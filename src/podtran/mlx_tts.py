from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import TYPE_CHECKING

from podtran.artifacts import atomic_write_bytes, atomic_write_text
from podtran.mlx_worker import SpeechRequest, SpeechResponse
from podtran.models import PresetVoiceSpec, ReferenceCloneSpec, VoiceSpec

if TYPE_CHECKING:
    from podtran.config import AppConfig

REQUIREMENTS = Path(__file__).with_name("mlx-tts-requirements.txt")
WORKER_SCRIPT = Path(__file__).with_name("mlx_worker.py")


def _uv_command() -> str:
    try:
        import uv

        return uv.find_uv_bin()
    except ImportError:
        command = shutil.which("uv")
        if command:
            return command
    raise RuntimeError(
        "MLX TTS setup requires uv. Reinstall podtran's Mac dependencies."
    )


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _check_cancelled(stop_event: threading.Event | None) -> None:
    if stop_event is not None and stop_event.is_set():
        raise KeyboardInterrupt


@contextmanager
def _environment_lock(path: Path, stop_event: threading.Event | None):
    import fcntl

    with path.open("a") as handle:
        while True:
            _check_cancelled(stop_event)
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(0.1)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def ensure_mlx_environment(
    workdir: Path, stop_event: threading.Event | None = None
) -> Path:
    digest = hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()[:16]
    runtime_dir = workdir.resolve() / "runtimes"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    environment = runtime_dir / f"mlx-tts-py311-{digest}"
    python = environment / "bin" / "python"
    ready = environment / ".podtran-ready"
    with _environment_lock(runtime_dir / f"{environment.name}.lock", stop_event):
        if python.is_file() and ready.is_file():
            return python
        if environment.exists():
            shutil.rmtree(environment)
        uv = _uv_command()
        print(
            "Preparing the isolated MLX TTS environment (first use)...", file=sys.stderr
        )
        log_path = runtime_dir / f"{environment.name}.log"
        with log_path.open("w", encoding="utf-8") as log:
            for command in (
                [uv, "venv", "--python", "3.11", str(environment)],
                [uv, "pip", "sync", "--python", str(python), str(REQUIREMENTS)],
            ):
                _check_cancelled(stop_event)
                process = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT
                )
                try:
                    while process.poll() is None:
                        _check_cancelled(stop_event)
                        time.sleep(0.1)
                    if process.returncode:
                        raise RuntimeError(
                            f"MLX TTS environment setup failed. See {log_path}. "
                            "Retry to rebuild the incomplete environment."
                        )
                finally:
                    _stop_process(process)
        atomic_write_text(ready, digest)
    return python


class MLXTTSBackend:
    supported_voice_kinds = frozenset({"preset", "reference_clone"})

    def __init__(
        self,
        config: AppConfig,
        workdir: Path,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.config = config
        self.workdir = workdir
        self.stop_event = stop_event
        self.process: subprocess.Popen | None = None
        self.responses: queue.Queue[str | None] = queue.Queue()
        self.temp_dir: tempfile.TemporaryDirectory | None = None
        self.log = None
        self.reader: threading.Thread | None = None

    def _start(self) -> None:
        if self.process is not None:
            return
        python = ensure_mlx_environment(self.workdir, self.stop_event)
        self.temp_dir = tempfile.TemporaryDirectory(prefix="mlx-tts-", dir=self.workdir)
        self.log = (Path(self.temp_dir.name) / "worker.log").open("w+")
        try:
            self.process = subprocess.Popen(
                [str(python), "-u", str(WORKER_SCRIPT)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                text=True,
                encoding="utf-8",
                env={
                    key: value
                    for key, value in os.environ.items()
                    if key not in {"PYTHONPATH", "PYTHONHOME"}
                },
            )
        except BaseException:
            self.close()
            raise
        self.reader = threading.Thread(target=self._read_responses, daemon=True)
        self.reader.start()

    def _read_responses(self) -> None:
        try:
            for line in self.process.stdout:
                self.responses.put(line)
        finally:
            self.responses.put(None)

    def synthesize(
        self, text: str, spec: VoiceSpec, model: str, output_path: Path
    ) -> None:
        from podtran.tts import _reference_x_vector_only_mode, _resolve_reference_text

        _check_cancelled(self.stop_event)
        self._start()
        settings = self.config.providers.qwen_local
        temporary_audio = Path(self.temp_dir.name) / "audio.wav"
        temporary_audio.unlink(missing_ok=True)
        kwargs = dict(
            kind=spec.kind,
            size=(
                settings.clone_model_size
                if isinstance(spec, ReferenceCloneSpec)
                else settings.preset_model_size
            ).strip()
            or "0.6B",
            text=text,
            output_path=str(temporary_audio),
            language=settings.language,
            instructions=settings.instructions,
        )
        if isinstance(spec, ReferenceCloneSpec):
            kwargs.update(
                reference_audio=spec.payload.reference_audio_path,
                reference_text=None
                if _reference_x_vector_only_mode(spec, settings.x_vector_only_mode)
                else _resolve_reference_text(spec),
            )
        elif isinstance(spec, PresetVoiceSpec):
            kwargs["voice"] = spec.voice_name
        else:
            raise RuntimeError(f"MLX TTS does not support {spec.kind}.")
        request = SpeechRequest.model_validate(kwargs)
        try:
            self.process.stdin.write(request.model_dump_json() + "\n")
            self.process.stdin.flush()
            while True:
                _check_cancelled(self.stop_event)
                try:
                    line = self.responses.get(timeout=0.1)
                    break
                except queue.Empty:
                    continue
            if line is None:
                self.log.flush()
                self.log.seek(0)
                detail = self.log.read()[-4000:]
                raise RuntimeError(f"MLX TTS worker exited unexpectedly. {detail}")
            response = SpeechResponse.model_validate_json(line)
            if response.error:
                raise RuntimeError(response.error)
            if not temporary_audio.is_file() or temporary_audio.stat().st_size == 0:
                raise RuntimeError("MLX TTS worker returned no audio file.")
            atomic_write_bytes(output_path, temporary_audio.read_bytes())
        except BaseException:
            self.close()
            raise
        finally:
            temporary_audio.unlink(missing_ok=True)

    def synthesize_batch(self, requests: list, spec: VoiceSpec, model: str) -> None:
        # One persistent model, sequential requests; bounded memory on laptops.
        for request in requests:
            self.synthesize(request.text, spec, model, request.output_path)

    def close(self) -> None:
        if self.process is not None:
            _stop_process(self.process)
            if self.reader is not None:
                self.reader.join(timeout=3)
            for stream in (self.process.stdin, self.process.stdout):
                if stream is not None:
                    stream.close()
            self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None
        if self.temp_dir is not None:
            self.temp_dir.cleanup()
            self.temp_dir = None
        self.responses = queue.Queue()
