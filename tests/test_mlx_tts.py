from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from podtran.config import AppConfig
from podtran.mlx_tts import MLXTTSBackend, ensure_mlx_environment
from podtran.mlx_worker import SpeechRequest, SpeechResponse
from podtran.models import PresetVoiceSpec, ReferenceClonePayload, ReferenceCloneSpec


@pytest.fixture
def fake_worker(tmp_path, monkeypatch):
    # Exercise the actual JSON protocol and worker loop in a real child process.
    worker = tmp_path / "fake_worker.py"
    source = Path(__file__).resolve().parents[1] / "src"
    requests = tmp_path / "requests.jsonl"
    worker.write_text(
        f"import sys, time\nsys.path.insert(0, {str(source)!r})\n"
        "from pathlib import Path\n"
        "from podtran import mlx_worker\n"
        "class Engine:\n"
        "    def generate(self, request):\n"
        f"        with open({str(requests)!r}, 'a', encoding='utf-8') as out: out.write(request.model_dump_json() + '\\n')\n"
        "        if request.text == 'wait': time.sleep(60)\n"
        "        if request.text == 'fail': raise RuntimeError('GPU failure')\n"
        "        print('model diagnostic')\n"
        "        Path(request.output_path).write_bytes(b'RIFF-fake-audio')\n"
        "mlx_worker.SpeechEngine = Engine\n"
        "mlx_worker.main()\n"
    )
    monkeypatch.setattr("podtran.mlx_tts.WORKER_SCRIPT", worker)
    monkeypatch.setattr(
        "podtran.mlx_tts.ensure_mlx_environment", lambda *args: Path(sys.executable)
    )
    return requests


def clone_spec(x_vector=False):
    return ReferenceCloneSpec(
        identity="test-reference",
        provider="qwen-local",
        payload=ReferenceClonePayload(
            reference_fingerprint="ref1",
            reference_audio_path="reference.wav",
            reference_text="reference text",
            x_vector_only_mode=x_vector,
        ),
    )


@pytest.mark.parametrize("stdio_encoding", ["utf-8", "ascii", "cp1252"])
def test_worker_reuses_process_and_preserves_clone_and_preset_inputs(
    fake_worker, tmp_path, monkeypatch, stdio_encoding
):
    monkeypatch.setenv("PYTHONIOENCODING", stdio_encoding)
    backend = MLXTTSBackend(AppConfig(), tmp_path)
    try:
        backend.synthesize("你好", clone_spec(), "unused", tmp_path / "one.wav")
        process = backend.process
        backend.synthesize("再见", clone_spec(True), "unused", tmp_path / "two.wav")
        backend.synthesize(
            "预设",
            PresetVoiceSpec(identity="preset:Vivian", voice_name="Vivian"),
            "unused",
            tmp_path / "three.wav",
        )
        assert backend.process is process
        assert (tmp_path / "one.wav").read_bytes() == b"RIFF-fake-audio"
        requests = [json.loads(line) for line in fake_worker.read_text(encoding="utf-8").splitlines()]
        assert [request["text"] for request in requests] == ["你好", "再见", "预设"]
        assert requests[0]["reference_text"] == "reference text"
        assert requests[1]["reference_text"] is None
        assert requests[2]["voice"] == "Vivian"
    finally:
        backend.close()
    assert process.poll() is not None
    assert backend.temp_dir is None


def test_worker_failure_does_not_publish_or_switch_to_cpu(fake_worker, tmp_path):
    backend = MLXTTSBackend(AppConfig(), tmp_path)
    output = tmp_path / "audio.wav"
    with pytest.raises(RuntimeError, match="GPU failure"):
        backend.synthesize("fail", clone_spec(), "unused", output)
    assert not output.exists()
    assert backend.process is None
    assert len(fake_worker.read_text(encoding="utf-8").splitlines()) == 1


def test_cancellation_terminates_inflight_worker(fake_worker, tmp_path):
    stop = threading.Event()
    backend = MLXTTSBackend(AppConfig(), tmp_path, stop)
    errors = []

    def run():
        try:
            backend.synthesize("wait", clone_spec(), "unused", tmp_path / "audio.wav")
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 5
    while not fake_worker.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    process = backend.process
    stop.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert isinstance(errors[0], KeyboardInterrupt)
    assert process.poll() is not None
    assert not (tmp_path / "audio.wav").exists()


@pytest.mark.skipif(os.name == "nt", reason="Mac runtime uses POSIX file locks")
def test_environment_reuses_ready_install_and_rebuilds_interrupted_install(
    monkeypatch, tmp_path
):
    monkeypatch.setattr("podtran.mlx_tts._uv_command", lambda: "uv")
    commands = []

    def popen(command, **kwargs):
        commands.append(command)
        if "venv" in command:
            python = Path(command[-1]) / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.touch()
        return SimpleNamespace(poll=lambda: 0, returncode=0)

    monkeypatch.setattr("podtran.mlx_tts.subprocess.Popen", popen)
    python = ensure_mlx_environment(tmp_path)
    assert len(commands) == 2
    assert ensure_mlx_environment(tmp_path) == python
    assert len(commands) == 2
    (python.parent.parent / ".podtran-ready").unlink()
    ensure_mlx_environment(tmp_path)
    assert len(commands) == 4


@pytest.mark.skipif(os.name == "nt", reason="Mac runtime uses POSIX file locks")
def test_environment_failure_is_not_marked_ready(monkeypatch, tmp_path):
    monkeypatch.setattr("podtran.mlx_tts._uv_command", lambda: "uv")
    monkeypatch.setattr(
        "podtran.mlx_tts.subprocess.Popen",
        lambda *a, **k: SimpleNamespace(poll=lambda: 1, returncode=1),
    )
    with pytest.raises(RuntimeError, match="setup failed"):
        ensure_mlx_environment(tmp_path)
    assert not list(tmp_path.rglob(".podtran-ready"))


def test_speech_protocol_validates_voice_and_response():
    with pytest.raises(ValueError, match="reference audio"):
        SpeechRequest(
            kind="reference_clone", size="0.6B", text="hello", output_path="out.wav"
        )
    with pytest.raises(ValueError):
        SpeechResponse.model_validate_json('{"unexpected": true}')


def test_mlx_backend_routing_requires_workdir(monkeypatch, tmp_path):
    from podtran.tts import build_tts_backend

    monkeypatch.setattr("podtran.macos.apple_gpu_available", lambda: True)
    config = AppConfig()
    config.providers.qwen_local.device = "gpu"
    with pytest.raises(RuntimeError, match="workdir"):
        build_tts_backend(config)
    backend = build_tts_backend(config, tmp_path)
    assert isinstance(backend, MLXTTSBackend)
    assert backend.config.providers.qwen_local.device == "mlx"
    assert backend.process is None


def test_synthesis_worker_closes_mlx_backend(monkeypatch, tmp_path):
    import queue
    from podtran.tts import _synthesis_worker

    config = AppConfig()
    config.providers.qwen_local.device = "mlx"
    backend = Mock()
    monkeypatch.setattr("podtran.tts.build_tts_backend", lambda *args: backend)
    monkeypatch.setattr("podtran.tts._synthesize_batch_work_item", lambda *args: [])
    work = queue.Queue()
    work.put("batch")
    _synthesis_worker(config, work, queue.Queue(), threading.Event(), tmp_path)
    backend.close.assert_called_once()


def test_speech_engine_consumes_all_chunks_and_reuses_model(monkeypatch, tmp_path):
    from contextlib import nullcontext
    from podtran.mlx_worker import SpeechEngine

    class Array(list):
        @property
        def size(self):
            return len(self)

        def reshape(self, *args):
            return self

    numpy = SimpleNamespace(
        float32=object(),
        asarray=lambda value, **kwargs: Array(value),
        concatenate=lambda chunks: [value for chunk in chunks for value in chunk],
    )
    core = SimpleNamespace(
        metal=SimpleNamespace(is_available=lambda: True),
        gpu=object(),
        stream=lambda device: nullcontext(),
        clear_cache=Mock(),
    )
    generate = Mock(
        side_effect=lambda **kwargs: iter(
            [
                SimpleNamespace(audio=[0.1, 0.2], sample_rate=24000),
                SimpleNamespace(audio=[0.3], sample_rate=24000),
            ]
        )
    )
    load = Mock(return_value=SimpleNamespace(generate=generate))
    write = Mock()
    for name, module in {
        "numpy": numpy,
        "mlx": SimpleNamespace(core=core),
        "mlx.core": core,
        "mlx_audio": SimpleNamespace(),
        "mlx_audio.tts": SimpleNamespace(),
        "mlx_audio.tts.utils": SimpleNamespace(load_model=load),
        "soundfile": SimpleNamespace(write=write),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    engine = SpeechEngine()
    request = SpeechRequest(
        kind="reference_clone",
        size="0.6B",
        text="你好",
        output_path=str(tmp_path / "out.wav"),
        reference_audio="ref.wav",
        reference_text="hello",
    )
    engine.generate(request)
    engine.generate(request.model_copy(update={"reference_text": None}))
    load.assert_called_once_with("mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16")
    assert write.call_args.args[1] == [0.1, 0.2, 0.3]
    assert write.call_args.args[2] == 24000
    assert generate.call_args.kwargs["ref_text"] is None
    assert generate.call_args.kwargs["ref_audio"] == "ref.wav"


def test_anchor_voice_uses_mlx_and_closes_worker(monkeypatch, tmp_path):
    from podtran.voices import QwenLocalCloneProvider

    config = AppConfig()
    config.providers.qwen_local.device = "mlx"
    backend = Mock()
    factory = Mock(return_value=backend)
    monkeypatch.setattr("podtran.tts.build_tts_backend", factory)
    provider = QwenLocalCloneProvider(config, tmp_path)
    provider._synthesize_anchor(
        tmp_path / "reference.wav", "hello", "model", "ref1", tmp_path / "anchor.wav"
    )
    factory.assert_called_once_with(config, tmp_path)
    assert backend.synthesize.call_args.args[1].payload.x_vector_only_mode is True
    backend.close.assert_called_once()
