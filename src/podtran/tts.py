from __future__ import annotations

import base64
from dataclasses import dataclass, field
import gc
import io
import json
import mimetypes
import os
from pathlib import Path
import queue
import threading
from typing import Callable, Protocol

import httpx
from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from podtran.artifacts import (
    ArtifactPaths,
    atomic_write_bytes,
    copy_path,
    read_model_list,
    remove_path,
    write_json,
)
from podtran.audio import FFPROBE_COMMAND, probe_duration
from podtran.cache_store import CacheStore
from podtran.config import AppConfig
from podtran.fingerprints import FingerprintService, TTS_CONFIG_KEYS, normalize_text
from podtran.models import (
    PresetVoiceSpec,
    ProviderCloneSpec,
    ReferenceCloneSpec,
    ResolvedVoiceTarget,
    SegmentRecord,
    StageManifest,
    StageProgressCallback,
    VoiceSpec,
)
from podtran.stage_versions import TTS_STAGE_VERSION
from podtran.voices import (
    UNKNOWN_SPEAKER_TTS_SKIP_MESSAGE,
    VoiceResolver,
    build_preset_targets,
    is_unknown_speaker,
    resolve_dashscope_api_key,
)

TTS_RETRY_ATTEMPTS = 3
CLONE_VOICE_KINDS = frozenset({"provider_clone", "reference_clone"})
QWEN_LOCAL_INSTALL_HINT = "Install local Qwen support with: uv sync --extra qwen-local"
QWEN_LOCAL_BASE_MODELS = {
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}
QWEN_LOCAL_CUSTOM_VOICE_MODELS = {
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
}
QWEN_LOCAL_ATTN_IMPLEMENTATIONS = frozenset({"flash_attention_2", "sdpa", "eager"})


class TTSBackend(Protocol):
    supported_voice_kinds: frozenset[str]

    def synthesize(
        self, text: str, spec: VoiceSpec, model: str, output_path: Path
    ) -> None: ...


class DashScopeTTSBackend:
    supported_voice_kinds = frozenset({"preset", "provider_clone"})

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = httpx.Client(timeout=config.tts.timeout_seconds)
        self.api_key = resolve_dashscope_api_key(config)

    @retry(
        reraise=True,
        stop=stop_after_attempt(TTS_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((RuntimeError, httpx.HTTPError)),
    )
    def synthesize(
        self, text: str, spec: VoiceSpec, model: str, output_path: Path
    ) -> None:
        response = self.client.post(
            f"{self.config.resolved_tts_base_url()}/services/aigc/multimodal-generation/generation",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": model,
                "input": {
                    "text": text,
                    "voice": self._voice_value(spec),
                },
            },
        )
        response.raise_for_status()
        payload = response.json()
        audio_url = payload.get("output", {}).get("audio", {}).get("url", "")
        if not audio_url:
            raise RuntimeError(
                f"DashScope TTS returned no audio URL: {json.dumps(payload, ensure_ascii=False)}"
            )
        audio_response = self.client.get(audio_url)
        audio_response.raise_for_status()
        atomic_write_bytes(output_path, audio_response.content)

    def _voice_value(self, spec: VoiceSpec) -> str:
        if isinstance(spec, PresetVoiceSpec):
            return spec.voice_name
        if isinstance(spec, ProviderCloneSpec):
            return spec.payload.voice_token
        if isinstance(spec, ReferenceCloneSpec):
            raise RuntimeError(
                "DashScope TTS backend does not support reference_clone specs."
            )
        raise RuntimeError(f"Unsupported voice spec for DashScope TTS: {spec.kind}")


class OpenAICompatibleTTSBackend:
    supported_voice_kinds = frozenset({"preset"})

    def __init__(self, config: AppConfig) -> None:
        self.client = OpenAI(
            api_key=_resolve_openai_compatible_api_key(config),
            base_url=config.resolved_tts_base_url(),
            timeout=config.tts.timeout_seconds,
        )
        self.config = config

    @retry(
        reraise=True,
        stop=stop_after_attempt(TTS_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(RuntimeError),
    )
    def synthesize(
        self, text: str, spec: VoiceSpec, model: str, output_path: Path
    ) -> None:
        if not isinstance(spec, PresetVoiceSpec):
            raise RuntimeError(
                f"OpenAI-compatible TTS does not support voice kind: {spec.kind}"
            )
        response = self.client.audio.speech.create(
            model=model,
            voice=spec.voice_name,
            input=text,
            response_format="wav",
        )
        payload = _read_binary_response(response)
        atomic_write_bytes(output_path, payload)


class VllmOmniTTSBackend:
    supported_voice_kinds = frozenset({"preset", "reference_clone"})

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = httpx.Client(timeout=config.tts.timeout_seconds)
        self.api_key = _resolve_vllm_omni_api_key(config)

    @retry(
        reraise=True,
        stop=stop_after_attempt(TTS_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((RuntimeError, httpx.HTTPError)),
    )
    def synthesize(
        self, text: str, spec: VoiceSpec, model: str, output_path: Path
    ) -> None:
        payload = {
            "model": model,
            "input": text,
            "response_format": "wav",
            "language": self.config.providers.vllm_omni.language,
            "instructions": self.config.providers.vllm_omni.instructions,
        }
        if isinstance(spec, PresetVoiceSpec):
            payload.update(
                {
                    "voice": spec.voice_name,
                    "task_type": "CustomVoice",
                }
            )
        elif isinstance(spec, ReferenceCloneSpec):
            payload.update(
                {
                    "task_type": "Base",
                    "ref_audio": _audio_data_uri(
                        Path(spec.payload.reference_audio_path)
                    ),
                    "ref_text": _resolve_reference_text(spec),
                    "x_vector_only_mode": _reference_x_vector_only_mode(
                        spec, self.config.providers.vllm_omni.x_vector_only_mode
                    ),
                }
            )
        else:
            raise RuntimeError(
                f"vLLM-Omni TTS does not support voice kind: {spec.kind}"
            )

        response = self.client.post(
            _vllm_omni_speech_url(self.config),
            headers=_bearer_headers(self.api_key),
            json=payload,
        )
        response.raise_for_status()
        atomic_write_bytes(output_path, response.content)


class QwenLocalTTSBackend:
    supported_voice_kinds = frozenset({"preset", "reference_clone"})

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.model: object | None = None
        self.model_kind = ""
        self.model_size = ""
        self.device = self._resolve_device()
        self._prompt_cache: dict[str, object] = {}

    def synthesize(
        self, text: str, spec: VoiceSpec, model: str, output_path: Path
    ) -> None:
        if isinstance(spec, PresetVoiceSpec):
            self._synthesize_preset(text, spec, output_path)
            return
        if isinstance(spec, ReferenceCloneSpec):
            self._synthesize_clone(text, spec, output_path)
            return
        raise RuntimeError(f"Qwen local TTS does not support voice kind: {spec.kind}")

    def _synthesize_preset(
        self, text: str, spec: PresetVoiceSpec, output_path: Path
    ) -> None:
        model = self._load_model(
            "customvoice", self.config.providers.qwen_local.preset_model_size
        )
        wavs, sample_rate = model.generate_custom_voice(
            text=text,
            language=self.config.providers.qwen_local.language,
            speaker=spec.voice_name,
            instruct=self.config.providers.qwen_local.instructions or None,
        )
        _write_wav(output_path, wavs[0], sample_rate)

    def _synthesize_clone(
        self, text: str, spec: ReferenceCloneSpec, output_path: Path
    ) -> None:
        model = self._load_model(
            "base", self.config.providers.qwen_local.clone_model_size
        )
        prompt = self._voice_prompt(model, spec)
        wavs, sample_rate = model.generate_voice_clone(
            text=text,
            language=self.config.providers.qwen_local.language,
            voice_clone_prompt=prompt,
            instruct=self.config.providers.qwen_local.instructions or None,
        )
        _write_wav(output_path, wavs[0], sample_rate)

    def _voice_prompt(self, model: object, spec: ReferenceCloneSpec) -> object:
        key = spec.payload.reference_fingerprint.strip() or spec.identity
        cached = self._prompt_cache.get(key)
        if cached is not None:
            return cached
        reference_text = _resolve_reference_text(spec)
        prompt = model.create_voice_clone_prompt(
            ref_audio=spec.payload.reference_audio_path,
            ref_text=reference_text,
            x_vector_only_mode=_reference_x_vector_only_mode(
                spec, self.config.providers.qwen_local.x_vector_only_mode
            ),
        )
        self._prompt_cache[key] = prompt
        return prompt

    def _load_model(self, kind: str, size: str) -> object:
        normalized_size = _normalize_qwen_model_size(size)
        if (
            self.model is not None
            and self.model_kind == kind
            and self.model_size == normalized_size
        ):
            return self.model
        self._unload_model()
        repo = _qwen_local_model_repo(kind, normalized_size)
        self.model = self._from_pretrained(repo)
        self.model_kind = kind
        self.model_size = normalized_size
        self._prompt_cache.clear()
        return self.model

    def _from_pretrained(self, repo: str) -> object:
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise RuntimeError(
                f"qwen-local dependencies are not installed. {QWEN_LOCAL_INSTALL_HINT}"
            ) from exc

        if self.device == "cpu":
            kwargs = {"dtype": torch.float32, "low_cpu_mem_usage": False}
        else:
            kwargs = {
                "device_map": self.device,
                "dtype": _resolve_qwen_local_torch_dtype(
                    torch, self.config.providers.qwen_local.torch_dtype, self.device
                ),
            }
            attn_implementation = _resolve_qwen_local_attn_implementation(
                self.config.providers.qwen_local.attn_implementation
            )
            if attn_implementation:
                kwargs["attn_implementation"] = attn_implementation
        return _qwen_local_from_pretrained(Qwen3TTSModel, repo, kwargs)

    def _unload_model(self) -> None:
        if self.model is None:
            return
        self.model = None
        self.model_kind = ""
        self.model_size = ""
        self._prompt_cache.clear()
        gc.collect()
        try:
            import torch
        except ImportError:
            return
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif self.device == "xpu" and hasattr(torch, "xpu"):
            torch.xpu.empty_cache()

    def _resolve_device(self) -> str:
        requested = self.config.providers.qwen_local.device.strip().lower()
        if requested and requested != "auto":
            return requested
        try:
            import torch
        except ImportError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        return "cpu"


@dataclass(slots=True)
class _PendingSegment:
    segment_index: int
    audio_path: Path


@dataclass(slots=True)
class _SynthesisWorkItem:
    text_zh: str
    target: ResolvedVoiceTarget
    output_path: Path
    model: str
    cache_key: str | None
    segments: list[_PendingSegment] = field(default_factory=list)


@dataclass(slots=True)
class _SynthesisResult:
    output_path: Path
    duration_ms: int


@dataclass(slots=True)
class _SynthesisWorkerOutcome:
    item: _SynthesisWorkItem
    payload: _SynthesisResult | BaseException


def synthesize_segments(
    input_path: Path,
    output_path: Path,
    config: AppConfig,
    paths: ArtifactPaths,
    source_audio: Path | None = None,
    source_audio_fingerprint: str | None = None,
    cache_store: CacheStore | None = None,
    fingerprints: FingerprintService | None = None,
    progress_callback: StageProgressCallback | None = None,
) -> list[SegmentRecord]:
    segments = _load_resume_segments(input_path, output_path)
    paths.tts_dir.mkdir(parents=True, exist_ok=True)

    segment_units = len(segments)
    stage_total = max(segment_units, 1)

    if progress_callback is not None:
        progress_callback(0, stage_total, "Resolving voices")

    voice_targets = _resolve_voice_targets(
        config,
        paths,
        segments,
        source_audio,
        source_audio_fingerprint,
        cache_store,
        fingerprints,
        progress_callback=(
            lambda completed, total, message: (
                progress_callback(completed, max(total, 1), message)
                if progress_callback is not None
                else None
            )
        ),
    )
    if (
        progress_callback is not None
        and config.tts.effective_mode(config.tts.provider) != "clone"
    ):
        progress_callback(0, stage_total, "Using preset voices")

    tts_config_fingerprint = (
        fingerprints.hash_config_subset(config, TTS_CONFIG_KEYS) if fingerprints else ""
    )
    processed_segments = 0
    work_items: dict[str, _SynthesisWorkItem] = {}

    for index, segment in enumerate(segments):
        audio_path = _segment_audio_path(paths, segment)
        existing_audio_paths = _existing_tts_audio_paths(segment, audio_path)
        reused_audio = False
        for existing_audio_path in existing_audio_paths:
            duration_ms = _valid_tts_duration_or_none(existing_audio_path)
            if duration_ms is None:
                _discard_invalid_tts_audio(existing_audio_path)
                _reset_segment_tts_state(segment)
                write_json(output_path, segments)
                continue
            _mark_segment_completed(segment, existing_audio_path, duration_ms)
            write_json(output_path, segments)
            processed_segments += 1
            _emit_segment_progress(
                progress_callback, processed_segments, stage_total, "Reusing audio"
            )
            reused_audio = True
            break
        if reused_audio:
            continue
        if not segment.text_zh.strip():
            segment.status = "failed"
            segment.error = "Missing translated text."
            write_json(output_path, segments)
            processed_segments += 1
            _emit_segment_progress(
                progress_callback, processed_segments, stage_total, "Skipping segment"
            )
            continue

        target = voice_targets.get(segment.speaker)
        if target is None:
            if config.tts.effective_mode(config.tts.provider) == "clone":
                if is_unknown_speaker(segment.speaker):
                    segment.status = "failed"
                    segment.error = UNKNOWN_SPEAKER_TTS_SKIP_MESSAGE
                    write_json(output_path, segments)
                    processed_segments += 1
                    _emit_segment_progress(
                        progress_callback,
                        processed_segments,
                        stage_total,
                        "Skipping UNKNOWN speaker",
                    )
                    continue
                raise RuntimeError(
                    f"Missing resolved clone voice target for {segment.speaker}."
                )
            target = ResolvedVoiceTarget(
                speaker=segment.speaker,
                spec=PresetVoiceSpec(
                    identity=f"preset:{segment.voice.strip()}",
                    voice_name=segment.voice.strip(),
                ),
            )

        model = _resolve_tts_model(config, target.spec)
        cache_key = _tts_cache_key(segment, target.spec, model, config, fingerprints)
        if cache_key and cache_store:
            entry = cache_store.lookup("tts", cache_key)
            if entry is not None:
                cache_store.restore(entry, {"audio": audio_path})
                duration_ms = _valid_tts_duration_or_none(audio_path)
                if duration_ms is not None:
                    _mark_segment_completed(segment, audio_path, duration_ms)
                    write_json(output_path, segments)
                    processed_segments += 1
                    _emit_segment_progress(
                        progress_callback,
                        processed_segments,
                        stage_total,
                        "Restored cached audio",
                    )
                    continue
                _discard_invalid_tts_audio(audio_path)
                remove_path(entry.entry_dir)

        work_key = _tts_work_key(segment, target.spec, model)
        work_item = work_items.get(work_key)
        if work_item is None:
            work_item = _SynthesisWorkItem(
                text_zh=segment.text_zh,
                target=target,
                output_path=audio_path,
                model=model,
                cache_key=cache_key,
            )
            work_items[work_key] = work_item
        work_item.segments.append(
            _PendingSegment(segment_index=index, audio_path=audio_path)
        )

    if work_items:
        _emit_segment_progress(
            progress_callback,
            processed_segments,
            stage_total,
            "Synthesizing audio",
        )
        max_workers = _synthesis_worker_count(config, len(work_items))
        work_queue: queue.Queue[_SynthesisWorkItem] = queue.Queue()
        result_queue: queue.Queue[_SynthesisWorkerOutcome] = queue.Queue()
        stop_event = threading.Event()

        for item in work_items.values():
            work_queue.put(item)

        worker_count = max_workers
        for index in range(worker_count):
            threading.Thread(
                target=_synthesis_worker,
                name=f"podtran-tts-{index + 1}",
                args=(config, work_queue, result_queue, stop_event),
                daemon=True,
            ).start()

        def handle_outcome(outcome: _SynthesisWorkerOutcome) -> None:
            nonlocal processed_segments
            item = outcome.item
            payload = outcome.payload
            if isinstance(payload, KeyboardInterrupt):
                raise payload
            if isinstance(payload, BaseException):
                for pending in item.segments:
                    segment = segments[pending.segment_index]
                    _mark_segment_failed(segment, str(payload))
                    processed_segments += 1
            else:
                if item.cache_key and cache_store and fingerprints:
                    cache_store.publish(
                        "tts",
                        item.cache_key,
                        {"audio": payload.output_path},
                        _build_tts_manifest(
                            item.text_zh,
                            item.target.spec,
                            item.model,
                            item.cache_key,
                            tts_config_fingerprint,
                            fingerprints,
                        ),
                    )
                for pending in item.segments:
                    destination = pending.audio_path
                    if destination.resolve() != payload.output_path.resolve():
                        copy_path(payload.output_path, destination)
                    _mark_segment_completed(
                        segments[pending.segment_index],
                        destination,
                        payload.duration_ms,
                    )
                    processed_segments += 1
            write_json(output_path, segments)
            _emit_segment_progress(
                progress_callback,
                processed_segments,
                stage_total,
                "Synthesizing audio",
            )

        completed_items = 0
        total_items = len(work_items)
        try:
            while completed_items < total_items:
                try:
                    outcome = result_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                completed_items += 1
                handle_outcome(outcome)
        except KeyboardInterrupt:
            stop_event.set()
            completed_items += _drain_ready_synthesis_outcomes(
                result_queue, handle_outcome
            )
            raise
        finally:
            stop_event.set()

    if progress_callback is not None:
        progress_callback(stage_total, stage_total, "Synthesis complete")
    return segments


def build_tts_backend(config: AppConfig) -> TTSBackend:
    provider = config.tts.provider.strip().lower()
    if provider == "dashscope":
        backend: TTSBackend = DashScopeTTSBackend(config)
    elif provider == "openai-compatible":
        backend = OpenAICompatibleTTSBackend(config)
    elif provider == "vllm-omni":
        backend = VllmOmniTTSBackend(config)
    elif provider == "qwen-local":
        backend = QwenLocalTTSBackend(config)
    else:
        raise RuntimeError(f"Unsupported TTS provider: {config.tts.provider}")

    if config.tts.effective_mode(config.tts.provider) == "clone" and not (
        backend.supported_voice_kinds & CLONE_VOICE_KINDS
    ):
        raise RuntimeError(
            f"Clone mode is not supported for TTS provider: {config.tts.provider}"
        )
    return backend


def _load_resume_segments(input_path: Path, output_path: Path) -> list[SegmentRecord]:
    source = output_path if output_path.exists() else input_path
    return read_model_list(source, SegmentRecord)


def _resolve_voice_targets(
    config: AppConfig,
    paths: ArtifactPaths,
    segments: list[SegmentRecord],
    source_audio: Path | None,
    source_audio_fingerprint: str | None,
    cache_store: CacheStore | None,
    fingerprints: FingerprintService | None,
    progress_callback: StageProgressCallback | None = None,
) -> dict[str, ResolvedVoiceTarget]:
    if config.tts.effective_mode(config.tts.provider) != "clone":
        return build_preset_targets(segments)
    if source_audio is None:
        raise RuntimeError("Clone mode requires source audio.")
    resolved_source_audio = source_audio.resolve()
    if not resolved_source_audio.exists():
        raise RuntimeError(
            f"Source audio not found for clone mode: {resolved_source_audio}"
        )
    return VoiceResolver(
        config, paths, cache_store=cache_store, fingerprints=fingerprints
    ).resolve_voice_targets(
        segments,
        resolved_source_audio,
        source_audio_fingerprint=source_audio_fingerprint,
        progress_callback=progress_callback,
    )


def _emit_segment_progress(
    progress_callback: StageProgressCallback | None,
    completed: int,
    stage_total: int,
    action: str,
) -> None:
    if progress_callback is None:
        return
    progress_callback(completed, stage_total, action)


def _mark_segment_completed(
    segment: SegmentRecord, audio_path: Path, duration_ms: int
) -> None:
    segment.tts_audio_path = str(audio_path.resolve())
    segment.tts_duration_ms = duration_ms
    segment.status = "completed"
    segment.error = None


def _mark_segment_failed(segment: SegmentRecord, error: str) -> None:
    _reset_segment_tts_state(segment)
    segment.status = "failed"
    segment.error = error


def _reset_segment_tts_state(segment: SegmentRecord) -> None:
    segment.tts_audio_path = ""
    segment.tts_duration_ms = 0
    segment.status = "pending"
    segment.error = None


def _segment_audio_path(paths: ArtifactPaths, segment: SegmentRecord) -> Path:
    return paths.tts_dir / f"{segment.segment_id}_{segment.speaker}.wav"


def _existing_tts_audio_paths(
    segment: SegmentRecord, default_audio_path: Path
) -> list[Path]:
    candidates: list[Path] = []
    if segment.tts_audio_path and segment.status == "completed":
        candidates.append(Path(segment.tts_audio_path))
    candidates.append(default_audio_path)

    existing_paths: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists():
            existing_paths.append(candidate)
    return existing_paths


def _valid_tts_duration_or_none(audio_path: Path) -> int | None:
    try:
        return _validated_tts_duration_ms(audio_path)
    except RuntimeError:
        return None


def _validated_tts_duration_ms(audio_path: Path) -> int:
    try:
        if not audio_path.exists():
            raise RuntimeError("file does not exist")
        if audio_path.stat().st_size <= 0:
            raise RuntimeError("file is empty")
        return int(probe_duration(FFPROBE_COMMAND, audio_path) * 1000)
    except Exception as exc:
        raise RuntimeError(f"{audio_path}: invalid TTS audio ({exc})") from exc


def _discard_invalid_tts_audio(audio_path: Path) -> None:
    if audio_path.exists() and audio_path.is_file():
        audio_path.unlink()


def _tts_work_key(segment: SegmentRecord, spec: VoiceSpec, model: str) -> str:
    return json.dumps(
        {
            "text_zh": normalize_text(segment.text_zh),
            "voice_spec": spec.model_dump(),
            "model": model,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _synthesize_work_item(
    backend: TTSBackend, item: _SynthesisWorkItem
) -> _SynthesisResult:
    _ensure_backend_supports_spec(backend, item.target.spec)
    backend.synthesize(item.text_zh, item.target.spec, item.model, item.output_path)
    try:
        duration_ms = _validated_tts_duration_ms(item.output_path)
    except RuntimeError:
        _discard_invalid_tts_audio(item.output_path)
        raise
    return _SynthesisResult(output_path=item.output_path, duration_ms=duration_ms)


def _synthesis_worker(
    config: AppConfig,
    work_queue: queue.Queue[_SynthesisWorkItem],
    result_queue: queue.Queue[_SynthesisWorkerOutcome],
    stop_event: threading.Event,
) -> None:
    backend: TTSBackend | None = None
    while not stop_event.is_set():
        try:
            item = work_queue.get_nowait()
        except queue.Empty:
            return

        if stop_event.is_set():
            work_queue.task_done()
            return

        try:
            if backend is None:
                backend = build_tts_backend(config)
            payload: _SynthesisResult | BaseException = _synthesize_work_item(
                backend, item
            )
        except (
            BaseException
        ) as exc:  # pragma: no cover - exercised through main-thread handling
            payload = exc
        result_queue.put(_SynthesisWorkerOutcome(item=item, payload=payload))
        work_queue.task_done()


def _drain_ready_synthesis_outcomes(
    result_queue: queue.Queue[_SynthesisWorkerOutcome],
    handle_outcome: Callable[[_SynthesisWorkerOutcome], None],
) -> int:
    drained = 0
    while True:
        try:
            outcome = result_queue.get_nowait()
        except queue.Empty:
            return drained
        drained += 1
        handle_outcome(outcome)


def _build_tts_manifest(
    text_zh: str,
    spec: VoiceSpec,
    model: str,
    cache_key: str,
    config_fingerprint: str,
    fingerprints: FingerprintService,
) -> StageManifest:
    return StageManifest(
        stage="tts",
        status="completed",
        stage_version=TTS_STAGE_VERSION,
        cache_key=cache_key,
        input_fingerprints={
            "text_zh": fingerprints.hash_value(normalize_text(text_zh)),
            "voice_spec": fingerprints.hash_value(
                {
                    "spec": spec.model_dump(),
                    "model": model,
                }
            ),
        },
        config_fingerprint=config_fingerprint,
        config_keys=TTS_CONFIG_KEYS,
    )


def _tts_cache_key(
    segment: SegmentRecord,
    spec: VoiceSpec,
    model: str,
    config: AppConfig,
    fingerprints: FingerprintService | None,
) -> str | None:
    if fingerprints is None:
        return None
    config_fingerprint = fingerprints.hash_config_subset(config, TTS_CONFIG_KEYS)
    return fingerprints.build_stage_cache_key(
        "tts",
        TTS_STAGE_VERSION,
        {
            "text_zh": fingerprints.hash_value(normalize_text(segment.text_zh)),
            "voice_spec": fingerprints.hash_value(
                {
                    "spec": spec.model_dump(),
                    "model": model,
                }
            ),
        },
        config_fingerprint,
    )


def _read_binary_response(response: object) -> bytes:
    if hasattr(response, "read"):
        data = response.read()
        if isinstance(data, bytes):
            return data
    if hasattr(response, "content"):
        content = getattr(response, "content")
        if isinstance(content, bytes):
            return content
    raise RuntimeError("Unsupported binary response from TTS backend.")


def _resolve_tts_model(config: AppConfig, spec: VoiceSpec) -> str:
    if spec.kind == "preset":
        return config.tts_preset_model()
    return config.tts_clone_model()


def _resolve_openai_compatible_api_key(config: AppConfig | None = None) -> str:
    if config is not None:
        configured = config.resolve_provider_api_key("openai-compatible", purpose="tts")
        if configured:
            return configured
    return os.getenv("PODTRAN_TTS_API_KEY", "")


def _resolve_vllm_omni_api_key(config: AppConfig) -> str:
    configured = config.providers.vllm_omni.api_key.strip()
    if configured:
        return configured
    return os.getenv("PODTRAN_VLLM_OMNI_API_KEY", "")


def _bearer_headers(api_key: str) -> dict[str, str]:
    if not api_key.strip():
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def _audio_data_uri(path: Path) -> str:
    payload = path.read_bytes()
    mime_type = mimetypes.guess_type(path.name)[0] or "audio/wav"
    encoded = base64.b64encode(payload).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _resolve_reference_text(spec: ReferenceCloneSpec) -> str:
    if spec.payload.reference_text.strip():
        return spec.payload.reference_text
    if spec.payload.reference_text_path.strip():
        return (
            Path(spec.payload.reference_text_path).read_text(encoding="utf-8").strip()
        )
    raise RuntimeError("reference_clone specs require reference text.")


def _reference_x_vector_only_mode(
    spec: ReferenceCloneSpec, default_value: bool
) -> bool:
    if spec.payload.x_vector_only_mode is not None:
        return spec.payload.x_vector_only_mode
    return default_value


def _vllm_omni_speech_url(config: AppConfig) -> str:
    base_url = config.resolved_tts_base_url()
    if not base_url:
        raise RuntimeError("vLLM-Omni TTS requires a TTS base URL.")
    return base_url.rstrip("/") + "/audio/speech"


def _ensure_backend_supports_spec(backend: TTSBackend, spec: VoiceSpec) -> None:
    if spec.kind in backend.supported_voice_kinds:
        return
    raise RuntimeError(f"TTS provider does not support voice kind: {spec.kind}")


def _synthesis_worker_count(config: AppConfig, work_item_count: int) -> int:
    if config.tts.provider.strip().lower() == "qwen-local":
        return 1
    return min(max(1, config.tts.max_concurrency), work_item_count)


def _normalize_qwen_model_size(size: str) -> str:
    normalized = size.strip() or "0.6B"
    if normalized not in QWEN_LOCAL_BASE_MODELS:
        raise RuntimeError(
            f"Unsupported qwen-local model size: {size}. Expected one of: 0.6B, 1.7B"
        )
    return normalized


def _resolve_qwen_local_torch_dtype(torch: object, value: str, device: str) -> object:
    normalized = value.strip().lower().replace("-", "").replace("_", "")
    if normalized in {"", "auto"}:
        if device == "cpu":
            return torch.float32
        if _qwen_local_bf16_supported(torch):
            return torch.bfloat16
        return torch.float16
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32", "full"}:
        return torch.float32
    raise RuntimeError(
        "Unsupported qwen-local torch_dtype: "
        f"{value}. Expected one of: auto, float16, bfloat16, float32"
    )


def _qwen_local_bf16_supported(torch: object) -> bool:
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not hasattr(cuda, "is_bf16_supported"):
        return False
    try:
        return bool(cuda.is_bf16_supported())
    except Exception:
        return False


def _resolve_qwen_local_attn_implementation(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"", "auto"}:
        return ""
    if normalized not in QWEN_LOCAL_ATTN_IMPLEMENTATIONS:
        raise RuntimeError(
            "Unsupported qwen-local attn_implementation: "
            f"{value}. Expected one of: auto, flash_attention_2, sdpa, eager"
        )
    return normalized


def _qwen_local_from_pretrained(
    model_cls: object, repo: str, kwargs: dict[str, object]
) -> object:
    candidates = [dict(kwargs)]
    if "attn_implementation" in kwargs:
        without_attn = dict(kwargs)
        without_attn.pop("attn_implementation", None)
        candidates.append(without_attn)

    last_error: TypeError | None = None
    for candidate in candidates:
        for compatible_kwargs in _qwen_local_dtype_key_candidates(candidate):
            try:
                return model_cls.from_pretrained(repo, **compatible_kwargs)
            except TypeError as exc:
                last_error = exc
    if last_error is not None:
        raise last_error
    return model_cls.from_pretrained(repo, **kwargs)


def _qwen_local_dtype_key_candidates(
    kwargs: dict[str, object],
) -> list[dict[str, object]]:
    candidates = [dict(kwargs)]
    if "dtype" in kwargs:
        compatible = dict(kwargs)
        compatible["torch_dtype"] = compatible.pop("dtype")
        candidates.append(compatible)
    return candidates


def _qwen_local_model_repo(kind: str, size: str) -> str:
    if kind == "base":
        return QWEN_LOCAL_BASE_MODELS[_normalize_qwen_model_size(size)]
    if kind == "customvoice":
        return QWEN_LOCAL_CUSTOM_VOICE_MODELS[_normalize_qwen_model_size(size)]
    raise RuntimeError(f"Unsupported qwen-local model kind: {kind}")


def _write_wav(output_path: Path, audio: object, sample_rate: int) -> None:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            f"qwen-local dependencies are not installed. {QWEN_LOCAL_INSTALL_HINT}"
        ) from exc
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    atomic_write_bytes(output_path, buffer.getvalue())
