from pathlib import Path
import signal
import threading
import time

import pytest

from podtran.artifacts import ArtifactPaths, read_model_list, write_json
from podtran.cache_store import CacheStore
from podtran.config import AppConfig, DEFAULT_TTS_CLONE_MODEL, DEFAULT_TTS_PRESET_MODEL, TTSConfig
from podtran.fingerprints import FingerprintService
from podtran.models import (
    PresetVoiceSpec,
    ProviderClonePayload,
    ProviderCloneSpec,
    ReferenceClonePayload,
    ReferenceCloneSpec,
    SegmentRecord,
)
from podtran.tts import (
    VllmOmniTTSBackend,
    _resolve_openai_compatible_api_key,
    _resolve_tts_model,
    _resolve_vllm_omni_api_key,
    build_tts_backend,
    synthesize_segments,
)


class _DummyBackend:
    supported_voice_kinds = frozenset({"preset", "provider_clone", "reference_clone"})

    def __init__(self) -> None:
        self.calls = 0

    def synthesize(self, text: str, spec, model: str, output_path: Path) -> None:
        _ = (text, spec, model)
        self.calls += 1
        output_path.write_bytes(b"wav")


class _TrackingBackend:
    supported_voice_kinds = frozenset({"preset", "provider_clone", "reference_clone"})

    def __init__(self, *, fail_texts: set[str] | None = None, sleep_seconds: float = 0.05) -> None:
        self.calls = 0
        self.active_calls = 0
        self.max_active_calls = 0
        self.fail_texts = fail_texts or set()
        self.sleep_seconds = sleep_seconds
        self._lock = threading.Lock()

    def synthesize(self, text: str, spec, model: str, output_path: Path) -> None:
        _ = (spec, model)
        with self._lock:
            self.calls += 1
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            time.sleep(self.sleep_seconds)
            if text in self.fail_texts:
                raise RuntimeError(f"boom: {text}")
            output_path.write_bytes(text.encode("utf-8"))
        finally:
            with self._lock:
                self.active_calls -= 1


class _InterruptingBackend:
    supported_voice_kinds = frozenset({"preset", "provider_clone", "reference_clone"})

    def __init__(self) -> None:
        self.calls = 0

    def synthesize(self, text: str, spec, model: str, output_path: Path) -> None:
        _ = (text, spec, model, output_path)
        self.calls += 1
        if self.calls == 1:
            output_path.write_bytes(b"wav")
            return
        raise KeyboardInterrupt()


def _paths(tmp_path: Path, task_id: str = "task-1") -> ArtifactPaths:
    return ArtifactPaths.from_task_id(tmp_path, task_id)


def _segment() -> SegmentRecord:
    return SegmentRecord(
        segment_id="seg_1",
        block_id="block_1",
        start=0.0,
        end=1.0,
        text="hello world from speaker",
        speaker="SPEAKER_00",
        voice="Cherry",
        text_zh="你好",
    )


def _segment_with_text(segment_id: str, text_zh: str, voice: str = "Cherry") -> SegmentRecord:
    return _segment().model_copy(update={"segment_id": segment_id, "block_id": f"block_{segment_id}", "text_zh": text_zh, "voice": voice})


def test_resolve_tts_model_switches_between_clone_and_preset_specs() -> None:
    config = AppConfig(tts=TTSConfig(mode="clone"))

    clone_spec = ProviderCloneSpec(
        identity="dashscope:provider_clone:voice-token",
        provider="dashscope",
        payload=ProviderClonePayload(voice_token="voice-token"),
    )
    preset_spec = PresetVoiceSpec(identity="preset:Cherry", voice_name="Cherry")

    assert _resolve_tts_model(config, clone_spec) == DEFAULT_TTS_CLONE_MODEL
    assert _resolve_tts_model(config, preset_spec) == DEFAULT_TTS_PRESET_MODEL


def test_resolve_tts_model_supports_reference_clone_specs() -> None:
    config = AppConfig(tts=TTSConfig(mode="clone"))
    reference_spec = ReferenceCloneSpec(
        identity="fake:reference_clone:ref-1",
        provider="fake",
        payload=ReferenceClonePayload(reference_fingerprint="ref-1"),
    )

    assert _resolve_tts_model(config, reference_spec) == DEFAULT_TTS_CLONE_MODEL


def test_build_tts_backend_rejects_clone_mode_for_backend_without_clone_capability() -> None:
    config = AppConfig(tts=TTSConfig(provider="openai-compatible", mode="clone"))

    with pytest.raises(RuntimeError, match="Clone mode is not supported"):
        build_tts_backend(config)


def test_build_tts_backend_supports_vllm_omni_clone_mode() -> None:
    config = AppConfig(tts=TTSConfig(provider="vllm-omni", mode="clone"))

    backend = build_tts_backend(config)

    assert isinstance(backend, VllmOmniTTSBackend)


def test_build_tts_backend_rejects_unknown_provider() -> None:
    config = AppConfig(tts=TTSConfig(provider="custom", mode="preset"))

    with pytest.raises(RuntimeError, match="Unsupported TTS provider"):
        build_tts_backend(config)


def test_build_tts_backend_rejects_legacy_openai_compatible_provider_name() -> None:
    config = AppConfig(tts=TTSConfig(provider="openai_compatible", mode="preset"))

    with pytest.raises(RuntimeError, match="Unsupported TTS provider"):
        build_tts_backend(config)


def test_resolve_openai_compatible_api_key_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PODTRAN_TTS_API_KEY", "env-key")

    assert _resolve_openai_compatible_api_key() == "env-key"


def test_resolve_vllm_omni_api_key_prefers_config_then_env(monkeypatch: pytest.MonkeyPatch) -> None:
    config = AppConfig(tts={"vllm_omni": {"api_key": "config-key"}})
    monkeypatch.setenv("PODTRAN_VLLM_OMNI_API_KEY", "env-key")

    assert _resolve_vllm_omni_api_key(config) == "config-key"


def test_vllm_omni_backend_sends_custom_voice_request_body(tmp_path: Path) -> None:
    config = AppConfig(
        tts={
            "provider": "vllm-omni",
            "mode": "preset",
            "base_url": "http://localhost:8091/v1",
            "vllm_omni": {"language": "zh", "instructions": "Warm broadcast tone.", "api_key": "secret"},
        }
    )
    backend = VllmOmniTTSBackend(config)
    output_path = tmp_path / "speech.wav"
    captured: dict[str, object] = {}

    class _Response:
        content = b"wav"

        def raise_for_status(self) -> None:
            return None

    class _Client:
        def post(self, url: str, headers: dict[str, str], json: dict[str, object]) -> _Response:
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return _Response()

    backend.client = _Client()  # type: ignore[assignment]

    backend.synthesize("你好", PresetVoiceSpec(identity="preset:vivian", voice_name="vivian"), "Qwen/Qwen3-TTS", output_path)

    assert output_path.read_bytes() == b"wav"
    assert captured["url"] == "http://localhost:8091/v1/audio/speech"
    assert captured["headers"] == {"Authorization": "Bearer secret"}
    assert captured["json"] == {
        "model": "Qwen/Qwen3-TTS",
        "input": "你好",
        "response_format": "wav",
        "language": "zh",
        "instructions": "Warm broadcast tone.",
        "voice": "vivian",
        "task_type": "CustomVoice",
    }


def test_vllm_omni_backend_sends_base_clone_request_body_without_auth_when_key_missing(tmp_path: Path) -> None:
    reference_audio = tmp_path / "reference.wav"
    reference_audio.write_bytes(b"wav")
    config = AppConfig(
        tts={
            "provider": "vllm-omni",
            "mode": "clone",
            "base_url": "http://localhost:8091/v1",
            "vllm_omni": {"language": "Auto", "instructions": "", "x_vector_only_mode": True},
        }
    )
    backend = VllmOmniTTSBackend(config)
    output_path = tmp_path / "speech.wav"
    captured: dict[str, object] = {}

    class _Response:
        content = b"wav"

        def raise_for_status(self) -> None:
            return None

    class _Client:
        def post(self, url: str, headers: dict[str, str], json: dict[str, object]) -> _Response:
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return _Response()

    backend.client = _Client()  # type: ignore[assignment]
    spec = ReferenceCloneSpec(
        identity="vllm-omni:reference_clone:ref-1",
        provider="vllm-omni",
        payload=ReferenceClonePayload(
            reference_fingerprint="ref-1",
            reference_audio_path=str(reference_audio),
            reference_text="reference text",
        ),
    )

    backend.synthesize("你好", spec, "Qwen/Qwen3-TTS-Base", output_path)

    payload = captured["json"]
    assert output_path.read_bytes() == b"wav"
    assert captured["url"] == "http://localhost:8091/v1/audio/speech"
    assert captured["headers"] == {}
    assert payload["task_type"] == "Base"
    assert payload["ref_text"] == "reference text"
    assert payload["x_vector_only_mode"] is True
    assert str(payload["ref_audio"]).startswith("data:audio/")


def test_synthesize_segments_requires_source_audio_in_clone_mode(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.ensure()
    write_json(paths.translated_json, [_segment()])
    config = AppConfig(tts=TTSConfig(mode="clone"))

    with pytest.raises(RuntimeError, match="Clone mode requires source audio"):
        synthesize_segments(paths.translated_json, paths.translated_json, config, paths)


def test_synthesize_segments_raises_when_clone_voice_target_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    paths.ensure()
    write_json(paths.translated_json, [_segment()])
    config = AppConfig(tts=TTSConfig(mode="clone"))

    monkeypatch.setattr("podtran.tts.build_tts_backend", lambda cfg: _DummyBackend())
    monkeypatch.setattr(
        "podtran.tts._resolve_voice_targets",
        lambda config, paths, segments, source_audio, source_audio_fingerprint, cache_store, fingerprints, progress_callback=None: {},
    )

    with pytest.raises(RuntimeError, match="Missing resolved clone voice target"):
        synthesize_segments(paths.translated_json, paths.translated_json, config, paths, source_audio=tmp_path / "source.wav")


def test_synthesize_segments_reuses_shared_tts_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first_paths = _paths(tmp_path, "task-1")
    second_paths = _paths(tmp_path, "task-2")
    first_paths.ensure()
    second_paths.ensure()
    write_json(first_paths.translated_json, [_segment()])
    write_json(second_paths.translated_json, [_segment()])

    config = AppConfig(tts=TTSConfig(mode="preset"))
    cache_store = CacheStore(first_paths.cache_dir)
    fingerprints = FingerprintService(first_paths.cache_indexes_dir)
    backend = _DummyBackend()

    monkeypatch.setattr("podtran.tts.build_tts_backend", lambda cfg: backend)
    monkeypatch.setattr("podtran.tts.probe_duration", lambda ffprobe_path, path: 1.0)

    synthesize_segments(
        first_paths.translated_json,
        first_paths.translated_json,
        config,
        first_paths,
        cache_store=cache_store,
        fingerprints=fingerprints,
    )
    synthesize_segments(
        second_paths.translated_json,
        second_paths.translated_json,
        config,
        second_paths,
        cache_store=cache_store,
        fingerprints=fingerprints,
    )

    second_segments = read_model_list(second_paths.translated_json, SegmentRecord)
    assert backend.calls == 1
    assert second_segments[0].status == "completed"
    assert Path(second_segments[0].tts_audio_path).exists()


def test_synthesize_segments_does_not_reuse_shared_tts_cache_when_preset_voice_changes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first_paths = _paths(tmp_path, "task-1")
    second_paths = _paths(tmp_path, "task-2")
    first_paths.ensure()
    second_paths.ensure()
    write_json(first_paths.translated_json, [_segment()])
    write_json(second_paths.translated_json, [_segment().model_copy(update={"voice": "Serena"})])

    config = AppConfig(tts=TTSConfig(mode="preset"))
    cache_store = CacheStore(first_paths.cache_dir)
    fingerprints = FingerprintService(first_paths.cache_indexes_dir)
    backend = _DummyBackend()

    monkeypatch.setattr("podtran.tts.build_tts_backend", lambda cfg: backend)
    monkeypatch.setattr("podtran.tts.probe_duration", lambda ffprobe_path, path: 1.0)

    synthesize_segments(
        first_paths.translated_json,
        first_paths.translated_json,
        config,
        first_paths,
        cache_store=cache_store,
        fingerprints=fingerprints,
    )
    synthesize_segments(
        second_paths.translated_json,
        second_paths.translated_json,
        config,
        second_paths,
        cache_store=cache_store,
        fingerprints=fingerprints,
    )

    assert backend.calls == 2


def test_synthesize_segments_reports_progress_for_preset_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    paths.ensure()
    segments = [_segment(), _segment().model_copy(update={"segment_id": "seg_2"})]
    write_json(paths.translated_json, segments)
    events: list[tuple[int, int, str]] = []

    monkeypatch.setattr("podtran.tts.build_tts_backend", lambda cfg: _DummyBackend())
    monkeypatch.setattr("podtran.tts.probe_duration", lambda ffprobe_path, path: 1.0)

    synthesize_segments(
        paths.translated_json,
        paths.translated_json,
        AppConfig(tts=TTSConfig(mode="preset")),
        paths,
        progress_callback=lambda completed, total, message: events.append((completed, total, message)),
    )

    assert events[0][2] == "Resolving voices"
    assert events[1][2] == "Using preset voices"
    assert events[-1][2] == "Synthesis complete"
    assert events[-1][0] == events[-1][1]


def test_synthesize_segments_runs_distinct_work_items_concurrently(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    paths.ensure()
    segments = [
        _segment_with_text("seg_1", "你好一"),
        _segment_with_text("seg_2", "你好二"),
    ]
    write_json(paths.translated_json, segments)
    backend = _TrackingBackend()

    monkeypatch.setattr("podtran.tts.build_tts_backend", lambda cfg: backend)
    monkeypatch.setattr("podtran.tts.probe_duration", lambda ffprobe_path, path: 1.0)

    synthesize_segments(
        paths.translated_json,
        paths.translated_json,
        AppConfig(tts=TTSConfig(mode="preset", max_concurrency=2)),
        paths,
    )

    synthesized = read_model_list(paths.translated_json, SegmentRecord)
    assert backend.calls == 2
    assert backend.max_active_calls >= 2
    assert all(item.status == "completed" for item in synthesized)


def test_synthesize_segments_deduplicates_identical_work_items(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    paths.ensure()
    segments = [
        _segment_with_text("seg_1", "重复文本"),
        _segment_with_text("seg_2", "重复文本"),
    ]
    write_json(paths.translated_json, segments)
    backend = _TrackingBackend(sleep_seconds=0.01)

    monkeypatch.setattr("podtran.tts.build_tts_backend", lambda cfg: backend)
    monkeypatch.setattr("podtran.tts.probe_duration", lambda ffprobe_path, path: 1.0)

    synthesize_segments(
        paths.translated_json,
        paths.translated_json,
        AppConfig(tts=TTSConfig(mode="preset", max_concurrency=4)),
        paths,
    )

    synthesized = read_model_list(paths.translated_json, SegmentRecord)
    assert backend.calls == 1
    assert synthesized[0].status == "completed"
    assert synthesized[1].status == "completed"
    assert synthesized[0].tts_audio_path != synthesized[1].tts_audio_path
    assert Path(synthesized[0].tts_audio_path).exists()
    assert Path(synthesized[1].tts_audio_path).exists()


def test_synthesize_segments_isolates_worker_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    paths.ensure()
    segments = [
        _segment_with_text("seg_1", "good"),
        _segment_with_text("seg_2", "bad"),
    ]
    write_json(paths.translated_json, segments)
    backend = _TrackingBackend(fail_texts={"bad"}, sleep_seconds=0.01)

    monkeypatch.setattr("podtran.tts.build_tts_backend", lambda cfg: backend)
    monkeypatch.setattr("podtran.tts.probe_duration", lambda ffprobe_path, path: 1.0)

    synthesized = synthesize_segments(
        paths.translated_json,
        paths.translated_json,
        AppConfig(tts=TTSConfig(mode="preset", max_concurrency=2)),
        paths,
    )

    assert backend.calls == 2
    assert synthesized[0].status == "completed"
    assert synthesized[1].status == "failed"
    assert synthesized[1].error == "boom: bad"


def test_synthesize_segments_respects_max_concurrency_of_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    paths.ensure()
    segments = [
        _segment_with_text("seg_1", "你好一"),
        _segment_with_text("seg_2", "你好二"),
    ]
    write_json(paths.translated_json, segments)
    backend = _TrackingBackend()

    monkeypatch.setattr("podtran.tts.build_tts_backend", lambda cfg: backend)
    monkeypatch.setattr("podtran.tts.probe_duration", lambda ffprobe_path, path: 1.0)

    synthesize_segments(
        paths.translated_json,
        paths.translated_json,
        AppConfig(tts=TTSConfig(mode="preset", max_concurrency=1)),
        paths,
    )

    assert backend.calls == 2
    assert backend.max_active_calls == 1


def test_synthesize_segments_builds_backend_once_per_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    paths.ensure()
    segments = [
        _segment_with_text("seg_1", "你好一"),
        _segment_with_text("seg_2", "你好二"),
    ]
    write_json(paths.translated_json, segments)
    backend = _TrackingBackend(sleep_seconds=0.01)
    build_calls = 0

    def build_backend(_config: AppConfig) -> _TrackingBackend:
        nonlocal build_calls
        build_calls += 1
        return backend

    monkeypatch.setattr("podtran.tts.build_tts_backend", build_backend)
    monkeypatch.setattr("podtran.tts.probe_duration", lambda ffprobe_path, path: 1.0)

    synthesize_segments(
        paths.translated_json,
        paths.translated_json,
        AppConfig(tts=TTSConfig(mode="preset", max_concurrency=1)),
        paths,
    )

    assert build_calls == 1
    assert backend.calls == 2


def test_synthesize_segments_preserves_completed_results_when_worker_interrupts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    paths.ensure()
    segments = [
        _segment_with_text("seg_1", "你好一"),
        _segment_with_text("seg_2", "你好二"),
    ]
    write_json(paths.translated_json, segments)
    backend = _InterruptingBackend()

    monkeypatch.setattr("podtran.tts.build_tts_backend", lambda cfg: backend)
    monkeypatch.setattr("podtran.tts.probe_duration", lambda ffprobe_path, path: 1.0)

    with pytest.raises(KeyboardInterrupt):
        synthesize_segments(
            paths.translated_json,
            paths.translated_json,
            AppConfig(tts=TTSConfig(mode="preset", max_concurrency=1)),
            paths,
        )

    synthesized = read_model_list(paths.translated_json, SegmentRecord)
    assert backend.calls == 2
    assert synthesized[0].status == "completed"
    assert Path(synthesized[0].tts_audio_path).exists()
    assert synthesized[1].status == "pending"
    assert synthesized[1].tts_audio_path == ""


def test_synthesize_segments_stops_queued_work_quickly_on_sigint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    paths.ensure()
    segments = [
        _segment_with_text("seg_1", "你好一"),
        _segment_with_text("seg_2", "你好二"),
    ]
    write_json(paths.translated_json, segments)
    backend = _TrackingBackend(sleep_seconds=0.2)

    monkeypatch.setattr("podtran.tts.build_tts_backend", lambda cfg: backend)
    monkeypatch.setattr("podtran.tts.probe_duration", lambda ffprobe_path, path: 1.0)

    timer = threading.Timer(0.05, lambda: signal.raise_signal(signal.SIGINT))
    timer.start()
    start = time.perf_counter()
    try:
        with pytest.raises(KeyboardInterrupt):
            synthesize_segments(
                paths.translated_json,
                paths.translated_json,
                AppConfig(tts=TTSConfig(mode="preset", max_concurrency=1)),
                paths,
            )
    finally:
        timer.cancel()

    elapsed = time.perf_counter() - start
    time.sleep(0.3)

    assert elapsed < 0.35
    assert backend.calls == 1
