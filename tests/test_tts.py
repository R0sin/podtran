from pathlib import Path
import threading
import time

import pytest

from podtran.artifacts import ArtifactPaths, read_model_list, write_json
from podtran.cache_store import CacheStore
from podtran.config import AppConfig, DEFAULT_TTS_CLONE_MODEL, DEFAULT_TTS_PRESET_MODEL, TTSConfig
from podtran.fingerprints import FingerprintService
from podtran.models import ResolvedVoiceTarget, SegmentRecord
from podtran.tts import _resolve_tts_key, _resolve_tts_model, build_tts_backend, synthesize_segments


class _DummyBackend:
    def __init__(self) -> None:
        self.calls = 0

    def synthesize(self, text: str, target: ResolvedVoiceTarget, output_path: Path) -> None:
        self.calls += 1
        output_path.write_bytes(b"wav")


class _TrackingBackend:
    def __init__(self, *, fail_texts: set[str] | None = None, sleep_seconds: float = 0.05) -> None:
        self.calls = 0
        self.active_calls = 0
        self.max_active_calls = 0
        self.fail_texts = fail_texts or set()
        self.sleep_seconds = sleep_seconds
        self._lock = threading.Lock()

    def synthesize(self, text: str, target: ResolvedVoiceTarget, output_path: Path) -> None:
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



def test_resolve_tts_model_switches_back_to_preset_model_for_clone_fallback() -> None:
    config = AppConfig(tts=TTSConfig(voice_mode="clone"))

    clone_target = ResolvedVoiceTarget(speaker="SPEAKER_00", provider="dashscope", mode="clone", voice="voice-token")
    preset_target = ResolvedVoiceTarget(speaker="SPEAKER_00", provider="dashscope", mode="preset", voice="Cherry")

    assert _resolve_tts_model(config, clone_target) == DEFAULT_TTS_CLONE_MODEL
    assert _resolve_tts_model(config, preset_target) == DEFAULT_TTS_PRESET_MODEL



def test_build_tts_backend_rejects_clone_mode_for_non_dashscope_backend() -> None:
    config = AppConfig(tts=TTSConfig(provider="custom", voice_mode="clone"))

    with pytest.raises(RuntimeError, match="Clone mode is not supported"):
        build_tts_backend(config)


def test_resolve_tts_key_prefers_provider_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    config = AppConfig(providers={"dashscope": {"api_key": "dash-key"}})
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key")

    assert _resolve_tts_key(config) == "dash-key"



def test_synthesize_segments_requires_source_audio_in_clone_mode(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.ensure()
    write_json(paths.translated_json, [_segment()])
    config = AppConfig(tts=TTSConfig(voice_mode="clone"))

    with pytest.raises(RuntimeError, match="Clone mode requires source audio"):
        synthesize_segments(paths.translated_json, paths.translated_json, config, paths)



def test_synthesize_segments_raises_when_clone_voice_target_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _paths(tmp_path)
    paths.ensure()
    write_json(paths.translated_json, [_segment()])
    config = AppConfig(tts=TTSConfig(voice_mode="clone"))

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

    config = AppConfig(tts=TTSConfig(voice_mode="preset"))
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

    config = AppConfig(tts=TTSConfig(voice_mode="preset"))
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
        AppConfig(tts=TTSConfig(voice_mode="preset")),
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
        AppConfig(tts=TTSConfig(voice_mode="preset", max_concurrency=2)),
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
        AppConfig(tts=TTSConfig(voice_mode="preset", max_concurrency=4)),
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
        AppConfig(tts=TTSConfig(voice_mode="preset", max_concurrency=2)),
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
        AppConfig(tts=TTSConfig(voice_mode="preset", max_concurrency=1)),
        paths,
    )

    assert backend.calls == 2
    assert backend.max_active_calls == 1
