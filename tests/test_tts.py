from pathlib import Path

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
