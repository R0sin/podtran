from pathlib import Path

import pytest

from podtran.artifacts import ArtifactPaths, read_model_list, write_json
from podtran.cache_store import CacheStore
from podtran.config import AppConfig, DEFAULT_TTS_ENROLLMENT_MODEL, TTSConfig
from podtran.fingerprints import FingerprintService
from podtran.models import (
    ProviderClonePayload,
    ReferenceClonePayload,
    ReferenceCloneSpec,
    SegmentRecord,
    VoiceProfile,
)
from podtran.voices import (
    UNKNOWN_SPEAKER_TTS_SKIP_MESSAGE,
    DashScopeCloneProvider,
    VoiceResolver,
    resolve_dashscope_api_key,
    select_reference_candidate,
)


class _UnexpectedProvider:
    def create_voice_spec(self, reference_audio: Path, reference_text: str, target_model: str, preferred_name: str, reference_fingerprint: str):
        _ = (reference_audio, reference_text, target_model, preferred_name, reference_fingerprint)
        raise AssertionError("provider should not be called")


class _FixedProvider:
    def __init__(self, token: str) -> None:
        self.token = token
        self.calls = 0

    def create_voice_spec(self, reference_audio: Path, reference_text: str, target_model: str, preferred_name: str, reference_fingerprint: str):
        _ = (reference_audio, reference_text, target_model, preferred_name)
        self.calls += 1
        return ReferenceCloneSpec(
            identity=f"fake:reference_clone:{reference_fingerprint}",
            provider="fake",
            payload=ReferenceClonePayload(
                reference_fingerprint=reference_fingerprint,
                reference_audio_path=str(reference_audio),
                reference_text=reference_text,
            ),
        )


def _paths(tmp_path: Path, task_id: str = "task-1") -> ArtifactPaths:
    return ArtifactPaths.from_task_id(tmp_path, task_id)


def _segment(segment_id: str, start: float, end: float, speaker: str = "SPEAKER_00", text: str | None = None) -> SegmentRecord:
    return SegmentRecord(
        segment_id=segment_id,
        block_id=f"block_{segment_id}",
        start=start,
        end=end,
        text=text or "this is a suitable reference sentence for cloning quality",
        speaker=speaker,
        voice="Cherry",
        text_zh="你好",
    )


def _stub_export(paths: ArtifactPaths):
    def _export(source_audio: Path, speaker: str, candidate, build_dir: Path) -> tuple[Path, Path]:
        _ = (source_audio, build_dir)
        ref_audio = paths.refs_dir / f"{speaker}.wav"
        ref_text = paths.refs_dir / f"{speaker}.txt"
        ref_audio.write_bytes(b"ref")
        ref_text.write_text(candidate.text, encoding="utf-8")
        return ref_audio, ref_text

    return _export


def test_select_reference_candidate_merges_adjacent_segments_to_target_duration() -> None:
    segments = [
        _segment("seg_1", 0.0, 6.0, text="this speaker says something useful for cloning"),
        _segment("seg_2", 6.2, 12.0, text="and keeps talking with enough words to qualify"),
    ]

    candidate = select_reference_candidate(segments, "SPEAKER_00", pause_threshold=0.8, preferred_min_duration=10.0, preferred_max_duration=20.0)

    assert candidate is not None
    assert candidate.start == 0.0
    assert candidate.end == 12.0
    assert len(candidate.segments) == 2


def test_select_reference_candidate_accepts_shorter_clip_when_it_has_clear_speech() -> None:
    segments = [
        _segment("seg_1", 0.0, 3.2, text="this opening phrase is clear enough for enrollment"),
        _segment("seg_2", 4.8, 7.5, text="and the speaker resumes after a short pause"),
    ]

    candidate = select_reference_candidate(
        segments,
        "SPEAKER_00",
        pause_threshold=2.0,
        preferred_min_duration=10.0,
        preferred_max_duration=20.0,
    )

    assert candidate is not None
    assert candidate.start == 0.0
    assert candidate.end == 7.5


def test_voice_resolver_preferred_name_uses_normalized_speaker_without_prefix(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    manager = VoiceResolver(AppConfig(tts=TTSConfig(mode="clone")), paths)

    assert manager._preferred_name("SPEAKER_00") == "speaker_00"
    assert manager._preferred_name("Host A / Guest B") == "host_a_guest_b"


def test_voice_resolver_reuses_cached_voice_profile(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.ensure()
    source_audio = tmp_path / "source.wav"
    source_audio.write_bytes(b"wav")
    ref_audio = paths.refs_dir / "SPEAKER_00.wav"
    ref_audio.write_bytes(b"ref")
    ref_text = paths.refs_dir / "SPEAKER_00.txt"
    ref_text.write_text("reference", encoding="utf-8")
    write_json(
        paths.voices_json,
        [
            {
                "speaker": "SPEAKER_00",
                "provider": "dashscope",
                "target_model": "qwen3-tts-vc-2026-01-22",
                "voice_token": "voice-token-1",
                "ref_audio_path": str(ref_audio.resolve()),
                "ref_text_path": str(ref_text.resolve()),
                "source_audio_path": str(source_audio.resolve()),
                "status": "completed",
            }
        ],
    )
    manager = VoiceResolver(AppConfig(tts=TTSConfig(mode="clone")), paths)
    manager._clone_provider = _UnexpectedProvider()
    manager._export_reference_audio = _stub_export(paths)  # type: ignore[method-assign]

    resolved = manager.resolve_voice_targets([_segment("seg_1", 0.0, 12.0)], source_audio)

    assert resolved["SPEAKER_00"].spec.kind == "provider_clone"
    assert resolved["SPEAKER_00"].spec.payload.voice_token == "voice-token-1"


def test_voice_resolver_raises_when_no_reference_is_available(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.ensure()
    source_audio = tmp_path / "source.wav"
    source_audio.write_bytes(b"wav")
    manager = VoiceResolver(AppConfig(tts=TTSConfig(mode="clone")), paths)
    manager._clone_provider = _UnexpectedProvider()

    with pytest.raises(RuntimeError, match="Voice clone failed for 1 speaker"):
        manager.resolve_voice_targets(
            [
                _segment("seg_1", 0.0, 2.4, text="short speech here"),
                _segment("seg_2", 3.5, 5.7, text="still not enough contiguous speech"),
            ],
            source_audio,
        )

    voice_profiles = read_model_list(paths.voices_json, VoiceProfile)
    assert voice_profiles[0].status == "failed"
    assert "No qualifying reference audio" in (voice_profiles[0].error or "")


def test_voice_resolver_skips_unknown_speaker_without_raising(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.ensure()
    source_audio = tmp_path / "source.wav"
    source_audio.write_bytes(b"wav")
    manager = VoiceResolver(AppConfig(tts=TTSConfig(mode="clone")), paths)
    manager._clone_provider = _UnexpectedProvider()

    resolved = manager.resolve_voice_targets(
        [
            _segment("seg_1", 0.0, 2.4, speaker="UNKNOWN", text="short speech here"),
            _segment("seg_2", 3.5, 5.7, speaker="UNKNOWN", text="still not enough contiguous speech"),
        ],
        source_audio,
    )

    voice_profiles = read_model_list(paths.voices_json, VoiceProfile)
    assert resolved == {}
    assert voice_profiles[0].speaker == "UNKNOWN"
    assert voice_profiles[0].status == "failed"
    assert voice_profiles[0].error == UNKNOWN_SPEAKER_TTS_SKIP_MESSAGE


def test_voice_resolver_uses_reference_clone_specs_for_vllm_omni(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.ensure()
    source_audio = tmp_path / "source.wav"
    source_audio.write_bytes(b"wav")
    manager = VoiceResolver(
        AppConfig(
            tts=TTSConfig(
                provider="vllm-omni",
                mode="clone",
            ),
            providers={"vllm_omni": {"base_url": "http://localhost:8091/v1"}},
        ),
        paths,
    )
    manager._export_reference_audio = _stub_export(paths)  # type: ignore[method-assign]

    resolved = manager.resolve_voice_targets([_segment("seg_1", 0.0, 12.0)], source_audio)

    spec = resolved["SPEAKER_00"].spec
    assert spec.kind == "reference_clone"
    assert spec.provider == "vllm-omni"
    assert spec.payload.reference_text == "this is a suitable reference sentence for cloning quality"
    assert Path(spec.payload.reference_audio_path).exists()


def test_voice_resolver_uses_reference_clone_specs_for_qwen_local(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.ensure()
    source_audio = tmp_path / "source.wav"
    source_audio.write_bytes(b"wav")
    manager = VoiceResolver(AppConfig(tts=TTSConfig(provider="qwen-local", mode="clone")), paths)
    manager._export_reference_audio = _stub_export(paths)  # type: ignore[method-assign]

    resolved = manager.resolve_voice_targets([_segment("seg_1", 0.0, 12.0)], source_audio)

    spec = resolved["SPEAKER_00"].spec
    assert spec.kind == "reference_clone"
    assert spec.provider == "qwen-local"
    assert spec.payload.reference_text == "this is a suitable reference sentence for cloning quality"
    assert Path(spec.payload.reference_audio_path).exists()


def test_voice_resolver_reuses_shared_voice_cache(tmp_path: Path) -> None:
    source_audio = tmp_path / "source.wav"
    source_audio.write_bytes(b"wav")
    first_paths = _paths(tmp_path, "task-1")
    second_paths = _paths(tmp_path, "task-2")
    first_paths.ensure()
    second_paths.ensure()
    cache_store = CacheStore(first_paths.cache_dir)
    fingerprints = FingerprintService(first_paths.cache_indexes_dir)

    first_manager = VoiceResolver(
        AppConfig(tts=TTSConfig(mode="clone")),
        first_paths,
        cache_store=cache_store,
        fingerprints=fingerprints,
    )
    first_manager._clone_provider = _FixedProvider("voice-token-1")
    first_manager._export_reference_audio = _stub_export(first_paths)  # type: ignore[method-assign]

    second_manager = VoiceResolver(
        AppConfig(tts=TTSConfig(mode="clone")),
        second_paths,
        cache_store=cache_store,
        fingerprints=fingerprints,
    )
    second_manager._clone_provider = _UnexpectedProvider()
    second_manager._export_reference_audio = _stub_export(second_paths)  # type: ignore[method-assign]

    segments = [_segment("seg_1", 0.0, 12.0)]
    first_resolved = first_manager.resolve_voice_targets(segments, source_audio, source_audio_fingerprint="audio-1")
    second_resolved = second_manager.resolve_voice_targets(segments, source_audio, source_audio_fingerprint="audio-1")

    assert first_resolved["SPEAKER_00"].spec.kind == "reference_clone"
    assert second_resolved["SPEAKER_00"].spec.kind == "reference_clone"
    assert first_resolved["SPEAKER_00"].spec.identity == second_resolved["SPEAKER_00"].spec.identity


def test_voice_resolver_reports_progress(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.ensure()
    source_audio = tmp_path / "source.wav"
    source_audio.write_bytes(b"wav")
    manager = VoiceResolver(AppConfig(tts=TTSConfig(mode="clone")), paths)
    manager._clone_provider = _FixedProvider("voice-token-1")
    manager._export_reference_audio = _stub_export(paths)  # type: ignore[method-assign]
    events: list[tuple[int, int, str]] = []

    manager.resolve_voice_targets(
        [_segment("seg_1", 0.0, 12.0)],
        source_audio,
        progress_callback=lambda completed, total, message: events.append((completed, total, message)),
    )

    assert events[0] == (0, 1, "Resolving cloned voices")
    assert events[-1] == (1, 1, "Voice resolution complete")
    assert any(("Enrolled voice" in message) or ("Prepared reference voice" in message) for _, _, message in events)


def test_dashscope_clone_provider_uses_fixed_enrollment_model_and_derived_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = AppConfig(providers={"dashscope": {"tts_base_url": "https://tts.example.com/root/"}})
    provider = DashScopeCloneProvider(config)
    captured: dict[str, object] = {}

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"output": {"voice": "voice-token-1"}}

    class _Client:
        def post(self, url: str, headers: dict[str, str], json: dict[str, object]) -> _Response:
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return _Response()

    provider.client = _Client()  # type: ignore[assignment]
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key")
    reference_audio = tmp_path / "reference.wav"
    reference_audio.write_bytes(b"wav")

    spec = provider.create_voice_spec(reference_audio, "ref text", "clone-model", "speaker_00", "ref-1")

    assert spec.kind == "provider_clone"
    assert spec.payload == ProviderClonePayload(voice_token="voice-token-1")
    assert captured["url"] == "https://tts.example.com/root/services/audio/tts/customization"
    assert captured["json"]["model"] == DEFAULT_TTS_ENROLLMENT_MODEL


def test_resolve_dashscope_api_key_prefers_provider_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    config = AppConfig(providers={"dashscope": {"api_key": "dash-key"}})
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key")

    assert resolve_dashscope_api_key(config) == "dash-key"
