import pytest

from pathlib import Path

from podtran.artifacts import ArtifactPaths, read_model_list, write_json
from podtran.cache_store import CacheStore
from podtran.config import AppConfig, TTSConfig
from podtran.fingerprints import FingerprintService
from podtran.models import SegmentRecord, VoiceProfile
from podtran.voices import VoiceProfileManager, select_reference_candidate


class _UnexpectedProvider:
    def ensure_voice(self, reference_audio: Path, target_model: str, preferred_name: str) -> str:
        raise AssertionError("provider should not be called")


class _FixedProvider:
    def __init__(self, token: str) -> None:
        self.token = token
        self.calls = 0

    def ensure_voice(self, reference_audio: Path, target_model: str, preferred_name: str) -> str:
        self.calls += 1
        return self.token



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



def test_voice_profile_manager_preferred_name_uses_normalized_speaker_without_prefix(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    manager = VoiceProfileManager(AppConfig(tts=TTSConfig(voice_mode="clone")), paths)

    assert manager._preferred_name("SPEAKER_00") == "speaker_00"
    assert manager._preferred_name("Host A / Guest B") == "host_a_guest_b"



def test_voice_profile_manager_reuses_cached_voice_profile(tmp_path: Path) -> None:
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
            VoiceProfile(
                speaker="SPEAKER_00",
                provider="dashscope",
                target_model="qwen3-tts-vc-2026-01-22",
                voice_token="voice-token-1",
                ref_audio_path=str(ref_audio.resolve()),
                ref_text_path=str(ref_text.resolve()),
                source_audio_path=str(source_audio.resolve()),
                status="completed",
            )
        ],
    )
    manager = VoiceProfileManager(AppConfig(tts=TTSConfig(voice_mode="clone")), paths)
    manager._provider = _UnexpectedProvider()

    resolved = manager.resolve_voice_targets([_segment("seg_1", 0.0, 12.0)], source_audio)

    assert resolved["SPEAKER_00"].mode == "clone"
    assert resolved["SPEAKER_00"].voice == "voice-token-1"



def test_voice_profile_manager_raises_when_no_reference_is_available(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.ensure()
    source_audio = tmp_path / "source.wav"
    source_audio.write_bytes(b"wav")
    manager = VoiceProfileManager(AppConfig(tts=TTSConfig(voice_mode="clone")), paths)
    manager._provider = _UnexpectedProvider()

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



def test_voice_profile_manager_reuses_shared_voice_cache(tmp_path: Path) -> None:
    source_audio = tmp_path / "source.wav"
    source_audio.write_bytes(b"wav")
    first_paths = _paths(tmp_path, "task-1")
    second_paths = _paths(tmp_path, "task-2")
    first_paths.ensure()
    second_paths.ensure()
    cache_store = CacheStore(first_paths.cache_dir)
    fingerprints = FingerprintService(first_paths.cache_indexes_dir)

    first_manager = VoiceProfileManager(
        AppConfig(tts=TTSConfig(voice_mode="clone")),
        first_paths,
        cache_store=cache_store,
        fingerprints=fingerprints,
    )
    first_manager._provider = _FixedProvider("voice-token-1")
    first_manager._export_reference_audio = _stub_export(first_paths)  # type: ignore[method-assign]

    second_manager = VoiceProfileManager(
        AppConfig(tts=TTSConfig(voice_mode="clone")),
        second_paths,
        cache_store=cache_store,
        fingerprints=fingerprints,
    )
    second_manager._provider = _UnexpectedProvider()
    second_manager._export_reference_audio = _stub_export(second_paths)  # type: ignore[method-assign]

    segments = [_segment("seg_1", 0.0, 12.0)]
    first_resolved = first_manager.resolve_voice_targets(segments, source_audio, source_audio_fingerprint="audio-1")
    second_resolved = second_manager.resolve_voice_targets(segments, source_audio, source_audio_fingerprint="audio-1")

    assert first_resolved["SPEAKER_00"].voice == "voice-token-1"
    assert second_resolved["SPEAKER_00"].voice == "voice-token-1"



def test_voice_profile_manager_reports_progress(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.ensure()
    source_audio = tmp_path / "source.wav"
    source_audio.write_bytes(b"wav")
    manager = VoiceProfileManager(AppConfig(tts=TTSConfig(voice_mode="clone")), paths)
    manager._provider = _FixedProvider("voice-token-1")
    manager._export_reference_audio = _stub_export(paths)  # type: ignore[method-assign]
    events: list[tuple[int, int, str]] = []

    manager.resolve_voice_targets(
        [_segment("seg_1", 0.0, 12.0)],
        source_audio,
        progress_callback=lambda completed, total, message: events.append((completed, total, message)),
    )

    assert events[0] == (0, 1, "Resolving cloned voices")
    assert events[-1] == (1, 1, "Voice resolution complete")
    assert any("Enrolled voice" in message for _, _, message in events)
