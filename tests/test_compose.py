from pathlib import Path

from podtran.compose import compose_output
from podtran.config import AppConfig
from podtran.models import SegmentRecord


class _ChunkRecorder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def extract_audio_chunk(
        self,
        ffmpeg_path: str,
        source: Path,
        output: Path,
        start: float | None,
        end: float | None,
    ) -> Path:
        self.calls.append(f"extract:{output.name}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"wav")
        return output

    def create_silence(self, ffmpeg_path: str, output: Path, duration_ms: int) -> Path:
        self.calls.append(f"silence:{output.name}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"wav")
        return output

    def normalize_audio(self, ffmpeg_path: str, source: Path, output: Path) -> Path:
        self.calls.append(f"normalize:{output.name}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"wav")
        return output


def _segment(tts_audio_path: str) -> SegmentRecord:
    return SegmentRecord(
        segment_id="seg_1",
        block_id="block_1",
        start=0.0,
        end=1.0,
        text="hello",
        speaker="SPEAKER_00",
        voice="Cherry",
        text_zh="你好",
        status="completed",
        tts_audio_path=tts_audio_path,
    )


def test_compose_output_reports_progress(tmp_path: Path, monkeypatch) -> None:
    source_audio = tmp_path / "source.wav"
    source_audio.write_bytes(b"source")
    tts_audio = tmp_path / "tts.wav"
    tts_audio.write_bytes(b"tts")
    output_path = tmp_path / "final" / "episode.interleave.mp3"
    temp_dir = tmp_path / "temp"
    recorder = _ChunkRecorder()
    events: list[tuple[int, int, str]] = []

    monkeypatch.setattr(
        "podtran.compose.reset_temp_dir",
        lambda path, root: path.mkdir(parents=True, exist_ok=True),
    )
    monkeypatch.setattr(
        "podtran.compose.probe_duration", lambda ffprobe_path, path: 2.0
    )
    monkeypatch.setattr(
        "podtran.compose.extract_audio_chunk", recorder.extract_audio_chunk
    )
    monkeypatch.setattr("podtran.compose.create_silence", recorder.create_silence)
    monkeypatch.setattr("podtran.compose.normalize_audio", recorder.normalize_audio)
    monkeypatch.setattr(
        "podtran.compose.concat_audio",
        lambda ffmpeg_path, chunks, output, bitrate: (
            output.parent.mkdir(parents=True, exist_ok=True),
            output.write_bytes(b"mp3"),
            output,
        )[2],
    )

    compose_output(
        source_audio,
        [_segment(str(tts_audio))],
        AppConfig(),
        temp_dir,
        output_path,
        progress_callback=lambda completed, total, message: events.append(
            (completed, total, message)
        ),
    )

    assert events[0] == (0, 6, "Scanning segments")
    assert events[-1] == (6, 6, "Compose complete")
    assert any(message == "Building chunks" for _, _, message in events)
    assert any(message == "Concatenating audio" for _, _, message in events)
