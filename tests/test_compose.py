from __future__ import annotations

from pathlib import Path

import pytest

from podtran.compose import (
    compose_output,
    build_interleave_chunks,
    build_replace_chunks,
)
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
        *,
        speed: float = 1.0,
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

    def normalize_audio(
        self, ffmpeg_path: str, source: Path, output: Path, *, speed: float = 1.0
    ) -> Path:
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


@pytest.mark.parametrize("mode", ["interleave", "replace"])
@pytest.mark.parametrize("speeds", [(1.0, 1.0), (0.5, 2.0), (1.25, 0.75)])
def test_compose_applies_independent_speeds_and_keeps_silence(
    tmp_path, monkeypatch, mode, speeds
):
    tts = tmp_path / "tts.wav"
    tts.write_bytes(b"tts")
    segment = _segment(str(tts)).model_copy(update={"start": 1.0, "end": 3.0})
    missing = segment.model_copy(
        update={
            "segment_id": "missing",
            "start": 4.0,
            "end": 6.0,
            "status": "failed",
            "tts_audio_path": None,
        }
    )
    config = AppConfig(compose={"english_speed": speeds[0], "chinese_speed": speeds[1]})
    extracts, translated, silences = [], [], []

    def extract(_, source, output, start, end, *, speed):
        extracts.append((start, end, speed))
        return output

    def normalize(_, source, output, *, speed):
        translated.append((source, speed))
        return output

    def silence(_, output, duration):
        silences.append(duration)
        return output

    monkeypatch.setattr("podtran.compose.extract_audio_chunk", extract)
    monkeypatch.setattr("podtran.compose.normalize_audio", normalize)
    monkeypatch.setattr("podtran.compose.create_silence", silence)
    if mode == "interleave":
        chunks = build_interleave_chunks(
            tmp_path / "source.wav",
            [segment, missing],
            config,
            tmp_path,
            audio_duration=8,
        )
        assert extracts == [
            (0.0, 3.0, speeds[0]),
            (3.0, 6.0, speeds[0]),
            (6.0, None, speeds[0]),
        ]
        assert silences == [200, 400]
        assert len(chunks) == 6
    else:
        chunks = build_replace_chunks(
            tmp_path / "source.wav", [segment, missing], config, tmp_path
        )
        assert extracts == []
        assert silences == [1000, 1000, 2000]
        assert len(chunks) == 4
    assert translated == [(tts, speeds[1])]
