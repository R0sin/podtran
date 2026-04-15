from pathlib import Path

from podtran.audio import extract_audio_chunk


def test_extract_audio_chunk_uses_relative_duration_when_start_and_end_are_provided(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_ffmpeg(ffmpeg_path: str, args: list[str]) -> None:
        captured["ffmpeg_path"] = ffmpeg_path
        captured["args"] = args

    monkeypatch.setattr("podtran.audio.run_ffmpeg", fake_run_ffmpeg)

    result = extract_audio_chunk(
        "ffmpeg",
        Path("input.wav"),
        Path("output.wav"),
        start=152.740,
        end=156.120,
    )

    assert result == Path("output.wav")
    assert captured["ffmpeg_path"] == "ffmpeg"
    assert captured["args"] == [
        "-ss",
        "152.740",
        "-i",
        "input.wav",
        "-t",
        "3.380",
        "-ar",
        "24000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        "output.wav",
    ]
