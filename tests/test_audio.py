from __future__ import annotations

from pathlib import Path

import pytest

from podtran.audio import extract_audio_chunk, normalize_audio, prepare_audio


def test_extract_audio_chunk_uses_relative_duration_when_start_and_end_are_provided(
    monkeypatch,
) -> None:
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
        "-t",
        "3.380",
        "-i",
        "input.wav",
        "-ar",
        "24000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        "output.wav",
    ]


@pytest.mark.parametrize("speed", [0.5, 1.0, 1.25, 2.0])
@pytest.mark.parametrize("end", [14.0, None])
def test_speed_filter_preserves_source_cut_boundaries(monkeypatch, speed, end):
    calls = []
    monkeypatch.setattr("podtran.audio.run_ffmpeg", lambda _, args: calls.append(args))
    extract_audio_chunk("ffmpeg", Path("in.wav"), Path("out.wav"), 10, end, speed=speed)
    normalize_audio("ffmpeg", Path("tts.wav"), Path("cn.wav"), speed=speed)
    extracted = calls[0]
    assert extracted[:2] == ["-ss", "10.000"]
    if end is not None:
        assert extracted.index("-t") < extracted.index("-i")
        assert extracted[extracted.index("-t") + 1] == "4.000"
    else:
        assert "-t" not in extracted
    for args in calls:
        if speed == 1.0:
            assert "-af" not in args
        else:
            assert args[args.index("-af") + 1] == f"atempo={speed}"


@pytest.mark.parametrize(
    "failure", [RuntimeError("decoder failed"), KeyboardInterrupt()]
)
def test_preparation_failure_preserves_source_and_existing_output(
    tmp_path, monkeypatch, failure
):
    source = tmp_path / "source.opus"
    output = tmp_path / "processing.flac"
    source.write_bytes(b"source")
    output.write_bytes(b"previous result")

    def fail(command, args):
        Path(args[-1]).write_bytes(b"partial encoding")
        raise failure

    monkeypatch.setattr("podtran.audio.run_ffmpeg", fail)
    with pytest.raises(type(failure)):
        prepare_audio(source, output)
    assert source.read_bytes() == b"source"
    assert output.read_bytes() == b"previous result"
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "processing.flac",
        "source.opus",
    ]


def test_source_change_during_preparation_does_not_publish_output(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.opus"
    output = tmp_path / "processing.flac"
    source.write_bytes(b"before")

    def encode(command, args):
        Path(args[-1]).write_bytes(b"encoded")
        source.write_bytes(b"changed during encoding")

    monkeypatch.setattr("podtran.audio.run_ffmpeg", encode)
    monkeypatch.setattr("podtran.audio.probe_duration", lambda *args: 1.0)
    with pytest.raises(RuntimeError, match="changed during preparation"):
        prepare_audio(source, output)
    assert not output.exists()
