from __future__ import annotations

import argparse
import array
import hashlib
import json
import random
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

from podtran.audio import (
    hash_audio_content,
    prepare_audio,
    probe_audio_format,
    probe_duration,
)
from podtran.models import AudioCut


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *args], check=True)


def pcm(path: Path) -> bytes:
    return subprocess.check_output(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-c:a",
            "pcm_s16le",
            "-f",
            "s16le",
            "-",
        ]
    )


def streamed_pcm_digest(path: Path, cuts: list[AudioCut]) -> tuple[str, int]:
    audio_format = probe_audio_format(path)
    frame_bytes = audio_format.channels * 2
    ranges = [
        (
            round(c.start_time * audio_format.sample_rate) * frame_bytes,
            round(c.end_time * audio_format.sample_rate) * frame_bytes,
        )
        for c in sorted(cuts, key=lambda c: c.start_time)
    ]
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-c:a",
            "pcm_s16le",
            "-f",
            "s16le",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    assert process.stdout is not None
    digest = hashlib.sha256()
    offset = kept = 0
    while block := process.stdout.read(1024 * 1024):
        position = 0
        for start, end in ranges:
            if start >= offset + len(block) or end <= offset:
                continue
            before = max(position, start - offset)
            digest.update(block[position:before])
            kept += before - position
            position = min(len(block), end - offset)
        digest.update(block[position:])
        kept += len(block) - position
        offset += len(block)
    assert process.wait() == 0
    return digest.hexdigest(), kept


def assert_continuous(path: Path) -> None:
    process = subprocess.Popen(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_packets",
            "-show_entries",
            "packet=pts,duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    expected = 0
    assert process.stdout is not None
    for line in process.stdout:
        pts, duration = map(int, line.strip().split(",")[:2])
        assert pts == expected, (pts, expected)
        expected += duration
    assert process.wait() == 0
    assert expected > 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify prepared audio with real FFmpeg."
    )
    parser.add_argument("--source", type=Path)
    parser.add_argument("--cut", action="append", default=[], metavar="START:END")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="podtran-preparation-check-") as directory:
        root = Path(directory)
        source = root / "source.wav"
        rng = random.Random(7)
        samples = array.array(
            "h", (rng.randrange(-12000, 12000) for _ in range(48000 * 2 * 4))
        )
        if sys.byteorder != "little":
            samples.byteswap()
        original = samples.tobytes()
        with wave.open(str(source), "wb") as out:
            out.setnchannels(2)
            out.setsampwidth(2)
            out.setframerate(48000)
            out.writeframes(original)
        prepared = prepare_audio(source, root / "prepared.flac")
        assert pcm(prepared) == original
        assert probe_audio_format(prepared).sample_rate == 48000
        assert probe_audio_format(prepared).channels == 2
        assert hash_audio_content(prepared) == hash_audio_content(source)
        assert_continuous(prepared)
        print("PASS: native format, exact PCM16 samples and content identity")

        cuts = [
            AudioCut(start_time=1.013, end_time=1.9),
            AudioCut(start_time=1.8, end_time=2.037),
        ]
        cut = prepare_audio(source, root / "cut.flac", cuts=cuts)
        expected = original[: 48624 * 4] + original[97776 * 4 :]
        assert pcm(cut) == expected
        assert_continuous(cut)
        preview = prepare_audio(source, root / "preview.flac", cuts=cuts, duration=1.5)
        assert pcm(preview) == expected[: 72000 * 4]
        multiple = prepare_audio(
            source,
            root / "multiple.flac",
            cuts=[
                AudioCut(start_time=1.013, end_time=2.037),
                AudioCut(start_time=2.8, end_time=3.1),
            ],
        )
        assert (
            pcm(multiple)
            == original[: 48624 * 4]
            + original[97776 * 4 : 134400 * 4]
            + original[148800 * 4 :]
        )
        assert_continuous(multiple)
        edges = prepare_audio(
            source,
            root / "edges.flac",
            cuts=[
                AudioCut(start_time=0, end_time=0.013),
                AudioCut(start_time=3.7, end_time=10),
            ],
        )
        assert pcm(edges) == original[624 * 4 : 177600 * 4]
        assert_continuous(edges)
        print(
            "PASS: sample-accurate cuts, overlapping ranges, start/end cuts and preview after cutting"
        )

        try:
            prepare_audio(source, prepared, cuts=[AudioCut(start_time=0, end_time=10)])
        except (RuntimeError, ValueError):
            pass
        else:
            raise AssertionError("Removing all audio must fail")
        assert pcm(prepared) == original
        print("PASS: failed preparation preserves existing output")

        gap = root / "gap.nut"
        ffmpeg(
            "-i",
            str(source),
            "-af",
            "asetpts=PTS+gte(T\\,2)*0.5/TB",
            "-c:a",
            "pcm_s16le",
            str(gap),
        )
        filled = prepare_audio(gap, root / "filled.flac")
        assert abs(probe_duration("ffprobe", filled) - 4.5) < 1 / 48000
        assert_continuous(filled)
        assert hash_audio_content(filled) != hash_audio_content(prepared)
        print("PASS: timestamp gap becomes real silence and changes content identity")

        overlap = root / "overlap.nut"
        ffmpeg(
            "-i",
            str(source),
            "-af",
            "asetpts=PTS-gte(T\\,2)*0.5/TB",
            "-c:a",
            "pcm_s16le",
            str(overlap),
        )
        trimmed = prepare_audio(overlap, root / "trimmed.flac")
        assert abs(probe_duration("ffprobe", trimmed) - 3.5) < 1 / 48000
        assert_continuous(trimmed)
        shifted = root / "shifted.nut"
        ffmpeg(
            "-i",
            str(source),
            "-af",
            "asetpts=PTS+3/TB",
            "-c:a",
            "pcm_s16le",
            str(shifted),
        )
        rebased = prepare_audio(shifted, root / "rebased.flac")
        assert pcm(rebased) == original
        assert_continuous(rebased)
        print("PASS: overlaps are trimmed and absolute start offsets are rebased")

        opus = root / "source.opus"
        ffmpeg("-i", str(source), "-c:a", "libopus", str(opus))
        baseline = pcm(opus)
        exact = prepare_audio(opus, root / "opus-cut.flac", cuts=cuts)
        assert pcm(exact) == baseline[: 48624 * 4] + baseline[97776 * 4 :]
        assert_continuous(exact)
        print("PASS: Opus decoding and precise cuts introduce no extra sample changes")

        if args.source:
            intervals = [
                AudioCut(start_time=float(start), end_time=float(end))
                for start, end in (item.split(":") for item in args.cut)
            ]
            started = time.monotonic()
            output = prepare_audio(args.source, root / "episode.flac", cuts=intervals)
            fingerprint = hash_audio_content(output)
            assert_continuous(output)
            duration = probe_duration("ffprobe", output)
            assert streamed_pcm_digest(args.source, intervals) == streamed_pcm_digest(
                output, []
            )
            print(
                json.dumps(
                    {
                        "source": str(args.source),
                        "size_MB": round(output.stat().st_size / 1e6, 2),
                        "duration": duration,
                        "format": probe_audio_format(output).model_dump(),
                        "fingerprint": fingerprint,
                        "elapsed_seconds": round(time.monotonic() - started, 2),
                    },
                    ensure_ascii=False,
                )
            )
            print(
                "PASS: full source preparation, exact retained PCM samples, fingerprint and continuous timestamps"
            )


if __name__ == "__main__":
    main()
