from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from podtran.fingerprints import FingerprintService


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify audio fingerprints with real FFmpeg."
    )
    parser.add_argument("equivalent_files", type=Path, nargs="*")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="podtran-fingerprint-check-") as directory:
        root = Path(directory)
        fingerprints = FingerprintService(root / "indexes")

        def ffmpeg(*arguments: str) -> None:
            subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-y", *arguments], check=True
            )

        original = root / "source.wav"
        ffmpeg(
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=24000:duration=0.3",
            str(original),
        )
        lossless = root / "source.flac"
        metadata = root / "metadata.wav"
        ffmpeg("-i", str(original), str(lossless))
        ffmpeg(
            "-i",
            str(original),
            "-c:a",
            "copy",
            "-metadata",
            "title=changed",
            str(metadata),
        )
        expected = fingerprints.hash_audio(original)
        assert (
            expected
            == fingerprints.hash_audio(lossless)
            == fingerprints.hash_audio(metadata)
        )
        print("PASS: lossless containers and metadata changes share a fingerprint")

        for name, audio_filter in [
            ("rate", "asetrate=48000"),
            ("channels", "pan=stereo|c0=c0|c1=c0"),
            ("silence", "apad=pad_len=10"),
            ("samples", "volume=0.5"),
            ("start", "asetpts=PTS+2400"),
            ("gap", "asetpts=PTS+gte(N\\,1024)*2400"),
        ]:
            changed = root / f"{name}.nut"
            ffmpeg(
                "-i",
                str(original),
                "-af",
                audio_filter,
                "-c:a",
                "pcm_s16le",
                str(changed),
            )
            assert fingerprints.hash_audio(changed) != expected, name
        print(
            "PASS: samples, rate, channels, silence, start time and internal gaps remain distinct"
        )

        first = root / "first.opus"
        second = root / "second.opus"
        ffmpeg("-i", str(original), "-c:a", "libopus", str(first))
        ffmpeg("-i", str(first), "-c:a", "copy", str(second))
        assert first.read_bytes() != second.read_bytes()
        assert fingerprints.hash_audio(first) == fingerprints.hash_audio(second)
        print(
            "PASS: remuxed Opus files share a fingerprint despite different container bytes"
        )

        if args.equivalent_files:
            values = {fingerprints.hash_audio(path) for path in args.equivalent_files}
            assert len(values) == 1, (
                "Provided files have different decoded content or timing"
            )
            print(
                f"PASS: {len(args.equivalent_files)} provided files share a fingerprint"
            )


if __name__ == "__main__":
    main()
