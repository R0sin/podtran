from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path

FFMPEG_COMMAND = "ffmpeg"
FFPROBE_COMMAND = "ffprobe"

# Changing the decoding/framing contract requires a new version, including the
# on-disk fingerprint index namespace.
AUDIO_FINGERPRINT_VERSION = 2


def hash_audio_content(path: Path) -> str:
    """Hash decoded samples, native rate/channels and presentation timestamps.

    Fixed sample blocks remove container packet boundaries from the identity.
    Float64 PCM preserves decoder precision without resampling, downmixing,
    trimming silence or padding the last block. Only the first audio stream is
    used, matching the processing pipeline.
    """
    command = [
        FFMPEG_COMMAND,
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-copyts",
        "-i",
        str(path),
        "-filter_complex",
        "[0:a:0]asplit=2[timing][samples];[samples]asetnsamples=n=4096:p=0[blocks]",
        "-map",
        "[timing]",
        "-map",
        "[blocks]",
        "-c:a",
        "pcm_f64le",
        "-f",
        "framehash",
        "-hash",
        "sha256",
        "pipe:1",
    ]
    # A file for stderr avoids a pipe deadlock without retaining decoder output
    # in memory. stdout contains only one small checksum record per PCM block.
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        try:
            assert process.stdout is not None
            value = _hash_audio_frames(process.stdout)
            returncode = process.wait()
            if returncode != 0:
                errors.seek(0)
                detail = errors.read(4096).decode("utf-8", errors="replace").strip()
                raise RuntimeError(
                    f"Cannot fingerprint audio {path}: {detail or 'ffmpeg failed'}"
                )
            if value is None:
                raise RuntimeError(
                    f"Cannot fingerprint audio {path}: no audio samples decoded"
                )
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            if process.stdout is not None:
                process.stdout.close()
    return value


def _hash_audio_frames(lines: Iterable[bytes]) -> str | None:
    """Canonicalize framehash records without retaining decoded audio.

    Stream 0 retains decoder timestamps. Merge adjacent frames into intervals
    so packet boundaries do not affect timing identity. Stream 1 contains fixed
    sample blocks; its timestamps are ignored because buffering can hide gaps.
    """
    samples = hashlib.sha256(f"podtran-audio-v{AUDIO_FINGERPRINT_VERSION}\n".encode())
    timing = hashlib.sha256()
    interval: tuple[int, int] | None = None
    has_samples = False
    for line in lines:
        if line.startswith(b"#software:"):
            continue
        normalized = b"".join(line.split())
        if not normalized:
            continue
        if normalized.startswith(b"#"):
            samples.update(normalized + b"\n")
            continue
        stream, _dts, pts, duration, size, checksum = normalized.split(b",")
        if stream == b"0":
            start = int(pts)
            end = start + int(duration)
            if interval is not None and start == interval[1]:
                interval = interval[0], end
            else:
                if interval is not None:
                    timing.update(f"{interval[0]}:{interval[1]}\n".encode())
                interval = start, end
        elif stream == b"1":
            samples.update(b",".join([duration, size, checksum]) + b"\n")
            has_samples = True
    if interval is None or not has_samples:
        return None
    timing.update(f"{interval[0]}:{interval[1]}\n".encode())
    samples.update(timing.digest())
    return samples.hexdigest()


def seconds_arg(value: float | None) -> str | None:
    if value is None:
        return None
    return f"{value:.3f}"


def run_ffmpeg(ffmpeg_path: str, args: list[str]) -> None:
    command = [ffmpeg_path, "-y", *args]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffmpeg command failed")


def probe_duration(ffprobe_path: str, path: Path) -> float:
    result = subprocess.run(
        [
            ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffprobe command failed")
    return float(result.stdout.strip())


def extract_audio_chunk(
    ffmpeg_path: str,
    source: Path,
    output: Path,
    start: float | None,
    end: float | None,
    *,
    speed: float = 1.0,
) -> Path:
    args: list[str] = []
    if start is not None:
        args.extend(["-ss", seconds_arg(start)])
    if end is not None:
        clip_start = start or 0.0
        duration = max(end - clip_start, 0.0)
        args.extend(["-t", seconds_arg(duration)])
    # Limit source duration before filtering so slow playback is not truncated.
    args.extend(["-i", str(source)])
    if speed != 1.0:
        args.extend(["-af", f"atempo={speed}"])
    args.extend(["-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(output)])
    run_ffmpeg(ffmpeg_path, args)
    return output


def normalize_audio(
    ffmpeg_path: str, source: Path, output: Path, *, speed: float = 1.0
) -> Path:
    args = ["-i", str(source)]
    if speed != 1.0:
        args.extend(["-af", f"atempo={speed}"])
    args.extend(
        [
            "-ar",
            "24000",
            "-ac",
            "1",
            "-c:a",
            "pcm_s16le",
            str(output),
        ]
    )
    run_ffmpeg(ffmpeg_path, args)
    return output


def create_silence(ffmpeg_path: str, output: Path, duration_ms: int) -> Path:
    duration = max(duration_ms / 1000.0, 0.01)
    args = [
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=24000:cl=mono",
        "-t",
        f"{duration:.3f}",
        "-c:a",
        "pcm_s16le",
        str(output),
    ]
    run_ffmpeg(ffmpeg_path, args)
    return output


def concat_wav_chunks(ffmpeg_path: str, chunks: list[Path], output: Path) -> Path:
    list_file = output.parent / f"{output.stem}.concat.txt"
    list_file.write_text(
        "".join(f"file '{chunk.as_posix()}'\n" for chunk in chunks),
        encoding="utf-8",
    )
    args = [
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-ar",
        "24000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output),
    ]
    try:
        run_ffmpeg(ffmpeg_path, args)
    finally:
        list_file.unlink(missing_ok=True)
    return output


def concat_audio(
    ffmpeg_path: str, chunks: list[Path], output: Path, bitrate: str
) -> Path:
    list_file = output.parent / f"{output.stem}.concat.txt"
    list_file.write_text(
        "".join(f"file '{chunk.as_posix()}'\n" for chunk in chunks),
        encoding="utf-8",
    )
    args = [
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c:a",
        "libmp3lame",
        "-b:a",
        bitrate,
        str(output),
    ]
    try:
        run_ffmpeg(ffmpeg_path, args)
    finally:
        list_file.unlink(missing_ok=True)
    return output


def reset_temp_dir(path: Path, workspace_root: Path) -> None:
    resolved = path.resolve()
    root = workspace_root.resolve()
    if root not in resolved.parents and resolved != root:
        raise RuntimeError(f"Refusing to clean temp dir outside workspace: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
