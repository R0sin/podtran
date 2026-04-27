from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

FFMPEG_COMMAND = "ffmpeg"
FFPROBE_COMMAND = "ffprobe"


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
) -> Path:
    args: list[str] = []
    if start is not None:
        args.extend(["-ss", seconds_arg(start)])
    args.extend(["-i", str(source)])
    if end is not None:
        clip_start = start or 0.0
        duration = max(end - clip_start, 0.0)
        args.extend(["-t", seconds_arg(duration)])
    args.extend(["-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(output)])
    run_ffmpeg(ffmpeg_path, args)
    return output


def normalize_audio(ffmpeg_path: str, source: Path, output: Path) -> Path:
    args = [
        "-i",
        str(source),
        "-ar",
        "24000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output),
    ]
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
