from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from podtran.artifacts import atomic_output_path
from podtran.models import AudioCut, AudioFormat


FFMPEG_COMMAND = "ffmpeg"
FFPROBE_COMMAND = "ffprobe"

# Change these when preparation or PCM identity semantics change.
AUDIO_PREPARATION_VERSION = 1
AUDIO_FINGERPRINT_VERSION = 3


def probe_audio_format(path: Path) -> AudioFormat:
    result = subprocess.run(
        [
            FFPROBE_COMMAND,
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=sample_rate,channels",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"Cannot inspect audio {path}: {result.stderr.strip()}")
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise RuntimeError(f"Cannot inspect audio {path}: no audio stream")
    return AudioFormat.model_validate(streams[0])


def hash_audio_content(path: Path) -> str:
    """Hash native-rate/channel PCM16 samples, independent of container framing.

    Task inputs have already been prepared onto a continuous sample clock.
    Hashing does not repair media or interpret its container timestamps.
    """
    audio_format = probe_audio_format(path)
    value = hashlib.sha256(f"podtran-pcm16-v{AUDIO_FINGERPRINT_VERSION}\n".encode())
    value.update(json.dumps(audio_format.model_dump(), sort_keys=True).encode())
    value.update(b"\n")
    command = [
        FFMPEG_COMMAND,
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-af",
        "asetpts=N/SR/TB",
        "-c:a",
        "pcm_s16le",
        "-f",
        "s16le",
        "pipe:1",
    ]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        try:
            assert process.stdout is not None
            size = 0
            while chunk := process.stdout.read(1024 * 1024):
                value.update(chunk)
                size += len(chunk)
            returncode = process.wait()
            if returncode != 0:
                errors.seek(0)
                detail = errors.read(4096).decode("utf-8", errors="replace").strip()
                raise RuntimeError(
                    f"Cannot fingerprint audio {path}: {detail or 'ffmpeg failed'}"
                )
            if not size:
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
    return value.hexdigest()


def prepare_audio(
    source: Path,
    output: Path,
    *,
    cuts: Sequence[AudioCut] = (),
    duration: float | None = None,
) -> Path:
    """Publish native-rate/channel PCM16 FLAC on a continuous sample timeline.

    Rebase source timestamps, materialize gaps as silence and trim overlaps.
    Cut decoded samples before concatenation; preview limits apply afterwards.
    Neither decoding nor hashing retains the full episode in memory.
    """
    if source.resolve() == output.resolve():
        raise ValueError("Prepared audio must not overwrite its source")
    if duration is not None and duration <= 0:
        raise ValueError("Preview duration must be positive")
    source_stat = source.stat()
    merged: list[AudioCut] = []
    for cut in sorted(cuts, key=lambda item: item.start_time):
        if merged and cut.start_time <= merged[-1].end_time:
            merged[-1] = merged[-1].model_copy(
                update={"end_time": max(merged[-1].end_time, cut.end_time)}
            )
        else:
            merged.append(cut)
    intervals: list[tuple[float, float | None]] = []
    cursor = 0.0
    for cut in merged:
        if cut.start_time > cursor:
            intervals.append((cursor, cut.start_time))
        cursor = cut.end_time
    intervals.append((cursor, None))

    # Keep source rate and channel layout. No whole-file PCM intermediate.
    filters = [
        "[0:a:0]asetpts=PTS-STARTPTS,aresample=async=1:min_comp=0:min_hard_comp=0:first_pts=0[clock]"
    ]
    if len(intervals) == 1:
        inputs = ["[clock]"]
    else:
        inputs = [f"[part{i}]" for i in range(len(intervals))]
        filters.append(f"[clock]asplit={len(inputs)}" + "".join(inputs))
    for i, (start, end) in enumerate(intervals):
        trim = f"atrim=start={start:.9f}"
        if end is not None:
            trim += f":end={end:.9f}"
        filters.append(f"{inputs[i]}{trim},asetpts=PTS-STARTPTS[keep{i}]")
    kept = "".join(f"[keep{i}]" for i in range(len(intervals)))
    if len(intervals) > 1:
        filters.append(f"{kept}concat=n={len(intervals)}:v=0:a=1[joined]")
        kept = "[joined]"
    preview = f",atrim=duration={duration:.9f}" if duration is not None else ""
    filters.append(f"{kept}asetpts=N/SR/TB{preview}[audio]")
    with atomic_output_path(output) as temporary:
        run_ffmpeg(
            FFMPEG_COMMAND,
            [
                "-nostdin",
                "-v",
                "error",
                "-xerror",
                "-i",
                str(source),
                "-filter_complex",
                ";".join(filters),
                "-map",
                "[audio]",
                "-map_metadata",
                "-1",
                "-map_chapters",
                "-1",
                "-c:a",
                "flac",
                "-sample_fmt",
                "s16",
                "-f",
                "flac",
                str(temporary),
            ],
        )
        # Removing the entire input must not publish an empty working file.
        try:
            prepared_duration = probe_duration(FFPROBE_COMMAND, temporary)
        except ValueError as exc:
            raise RuntimeError("Audio preparation produced no audio samples") from exc
        if prepared_duration <= 0:
            raise RuntimeError("Audio preparation produced no audio samples")
        after = source.stat()
        if (source_stat.st_size, source_stat.st_mtime_ns, source_stat.st_ctime_ns) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise RuntimeError(f"Audio changed during preparation: {source}")
    return output


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
