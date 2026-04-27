from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from podtran.audio import (
    FFMPEG_COMMAND,
    FFPROBE_COMMAND,
    concat_audio,
    create_silence,
    extract_audio_chunk,
    normalize_audio,
    probe_duration,
    reset_temp_dir,
)
from podtran.config import AppConfig
from podtran.models import SegmentRecord, StageProgressCallback

ChunkStepCallback = Callable[[str], None]


def compose_output(
    source_audio: Path,
    segments: list[SegmentRecord],
    config: AppConfig,
    temp_dir: Path,
    output_path: Path,
    mode: str | None = None,
    progress_callback: StageProgressCallback | None = None,
) -> Path:
    selected_mode = (mode or config.compose.mode).lower()
    reset_temp_dir(temp_dir, temp_dir.parent.parent)

    audio_duration = (
        probe_duration(FFPROBE_COMMAND, source_audio)
        if selected_mode != "replace"
        else 0.0
    )
    chunk_steps = _count_chunk_steps(selected_mode, segments, audio_duration)
    total_steps = max(chunk_steps + 1, 1)
    if progress_callback is not None:
        progress_callback(0, total_steps, "Scanning segments")

    completed_steps = 0

    def step(message: str) -> None:
        nonlocal completed_steps
        completed_steps += 1
        if progress_callback is not None:
            progress_callback(completed_steps, total_steps, message)

    if selected_mode == "replace":
        chunks = build_replace_chunks(
            source_audio, segments, config, temp_dir, step_callback=step
        )
    else:
        chunks = build_interleave_chunks(
            source_audio,
            segments,
            config,
            temp_dir,
            audio_duration=audio_duration,
            step_callback=step,
        )
    if not chunks:
        raise RuntimeError("No audio chunks were produced for compose.")
    if progress_callback is not None:
        progress_callback(total_steps - 1, total_steps, "Concatenating audio")
    output = concat_audio(
        FFMPEG_COMMAND, chunks, output_path, config.compose.output_bitrate
    )
    if progress_callback is not None:
        progress_callback(total_steps, total_steps, "Compose complete")
    return output


def build_interleave_chunks(
    source_audio: Path,
    segments: list[SegmentRecord],
    config: AppConfig,
    temp_dir: Path,
    audio_duration: float | None = None,
    step_callback: ChunkStepCallback | None = None,
) -> list[Path]:
    chunks: list[Path] = []
    cursor = 0.0
    resolved_audio_duration = (
        audio_duration
        if audio_duration is not None
        else probe_duration(FFPROBE_COMMAND, source_audio)
    )
    for index, segment in enumerate(sorted(segments, key=lambda item: item.start)):
        if segment.end > cursor:
            english_chunk = temp_dir / f"{index:05d}_en.wav"
            extract_audio_chunk(
                FFMPEG_COMMAND,
                source_audio,
                english_chunk,
                cursor,
                segment.end,
            )
            chunks.append(english_chunk)
            cursor = segment.end
            _emit_step(step_callback, "Building chunks")

        if (
            segment.status == "completed"
            and segment.tts_audio_path
            and Path(segment.tts_audio_path).exists()
        ):
            pre_gap = temp_dir / f"{index:05d}_gap_before.wav"
            create_silence(FFMPEG_COMMAND, pre_gap, config.compose.gap_en_to_cn_ms)
            chunks.append(pre_gap)
            _emit_step(step_callback, "Building chunks")
            cn_chunk = temp_dir / f"{index:05d}_cn.wav"
            normalize_audio(FFMPEG_COMMAND, Path(segment.tts_audio_path), cn_chunk)
            chunks.append(cn_chunk)
            _emit_step(step_callback, "Building chunks")
            post_gap = temp_dir / f"{index:05d}_gap_after.wav"
            create_silence(FFMPEG_COMMAND, post_gap, config.compose.gap_cn_to_en_ms)
            chunks.append(post_gap)
            _emit_step(step_callback, "Building chunks")

    if cursor < resolved_audio_duration:
        tail = temp_dir / "tail_en.wav"
        extract_audio_chunk(FFMPEG_COMMAND, source_audio, tail, cursor, None)
        chunks.append(tail)
        _emit_step(step_callback, "Building chunks")

    return chunks


def build_replace_chunks(
    source_audio: Path,
    segments: list[SegmentRecord],
    config: AppConfig,
    temp_dir: Path,
    step_callback: ChunkStepCallback | None = None,
) -> list[Path]:
    _ = source_audio
    chunks: list[Path] = []
    cursor = 0.0
    for index, segment in enumerate(sorted(segments, key=lambda item: item.start)):
        gap = segment.start - cursor
        if gap > 0:
            gap_chunk = temp_dir / f"{index:05d}_gap.wav"
            create_silence(FFMPEG_COMMAND, gap_chunk, int(gap * 1000))
            chunks.append(gap_chunk)
            _emit_step(step_callback, "Building chunks")
        if (
            segment.status == "completed"
            and segment.tts_audio_path
            and Path(segment.tts_audio_path).exists()
        ):
            cn_chunk = temp_dir / f"{index:05d}_replace_cn.wav"
            normalize_audio(FFMPEG_COMMAND, Path(segment.tts_audio_path), cn_chunk)
            chunks.append(cn_chunk)
            _emit_step(step_callback, "Building chunks")
        else:
            fallback = temp_dir / f"{index:05d}_missing.wav"
            create_silence(
                FFMPEG_COMMAND, fallback, int((segment.end - segment.start) * 1000)
            )
            chunks.append(fallback)
            _emit_step(step_callback, "Building chunks")
        cursor = segment.end
    return chunks


def _count_chunk_steps(
    selected_mode: str, segments: list[SegmentRecord], audio_duration: float
) -> int:
    if selected_mode == "replace":
        return _count_replace_steps(segments)
    return _count_interleave_steps(segments, audio_duration)


def _count_interleave_steps(
    segments: list[SegmentRecord], audio_duration: float
) -> int:
    steps = 0
    cursor = 0.0
    for segment in sorted(segments, key=lambda item: item.start):
        if segment.end > cursor:
            steps += 1
            cursor = segment.end
        if (
            segment.status == "completed"
            and segment.tts_audio_path
            and Path(segment.tts_audio_path).exists()
        ):
            steps += 3
    if cursor < audio_duration:
        steps += 1
    return steps


def _count_replace_steps(segments: list[SegmentRecord]) -> int:
    steps = 0
    cursor = 0.0
    for segment in sorted(segments, key=lambda item: item.start):
        if segment.start > cursor:
            steps += 1
        steps += 1
        cursor = segment.end
    return steps


def _emit_step(step_callback: ChunkStepCallback | None, message: str) -> None:
    if step_callback is None:
        return
    step_callback(message)
