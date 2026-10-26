from __future__ import annotations

import gc
import inspect
import re
from collections.abc import Callable
from pathlib import Path

from podtran.config import ASRConfig
from podtran.models import TranscriptSegment, WordAlignment
from podtran.macos import apple_device


TranscriptionProgressCallback = Callable[[int, int, str], None]

_TRANSCRIPTION_STAGE_LABELS = (
    "Loading audio",
    "Loading ASR model",
    "Running ASR",
    "Loading alignment model",
    "Aligning words",
    "Running diarization",
    "Assigning speakers",
)


def transcribe_audio(
    audio_path: Path,
    config: ASRConfig,
    hf_token: str,
    min_speakers: int = 2,
    max_speakers: int = 5,
    progress_callback: TranscriptionProgressCallback | None = None,
) -> list[TranscriptSegment]:
    import whisperx

    device = _resolve_asr_device(config.device)
    alignment_device = "cpu" if device == "mlx" else device
    progress = _TranscriptionProgress(progress_callback)
    progress.start(_TRANSCRIPTION_STAGE_LABELS[0])
    audio = whisperx.load_audio(str(audio_path))
    progress.advance(f"{_TRANSCRIPTION_STAGE_LABELS[1]} (device: {device})")
    progress.advance(_TRANSCRIPTION_STAGE_LABELS[2])
    if device == "mlx":
        result = _transcribe_mlx(audio, config)
    else:
        device_type, _, device_index = device.partition(":")
        model = whisperx.load_model(
            config.model,
            device_type,
            device_index=int(device_index or 0),
            compute_type=config.compute_type,
            language=config.language or None,
            asr_options=_build_asr_options(),
        )
        result = model.transcribe(audio, batch_size=max(1, config.batch_size))
        del model
        gc.collect()

    language_code = result.get("language") or config.language or "en"
    progress.advance(_TRANSCRIPTION_STAGE_LABELS[3])
    if config.align_model.strip():
        model_a, metadata = whisperx.load_align_model(
            language_code=language_code,
            device=alignment_device,
            model_name=config.align_model,
        )
    else:
        model_a, metadata = whisperx.load_align_model(
            language_code=language_code,
            device=alignment_device,
        )
    progress.advance(_TRANSCRIPTION_STAGE_LABELS[4])
    aligned = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        alignment_device,
        return_char_alignments=False,
    )
    del model_a
    gc.collect()

    diarization_pipeline_cls = _get_diarization_pipeline_class(whisperx)
    progress.advance(_TRANSCRIPTION_STAGE_LABELS[5])
    diarize_model = diarization_pipeline_cls(token=hf_token, device=alignment_device)
    diarize_kwargs = {}
    if min_speakers > 0:
        diarize_kwargs["min_speakers"] = min_speakers
    if max_speakers > 0:
        diarize_kwargs["max_speakers"] = max_speakers
    diarize_segments = diarize_model(audio, **diarize_kwargs)
    progress.advance(_TRANSCRIPTION_STAGE_LABELS[6])
    enriched = whisperx.assign_word_speakers(diarize_segments, aligned)
    del diarize_model
    gc.collect()

    records: list[TranscriptSegment] = []
    for index, segment in enumerate(enriched.get("segments", [])):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        words = [
            WordAlignment(
                word=str(word.get("word", "")).strip(),
                start=_to_float(word.get("start")),
                end=_to_float(word.get("end")),
                score=_to_float(word.get("score")),
                speaker=(
                    str(word.get("speaker"))
                    if word.get("speaker") is not None
                    else None
                ),
            )
            for word in segment.get("words", [])
            if str(word.get("word", "")).strip()
        ]
        records.append(
            TranscriptSegment(
                segment_id=f"ts_{index:05d}",
                start=float(segment.get("start", 0.0)),
                end=float(segment.get("end", 0.0)),
                text=text,
                language=language_code,
                speaker=str(
                    segment.get("speaker") or infer_speaker(words) or "UNKNOWN"
                ),
                words=words,
            )
        )

    progress.finish()
    return records


def _resolve_asr_device(value: str) -> str:
    requested = apple_device(value)
    if requested == "mlx":
        return requested
    if requested == "auto":
        import ctranslate2
        import torch

        if torch.cuda.is_available() and ctranslate2.get_cuda_device_count() > 0:
            return "cuda:0"
        return "cpu"
    if requested == "cuda":
        return "cuda:0"
    if requested == "cpu" or re.fullmatch(r"cuda:[0-9]+", requested):
        return requested
    raise ValueError(
        f"Unsupported ASR device: {value!r}. Expected auto, cpu, cuda, or cuda:N"
    )


def infer_speaker(words: list[WordAlignment]) -> str | None:
    for word in words:
        if word.speaker:
            return word.speaker
    return None


def transcription_stage_count() -> int:
    return len(_TRANSCRIPTION_STAGE_LABELS)


def _build_asr_options(
    transcription_options_type: type | None = None,
) -> dict[str, object]:
    if transcription_options_type is None:
        try:
            from faster_whisper.transcribe import (
                TranscriptionOptions as transcription_options_type,
            )
        except Exception:
            return {"condition_on_previous_text": False}

    try:
        params = inspect.signature(transcription_options_type).parameters
    except (TypeError, ValueError):
        return {"condition_on_previous_text": False}

    if "condition_on_previous_text" in params:
        return {"condition_on_previous_text": False}
    if "condition_on_prev_text" in params:
        return {"condition_on_prev_text": False}
    return {}


def _get_diarization_pipeline_class(whisperx_module: object | None = None) -> type:
    if whisperx_module is None:
        import whisperx as whisperx_module

    pipeline_cls = getattr(whisperx_module, "DiarizationPipeline", None)
    if pipeline_cls is not None:
        return pipeline_cls

    from whisperx.diarize import DiarizationPipeline

    return DiarizationPipeline


def _to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class _TranscriptionProgress:
    def __init__(self, callback: TranscriptionProgressCallback | None) -> None:
        self._callback = callback
        self._completed = 0
        self._total = len(_TRANSCRIPTION_STAGE_LABELS)

    def start(self, message: str) -> None:
        self._emit(message)

    def advance(self, message: str) -> None:
        self._completed = min(self._completed + 1, self._total)
        self._emit(message)

    def finish(self) -> None:
        self._completed = self._total
        self._emit("Transcription complete")

    def _emit(self, message: str) -> None:
        if self._callback is None:
            return
        self._callback(self._completed, self._total, message)


def _transcribe_mlx(audio: object, config: ASRConfig) -> dict:
    import mlx.core as mx
    import mlx_whisper
    from mlx_whisper.transcribe import ModelHolder

    model = config.model.strip()
    repo = model if "/" in model else f"mlx-community/whisper-{model}-mlx"
    try:
        with mx.stream(mx.gpu):
            return mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=repo,
                language=config.language or None,
                condition_on_previous_text=False,
                verbose=False,
            )
    finally:
        # mlx-whisper holds the model globally, beyond this function's lifetime.
        ModelHolder.model = None
        ModelHolder.model_path = None
        gc.collect()
        mx.clear_cache()
