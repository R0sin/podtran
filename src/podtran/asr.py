from __future__ import annotations

import gc
import inspect
from collections.abc import Callable
from pathlib import Path

from podtran.config import ASRConfig
from podtran.models import TranscriptSegment, WordAlignment


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

    progress = _TranscriptionProgress(progress_callback)
    progress.start(_TRANSCRIPTION_STAGE_LABELS[0])
    audio = whisperx.load_audio(str(audio_path))
    batch_size = max(1, config.batch_size)

    progress.advance(_TRANSCRIPTION_STAGE_LABELS[1])
    model = whisperx.load_model(
        config.model,
        config.device,
        compute_type=config.compute_type,
        language=config.language or None,
        asr_options=_build_asr_options(),
    )
    progress.advance(_TRANSCRIPTION_STAGE_LABELS[2])
    result = model.transcribe(audio, batch_size=batch_size)
    del model
    gc.collect()

    language_code = result.get("language") or config.language or "en"
    progress.advance(_TRANSCRIPTION_STAGE_LABELS[3])
    if config.align_model.strip():
        model_a, metadata = whisperx.load_align_model(
            language_code=language_code,
            device=config.device,
            model_name=config.align_model,
        )
    else:
        model_a, metadata = whisperx.load_align_model(
            language_code=language_code,
            device=config.device,
        )
    progress.advance(_TRANSCRIPTION_STAGE_LABELS[4])
    aligned = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        config.device,
        return_char_alignments=False,
    )
    del model_a
    gc.collect()

    diarization_pipeline_cls = _get_diarization_pipeline_class(whisperx)
    progress.advance(_TRANSCRIPTION_STAGE_LABELS[5])
    diarize_model = diarization_pipeline_cls(token=hf_token, device=config.device)
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
                speaker=(str(word.get("speaker")) if word.get("speaker") is not None else None),
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
                speaker=str(segment.get("speaker") or infer_speaker(words) or "UNKNOWN"),
                words=words,
            )
        )

    progress.finish()
    return records


def infer_speaker(words: list[WordAlignment]) -> str | None:
    for word in words:
        if word.speaker:
            return word.speaker
    return None


def transcription_stage_count() -> int:
    return len(_TRANSCRIPTION_STAGE_LABELS)


def _build_asr_options(transcription_options_type: type | None = None) -> dict[str, object]:
    if transcription_options_type is None:
        try:
            from faster_whisper.transcribe import TranscriptionOptions as transcription_options_type
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
