import sys
from pathlib import Path

from podtran.asr import _build_asr_options, _get_diarization_pipeline_class, transcribe_audio, transcription_stage_count
from podtran.config import ASRConfig


class _CurrentOptions:
    def __init__(self, condition_on_previous_text: bool) -> None:
        self.condition_on_previous_text = condition_on_previous_text


class _LegacyOptions:
    def __init__(self, condition_on_prev_text: bool) -> None:
        self.condition_on_prev_text = condition_on_prev_text


class _UnknownOptions:
    def __init__(self, beam_size: int) -> None:
        self.beam_size = beam_size


class _TopLevelWhisperX:
    class DiarizationPipeline:
        pass


class _FakeModel:
    def transcribe(self, audio: object, batch_size: int) -> dict[str, object]:
        return {
            "language": "en",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": " hello ",
                    "words": [
                        {
                            "word": "hello",
                            "start": 0.0,
                            "end": 1.0,
                            "score": 0.99,
                        }
                    ],
                }
            ],
        }


class _FakeDiarizationPipeline:
    calls: list[dict[str, object]] = []

    def __init__(self, token: str, device: str) -> None:
        self.token = token
        self.device = device

    def __call__(self, audio: object, **kwargs: object) -> list[dict[str, object]]:
        self.__class__.calls.append(dict(kwargs))
        return [{"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0}]


class _FakeWhisperXModule:
    DiarizationPipeline = _FakeDiarizationPipeline

    @staticmethod
    def load_audio(path: str) -> str:
        return path

    @staticmethod
    def load_model(*args: object, **kwargs: object) -> _FakeModel:
        return _FakeModel()

    @staticmethod
    def load_align_model(language_code: str, device: str, model_name: str | None = None) -> tuple[object, dict[str, str]]:
        return object(), {"language": language_code}

    @staticmethod
    def align(
        segments: list[dict[str, object]],
        model_a: object,
        metadata: dict[str, str],
        audio: object,
        device: str,
        return_char_alignments: bool = False,
    ) -> dict[str, object]:
        return {"segments": segments}

    @staticmethod
    def assign_word_speakers(
        diarize_segments: list[dict[str, object]],
        aligned: dict[str, object],
    ) -> dict[str, object]:
        segment = dict(aligned["segments"][0])
        segment["speaker"] = "SPEAKER_00"
        segment["words"] = [
            {
                "word": "hello",
                "start": 0.0,
                "end": 1.0,
                "score": 0.99,
                "speaker": "SPEAKER_00",
            }
        ]
        return {"segments": [segment]}


def test_build_asr_options_prefers_current_parameter_name() -> None:
    assert _build_asr_options(_CurrentOptions) == {"condition_on_previous_text": False}


def test_build_asr_options_supports_legacy_parameter_name() -> None:
    assert _build_asr_options(_LegacyOptions) == {"condition_on_prev_text": False}


def test_build_asr_options_skips_unknown_parameter_sets() -> None:
    assert _build_asr_options(_UnknownOptions) == {}


def test_get_diarization_pipeline_class_prefers_top_level_export() -> None:
    assert _get_diarization_pipeline_class(_TopLevelWhisperX) is _TopLevelWhisperX.DiarizationPipeline


def test_get_diarization_pipeline_class_falls_back_to_nested_module() -> None:
    pipeline_cls = _get_diarization_pipeline_class(type("_NoTopLevelWhisperX", (), {}))
    assert pipeline_cls.__name__ == "DiarizationPipeline"
    assert pipeline_cls.__module__ == "whisperx.diarize"


def test_transcribe_audio_reports_stage_progress(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "whisperx", _FakeWhisperXModule)
    _FakeDiarizationPipeline.calls.clear()
    events: list[tuple[int, int, str]] = []

    result = transcribe_audio(
        Path("fake.wav"),
        ASRConfig(),
        "hf-token",
        min_speakers=2,
        max_speakers=5,
        progress_callback=lambda completed, total, message: events.append((completed, total, message)),
    )

    assert len(result) == 1
    assert result[0].speaker == "SPEAKER_00"
    assert _FakeDiarizationPipeline.calls[-1] == {"min_speakers": 2, "max_speakers": 5}
    assert [event[0] for event in events] == list(range(transcription_stage_count() + 1))
    assert all(event[1] == transcription_stage_count() for event in events)
    assert events[0][2] == "Loading audio"
    assert events[-1][2] == "Transcription complete"
