from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest

from podtran.asr import (
    _build_asr_options,
    _get_diarization_pipeline_class,
    _resolve_asr_device,
    transcribe_audio,
    transcription_stage_count,
)
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
    def load_align_model(
        language_code: str, device: str, model_name: str | None = None
    ) -> tuple[object, dict[str, str]]:
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
    assert (
        _get_diarization_pipeline_class(_TopLevelWhisperX)
        is _TopLevelWhisperX.DiarizationPipeline
    )


def test_get_diarization_pipeline_class_falls_back_to_nested_module(
    monkeypatch,
) -> None:
    module = ModuleType("whisperx.diarize")
    module.DiarizationPipeline = _FakeDiarizationPipeline
    monkeypatch.setitem(sys.modules, "whisperx.diarize", module)
    pipeline_cls = _get_diarization_pipeline_class(type("_NoTopLevelWhisperX", (), {}))
    assert pipeline_cls is _FakeDiarizationPipeline


@pytest.fixture
def device_backends(monkeypatch):
    torch = Mock()
    torch.cuda.is_available.return_value = True
    ctranslate2 = Mock()
    ctranslate2.get_cuda_device_count.return_value = 2
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "ctranslate2", ctranslate2)
    return torch, ctranslate2


@pytest.mark.parametrize(
    "torch_available,ct2_count,expected",
    [(True, 2, "cuda:0"), (True, 0, "cpu"), (False, 2, "cpu"), (False, 0, "cpu")],
)
def test_auto_device_requires_both_cuda_backends(
    device_backends, torch_available, ct2_count, expected
) -> None:
    torch, ctranslate2 = device_backends
    torch.cuda.is_available.return_value = torch_available
    ctranslate2.get_cuda_device_count.return_value = ct2_count
    assert _resolve_asr_device("auto") == expected


@pytest.mark.parametrize(
    "requested,expected",
    [("cpu", "cpu"), ("cuda", "cuda:0"), (" CUDA:1 ", "cuda:1")],
)
def test_explicit_device_does_not_probe_or_fall_back(
    device_backends, requested, expected
) -> None:
    torch, ctranslate2 = device_backends
    assert _resolve_asr_device(requested) == expected
    torch.cuda.is_available.assert_not_called()
    ctranslate2.get_cuda_device_count.assert_not_called()


@pytest.mark.parametrize("requested", ["", "xpu", "mps", "cuda:-1", "cuda:abc"])
def test_invalid_asr_device_is_rejected(requested) -> None:
    with pytest.raises(ValueError, match="Unsupported ASR device"):
        _resolve_asr_device(requested)


@pytest.mark.parametrize(
    "requested,expected",
    [("auto", "cuda:0"), ("cpu", "cpu"), ("cuda", "cuda:0"), ("cuda:1", "cuda:1")],
)
@pytest.mark.parametrize("align_model", ["", "custom-align-model"])
def test_transcribe_audio_reports_stage_progress(
    monkeypatch, device_backends, requested, expected, align_model
) -> None:
    monkeypatch.setitem(sys.modules, "whisperx", _FakeWhisperXModule)
    options_module = ModuleType("faster_whisper.transcribe")
    options_module.TranscriptionOptions = _CurrentOptions
    monkeypatch.setitem(sys.modules, "faster_whisper.transcribe", options_module)
    spies = {}
    for name in ["load_model", "load_align_model", "align", "DiarizationPipeline"]:
        spies[name] = Mock(wraps=getattr(_FakeWhisperXModule, name))
        monkeypatch.setattr(_FakeWhisperXModule, name, spies[name])
    _FakeDiarizationPipeline.calls.clear()
    events: list[tuple[int, int, str]] = []

    result = transcribe_audio(
        Path("fake.wav"),
        ASRConfig(device=requested, align_model=align_model),
        "hf-token",
        min_speakers=2,
        max_speakers=5,
        progress_callback=lambda completed, total, message: events.append(
            (completed, total, message)
        ),
    )

    assert len(result) == 1
    assert result[0].speaker == "SPEAKER_00"
    assert _FakeDiarizationPipeline.calls[-1] == {"min_speakers": 2, "max_speakers": 5}
    assert [event[0] for event in events] == list(
        range(transcription_stage_count() + 1)
    )
    assert all(event[1] == transcription_stage_count() for event in events)
    assert events[0][2] == "Loading audio"
    assert events[1][2] == f"Loading ASR model (device: {expected})"
    assert events[-1][2] == "Transcription complete"
    device_type, _, device_index = expected.partition(":")
    assert spies["load_model"].call_args.args[1] == device_type
    assert spies["load_model"].call_args.kwargs["device_index"] == int(
        device_index or 0
    )
    assert spies["load_align_model"].call_args.kwargs["device"] == expected
    assert spies["align"].call_args.args[4] == expected
    assert spies["DiarizationPipeline"].call_args.kwargs["device"] == expected


def test_cuda_load_failure_is_not_retried_on_cpu(monkeypatch, device_backends) -> None:
    monkeypatch.setitem(sys.modules, "whisperx", _FakeWhisperXModule)
    monkeypatch.setattr("podtran.asr._build_asr_options", lambda: {})
    load_model = Mock(side_effect=RuntimeError("CUDA out of memory"))
    monkeypatch.setattr(_FakeWhisperXModule, "load_model", load_model)

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        transcribe_audio(Path("fake.wav"), ASRConfig(), "hf-token")

    load_model.assert_called_once()
    assert load_model.call_args.args[1] == "cuda"
