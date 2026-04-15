from pathlib import Path
import threading
import time

from podtran.artifacts import read_model_list
from podtran.config import AppConfig
from podtran.models import SegmentRecord
from podtran.translate import Translator, _format_batch_error, _parse_translation_response, _resolve_translation_key


def _segment(segment_id: str) -> SegmentRecord:
    return SegmentRecord(
        segment_id=segment_id,
        block_id="b1",
        start=0.0,
        end=1.0,
        text="hello",
        speaker="SPEAKER_00",
        voice="Cherry",
    )


class _TranslatorHarness(Translator):
    def __init__(self) -> None:
        self.config = AppConfig()

    def _translate_batch(self, batch: list[SegmentRecord]) -> list[dict[str, str]]:
        return [{"segment_id": item.segment_id, "text_zh": f"zh-{item.segment_id}"} for item in batch]


def test_resolve_translation_key_prefers_provider_credentials(monkeypatch) -> None:
    config = AppConfig(providers={"dashscope": {"api_key": "dash-key"}})
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    assert _resolve_translation_key(config) == "dash-key"



def test_parse_translation_response_accepts_valid_payload() -> None:
    batch = [_segment("seg_1"), _segment("seg_2")]
    content = '{"translations":[{"segment_id":"seg_1","text_zh":"你好"},{"segment_id":"seg_2","text_zh":"世界"}]}'
    assert _parse_translation_response(content, batch) == [
        {"segment_id": "seg_1", "text_zh": "你好"},
        {"segment_id": "seg_2", "text_zh": "世界"},
    ]



def test_parse_translation_response_reports_json_error_with_response_excerpt() -> None:
    batch = [_segment("seg_1")]
    content = 'not json at all'
    try:
        _parse_translation_response(content, batch)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected ValueError")
    assert "not valid JSON" in message
    assert "not json at all" in message



def test_parse_translation_response_reports_shape_mismatch() -> None:
    batch = [_segment("seg_1"), _segment("seg_2")]
    content = '{"translations":[{"segment_id":"seg_1","text_zh":"你好"}]}'
    try:
        _parse_translation_response(content, batch)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected ValueError")
    assert "expected 2 items, got 1" in message



def test_format_batch_error_includes_exception_type_and_segment_ids() -> None:
    batch = [_segment("seg_1"), _segment("seg_2")]
    message = _format_batch_error(RuntimeError("boom"), batch)
    assert "RuntimeError: boom" in message
    assert "seg_1, seg_2" in message



def test_translate_segments_reports_segment_progress(tmp_path: Path) -> None:
    input_path = tmp_path / "segments.json"
    output_path = tmp_path / "translated.json"
    segments = [_segment("seg_1"), _segment("seg_2"), _segment("seg_3")]
    input_path.write_text(
        "[" + ",".join(segment.model_dump_json() for segment in segments) + "]",
        encoding="utf-8",
    )
    translator = _TranslatorHarness()
    translator.config.translation.batch_size = 2
    translator.config.translation.max_concurrency = 1
    events: list[tuple[int, int, str]] = []

    translated = translator.translate_segments(input_path, output_path, progress_callback=lambda completed, total, message: events.append((completed, total, message)))

    assert [event[0] for event in events] == [0, 2, 3, 3]
    assert all(event[1] == 3 for event in events)
    assert events[0][2] == "Preparing translation batches"
    assert events[-1][2] == "Translation complete"
    assert all(item.text_zh for item in translated)


def test_translate_segments_reports_resume_progress_as_completed_over_total(tmp_path: Path) -> None:
    input_path = tmp_path / "segments.json"
    output_path = tmp_path / "translated.json"
    segments = [_segment("seg_1"), _segment("seg_2"), _segment("seg_3"), _segment("seg_4")]
    input_path.write_text(
        "[" + ",".join(segment.model_dump_json() for segment in segments) + "]",
        encoding="utf-8",
    )
    resumed = [
        segments[0].model_copy(update={"text_zh": "zh-seg_1"}),
        segments[1].model_copy(update={"text_zh": "zh-seg_2"}),
        segments[2],
        segments[3],
    ]
    output_path.write_text(
        "[" + ",".join(segment.model_dump_json() for segment in resumed) + "]",
        encoding="utf-8",
    )
    translator = _TranslatorHarness()
    translator.config.translation.batch_size = 1
    translator.config.translation.max_concurrency = 1
    events: list[tuple[int, int, str]] = []

    translated = translator.translate_segments(input_path, output_path, progress_callback=lambda completed, total, message: events.append((completed, total, message)))

    assert [event[0] for event in events] == [2, 3, 4, 4]
    assert all(event[1] == 4 for event in events)
    assert events[1][2] == "Translating segments 3/4"
    assert events[2][2] == "Translating segments 4/4"
    assert all(item.text_zh for item in translated)


def test_translate_segments_runs_batches_concurrently_and_preserves_segment_order(tmp_path: Path) -> None:
    input_path = tmp_path / "segments.json"
    output_path = tmp_path / "translated.json"
    segments = [_segment(f"seg_{index}") for index in range(1, 5)]
    input_path.write_text(
        "[" + ",".join(segment.model_dump_json() for segment in segments) + "]",
        encoding="utf-8",
    )

    class _DelayedTranslator(_TranslatorHarness):
        def _translate_batch(self, batch: list[SegmentRecord]) -> list[dict[str, str]]:
            if batch[0].segment_id == "seg_1":
                time.sleep(0.05)
            return super()._translate_batch(batch)

    translator = _DelayedTranslator()
    translator.config.translation.batch_size = 2
    translator.config.translation.max_concurrency = 2

    translated = translator.translate_segments(input_path, output_path)
    persisted = read_model_list(output_path, SegmentRecord)

    assert [item.segment_id for item in translated] == ["seg_1", "seg_2", "seg_3", "seg_4"]
    assert [item.text_zh for item in translated] == ["zh-seg_1", "zh-seg_2", "zh-seg_3", "zh-seg_4"]
    assert [item.segment_id for item in persisted] == ["seg_1", "seg_2", "seg_3", "seg_4"]
    assert [item.text_zh for item in persisted] == ["zh-seg_1", "zh-seg_2", "zh-seg_3", "zh-seg_4"]


def test_translate_segments_marks_only_failed_batch_errors(tmp_path: Path) -> None:
    input_path = tmp_path / "segments.json"
    output_path = tmp_path / "translated.json"
    segments = [_segment(f"seg_{index}") for index in range(1, 5)]
    input_path.write_text(
        "[" + ",".join(segment.model_dump_json() for segment in segments) + "]",
        encoding="utf-8",
    )

    class _PartiallyFailingTranslator(_TranslatorHarness):
        def _translate_batch(self, batch: list[SegmentRecord]) -> list[dict[str, str]]:
            if batch[0].segment_id == "seg_1":
                raise RuntimeError("boom")
            return super()._translate_batch(batch)

    translator = _PartiallyFailingTranslator()
    translator.config.translation.batch_size = 2
    translator.config.translation.max_concurrency = 2

    translated = translator.translate_segments(input_path, output_path)

    assert [item.text_zh for item in translated[2:]] == ["zh-seg_3", "zh-seg_4"]
    assert translated[2].error is None
    assert translated[3].error is None
    assert translated[0].text_zh == ""
    assert translated[1].text_zh == ""
    assert translated[0].error is not None
    assert translated[1].error is not None
    assert "RuntimeError: boom" in translated[0].error
    assert translated[0].error == translated[1].error


def test_translate_segments_writes_merged_state_from_main_thread(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "segments.json"
    output_path = tmp_path / "translated.json"
    segments = [_segment(f"seg_{index}") for index in range(1, 5)]
    input_path.write_text(
        "[" + ",".join(segment.model_dump_json() for segment in segments) + "]",
        encoding="utf-8",
    )

    ready = threading.Barrier(3)
    release = threading.Event()
    snapshots: list[list[dict[str, str | None]]] = []
    errors: list[BaseException] = []

    class _BarrierTranslator(_TranslatorHarness):
        def _translate_batch(self, batch: list[SegmentRecord]) -> list[dict[str, str]]:
            ready.wait(timeout=1)
            release.wait(timeout=1)
            if batch[0].segment_id == "seg_1":
                time.sleep(0.02)
            return super()._translate_batch(batch)

    def capture_write_json(path: Path, data) -> None:
        snapshots.append(
            [
                {
                    "segment_id": item.segment_id,
                    "text_zh": item.text_zh,
                    "error": item.error,
                }
                for item in data
            ]
        )
        path.write_text("[]", encoding="utf-8")

    monkeypatch.setattr("podtran.translate.write_json", capture_write_json)

    translator = _BarrierTranslator()
    translator.config.translation.batch_size = 2
    translator.config.translation.max_concurrency = 2

    def run_translation() -> None:
        try:
            translator.translate_segments(input_path, output_path)
        except BaseException as exc:  # pragma: no cover - test should fail below if hit
            errors.append(exc)

    worker = threading.Thread(target=run_translation)
    worker.start()
    ready.wait(timeout=1)
    release.set()
    worker.join(timeout=2)

    assert not worker.is_alive()
    assert not errors
    assert len(snapshots) == 2
    assert snapshots[-1] == [
        {"segment_id": "seg_1", "text_zh": "zh-seg_1", "error": None},
        {"segment_id": "seg_2", "text_zh": "zh-seg_2", "error": None},
        {"segment_id": "seg_3", "text_zh": "zh-seg_3", "error": None},
        {"segment_id": "seg_4", "text_zh": "zh-seg_4", "error": None},
    ]
