from pathlib import Path

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
    events: list[tuple[int, int, str]] = []

    translated = translator.translate_segments(input_path, output_path, progress_callback=lambda completed, total, message: events.append((completed, total, message)))

    assert [event[0] for event in events] == [0, 2, 3, 3]
    assert all(event[1] == 3 for event in events[1:])
    assert events[0][2] == "Preparing translation batches"
    assert events[-1][2] == "Translation complete"
    assert all(item.text_zh for item in translated)
