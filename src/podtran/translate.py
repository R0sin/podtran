from __future__ import annotations

import json
from pathlib import Path

from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from podtran.artifacts import read_model_list, write_json
from podtran.config import AppConfig
from podtran.models import SegmentRecord, StageProgressCallback

TRANSLATION_RETRY_ATTEMPTS = 3


class Translator:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = OpenAI(
            api_key=_resolve_translation_key(config),
            base_url=config.translation.resolved_base_url(),
            timeout=config.translation.timeout_seconds,
        )

    def translate_segments(
        self,
        input_path: Path,
        output_path: Path,
        progress_callback: StageProgressCallback | None = None,
    ) -> list[SegmentRecord]:
        segments = _load_resume_segments(input_path, output_path)
        pending = [segment for segment in segments if not segment.text_zh.strip()]
        total_segments = len(pending)
        if progress_callback is not None:
            progress_callback(0, max(total_segments, 1), "Preparing translation batches")
        if not pending:
            if progress_callback is not None:
                progress_callback(1, 1, "Translation complete")
            return segments

        batch_size = max(1, self.config.translation.batch_size)
        processed = 0
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            try:
                translations = self._translate_batch(batch)
                mapping = {item["segment_id"]: item["text_zh"] for item in translations}
                for segment in segments:
                    if segment.segment_id in mapping:
                        segment.text_zh = mapping[segment.segment_id].strip()
                        segment.error = None
            except Exception as exc:
                error_message = _format_batch_error(exc, batch)
                for segment in batch:
                    segment.error = error_message
            processed += len(batch)
            write_json(output_path, segments)
            if progress_callback is not None:
                progress_callback(processed, total_segments, f"Translating segments {processed}/{total_segments}")

        if progress_callback is not None:
            progress_callback(total_segments, total_segments, "Translation complete")
        return segments

    @retry(
        reraise=True,
        stop=stop_after_attempt(TRANSLATION_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(
            (ValueError, RuntimeError, APIError, APITimeoutError, RateLimitError)
        ),
    )
    def _translate_batch(self, batch: list[SegmentRecord]) -> list[dict[str, str]]:
        payload = [{"segment_id": item.segment_id, "text": item.text} for item in batch]
        system_prompt = (
            "You are translating English podcast transcripts into natural spoken Chinese. "
            "Return valid JSON with this schema only: "
            '{"translations":[{"segment_id":"seg_x","text_zh":"..."}]}'
        )
        user_prompt = json.dumps(payload, ensure_ascii=False)
        response = self.client.chat.completions.create(
            model=self.config.translation.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        return _parse_translation_response(content, batch)



def _load_resume_segments(input_path: Path, output_path: Path) -> list[SegmentRecord]:
    source = output_path if output_path.exists() else input_path
    return read_model_list(source, SegmentRecord)



def _parse_translation_response(content: str, batch: list[SegmentRecord]) -> list[dict[str, str]]:
    cleaned = _strip_fences(content)
    if not cleaned:
        raise ValueError("Translation response was empty.")

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Translation response was not valid JSON: {exc.msg}. Response: {_excerpt(cleaned)}"
        ) from exc

    translations = parsed.get("translations")
    if not isinstance(translations, list):
        raise ValueError(
            "Translation response missing 'translations' list. "
            f"Response: {_excerpt(cleaned)}"
        )
    if len(translations) != len(batch):
        raise ValueError(
            f"Translation response shape mismatch: expected {len(batch)} items, got {len(translations)}. "
            f"Response: {_excerpt(cleaned)}"
        )

    expected_ids = {item.segment_id for item in batch}
    seen_ids: set[str] = set()
    normalized: list[dict[str, str]] = []
    for item in translations:
        if not isinstance(item, dict):
            raise ValueError(f"Translation item was not an object: {_excerpt(json.dumps(item, ensure_ascii=False))}")
        segment_id = str(item.get("segment_id", "")).strip()
        text_zh = str(item.get("text_zh", "")).strip()
        if not segment_id or segment_id not in expected_ids:
            raise ValueError(
                f"Translation response returned unexpected segment_id '{segment_id}'. "
                f"Expected one of {sorted(expected_ids)}."
            )
        if segment_id in seen_ids:
            raise ValueError(f"Translation response returned duplicate segment_id '{segment_id}'.")
        if not text_zh:
            raise ValueError(f"Translation response returned empty text_zh for '{segment_id}'.")
        seen_ids.add(segment_id)
        normalized.append({"segment_id": segment_id, "text_zh": text_zh})

    missing_ids = expected_ids - seen_ids
    if missing_ids:
        raise ValueError(f"Translation response omitted segment_ids: {sorted(missing_ids)}")

    return normalized



def _strip_fences(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("\n", 1)[0]
    return text.strip()



def _format_batch_error(exc: Exception, batch: list[SegmentRecord]) -> str:
    ids = [segment.segment_id for segment in batch]
    id_text = ", ".join(ids[:3])
    if len(ids) > 3:
        id_text += f", ... ({len(ids)} segments)"
    if not id_text:
        id_text = "no segment ids"
    message = str(exc).strip() or repr(exc)
    return f"{type(exc).__name__}: {message} | batch={id_text}"



def _excerpt(text: str, limit: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."



def _resolve_translation_key(config: AppConfig) -> str:
    resolved = config.resolve_provider_api_key(config.translation.provider)
    if resolved:
        return resolved
    return __import__("os").getenv("OPENAI_API_KEY", "")
