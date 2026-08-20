from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Callable, Protocol
from urllib.parse import urlencode

import httpx
from openai import APIError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from podtran.artifacts import read_model_list, write_json
from podtran.config import AppConfig
from podtran.models import SegmentRecord, StageProgressCallback

TRANSLATION_RETRY_ATTEMPTS = 3
GOOGLE_FREE_TRANSLATE_URL = "https://translate.google.com/translate_a/t"
# Protocol reference (MIT): https://github.com/plainheart/bing-translate-api
BING_FREE_TRANSLATOR_URL = "https://cn.bing.com/Translator"
BING_FREE_TRANSLATE_URL = "https://cn.bing.com/ttranslatev3"
BING_FREE_TEXT_LIMIT = 1000
BING_FREE_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/150.0.0.0 Safari/537.36 Edg/151.0.4129.59"
)


class TranslationRuntime(BaseModel):
    provider: str
    base_url: str = ""
    model: str = ""


class _BingSessionState(BaseModel):
    ig: str
    iid: str
    key: str
    token: str
    expires_at: float


class _BingTranslatorPageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.iid = ""
        self._in_script = False
        self.script_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "script":
            self._in_script = True
        for name, value in attrs:
            if name == "data-iid" and value:
                self.iid = value

    def handle_endtag(self, tag: str) -> None:
        if tag == "script":
            self._in_script = False

    def handle_data(self, data: str) -> None:
        if self._in_script:
            self.script_parts.append(data)


def _is_retryable_bing_error(exc: BaseException) -> bool:
    if isinstance(exc, httpx.RequestError):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500


def _retry_bing_free_request():
    return retry(
        reraise=True,
        stop=stop_after_attempt(TRANSLATION_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception(_is_retryable_bing_error),
    )


class TranslationBackend(Protocol):
    batch_size_limit: int | None

    def translate_batch(self, batch: list[SegmentRecord]) -> list[dict[str, str]]: ...


class OpenAICompatibleTranslationBackend:
    batch_size_limit = None

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.runtime = resolve_translation_runtime(config)
        self.client = OpenAI(
            api_key=_resolve_translation_key(config),
            base_url=self.runtime.base_url,
            timeout=config.translation.timeout_seconds,
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(TRANSLATION_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(
            (ValueError, RuntimeError, APIError, APITimeoutError, RateLimitError)
        ),
    )
    def translate_batch(self, batch: list[SegmentRecord]) -> list[dict[str, str]]:
        payload = [{"segment_id": item.segment_id, "text": item.text} for item in batch]
        system_prompt = (
            "You are translating English podcast transcripts into natural spoken Chinese. "
            "Return valid JSON with this schema only: "
            '{"translations":[{"segment_id":"seg_x","text_zh":"..."}]}'
        )
        user_prompt = json.dumps(payload, ensure_ascii=False)
        response = self.client.chat.completions.create(
            model=self.runtime.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        return _parse_translation_response(content, batch)


class GoogleFreeTranslationBackend:
    batch_size_limit = None

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = httpx.Client(
            timeout=config.translation.timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0"},
        )

    def translate_batch(self, batch: list[SegmentRecord]) -> list[dict[str, str]]:
        return self._translate_segments(batch)

    @retry(
        reraise=True,
        stop=stop_after_attempt(TRANSLATION_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((RuntimeError, httpx.HTTPError)),
    )
    def _translate_segments(self, batch: list[SegmentRecord]) -> list[dict[str, str]]:
        form_body = urlencode([("q", item.text) for item in batch], doseq=True)
        response = self.client.post(
            GOOGLE_FREE_TRANSLATE_URL,
            params={
                "client": "at",
                "sl": "en",
                "tl": "zh-CN",
                "ie": "UTF-8",
                "oe": "UTF-8",
                "dj": "1",
                "format": "text",
                "v": "1.0",
            },
            content=form_body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response.raise_for_status()
        return _parse_google_free_translation_response(response.text, batch)


class BingFreeTranslationBackend:
    batch_size_limit = 1

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = httpx.Client(
            timeout=config.translation.timeout_seconds,
            follow_redirects=False,
            headers={"User-Agent": BING_FREE_USER_AGENT},
        )
        self._session: _BingSessionState | None = None
        self._session_lock = threading.Lock()

    def translate_batch(self, batch: list[SegmentRecord]) -> list[dict[str, str]]:
        if len(batch) != 1:
            raise RuntimeError(
                f"Bing-free expects one segment per batch, got {len(batch)}."
            )
        segment = batch[0]
        translated = "".join(
            self._translate_text(chunk) for chunk in _split_bing_free_text(segment.text)
        )
        return [{"segment_id": segment.segment_id, "text_zh": translated}]

    def _translate_text(self, text: str) -> str:
        for refresh_attempt in range(2):
            session = self._get_session(force=refresh_attempt == 1)
            response = self._request_translation(text, session)
            parsed = _decode_bing_free_response(response.text)
            blocking_error = _bing_free_blocking_error(parsed)
            if blocking_error is not None:
                raise RuntimeError(blocking_error)
            if response.status_code == 400 and refresh_attempt == 0:
                continue
            if response.status_code == 401:
                raise RuntimeError(
                    "Bing-free translation limit was exceeded (HTTP 401)."
                )
            if response.status_code == 429:
                raise RuntimeError("Bing-free translation was rate limited (HTTP 429).")
            response.raise_for_status()
            return _parse_bing_free_translation_response(response.text, parsed)
        raise RuntimeError("Bing-free translation failed after refreshing its session.")

    @_retry_bing_free_request()
    def _request_translation(
        self, text: str, session: _BingSessionState
    ) -> httpx.Response:
        response = self.client.post(
            BING_FREE_TRANSLATE_URL,
            params={"isVertical": "1", "IG": session.ig, "IID": session.iid},
            data={
                "fromLang": "en",
                "to": "zh-Hans",
                "text": text,
                "key": session.key,
                "token": session.token,
            },
            headers={"Referer": BING_FREE_TRANSLATOR_URL},
        )
        if response.status_code >= 500:
            response.raise_for_status()
        return response

    def _get_session(self, *, force: bool = False) -> _BingSessionState:
        with self._session_lock:
            if (
                not force
                and self._session is not None
                and time.monotonic() < self._session.expires_at
            ):
                return self._session
            response = self._request_session_page()
            self._session = _parse_bing_free_session(response.text)
            return self._session

    @_retry_bing_free_request()
    def _request_session_page(self) -> httpx.Response:
        response = self.client.get(BING_FREE_TRANSLATOR_URL)
        response.raise_for_status()
        return response


def _parse_bing_free_session(content: str) -> _BingSessionState:
    page = _BingTranslatorPageParser()
    page.feed(content)
    scripts = "\n".join(page.script_parts)
    ig_match = re.search(r'IG:"([^"]+)"', scripts)
    abuse_match = re.search(r"params_AbusePreventionHelper\s*=\s*(\[[^\]]+\])", scripts)
    if not ig_match or not page.iid or not abuse_match:
        raise RuntimeError("Bing-free translator page was missing session data.")
    try:
        key, token, expires_ms = json.loads(abuse_match.group(1))
        issued_at_ms = float(key)
        expires_ms = float(expires_ms)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Bing-free translator page had invalid session data."
        ) from exc
    if not key or not token or expires_ms <= 0:
        raise RuntimeError("Bing-free translator page had incomplete session data.")
    remaining_ms = min(
        expires_ms,
        max(issued_at_ms + expires_ms - time.time() * 1000, 0),
    )
    return _BingSessionState(
        ig=ig_match.group(1),
        iid=page.iid,
        key=str(key),
        token=str(token),
        expires_at=time.monotonic() + remaining_ms / 1000,
    )


def _split_bing_free_text(text: str) -> list[str]:
    remaining = text.strip()
    if not remaining:
        raise RuntimeError("Bing-free translation input was empty.")
    chunks: list[str] = []
    while len(remaining) > BING_FREE_TEXT_LIMIT:
        window = remaining[:BING_FREE_TEXT_LIMIT]
        sentence_matches = list(re.finditer(r'[.!?](?:["\')\]]*)\s+', window))
        if sentence_matches:
            split_at = sentence_matches[-1].end()
        else:
            split_at = window.rfind(" ")
            if split_at <= 0:
                split_at = BING_FREE_TEXT_LIMIT
        chunk = remaining[:split_at].strip()
        if not chunk:
            split_at = BING_FREE_TEXT_LIMIT
            chunk = remaining[:split_at]
        chunks.append(chunk)
        remaining = remaining[split_at:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def _decode_bing_free_response(content: str) -> object:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def _parse_bing_free_translation_response(content: str, parsed: object) -> str:
    cleaned = content.strip()
    if not cleaned:
        raise RuntimeError("Bing-free translation response was empty.")
    if parsed is None:
        raise RuntimeError(
            "Bing-free translation response was not valid JSON. "
            f"Response: {_excerpt(cleaned)}"
        )
    blocking_error = _bing_free_blocking_error(parsed)
    if blocking_error is not None:
        raise RuntimeError(blocking_error)
    if isinstance(parsed, dict):
        raise RuntimeError(
            "Bing-free translation returned an error response. "
            f"Response: {_excerpt(cleaned)}"
        )
    try:
        translated = parsed[0]["translations"][0]["text"]
    except (IndexError, KeyError, TypeError) as exc:
        raise RuntimeError(
            "Bing-free translation response structure was unexpected. "
            f"Response: {_excerpt(cleaned)}"
        ) from exc
    translated_text = str(translated).strip()
    if not translated_text:
        raise RuntimeError("Bing-free translation response returned empty text.")
    return translated_text


def _bing_free_blocking_error(parsed: object) -> str | None:
    if not isinstance(parsed, dict):
        return None
    if parsed.get("ShowCaptcha"):
        return "Bing-free translation requested a captcha."
    status_code = parsed.get("StatusCode", parsed.get("statusCode"))
    if status_code == 401:
        return "Bing-free translation limit was exceeded."
    if status_code == 429:
        return "Bing-free translation was rate limited."
    return None


def _resolve_google_free_runtime(config: AppConfig) -> TranslationRuntime:
    _ = config
    return TranslationRuntime(provider="google-free")


def _resolve_bing_free_runtime(config: AppConfig) -> TranslationRuntime:
    _ = config
    return TranslationRuntime(provider="bing-free")


def _resolve_openai_compatible_runtime(config: AppConfig) -> TranslationRuntime:
    return TranslationRuntime(
        provider="openai-compatible",
        base_url=config.resolved_translation_base_url(),
        model=config.translation_model().strip(),
    )


TranslationProviderRegistration = tuple[
    Callable[[AppConfig], TranslationRuntime], type[TranslationBackend]
]
_TRANSLATION_PROVIDERS: dict[str, TranslationProviderRegistration] = {
    "google-free": (_resolve_google_free_runtime, GoogleFreeTranslationBackend),
    "bing-free": (_resolve_bing_free_runtime, BingFreeTranslationBackend),
    "openai-compatible": (
        _resolve_openai_compatible_runtime,
        OpenAICompatibleTranslationBackend,
    ),
}


def translation_provider_names() -> tuple[str, ...]:
    return tuple(_TRANSLATION_PROVIDERS)


def resolve_translation_runtime(config: AppConfig) -> TranslationRuntime:
    """Resolve the active translation settings without credentials or I/O."""
    provider = config.translation.provider.strip().lower()
    registration = _TRANSLATION_PROVIDERS.get(provider)
    if registration is None:
        raise RuntimeError(f"Unsupported translation provider: {provider}")
    resolve_runtime, _ = registration
    return resolve_runtime(config)


class Translator:
    def __init__(
        self, config: AppConfig, backend: TranslationBackend | None = None
    ) -> None:
        self.config = config
        self.backend = backend or build_translation_backend(config)

    def translate_segments(
        self,
        input_path: Path,
        output_path: Path,
        progress_callback: StageProgressCallback | None = None,
    ) -> list[SegmentRecord]:
        segments = _load_resume_segments(input_path, output_path)
        pending = [segment for segment in segments if not segment.text_zh.strip()]
        total_segments = len(segments)
        completed_segments = total_segments - len(pending)
        if progress_callback is not None:
            progress_callback(
                completed_segments,
                max(total_segments, 1),
                "Preparing translation batches",
            )
        if not pending:
            if progress_callback is not None:
                progress_callback(
                    total_segments, max(total_segments, 1), "Translation complete"
                )
            return segments

        configured_batch_size = max(1, self.config.translation.batch_size)
        batch_limit = self.backend.batch_size_limit
        batch_size = (
            min(configured_batch_size, batch_limit)
            if batch_limit
            else configured_batch_size
        )
        max_concurrency = max(1, self.config.translation.max_concurrency)
        batches = [
            pending[start : start + batch_size]
            for start in range(0, len(pending), batch_size)
        ]
        processed = 0
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            future_to_batch = {
                executor.submit(self._translate_batch, batch): batch
                for batch in batches
            }
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    _apply_translations(segments, future.result())
                except Exception as exc:
                    _apply_batch_error(batch, exc)
                processed += len(batch)
                write_json(output_path, segments)
                if progress_callback is not None:
                    progress_callback(
                        completed_segments + processed,
                        total_segments,
                        f"Translating segments {completed_segments + processed}/{total_segments}",
                    )

        if progress_callback is not None:
            progress_callback(total_segments, total_segments, "Translation complete")
        return segments

    def _translate_batch(self, batch: list[SegmentRecord]) -> list[dict[str, str]]:
        return self.backend.translate_batch(batch)


def build_translation_backend(config: AppConfig) -> TranslationBackend:
    runtime = resolve_translation_runtime(config)
    _, backend_type = _TRANSLATION_PROVIDERS[runtime.provider]
    return backend_type(config)


def _load_resume_segments(input_path: Path, output_path: Path) -> list[SegmentRecord]:
    source = output_path if output_path.exists() else input_path
    return read_model_list(source, SegmentRecord)


def _apply_translations(
    segments: list[SegmentRecord], translations: list[dict[str, str]]
) -> None:
    mapping = {item["segment_id"]: item["text_zh"] for item in translations}
    for segment in segments:
        if segment.segment_id in mapping:
            segment.text_zh = mapping[segment.segment_id].strip()
            segment.error = None


def _apply_batch_error(batch: list[SegmentRecord], exc: Exception) -> None:
    error_message = _format_batch_error(exc, batch)
    for segment in batch:
        segment.error = error_message


def _parse_translation_response(
    content: str, batch: list[SegmentRecord]
) -> list[dict[str, str]]:
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
            raise ValueError(
                f"Translation item was not an object: {_excerpt(json.dumps(item, ensure_ascii=False))}"
            )
        segment_id = str(item.get("segment_id", "")).strip()
        text_zh = str(item.get("text_zh", "")).strip()
        if not segment_id or segment_id not in expected_ids:
            raise ValueError(
                f"Translation response returned unexpected segment_id '{segment_id}'. "
                f"Expected one of {sorted(expected_ids)}."
            )
        if segment_id in seen_ids:
            raise ValueError(
                f"Translation response returned duplicate segment_id '{segment_id}'."
            )
        if not text_zh:
            raise ValueError(
                f"Translation response returned empty text_zh for '{segment_id}'."
            )
        seen_ids.add(segment_id)
        normalized.append({"segment_id": segment_id, "text_zh": text_zh})

    missing_ids = expected_ids - seen_ids
    if missing_ids:
        raise ValueError(
            f"Translation response omitted segment_ids: {sorted(missing_ids)}"
        )

    return normalized


def _parse_google_free_translation_response(
    content: str,
    batch: list[SegmentRecord],
) -> list[dict[str, str]]:
    cleaned = content.strip()
    if not cleaned:
        raise RuntimeError("Google-free translation response was empty.")
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Google-free translation response was not valid JSON: {exc.msg}. Response: {_excerpt(cleaned)}"
        ) from exc
    if isinstance(parsed, dict):
        return _parse_google_free_dict_response(parsed, batch, cleaned)
    if isinstance(parsed, list):
        return _parse_google_free_list_response(parsed, batch, cleaned)
    raise RuntimeError(
        "Google-free translation response JSON structure was unexpected. "
        f"Response: {_excerpt(cleaned)}"
    )


def _parse_google_free_dict_response(
    parsed: dict[str, object],
    batch: list[SegmentRecord],
    raw: str,
) -> list[dict[str, str]]:
    sentences = parsed.get("sentences")
    if not isinstance(sentences, list):
        raise RuntimeError(
            "Google-free translation response missing 'sentences' list. "
            f"Response: {_excerpt(raw)}"
        )
    if len(sentences) != len(batch):
        raise RuntimeError(
            f"Google-free translation response shape mismatch: expected {len(batch)} items, got {len(sentences)}. "
            f"Response: {_excerpt(raw)}"
        )

    normalized: list[dict[str, str]] = []
    for segment, item in zip(batch, sentences, strict=True):
        if not isinstance(item, dict):
            raise RuntimeError(
                "Google-free translation sentence item was not an object. "
                f"Response: {_excerpt(json.dumps(item, ensure_ascii=False))}"
            )
        translated_text = str(item.get("trans", "")).strip()
        if not translated_text:
            raise RuntimeError(
                f"Google-free translation response returned empty text for '{segment.segment_id}'. "
                f"Response: {_excerpt(raw)}"
            )
        normalized.append(
            {"segment_id": segment.segment_id, "text_zh": translated_text}
        )
    return normalized


def _parse_google_free_list_response(
    parsed: list[object],
    batch: list[SegmentRecord],
    raw: str,
) -> list[dict[str, str]]:
    if len(parsed) != len(batch):
        raise RuntimeError(
            f"Google-free translation response shape mismatch: expected {len(batch)} items, got {len(parsed)}. "
            f"Response: {_excerpt(raw)}"
        )

    normalized: list[dict[str, str]] = []
    for segment, item in zip(batch, parsed, strict=True):
        translated_text = item[0] if isinstance(item, list) and item else item
        translated_text = str(translated_text).strip()
        if not translated_text:
            raise RuntimeError(
                f"Google-free translation response returned empty text for '{segment.segment_id}'. "
                f"Response: {_excerpt(raw)}"
            )
        normalized.append(
            {"segment_id": segment.segment_id, "text_zh": translated_text}
        )
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
    resolved = config.resolve_provider_api_key(
        config.translation.provider, purpose="translation"
    )
    if resolved:
        return resolved
    return os.getenv("OPENAI_API_KEY", "")
