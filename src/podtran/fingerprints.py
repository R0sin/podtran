from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from podtran.artifacts import atomic_write_text, read_json_data
from podtran.config import AppConfig, model_dump


TRANSCRIBE_CONFIG_KEYS = [
    "asr.model",
    "asr.compute_type",
    "asr.device",
    "asr.language",
    "asr.align_model",
]
TRANSLATE_CONFIG_KEYS = [
    "translation.provider",
    "translation.base_url",
    "translation.model",
]
VOICE_CLONE_CONFIG_KEYS = [
    "tts.provider",
    "tts.base_url",
    "tts.voice_mode",
    "tts.model",
    "tts.enrollment_model",
    "tts.language_type",
    "tts.clone_min_ref_seconds",
    "tts.clone_max_ref_seconds",
    "tts.customization_url",
]
TTS_CONFIG_KEYS = [
    "tts.provider",
    "tts.base_url",
    "tts.voice_mode",
    "tts.model",
    "tts.language_type",
]
SYNTHESIZE_CONFIG_KEYS = list(dict.fromkeys([*VOICE_CLONE_CONFIG_KEYS, *TTS_CONFIG_KEYS]))
COMPOSE_CONFIG_KEYS = [
    "compose.mode",
    "compose.gap_en_to_cn_ms",
    "compose.gap_cn_to_en_ms",
    "compose.output_bitrate",
]


def normalize_text(text: str) -> str:
    return " ".join(text.split()).strip()


def stable_hash(value: Any) -> str:
    payload = json.dumps(_normalize_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class FingerprintService:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.index_path = index_dir / "audio_hashes.json"
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def hash_audio(self, path: Path) -> str:
        resolved = path.resolve()
        stat = resolved.stat()
        cache_key = f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}"
        index = self._load_index()
        cached = index.get(cache_key)
        if cached:
            return str(cached)

        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        value = digest.hexdigest()
        index[cache_key] = value
        self._save_index(index)
        return value

    def hash_json(self, value: Path | BaseModel | dict | list) -> str:
        if isinstance(value, Path):
            payload = read_json_data(value)
        else:
            payload = model_dump(value) if isinstance(value, BaseModel) else value
        return stable_hash(payload)

    def hash_config_subset(self, config: AppConfig, keys: list[str]) -> str:
        return stable_hash(self.config_subset(config, keys))

    def config_subset(self, config: AppConfig, keys: list[str]) -> dict[str, Any]:
        raw = model_dump(config)
        return {key: _resolve_dotted_key(raw, key) for key in keys}

    def hash_value(self, value: Any) -> str:
        return stable_hash(value)

    def build_stage_cache_key(
        self,
        stage: str,
        stage_version: int,
        input_fingerprints: dict[str, str],
        config_fingerprint: str,
    ) -> str:
        return stable_hash(
            {
                "stage": stage,
                "stage_version": stage_version,
                "input_fingerprints": input_fingerprints,
                "config_fingerprint": config_fingerprint,
            }
        )

    def _load_index(self) -> dict[str, str]:
        if not self.index_path.exists():
            return {}
        data = read_json_data(self.index_path)
        if not isinstance(data, dict):
            return {}
        return {str(key): str(value) for key, value in data.items()}

    def _save_index(self, data: dict[str, str]) -> None:
        atomic_write_text(
            self.index_path,
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        )


def _normalize_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _normalize_value(model_dump(value))
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(key): _normalize_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    return value


def _resolve_dotted_key(data: dict[str, Any], key: str) -> Any:
    current: Any = data
    for part in key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current

