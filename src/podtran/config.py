from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


DEFAULT_TRANSLATION_PROVIDER = "dashscope"
DEFAULT_TRANSLATION_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_TRANSLATION_MODEL = "qwen-flash"
DEFAULT_TTS_PROVIDER = "dashscope"
DEFAULT_TTS_PRESET_MODEL = "qwen3-tts-flash"
DEFAULT_TTS_CLONE_MODEL = "qwen3-tts-vc-2026-01-22"
DEFAULT_TTS_ENROLLMENT_MODEL = "qwen-voice-enrollment"
DEFAULT_TTS_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
DEFAULT_WORKDIR = Path("~/.podtran")
DEFAULT_CONFIG_FILENAME = "podtran.toml"


class ProviderConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    api_key: str = ""


class ProvidersConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    dashscope: ProviderConfig = Field(default_factory=ProviderConfig)


class TranslationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    provider: str = DEFAULT_TRANSLATION_PROVIDER
    base_url: str = ""
    model: str = DEFAULT_TRANSLATION_MODEL
    timeout_seconds: int = 120
    batch_size: int = 8
    max_concurrency: int = 4

    def resolved_base_url(self) -> str:
        base_url = self.base_url.strip().rstrip("/")
        if base_url:
            return base_url
        if self.provider.strip().lower() == "dashscope":
            return DEFAULT_TRANSLATION_BASE_URL
        return ""


class TTSConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    provider: str = DEFAULT_TTS_PROVIDER
    base_url: str = ""
    model: str = DEFAULT_TTS_CLONE_MODEL
    voice_mode: str = "clone"
    enrollment_model: str = DEFAULT_TTS_ENROLLMENT_MODEL
    voice_map: dict[str, str] = Field(default_factory=dict)
    fallback_voices: list[str] = Field(default_factory=lambda: ["Cherry", "Serena", "Ethan", "Chelsie"])
    language_type: str = "Chinese"
    timeout_seconds: int = 120
    max_concurrency: int = 4
    clone_min_ref_seconds: int = 10
    clone_max_ref_seconds: int = 20
    customization_url: str = ""

    def resolved_model(self) -> str:
        model = self.model.strip()
        voice_mode = self.voice_mode.strip().lower()
        if voice_mode == "clone" and (not model or model == DEFAULT_TTS_PRESET_MODEL):
            return DEFAULT_TTS_CLONE_MODEL
        if voice_mode != "clone" and (not model or model == DEFAULT_TTS_CLONE_MODEL):
            return DEFAULT_TTS_PRESET_MODEL
        return model

    def resolved_backend(self) -> str:
        if self.provider.strip().lower() == "dashscope":
            return "dashscope"
        return "openai_compatible"

    def resolved_base_url(self) -> str:
        base_url = self.base_url.strip().rstrip("/")
        if base_url:
            return base_url
        if self.provider.strip().lower() == "dashscope":
            return DEFAULT_TTS_BASE_URL
        return ""

    def resolved_customization_url(self) -> str:
        customization_url = self.customization_url.strip().rstrip("/")
        if customization_url:
            return customization_url
        base_url = self.resolved_base_url()
        if not base_url:
            return ""
        return base_url + "/services/audio/tts/customization"


class ASRConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = "medium"
    compute_type: str = "int8"
    device: str = "cpu"
    language: str = "en"
    batch_size: int = 4
    align_model: str = ""


class ComposeConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mode: str = "interleave"
    block_pause_threshold: float = 0.8
    max_block_duration: float = 15.0
    gap_en_to_cn_ms: int = 200
    gap_cn_to_en_ms: int = 400
    output_bitrate: str = "192k"


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hf_token: str = ""
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    translation: TranslationConfig = Field(default_factory=TranslationConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    compose: ComposeConfig = Field(default_factory=ComposeConfig)

    def resolve_provider_api_key(self, provider: str) -> str:
        normalized = provider.strip().lower()
        if normalized == "dashscope":
            return self.providers.dashscope.api_key.strip()
        return ""


def build_init_config(
    hf_token: str,
    dashscope_api_key: str,
    translation_model: str,
    tts_model: str,
) -> AppConfig:
    config = AppConfig()
    config.hf_token = hf_token.strip()
    config.providers.dashscope.api_key = dashscope_api_key.strip()
    config.translation.provider = DEFAULT_TRANSLATION_PROVIDER
    config.translation.model = translation_model.strip() or DEFAULT_TRANSLATION_MODEL
    config.tts.provider = DEFAULT_TTS_PROVIDER
    config.tts.model = tts_model.strip() or DEFAULT_TTS_CLONE_MODEL
    return config


def load_config(path: Path) -> AppConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}\nRun 'podtran init' to create a default config.")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return AppConfig.model_validate(data)


def resolve_workdir(workdir_override: Path | str | None = None, config_path: Path | None = None) -> Path:
    if workdir_override is not None:
        return Path(workdir_override).expanduser().resolve()
    if config_path is not None:
        return config_path.expanduser().resolve().parent
    return DEFAULT_WORKDIR.expanduser().resolve()


def resolve_config_path(config_override: Path | str | None = None, workdir_override: Path | str | None = None) -> Path:
    if config_override is not None:
        return Path(config_override).expanduser().resolve()
    return resolve_workdir(workdir_override) / DEFAULT_CONFIG_FILENAME


def render_config_toml(config: AppConfig) -> str:
    lines = [
        "# This config lives under ~/.podtran/podtran.toml by default.",
        "# Use --workdir to move config, tasks, and cache into a different directory.",
        '# Leave base_url and customization_url empty to use the provider defaults.',
        f'hf_token = "{config.hf_token}"',
        "",
        "[providers.dashscope]",
        f'api_key = "{config.providers.dashscope.api_key}"',
        "",
        "[translation]",
        f'provider = "{config.translation.provider}"',
        f'base_url = "{config.translation.base_url}"',
        f'model = "{config.translation.model}"',
        f"timeout_seconds = {config.translation.timeout_seconds}",
        f"batch_size = {config.translation.batch_size}",
        f"max_concurrency = {config.translation.max_concurrency}",
        "",
        "[tts]",
        '# Clone-first defaults. Switch voice_mode to "preset" to use preset voices below.',
        f'provider = "{config.tts.provider}"',
        f'base_url = "{config.tts.base_url}"',
        f'voice_mode = "{config.tts.voice_mode}"',
        f'model = "{config.tts.resolved_model()}"',
        f'enrollment_model = "{config.tts.enrollment_model}"',
        f'language_type = "{config.tts.language_type}"',
        f"timeout_seconds = {config.tts.timeout_seconds}",
        "# Segment-level synthesis concurrency. Lower this if your TTS provider rate-limits aggressively.",
        f"max_concurrency = {config.tts.max_concurrency}",
        f"clone_min_ref_seconds = {config.tts.clone_min_ref_seconds}",
        f"clone_max_ref_seconds = {config.tts.clone_max_ref_seconds}",
        f'customization_url = "{config.tts.customization_url}"',
        "# Preset-only voice assignment settings.",
        f"fallback_voices = {_render_list(config.tts.fallback_voices)}",
        "",
        "[tts.voice_map]",
        *_render_mapping(config.tts.voice_map),
        "",
        "[asr]",
        f'model = "{config.asr.model}"',
        f'compute_type = "{config.asr.compute_type}"',
        f'device = "{config.asr.device}"',
        f'language = "{config.asr.language}"',
        f"batch_size = {config.asr.batch_size}",
        f'align_model = "{config.asr.align_model}"',
        "",
        "[compose]",
        "# interleave = English + Chinese, replace = Chinese only.",
        f'mode = "{config.compose.mode}"',
        f"block_pause_threshold = {config.compose.block_pause_threshold}",
        f"max_block_duration = {config.compose.max_block_duration}",
        f"gap_en_to_cn_ms = {config.compose.gap_en_to_cn_ms}",
        f"gap_cn_to_en_ms = {config.compose.gap_cn_to_en_ms}",
        f'output_bitrate = "{config.compose.output_bitrate}"',
        "",
    ]
    return "\n".join(lines)


def write_default_config(path: Path, force: bool = False) -> Path:
    if path.exists() and not force:
        raise FileExistsError(f"Config already exists: {path}")

    config = AppConfig()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_config_toml(config), encoding="utf-8")
    return path


def _render_list(values: list[str]) -> str:
    return "[" + ", ".join(f'"{item}"' for item in values) + "]"


def _render_mapping(values: dict[str, str]) -> list[str]:
    if not values:
        return ['# SPEAKER_00 = "Cherry"', '# SPEAKER_01 = "Ethan"']
    return [f'{key} = "{value}"' for key, value in values.items()]


def model_dump(data: BaseModel | list[BaseModel] | dict[str, Any]) -> Any:
    if isinstance(data, BaseModel):
        return data.model_dump()
    if isinstance(data, list):
        return [model_dump(item) for item in data]
    return data
