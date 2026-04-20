from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from podtran.models import VoiceMode

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
DEFAULT_FALLBACK_VOICES = ["Cherry", "Serena", "Ethan", "Chelsie"]
LEGACY_TTS_KEYS = (
    "voice_mode",
    "model",
    "enrollment_model",
    "clone_min_ref_seconds",
    "clone_max_ref_seconds",
    "customization_url",
    "fallback_voices",
    "voice_map",
)


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


class TTSPresetConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = DEFAULT_TTS_PRESET_MODEL
    voice_map: dict[str, str] = Field(default_factory=dict)
    fallback_voices: list[str] = Field(default_factory=lambda: list(DEFAULT_FALLBACK_VOICES))


class TTSCloneConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = DEFAULT_TTS_CLONE_MODEL
    min_ref_seconds: int = 10
    max_ref_seconds: int = 20


class TTSConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    provider: str = DEFAULT_TTS_PROVIDER
    base_url: str = ""
    mode: VoiceMode = "clone"
    timeout_seconds: int = 120
    max_concurrency: int = 4
    preset: TTSPresetConfig = Field(default_factory=TTSPresetConfig)
    clone: TTSCloneConfig = Field(default_factory=TTSCloneConfig)

    def resolved_base_url(self) -> str:
        base_url = self.base_url.strip().rstrip("/")
        if base_url:
            return base_url
        if self.provider.strip().lower() == "dashscope":
            return DEFAULT_TTS_BASE_URL
        return ""

    def normalized_mode(self) -> str:
        return self.mode.strip().lower()

    def preset_model(self) -> str:
        model = self.preset.model.strip()
        return model or DEFAULT_TTS_PRESET_MODEL

    def clone_model(self) -> str:
        model = self.clone.model.strip()
        return model or DEFAULT_TTS_CLONE_MODEL


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
    config.tts.clone.model = tts_model.strip() or DEFAULT_TTS_CLONE_MODEL
    return config


def load_config(path: Path) -> AppConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}\nRun 'podtran init' to create a default config.")
    data = load_config_data(path)
    _raise_for_legacy_tts_config(path, data)
    return AppConfig.model_validate(data)


def load_config_data(path: Path) -> dict[str, Any]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a TOML table at the top level: {path}")
    return data


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
        '# Set mode to "preset" to synthesize with preset voices only.',
        f'provider = "{config.tts.provider}"',
        f'base_url = "{config.tts.base_url}"',
        f'mode = "{config.tts.mode}"',
        f"timeout_seconds = {config.tts.timeout_seconds}",
        "# Segment-level synthesis concurrency. Lower this if your TTS provider rate-limits aggressively.",
        f"max_concurrency = {config.tts.max_concurrency}",
        "",
        "[tts.preset]",
        f'model = "{config.tts.preset_model()}"',
        f"fallback_voices = {_render_list(config.tts.preset.fallback_voices)}",
        "",
        "[tts.preset.voice_map]",
        *_render_mapping(config.tts.preset.voice_map),
        "",
        "[tts.clone]",
        f'model = "{config.tts.clone_model()}"',
        f"min_ref_seconds = {config.tts.clone.min_ref_seconds}",
        f"max_ref_seconds = {config.tts.clone.max_ref_seconds}",
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


def detect_legacy_tts_keys(data: dict[str, Any]) -> list[str]:
    tts = data.get("tts")
    if not isinstance(tts, dict):
        return []
    return [key for key in LEGACY_TTS_KEYS if key in tts]


def _raise_for_legacy_tts_config(path: Path, data: dict[str, Any]) -> None:
    legacy_keys = detect_legacy_tts_keys(data)
    if not legacy_keys:
        return

    formatted_keys = ", ".join(f"tts.{key}" for key in legacy_keys)
    raise ValueError(
        f"Legacy TTS config detected in {path}: {formatted_keys}. "
        "Please run 'podtran init' to rebuild the config. "
        "The init command will back up the old file and preserve hf_token and provider API keys."
    )


def model_dump(data: BaseModel | list[BaseModel] | dict[str, Any]) -> Any:
    if isinstance(data, BaseModel):
        return data.model_dump()
    if isinstance(data, list):
        return [model_dump(item) for item in data]
    return data
