from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from podtran.models import VoiceMode

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


DEFAULT_TRANSLATION_PROVIDER = "google-free"
DEFAULT_TRANSLATION_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_TRANSLATION_MODEL = "qwen-flash"
DEFAULT_TTS_PROVIDER = "qwen-local"
DEFAULT_TTS_PRESET_MODEL = "qwen3-tts-flash"
DEFAULT_TTS_CLONE_MODEL = "qwen3-tts-vc-2026-01-22"
DEFAULT_TTS_ENROLLMENT_MODEL = "qwen-voice-enrollment"
DEFAULT_TTS_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"
DEFAULT_TTS_TIMEOUT_SECONDS = 300
DEFAULT_VLLM_OMNI_LANGUAGE = "Auto"
DEFAULT_QWEN_LOCAL_MODEL_SIZE = "0.6B"
DEFAULT_QWEN_LOCAL_LANGUAGE = "Chinese"
DEFAULT_QWEN_LOCAL_TORCH_DTYPE = "auto"
DEFAULT_QWEN_LOCAL_ATTN_IMPLEMENTATION = "auto"
DEFAULT_WORKDIR = Path("~/.podtran")
DEFAULT_CONFIG_FILENAME = "config.toml"
DEFAULT_FALLBACK_VOICES = ["Cherry", "Serena", "Ethan", "Chelsie"]
LEGACY_TRANSLATION_KEYS = (
    "base_url",
    "model",
)
LEGACY_TTS_KEYS = (
    "base_url",
    "voice_mode",
    "model",
    "enrollment_model",
    "clone_min_ref_seconds",
    "clone_max_ref_seconds",
    "customization_url",
    "fallback_voices",
    "voice_map",
    "vllm_omni",
)


class DashScopeProviderConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    api_key: str = ""
    tts_base_url: str = DEFAULT_TTS_BASE_URL
    tts_preset_model: str = DEFAULT_TTS_PRESET_MODEL
    tts_clone_model: str = DEFAULT_TTS_CLONE_MODEL
    tts_enrollment_model: str = DEFAULT_TTS_ENROLLMENT_MODEL


class OpenAICompatibleProviderConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    translation_base_url: str = DEFAULT_TRANSLATION_BASE_URL
    translation_api_key: str = ""
    translation_model: str = DEFAULT_TRANSLATION_MODEL
    tts_base_url: str = ""
    tts_api_key: str = ""
    tts_model: str = ""


class VllmOmniProviderConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    base_url: str = ""
    api_key: str = ""
    model: str = ""
    language: str = DEFAULT_VLLM_OMNI_LANGUAGE
    instructions: str = ""
    x_vector_only_mode: bool = False


class QwenLocalProviderConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    clone_model_size: str = DEFAULT_QWEN_LOCAL_MODEL_SIZE
    preset_model_size: str = DEFAULT_QWEN_LOCAL_MODEL_SIZE
    device: str = "auto"
    torch_dtype: str = DEFAULT_QWEN_LOCAL_TORCH_DTYPE
    attn_implementation: str = DEFAULT_QWEN_LOCAL_ATTN_IMPLEMENTATION
    language: str = DEFAULT_QWEN_LOCAL_LANGUAGE
    instructions: str = ""
    x_vector_only_mode: bool = False


class ProvidersConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    dashscope: DashScopeProviderConfig = Field(default_factory=DashScopeProviderConfig)
    openai_compatible: OpenAICompatibleProviderConfig = Field(
        default_factory=OpenAICompatibleProviderConfig
    )
    vllm_omni: VllmOmniProviderConfig = Field(default_factory=VllmOmniProviderConfig)
    qwen_local: QwenLocalProviderConfig = Field(default_factory=QwenLocalProviderConfig)


class TranslationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    provider: str = DEFAULT_TRANSLATION_PROVIDER
    timeout_seconds: int = 120
    batch_size: int = 8
    max_concurrency: int = 4


class TTSPresetConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    voice_map: dict[str, str] = Field(default_factory=dict)
    fallback_voices: list[str] = Field(
        default_factory=lambda: list(DEFAULT_FALLBACK_VOICES)
    )


class TTSCloneConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    min_ref_seconds: int = 10
    max_ref_seconds: int = 20


class TTSConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    provider: str = DEFAULT_TTS_PROVIDER
    mode: VoiceMode = "auto"
    timeout_seconds: int = DEFAULT_TTS_TIMEOUT_SECONDS
    batch_size: int = 1
    max_concurrency: int = 4
    preset: TTSPresetConfig = Field(default_factory=TTSPresetConfig)
    clone: TTSCloneConfig = Field(default_factory=TTSCloneConfig)

    def normalized_mode(self) -> str:
        return self.mode.strip().lower()

    def effective_mode(self, provider: str) -> Literal["preset", "clone"]:
        mode = self.normalized_mode()
        if mode in {"preset", "clone"}:
            return mode  # type: ignore[return-value]

        normalized_provider = provider.strip().lower()
        if normalized_provider == "openai-compatible":
            return "preset"
        return "clone"


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

    def resolve_provider_api_key(self, provider: str, purpose: str = "") -> str:
        normalized = provider.strip().lower()
        if normalized == "dashscope":
            return self.providers.dashscope.api_key.strip()
        if normalized == "openai-compatible":
            if purpose == "translation":
                return self.providers.openai_compatible.translation_api_key.strip()
            if purpose == "tts":
                return self.providers.openai_compatible.tts_api_key.strip()
        if normalized == "vllm-omni" and purpose == "tts":
            return self.providers.vllm_omni.api_key.strip()
        return ""

    def resolved_translation_base_url(self) -> str:
        provider = self.translation.provider.strip().lower()
        if provider == "openai-compatible":
            return self.providers.openai_compatible.translation_base_url.strip().rstrip(
                "/"
            )
        return ""

    def translation_model(self) -> str:
        model = self.providers.openai_compatible.translation_model.strip()
        return model or DEFAULT_TRANSLATION_MODEL

    def resolved_tts_base_url(self) -> str:
        provider = self.tts.provider.strip().lower()
        if provider == "dashscope":
            return (
                self.providers.dashscope.tts_base_url.strip().rstrip("/")
                or DEFAULT_TTS_BASE_URL
            )
        if provider == "openai-compatible":
            return self.providers.openai_compatible.tts_base_url.strip().rstrip("/")
        if provider == "vllm-omni":
            return self.providers.vllm_omni.base_url.strip().rstrip("/")
        return ""

    def tts_preset_model(self) -> str:
        provider = self.tts.provider.strip().lower()
        if provider == "dashscope":
            return (
                self.providers.dashscope.tts_preset_model.strip()
                or DEFAULT_TTS_PRESET_MODEL
            )
        if provider == "openai-compatible":
            return (
                self.providers.openai_compatible.tts_model.strip()
                or DEFAULT_TTS_PRESET_MODEL
            )
        if provider == "vllm-omni":
            return self.providers.vllm_omni.model.strip()
        if provider == "qwen-local":
            return f"qwen-local:customvoice:{self.providers.qwen_local.preset_model_size.strip() or DEFAULT_QWEN_LOCAL_MODEL_SIZE}"
        return ""

    def tts_clone_model(self) -> str:
        provider = self.tts.provider.strip().lower()
        if provider == "dashscope":
            return (
                self.providers.dashscope.tts_clone_model.strip()
                or DEFAULT_TTS_CLONE_MODEL
            )
        if provider == "vllm-omni":
            return self.providers.vllm_omni.model.strip()
        if provider == "qwen-local":
            return f"qwen-local:base:{self.providers.qwen_local.clone_model_size.strip() or DEFAULT_QWEN_LOCAL_MODEL_SIZE}"
        return ""

    def tts_enrollment_model(self) -> str:
        return (
            self.providers.dashscope.tts_enrollment_model.strip()
            or DEFAULT_TTS_ENROLLMENT_MODEL
        )


def build_init_config(
    hf_token: str,
    dashscope_api_key: str,
    translation_model: str,
    tts_model_size: str,
) -> AppConfig:
    config = AppConfig()
    config.hf_token = hf_token.strip()
    config.providers.dashscope.api_key = dashscope_api_key.strip()
    config.translation.provider = DEFAULT_TRANSLATION_PROVIDER
    config.providers.openai_compatible.translation_model = (
        translation_model.strip() or DEFAULT_TRANSLATION_MODEL
    )
    config.tts.provider = DEFAULT_TTS_PROVIDER
    config.providers.qwen_local.clone_model_size = (
        tts_model_size.strip() or DEFAULT_QWEN_LOCAL_MODEL_SIZE
    )
    return config


def load_config(path: Path) -> AppConfig:
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found: {path}\nRun 'podtran init' to create a default config."
        )
    data = load_config_data(path)
    _raise_for_legacy_translation_config(path, data)
    _raise_for_legacy_tts_config(path, data)
    return AppConfig.model_validate(data)


def load_config_data(path: Path) -> dict[str, Any]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(
            f"Config file must contain a TOML table at the top level: {path}"
        )
    return data


def resolve_workdir(
    workdir_override: Path | str | None = None, config_path: Path | None = None
) -> Path:
    if workdir_override is not None:
        return Path(workdir_override).expanduser().resolve()
    if config_path is not None:
        return config_path.expanduser().resolve().parent
    return DEFAULT_WORKDIR.expanduser().resolve()


def resolve_config_path(
    config_override: Path | str | None = None,
    workdir_override: Path | str | None = None,
) -> Path:
    if config_override is not None:
        return Path(config_override).expanduser().resolve()
    return resolve_workdir(workdir_override) / DEFAULT_CONFIG_FILENAME


def render_config_toml(config: AppConfig) -> str:
    lines = [
        "# This config lives under ~/.podtran/config.toml by default.",
        "# Use --workdir to move config, tasks, and cache into a different directory.",
        f'hf_token = "{config.hf_token}"',
        "",
        "[providers.dashscope]",
        f'api_key = "{config.providers.dashscope.api_key}"',
        f'tts_base_url = "{config.providers.dashscope.tts_base_url}"',
        f'tts_preset_model = "{config.providers.dashscope.tts_preset_model}"',
        f'tts_clone_model = "{config.providers.dashscope.tts_clone_model}"',
        f'tts_enrollment_model = "{config.providers.dashscope.tts_enrollment_model}"',
        "",
        "[providers.openai_compatible]",
        f'translation_base_url = "{config.providers.openai_compatible.translation_base_url}"',
        f'translation_api_key = "{config.providers.openai_compatible.translation_api_key}"',
        f'translation_model = "{config.providers.openai_compatible.translation_model}"',
        f'tts_base_url = "{config.providers.openai_compatible.tts_base_url}"',
        f'tts_api_key = "{config.providers.openai_compatible.tts_api_key}"',
        f'tts_model = "{config.providers.openai_compatible.tts_model}"',
        "",
        "[providers.vllm_omni]",
        f'base_url = "{config.providers.vllm_omni.base_url}"',
        f'api_key = "{config.providers.vllm_omni.api_key}"',
        f'model = "{config.providers.vllm_omni.model}"',
        f'language = "{config.providers.vllm_omni.language}"',
        f'instructions = "{config.providers.vllm_omni.instructions}"',
        f"x_vector_only_mode = {str(config.providers.vllm_omni.x_vector_only_mode).lower()}",
        "",
        "[providers.qwen_local]",
        f'clone_model_size = "{config.providers.qwen_local.clone_model_size}"',
        f'preset_model_size = "{config.providers.qwen_local.preset_model_size}"',
        f'device = "{config.providers.qwen_local.device}"',
        f'torch_dtype = "{config.providers.qwen_local.torch_dtype}"',
        f'attn_implementation = "{config.providers.qwen_local.attn_implementation}"',
        f'language = "{config.providers.qwen_local.language}"',
        f'instructions = "{config.providers.qwen_local.instructions}"',
        f"x_vector_only_mode = {str(config.providers.qwen_local.x_vector_only_mode).lower()}",
        "",
        "[asr]",
        f'model = "{config.asr.model}"',
        f'compute_type = "{config.asr.compute_type}"',
        f'device = "{config.asr.device}"',
        f'language = "{config.asr.language}"',
        f"batch_size = {config.asr.batch_size}",
        f'align_model = "{config.asr.align_model}"',
        "",
        "[translation]",
        "# google-free uses the unofficial Google Translate web endpoint and ignores base_url/model.",
        f'provider = "{config.translation.provider}"',
        f"timeout_seconds = {config.translation.timeout_seconds}",
        f"batch_size = {config.translation.batch_size}",
        f"max_concurrency = {config.translation.max_concurrency}",
        "",
        "[tts]",
        '# Mode: "auto" chooses the recommended backend behavior; "preset" and "clone" force a mode.',
        f'provider = "{config.tts.provider}"',
        f'mode = "{config.tts.mode}"',
        f"timeout_seconds = {config.tts.timeout_seconds}",
        "# qwen-local only: number of text segments per local model call.",
        f"batch_size = {config.tts.batch_size}",
        "# API TTS worker concurrency. qwen-local always uses a single local model worker.",
        f"max_concurrency = {config.tts.max_concurrency}",
        "",
        "[tts.preset]",
        f"fallback_voices = {_render_list(config.tts.preset.fallback_voices)}",
        "",
        "[tts.preset.voice_map]",
        *_render_mapping(config.tts.preset.voice_map),
        "",
        "[tts.clone]",
        f"min_ref_seconds = {config.tts.clone.min_ref_seconds}",
        f"max_ref_seconds = {config.tts.clone.max_ref_seconds}",
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
    keys = [key for key in LEGACY_TTS_KEYS if key in tts]
    preset = tts.get("preset")
    if isinstance(preset, dict) and "model" in preset:
        keys.append("preset.model")
    clone = tts.get("clone")
    if isinstance(clone, dict) and "model" in clone:
        keys.append("clone.model")
    return keys


def detect_legacy_translation_keys(data: dict[str, Any]) -> list[str]:
    translation = data.get("translation")
    if not isinstance(translation, dict):
        return []
    keys = [key for key in LEGACY_TRANSLATION_KEYS if key in translation]
    provider = str(translation.get("provider", "") or "").strip().lower()
    if provider == "dashscope":
        keys.append("provider=dashscope")
    return keys


def _raise_for_legacy_translation_config(path: Path, data: dict[str, Any]) -> None:
    legacy_keys = detect_legacy_translation_keys(data)
    if not legacy_keys:
        return

    formatted_keys = ", ".join(f"translation.{key}" for key in legacy_keys)
    raise ValueError(
        f"Legacy translation config detected in {path}: {formatted_keys}. "
        "Please run 'podtran init' to rebuild the config. "
        "DashScope compatible-mode translation is now configured as provider 'openai-compatible' "
        "under [providers.openai_compatible]."
    )


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
