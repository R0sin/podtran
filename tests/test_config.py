from pathlib import Path

import pytest

from podtran.config import (
    AppConfig,
    DEFAULT_QWEN_LOCAL_MODEL_SIZE,
    DEFAULT_QWEN_LOCAL_ATTN_IMPLEMENTATION,
    DEFAULT_QWEN_LOCAL_TORCH_DTYPE,
    DEFAULT_TRANSLATION_BASE_URL,
    DEFAULT_TRANSLATION_MODEL,
    DEFAULT_TTS_BASE_URL,
    DEFAULT_TTS_CLONE_MODEL,
    DEFAULT_TTS_PRESET_MODEL,
    DEFAULT_TTS_TIMEOUT_SECONDS,
    DEFAULT_VLLM_OMNI_LANGUAGE,
    build_init_config,
    detect_legacy_translation_keys,
    load_config,
    render_config_toml,
    resolve_config_path,
    resolve_workdir,
    write_default_config,
)
from podtran.fingerprints import (
    FingerprintService,
    TTS_CONFIG_KEYS,
    VOICE_CLONE_CONFIG_KEYS,
)


def test_load_config_accepts_provider_scoped_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
hf_token = ""

[providers.dashscope]
api_key = "dash-key"
tts_base_url = "https://dash.example/api/v1"
tts_preset_model = "dash-preset"
tts_clone_model = "dash-clone"
tts_enrollment_model = "dash-enroll"

[providers.openai_compatible]
translation_base_url = "http://localhost:9000/v1"
translation_api_key = "translate-key"
translation_model = "translate-model"
tts_base_url = "http://localhost:9001/v1"
tts_api_key = "tts-key"
tts_model = "tts-model"

[providers.vllm_omni]
base_url = "http://localhost:8091/v1"
api_key = "local-key"
model = "Qwen/Qwen3-TTS"
language = "zh"
instructions = "Warm broadcast tone."
x_vector_only_mode = true

[providers.qwen_local]
clone_model_size = "1.7B"
preset_model_size = "0.6B"
device = "cuda"
torch_dtype = "float16"
attn_implementation = "flash_attention_2"
language = "Chinese"
instructions = "steady"
x_vector_only_mode = true

[translation]
provider = "openai-compatible"
timeout_seconds = 90
batch_size = 2
max_concurrency = 3

[tts]
provider = "qwen-local"
mode = "clone"
timeout_seconds = 60
batch_size = 4
max_concurrency = 2

[tts.preset]
fallback_voices = ["Vivian"]

[tts.preset.voice_map]
SPEAKER_00 = "Vivian"

[tts.clone]
min_ref_seconds = 8
max_ref_seconds = 18
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.providers.dashscope.api_key == "dash-key"
    assert config.providers.dashscope.tts_clone_model == "dash-clone"
    assert (
        config.providers.openai_compatible.translation_base_url
        == "http://localhost:9000/v1"
    )
    assert config.providers.vllm_omni.instructions == "Warm broadcast tone."
    assert config.providers.qwen_local.clone_model_size == "1.7B"
    assert config.providers.qwen_local.torch_dtype == "float16"
    assert config.providers.qwen_local.attn_implementation == "flash_attention_2"
    assert config.translation.provider == "openai-compatible"
    assert config.translation.timeout_seconds == 90
    assert config.tts.provider == "qwen-local"
    assert config.tts.batch_size == 4
    assert config.tts.clone.min_ref_seconds == 8
    assert config.tts.preset.voice_map == {"SPEAKER_00": "Vivian"}


def test_load_config_rejects_legacy_translation_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[translation]
provider = "openai-compatible"
base_url = "http://localhost:9000/v1"
model = "translate-model"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Legacy translation config detected"):
        load_config(config_path)


def test_load_config_rejects_dashscope_translation_provider(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[translation]
provider = "dashscope"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="provider 'openai-compatible'"):
        load_config(config_path)


def test_load_config_rejects_legacy_tts_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[tts]
provider = "vllm-omni"
base_url = "http://localhost:8091/v1"

[tts.preset]
model = "old-preset"

[tts.clone]
model = "old-clone"

[tts.vllm_omni]
language = "Auto"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Legacy TTS config detected"):
        load_config(config_path)


def test_provider_helpers_resolve_defaults_and_overrides() -> None:
    config = AppConfig()

    assert config.translation.provider == "google-free"
    assert config.tts.provider == "qwen-local"
    assert config.tts.mode == "auto"
    assert config.tts.effective_mode("dashscope") == "clone"
    assert config.tts.effective_mode("vllm-omni") == "clone"
    assert config.tts.effective_mode("qwen-local") == "clone"
    assert config.tts.effective_mode("openai-compatible") == "preset"
    assert config.resolved_translation_base_url() == ""
    assert config.translation_model() == DEFAULT_TRANSLATION_MODEL
    assert (
        config.providers.openai_compatible.translation_base_url
        == DEFAULT_TRANSLATION_BASE_URL
    )
    assert config.resolved_tts_base_url() == ""
    assert config.tts_preset_model() == "qwen-local:customvoice:0.6B"
    assert config.tts_clone_model() == "qwen-local:base:0.6B"
    assert config.tts.timeout_seconds == DEFAULT_TTS_TIMEOUT_SECONDS
    assert config.tts.batch_size == 1
    assert config.tts.max_concurrency == 4
    assert config.providers.vllm_omni.language == DEFAULT_VLLM_OMNI_LANGUAGE
    assert config.providers.qwen_local.clone_model_size == DEFAULT_QWEN_LOCAL_MODEL_SIZE
    assert config.providers.qwen_local.torch_dtype == DEFAULT_QWEN_LOCAL_TORCH_DTYPE
    assert (
        config.providers.qwen_local.attn_implementation
        == DEFAULT_QWEN_LOCAL_ATTN_IMPLEMENTATION
    )

    dashscope = AppConfig(tts={"provider": "dashscope"})
    assert dashscope.resolved_tts_base_url() == DEFAULT_TTS_BASE_URL
    assert dashscope.tts_preset_model() == DEFAULT_TTS_PRESET_MODEL
    assert dashscope.tts_clone_model() == DEFAULT_TTS_CLONE_MODEL

    custom = AppConfig(
        translation={"provider": "openai-compatible"},
        tts={"provider": "vllm-omni"},
        providers={
            "openai_compatible": {
                "translation_base_url": "https://example.com/v1/",
                "translation_model": "m",
            },
            "vllm_omni": {"base_url": "https://tts.example.com/root/", "model": "tts"},
        },
    )
    assert custom.resolved_translation_base_url() == "https://example.com/v1"
    assert custom.translation_model() == "m"
    assert custom.resolved_tts_base_url() == "https://tts.example.com/root"
    assert custom.tts_clone_model() == "tts"


def test_build_init_config_sets_provider_managed_auth_and_models() -> None:
    config = build_init_config(
        hf_token="hf-token",
        dashscope_api_key="dash-key",
        translation_model="custom-translate",
        tts_model_size="1.7B",
    )

    assert config.hf_token == "hf-token"
    assert config.providers.dashscope.api_key == "dash-key"
    assert config.translation.provider == "google-free"
    assert config.providers.openai_compatible.translation_model == "custom-translate"
    assert config.tts.provider == "qwen-local"
    assert config.providers.qwen_local.clone_model_size == "1.7B"


def test_write_default_config_renders_provider_structure(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"

    write_default_config(config_path)
    rendered = config_path.read_text(encoding="utf-8")

    assert "[providers.dashscope]" in rendered
    assert "[providers.openai_compatible]" in rendered
    assert "[providers.vllm_omni]" in rendered
    assert "[providers.qwen_local]" in rendered
    assert 'provider = "google-free"' in rendered
    assert 'provider = "qwen-local"' in rendered
    assert f'tts_clone_model = "{DEFAULT_TTS_CLONE_MODEL}"' in rendered
    assert f'clone_model_size = "{DEFAULT_QWEN_LOCAL_MODEL_SIZE}"' in rendered
    qwen_local_section = rendered.split("[providers.qwen_local]", 1)[1].split(
        "[asr]", 1
    )[0]
    assert "max_concurrency" not in qwen_local_section
    assert f'torch_dtype = "{DEFAULT_QWEN_LOCAL_TORCH_DTYPE}"' in rendered
    assert (
        f'attn_implementation = "{DEFAULT_QWEN_LOCAL_ATTN_IMPLEMENTATION}"' in rendered
    )
    assert "\n[tts.preset]\n" in rendered
    assert "\n[tts.clone]\n" in rendered
    assert "\n[tts.vllm_omni]\n" not in rendered
    assert (
        "\nbase_url =" not in rendered.split("[translation]", 1)[1].split("[tts]", 1)[0]
    )


def test_render_config_toml_uses_provider_scoped_tts_sections() -> None:
    rendered = render_config_toml(AppConfig())

    section_order = [
        "[providers.dashscope]",
        "[providers.openai_compatible]",
        "[providers.vllm_omni]",
        "[providers.qwen_local]",
        "[asr]",
        "[translation]",
        "[tts]",
        "[tts.preset]",
        "[tts.preset.voice_map]",
        "[tts.clone]",
        "[compose]",
    ]
    assert [rendered.index(section) for section in section_order] == sorted(
        rendered.index(section) for section in section_order
    )
    assert "[tts]" in rendered
    assert 'mode = "auto"' in rendered
    assert "batch_size = 1" in rendered
    assert "[tts.preset]" in rendered
    assert "[tts.preset.voice_map]" in rendered
    assert "[tts.clone]" in rendered
    assert "[providers.vllm_omni]" in rendered
    assert "voice_mode" not in rendered
    assert "customization_url" not in rendered


def test_provider_api_keys_do_not_affect_tts_or_voice_clone_fingerprints(
    tmp_path: Path,
) -> None:
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    first = AppConfig(
        tts={"provider": "vllm-omni"}, providers={"vllm_omni": {"api_key": "key-1"}}
    )
    second = AppConfig(
        tts={"provider": "vllm-omni"}, providers={"vllm_omni": {"api_key": "key-2"}}
    )

    assert fingerprints.hash_config_subset(
        first, TTS_CONFIG_KEYS
    ) == fingerprints.hash_config_subset(second, TTS_CONFIG_KEYS)
    assert fingerprints.hash_config_subset(
        first, VOICE_CLONE_CONFIG_KEYS
    ) == fingerprints.hash_config_subset(second, VOICE_CLONE_CONFIG_KEYS)


def test_provider_runtime_fields_affect_tts_and_voice_clone_fingerprints(
    tmp_path: Path,
) -> None:
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    first = AppConfig(
        tts={"provider": "vllm-omni"},
        providers={
            "vllm_omni": {
                "language": "Auto",
                "instructions": "",
                "x_vector_only_mode": False,
            }
        },
    )
    second = AppConfig(
        tts={"provider": "vllm-omni"},
        providers={
            "vllm_omni": {
                "language": "zh",
                "instructions": "Warm",
                "x_vector_only_mode": True,
            }
        },
    )

    assert fingerprints.hash_config_subset(
        first, TTS_CONFIG_KEYS
    ) != fingerprints.hash_config_subset(second, TTS_CONFIG_KEYS)
    assert fingerprints.hash_config_subset(
        first, VOICE_CLONE_CONFIG_KEYS
    ) != fingerprints.hash_config_subset(second, VOICE_CLONE_CONFIG_KEYS)


def test_qwen_local_runtime_fields_affect_tts_and_voice_clone_fingerprints(
    tmp_path: Path,
) -> None:
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    first = AppConfig(
        tts={"provider": "qwen-local"},
        providers={
            "qwen_local": {"torch_dtype": "auto", "attn_implementation": "auto"}
        },
    )
    second = AppConfig(
        tts={"provider": "qwen-local"},
        providers={
            "qwen_local": {
                "torch_dtype": "float16",
                "attn_implementation": "flash_attention_2",
            }
        },
    )

    assert fingerprints.hash_config_subset(
        first, TTS_CONFIG_KEYS
    ) != fingerprints.hash_config_subset(second, TTS_CONFIG_KEYS)
    assert fingerprints.hash_config_subset(
        first, VOICE_CLONE_CONFIG_KEYS
    ) != fingerprints.hash_config_subset(second, VOICE_CLONE_CONFIG_KEYS)


def test_tts_runtime_scheduling_fields_do_not_affect_fingerprints(
    tmp_path: Path,
) -> None:
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    first = AppConfig(tts={"provider": "qwen-local"})
    second = AppConfig(
        tts={"provider": "qwen-local", "batch_size": 4, "max_concurrency": 8},
    )

    assert fingerprints.hash_config_subset(
        first, TTS_CONFIG_KEYS
    ) == fingerprints.hash_config_subset(second, TTS_CONFIG_KEYS)
    assert fingerprints.hash_config_subset(
        first, VOICE_CLONE_CONFIG_KEYS
    ) == fingerprints.hash_config_subset(second, VOICE_CLONE_CONFIG_KEYS)


def test_detect_legacy_translation_keys_detects_dashscope_provider() -> None:
    assert detect_legacy_translation_keys(
        {"translation": {"provider": "dashscope"}}
    ) == ["provider=dashscope"]


def test_resolve_workdir_uses_default_and_override() -> None:
    default_config = resolve_config_path()
    assert resolve_workdir() == Path("~/.podtran").expanduser().resolve()
    assert default_config == Path("~/.podtran/config.toml").expanduser().resolve()
    assert (
        resolve_workdir(Path("~/.podtran-tests"))
        == Path("~/.podtran-tests").expanduser().resolve()
    )
    assert (
        resolve_workdir(config_path=Path("~/profiles/config.toml"))
        == Path("~/profiles").expanduser().resolve()
    )
    assert (
        resolve_config_path(workdir_override=Path("~/.podtran-tests"))
        == Path("~/.podtran-tests/config.toml").expanduser().resolve()
    )
    assert (
        resolve_config_path(Path("~/custom.toml"))
        == Path("~/custom.toml").expanduser().resolve()
    )
