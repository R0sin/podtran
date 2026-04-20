from pathlib import Path

import pytest

from podtran.config import (
    AppConfig,
    DEFAULT_TRANSLATION_BASE_URL,
    DEFAULT_TRANSLATION_MODEL,
    DEFAULT_TTS_BASE_URL,
    DEFAULT_TTS_CLONE_MODEL,
    DEFAULT_TTS_PRESET_MODEL,
    TTSConfig,
    build_init_config,
    load_config,
    render_config_toml,
    resolve_config_path,
    resolve_workdir,
    write_default_config,
)


def test_load_config_accepts_new_supported_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "podtran.toml"
    config_path.write_text(
        """
hf_token = ""

[providers.dashscope]
api_key = "dash-key"

[translation]
provider = "dashscope"
base_url = ""
model = "qwen3.5-flash"
timeout_seconds = 90
batch_size = 2
max_concurrency = 3

[tts]
provider = "dashscope"
base_url = ""
mode = "clone"
timeout_seconds = 60
max_concurrency = 2

[tts.preset]
model = "qwen3-tts-flash"
fallback_voices = ["Cherry"]

[tts.preset.voice_map]
SPEAKER_00 = "Cherry"

[tts.clone]
model = "qwen3-tts-vc-2026-01-22"
min_ref_seconds = 8
max_ref_seconds = 18

[asr]
model = "medium"
compute_type = "int8"
device = "cpu"
language = "en"
batch_size = 4
align_model = ""

[compose]
mode = "interleave"
block_pause_threshold = 0.8
max_block_duration = 15.0
gap_en_to_cn_ms = 200
gap_cn_to_en_ms = 400
output_bitrate = "192k"
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.providers.dashscope.api_key == "dash-key"
    assert config.translation.provider == "dashscope"
    assert config.translation.timeout_seconds == 90
    assert config.translation.batch_size == 2
    assert config.translation.max_concurrency == 3
    assert config.tts.provider == "dashscope"
    assert config.tts.mode == "clone"
    assert config.tts.max_concurrency == 2
    assert config.tts.clone.min_ref_seconds == 8
    assert config.tts.clone.max_ref_seconds == 18
    assert config.tts.preset.voice_map == {"SPEAKER_00": "Cherry"}
    assert config.asr.compute_type == "int8"
    assert config.compose.output_bitrate == "192k"


def test_load_config_rejects_legacy_tts_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "podtran.toml"
    config_path.write_text(
        """
[tts]
voice_mode = "clone"
model = "old-model"
clone_min_ref_seconds = 8
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Please run 'podtran init' to rebuild the config"):
        load_config(config_path)


def test_translation_and_tts_resolve_provider_defaults_and_overrides() -> None:
    config = AppConfig()
    assert config.translation.provider == "dashscope"
    assert config.translation.model == DEFAULT_TRANSLATION_MODEL
    assert config.translation.batch_size == 8
    assert config.translation.max_concurrency == 4
    assert config.translation.resolved_base_url() == DEFAULT_TRANSLATION_BASE_URL
    assert config.asr.batch_size == 4
    assert config.tts.resolved_base_url() == DEFAULT_TTS_BASE_URL
    assert config.tts.max_concurrency == 4

    custom = AppConfig(
        translation={"base_url": "https://example.com/v1/"},
        tts={"base_url": "https://tts.example.com/root/"},
    )
    assert custom.translation.resolved_base_url() == "https://example.com/v1"
    assert custom.tts.resolved_base_url() == "https://tts.example.com/root"


def test_tts_config_resolves_clone_and_preset_defaults() -> None:
    clone_config = AppConfig()
    preset_config = AppConfig(tts=TTSConfig(mode="preset"))

    assert clone_config.tts.clone_model() == DEFAULT_TTS_CLONE_MODEL
    assert preset_config.tts.preset_model() == DEFAULT_TTS_PRESET_MODEL


def test_tts_config_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="preset|clone"):
        AppConfig(tts={"mode": "clonee"})


def test_build_init_config_sets_provider_managed_auth() -> None:
    config = build_init_config(
        hf_token="hf-token",
        dashscope_api_key="dash-key",
        translation_model="custom-translate",
        tts_model="custom-tts",
    )

    assert config.hf_token == "hf-token"
    assert config.providers.dashscope.api_key == "dash-key"
    assert config.translation.provider == "dashscope"
    assert config.translation.model == "custom-translate"
    assert config.tts.provider == "dashscope"
    assert config.tts.clone.model == "custom-tts"


def test_write_default_config_renders_provider_structure(tmp_path: Path) -> None:
    config_path = tmp_path / "podtran.toml"

    write_default_config(config_path)
    rendered = config_path.read_text(encoding="utf-8")

    assert "\nworkdir =" not in rendered
    assert "\nffmpeg_path =" not in rendered
    assert "\nffprobe_path =" not in rendered
    assert "max_retries" not in rendered
    assert "[providers.dashscope]" in rendered
    assert 'provider = "dashscope"' in rendered
    assert 'model = "qwen-flash"' in rendered
    assert "batch_size = 8" in rendered
    assert rendered.count("max_concurrency = 4") == 2
    assert 'mode = "clone"' in rendered
    assert f'model = "{DEFAULT_TTS_PRESET_MODEL}"' in rendered
    assert f'model = "{DEFAULT_TTS_CLONE_MODEL}"' in rendered
    assert 'api_key = ""' in rendered
    assert "\n[tts.preset]\n" in rendered
    assert "\n[tts.clone]\n" in rendered
    assert "enrollment_model" not in rendered
    assert "customization_url" not in rendered
    assert "\n[asr]\n" in rendered
    assert 'backend = ' not in rendered


def test_render_config_toml_uses_new_nested_tts_sections() -> None:
    rendered = render_config_toml(AppConfig())

    assert "[tts]" in rendered
    assert "[tts.preset]" in rendered
    assert "[tts.preset.voice_map]" in rendered
    assert "[tts.clone]" in rendered
    assert "voice_mode" not in rendered
    assert "customization_url" not in rendered


def test_resolve_workdir_uses_default_and_override() -> None:
    default_config = resolve_config_path()
    assert resolve_workdir() == Path("~/.podtran").expanduser().resolve()
    assert default_config == Path("~/.podtran/podtran.toml").expanduser().resolve()
    assert resolve_workdir(Path("~/.podtran-tests")) == Path("~/.podtran-tests").expanduser().resolve()
    assert resolve_workdir(config_path=Path("~/profiles/podtran.toml")) == Path("~/profiles").expanduser().resolve()
    assert resolve_config_path(workdir_override=Path("~/.podtran-tests")) == Path("~/.podtran-tests/podtran.toml").expanduser().resolve()
    assert resolve_config_path(Path("~/custom.toml")) == Path("~/custom.toml").expanduser().resolve()
