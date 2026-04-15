from pathlib import Path

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
    resolve_config_path,
    resolve_workdir,
    write_default_config,
)


def test_load_config_accepts_current_supported_fields(tmp_path: Path) -> None:
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

[tts]
provider = "dashscope"
base_url = ""
voice_mode = "clone"
model = "qwen3-tts-vc"
enrollment_model = "qwen-voice-enrollment"
language_type = "Chinese"
timeout_seconds = 60
clone_min_ref_seconds = 8
clone_max_ref_seconds = 18
customization_url = ""
fallback_voices = ["Cherry"]

[tts.voice_map]
SPEAKER_00 = "Cherry"

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
    assert config.tts.provider == "dashscope"
    assert config.tts.voice_mode == "clone"
    assert config.tts.clone_min_ref_seconds == 8
    assert config.tts.clone_max_ref_seconds == 18
    assert config.tts.voice_map == {"SPEAKER_00": "Cherry"}
    assert config.asr.compute_type == "int8"
    assert config.compose.output_bitrate == "192k"


def test_translation_and_tts_resolve_provider_defaults_and_overrides() -> None:
    config = AppConfig()
    assert config.translation.provider == "dashscope"
    assert config.translation.model == DEFAULT_TRANSLATION_MODEL
    assert config.translation.resolved_base_url() == DEFAULT_TRANSLATION_BASE_URL
    assert config.tts.resolved_base_url() == DEFAULT_TTS_BASE_URL
    assert config.tts.resolved_customization_url() == "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/customization"

    custom = AppConfig(
        translation={"base_url": "https://example.com/v1/"},
        tts={"base_url": "https://tts.example.com/root/", "customization_url": "https://tts.example.com/custom/"},
    )
    assert custom.translation.resolved_base_url() == "https://example.com/v1"
    assert custom.tts.resolved_base_url() == "https://tts.example.com/root"
    assert custom.tts.resolved_customization_url() == "https://tts.example.com/custom"


def test_tts_config_resolves_clone_and_preset_defaults() -> None:
    clone_config = AppConfig()
    preset_config = AppConfig(tts=TTSConfig(voice_mode="preset"))

    assert clone_config.tts.resolved_model() == DEFAULT_TTS_CLONE_MODEL
    assert preset_config.tts.resolved_model() == DEFAULT_TTS_PRESET_MODEL
    assert clone_config.tts.resolved_backend() == "dashscope"
    assert AppConfig(tts=TTSConfig(provider="custom", voice_mode="preset")).tts.resolved_backend() == "openai_compatible"


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
    assert config.tts.model == "custom-tts"


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
    assert 'model = "qwen3.5-flash"' in rendered
    assert 'voice_mode = "clone"' in rendered
    assert f'model = "{DEFAULT_TTS_CLONE_MODEL}"' in rendered
    assert 'api_key = ""' in rendered
    assert 'backend = ' not in rendered


def test_resolve_workdir_uses_default_and_override() -> None:
    default_config = resolve_config_path()
    assert resolve_workdir() == Path("~/.podtran").expanduser().resolve()
    assert default_config == Path("~/.podtran/podtran.toml").expanduser().resolve()
    assert resolve_workdir(Path("~/.podtran-tests")) == Path("~/.podtran-tests").expanduser().resolve()
    assert resolve_workdir(config_path=Path("~/profiles/podtran.toml")) == Path("~/profiles").expanduser().resolve()
    assert resolve_config_path(workdir_override=Path("~/.podtran-tests")) == Path("~/.podtran-tests/podtran.toml").expanduser().resolve()
    assert resolve_config_path(Path("~/custom.toml")) == Path("~/custom.toml").expanduser().resolve()
