from datetime import datetime, timezone
from pathlib import Path
import re
import os
import subprocess
import sys

from click.exceptions import Exit as ClickExit
import pytest
from rich.console import Console
from typer.testing import CliRunner

import podtran.cli as cli
from podtran.artifacts import read_model, read_model_list, write_json
from podtran.cache_store import CacheStore
from podtran.config import AppConfig, load_config, write_default_config
from podtran.fingerprints import FingerprintService
from podtran.models import SegmentRecord, StageManifest, TaskManifest, TranscriptSegment
from podtran.stage_executor import StageExecutor
from podtran.tasks import TaskStore

runner = CliRunner()


def _normalize_help_output(output: str) -> str:
    without_ansi = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", output)
    return " ".join(without_ansi.split())


def _segment(
    segment_id: str,
    error: str | None,
    status: str = "pending",
    tts_audio_path: str = "",
    text_zh: str = "",
) -> SegmentRecord:
    return SegmentRecord(
        segment_id=segment_id,
        block_id="b1",
        start=0.0,
        end=1.0,
        text="hello",
        speaker="SPEAKER_00",
        voice="Cherry",
        text_zh=text_zh,
        status=status,
        tts_audio_path=tts_audio_path,
        error=error,
    )


def _create_config(tmp_path: Path) -> tuple[Path, AppConfig]:
    config_path = tmp_path / "config.toml"
    write_default_config(config_path)
    return config_path, load_config(config_path)


def _preview_task(tmp_path: Path) -> tuple[AppConfig, TaskStore, TaskManifest]:
    source = tmp_path / "episode.mp3"
    source.write_bytes(b"source-audio")
    preview = tmp_path / "preview.wav"
    preview.write_bytes(b"preview-audio")
    cfg = AppConfig()
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)
    task = store.create_task_with_processing_audio(
        source,
        cfg,
        entry_command="podtran --preview episode.mp3",
        task_id="20260408-120000-aaaaaa",
        source_audio_sha256=fingerprints.hash_audio(source),
        processing_audio=preview,
        processing_audio_sha256=fingerprints.hash_audio(preview),
        preview=True,
        preview_start_seconds=0.0,
        preview_duration_seconds=300.0,
    )
    return cfg, store, task


def test_print_stage_failure_summary_renders_unique_messages() -> None:
    console = Console(record=True, width=120)

    previous_console = cli.console
    cli.console = console
    try:
        cli._print_stage_failure_summary(
            "translate",
            [
                _segment("seg_1", "RuntimeError: boom"),
                _segment("seg_2", "RuntimeError: boom"),
                _segment("seg_3", "ValueError: bad json"),
            ],
        )
        rendered = console.export_text()
    finally:
        cli.console = previous_console

    assert "translate failures: 3 segment(s)" in rendered
    assert "RuntimeError: boom" in rendered
    assert "ValueError: bad json" in rendered


def test_root_help_documents_run_and_shortcut_entrypoints() -> None:
    result = runner.invoke(cli.app, ["--help"])
    normalized = _normalize_help_output(result.output)

    assert result.exit_code == 0
    assert "stop" in normalized
    assert "run" in normalized
    assert "resume" in normalized
    assert "tasks" in normalized
    assert "status" in normalized
    assert "version" in normalized
    assert "Recommended entrypoint" in normalized
    assert "podtran run AUDIO [--preview]" in normalized
    assert "Shortcut" in normalized
    assert "podtran AUDIO [--preview]" in normalized
    assert "podtran resume [TASK]" in normalized
    assert "podtran stop [TASK]" in normalized


def test_version_command_prints_package_version() -> None:
    result = runner.invoke(cli.app, ["version"])

    assert result.exit_code == 0
    assert result.output.strip() == f"podtran {cli.__version__}"


def test_cache_help_only_lists_clean() -> None:
    result = runner.invoke(cli.app, ["cache", "--help"])

    assert result.exit_code == 0
    assert "clean" in result.output
    assert "only exposed maintenance action is" in result.output
    assert "inspect" not in result.output
    assert "\n| list" not in result.output


def test_run_help_exposes_audio_and_preview_options() -> None:
    result = runner.invoke(cli.app, ["run", "--help"])
    normalized = _normalize_help_output(result.output)

    assert result.exit_code == 0
    assert "AUDIO" in normalized
    assert "--preview" in normalized
    assert "--background" in normalized
    assert "--min_speakers" in normalized
    assert "--max_speakers" in normalized
    assert "Create a new task for AUDIO and run the full pipeline." in normalized


def test_stage_help_documents_task_requirements() -> None:
    stop_result = runner.invoke(cli.app, ["stop", "--help"])
    status_result = runner.invoke(cli.app, ["status", "--help"])
    translate_result = runner.invoke(cli.app, ["translate", "--help"])
    synthesize_result = runner.invoke(cli.app, ["synthesize", "--help"])
    compose_result = runner.invoke(cli.app, ["compose", "--help"])
    cache_clean_result = runner.invoke(cli.app, ["cache", "clean", "--help"])
    normalized_synthesize = _normalize_help_output(synthesize_result.output)
    normalized_compose = _normalize_help_output(compose_result.output)

    assert stop_result.exit_code == 0
    assert "running background task" in stop_result.output
    assert "Defaults to latest" in stop_result.output

    assert status_result.exit_code == 0
    assert "If TASK is omitted" in status_result.output
    assert "latest task" in status_result.output

    assert translate_result.exit_code == 0
    assert "Requires `transcript.json`" in translate_result.output

    assert synthesize_result.exit_code == 0
    assert "Requires `translated.json`" in normalized_synthesize
    assert "all translations completed successfully" in normalized_synthesize

    assert compose_result.exit_code == 0
    assert "Requires `translated.json`" in normalized_compose
    assert "successful TTS output for every required segment" in normalized_compose

    assert cache_clean_result.exit_code == 0
    assert "2026-04-01" in cache_clean_result.output
    assert "2026-04-01T12:30:00+08:00" in cache_clean_result.output


def test_init_prompts_for_qwen_local_defaults(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"

    result = runner.invoke(
        cli.app,
        ["init", "--config", str(config_path), "--workdir", str(tmp_path)],
        input="hf-token\n\n\n\n\n",
    )

    assert result.exit_code == 0
    config = load_config(config_path)
    assert config.hf_token == "hf-token"
    assert config.providers.dashscope.api_key == ""
    assert config.translation.provider == "google-free"
    assert config.providers.openai_compatible.translation_model == "qwen-flash"
    assert config.tts.provider == "qwen-local"
    assert config.tts.mode == "auto"
    assert config.providers.qwen_local.clone_model_size == "0.6B"
    assert (tmp_path / "artifacts").exists()
    assert "speaker-diarization-community-1" in result.output
    assert "hf.co/settings/tokens" in result.output
    assert "DashScope API key" not in result.output


def test_init_can_write_openai_compatible_tts_provider(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"

    result = runner.invoke(
        cli.app,
        ["init", "--config", str(config_path), "--workdir", str(tmp_path)],
        input="hf-token\n\nopenai-compatible\nhttp://localhost:9000/v1\n\ntts-1\n",
    )

    assert result.exit_code == 0
    config = load_config(config_path)
    assert config.tts.provider == "openai-compatible"
    assert config.tts.mode == "auto"
    assert config.tts.effective_mode(config.tts.provider) == "preset"
    assert config.providers.openai_compatible.tts_base_url == "http://localhost:9000/v1"
    assert config.providers.openai_compatible.tts_model == "tts-1"
    assert "DashScope API key" not in result.output


def test_init_can_write_vllm_omni_tts_provider(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"

    result = runner.invoke(
        cli.app,
        ["init", "--config", str(config_path), "--workdir", str(tmp_path)],
        input="hf-token\n\nvllm-omni\nhttp://localhost:8091/v1\nlocal-key\nclone\n\n",
    )

    assert result.exit_code == 0
    config = load_config(config_path)
    assert config.tts.provider == "vllm-omni"
    assert config.providers.vllm_omni.base_url == "http://localhost:8091/v1"
    assert config.tts.mode == "clone"
    assert config.providers.vllm_omni.api_key == "local-key"
    assert "DashScope API key" not in result.output


def test_init_can_write_mimo_tts_provider(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"

    result = runner.invoke(
        cli.app,
        ["init", "--config", str(config_path), "--workdir", str(tmp_path)],
        input="hf-token\n\nmimo\n\nmimo-key\nclone\n\n",
    )

    assert result.exit_code == 0
    config = load_config(config_path)
    assert config.tts.provider == "mimo"
    assert config.tts.mode == "clone"
    assert config.providers.mimo.api_key == "mimo-key"
    assert config.providers.mimo.base_url == "https://api.xiaomimimo.com/v1"
    assert config.providers.mimo.clone_model == "mimo-v2.5-tts-voiceclone"
    assert "DashScope API key" not in result.output
    assert "OpenAI-compatible TTS base URL" not in result.output
    assert "vLLM-Omni TTS base URL" not in result.output


def test_init_reprompts_required_values(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"

    result = runner.invoke(
        cli.app,
        ["init", "--config", str(config_path), "--workdir", str(tmp_path)],
        input="\nhf-token\n\n\ninvalid\n\n1.7B\n",
    )

    assert result.exit_code == 0
    config = load_config(config_path)
    assert config.hf_token == "hf-token"
    assert config.providers.dashscope.api_key == ""
    assert config.translation.provider == "google-free"
    assert config.providers.openai_compatible.translation_model == "qwen-flash"
    assert config.tts.provider == "qwen-local"
    assert config.providers.qwen_local.clone_model_size == "1.7B"
    assert result.output.count("Hugging Face token") >= 2
    assert "TTS mode must be one of" in result.output


def test_init_uses_existing_values_as_defaults_and_preserves_other_fields(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"
    config = AppConfig()
    config.hf_token = "hf-existing"
    config.providers.dashscope.api_key = "dash-existing"
    config.providers.openai_compatible.translation_model = "existing-translate"
    config.providers.dashscope.tts_clone_model = "existing-tts"
    config.providers.dashscope.tts_preset_model = "existing-preset"
    config.tts.mode = "preset"
    config.tts.preset.voice_map = {"SPEAKER_00": "Cherry"}
    config.asr.batch_size = 7
    config.compose.output_bitrate = "256k"
    config_path.write_text(cli.render_config_toml(config), encoding="utf-8")

    result = runner.invoke(
        cli.app,
        ["init", "--config", str(config_path), "--workdir", str(tmp_path)],
        input="\n\n\n\n\n",
    )

    assert result.exit_code == 0
    updated = load_config(config_path)
    assert updated.hf_token == "hf-existing"
    assert updated.providers.dashscope.api_key == "dash-existing"
    assert updated.translation.provider == "google-free"
    assert updated.providers.openai_compatible.translation_model == "existing-translate"
    assert updated.providers.dashscope.tts_clone_model == "existing-tts"
    assert updated.providers.dashscope.tts_preset_model == "existing-preset"
    assert updated.tts.mode == "preset"
    assert updated.tts.preset.voice_map == {"SPEAKER_00": "Cherry"}
    assert updated.asr.batch_size == 7
    assert updated.compose.output_bitrate == "256k"


def test_init_rebuilds_legacy_tts_config_with_backup_and_preserved_auth(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
hf_token = "hf-legacy"

[providers.dashscope]
api_key = "dash-legacy"

[translation]
provider = "dashscope"
model = "legacy-translate"

[tts]
provider = "dashscope"
voice_mode = "clone"
model = "legacy-tts"
fallback_voices = ["Cherry"]
""".strip(),
        encoding="utf-8",
    )

    result = runner.invoke(
        cli.app,
        ["init", "--config", str(config_path), "--workdir", str(tmp_path)],
        input="\n\n\n\n\n",
    )

    assert result.exit_code == 0
    rebuilt = load_config(config_path)
    backup_path = tmp_path / "config.toml.bak"
    assert backup_path.exists()
    assert 'voice_mode = "clone"' in backup_path.read_text(encoding="utf-8")
    assert rebuilt.hf_token == "hf-legacy"
    assert rebuilt.providers.dashscope.api_key == "dash-legacy"
    assert rebuilt.translation.provider == "google-free"
    assert rebuilt.providers.openai_compatible.translation_model == "qwen-flash"
    assert rebuilt.providers.dashscope.tts_clone_model == "qwen3-tts-vc-2026-01-22"
    assert "Legacy TTS config detected." in result.output
    assert (
        "Only hf_token and provider API keys were preserved."
        in _normalize_help_output(result.output)
    )


def test_rebuild_legacy_config_preserves_provider_api_keys() -> None:
    rebuilt = cli._rebuild_config_with_preserved_auth(
        {
            "hf_token": "hf-legacy",
            "providers": {
                "dashscope": {"api_key": "dash-key"},
                "openai_compatible": {
                    "translation_api_key": "translate-key",
                    "tts_api_key": "tts-key",
                },
                "vllm_omni": {"api_key": "new-vllm-key"},
                "mimo": {"api_key": "mimo-key"},
            },
            "tts": {
                "vllm_omni": {"api_key": "legacy-vllm-key"},
            },
        }
    )

    assert rebuilt.hf_token == "hf-legacy"
    assert rebuilt.providers.dashscope.api_key == "dash-key"
    assert rebuilt.providers.openai_compatible.translation_api_key == "translate-key"
    assert rebuilt.providers.openai_compatible.tts_api_key == "tts-key"
    assert rebuilt.providers.vllm_omni.api_key == "new-vllm-key"
    assert rebuilt.providers.mimo.api_key == "mimo-key"


def test_rebuild_legacy_config_preserves_legacy_vllm_omni_api_key() -> None:
    rebuilt = cli._rebuild_config_with_preserved_auth(
        {
            "tts": {
                "vllm_omni": {"api_key": "legacy-vllm-key"},
            },
        }
    )

    assert rebuilt.providers.vllm_omni.api_key == "legacy-vllm-key"


def test_init_skips_dashscope_key_when_neither_translation_nor_tts_use_dashscope(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"

    result = runner.invoke(
        cli.app,
        ["init", "--config", str(config_path), "--workdir", str(tmp_path)],
        input="hf-token\nopenai-compatible\nhttp://localhost:9000/v1\n\ntranslate-model\nopenai-compatible\nhttp://localhost:9001/v1\n\ntts-1\n",
    )

    assert result.exit_code == 0
    config = load_config(config_path)
    assert config.translation.provider == "openai-compatible"
    assert (
        config.providers.openai_compatible.translation_base_url
        == "http://localhost:9000/v1"
    )
    assert config.providers.openai_compatible.translation_model == "translate-model"
    assert config.tts.provider == "openai-compatible"
    assert config.providers.openai_compatible.tts_base_url == "http://localhost:9001/v1"
    assert config.providers.openai_compatible.tts_model == "tts-1"
    assert config.providers.dashscope.api_key == ""
    assert "DashScope API key" not in result.output


def test_init_writes_openai_compatible_translation_provider(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"

    result = runner.invoke(
        cli.app,
        ["init", "--config", str(config_path), "--workdir", str(tmp_path)],
        input="hf-token\nopenai-compatible\nhttp://localhost:9000/v1\ntranslate-key\ntranslate-model\nopenai-compatible\nhttp://localhost:9001/v1\n\ntts-1\n",
    )

    assert result.exit_code == 0
    config = load_config(config_path)
    assert config.translation.provider == "openai-compatible"
    assert (
        config.providers.openai_compatible.translation_base_url
        == "http://localhost:9000/v1"
    )
    assert config.providers.openai_compatible.translation_api_key == "translate-key"
    assert config.providers.openai_compatible.translation_model == "translate-model"
    assert config.tts.provider == "openai-compatible"
    assert result.output.index("Translation provider") < result.output.index(
        "OpenAI-compatible translation base URL"
    )
    assert result.output.index(
        "OpenAI-compatible translation API key"
    ) < result.output.index("Translation model")


def test_init_prompts_dashscope_key_immediately_for_dashscope_tts(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"

    result = runner.invoke(
        cli.app,
        ["init", "--config", str(config_path), "--workdir", str(tmp_path)],
        input="hf-token\n\ndashscope\ndash-key\nclone\ncustom-tts\n",
    )

    assert result.exit_code == 0
    config = load_config(config_path)
    assert config.translation.provider == "google-free"
    assert config.tts.provider == "dashscope"
    assert config.providers.dashscope.api_key == "dash-key"
    assert config.providers.dashscope.tts_clone_model == "custom-tts"
    assert result.output.index("TTS provider") < result.output.index(
        "DashScope API key"
    )
    assert result.output.index("DashScope API key") < result.output.index("TTS mode")


def test_backup_legacy_config_preserves_existing_backup(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text('mode = "new"', encoding="utf-8")
    backup_path = tmp_path / "config.toml.bak"
    backup_path.write_text('mode = "old"', encoding="utf-8")

    preserved = cli._backup_legacy_config(config_path)

    assert preserved == backup_path
    assert backup_path.read_text(encoding="utf-8") == 'mode = "old"'


@pytest.mark.parametrize(
    "argv",
    [
        ["podtran", "{audio}", "--preview"],
        ["podtran", "--preview", "{audio}"],
    ],
)
def test_root_preview_creates_preview_task_and_executes_pipeline(
    tmp_path: Path, monkeypatch, capsys, argv: list[str]
) -> None:
    config_path, _ = _create_config(tmp_path)
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    captured: dict[str, TaskManifest] = {}
    clip_args: dict[str, object] = {}

    def fake_execute_pipeline(
        task_manifest,
        cfg,
        task_store,
        cache_store,
        fingerprints,
        min_speakers=2,
        max_speakers=5,
    ):
        captured["task"] = task_manifest
        captured["min_speakers"] = min_speakers
        captured["max_speakers"] = max_speakers
        return []

    def fake_extract_audio_chunk(
        ffmpeg_path: str,
        source: Path,
        output: Path,
        start: float | None,
        end: float | None,
    ) -> Path:
        clip_args.update(
            {
                "ffmpeg_path": ffmpeg_path,
                "source": source,
                "output": output,
                "start": start,
                "end": end,
            }
        )
        output.write_bytes(b"preview")
        return output

    argv = [part.format(audio=audio) for part in argv]

    monkeypatch.setattr(cli, "_execute_pipeline", fake_execute_pipeline)
    monkeypatch.setattr(cli, "extract_audio_chunk", fake_extract_audio_chunk)
    monkeypatch.setattr(cli, "ensure_command", lambda command: None)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [argv[0], "--config", str(config_path), "--workdir", str(tmp_path), *argv[1:]],
    )

    cli.main()
    stdout = capsys.readouterr().out

    assert "Created task:" in stdout
    task = captured["task"]
    assert task.preview is True
    assert task.preview_start_seconds == 0.0
    assert task.preview_duration_seconds == 300.0
    assert Path(task.processing_audio_path).name == "preview.wav"
    assert Path(task.processing_audio_path).read_bytes() == b"preview"
    assert task.processing_audio_path != task.source_audio_path
    assert clip_args == {
        "ffmpeg_path": "ffmpeg",
        "source": audio.resolve(),
        "output": Path(task.processing_audio_path),
        "start": 0.0,
        "end": 300.0,
    }


def test_root_audio_creates_task_and_executes_pipeline(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config_path, _ = _create_config(tmp_path)
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    captured: dict[str, str] = {}

    def fake_execute_pipeline(
        task_manifest,
        cfg,
        task_store,
        cache_store,
        fingerprints,
        min_speakers=2,
        max_speakers=5,
    ):
        captured["task_id"] = task_manifest.task_id
        captured["entry_command"] = task_manifest.entry_command
        captured["min_speakers"] = min_speakers
        captured["max_speakers"] = max_speakers
        return []

    monkeypatch.setattr(cli, "_execute_pipeline", fake_execute_pipeline)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "podtran",
            "--config",
            str(config_path),
            "--workdir",
            str(tmp_path),
            "--min_speakers",
            "3",
            "--max_speakers",
            "6",
            str(audio),
        ],
    )

    cli.main()
    stdout = capsys.readouterr().out

    assert "Created task:" in stdout
    assert captured["task_id"]
    assert captured["min_speakers"] == 3
    assert captured["max_speakers"] == 6
    assert "--min_speakers 3" in captured["entry_command"]
    assert "--max_speakers 6" in captured["entry_command"]


def test_run_background_creates_task_and_starts_resume_process(
    tmp_path: Path, monkeypatch
) -> None:
    config_path, _ = _create_config(tmp_path)
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    captured: dict[str, object] = {}

    class FakeProcess:
        pid = 4321

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        captured["stdout_name"] = kwargs["stdout"].name
        return FakeProcess()

    def fail_execute_pipeline(*args, **kwargs):
        raise AssertionError("background run must not execute pipeline inline")

    monkeypatch.setattr(cli.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(cli, "_execute_pipeline", fail_execute_pipeline)

    result = runner.invoke(
        cli.app,
        [
            "run",
            str(audio),
            "--background",
            "--config",
            str(config_path),
            "--workdir",
            str(tmp_path),
            "--min_speakers",
            "3",
            "--max_speakers",
            "6",
        ],
    )

    assert result.exit_code == 0
    store = TaskStore(
        tmp_path, FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    )
    task = store.load_latest_task()
    task_dir = tmp_path / "artifacts" / "tasks" / task.task_id
    assert (
        task.entry_command
        == f"podtran --background --min_speakers 3 --max_speakers 6 {audio}"
    )
    assert (task_dir / "run.pid").read_text(encoding="utf-8") == "4321\n"
    assert Path(captured["stdout_name"]) == task_dir / "run.log"
    run_token = (task_dir / "run.token").read_text(encoding="utf-8").strip()
    assert len(run_token) >= 32
    assert captured["kwargs"]["env"]["PODTRAN_RUN_TOKEN"] == run_token
    assert captured["kwargs"]["stderr"] == cli.subprocess.STDOUT
    assert captured["command"] == [
        cli.sys.executable,
        "-m",
        "podtran",
        "resume",
        task.task_id,
        "--config",
        str(config_path),
        "--workdir",
        str(tmp_path.resolve()),
        "--min_speakers",
        "3",
        "--max_speakers",
        "6",
    ]
    assert "Started background task:" in result.output
    assert "pid=4321" in result.output
    assert "run.log" in result.output
    assert f"podtran status {task.task_id}" in result.output


def test_stop_running_background_task_marks_task_and_stage_interrupted(
    tmp_path: Path,
) -> None:
    config_path, cfg = _create_config(tmp_path)
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)
    task = store.create_task(audio, cfg, entry_command=f"podtran {audio}")
    paths = store.paths_for(task)

    fake_package = tmp_path / "fake-package" / "podtran"
    fake_package.mkdir(parents=True)
    (fake_package / "__init__.py").write_text("", encoding="utf-8")
    (fake_package / "__main__.py").write_text(
        "import time\ntime.sleep(60)\n", encoding="utf-8"
    )
    run_token = "test-background-run-token"
    process_env = os.environ.copy()
    process_env["PODTRAN_RUN_TOKEN"] = run_token
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "podtran",
            "resume",
            task.task_id,
            "--config",
            str(config_path),
            "--workdir",
            str(tmp_path.resolve()),
            "--min_speakers",
            "2",
            "--max_speakers",
            "5",
        ],
        cwd=fake_package.parent,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=process_env,
        creationflags=(
            subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        ),
        start_new_session=sys.platform != "win32",
    )
    try:
        task.status = "running"
        task.current_stage = "translate"
        store.save_task(task)
        write_json(
            paths.manifest_path("translate"),
            StageManifest(stage="translate", status="running", pid=process.pid),
        )
        (paths.task_dir / "run.pid").write_text(f"{process.pid}\n", encoding="utf-8")
        (paths.task_dir / "run.token").write_text(f"{run_token}\n", encoding="utf-8")

        result = runner.invoke(
            cli.app,
            [
                "stop",
                task.task_id,
                "--config",
                str(config_path),
                "--workdir",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 0
        assert f"Stopped task: {task.task_id}" in result.output
        assert process.wait(timeout=5) is not None
        assert store.load_task(task.task_id).status == "interrupted"
        stage = read_model(paths.manifest_path("translate"), StageManifest)
        assert stage.status == "interrupted"
        assert stage.error == "Interrupted by user"
        assert not (paths.task_dir / "run.pid").exists()
        assert not (paths.task_dir / "run.token").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_stop_stale_background_task_marks_latest_task_interrupted(
    tmp_path: Path,
) -> None:
    config_path, cfg = _create_config(tmp_path)
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)
    task = store.create_task(audio, cfg, entry_command=f"podtran {audio}")
    paths = store.paths_for(task)
    stale_pid = 2_147_483_647
    task.status = "running"
    task.current_stage = "translate"
    store.save_task(task)
    write_json(
        paths.manifest_path("translate"),
        StageManifest(stage="translate", status="running", pid=stale_pid),
    )
    (paths.task_dir / "run.pid").write_text(f"{stale_pid}\n", encoding="utf-8")

    result = runner.invoke(
        cli.app,
        [
            "stop",
            "--config",
            str(config_path),
            "--workdir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert f"Stopped task: {task.task_id}" in result.output
    assert store.load_task(task.task_id).status == "interrupted"
    stage = read_model(paths.manifest_path("translate"), StageManifest)
    assert stage.status == "interrupted"
    assert not (paths.task_dir / "run.pid").exists()


def test_stop_refuses_pid_owned_by_another_process(tmp_path: Path) -> None:
    config_path, cfg = _create_config(tmp_path)
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)
    task = store.create_task(audio, cfg, entry_command=f"podtran {audio}")
    paths = store.paths_for(task)

    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        task.status = "running"
        task.current_stage = "translate"
        store.save_task(task)
        write_json(
            paths.manifest_path("translate"),
            StageManifest(stage="translate", status="running", pid=process.pid),
        )
        (paths.task_dir / "run.pid").write_text(f"{process.pid}\n", encoding="utf-8")

        result = runner.invoke(
            cli.app,
            [
                "stop",
                task.task_id,
                "--config",
                str(config_path),
                "--workdir",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 1
        assert f"Refusing to stop PID {process.pid}" in result.output
        assert process.poll() is None
        assert store.load_task(task.task_id).status == "running"
        assert (paths.task_dir / "run.pid").exists()
    finally:
        process.kill()
        process.wait(timeout=5)


def test_stop_refuses_mismatched_running_stage_pid(tmp_path: Path) -> None:
    config_path, cfg = _create_config(tmp_path)
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)
    task = store.create_task(audio, cfg, entry_command=f"podtran {audio}")
    paths = store.paths_for(task)
    task.status = "running"
    task.current_stage = "translate"
    store.save_task(task)
    write_json(
        paths.manifest_path("translate"),
        StageManifest(stage="translate", status="running", pid=1234),
    )
    (paths.task_dir / "run.pid").write_text("5678\n", encoding="utf-8")

    result = runner.invoke(
        cli.app,
        [
            "stop",
            task.task_id,
            "--config",
            str(config_path),
            "--workdir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "run.pid contains 5678" in result.output
    assert "stage belongs to PID 1234" in result.output
    assert store.load_task(task.task_id).status == "running"
    stage = read_model(paths.manifest_path("translate"), StageManifest)
    assert stage.status == "running"
    assert (paths.task_dir / "run.pid").exists()


def test_root_background_shortcut_creates_task_and_starts_resume_process(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config_path, _ = _create_config(tmp_path)
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    captured: dict[str, object] = {}

    class FakeProcess:
        pid = 9876

    def fake_popen(command, **kwargs):
        captured["command"] = command
        return FakeProcess()

    monkeypatch.setattr(cli.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        cli,
        "_execute_pipeline",
        lambda *args, **kwargs: pytest.fail("background run executed inline"),
    )
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "podtran",
            "--config",
            str(config_path),
            "--workdir",
            str(tmp_path),
            str(audio),
            "--background",
        ],
    )

    cli.main()
    stdout = capsys.readouterr().out

    store = TaskStore(
        tmp_path, FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    )
    task = store.load_latest_task()
    assert task.entry_command == f"podtran --background {audio}"
    assert (tmp_path / "artifacts" / "tasks" / task.task_id / "run.pid").read_text(
        encoding="utf-8"
    ) == "9876\n"
    assert captured["command"][3:5] == ["resume", task.task_id]
    assert "Started background task:" in stdout


def test_background_preview_creates_preview_before_starting_resume_process(
    tmp_path: Path, monkeypatch
) -> None:
    config_path, _ = _create_config(tmp_path)
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    captured: dict[str, object] = {}

    class FakeProcess:
        pid = 2468

    def fake_extract_audio_chunk(
        ffmpeg_path: str,
        source: Path,
        output: Path,
        start: float | None,
        end: float | None,
    ) -> Path:
        captured["preview_exists_before_popen"] = False
        output.write_bytes(b"preview")
        return output

    def fake_popen(command, **kwargs):
        task_id = command[4]
        preview_path = tmp_path / "artifacts" / "tasks" / task_id / "preview.wav"
        captured["preview_exists_before_popen"] = preview_path.exists()
        return FakeProcess()

    monkeypatch.setattr(cli, "extract_audio_chunk", fake_extract_audio_chunk)
    monkeypatch.setattr(cli, "ensure_command", lambda command: None)
    monkeypatch.setattr(cli.subprocess, "Popen", fake_popen)

    result = runner.invoke(
        cli.app,
        [
            "run",
            str(audio),
            "--preview",
            "--background",
            "--config",
            str(config_path),
            "--workdir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    store = TaskStore(
        tmp_path, FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    )
    task = store.load_latest_task()
    assert task.preview is True
    assert Path(task.processing_audio_path).read_bytes() == b"preview"
    assert captured["preview_exists_before_popen"] is True


def test_status_reports_factual_stage_statuses_and_preview_mode(tmp_path: Path) -> None:
    config_path, cfg = _create_config(tmp_path)
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    preview_audio = tmp_path / "preview.wav"
    preview_audio.write_bytes(b"preview")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)
    task_manifest = store.create_task_with_processing_audio(
        audio,
        cfg,
        entry_command=f"podtran --preview {audio}",
        task_id="20260408-120000-aaaaaa",
        source_audio_sha256=fingerprints.hash_audio(audio),
        processing_audio=preview_audio,
        processing_audio_sha256=fingerprints.hash_audio(preview_audio),
        preview=True,
        preview_start_seconds=0.0,
        preview_duration_seconds=300.0,
    )
    task_manifest.status = "failed"
    task_manifest.current_stage = "translate"
    store.save_task(task_manifest)
    paths = store.paths_for(task_manifest)
    paths.ensure()
    write_json(paths.transcript_json, [])
    write_json(
        paths.manifest_path("transcribe"),
        StageManifest(
            stage="transcribe",
            status="completed",
            output_refs={"transcript_json": "transcript.json"},
        ),
    )
    write_json(
        paths.manifest_path("translate"),
        StageManifest(
            stage="translate",
            status="failed",
            error="translation exploded",
            output_refs={
                "segments_json": "segments.json",
                "translated_json": "translated.json",
            },
        ),
    )

    status_result = runner.invoke(
        cli.app,
        [
            "status",
            task_manifest.task_id,
            "--config",
            str(config_path),
            "--workdir",
            str(tmp_path),
        ],
    )
    tasks_result = runner.invoke(
        cli.app, ["tasks", "--config", str(config_path), "--workdir", str(tmp_path)]
    )

    assert status_result.exit_code == 0
    assert f"task_id={task_manifest.task_id}" in status_result.output
    assert "mode=preview" in status_result.output
    assert "clip=0-300s" in status_result.output
    assert "podtran status" in status_result.output
    assert "translation exploded" in status_result.output
    assert "cache hit" not in status_result.output
    assert "up-to-date" not in status_result.output
    assert tasks_result.exit_code == 0
    assert "preview" in tasks_result.output


def test_translate_requires_transcript_json(tmp_path: Path) -> None:
    config_path, cfg = _create_config(tmp_path)
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)
    task_manifest = store.create_task(audio, cfg, f"podtran {audio}")

    result = runner.invoke(
        cli.app,
        [
            "translate",
            task_manifest.task_id,
            "--config",
            str(config_path),
            "--workdir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "Cannot run translate: missing transcript.json." in result.output


def test_translate_shared_cache_ignores_voice_map_and_syncs_current_voice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import podtran.translate as translate_module

    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)
    cache_store = CacheStore(tmp_path / "artifacts" / "cache")

    first_cfg = AppConfig()
    first_cfg.tts.mode = "preset"
    first_cfg.tts.preset.voice_map = {"SPEAKER_00": "Cherry"}

    second_cfg = first_cfg.model_copy(deep=True)
    second_cfg.tts.preset.voice_map = {"SPEAKER_00": "Serena"}

    first_task = store.create_task(audio, first_cfg, f"podtran {audio}")
    second_task = store.create_task(audio, second_cfg, f"podtran {audio}")
    first_paths = store.paths_for(first_task)
    second_paths = store.paths_for(second_task)
    transcript = [
        TranscriptSegment(
            segment_id="ts_1",
            start=0.0,
            end=1.0,
            text="hello world",
            speaker="SPEAKER_00",
        )
    ]
    write_json(first_paths.transcript_json, transcript)
    write_json(second_paths.transcript_json, transcript)

    class FakeTranslator:
        calls = 0

        def __init__(self, config: AppConfig) -> None:
            self.config = config

        def translate_segments(
            self, input_path: Path, output_path: Path, progress_callback=None
        ) -> list[SegmentRecord]:
            FakeTranslator.calls += 1
            segments = read_model_list(input_path, SegmentRecord)
            translated = [
                segment.model_copy(update={"text_zh": f"zh-{segment.segment_id}"})
                for segment in segments
            ]
            write_json(output_path, translated)
            return translated

    monkeypatch.setattr(translate_module, "Translator", FakeTranslator)

    first_executor = StageExecutor(store, first_task, first_paths)
    second_executor = StageExecutor(store, second_task, second_paths)

    first_result = cli._ensure_translate(
        first_task, first_cfg, first_paths, first_executor, cache_store, fingerprints
    )
    second_result = cli._ensure_translate(
        second_task,
        second_cfg,
        second_paths,
        second_executor,
        cache_store,
        fingerprints,
    )
    restored = read_model_list(second_paths.translated_json, SegmentRecord)

    assert first_result.action == "run"
    assert second_result.action == "cache hit"
    assert FakeTranslator.calls == 1
    assert restored[0].text_zh == "zh-seg_00000"
    assert restored[0].voice == "Serena"


def test_preview_transcribe_uses_processing_audio(tmp_path: Path, monkeypatch) -> None:
    cfg, store, task_manifest = _preview_task(tmp_path)
    paths = store.paths_for(task_manifest)
    executor = StageExecutor(store, task_manifest, paths)
    cache_store = CacheStore(paths.cache_dir)
    fingerprints = FingerprintService(paths.cache_indexes_dir)
    captured: dict[str, object] = {}

    def fake_run_transcription_with_progress(
        audio: Path,
        config: AppConfig,
        min_speakers: int,
        max_speakers: int,
        progress_callback=None,
    ):
        captured["audio"] = audio
        captured["min_speakers"] = min_speakers
        captured["max_speakers"] = max_speakers
        return []

    monkeypatch.setattr(cli, "ensure_command", lambda command: None)
    monkeypatch.setattr(cli, "ensure_hf_token", lambda token: None)
    monkeypatch.setattr(
        cli, "_run_transcription_with_progress", fake_run_transcription_with_progress
    )

    cli._ensure_transcribe(
        task_manifest,
        cfg,
        paths,
        executor,
        cache_store,
        fingerprints,
        min_speakers=3,
        max_speakers=6,
    )

    manifest = read_model(paths.manifest_path("transcribe"), StageManifest)
    assert captured["audio"] == Path(task_manifest.processing_audio_path)
    assert captured["min_speakers"] == 3
    assert captured["max_speakers"] == 6
    assert manifest.input_fingerprints["audio"] == task_manifest.processing_audio_sha256


def test_preview_synthesize_uses_processing_audio(tmp_path: Path, monkeypatch) -> None:
    import podtran.tts as tts

    cfg, store, task_manifest = _preview_task(tmp_path)
    paths = store.paths_for(task_manifest)
    executor = StageExecutor(store, task_manifest, paths)
    cache_store = CacheStore(paths.cache_dir)
    fingerprints = FingerprintService(paths.cache_indexes_dir)
    write_json(paths.translated_json, [_segment("seg_1", None, text_zh="你好")])
    captured: dict[str, object] = {}

    def fake_synthesize_segments(*args, **kwargs):
        captured["source_audio"] = kwargs["source_audio"]
        captured["source_audio_fingerprint"] = kwargs["source_audio_fingerprint"]
        output = paths.tts_dir / "seg_1.wav"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"tts")
        completed = _segment(
            "seg_1",
            None,
            status="completed",
            tts_audio_path=str(output.resolve()),
            text_zh="你好",
        )
        return [completed]

    monkeypatch.setattr(cli, "ensure_command", lambda command: None)
    monkeypatch.setattr(tts, "synthesize_segments", fake_synthesize_segments)

    cli._ensure_synthesize(
        task_manifest, cfg, paths, executor, cache_store, fingerprints
    )

    assert captured["source_audio"] == Path(task_manifest.processing_audio_path)
    assert captured["source_audio_fingerprint"] == task_manifest.processing_audio_sha256


def test_synthesize_is_up_to_date_after_successful_run(
    tmp_path: Path, monkeypatch
) -> None:
    import podtran.tts as tts

    cfg, store, task_manifest = _preview_task(tmp_path)
    cfg.tts.mode = "preset"
    paths = store.paths_for(task_manifest)
    executor = StageExecutor(store, task_manifest, paths)
    cache_store = CacheStore(paths.cache_dir)
    fingerprints = FingerprintService(paths.cache_indexes_dir)
    write_json(paths.translated_json, [_segment("seg_1", None, text_zh="你好")])

    def fake_synthesize_segments(*args, **kwargs):
        output = paths.tts_dir / "seg_1.wav"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"tts")
        return [
            _segment(
                "seg_1",
                None,
                status="completed",
                tts_audio_path=str(output.resolve()),
                text_zh="你好",
            )
        ]

    monkeypatch.setattr(cli, "ensure_command", lambda command: None)
    monkeypatch.setattr(tts, "synthesize_segments", fake_synthesize_segments)

    first = cli._ensure_synthesize(
        task_manifest, cfg, paths, executor, cache_store, fingerprints
    )
    second = cli._ensure_synthesize(
        task_manifest, cfg, paths, executor, cache_store, fingerprints
    )

    assert first.action == "run"
    assert second.action == "up-to-date"


def test_preview_compose_uses_processing_audio_and_preview_output_name(
    tmp_path: Path, monkeypatch
) -> None:
    cfg, store, task_manifest = _preview_task(tmp_path)
    paths = store.paths_for(task_manifest)
    executor = StageExecutor(store, task_manifest, paths)
    fingerprints = FingerprintService(paths.cache_indexes_dir)
    synthesized_audio = paths.tts_dir / "seg_1.wav"
    synthesized_audio.parent.mkdir(parents=True, exist_ok=True)
    synthesized_audio.write_bytes(b"tts")
    write_json(
        paths.translated_json,
        [
            _segment(
                "seg_1",
                None,
                status="completed",
                tts_audio_path=str(synthesized_audio),
                text_zh="你好",
            )
        ],
    )
    captured: dict[str, Path] = {}

    def fake_compose_output(
        source_audio: Path,
        segments,
        config,
        temp_dir: Path,
        output_path: Path,
        mode: str | None = None,
        progress_callback=None,
    ) -> Path:
        captured["source_audio"] = source_audio
        captured["output_path"] = output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"final")
        return output_path

    monkeypatch.setattr(cli, "ensure_command", lambda command: None)
    monkeypatch.setattr(cli, "compose_output", fake_compose_output)

    cli._ensure_compose(task_manifest, cfg, paths, executor, fingerprints)

    manifest = read_model(paths.manifest_path("compose"), StageManifest)
    assert captured["source_audio"] == Path(task_manifest.processing_audio_path)
    assert captured["output_path"].name == "episode.preview.interleave.mp3"
    assert (
        manifest.input_fingerprints["source_audio"]
        == task_manifest.processing_audio_sha256
    )


def test_execute_pipeline_prints_absolute_output_path_for_preview_task(
    tmp_path: Path, monkeypatch
) -> None:
    cfg, store, task_manifest = _preview_task(tmp_path)
    cache_store = CacheStore(tmp_path / "artifacts" / "cache")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    recorded_console = Console(record=True, width=160)
    expected_output_path = (
        store.paths_for(task_manifest).final_dir / "episode.preview.interleave.mp3"
    )
    previous_console = cli.console

    monkeypatch.setattr(
        cli,
        "_ensure_transcribe",
        lambda *args, **kwargs: cli.StageDecision("transcribe", "up-to-date", "test"),
    )
    monkeypatch.setattr(
        cli,
        "_ensure_translate",
        lambda *args, **kwargs: cli.StageDecision("translate", "up-to-date", "test"),
    )
    monkeypatch.setattr(
        cli,
        "_ensure_synthesize",
        lambda *args, **kwargs: cli.StageDecision("synthesize", "up-to-date", "test"),
    )

    def fake_ensure_compose(
        task_manifest, cfg, paths, executor, fingerprints, reporter=None
    ):
        output_path = cli._compose_output_path(task_manifest, cfg, paths)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"final")
        return cli.StageDecision("compose", "run", "test")

    monkeypatch.setattr(cli, "_ensure_compose", fake_ensure_compose)
    cli.console = recorded_console
    try:
        cli._execute_pipeline(task_manifest, cfg, store, cache_store, fingerprints)
        rendered = recorded_console.export_text()
    finally:
        cli.console = previous_console

    assert "Task complete:" in rendered
    assert f"Output file: {expected_output_path.resolve()}".replace(
        "\n", ""
    ) in rendered.replace("\n", "")


def test_cache_clean_accepts_date_only_before(tmp_path: Path) -> None:
    config_path, cfg = _create_config(tmp_path)
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)
    task_manifest = store.create_task(audio, cfg, f"podtran {audio}")
    paths = store.paths_for(task_manifest)
    cache_store = CacheStore(paths.cache_dir)
    cache_payload = tmp_path / "translated.json"
    cache_payload.write_text('{"ok": true}', encoding="utf-8")
    manifest = StageManifest(
        stage="translate",
        status="completed",
        stage_version=1,
        cache_key="abc123",
        input_fingerprints={"segments_json": "hash-1"},
        config_fingerprint="cfg-1",
        config_keys=["translation.model"],
        finished_at="2026-03-01T00:00:00+00:00",
    )
    cache_store.publish(
        "translate", "abc123", {"translated_json": cache_payload}, manifest
    )

    result = runner.invoke(
        cli.app,
        [
            "cache",
            "clean",
            "--before",
            "2026-04-01",
            "--config",
            str(config_path),
            "--workdir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Removed cache entries:" in result.output
    assert cache_store.lookup("translate", "abc123") is None


def test_parse_before_normalizes_to_utc() -> None:
    assert cli._parse_before("2026-04-01") == datetime(2026, 4, 1, tzinfo=timezone.utc)
    assert cli._parse_before("2026-04-01T08:00:00+08:00") == datetime(
        2026, 4, 1, tzinfo=timezone.utc
    )


def test_translate_config_fingerprint_ignores_batch_size(tmp_path: Path) -> None:
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    first = AppConfig()
    second = AppConfig()
    second.translation.batch_size = first.translation.batch_size + 1

    assert cli._translation_config_fingerprint(
        fingerprints, first
    ) == cli._translation_config_fingerprint(fingerprints, second)


def test_translate_config_fingerprint_ignores_max_concurrency(tmp_path: Path) -> None:
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    first = AppConfig()
    second = AppConfig()
    second.translation.max_concurrency = first.translation.max_concurrency + 1

    assert cli._translation_config_fingerprint(
        fingerprints, first
    ) == cli._translation_config_fingerprint(fingerprints, second)


def test_translate_config_fingerprint_uses_only_canonical_active_runtime(
    tmp_path: Path,
) -> None:
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    first = AppConfig(
        translation={"provider": "google-free"},
        providers={
            "openai_compatible": {
                "translation_base_url": "https://first.example/v1",
                "translation_model": "first-model",
            }
        },
    )
    second = AppConfig(
        translation={"provider": " GOOGLE-FREE "},
        providers={
            "openai_compatible": {
                "translation_base_url": "https://second.example/v1/",
                "translation_model": "second-model",
            }
        },
    )

    assert cli._translation_config_fingerprint(
        fingerprints, first
    ) == cli._translation_config_fingerprint(fingerprints, second)


def test_pipeline_progress_reporter_renders_overall_and_stage_lines() -> None:
    console = Console(record=True, width=120)

    with cli.PipelineProgressReporter(console, show_overall=True) as reporter:
        reporter.start_stage("translate", 4, "Preparing segments")
        reporter.update_stage("translate", 2, 4, "Translating segments 2/4")
        descriptions = [task.description for task in reporter._progress.tasks]
        reporter.complete_stage("translate", "translate done: 4/4 translated, 0 failed")

    rendered = console.export_text()
    assert any("Pipeline Translate" in description for description in descriptions)
    assert any(
        "Translate: Translating segments 2/4" in description
        for description in descriptions
    )
    assert "translate done: 4/4 translated, 0 failed" in rendered


def test_pipeline_progress_reporter_handles_removed_stage_task_ids() -> None:
    console = Console(record=True, width=120)

    with cli.PipelineProgressReporter(console, show_overall=True) as reporter:
        reporter.skip_stage("transcribe", "cache hit")
        reporter.start_stage("translate", 1, "Translating")
        reporter.complete_stage("translate", "translate done: 1/1 translated, 0 failed")
        reporter.start_stage("synthesize", 4, "Synthesizing")
        reporter.complete_stage(
            "synthesize", "synthesize done: 4/4 completed, 0 failed"
        )

    rendered = console.export_text()
    assert "translate done: 1/1 translated, 0 failed" in rendered
    assert "synthesize done: 4/4 completed, 0 failed" in rendered


def test_pipeline_progress_reporter_stage_only_skip_message_is_concise() -> None:
    console = Console(record=True, width=120)

    with cli.PipelineProgressReporter(console, show_overall=False) as reporter:
        reporter.skip_stage("translate", "cache hit")

    rendered = console.export_text()
    assert "translate skipped (cache hit)" in rendered
    assert "Pipeline Translate" not in rendered


def test_resume_help_documents_task_argument_and_latest_default() -> None:
    result = runner.invoke(cli.app, ["resume", "--help"])
    normalized = _normalize_help_output(result.output)

    assert result.exit_code == 0
    assert "TASK" in normalized
    assert "latest" in normalized.lower()
    assert "Interrupted or failed translate/tts stages resume" in normalized
    assert "--translation-provider" in normalized
    assert "--background" in normalized


def test_resume_loads_latest_task_and_executes_pipeline(
    tmp_path: Path, monkeypatch
) -> None:
    config_path, cfg = _create_config(tmp_path)
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)
    task_manifest = store.create_task(audio, cfg, f"podtran {audio}")
    captured: dict[str, object] = {}

    def fake_execute_pipeline(
        task_manifest,
        cfg,
        task_store,
        cache_store,
        fingerprints,
        min_speakers=2,
        max_speakers=5,
    ):
        captured["task_id"] = task_manifest.task_id
        return []

    monkeypatch.setattr(cli, "_execute_pipeline", fake_execute_pipeline)

    result = runner.invoke(
        cli.app,
        ["resume", "--config", str(config_path), "--workdir", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "Resuming task:" in result.output
    assert captured["task_id"] == task_manifest.task_id


def test_resume_translation_provider_override_is_one_shot(
    tmp_path: Path, monkeypatch
) -> None:
    config_path, cfg = _create_config(tmp_path)
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)
    task_manifest = store.create_task(audio, cfg, f"podtran {audio}")
    captured: dict[str, str] = {}

    def fake_execute_pipeline(task_manifest, cfg, *args, **kwargs):
        captured["provider"] = cfg.translation.provider
        return []

    monkeypatch.setattr(cli, "_execute_pipeline", fake_execute_pipeline)

    result = runner.invoke(
        cli.app,
        [
            "resume",
            task_manifest.task_id,
            "--translation-provider",
            " OPENAI-COMPATIBLE ",
            "--config",
            str(config_path),
            "--workdir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert captured["provider"] == "openai-compatible"
    assert load_config(config_path).translation.provider == "google-free"


def test_resume_background_passes_translation_provider_to_child(
    tmp_path: Path, monkeypatch
) -> None:
    config_path, cfg = _create_config(tmp_path)
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)
    task_manifest = store.create_task(audio, cfg, f"podtran {audio}")
    captured: dict[str, object] = {}

    class FakeProcess:
        pid = 4321

    def fake_popen(command, **kwargs):
        captured["command"] = command
        return FakeProcess()

    monkeypatch.setattr(cli.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        cli,
        "_execute_pipeline",
        lambda *args, **kwargs: pytest.fail("background resume executed inline"),
    )

    result = runner.invoke(
        cli.app,
        [
            "resume",
            task_manifest.task_id,
            "--background",
            "--translation-provider",
            "openai-compatible",
            "--config",
            str(config_path),
            "--workdir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    command = captured["command"]
    assert command[-2:] == ["--translation-provider", "openai-compatible"]
    assert "--background" not in command


def test_resume_background_does_not_start_completed_task(
    tmp_path: Path, monkeypatch
) -> None:
    config_path, cfg = _create_config(tmp_path)
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)
    task_manifest = store.create_task(audio, cfg, f"podtran {audio}")
    task_manifest.status = "completed"
    store.save_task(task_manifest)
    monkeypatch.setattr(
        cli.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("completed task started a child process"),
    )

    result = runner.invoke(
        cli.app,
        [
            "resume",
            task_manifest.task_id,
            "--background",
            "--config",
            str(config_path),
            "--workdir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Task is already complete" in result.output


def test_resume_background_rejects_live_background_process(
    tmp_path: Path, monkeypatch
) -> None:
    config_path, cfg = _create_config(tmp_path)
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)
    task_manifest = store.create_task(audio, cfg, f"podtran {audio}")
    paths = store.paths_for(task_manifest)
    (paths.task_dir / "run.pid").write_text("4321\n", encoding="utf-8")
    (paths.task_dir / "run.token").write_text("active-token\n", encoding="utf-8")
    monkeypatch.setattr(
        cli, "_verified_background_process", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        cli.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("duplicate background process started"),
    )

    result = runner.invoke(
        cli.app,
        [
            "resume",
            task_manifest.task_id,
            "--background",
            "--config",
            str(config_path),
            "--workdir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "already running in background with PID 4321" in result.output


def test_resume_background_replaces_stale_process_files(
    tmp_path: Path, monkeypatch
) -> None:
    config_path, cfg = _create_config(tmp_path)
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)
    task_manifest = store.create_task(audio, cfg, f"podtran {audio}")
    paths = store.paths_for(task_manifest)
    (paths.task_dir / "run.pid").write_text("999999\n", encoding="utf-8")
    (paths.task_dir / "run.token").write_text("stale-token\n", encoding="utf-8")

    class FakeProcess:
        pid = 4321

    monkeypatch.setattr(
        cli, "_verified_background_process", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())

    result = runner.invoke(
        cli.app,
        [
            "resume",
            task_manifest.task_id,
            "--background",
            "--config",
            str(config_path),
            "--workdir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert (paths.task_dir / "run.pid").read_text(encoding="utf-8") == "4321\n"
    assert (paths.task_dir / "run.token").read_text(
        encoding="utf-8"
    ).strip() != "stale-token"


def test_resume_background_rejects_unknown_provider_before_spawning(
    tmp_path: Path, monkeypatch
) -> None:
    config_path, cfg = _create_config(tmp_path)
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)
    task_manifest = store.create_task(audio, cfg, f"podtran {audio}")
    monkeypatch.setattr(
        cli.subprocess,
        "Popen",
        lambda *args, **kwargs: pytest.fail("invalid provider spawned a process"),
    )

    result = runner.invoke(
        cli.app,
        [
            "resume",
            task_manifest.task_id,
            "--background",
            "--translation-provider",
            "future-provider",
            "--config",
            str(config_path),
            "--workdir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 2
    assert not (store.paths_for(task_manifest).task_dir / "run.pid").exists()


def test_background_resume_options_accept_optional_translation_provider() -> None:
    options = cli._background_resume_options(
        [
            "--config",
            "config.toml",
            "--workdir",
            "workdir",
            "--min_speakers",
            "2",
            "--max_speakers",
            "5",
            "--translation-provider",
            "openai-compatible",
        ]
    )

    assert options is not None
    assert options["--translation-provider"] == "openai-compatible"


def test_resume_with_explicit_task_id(tmp_path: Path, monkeypatch) -> None:
    config_path, cfg = _create_config(tmp_path)
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)
    task_manifest = store.create_task(audio, cfg, f"podtran {audio}")
    captured: dict[str, object] = {}

    def fake_execute_pipeline(
        task_manifest,
        cfg,
        task_store,
        cache_store,
        fingerprints,
        min_speakers=2,
        max_speakers=5,
    ):
        captured["task_id"] = task_manifest.task_id
        return []

    monkeypatch.setattr(cli, "_execute_pipeline", fake_execute_pipeline)

    result = runner.invoke(
        cli.app,
        [
            "resume",
            task_manifest.task_id,
            "--config",
            str(config_path),
            "--workdir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert captured["task_id"] == task_manifest.task_id


def test_resume_aborts_when_no_tasks_exist(tmp_path: Path) -> None:
    config_path, _ = _create_config(tmp_path)

    result = runner.invoke(
        cli.app,
        ["resume", "--config", str(config_path), "--workdir", str(tmp_path)],
    )

    assert result.exit_code == 1
    assert "No tasks found" in result.output


def test_can_resume_partial_returns_true_for_interrupted_with_matching_fingerprints(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)
    cfg = AppConfig()
    task = store.create_task(audio, cfg, f"podtran {audio}")
    paths = store.paths_for(task)
    executor = StageExecutor(store, task, paths)
    input_fps = {"segments_json": "abc123"}
    config_fp = "cfg-hash"
    manifest = StageManifest(
        stage="translate",
        status="running",
        input_fingerprints=input_fps,
        config_fingerprint=config_fp,
        output_refs={"translated_json": "translated.json"},
    )
    write_json(paths.manifest_path("translate"), manifest)

    assert (
        cli._can_resume_partial(executor, "translate", 1, input_fps, config_fp) is True
    )


def test_can_resume_translate_partial_returns_true_when_config_changed(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)
    cfg = AppConfig()
    task = store.create_task(audio, cfg, f"podtran {audio}")
    paths = store.paths_for(task)
    executor = StageExecutor(store, task, paths)
    manifest = StageManifest(
        stage="translate",
        status="running",
        input_fingerprints={"segments_json": "abc123"},
        config_fingerprint="old-cfg-hash",
        output_refs={"translated_json": "translated.json"},
    )
    write_json(paths.manifest_path("translate"), manifest)

    assert cli._can_resume_partial(
        executor,
        "translate",
        1,
        {"segments_json": "abc123"},
        "new-cfg-hash",
        require_config_match=False,
    )


def test_can_resume_partial_returns_false_when_stage_completed(tmp_path: Path) -> None:
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)
    cfg = AppConfig()
    task = store.create_task(audio, cfg, f"podtran {audio}")
    paths = store.paths_for(task)
    executor = StageExecutor(store, task, paths)
    manifest = StageManifest(
        stage="translate",
        status="completed",
        input_fingerprints={"segments_json": "abc123"},
        config_fingerprint="cfg-hash",
        output_refs={"translated_json": "translated.json"},
    )
    write_json(paths.manifest_path("translate"), manifest)

    assert (
        cli._can_resume_partial(
            executor, "translate", 1, {"segments_json": "abc123"}, "cfg-hash"
        )
        is False
    )


def test_ensure_translate_preserves_partial_results_on_resume(
    tmp_path: Path, monkeypatch
) -> None:
    import podtran.translate as translate_module

    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)
    cache_store = CacheStore(tmp_path / "artifacts" / "cache")
    cfg = AppConfig()
    task = store.create_task(audio, cfg, f"podtran {audio}")
    paths = store.paths_for(task)
    executor = StageExecutor(store, task, paths)
    transcript = [
        TranscriptSegment(
            segment_id="ts_1", start=0.0, end=1.0, text="hello", speaker="SPEAKER_00"
        )
    ]
    write_json(paths.transcript_json, transcript)

    # Simulate an interrupted translate with partial results
    segments = cli._write_segments(paths, cfg)
    input_fps = {
        "segments_json": cli._translate_input_fingerprint(fingerprints, segments)
    }
    config_fp = fingerprints.hash_config_subset(cfg, cli.TRANSLATE_CONFIG_KEYS)
    interrupted_manifest = StageManifest(
        stage="translate",
        status="running",
        stage_version=cli.TRANSLATE_STAGE_VERSION,
        input_fingerprints=input_fps,
        config_fingerprint=config_fp,
        output_refs={
            "translated_json": "translated.json",
            "segments_json": "segments.json",
        },
    )
    write_json(paths.manifest_path("translate"), interrupted_manifest)
    # Write partial translated output
    partial = [segments[0].model_copy(update={"text_zh": "你好"})]
    write_json(paths.translated_json, partial)

    class FakeTranslator:
        def __init__(self, config: AppConfig) -> None:
            self.config = config

        def translate_segments(self, input_path, output_path, progress_callback=None):
            # _load_resume_segments should find the existing translated.json
            loaded = (
                read_model_list(output_path, SegmentRecord)
                if output_path.exists()
                else read_model_list(input_path, SegmentRecord)
            )
            return loaded

    monkeypatch.setattr(translate_module, "Translator", FakeTranslator)

    result = cli._ensure_translate(
        task, cfg, paths, executor, cache_store, fingerprints
    )

    assert result.action == "up-to-date"
    assert paths.translated_json.exists()
    restored = read_model_list(paths.translated_json, SegmentRecord)
    assert restored[0].text_zh == "你好"


def test_ensure_translate_preserves_partial_across_provider_change_and_skips_cache(
    tmp_path: Path, monkeypatch
) -> None:
    import podtran.translate as translate_module

    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)
    cache_store = CacheStore(tmp_path / "artifacts" / "cache")
    old_cfg = AppConfig(translation={"provider": "google-free"})
    new_cfg = AppConfig(translation={"provider": "openai-compatible"})
    task = store.create_task(audio, old_cfg, f"podtran {audio}")
    paths = store.paths_for(task)
    executor = StageExecutor(store, task, paths)
    transcript = [
        TranscriptSegment(
            segment_id="ts_1",
            start=0.0,
            end=1.0,
            text="hello",
            speaker="SPEAKER_00",
        ),
        TranscriptSegment(
            segment_id="ts_2",
            start=1.0,
            end=2.0,
            text="world",
            speaker="SPEAKER_01",
        ),
    ]
    write_json(paths.transcript_json, transcript)
    segments = cli._write_segments(paths, old_cfg)
    input_fps = {
        "segments_json": cli._translate_input_fingerprint(fingerprints, segments)
    }
    write_json(
        paths.manifest_path("translate"),
        StageManifest(
            stage="translate",
            status="failed",
            stage_version=cli.TRANSLATE_STAGE_VERSION,
            input_fingerprints=input_fps,
            config_fingerprint=cli._translation_config_fingerprint(
                fingerprints, old_cfg
            ),
            output_refs={
                "translated_json": "translated.json",
                "segments_json": "segments.json",
            },
        ),
    )
    write_json(
        paths.translated_json,
        [
            segments[0].model_copy(update={"text_zh": "你好"}),
            segments[1].model_copy(update={"error": "blocked"}),
        ],
    )

    class FakeTranslator:
        def __init__(self, config: AppConfig) -> None:
            assert config.translation.provider == "openai-compatible"

        def translate_segments(self, input_path, output_path, progress_callback=None):
            loaded = read_model_list(output_path, SegmentRecord)
            assert loaded[0].text_zh == "你好"
            return [
                loaded[0],
                loaded[1].model_copy(update={"text_zh": "世界", "error": None}),
            ]

    monkeypatch.setattr(translate_module, "Translator", FakeTranslator)
    monkeypatch.setattr(
        cache_store,
        "lookup",
        lambda *args, **kwargs: pytest.fail("partial task consulted shared cache"),
    )
    monkeypatch.setattr(
        cache_store,
        "publish",
        lambda *args, **kwargs: pytest.fail("resumed partial was published"),
    )

    result = cli._ensure_translate(
        task, new_cfg, paths, executor, cache_store, fingerprints
    )

    assert result.action == "run"
    translated = read_model_list(paths.translated_json, SegmentRecord)
    assert [item.text_zh for item in translated] == ["你好", "世界"]


def test_ensure_translate_complete_output_ignores_provider_change(
    tmp_path: Path, monkeypatch
) -> None:
    import podtran.translate as translate_module

    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)
    cache_store = CacheStore(tmp_path / "artifacts" / "cache")
    old_cfg = AppConfig(translation={"provider": "google-free"})
    new_cfg = AppConfig(translation={"provider": "openai-compatible"})
    task = store.create_task(audio, old_cfg, f"podtran {audio}")
    paths = store.paths_for(task)
    executor = StageExecutor(store, task, paths)
    write_json(
        paths.transcript_json,
        [
            TranscriptSegment(
                segment_id="ts_1",
                start=0.0,
                end=1.0,
                text="hello",
                speaker="SPEAKER_00",
            )
        ],
    )
    segments = cli._write_segments(paths, old_cfg)
    input_fps = {
        "segments_json": cli._translate_input_fingerprint(fingerprints, segments)
    }
    write_json(
        paths.translated_json,
        [segments[0].model_copy(update={"text_zh": "你好"})],
    )
    write_json(
        paths.manifest_path("translate"),
        StageManifest(
            stage="translate",
            status="completed",
            stage_version=cli.TRANSLATE_STAGE_VERSION,
            input_fingerprints=input_fps,
            config_fingerprint=cli._translation_config_fingerprint(
                fingerprints, old_cfg
            ),
            output_refs={
                "translated_json": "translated.json",
                "segments_json": "segments.json",
            },
        ),
    )
    monkeypatch.setattr(
        translate_module,
        "Translator",
        lambda *args, **kwargs: pytest.fail("complete translation was re-run"),
    )

    result = cli._ensure_translate(
        task, new_cfg, paths, executor, cache_store, fingerprints
    )

    assert result.action == "up-to-date"


def test_ensure_translate_discards_partial_when_segment_structure_changes(
    tmp_path: Path, monkeypatch
) -> None:
    import podtran.translate as translate_module

    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)
    cache_store = CacheStore(tmp_path / "artifacts" / "cache")
    cfg = AppConfig()
    task = store.create_task(audio, cfg, f"podtran {audio}")
    paths = store.paths_for(task)
    executor = StageExecutor(store, task, paths)
    write_json(
        paths.transcript_json,
        [
            TranscriptSegment(
                segment_id="ts_1",
                start=0.0,
                end=1.0,
                text="new source text",
                speaker="SPEAKER_00",
            )
        ],
    )
    old_segment = _segment("seg_00000", None, text_zh="旧译文")
    old_input_fps = {
        "segments_json": cli._translate_input_fingerprint(fingerprints, [old_segment])
    }
    write_json(paths.translated_json, [old_segment])
    write_json(
        paths.manifest_path("translate"),
        StageManifest(
            stage="translate",
            status="failed",
            stage_version=cli.TRANSLATE_STAGE_VERSION,
            input_fingerprints=old_input_fps,
            config_fingerprint=cli._translation_config_fingerprint(fingerprints, cfg),
            output_refs={
                "translated_json": "translated.json",
                "segments_json": "segments.json",
            },
        ),
    )

    class FakeTranslator:
        def __init__(self, config: AppConfig) -> None:
            pass

        def translate_segments(self, input_path, output_path, progress_callback=None):
            assert not output_path.exists()
            current = read_model_list(input_path, SegmentRecord)
            return [current[0].model_copy(update={"text_zh": "新译文"})]

    monkeypatch.setattr(translate_module, "Translator", FakeTranslator)

    result = cli._ensure_translate(
        task, cfg, paths, executor, cache_store, fingerprints
    )

    assert result.action == "run"
    translated = read_model_list(paths.translated_json, SegmentRecord)
    assert translated[0].text == "new source text"
    assert translated[0].text_zh == "新译文"


def test_ensure_translate_fails_when_any_segment_fails_and_preserves_output(
    tmp_path: Path, monkeypatch
) -> None:
    import podtran.translate as translate_module

    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)
    cache_store = CacheStore(tmp_path / "artifacts" / "cache")
    cfg = AppConfig()
    task = store.create_task(audio, cfg, f"podtran {audio}")
    paths = store.paths_for(task)
    executor = StageExecutor(store, task, paths)
    segments = [_segment("seg_1", None), _segment("seg_2", None)]

    def fake_write_segments(_paths: Path, _cfg: AppConfig) -> list[SegmentRecord]:
        write_json(paths.segments_json, segments)
        return segments

    class FakeTranslator:
        def __init__(self, config: AppConfig) -> None:
            self.config = config

        def translate_segments(self, input_path, output_path, progress_callback=None):
            return [
                segments[0].model_copy(update={"text_zh": "你好"}),
                segments[1].model_copy(update={"error": "RuntimeError: boom"}),
            ]

    monkeypatch.setattr(cli, "_write_segments", fake_write_segments)
    monkeypatch.setattr(translate_module, "Translator", FakeTranslator)

    with pytest.raises(
        RuntimeError, match="Translation failed for one or more segments."
    ):
        cli._ensure_translate(task, cfg, paths, executor, cache_store, fingerprints)

    manifest = read_model(paths.manifest_path("translate"), StageManifest)
    translated = read_model_list(paths.translated_json, SegmentRecord)
    assert manifest.status == "failed"
    assert translated[0].text_zh == "你好"
    assert translated[1].error == "RuntimeError: boom"
    assert cache_store.lookup("translate", manifest.cache_key) is None


def test_can_resume_partial_returns_true_for_interrupted_status(tmp_path: Path) -> None:
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)
    cfg = AppConfig()
    task = store.create_task(audio, cfg, f"podtran {audio}")
    paths = store.paths_for(task)
    executor = StageExecutor(store, task, paths)
    input_fps = {"segments_json": "abc123"}
    config_fp = "cfg-hash"
    manifest = StageManifest(
        stage="translate",
        status="interrupted",
        input_fingerprints=input_fps,
        config_fingerprint=config_fp,
        output_refs={"translated_json": "translated.json"},
        error="Interrupted by user",
    )
    write_json(paths.manifest_path("translate"), manifest)

    assert (
        cli._can_resume_partial(executor, "translate", 1, input_fps, config_fp) is True
    )


def test_can_resume_partial_returns_true_for_failed_status(tmp_path: Path) -> None:
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)
    cfg = AppConfig()
    task = store.create_task(audio, cfg, f"podtran {audio}")
    paths = store.paths_for(task)
    executor = StageExecutor(store, task, paths)
    input_fps = {"segments_json": "abc123"}
    config_fp = "cfg-hash"
    manifest = StageManifest(
        stage="translate",
        status="failed",
        input_fingerprints=input_fps,
        config_fingerprint=config_fp,
        output_refs={"translated_json": "translated.json"},
        error="Translation failed for one or more segments.",
    )
    write_json(paths.manifest_path("translate"), manifest)

    assert (
        cli._can_resume_partial(executor, "translate", 1, input_fps, config_fp) is True
    )


def test_can_resume_partial_returns_false_when_stage_version_changed(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)
    cfg = AppConfig()
    task = store.create_task(audio, cfg, f"podtran {audio}")
    paths = store.paths_for(task)
    executor = StageExecutor(store, task, paths)
    input_fps = {"segments_json": "abc123"}
    config_fp = "cfg-hash"
    manifest = StageManifest(
        stage="translate",
        status="interrupted",
        stage_version=2,
        input_fingerprints=input_fps,
        config_fingerprint=config_fp,
        output_refs={"translated_json": "translated.json"},
        error="Interrupted by user",
    )
    write_json(paths.manifest_path("translate"), manifest)

    # Old manifest has stage_version=2, current version is 3 → should not resume
    assert (
        cli._can_resume_partial(executor, "translate", 3, input_fps, config_fp) is False
    )
    # Same version → should resume
    assert (
        cli._can_resume_partial(executor, "translate", 2, input_fps, config_fp) is True
    )


def test_keyboard_interrupt_sets_interrupted_status(
    tmp_path: Path, monkeypatch
) -> None:
    import podtran.translate as translate_module

    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)
    cache_store = CacheStore(tmp_path / "artifacts" / "cache")
    cfg = AppConfig()
    task = store.create_task(audio, cfg, f"podtran {audio}")
    paths = store.paths_for(task)
    executor = StageExecutor(store, task, paths)
    transcript = [
        TranscriptSegment(
            segment_id="ts_1", start=0.0, end=1.0, text="hello", speaker="SPEAKER_00"
        )
    ]
    write_json(paths.transcript_json, transcript)

    class InterruptingTranslator:
        def __init__(self, config: AppConfig) -> None:
            self.config = config

        def translate_segments(self, input_path, output_path, progress_callback=None):
            raise KeyboardInterrupt()

    monkeypatch.setattr(translate_module, "Translator", InterruptingTranslator)

    with pytest.raises(KeyboardInterrupt):
        cli._ensure_translate(task, cfg, paths, executor, cache_store, fingerprints)

    manifest = read_model(paths.manifest_path("translate"), StageManifest)
    assert manifest.status == "interrupted"
    assert manifest.error == "Interrupted by user"
    reloaded_task = store.load_task(task.task_id)
    assert reloaded_task.status == "interrupted"


def test_synthesize_keyboard_interrupt_sets_interrupted_status(
    tmp_path: Path, monkeypatch
) -> None:
    import podtran.tts as tts_module

    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)
    cache_store = CacheStore(tmp_path / "artifacts" / "cache")
    cfg = AppConfig()
    task = store.create_task(audio, cfg, f"podtran {audio}")
    paths = store.paths_for(task)
    paths.ensure()
    write_json(paths.translated_json, [_segment("seg_1", None, text_zh="你好")])

    def interrupting_synthesize(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(tts_module, "synthesize_segments", interrupting_synthesize)
    monkeypatch.setattr(cli, "ensure_command", lambda command: None)

    with pytest.raises(KeyboardInterrupt):
        cli._ensure_synthesize(
            task,
            cfg,
            paths,
            StageExecutor(store, task, paths),
            cache_store,
            fingerprints,
        )

    manifest = read_model(paths.manifest_path("synthesize"), StageManifest)
    assert manifest.status == "interrupted"
    assert manifest.error == "Interrupted by user"
    reloaded_task = store.load_task(task.task_id)
    assert reloaded_task.status == "interrupted"


def test_ensure_synthesize_rejects_incomplete_translation_without_mutating_output(
    tmp_path: Path,
) -> None:
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)
    cache_store = CacheStore(tmp_path / "artifacts" / "cache")
    cfg = AppConfig()
    task = store.create_task(audio, cfg, f"podtran {audio}")
    paths = store.paths_for(task)
    paths.ensure()
    write_json(paths.translated_json, [_segment("seg_1", "RuntimeError: boom")])
    original = paths.translated_json.read_text(encoding="utf-8")

    with pytest.raises(ClickExit):
        cli._ensure_synthesize(
            task,
            cfg,
            paths,
            StageExecutor(store, task, paths),
            cache_store,
            fingerprints,
        )

    assert paths.translated_json.read_text(encoding="utf-8") == original
    assert not paths.manifest_path("synthesize").exists()


def test_require_completed_tts_rejects_partial_outputs(tmp_path: Path) -> None:
    translated_json = tmp_path / "translated.json"
    audio = tmp_path / "seg_1.wav"
    audio.write_bytes(b"tts")
    write_json(
        translated_json,
        [
            _segment(
                "seg_1",
                None,
                status="completed",
                tts_audio_path=str(audio.resolve()),
                text_zh="你好一",
            ),
            _segment("seg_2", "boom: bad", status="failed", text_zh="你好二"),
        ],
    )

    with pytest.raises(ClickExit):
        cli._require_completed_tts(translated_json)


def test_require_completed_tts_allows_skipped_unknown_segments(tmp_path: Path) -> None:
    translated_json = tmp_path / "translated.json"
    audio = tmp_path / "seg_1.wav"
    audio.write_bytes(b"tts")
    write_json(
        translated_json,
        [
            _segment(
                "seg_1",
                None,
                status="completed",
                tts_audio_path=str(audio.resolve()),
                text_zh="你好一",
            ),
            _segment(
                "seg_2",
                f"{cli.UNKNOWN_SPEAKER_TTS_SKIP_PREFIX}: speaker diarization did not identify this segment.",
                status="failed",
                text_zh="你好二",
            ).model_copy(update={"speaker": "UNKNOWN"}),
        ],
    )

    cli._require_completed_tts(translated_json)


def test_ensure_synthesize_fails_when_any_segment_fails_and_preserves_output(
    tmp_path: Path, monkeypatch
) -> None:
    import podtran.tts as tts_module

    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)
    cache_store = CacheStore(tmp_path / "artifacts" / "cache")
    cfg = AppConfig()
    task = store.create_task(audio, cfg, f"podtran {audio}")
    paths = store.paths_for(task)
    paths.ensure()
    translated = [
        _segment("seg_1", None, text_zh="你好一"),
        _segment("seg_2", None, text_zh="你好二"),
    ]
    write_json(paths.translated_json, translated)

    def fake_synthesize_segments(*args, **kwargs):
        completed_audio = paths.tts_dir / "seg_1_SPEAKER_00.wav"
        completed_audio.parent.mkdir(parents=True, exist_ok=True)
        completed_audio.write_bytes(b"tts-1")
        return [
            translated[0].model_copy(
                update={
                    "status": "completed",
                    "tts_audio_path": str(completed_audio.resolve()),
                    "tts_duration_ms": 1000,
                }
            ),
            translated[1].model_copy(update={"status": "failed", "error": "boom: bad"}),
        ]

    monkeypatch.setattr(cli, "ensure_command", lambda command: None)
    monkeypatch.setattr(tts_module, "synthesize_segments", fake_synthesize_segments)

    with pytest.raises(
        RuntimeError, match="Synthesis failed for one or more segments."
    ):
        cli._ensure_synthesize(
            task,
            cfg,
            paths,
            StageExecutor(store, task, paths),
            cache_store,
            fingerprints,
        )

    manifest = read_model(paths.manifest_path("synthesize"), StageManifest)
    synthesized = read_model_list(paths.translated_json, SegmentRecord)
    assert manifest.status == "failed"
    assert synthesized[0].status == "completed"
    assert synthesized[1].status == "failed"
    assert synthesized[1].error == "boom: bad"


def test_ensure_synthesize_warns_when_unknown_segments_are_skipped(
    tmp_path: Path, monkeypatch
) -> None:
    import podtran.tts as tts_module

    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)
    cache_store = CacheStore(tmp_path / "artifacts" / "cache")
    cfg = AppConfig(tts={"mode": "clone"})
    task = store.create_task(audio, cfg, f"podtran {audio}")
    paths = store.paths_for(task)
    paths.ensure()
    translated = [
        _segment("seg_1", None, text_zh="你好一"),
        _segment("seg_2", None, text_zh="你好二").model_copy(
            update={"speaker": "UNKNOWN"}
        ),
    ]
    write_json(paths.translated_json, translated)
    recorded_console = Console(record=True, width=160)
    previous_console = cli.console

    def fake_synthesize_segments(*args, **kwargs):
        completed_audio = paths.tts_dir / "seg_1_SPEAKER_00.wav"
        completed_audio.parent.mkdir(parents=True, exist_ok=True)
        completed_audio.write_bytes(b"tts-1")
        paths.voices_json.write_text("[]", encoding="utf-8")
        return [
            translated[0].model_copy(
                update={
                    "status": "completed",
                    "tts_audio_path": str(completed_audio.resolve()),
                    "tts_duration_ms": 1000,
                }
            ),
            translated[1].model_copy(
                update={
                    "status": "failed",
                    "error": f"{cli.UNKNOWN_SPEAKER_TTS_SKIP_PREFIX}: speaker diarization did not identify this segment.",
                }
            ),
        ]

    monkeypatch.setattr(cli, "ensure_command", lambda command: None)
    monkeypatch.setattr(tts_module, "synthesize_segments", fake_synthesize_segments)
    cli.console = recorded_console
    try:
        result = cli._ensure_synthesize(
            task,
            cfg,
            paths,
            StageExecutor(store, task, paths),
            cache_store,
            fingerprints,
        )
        rendered = recorded_console.export_text()
    finally:
        cli.console = previous_console

    manifest = read_model(paths.manifest_path("synthesize"), StageManifest)
    synthesized = read_model_list(paths.translated_json, SegmentRecord)
    assert result.action == "run"
    assert manifest.status == "completed"
    assert synthesized[1].status == "failed"
    assert "tts warnings:" in rendered
    assert "skipped 1 UNKNOWN segment" in rendered


def test_ensure_synthesize_resumes_failed_stage_and_reuses_completed_audio(
    tmp_path: Path, monkeypatch
) -> None:
    import podtran.tts as tts_module

    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)
    cache_store = CacheStore(tmp_path / "artifacts" / "cache")
    cfg = AppConfig()
    task = store.create_task(audio, cfg, f"podtran {audio}")
    paths = store.paths_for(task)
    paths.ensure()
    existing_audio = paths.tts_dir / "seg_1_SPEAKER_00.wav"
    existing_audio.parent.mkdir(parents=True, exist_ok=True)
    existing_audio.write_bytes(b"tts-1")
    translated = [
        _segment(
            "seg_1",
            None,
            status="completed",
            tts_audio_path=str(existing_audio.resolve()),
            text_zh="你好一",
        ),
        _segment("seg_2", "boom: bad", status="failed", text_zh="你好二"),
    ]
    write_json(paths.translated_json, translated)
    input_fps = {
        "translated_json": fingerprints.hash_json(
            cli._synthesize_input_segments(translated)
        )
    }
    config_fp = fingerprints.hash_config_subset(cfg, cli.SYNTHESIZE_CONFIG_KEYS)
    write_json(
        paths.manifest_path("synthesize"),
        StageManifest(
            stage="synthesize",
            status="failed",
            stage_version=cli.SYNTHESIZE_STAGE_VERSION,
            input_fingerprints=input_fps,
            config_fingerprint=config_fp,
            output_refs=cli._synthesize_output_refs(cfg),
            error="Synthesis failed for one or more segments.",
        ),
    )
    captured: dict[str, object] = {}

    def fake_synthesize_segments(input_path, output_path, *args, **kwargs):
        loaded = read_model_list(output_path, SegmentRecord)
        captured["loaded"] = loaded
        new_audio = paths.tts_dir / "seg_2_SPEAKER_00.wav"
        new_audio.write_bytes(b"tts-2")
        return [
            loaded[0],
            loaded[1].model_copy(
                update={
                    "status": "completed",
                    "error": None,
                    "tts_audio_path": str(new_audio.resolve()),
                    "tts_duration_ms": 1000,
                }
            ),
        ]

    monkeypatch.setattr(cli, "ensure_command", lambda command: None)
    monkeypatch.setattr(tts_module, "synthesize_segments", fake_synthesize_segments)

    result = cli._ensure_synthesize(
        task, cfg, paths, StageExecutor(store, task, paths), cache_store, fingerprints
    )

    manifest = read_model(paths.manifest_path("synthesize"), StageManifest)
    synthesized = read_model_list(paths.translated_json, SegmentRecord)
    loaded = captured["loaded"]
    assert result.action == "run"
    assert manifest.status == "completed"
    assert loaded[0].status == "completed"
    assert loaded[0].tts_audio_path == str(existing_audio.resolve())
    assert synthesized[0].tts_audio_path == str(existing_audio.resolve())
    assert synthesized[1].status == "completed"
    assert Path(synthesized[1].tts_audio_path).exists()


def test_execute_pipeline_prints_resume_hint_on_interrupt(
    tmp_path: Path, monkeypatch
) -> None:
    cfg, store, task_manifest = _preview_task(tmp_path)
    cache_store = CacheStore(tmp_path / "artifacts" / "cache")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    recorded_console = Console(record=True, width=160)
    previous_console = cli.console

    def interrupting_transcribe(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(cli, "_ensure_transcribe", interrupting_transcribe)
    cli.console = recorded_console
    try:
        with pytest.raises((SystemExit, Exception)):
            cli._execute_pipeline(task_manifest, cfg, store, cache_store, fingerprints)
        rendered = recorded_console.export_text()
    finally:
        cli.console = previous_console

    assert "Interrupted" in rendered
    assert f"podtran resume {task_manifest.task_id}" in rendered
    assert "Task complete" not in rendered


def test_execute_pipeline_prints_resume_hint_on_stage_failure(
    tmp_path: Path, monkeypatch
) -> None:
    cfg, store, task_manifest = _preview_task(tmp_path)
    cache_store = CacheStore(tmp_path / "artifacts" / "cache")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    recorded_console = Console(record=True, width=160)
    previous_console = cli.console

    monkeypatch.setattr(
        cli,
        "_ensure_transcribe",
        lambda *args, **kwargs: cli.StageDecision("transcribe", "up-to-date", "test"),
    )

    def failing_translate(*args, **kwargs):
        task_manifest.current_stage = "translate"
        task_manifest.status = "failed"
        raise RuntimeError("Translation failed for one or more segments.")

    monkeypatch.setattr(cli, "_ensure_translate", failing_translate)
    cli.console = recorded_console
    try:
        with pytest.raises(ClickExit):
            cli._execute_pipeline(task_manifest, cfg, store, cache_store, fingerprints)
        rendered = recorded_console.export_text()
    finally:
        cli.console = previous_console

    assert "Task failed at translate." in rendered
    assert f"podtran resume {task_manifest.task_id}" in rendered
    assert "Task complete" not in rendered


@pytest.mark.parametrize("background", [False, True])
def test_url_shortcut_creates_task_before_download(tmp_path, monkeypatch, background):
    config_path, _ = _create_config(tmp_path)
    url = "https://example.com/watch?v=abc&list=xyz"
    captured = []

    def capture(task, *args, **kwargs):
        captured.append(task)

    monkeypatch.setattr(cli, "_start_background_resume", capture)
    monkeypatch.setattr(cli, "_execute_pipeline", capture)
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "podtran",
            url,
            "--config",
            str(config_path),
            "--workdir",
            str(tmp_path),
            *(["--background"] if background else []),
        ],
    )
    cli.main()
    task = captured[0]
    assert task.source_url == url
    assert task.source_audio_path == ""
    assert task.source_audio_sha256 == ""
    assert task.current_stage == "download"


@pytest.mark.parametrize("error", [RuntimeError("offline"), KeyboardInterrupt()])
@pytest.mark.parametrize("sponsorblock", [False, True])
def test_download_failure_and_resume(tmp_path, monkeypatch, error, sponsorblock):
    cfg = AppConfig()
    cfg.download.sponsorblock = sponsorblock
    fingerprints = FingerprintService(tmp_path / "indexes")
    store = TaskStore(tmp_path, fingerprints)
    task = store.create_url_task("https://example.com/audio", cfg, "podtran URL")
    paths = store.paths_for(task)
    executor = StageExecutor(store, task, paths)
    monkeypatch.setattr(cli, "ensure_command", lambda command: None)

    def fail(url, directory, enabled):
        assert enabled is sponsorblock
        raise error

    monkeypatch.setattr(cli, "download_audio", fail)
    with pytest.raises(type(error)):
        cli._ensure_download(task, cfg, paths, executor, fingerprints)
    assert store.load_task(task.task_id).status == (
        "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
    )
    assert store.load_task(task.task_id).source_audio_path == ""
    audio = paths.task_dir / "source" / "episode.opus"
    audio.parent.mkdir()
    audio.write_bytes(b"downloaded")
    monkeypatch.setattr(cli, "download_audio", lambda *args: audio)
    task = store.load_task(task.task_id)
    executor = StageExecutor(store, task, paths)
    cli._ensure_download(task, cfg, paths, executor, fingerprints)
    assert task.processing_audio_sha256 == fingerprints.hash_audio(audio)
    assert executor.load_manifest("download").status == "completed"
    monkeypatch.setattr(cli, "download_audio", fail)
    cli._ensure_download(task, cfg, paths, executor, fingerprints)


def test_url_preview_failure_reuses_download_and_stop_records_stage(
    tmp_path, monkeypatch
):
    cfg = AppConfig()
    fingerprints = FingerprintService(tmp_path / "indexes")
    store = TaskStore(tmp_path, fingerprints)
    task = store.create_url_task(
        "https://example.com/audio", cfg, "podtran URL", preview=True
    )
    paths = store.paths_for(task)
    executor = StageExecutor(store, task, paths)
    monkeypatch.setattr(cli, "ensure_command", lambda command: None)
    audio = paths.task_dir / "episode.opus"
    audio.write_bytes(b"full episode")
    monkeypatch.setattr(cli, "download_audio", lambda *args: audio)

    def fail_preview(*args):
        raise RuntimeError("preview failed")

    monkeypatch.setattr(cli, "_create_preview_audio", fail_preview)
    with pytest.raises(RuntimeError):
        cli._ensure_download(task, cfg, paths, executor, fingerprints)
    task = store.load_task(task.task_id)
    executor = StageExecutor(store, task, paths)
    monkeypatch.setattr(
        cli, "download_audio", lambda *args: pytest.fail("download repeated")
    )

    def preview(*args):
        paths.preview_audio_path.write_bytes(b"preview")
        return paths.preview_audio_path

    monkeypatch.setattr(cli, "_create_preview_audio", preview)
    cli._ensure_download(task, cfg, paths, executor, fingerprints)
    assert task.processing_audio_path == str(paths.preview_audio_path)
    assert task.processing_audio_sha256 != task.source_audio_sha256
    stage = executor.start(StageManifest(stage="download"))
    assert cli._record_stopped_task(store, task.task_id, stage.pid)
    assert executor.load_manifest("download").status == "interrupted"
    assert audio.exists()


def test_url_pipeline_downloads_before_transcribe_and_status_shows_download(
    tmp_path, monkeypatch, capsys
):
    cfg = AppConfig()
    fingerprints = FingerprintService(tmp_path / "indexes")
    store = TaskStore(tmp_path, fingerprints)
    task = store.create_url_task("https://example.com/audio", cfg, "podtran URL")
    paths = store.paths_for(task)
    audio = paths.task_dir / "episode.opus"
    audio.write_bytes(b"audio")
    monkeypatch.setattr(cli, "ensure_command", lambda command: None)
    monkeypatch.setattr(cli, "download_audio", lambda *args: audio)
    seen = []

    def stage(task, *args, **kwargs):
        assert task.processing_audio_path == str(audio)
        assert task.processing_audio_sha256 == fingerprints.hash_audio(audio)
        seen.append(task.task_id)

    for name in [
        "_ensure_transcribe",
        "_ensure_translate",
        "_ensure_synthesize",
        "_ensure_compose",
    ]:
        monkeypatch.setattr(cli, name, stage)
    cli._execute_pipeline(task, cfg, store, CacheStore(paths.cache_dir), fingerprints)
    assert len(seen) == 4
    cli._print_task_stage_status(task, cfg, paths, StageExecutor(store, task, paths))
    assert "download" in capsys.readouterr().out


def test_transcribe_rejects_url_with_incomplete_download(tmp_path):
    cfg = AppConfig()
    fingerprints = FingerprintService(tmp_path / "indexes")
    store = TaskStore(tmp_path, fingerprints)
    task = store.create_url_task("https://example.com/audio", cfg, "podtran URL")
    paths = store.paths_for(task)
    with pytest.raises(ClickExit):
        cli._ensure_transcribe(
            task,
            cfg,
            paths,
            StageExecutor(store, task, paths),
            CacheStore(paths.cache_dir),
            fingerprints,
        )


@pytest.mark.parametrize(
    "command", ["run", "resume", "transcribe", "translate", "synthesize", "compose"]
)
def test_proxy_flags_are_mutually_exclusive(command):
    args = [command] + (["audio.mp3"] if command == "run" else ["task"])
    result = runner.invoke(
        cli.app, args + ["--proxy", "http://proxy:7890", "--no-proxy"]
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in _normalize_help_output(result.output)


def test_proxy_persistence_precedence_and_current_bypass(tmp_path, monkeypatch):
    from podtran.config import render_config_toml

    config_path, cfg = _create_config(tmp_path)
    cfg.proxy = "http://configured:7890"
    config_path.write_text(render_config_toml(cfg), encoding="utf-8")
    monkeypatch.setenv("HTTPS_PROXY", "http://environment:7890")
    observed = []
    monkeypatch.setattr(
        cli,
        "_execute_pipeline",
        lambda *a, **kw: observed.append(
            (os.environ["https_proxy"], os.environ["no_proxy"])
        ),
    )
    common = ["--config", str(config_path), "--workdir", str(tmp_path)]
    result = runner.invoke(
        cli.app,
        [
            "run",
            "https://example.com/episode",
            *common,
            "--proxy",
            "http://temporary:7890",
        ],
    )
    assert result.exit_code == 0, result.exception
    store = cli._load_runtime(config_path, tmp_path)[2]
    assert store.load_latest_task().proxy_override == "http://temporary:7890"
    cfg.no_proxy = ["new.internal"]
    config_path.write_text(render_config_toml(cfg), encoding="utf-8")
    for flags, expected in [
        ([], "http://temporary:7890"),
        (["--proxy", "http://replacement:7890"], "http://replacement:7890"),
        ([], "http://replacement:7890"),
        (["--no-proxy"], ""),
        ([], ""),
    ]:
        result = runner.invoke(cli.app, ["resume", *common, *flags])
        assert result.exit_code == 0, result.exception
        assert observed[-1][0] == expected
        assert (
            observed[-1][1] == "*"
            if not expected
            else "new.internal" in observed[-1][1]
        )
        assert store.load_latest_task().proxy_override == expected
    assert os.environ["HTTPS_PROXY"] == "http://environment:7890"


def test_background_proxy_inherits_and_is_saved(tmp_path, monkeypatch):
    from types import SimpleNamespace

    config_path, _ = _create_config(tmp_path)
    captured = {}

    def popen(command, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(pid=12345)

    monkeypatch.setattr(cli.subprocess, "Popen", popen)
    result = runner.invoke(
        cli.app,
        [
            "run",
            "https://example.com/audio",
            "--background",
            "--config",
            str(config_path),
            "--workdir",
            str(tmp_path),
            "--proxy",
            "http://proxy:7890",
        ],
    )
    assert result.exit_code == 0, result.exception
    assert captured["env"]["HTTPS_PROXY"] == "http://proxy:7890"
    assert (
        cli._load_runtime(config_path, tmp_path)[2].load_latest_task().proxy_override
        == "http://proxy:7890"
    )


def test_root_proxy_shortcut():
    assert cli._should_dispatch_root_task(
        ["--proxy", "http://proxy:7890", "episode.mp3"]
    )
    assert cli._should_dispatch_root_task(["--no-proxy", "episode.mp3"])


@pytest.mark.parametrize("configured", ["", "http://config:7890"])
def test_proxy_config_and_environment_fallback(tmp_path, monkeypatch, configured):
    from podtran.config import render_config_toml

    config_path, cfg = _create_config(tmp_path)
    cfg.proxy = configured
    config_path.write_text(render_config_toml(cfg), encoding="utf-8")
    monkeypatch.delenv("https_proxy", raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://environment:7890")
    observed = []
    monkeypatch.setattr(
        cli,
        "_execute_pipeline",
        lambda *a, **kw: observed.append(os.environ["HTTPS_PROXY"]),
    )
    result = runner.invoke(
        cli.app,
        [
            "run",
            "https://example.com/audio",
            "--config",
            str(config_path),
            "--workdir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.exception
    assert observed == [configured or "http://environment:7890"]
    assert (
        cli._load_runtime(config_path, tmp_path)[2].load_latest_task().proxy_override
        is None
    )


@pytest.mark.parametrize(
    "command", ["transcribe", "translate", "synthesize", "compose"]
)
def test_single_stage_proxy_override(tmp_path, monkeypatch, command):
    config_path, cfg = _create_config(tmp_path)
    store = cli._load_runtime(config_path, tmp_path)[2]
    task = store.create_url_task("https://example.com/audio", cfg, "podtran run")
    observed = []
    monkeypatch.setattr(cli, "_require_artifact", lambda *args: None)
    monkeypatch.setattr(cli, "_require_completed_tts", lambda *args: None)
    monkeypatch.setattr(
        cli,
        "_ensure_" + command,
        lambda *a, **kw: observed.append(os.environ["HTTPS_PROXY"]),
    )
    result = runner.invoke(
        cli.app,
        [
            command,
            task.task_id,
            "--config",
            str(config_path),
            "--workdir",
            str(tmp_path),
            "--proxy",
            "http://stage:7890",
        ],
    )
    assert result.exit_code == 0, result.exception
    assert observed == ["http://stage:7890"]
    assert store.load_task(task.task_id).proxy_override == "http://stage:7890"


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize(
    "configured,flag,expected",
    [
        (False, None, False),
        (True, None, True),
        (False, "--sponsorblock", True),
        (True, "--no-sponsorblock", False),
    ],
)
def test_sponsorblock_cli_precedence_and_task_persistence(
    tmp_path, monkeypatch, background, configured, flag, expected
):
    config_path, cfg = _create_config(tmp_path)
    cfg.download.sponsorblock = configured
    config_path.write_text(cli.render_config_toml(cfg), encoding="utf-8")
    captured = []
    monkeypatch.setattr(
        cli, "_execute_pipeline", lambda task, *a, **k: captured.append(task)
    )
    monkeypatch.setattr(
        cli, "_start_background_resume", lambda task, *a, **k: captured.append(task)
    )
    monkeypatch.setattr(
        cli.sys,
        "argv",
        [
            "podtran",
            *([flag] if flag else []),
            "https://www.youtube.com/watch?v=abc",
            "--config",
            str(config_path),
            "--workdir",
            str(tmp_path),
            *(["--background"] if background else []),
        ],
    )
    cli.main()
    task = captured[0]
    store = TaskStore(tmp_path, FingerprintService(tmp_path / "indexes"))
    assert store.load_task(task.task_id).sponsorblock is expected
    assert load_config(config_path).download.sponsorblock is configured


def test_sponsorblock_resume_uses_saved_choice_before_preview(tmp_path, monkeypatch):
    cfg = AppConfig(download={"sponsorblock": True})
    fingerprints = FingerprintService(tmp_path / "indexes")
    store = TaskStore(tmp_path, fingerprints)
    task = store.create_url_task(
        "https://www.youtube.com/watch?v=abc", cfg, "podtran URL", preview=True
    )
    task = store.load_task(task.task_id)
    cfg.download.sponsorblock = False
    paths = store.paths_for(task)
    executor = StageExecutor(store, task, paths)
    monkeypatch.setattr(cli, "ensure_command", lambda command: None)
    audio = paths.task_dir / "cut.opus"

    def download(url, directory, enabled):
        assert enabled is True
        audio.write_bytes(b"sponsor removed")
        return audio

    def preview(source, *args):
        assert source.read_bytes() == b"sponsor removed"
        paths.preview_audio_path.write_bytes(b"preview")
        return paths.preview_audio_path

    monkeypatch.setattr(cli, "download_audio", download)
    monkeypatch.setattr(cli, "_create_preview_audio", preview)
    cli._ensure_download(task, cfg, paths, executor, fingerprints)
    assert task.source_audio_sha256 == fingerprints.hash_audio(audio)
    monkeypatch.setattr(
        cli, "download_audio", lambda *args: pytest.fail("download repeated")
    )
    cli._ensure_download(task, cfg, paths, executor, fingerprints)
