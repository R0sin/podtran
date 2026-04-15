from datetime import datetime, timezone
from pathlib import Path

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


def _segment(segment_id: str, error: str | None, status: str = "pending", tts_audio_path: str = "") -> SegmentRecord:
    return SegmentRecord(
        segment_id=segment_id,
        block_id="b1",
        start=0.0,
        end=1.0,
        text="hello",
        speaker="SPEAKER_00",
        voice="Cherry",
        status=status,
        tts_audio_path=tts_audio_path,
        error=error,
    )


def _create_config(tmp_path: Path) -> tuple[Path, AppConfig]:
    config_path = tmp_path / "podtran.toml"
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

    assert result.exit_code == 0
    assert "run" in result.output
    assert "tasks" in result.output
    assert "status" in result.output
    assert "Recommended entrypoint" in result.output
    assert "podtran run AUDIO" in result.output
    assert "Shortcut" in result.output
    assert "podtran AUDIO [--preview]" in result.output
    assert "resume" not in result.output


def test_cache_help_only_lists_clean() -> None:
    result = runner.invoke(cli.app, ["cache", "--help"])

    assert result.exit_code == 0
    assert "clean" in result.output
    assert "only exposed maintenance action is" in result.output
    assert "inspect" not in result.output
    assert "\n| list" not in result.output


def test_run_help_exposes_audio_and_preview_options() -> None:
    result = runner.invoke(cli.app, ["run", "--help"])

    assert result.exit_code == 0
    assert "AUDIO" in result.output
    assert "--preview" in result.output
    assert "--min_speakers" in result.output
    assert "--max_speakers" in result.output
    assert "Create a new task for AUDIO and run the full pipeline." in result.output


def test_stage_help_documents_task_requirements() -> None:
    status_result = runner.invoke(cli.app, ["status", "--help"])
    translate_result = runner.invoke(cli.app, ["translate", "--help"])
    synthesize_result = runner.invoke(cli.app, ["synthesize", "--help"])
    compose_result = runner.invoke(cli.app, ["compose", "--help"])
    cache_clean_result = runner.invoke(cli.app, ["cache", "clean", "--help"])

    assert status_result.exit_code == 0
    assert "If TASK is omitted" in status_result.output
    assert "latest task" in status_result.output

    assert translate_result.exit_code == 0
    assert "Requires `transcript.json`" in translate_result.output

    assert synthesize_result.exit_code == 0
    assert "Requires `translated.json`" in synthesize_result.output

    assert compose_result.exit_code == 0
    assert "Requires `translated.json`" in compose_result.output
    assert "completed TTS output" in compose_result.output

    assert cache_clean_result.exit_code == 0
    assert "2026-04-01" in cache_clean_result.output
    assert "2026-04-01T12:30:00+08:00" in cache_clean_result.output


def test_init_prompts_for_provider_managed_config_and_writes_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "podtran.toml"

    result = runner.invoke(
        cli.app,
        ["init", "--config", str(config_path), "--workdir", str(tmp_path)],
        input="hf-token\ndash-key\n\n\n",
    )

    assert result.exit_code == 0
    config = load_config(config_path)
    assert config.hf_token == "hf-token"
    assert config.providers.dashscope.api_key == "dash-key"
    assert config.translation.provider == "dashscope"
    assert config.translation.model == "qwen3.5-flash"
    assert config.tts.provider == "dashscope"
    assert config.tts.model == "qwen3-tts-vc-2026-01-22"
    assert (tmp_path / "artifacts").exists()
    assert "speaker-diarization-community-1" in result.output
    assert "hf.co/settings/tokens" in result.output


def test_init_reprompts_required_values(tmp_path: Path) -> None:
    config_path = tmp_path / "podtran.toml"

    result = runner.invoke(
        cli.app,
        ["init", "--config", str(config_path), "--workdir", str(tmp_path)],
        input="\nhf-token\n\ndash-key\ncustom-translator\ncustom-tts\n",
    )

    assert result.exit_code == 0
    config = load_config(config_path)
    assert config.hf_token == "hf-token"
    assert config.providers.dashscope.api_key == "dash-key"
    assert config.translation.model == "custom-translator"
    assert config.tts.model == "custom-tts"
    assert result.output.count("Hugging Face token") >= 2
    assert result.output.count("DashScope API key") >= 2


def test_init_uses_existing_values_as_defaults_and_preserves_other_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "podtran.toml"
    config = AppConfig()
    config.hf_token = "hf-existing"
    config.providers.dashscope.api_key = "dash-existing"
    config.translation.model = "existing-translate"
    config.tts.model = "existing-tts"
    config.tts.voice_mode = "preset"
    config.tts.voice_map = {"SPEAKER_00": "Cherry"}
    config.asr.batch_size = 7
    config.compose.output_bitrate = "256k"
    config_path.write_text(cli.render_config_toml(config), encoding="utf-8")

    result = runner.invoke(
        cli.app,
        ["init", "--config", str(config_path), "--workdir", str(tmp_path)],
        input="\n\nupdated-translate\n\n",
    )

    assert result.exit_code == 0
    updated = load_config(config_path)
    assert updated.hf_token == "hf-existing"
    assert updated.providers.dashscope.api_key == "dash-existing"
    assert updated.translation.model == "updated-translate"
    assert updated.tts.model == "existing-tts"
    assert updated.tts.voice_mode == "preset"
    assert updated.tts.voice_map == {"SPEAKER_00": "Cherry"}
    assert updated.asr.batch_size == 7
    assert updated.compose.output_bitrate == "256k"


@pytest.mark.parametrize(
    "argv",
    [
        ["podtran", "{audio}", "--preview"],
        ["podtran", "--preview", "{audio}"],
    ],
)
def test_root_preview_creates_preview_task_and_executes_pipeline(tmp_path: Path, monkeypatch, capsys, argv: list[str]) -> None:
    config_path, _ = _create_config(tmp_path)
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    captured: dict[str, TaskManifest] = {}
    clip_args: dict[str, object] = {}

    def fake_execute_pipeline(task_manifest, cfg, task_store, cache_store, fingerprints, min_speakers=2, max_speakers=5):
        captured["task"] = task_manifest
        captured["min_speakers"] = min_speakers
        captured["max_speakers"] = max_speakers
        return []

    def fake_extract_audio_chunk(ffmpeg_path: str, source: Path, output: Path, start: float | None, end: float | None) -> Path:
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
    monkeypatch.setattr(cli.sys, "argv", [argv[0], "--config", str(config_path), "--workdir", str(tmp_path), *argv[1:]])

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


def test_root_audio_creates_task_and_executes_pipeline(tmp_path: Path, monkeypatch, capsys) -> None:
    config_path, _ = _create_config(tmp_path)
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    captured: dict[str, str] = {}

    def fake_execute_pipeline(task_manifest, cfg, task_store, cache_store, fingerprints, min_speakers=2, max_speakers=5):
        captured["task_id"] = task_manifest.task_id
        captured["entry_command"] = task_manifest.entry_command
        captured["min_speakers"] = min_speakers
        captured["max_speakers"] = max_speakers
        return []

    monkeypatch.setattr(cli, "_execute_pipeline", fake_execute_pipeline)
    monkeypatch.setattr(cli.sys, "argv", ["podtran", "--config", str(config_path), "--workdir", str(tmp_path), "--min_speakers", "3", "--max_speakers", "6", str(audio)])

    cli.main()
    stdout = capsys.readouterr().out

    assert "Created task:" in stdout
    assert captured["task_id"]
    assert captured["min_speakers"] == 3
    assert captured["max_speakers"] == 6
    assert "--min_speakers 3" in captured["entry_command"]
    assert "--max_speakers 6" in captured["entry_command"]


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
        StageManifest(stage="transcribe", status="completed", output_refs={"transcript_json": "transcript.json"}),
    )
    write_json(
        paths.manifest_path("translate"),
        StageManifest(
            stage="translate",
            status="failed",
            error="translation exploded",
            output_refs={"segments_json": "segments.json", "translated_json": "translated.json"},
        ),
    )

    status_result = runner.invoke(cli.app, ["status", task_manifest.task_id, "--config", str(config_path), "--workdir", str(tmp_path)])
    tasks_result = runner.invoke(cli.app, ["tasks", "--config", str(config_path), "--workdir", str(tmp_path)])

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
    assert "0-300s" in tasks_result.output


def test_translate_requires_transcript_json(tmp_path: Path) -> None:
    config_path, cfg = _create_config(tmp_path)
    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)
    task_manifest = store.create_task(audio, cfg, f"podtran {audio}")

    result = runner.invoke(cli.app, ["translate", task_manifest.task_id, "--config", str(config_path), "--workdir", str(tmp_path)])

    assert result.exit_code == 1
    assert "Cannot run translate: missing transcript.json." in result.output


def test_translate_shared_cache_ignores_voice_map_and_syncs_current_voice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import podtran.translate as translate_module

    audio = tmp_path / "episode.mp3"
    audio.write_bytes(b"audio")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)
    cache_store = CacheStore(tmp_path / "artifacts" / "cache")

    first_cfg = AppConfig()
    first_cfg.tts.voice_mode = "preset"
    first_cfg.tts.voice_map = {"SPEAKER_00": "Cherry"}

    second_cfg = first_cfg.model_copy(deep=True)
    second_cfg.tts.voice_map = {"SPEAKER_00": "Serena"}

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

        def translate_segments(self, input_path: Path, output_path: Path, progress_callback=None) -> list[SegmentRecord]:
            FakeTranslator.calls += 1
            segments = read_model_list(input_path, SegmentRecord)
            translated = [segment.model_copy(update={"text_zh": f"zh-{segment.segment_id}"}) for segment in segments]
            write_json(output_path, translated)
            return translated

    monkeypatch.setattr(translate_module, "Translator", FakeTranslator)

    first_executor = StageExecutor(store, first_task, first_paths)
    second_executor = StageExecutor(store, second_task, second_paths)

    first_result = cli._ensure_translate(first_task, first_cfg, first_paths, first_executor, cache_store, fingerprints)
    second_result = cli._ensure_translate(second_task, second_cfg, second_paths, second_executor, cache_store, fingerprints)
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

    def fake_run_transcription_with_progress(audio: Path, config: AppConfig, min_speakers: int, max_speakers: int, progress_callback=None):
        captured["audio"] = audio
        captured["min_speakers"] = min_speakers
        captured["max_speakers"] = max_speakers
        return []

    monkeypatch.setattr(cli, "ensure_command", lambda command: None)
    monkeypatch.setattr(cli, "ensure_hf_token", lambda token: None)
    monkeypatch.setattr(cli, "_run_transcription_with_progress", fake_run_transcription_with_progress)

    cli._ensure_transcribe(task_manifest, cfg, paths, executor, cache_store, fingerprints, min_speakers=3, max_speakers=6)

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
    write_json(paths.translated_json, [_segment("seg_1", None)])
    captured: dict[str, object] = {}

    def fake_synthesize_segments(*args, **kwargs):
        captured["source_audio"] = kwargs["source_audio"]
        captured["source_audio_fingerprint"] = kwargs["source_audio_fingerprint"]
        completed = _segment("seg_1", None, status="completed", tts_audio_path=str(paths.tts_dir / "seg_1.wav"))
        return [completed]

    monkeypatch.setattr(cli, "ensure_command", lambda command: None)
    monkeypatch.setattr(tts, "synthesize_segments", fake_synthesize_segments)

    cli._ensure_synthesize(task_manifest, cfg, paths, executor, cache_store, fingerprints)

    assert captured["source_audio"] == Path(task_manifest.processing_audio_path)
    assert captured["source_audio_fingerprint"] == task_manifest.processing_audio_sha256


def test_synthesize_is_up_to_date_after_successful_run(tmp_path: Path, monkeypatch) -> None:
    import podtran.tts as tts

    cfg, store, task_manifest = _preview_task(tmp_path)
    cfg.tts.voice_mode = "preset"
    paths = store.paths_for(task_manifest)
    executor = StageExecutor(store, task_manifest, paths)
    cache_store = CacheStore(paths.cache_dir)
    fingerprints = FingerprintService(paths.cache_indexes_dir)
    write_json(paths.translated_json, [_segment("seg_1", None)])

    def fake_synthesize_segments(*args, **kwargs):
        output = paths.tts_dir / "seg_1.wav"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"tts")
        return [_segment("seg_1", None, status="completed", tts_audio_path=str(output.resolve()))]

    monkeypatch.setattr(cli, "ensure_command", lambda command: None)
    monkeypatch.setattr(tts, "synthesize_segments", fake_synthesize_segments)

    first = cli._ensure_synthesize(task_manifest, cfg, paths, executor, cache_store, fingerprints)
    second = cli._ensure_synthesize(task_manifest, cfg, paths, executor, cache_store, fingerprints)

    assert first.action == "run"
    assert second.action == "up-to-date"


def test_preview_compose_uses_processing_audio_and_preview_output_name(tmp_path: Path, monkeypatch) -> None:
    cfg, store, task_manifest = _preview_task(tmp_path)
    paths = store.paths_for(task_manifest)
    executor = StageExecutor(store, task_manifest, paths)
    fingerprints = FingerprintService(paths.cache_indexes_dir)
    synthesized_audio = paths.tts_dir / "seg_1.wav"
    synthesized_audio.parent.mkdir(parents=True, exist_ok=True)
    synthesized_audio.write_bytes(b"tts")
    write_json(paths.translated_json, [_segment("seg_1", None, status="completed", tts_audio_path=str(synthesized_audio))])
    captured: dict[str, Path] = {}

    def fake_compose_output(source_audio: Path, segments, config, temp_dir: Path, output_path: Path, mode: str | None = None, progress_callback=None) -> Path:
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
    assert manifest.input_fingerprints["source_audio"] == task_manifest.processing_audio_sha256


def test_execute_pipeline_prints_absolute_output_path_for_preview_task(tmp_path: Path, monkeypatch) -> None:
    cfg, store, task_manifest = _preview_task(tmp_path)
    cache_store = CacheStore(tmp_path / "artifacts" / "cache")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    recorded_console = Console(record=True, width=160)
    expected_output_path = store.paths_for(task_manifest).final_dir / "episode.preview.interleave.mp3"
    previous_console = cli.console

    monkeypatch.setattr(cli, "_ensure_transcribe", lambda *args, **kwargs: cli.StageDecision("transcribe", "up-to-date", "test"))
    monkeypatch.setattr(cli, "_ensure_translate", lambda *args, **kwargs: cli.StageDecision("translate", "up-to-date", "test"))
    monkeypatch.setattr(cli, "_ensure_synthesize", lambda *args, **kwargs: cli.StageDecision("synthesize", "up-to-date", "test"))

    def fake_ensure_compose(task_manifest, cfg, paths, executor, fingerprints, reporter=None):
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
    assert f"Output file: {expected_output_path.resolve()}".replace("\n", "") in rendered.replace("\n", "")


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
    cache_store.publish("translate", "abc123", {"translated_json": cache_payload}, manifest)

    result = runner.invoke(
        cli.app,
        ["cache", "clean", "--before", "2026-04-01", "--config", str(config_path), "--workdir", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "Removed cache entries:" in result.output
    assert cache_store.lookup("translate", "abc123") is None


def test_parse_before_normalizes_to_utc() -> None:
    assert cli._parse_before("2026-04-01") == datetime(2026, 4, 1, tzinfo=timezone.utc)
    assert cli._parse_before("2026-04-01T08:00:00+08:00") == datetime(2026, 4, 1, tzinfo=timezone.utc)


def test_translate_config_fingerprint_includes_batch_size(tmp_path: Path) -> None:
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    first = AppConfig()
    second = AppConfig()
    second.translation.batch_size = first.translation.batch_size + 1

    assert (
        fingerprints.hash_config_subset(first, cli.TRANSLATE_CONFIG_KEYS)
        != fingerprints.hash_config_subset(second, cli.TRANSLATE_CONFIG_KEYS)
    )


def test_translate_config_fingerprint_ignores_max_concurrency(tmp_path: Path) -> None:
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    first = AppConfig()
    second = AppConfig()
    second.translation.max_concurrency = first.translation.max_concurrency + 1

    assert (
        fingerprints.hash_config_subset(first, cli.TRANSLATE_CONFIG_KEYS)
        == fingerprints.hash_config_subset(second, cli.TRANSLATE_CONFIG_KEYS)
    )


def test_pipeline_progress_reporter_renders_overall_and_stage_lines() -> None:
    console = Console(record=True, width=120)

    with cli.PipelineProgressReporter(console, show_overall=True) as reporter:
        reporter.start_stage("translate", 4, "Preparing segments")
        reporter.update_stage("translate", 2, 4, "Translating segments 2/4")
        descriptions = [task.description for task in reporter._progress.tasks]
        reporter.complete_stage("translate", "translate done: 4/4 translated, 0 failed")

    rendered = console.export_text()
    assert any("Pipeline Translate" in description for description in descriptions)
    assert any("Translate: Translating segments 2/4" in description for description in descriptions)
    assert "translate done: 4/4 translated, 0 failed" in rendered


def test_pipeline_progress_reporter_handles_removed_stage_task_ids() -> None:
    console = Console(record=True, width=120)

    with cli.PipelineProgressReporter(console, show_overall=True) as reporter:
        reporter.skip_stage("transcribe", "cache hit")
        reporter.start_stage("translate", 1, "Translating")
        reporter.complete_stage("translate", "translate done: 1/1 translated, 0 failed")
        reporter.start_stage("synthesize", 4, "Synthesizing")
        reporter.complete_stage("synthesize", "synthesize done: 4/4 completed, 0 failed")

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
