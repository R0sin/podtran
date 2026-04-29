from __future__ import annotations

import gc
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from podtran import __version__
from podtran.artifacts import (
    ArtifactPaths,
    output_refs_exist,
    read_model_list,
    remove_path,
    write_json,
)
from podtran.audio import FFMPEG_COMMAND, FFPROBE_COMMAND, extract_audio_chunk
from podtran.cache_store import CacheStore
from podtran.checks import ensure_audio_file, ensure_command, ensure_hf_token
from podtran.compose import compose_output
from podtran.config import (
    AppConfig,
    DEFAULT_TRANSLATION_PROVIDER,
    DEFAULT_TRANSLATION_MODEL,
    DEFAULT_TTS_PRESET_MODEL,
    DEFAULT_TTS_PROVIDER,
    DEFAULT_TTS_CLONE_MODEL,
    DEFAULT_QWEN_LOCAL_MODEL_SIZE,
    detect_legacy_tts_keys,
    detect_legacy_translation_keys,
    load_config,
    load_config_data,
    render_config_toml,
    resolve_config_path,
    resolve_workdir,
)
from podtran.fingerprints import (
    COMPOSE_CONFIG_KEYS,
    SYNTHESIZE_CONFIG_KEYS,
    TRANSCRIBE_CONFIG_KEYS,
    TRANSLATE_CONFIG_KEYS,
    FingerprintService,
)
from podtran.merge import merge_transcript_segments
from podtran.models import (
    TaskManifest,
    SegmentRecord,
    StageManifest,
    TranscriptSegment,
    VoiceProfile,
)
from podtran.stage_executor import StageExecutor
from podtran.stage_versions import (
    COMPOSE_STAGE_VERSION,
    SYNTHESIZE_STAGE_VERSION,
    TRANSCRIBE_STAGE_VERSION,
    TRANSLATE_STAGE_VERSION,
)
from podtran.tasks import TaskStore

PREVIEW_DURATION_SECONDS = 300.0
PREVIEW_START_SECONDS = 0.0

ROOT_HELP = """Translate English podcast audio into Chinese with a staged CLI.

Recommended entrypoint:
  podtran run AUDIO [--preview]

Shortcut:
  podtran AUDIO [--preview]

Resume an interrupted task:
  podtran resume [TASK]

Examples:
  podtran run path\\to\\episode.mp3 --preview
  podtran path\\to\\episode.mp3
  podtran resume
  podtran status
"""

RUN_HELP = """Create a new task for AUDIO and run the full pipeline.

This command executes transcribe, translate, synthesize, and compose for a new task.
Use --preview to process only the first five minutes before running the full audio.
"""

app = typer.Typer(no_args_is_help=True, add_completion=False, help=ROOT_HELP)
cache_app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Manage the shared artifact cache. The only exposed maintenance action is `clean`.",
)
app.add_typer(cache_app, name="cache", short_help="Manage the shared cache.")
console = Console()
KNOWN_COMMANDS = {
    "run",
    "resume",
    "init",
    "tasks",
    "status",
    "version",
    "transcribe",
    "translate",
    "synthesize",
    "compose",
    "cache",
}
DEFAULT_MIN_SPEAKERS = 2
DEFAULT_MAX_SPEAKERS = 5
TTS_PROVIDER_CHOICES = ("dashscope", "openai-compatible", "vllm-omni", "qwen-local")
TRANSLATION_PROVIDER_CHOICES = ("google-free", "openai-compatible")
TTS_MODE_CHOICES = ("auto", "preset", "clone")
DEFAULT_VLLM_OMNI_BASE_URL = "http://localhost:8091/v1"
UNKNOWN_SPEAKER = "UNKNOWN"
UNKNOWN_SPEAKER_TTS_SKIP_PREFIX = "Skipped TTS for UNKNOWN speaker"


@dataclass(slots=True)
class StageDecision:
    stage: str
    action: str
    reason: str


PIPELINE_STAGE_ORDER = ["transcribe", "translate", "synthesize", "compose"]
PIPELINE_STAGE_LABELS = {
    "transcribe": "Transcribe",
    "translate": "Translate",
    "synthesize": "Synthesize",
    "compose": "Compose",
}


class PipelineProgressReporter:
    def __init__(self, target_console: Console, show_overall: bool) -> None:
        self.console = target_console
        self.show_overall = show_overall
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=target_console,
        )
        self._overall_task_id: int | None = None
        self._stage_task_id: int | None = None
        self._current_stage: str | None = None
        self._stage_total: int = 1

    def __enter__(self) -> "PipelineProgressReporter":
        self._progress.__enter__()
        if self.show_overall:
            self._overall_task_id = self._progress.add_task(
                "Pipeline waiting", total=len(PIPELINE_STAGE_ORDER), completed=0
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._remove_stage_task()
        self._progress.__exit__(exc_type, exc, tb)

    def start_stage(self, stage: str, total: int, message: str) -> None:
        normalized_total = max(total, 1)
        self._remove_stage_task()
        self._current_stage = stage
        self._stage_total = normalized_total
        self._stage_task_id = self._progress.add_task(
            self._stage_description(stage, message),
            total=normalized_total,
            completed=0,
        )
        if self._overall_task_id is not None:
            self._progress.update(
                self._overall_task_id,
                completed=self._stage_start(stage),
                description=self._overall_description(stage, message),
            )

    def update_stage(
        self, stage: str, completed: int, total: int, message: str
    ) -> None:
        normalized_total = max(total, 1)
        normalized_completed = min(max(completed, 0), normalized_total)
        if self._current_stage != stage or self._stage_task_id is None:
            self.start_stage(stage, normalized_total, message)
        self._stage_total = normalized_total
        if self._stage_task_id is not None:
            self._progress.update(
                self._stage_task_id,
                total=normalized_total,
                completed=normalized_completed,
                description=self._stage_description(stage, message),
            )
        if self._overall_task_id is not None:
            self._progress.update(
                self._overall_task_id,
                completed=self._stage_start(stage),
                description=self._overall_description(stage, message),
            )

    def complete_stage(self, stage: str, summary: str) -> None:
        if self._stage_task_id is not None:
            self._progress.update(
                self._stage_task_id,
                total=self._stage_total,
                completed=self._stage_total,
                description=self._stage_description(stage, "Complete"),
            )
        if self._overall_task_id is not None:
            self._progress.update(
                self._overall_task_id,
                completed=self._stage_end(stage),
                description=self._overall_description(stage, "Complete"),
            )
        self.print(summary)
        self._remove_stage_task()

    def skip_stage(self, stage: str, action: str) -> None:
        if self._overall_task_id is not None:
            self._progress.update(
                self._overall_task_id,
                completed=self._stage_end(stage),
                description=self._overall_description(stage, f"Skipped ({action})"),
            )
        self.print(f"{stage} skipped ({action})")
        self._remove_stage_task()

    def fail_stage(self, stage: str, error: str) -> None:
        self.print(f"{stage} failed: {_truncate(error)}")
        self._remove_stage_task()

    def print(self, message: str) -> None:
        self._progress.console.print(message)

    def _remove_stage_task(self) -> None:
        if self._stage_task_id is None:
            self._current_stage = None
            self._stage_total = 1
            return
        self._progress.remove_task(self._stage_task_id)
        self._stage_task_id = None
        self._current_stage = None
        self._stage_total = 1

    def _stage_start(self, stage: str) -> int:
        return PIPELINE_STAGE_ORDER.index(stage)

    def _stage_end(self, stage: str) -> int:
        return self._stage_start(stage) + 1

    def _stage_description(self, stage: str, message: str) -> str:
        return f"{PIPELINE_STAGE_LABELS.get(stage, stage.title())}: {message}"

    def _overall_description(self, stage: str, message: str) -> str:
        if not self.show_overall:
            return "Pipeline"
        return f"Pipeline {PIPELINE_STAGE_LABELS.get(stage, stage.title())}: {message}"


@app.command(
    short_help="Create a task and run the full pipeline.",
    help=RUN_HELP,
)
def run(
    audio: Path = typer.Argument(
        ..., metavar="AUDIO", help="Input podcast audio for a new task."
    ),
    config: Optional[Path] = typer.Option(None, "--config", help="Config file path."),
    workdir: Optional[Path] = typer.Option(None, "--workdir", help="Override workdir."),
    preview: bool = typer.Option(
        False, "--preview", help="Run only the first five minutes as a preview task."
    ),
    min_speakers: int = typer.Option(
        DEFAULT_MIN_SPEAKERS,
        "--min_speakers",
        min=1,
        help="Minimum speaker count hint for diarization.",
    ),
    max_speakers: int = typer.Option(
        DEFAULT_MAX_SPEAKERS,
        "--max_speakers",
        min=1,
        help="Maximum speaker count hint for diarization.",
    ),
) -> None:
    _validate_speaker_bounds(min_speakers, max_speakers)
    _run_task(
        audio,
        config,
        workdir,
        preview=preview,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )


@app.command(
    short_help="Create or update config interactively.",
    help="Launch an interactive config wizard, write `podtran.toml`, and prepare the artifact workdir.",
)
def init(
    config: Optional[Path] = typer.Option(None, "--config", help="Config file path."),
    workdir: Optional[Path] = typer.Option(None, "--workdir", help="Artifact workdir."),
    force: bool = typer.Option(False, "--force", help="Overwrite existing config."),
) -> None:
    config_path = resolve_config_path(config, workdir)
    existing_config: AppConfig | None = None
    legacy_rebuild = False
    if config_path.exists():
        if force:
            existing_config = None
        else:
            raw_config = load_config_data(config_path)
            if detect_legacy_tts_keys(raw_config) or detect_legacy_translation_keys(
                raw_config
            ):
                existing_config = _rebuild_config_with_preserved_auth(raw_config)
                legacy_rebuild = True
            else:
                existing_config = AppConfig.model_validate(raw_config)
    config_data = _prompt_init_config(existing_config)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if legacy_rebuild:
        backup_path = _backup_legacy_config(config_path)
        console.print(
            f"[yellow]Legacy TTS config detected.[/yellow] Backed up old config to {backup_path} "
            "and rebuilt a fresh config shape. Only hf_token and provider API keys were preserved."
        )
    config_path.write_text(render_config_toml(config_data), encoding="utf-8")
    resolved_workdir = resolve_workdir(workdir, config_path=config_path)
    paths = ArtifactPaths.from_task_id(resolved_workdir, "example-run")
    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    paths.tasks_dir.mkdir(parents=True, exist_ok=True)
    paths.cache_indexes_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]Wrote config:[/green] {config_path}")
    console.print(f"[green]Prepared workdir:[/green] {resolved_workdir}")


def _prompt_init_config(existing_config: AppConfig | None = None) -> AppConfig:
    config = (
        existing_config.model_copy(deep=True)
        if existing_config is not None
        else AppConfig()
    )
    console.print("[bold]podtran init[/bold]")
    console.print(
        "Translation defaults to google-free. Choose the translation and TTS providers you want to configure."
    )
    console.print(
        "Before entering your Hugging Face token, accept the model terms at "
        "https://huggingface.co/pyannote/speaker-diarization-community-1 "
        "and create a token at https://hf.co/settings/tokens ."
    )
    config.hf_token = _prompt_required(
        "Hugging Face token", current_value=config.hf_token
    )
    config.translation.provider = _prompt_choice(
        "Translation provider",
        TRANSLATION_PROVIDER_CHOICES,
        config.translation.provider or DEFAULT_TRANSLATION_PROVIDER,
    )
    if config.translation.provider == "openai-compatible":
        config.providers.openai_compatible.translation_base_url = _prompt_required(
            "OpenAI-compatible translation base URL",
            current_value=config.providers.openai_compatible.translation_base_url,
        )
        config.providers.openai_compatible.translation_api_key = _prompt_optional(
            "OpenAI-compatible translation API key",
            current_value=config.providers.openai_compatible.translation_api_key,
        )
        config.providers.openai_compatible.translation_model = _prompt_with_default(
            "Translation model",
            config.providers.openai_compatible.translation_model
            or DEFAULT_TRANSLATION_MODEL,
        )
    provider = _prompt_choice(
        "TTS provider",
        TTS_PROVIDER_CHOICES,
        config.tts.provider or DEFAULT_TTS_PROVIDER,
    )
    config.tts.provider = provider
    if provider == "dashscope" and not config.providers.dashscope.api_key.strip():
        config.providers.dashscope.api_key = _prompt_required(
            "DashScope API key",
            hide_input=True,
            current_value=config.providers.dashscope.api_key,
        )

    if provider == "openai-compatible":
        config.tts.mode = "auto"
        config.providers.openai_compatible.tts_base_url = _prompt_required(
            "OpenAI-compatible TTS base URL",
            current_value=config.providers.openai_compatible.tts_base_url,
        )
        config.providers.openai_compatible.tts_api_key = _prompt_optional(
            "OpenAI-compatible TTS API key",
            current_value=config.providers.openai_compatible.tts_api_key,
        )
        config.providers.openai_compatible.tts_model = _prompt_with_default(
            "TTS preset model",
            config.providers.openai_compatible.tts_model or DEFAULT_TTS_PRESET_MODEL,
        )
        return config

    if provider == "vllm-omni":
        config.providers.vllm_omni.base_url = _prompt_with_default(
            "vLLM-Omni TTS base URL",
            config.providers.vllm_omni.base_url or DEFAULT_VLLM_OMNI_BASE_URL,
        )
        config.providers.vllm_omni.api_key = _prompt_optional(
            "vLLM-Omni API key",
            current_value=config.providers.vllm_omni.api_key,
        )

    config.tts.mode = _prompt_choice("TTS mode", TTS_MODE_CHOICES, config.tts.mode)
    effective_tts_mode = config.tts.effective_mode(provider)
    if effective_tts_mode == "preset":
        if provider == "qwen-local":
            config.providers.qwen_local.preset_model_size = _prompt_with_default(
                "Qwen local preset model size",
                config.providers.qwen_local.preset_model_size
                or DEFAULT_QWEN_LOCAL_MODEL_SIZE,
            )
        else:
            config.providers.dashscope.tts_preset_model = _prompt_with_default(
                "TTS preset model",
                config.providers.dashscope.tts_preset_model or DEFAULT_TTS_PRESET_MODEL,
            )
    else:
        if provider == "qwen-local":
            config.providers.qwen_local.clone_model_size = _prompt_with_default(
                "Qwen local clone model size",
                config.providers.qwen_local.clone_model_size
                or DEFAULT_QWEN_LOCAL_MODEL_SIZE,
            )
        elif provider == "vllm-omni":
            config.providers.vllm_omni.model = _prompt_optional(
                "vLLM-Omni model",
                current_value=config.providers.vllm_omni.model,
            )
        else:
            config.providers.dashscope.tts_clone_model = _prompt_with_default(
                "TTS clone model",
                config.providers.dashscope.tts_clone_model or DEFAULT_TTS_CLONE_MODEL,
            )
    return config


def _rebuild_config_with_preserved_auth(raw_config: dict[str, object]) -> AppConfig:
    rebuilt = AppConfig()
    rebuilt.hf_token = str(raw_config.get("hf_token", "") or "").strip()
    providers = raw_config.get("providers")
    if isinstance(providers, dict):
        dashscope = providers.get("dashscope")
        if isinstance(dashscope, dict):
            rebuilt.providers.dashscope.api_key = str(
                dashscope.get("api_key", "") or ""
            ).strip()
        openai_compatible = providers.get("openai_compatible")
        if isinstance(openai_compatible, dict):
            rebuilt.providers.openai_compatible.translation_api_key = str(
                openai_compatible.get("translation_api_key", "") or ""
            ).strip()
            rebuilt.providers.openai_compatible.tts_api_key = str(
                openai_compatible.get("tts_api_key", "") or ""
            ).strip()
        vllm_omni = providers.get("vllm_omni")
        if isinstance(vllm_omni, dict):
            rebuilt.providers.vllm_omni.api_key = str(
                vllm_omni.get("api_key", "") or ""
            ).strip()
    tts = raw_config.get("tts")
    if isinstance(tts, dict):
        legacy_vllm_omni = tts.get("vllm_omni")
        if (
            isinstance(legacy_vllm_omni, dict)
            and not rebuilt.providers.vllm_omni.api_key
        ):
            rebuilt.providers.vllm_omni.api_key = str(
                legacy_vllm_omni.get("api_key", "") or ""
            ).strip()
    return rebuilt


def _backup_legacy_config(config_path: Path) -> Path:
    backup_path = config_path.with_name(f"{config_path.name}.bak")
    if not backup_path.exists():
        backup_path.write_text(
            config_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
    return backup_path


def _prompt_required(
    label: str, *, hide_input: bool = False, current_value: str = ""
) -> str:
    while True:
        prompt_label = (
            label
            if not current_value.strip()
            else f"{label} (press Enter to keep existing)"
        )
        value = typer.prompt(
            prompt_label, hide_input=hide_input, default="", show_default=False
        ).strip()
        if value:
            return value
        if current_value.strip():
            return current_value.strip()
        console.print(f"[yellow]{label} cannot be empty.[/yellow]")


def _prompt_with_default(label: str, default: str) -> str:
    return typer.prompt(label, default=default, show_default=True).strip()


def _prompt_optional(label: str, current_value: str = "") -> str:
    prompt_label = (
        label
        if not current_value.strip()
        else f"{label} (press Enter to keep existing)"
    )
    return (
        typer.prompt(prompt_label, default="", show_default=False).strip()
        or current_value.strip()
    )


def _prompt_choice(label: str, options: tuple[str, ...], default: str) -> str:
    normalized_options = tuple(option.strip() for option in options)
    fallback = (
        default.strip()
        if default.strip() in normalized_options
        else normalized_options[0]
    )
    while True:
        prompt = f"{label} ({'/'.join(normalized_options)})"
        value = typer.prompt(prompt, default=fallback, show_default=True).strip()
        if value in normalized_options:
            return value
        console.print(
            f"[yellow]{label} must be one of: {', '.join(normalized_options)}[/yellow]"
        )


RESUME_HELP = """Resume an existing task and re-run the pipeline from where it left off.

If TASK is omitted, picks the latest task. Completed stages are skipped automatically.
Interrupted or failed translate/tts stages resume from the last compatible checkpoint.
"""


@app.command(
    short_help="Resume an existing task.",
    help=RESUME_HELP,
)
def resume(
    task: Optional[str] = typer.Argument(
        None, metavar="TASK", help="Task id or unique prefix. Defaults to latest."
    ),
    config: Optional[Path] = typer.Option(None, "--config", help="Config file path."),
    workdir: Optional[Path] = typer.Option(None, "--workdir", help="Override workdir."),
    min_speakers: int = typer.Option(
        DEFAULT_MIN_SPEAKERS,
        "--min_speakers",
        min=1,
        help="Minimum speaker count hint for diarization.",
    ),
    max_speakers: int = typer.Option(
        DEFAULT_MAX_SPEAKERS,
        "--max_speakers",
        min=1,
        help="Maximum speaker count hint for diarization.",
    ),
) -> None:
    _validate_speaker_bounds(min_speakers, max_speakers)
    cfg, _, task_store, cache_store, fingerprints = _load_runtime(config, workdir)
    try:
        task_manifest = (
            task_store.load_task(task) if task else task_store.load_latest_task()
        )
    except FileNotFoundError:
        _abort("No tasks found. Use 'podtran run AUDIO' to create a new task.")
        return  # unreachable, _abort raises
    console.print(f"[green]Resuming task:[/green] {task_manifest.task_id}")
    _execute_pipeline(
        task_manifest,
        cfg,
        task_store,
        cache_store,
        fingerprints,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )


@app.command(
    short_help="List recent tasks.",
    help="List recent tasks from the workdir so you can inspect task ids, modes, stages, and status.",
)
def tasks(
    config: Optional[Path] = typer.Option(None, "--config", help="Config file path."),
    workdir: Optional[Path] = typer.Option(None, "--workdir", help="Override workdir."),
    limit: int = typer.Option(20, "--limit", min=1, help="Maximum tasks to show."),
) -> None:
    _, _, task_store, _, _ = _load_runtime(config, workdir)
    entries = task_store.list_tasks(limit=limit)
    if not entries:
        console.print("[yellow]No tasks found.[/yellow]")
        return

    table = Table(title="podtran tasks")
    table.add_column("Task ID")
    table.add_column("Audio")
    table.add_column("Mode")
    table.add_column("Status")
    table.add_column("Stage")
    table.add_column("Updated")
    for item in entries:
        table.add_row(
            item.task_id,
            item.source_audio_name,
            _task_mode_label(item),
            item.status,
            item.current_stage or "-",
            item.updated_at,
        )
    console.print(table)


@app.command(
    short_help="Show task status.",
    help="Show the factual status for a task. If TASK is omitted, podtran loads the latest task in the workdir.",
)
def status(
    task: Optional[str] = typer.Argument(
        None, metavar="TASK", help="Task id or unique prefix."
    ),
    config: Optional[Path] = typer.Option(None, "--config", help="Config file path."),
    workdir: Optional[Path] = typer.Option(None, "--workdir", help="Override workdir."),
) -> None:
    cfg, _, task_store, _, _ = _load_runtime(config, workdir)
    try:
        task_manifest = (
            task_store.load_task(task) if task else task_store.load_latest_task()
        )
    except FileNotFoundError:
        console.print("[yellow]No tasks found.[/yellow]")
        return

    paths = task_store.paths_for(task_manifest)
    executor = StageExecutor(task_store, task_manifest, paths)
    preview_fields = (
        f" mode=preview clip={_preview_window_label(task_manifest)}"
        if task_manifest.preview
        else " mode=full"
    )
    console.print(
        f"task_id={task_manifest.task_id} audio={task_manifest.source_audio_name}{preview_fields} "
        f"status={task_manifest.status} stage={task_manifest.current_stage or '-'}"
    )
    _print_task_stage_status(task_manifest, cfg, paths, executor)

    if paths.translated_json.exists():
        segments = read_model_list(paths.translated_json, SegmentRecord)
        completed = sum(1 for item in segments if item.status == "completed")
        failed = sum(1 for item in segments if item.status == "failed")
        translated = sum(1 for item in segments if item.text_zh.strip())
        console.print(
            f"segments={len(segments)} translated={translated} completed_tts={completed} failed={failed}"
        )
    if paths.voices_json.exists():
        profiles = read_model_list(paths.voices_json, VoiceProfile)
        cloned = sum(1 for item in profiles if item.status == "completed")
        failed_profiles = sum(1 for item in profiles if item.status == "failed")
        console.print(
            f"voices={len(profiles)} cloned={cloned} failed_voice_profiles={failed_profiles}"
        )


@app.command(
    short_help="Show the installed podtran version.",
    help="Print the installed podtran package version.",
)
def version() -> None:
    console.print(f"podtran {__version__}")


@app.command(
    short_help="Run transcription for an existing task.",
    help="Run only the transcription stage for TASK. This command operates on an existing task and refreshes `transcript.json` when needed.",
)
def transcribe(
    task: str = typer.Argument(..., metavar="TASK", help="Task id or unique prefix."),
    config: Optional[Path] = typer.Option(None, "--config", help="Config file path."),
    workdir: Optional[Path] = typer.Option(None, "--workdir", help="Override workdir."),
    min_speakers: int = typer.Option(
        DEFAULT_MIN_SPEAKERS,
        "--min_speakers",
        min=1,
        help="Minimum speaker count hint for diarization.",
    ),
    max_speakers: int = typer.Option(
        DEFAULT_MAX_SPEAKERS,
        "--max_speakers",
        min=1,
        help="Maximum speaker count hint for diarization.",
    ),
) -> None:
    _validate_speaker_bounds(min_speakers, max_speakers)
    cfg, task_manifest, paths, executor, cache_store, fingerprints = _load_task_context(
        task, config, workdir
    )
    with PipelineProgressReporter(console, show_overall=False) as reporter:
        _ensure_transcribe(
            task_manifest,
            cfg,
            paths,
            executor,
            cache_store,
            fingerprints,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            reporter=reporter,
        )


@app.command(
    short_help="Run translation for an existing task.",
    help="Run only the translation stage for TASK. Requires `transcript.json` to exist for that task.",
)
def translate(
    task: str = typer.Argument(..., metavar="TASK", help="Task id or unique prefix."),
    config: Optional[Path] = typer.Option(None, "--config", help="Config file path."),
    workdir: Optional[Path] = typer.Option(None, "--workdir", help="Override workdir."),
) -> None:
    cfg, task_manifest, paths, executor, cache_store, fingerprints = _load_task_context(
        task, config, workdir
    )
    _require_artifact(paths.transcript_json, "translate", "transcript.json")
    with PipelineProgressReporter(console, show_overall=False) as reporter:
        _ensure_translate(
            task_manifest,
            cfg,
            paths,
            executor,
            cache_store,
            fingerprints,
            reporter=reporter,
        )


@app.command(
    short_help="Run TTS synthesis for an existing task.",
    help="Run only the synthesis stage for TASK. Requires `translated.json` with all translations completed successfully.",
)
def synthesize(
    task: str = typer.Argument(..., metavar="TASK", help="Task id or unique prefix."),
    config: Optional[Path] = typer.Option(None, "--config", help="Config file path."),
    workdir: Optional[Path] = typer.Option(None, "--workdir", help="Override workdir."),
) -> None:
    cfg, task_manifest, paths, executor, cache_store, fingerprints = _load_task_context(
        task, config, workdir
    )
    _require_artifact(paths.translated_json, "synthesize", "translated.json")
    with PipelineProgressReporter(console, show_overall=False) as reporter:
        _ensure_synthesize(
            task_manifest,
            cfg,
            paths,
            executor,
            cache_store,
            fingerprints,
            reporter=reporter,
        )


@app.command(
    short_help="Compose the final output for an existing task.",
    help="Run only the compose stage for TASK. Requires `translated.json` and successful TTS output for every required segment.",
)
def compose(
    task: str = typer.Argument(..., metavar="TASK", help="Task id or unique prefix."),
    config: Optional[Path] = typer.Option(None, "--config", help="Config file path."),
    workdir: Optional[Path] = typer.Option(None, "--workdir", help="Override workdir."),
) -> None:
    cfg, task_manifest, paths, executor, _, fingerprints = _load_task_context(
        task, config, workdir
    )
    _require_artifact(paths.translated_json, "compose", "translated.json")
    _require_completed_tts(paths.translated_json)
    with PipelineProgressReporter(console, show_overall=False) as reporter:
        _ensure_compose(
            task_manifest, cfg, paths, executor, fingerprints, reporter=reporter
        )


@cache_app.command(
    "clean",
    short_help="Delete old cache entries.",
    help="Delete shared cache entries, optionally limited by `--before`. Example values: `2026-04-01` or `2026-04-01T12:30:00+08:00`.",
)
def cache_clean(
    before: Optional[str] = typer.Option(
        None, "--before", help="Delete cache entries finished before ISO datetime/date."
    ),
    config: Optional[Path] = typer.Option(None, "--config", help="Config file path."),
    workdir: Optional[Path] = typer.Option(None, "--workdir", help="Override workdir."),
) -> None:
    _, _, _, cache_store, _ = _load_runtime(config, workdir)
    cutoff = _parse_before(before) if before else None
    removed = cache_store.clean(cutoff)
    console.print(f"[green]Removed cache entries:[/green] {removed}")


def main() -> None:
    argv = sys.argv[1:]
    original_argv = sys.argv[:]
    if _should_dispatch_root_task(argv):
        sys.argv = [sys.argv[0], "run", *argv]
    try:
        app(standalone_mode=False)
    finally:
        sys.argv = original_argv


def _should_dispatch_root_task(argv: list[str]) -> bool:
    option_with_value = {"--config", "--workdir", "--min_speakers", "--max_speakers"}
    boolean_options = {"--preview"}
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in {"-h", "--help"}:
            return False
        if token in option_with_value:
            index += 2
            continue
        if token in boolean_options:
            index += 1
            continue
        if token.startswith("-"):
            return False
        return token not in KNOWN_COMMANDS
    return False


def _load_runtime(
    config_path: Optional[Path],
    workdir_override: Optional[Path],
) -> tuple[AppConfig, Path, TaskStore, CacheStore, FingerprintService]:
    resolved_config_path = resolve_config_path(config_path, workdir_override)
    cfg = load_config(resolved_config_path)
    workdir = resolve_workdir(workdir_override, config_path=resolved_config_path)
    artifacts_dir = workdir / "artifacts"
    cache_dir = artifacts_dir / "cache"
    cache_indexes_dir = cache_dir / "_indexes"
    fingerprints = FingerprintService(cache_indexes_dir)
    task_store = TaskStore(workdir, fingerprints)
    cache_store = CacheStore(cache_dir)
    return cfg, workdir, task_store, cache_store, fingerprints


def _run_task(
    audio: Path,
    config_path: Optional[Path],
    workdir_override: Optional[Path],
    preview: bool = False,
    min_speakers: int = DEFAULT_MIN_SPEAKERS,
    max_speakers: int = DEFAULT_MAX_SPEAKERS,
) -> None:
    ensure_audio_file(audio)
    cfg, _, task_store, cache_store, fingerprints = _load_runtime(
        config_path, workdir_override
    )
    entry_command = _entry_command(audio, preview, min_speakers, max_speakers)
    if preview:
        ensure_command(FFMPEG_COMMAND)
        task_id, source_audio_sha256 = task_store.reserve_task_id(audio)
        paths = ArtifactPaths.from_task_id(task_store.workdir, task_id)
        paths.ensure()
        try:
            processing_audio = _create_preview_audio(audio.resolve(), cfg, paths)
            processing_audio_sha256 = fingerprints.hash_audio(processing_audio)
        except Exception:
            remove_path(paths.task_dir)
            raise
        task_manifest = task_store.create_task_with_processing_audio(
            audio,
            cfg,
            entry_command=entry_command,
            task_id=task_id,
            source_audio_sha256=source_audio_sha256,
            processing_audio=processing_audio,
            processing_audio_sha256=processing_audio_sha256,
            preview=True,
            preview_start_seconds=PREVIEW_START_SECONDS,
            preview_duration_seconds=PREVIEW_DURATION_SECONDS,
        )
    else:
        task_manifest = task_store.create_task(audio, cfg, entry_command=entry_command)
    console.print(f"[green]Created task:[/green] {task_manifest.task_id}")
    _execute_pipeline(
        task_manifest,
        cfg,
        task_store,
        cache_store,
        fingerprints,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )


def _create_preview_audio(audio: Path, cfg: AppConfig, paths: ArtifactPaths) -> Path:
    remove_path(paths.preview_audio_path)
    return extract_audio_chunk(
        FFMPEG_COMMAND,
        audio,
        paths.preview_audio_path,
        PREVIEW_START_SECONDS,
        PREVIEW_START_SECONDS + PREVIEW_DURATION_SECONDS,
    )


def _entry_command(
    audio: Path, preview: bool, min_speakers: int, max_speakers: int
) -> str:
    preview_flag = " --preview" if preview else ""
    speaker_flags = ""
    if min_speakers != DEFAULT_MIN_SPEAKERS:
        speaker_flags += f" --min_speakers {min_speakers}"
    if max_speakers != DEFAULT_MAX_SPEAKERS:
        speaker_flags += f" --max_speakers {max_speakers}"
    return f"podtran{preview_flag}{speaker_flags} {audio}"


def _load_task_context(
    task: str,
    config_path: Optional[Path],
    workdir_override: Optional[Path],
) -> tuple[
    AppConfig,
    TaskManifest,
    ArtifactPaths,
    StageExecutor,
    CacheStore,
    FingerprintService,
]:
    cfg, _, task_store, cache_store, fingerprints = _load_runtime(
        config_path, workdir_override
    )
    task_manifest = task_store.load_task(task)
    paths = task_store.paths_for(task_manifest)
    paths.ensure()
    executor = StageExecutor(task_store, task_manifest, paths)
    return cfg, task_manifest, paths, executor, cache_store, fingerprints


def _execute_pipeline(
    task_manifest: TaskManifest,
    cfg: AppConfig,
    task_store: TaskStore,
    cache_store: CacheStore,
    fingerprints: FingerprintService,
    min_speakers: int = DEFAULT_MIN_SPEAKERS,
    max_speakers: int = DEFAULT_MAX_SPEAKERS,
) -> list[StageDecision]:
    paths = task_store.paths_for(task_manifest)
    paths.ensure()
    executor = StageExecutor(task_store, task_manifest, paths)
    decisions: list[StageDecision] = []

    with PipelineProgressReporter(console, show_overall=True) as reporter:
        try:
            decisions.append(
                _ensure_transcribe(
                    task_manifest,
                    cfg,
                    paths,
                    executor,
                    cache_store,
                    fingerprints,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    reporter=reporter,
                )
            )
            decisions.append(
                _ensure_translate(
                    task_manifest,
                    cfg,
                    paths,
                    executor,
                    cache_store,
                    fingerprints,
                    reporter=reporter,
                )
            )
            decisions.append(
                _ensure_synthesize(
                    task_manifest,
                    cfg,
                    paths,
                    executor,
                    cache_store,
                    fingerprints,
                    reporter=reporter,
                )
            )
            decisions.append(
                _ensure_compose(
                    task_manifest, cfg, paths, executor, fingerprints, reporter=reporter
                )
            )
        except KeyboardInterrupt:
            console.print(
                f"\n[yellow]Interrupted.[/yellow] Resume with: podtran resume {task_manifest.task_id}"
            )
            raise typer.Exit(code=1)
        except typer.Exit:
            raise
        except Exception:
            if task_manifest.status != "failed":
                raise
            stage = task_manifest.current_stage or "unknown"
            console.print(
                f"\n[red]Task failed at {stage}.[/red] Resume with: podtran resume {task_manifest.task_id}"
            )
            raise typer.Exit(code=1)
    console.print(f"[green]Task complete:[/green] {task_manifest.task_id}")
    final_output_path = _compose_output_path(task_manifest, cfg, paths)
    if final_output_path.exists():
        console.print(f"[green]Output file:[/green] {final_output_path.resolve()}")
    return decisions


def _print_task_stage_status(
    task_manifest: TaskManifest,
    cfg: AppConfig,
    paths: ArtifactPaths,
    executor: StageExecutor,
) -> None:
    stage_specs = [
        ("transcribe", {"transcript_json": "transcript.json"}),
        (
            "translate",
            {"segments_json": "segments.json", "translated_json": "translated.json"},
        ),
        ("synthesize", _synthesize_output_refs(cfg)),
        ("compose", _compose_output_refs(task_manifest, cfg)),
    ]

    table = Table(title="podtran status")
    table.add_column("Stage")
    table.add_column("Status")
    table.add_column("Outputs")
    table.add_column("Error")
    for stage, output_refs in stage_specs:
        manifest = executor.load_manifest(stage)
        stage_status = manifest.status if manifest else "pending"
        outputs_present = (
            "yes" if output_refs_exist(paths.task_dir, output_refs) else "no"
        )
        error = _truncate(manifest.error) if manifest and manifest.error else "-"
        table.add_row(stage, stage_status, outputs_present, error)
    console.print(table)


def _require_artifact(path: Path, stage_name: str, artifact_name: str) -> None:
    if path.exists():
        return
    _abort(f"Cannot run {stage_name}: missing {artifact_name}.")


def _require_completed_tts(translated_json: Path) -> None:
    segments = read_model_list(translated_json, SegmentRecord)
    if _all_segments_synthesized(segments):
        return
    _abort("Cannot run compose: not all required synthesized audio is available.")


def _require_translated_segments_ready(translated_json: Path) -> list[SegmentRecord]:
    segments = read_model_list(translated_json, SegmentRecord)
    if _all_segments_translated(segments):
        return segments
    _abort(
        "Cannot run synthesize: translation is incomplete or contains failed segments."
    )


def _abort(message: str) -> None:
    console.print(f"[red]{message}[/red]")
    raise typer.Exit(code=1)


def _validate_speaker_bounds(min_speakers: int, max_speakers: int) -> None:
    if min_speakers > max_speakers:
        raise typer.BadParameter(
            "--min_speakers cannot be greater than --max_speakers."
        )


def _can_resume_partial(
    executor: StageExecutor,
    stage: str,
    stage_version: int,
    input_fingerprints: dict[str, str],
    config_fingerprint: str,
) -> bool:
    """Check if a non-current stage can resume with partial results intact.

    Returns True only when the previous manifest exists with status 'running',
    'interrupted', or 'failed' AND the stage version, input and config
    fingerprints all still match, meaning the partial results are compatible
    with the current run.
    """
    manifest = executor.load_manifest(stage)
    if manifest is None or manifest.status not in ("running", "interrupted", "failed"):
        return False
    if manifest.stage_version != stage_version:
        return False
    if manifest.config_fingerprint != config_fingerprint:
        return False
    for key, value in input_fingerprints.items():
        if manifest.input_fingerprints.get(key) != value:
            return False
    return True


def _transcribe_config_fingerprint(
    fingerprints: FingerprintService,
    cfg: AppConfig,
    min_speakers: int,
    max_speakers: int,
) -> str:
    return fingerprints.hash_value(
        {
            "config": fingerprints.config_subset(cfg, TRANSCRIBE_CONFIG_KEYS),
            "runtime": {
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
            },
        }
    )


def _ensure_transcribe(
    task_manifest: TaskManifest,
    cfg: AppConfig,
    paths: ArtifactPaths,
    executor: StageExecutor,
    cache_store: CacheStore,
    fingerprints: FingerprintService,
    min_speakers: int = DEFAULT_MIN_SPEAKERS,
    max_speakers: int = DEFAULT_MAX_SPEAKERS,
    reporter: PipelineProgressReporter | None = None,
) -> StageDecision:
    from podtran.asr import transcription_stage_count

    audio = Path(task_manifest.processing_audio_path)
    input_fingerprints = {"audio": task_manifest.processing_audio_sha256}
    config_fingerprint = _transcribe_config_fingerprint(
        fingerprints, cfg, min_speakers, max_speakers
    )
    output_refs = {"transcript_json": "transcript.json"}
    cache_key = fingerprints.build_stage_cache_key(
        "transcribe", TRANSCRIBE_STAGE_VERSION, input_fingerprints, config_fingerprint
    )
    current, reason = executor.is_current(
        "transcribe", input_fingerprints, config_fingerprint, output_refs
    )
    manifest = _build_stage_manifest(
        "transcribe",
        TRANSCRIBE_STAGE_VERSION,
        cache_key,
        input_fingerprints,
        config_fingerprint,
        [*TRANSCRIBE_CONFIG_KEYS, "runtime.min_speakers", "runtime.max_speakers"],
        output_refs,
    )
    if current:
        if reporter is not None:
            reporter.skip_stage("transcribe", "up-to-date")
        return StageDecision("transcribe", "up-to-date", "task outputs current")

    entry = cache_store.lookup("transcribe", cache_key)
    if entry is not None:
        cache_store.restore(entry, {"transcript_json": paths.transcript_json})
        executor.save_completed(manifest)
        if reporter is not None:
            reporter.skip_stage("transcribe", "cache hit")
        return StageDecision("transcribe", "cache hit", reason or "cache available")

    executor.start(manifest)
    try:
        ensure_command(FFMPEG_COMMAND)
        ensure_hf_token(cfg.hf_token)
        if reporter is not None:
            reporter.start_stage(
                "transcribe", transcription_stage_count(), "Loading audio"
            )
        transcript = _run_transcription_with_progress(
            audio,
            cfg,
            min_speakers,
            max_speakers,
            progress_callback=(
                lambda completed, total_steps, message: reporter.update_stage(
                    "transcribe", completed, total_steps, message
                )
            )
            if reporter is not None
            else None,
        )
        write_json(paths.transcript_json, transcript)
        executor.complete(manifest)
        cache_store.publish(
            "transcribe",
            cache_key,
            {"transcript_json": paths.transcript_json},
            manifest,
        )
        _release_memory()
        if reporter is not None:
            reporter.complete_stage(
                "transcribe", f"transcribe done: {len(transcript)} transcript segments"
            )
        return StageDecision("transcribe", "run", reason or "cache miss")
    except KeyboardInterrupt:
        if reporter is not None:
            reporter.fail_stage("transcribe", "Interrupted by user")
        executor.interrupt(manifest)
        raise
    except Exception as exc:
        if reporter is not None:
            reporter.fail_stage("transcribe", str(exc))
        executor.fail(manifest, exc)
        raise


def _write_segments(paths: ArtifactPaths, cfg: AppConfig) -> list[SegmentRecord]:
    transcript = read_model_list(paths.transcript_json, TranscriptSegment)
    merged = _build_segments_from_transcript(transcript, cfg)
    write_json(paths.segments_json, merged)
    return merged


def _build_segments_from_transcript(
    transcript: list[TranscriptSegment], cfg: AppConfig
) -> list[SegmentRecord]:
    return merge_transcript_segments(
        transcript,
        pause_threshold=cfg.compose.block_pause_threshold,
        max_block_duration=cfg.compose.max_block_duration,
        configured_voice_map=cfg.tts.preset.voice_map,
        fallback_voices=cfg.tts.preset.fallback_voices,
    )


def _translate_input_fingerprint(
    fingerprints: FingerprintService, segments: list[SegmentRecord]
) -> str:
    return fingerprints.hash_value(
        [
            {
                "segment_id": item.segment_id,
                "block_id": item.block_id,
                "start": item.start,
                "end": item.end,
                "text": item.text,
                "speaker": item.speaker,
                "words": [word.model_dump() for word in item.words],
            }
            for item in segments
        ]
    )


def _sync_translated_output(
    paths: ArtifactPaths, segments: list[SegmentRecord]
) -> None:
    if not paths.translated_json.exists():
        return

    existing = read_model_list(paths.translated_json, SegmentRecord)
    translated_by_id = {item.segment_id: item for item in existing}
    synced = [
        segment.model_copy(
            update={
                "text_zh": previous.text_zh,
                "tts_audio_path": previous.tts_audio_path,
                "tts_duration_ms": previous.tts_duration_ms,
                "status": previous.status,
                "error": previous.error,
            }
        )
        if (previous := translated_by_id.get(segment.segment_id)) is not None
        else segment
        for segment in segments
    ]

    if [item.model_dump() for item in existing] == [
        item.model_dump() for item in synced
    ]:
        return
    write_json(paths.translated_json, synced)


def _ensure_translate(
    task_manifest: TaskManifest,
    cfg: AppConfig,
    paths: ArtifactPaths,
    executor: StageExecutor,
    cache_store: CacheStore,
    fingerprints: FingerprintService,
    reporter: PipelineProgressReporter | None = None,
) -> StageDecision:
    from podtran.translate import Translator

    _ = task_manifest
    segments = _write_segments(paths, cfg)
    input_fingerprints = {
        "segments_json": _translate_input_fingerprint(fingerprints, segments)
    }
    config_fingerprint = fingerprints.hash_config_subset(cfg, TRANSLATE_CONFIG_KEYS)
    output_refs = {
        "translated_json": "translated.json",
        "segments_json": "segments.json",
    }
    cache_key = fingerprints.build_stage_cache_key(
        "translate", TRANSLATE_STAGE_VERSION, input_fingerprints, config_fingerprint
    )
    current, reason = executor.is_current(
        "translate", input_fingerprints, config_fingerprint, output_refs
    )
    manifest = _build_stage_manifest(
        "translate",
        TRANSLATE_STAGE_VERSION,
        cache_key,
        input_fingerprints,
        config_fingerprint,
        TRANSLATE_CONFIG_KEYS,
        output_refs,
    )
    if current:
        _sync_translated_output(paths, segments)
        if reporter is not None:
            reporter.skip_stage("translate", "up-to-date")
        return StageDecision("translate", "up-to-date", "task outputs current")

    entry = cache_store.lookup("translate", cache_key)
    if entry is not None:
        cache_store.restore(entry, {"translated_json": paths.translated_json})
        _sync_translated_output(paths, segments)
        executor.save_completed(manifest)
        if reporter is not None:
            reporter.skip_stage("translate", "cache hit")
        return StageDecision("translate", "cache hit", reason or "cache available")

    resumable = _can_resume_partial(
        executor,
        "translate",
        TRANSLATE_STAGE_VERSION,
        input_fingerprints,
        config_fingerprint,
    )
    executor.start(manifest)
    try:
        if reporter is not None:
            reporter.start_stage(
                "translate", max(len(segments), 1), "Preparing segments"
            )
        if not resumable:
            remove_path(paths.translated_json)
        translated = Translator(cfg).translate_segments(
            paths.segments_json,
            paths.translated_json,
            progress_callback=(
                lambda completed, total_steps, message: reporter.update_stage(
                    "translate", completed, total_steps, message
                )
            )
            if reporter is not None
            else None,
        )
        write_json(paths.translated_json, translated)
        if not _all_segments_translated(translated):
            _print_stage_failure_summary("translate", translated)
            raise RuntimeError("Translation failed for one or more segments.")
        executor.complete(manifest)
        cache_store.publish(
            "translate", cache_key, {"translated_json": paths.translated_json}, manifest
        )
        if reporter is not None:
            reporter.complete_stage("translate", _translate_stage_summary(translated))
        return StageDecision("translate", "run", reason or "cache miss")
    except KeyboardInterrupt:
        if reporter is not None:
            reporter.fail_stage("translate", "Interrupted by user")
        executor.interrupt(manifest)
        raise
    except Exception as exc:
        if reporter is not None:
            reporter.fail_stage("translate", str(exc))
        executor.fail(manifest, exc)
        raise


def _ensure_synthesize(
    task_manifest: TaskManifest,
    cfg: AppConfig,
    paths: ArtifactPaths,
    executor: StageExecutor,
    cache_store: CacheStore,
    fingerprints: FingerprintService,
    reporter: PipelineProgressReporter | None = None,
) -> StageDecision:
    from podtran.tts import synthesize_segments

    translation_segments = _require_translated_segments_ready(paths.translated_json)
    synthesis_input = _synthesize_input_segments(translation_segments)
    input_fingerprints = {"translated_json": fingerprints.hash_json(synthesis_input)}
    config_fingerprint = fingerprints.hash_config_subset(cfg, SYNTHESIZE_CONFIG_KEYS)
    output_refs = _synthesize_output_refs(cfg)
    cache_key = fingerprints.build_stage_cache_key(
        "synthesize", SYNTHESIZE_STAGE_VERSION, input_fingerprints, config_fingerprint
    )
    current, reason = executor.is_current(
        "synthesize", input_fingerprints, config_fingerprint, output_refs
    )
    manifest = _build_stage_manifest(
        "synthesize",
        SYNTHESIZE_STAGE_VERSION,
        cache_key,
        input_fingerprints,
        config_fingerprint,
        SYNTHESIZE_CONFIG_KEYS,
        output_refs,
    )
    if current:
        if reporter is not None:
            reporter.skip_stage("synthesize", "up-to-date")
        return StageDecision("synthesize", "up-to-date", "task outputs current")

    resumable = _can_resume_partial(
        executor,
        "synthesize",
        SYNTHESIZE_STAGE_VERSION,
        input_fingerprints,
        config_fingerprint,
    )
    executor.start(manifest)
    try:
        ensure_command(FFMPEG_COMMAND)
        ensure_command(FFPROBE_COMMAND)
        if reporter is not None:
            reporter.start_stage(
                "synthesize",
                max(len(synthesis_input), 1),
                "Resolving voices",
            )
        if not resumable:
            remove_path(paths.tts_dir)
            remove_path(paths.refs_dir)
            remove_path(paths.voices_json)
            segments = synthesis_input
            write_json(paths.translated_json, segments)
        else:
            segments = translation_segments
        paths.tts_dir.mkdir(parents=True, exist_ok=True)
        paths.refs_dir.mkdir(parents=True, exist_ok=True)
        synthesized = synthesize_segments(
            paths.translated_json,
            paths.translated_json,
            cfg,
            paths,
            source_audio=Path(task_manifest.processing_audio_path),
            source_audio_fingerprint=task_manifest.processing_audio_sha256,
            cache_store=cache_store,
            fingerprints=fingerprints,
            progress_callback=(
                lambda completed, total_steps, message: reporter.update_stage(
                    "synthesize", completed, total_steps, message
                )
            )
            if reporter is not None
            else None,
        )
        write_json(paths.translated_json, synthesized)
        if not _all_segments_synthesized(synthesized):
            _print_stage_failure_summary(
                "tts", _non_skipped_synthesis_failures(synthesized)
            )
            raise RuntimeError("Synthesis failed for one or more segments.")
        _print_synthesis_warning_summary(synthesized)
        executor.complete(manifest)
        if reporter is not None:
            reporter.complete_stage(
                "synthesize", _synthesize_stage_summary(synthesized)
            )
        return StageDecision("synthesize", "run", reason or "stage needs execution")
    except KeyboardInterrupt:
        if reporter is not None:
            reporter.fail_stage("synthesize", "Interrupted by user")
        executor.interrupt(manifest)
        raise
    except Exception as exc:
        if reporter is not None:
            reporter.fail_stage("synthesize", str(exc))
        executor.fail(manifest, exc)
        raise


def _ensure_compose(
    task_manifest: TaskManifest,
    cfg: AppConfig,
    paths: ArtifactPaths,
    executor: StageExecutor,
    fingerprints: FingerprintService,
    reporter: PipelineProgressReporter | None = None,
) -> StageDecision:
    audio = Path(task_manifest.processing_audio_path)
    output_refs = _compose_output_refs(task_manifest, cfg)
    input_fingerprints = {
        "source_audio": task_manifest.processing_audio_sha256,
        "translated_json": fingerprints.hash_json(paths.translated_json),
    }
    config_fingerprint = fingerprints.hash_config_subset(cfg, COMPOSE_CONFIG_KEYS)
    cache_key = fingerprints.build_stage_cache_key(
        "compose", COMPOSE_STAGE_VERSION, input_fingerprints, config_fingerprint
    )
    current, reason = executor.is_current(
        "compose", input_fingerprints, config_fingerprint, output_refs
    )
    manifest = _build_stage_manifest(
        "compose",
        COMPOSE_STAGE_VERSION,
        cache_key,
        input_fingerprints,
        config_fingerprint,
        COMPOSE_CONFIG_KEYS,
        output_refs,
    )
    if current:
        if reporter is not None:
            reporter.skip_stage("compose", "up-to-date")
        return StageDecision("compose", "up-to-date", "task outputs current")

    executor.start(manifest)
    try:
        ensure_command(FFMPEG_COMMAND)
        ensure_command(FFPROBE_COMMAND)
        segments = read_model_list(paths.translated_json, SegmentRecord)
        selected_mode = cfg.compose.mode
        output_path = _compose_output_path(task_manifest, cfg, paths)
        if reporter is not None:
            reporter.start_stage("compose", 1, "Scanning segments")
        compose_output(
            audio,
            segments,
            cfg,
            paths.temp_dir / f"compose_{_compose_output_suffix(cfg)}",
            output_path,
            selected_mode,
            progress_callback=(
                lambda completed, total_steps, message: reporter.update_stage(
                    "compose", completed, total_steps, message
                )
            )
            if reporter is not None
            else None,
        )
        executor.complete(manifest)
        if reporter is not None:
            reporter.complete_stage("compose", f"compose done: {output_path.name}")
        return StageDecision("compose", "run", reason or "stage needs execution")
    except KeyboardInterrupt:
        if reporter is not None:
            reporter.fail_stage("compose", "Interrupted by user")
        executor.interrupt(manifest)
        raise
    except Exception as exc:
        if reporter is not None:
            reporter.fail_stage("compose", str(exc))
        executor.fail(manifest, exc)
        raise


def _build_stage_manifest(
    stage: str,
    stage_version: int,
    cache_key: str,
    input_fingerprints: dict[str, str],
    config_fingerprint: str,
    config_keys: list[str],
    output_refs: dict[str, str],
) -> StageManifest:
    return StageManifest(
        stage=stage,
        status="pending",
        stage_version=stage_version,
        cache_key=cache_key,
        input_fingerprints=input_fingerprints,
        config_fingerprint=config_fingerprint,
        config_keys=config_keys,
        output_refs=output_refs,
    )


def _synthesize_output_refs(cfg: AppConfig) -> dict[str, str]:
    refs = {
        "translated_json": "translated.json",
        "tts_dir": "tts",
        "refs_dir": "refs",
    }
    if cfg.tts.effective_mode(cfg.tts.provider) == "clone":
        refs["voices_json"] = "voices.json"
    return refs


def _compose_output_refs(task_manifest: TaskManifest, cfg: AppConfig) -> dict[str, str]:
    return {
        "final_output": f"final/{_compose_output_filename(task_manifest, _compose_output_suffix(cfg))}"
    }


def _compose_output_filename(task_manifest: TaskManifest, suffix: str) -> str:
    stem = Path(task_manifest.source_audio_name).stem
    if task_manifest.preview:
        stem = f"{stem}.preview"
    return f"{stem}.{suffix}.mp3"


def _compose_output_path(
    task_manifest: TaskManifest, cfg: AppConfig, paths: ArtifactPaths
) -> Path:
    return paths.final_dir / _compose_output_filename(
        task_manifest, _compose_output_suffix(cfg)
    )


def _compose_output_suffix(cfg: AppConfig) -> str:
    return "replace" if cfg.compose.mode.lower() == "replace" else "interleave"


def _task_mode_label(task_manifest: TaskManifest) -> str:
    if not task_manifest.preview:
        return "full"
    return "preview"


def _preview_window_label(task_manifest: TaskManifest) -> str:
    start = f"{task_manifest.preview_start_seconds:g}"
    end = f"{task_manifest.preview_start_seconds + task_manifest.preview_duration_seconds:g}"
    return f"{start}-{end}s"


def _reset_tts_state(segments: list[SegmentRecord]) -> list[SegmentRecord]:
    reset: list[SegmentRecord] = []
    for item in segments:
        reset.append(
            item.model_copy(
                update={
                    "tts_audio_path": "",
                    "tts_duration_ms": 0,
                    "status": "pending",
                    "error": None,
                }
            )
        )
    return reset


def _synthesize_input_segments(segments: list[SegmentRecord]) -> list[SegmentRecord]:
    return _reset_tts_state(segments)


def _run_transcription_with_progress(
    audio: Path,
    cfg: AppConfig,
    min_speakers: int = DEFAULT_MIN_SPEAKERS,
    max_speakers: int = DEFAULT_MAX_SPEAKERS,
    progress_callback=None,
) -> list[TranscriptSegment]:
    from podtran.asr import transcribe_audio

    return transcribe_audio(
        audio,
        cfg.asr,
        cfg.hf_token,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        progress_callback=progress_callback,
    )


def _release_memory() -> None:
    gc.collect()


def _translate_stage_summary(segments: list[SegmentRecord]) -> str:
    translated = sum(1 for item in segments if item.text_zh.strip())
    failed = sum(1 for item in segments if item.error)
    return f"translate done: {translated}/{len(segments)} translated, {failed} failed"


def _synthesize_stage_summary(segments: list[SegmentRecord]) -> str:
    completed = sum(1 for item in segments if item.status == "completed")
    skipped = sum(1 for item in segments if _is_unknown_speaker_tts_skip(item))
    failed = sum(
        1
        for item in segments
        if item.status == "failed" and not _is_unknown_speaker_tts_skip(item)
    )
    return f"synthesize done: {completed}/{len(segments)} audio ready, {skipped} skipped, {failed} failed"


def _all_segments_translated(segments: list[SegmentRecord]) -> bool:
    return bool(segments) and all(item.text_zh.strip() for item in segments)


def _all_segments_synthesized(segments: list[SegmentRecord]) -> bool:
    return bool(segments) and all(
        _segment_has_tts_audio(item) or _is_unknown_speaker_tts_skip(item)
        for item in segments
    )


def _segment_has_tts_audio(segment: SegmentRecord) -> bool:
    return (
        segment.status == "completed"
        and segment.tts_audio_path.strip()
        and Path(segment.tts_audio_path).exists()
    )


def _is_unknown_speaker_tts_skip(segment: SegmentRecord) -> bool:
    return (
        segment.status == "failed"
        and segment.speaker.strip().upper() == UNKNOWN_SPEAKER
        and bool(segment.error)
        and segment.error.startswith(UNKNOWN_SPEAKER_TTS_SKIP_PREFIX)
    )


def _non_skipped_synthesis_failures(
    segments: list[SegmentRecord],
) -> list[SegmentRecord]:
    return [
        item
        for item in segments
        if item.error and not _is_unknown_speaker_tts_skip(item)
    ]


def _print_synthesis_warning_summary(segments: list[SegmentRecord]) -> None:
    skipped = [item for item in segments if _is_unknown_speaker_tts_skip(item)]
    if not skipped:
        return
    console.print(
        f"[yellow]tts warnings:[/yellow] skipped {len(skipped)} UNKNOWN segment(s) without TTS audio."
    )
    messages = {item.error for item in skipped if item.error}
    for message in sorted(messages):
        console.print(f"[yellow]-[/yellow] {_truncate(message)}")


def _print_stage_failure_summary(
    stage: str, segments: list[SegmentRecord], limit: int = 3
) -> None:
    failures = [item for item in segments if item.error]
    if not failures:
        return

    unique_messages: list[str] = []
    seen: set[str] = set()
    for item in failures:
        message = item.error or "Unknown error"
        if message in seen:
            continue
        seen.add(message)
        unique_messages.append(message)

    console.print(f"[yellow]{stage} failures:[/yellow] {len(failures)} segment(s)")
    for message in unique_messages[:limit]:
        console.print(f"[yellow]-[/yellow] {_truncate(message)}")
    if len(unique_messages) > limit:
        console.print(
            f"[yellow]-[/yellow] ... {len(unique_messages) - limit} more unique error(s)"
        )


def _truncate(text: str, limit: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _parse_before(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        parsed = datetime.fromisoformat(f"{value}T00:00:00")
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
