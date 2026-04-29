# AGENT.md — podtran

## Project Overview

**podtran** is a staged podcast translation CLI designed for low-memory laptops (16 GB RAM).
It translates English podcasts into Chinese using a multi-stage pipeline:
`transcribe → merge → translate → synthesize → compose`.

The execution model uses **task-isolated directories** with a **content-addressable shared cache**,
ensuring reproducibility and efficient reuse across tasks.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 (supports `>=3.10,<3.13`) |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| Build backend | Hatchling |
| CLI framework | Typer + Rich |
| Config | TOML (`config.toml`), parsed via `tomllib` / `tomli` |
| Data models | Pydantic v2 (`BaseModel`, `model_dump`, `model_copy`) |
| ASR | WhisperX (word-level alignment + speaker diarization) |
| Translation | OpenAI-compatible LLM API (default: Qwen3 via DashScope) |
| TTS | qwen-local Qwen3 TTS, DashScope Qwen3 TTS, OpenAI-compatible TTS |
| Audio toolchain | ffmpeg / ffprobe (external binaries) |
| HTTP client | httpx (TTS), openai SDK (translation, OpenAI-compat TTS) |
| Retry | tenacity |
| Linter | Ruff |
| Tests | pytest |

## Repository Layout

```text
podtran/
├── src/podtran/          # Main package (hatchling src-layout)
│   ├── cli.py            # Typer CLI: root AUDIO entry, resume, init, tasks, status, stages, cache clean
│   ├── config.py         # Pydantic config models, TOML loader, render_config_toml
│   ├── models.py         # Domain models: TaskManifest, StageManifest, SegmentRecord, VoiceProfile, etc.
│   ├── asr.py            # WhisperX transcription with progress callbacks
│   ├── merge.py          # Transcript → block-aggregated segments
│   ├── translate.py      # Batch translation via OpenAI-compatible API
│   ├── tts.py            # TTS backends (DashScope, OpenAI-compatible), backend dispatch, segment-level synthesis + cache
│   ├── voices.py         # Voice clone: reference audio selection, DashScope enrollment, VoiceResolver, clone profile/cache handling
│   ├── compose.py        # Final audio compositing (interleave / replace modes)
│   ├── artifacts.py      # ArtifactPaths helper, JSON I/O, atomic writes, copy/remove utilities
│   ├── tasks.py           # TaskStore: create/load/list/save task manifests under artifacts/tasks/<task_id>/
│   ├── cache_store.py    # CacheStore: content-addressable shared cache under artifacts/cache/
│   ├── fingerprints.py   # FingerprintService: SHA-256 hashing, config-subset hashing, cache key generation
│   ├── stage_executor.py # StageExecutor: orchestrates stage lifecycle (start, complete, fail, is_current)
│   ├── stage_versions.py # Stage version constants for cache invalidation
│   ├── audio.py          # Audio utilities: ffprobe duration, ffmpeg extract/convert
│   └── checks.py         # Pre-flight checks: audio file, ffmpeg, hf_token
│   ├── __init__.py       # Package version
│   └── __main__.py       # python -m podtran entry
├── tests/                # pytest test suite
├── docs/                 # Design docs (gitignored, local only)
├── main.py               # Dev entry point (adds src/ to sys.path, calls cli:main)
├── pyproject.toml        # Project metadata, dependencies, scripts, dev tools
└── .python-version       # Pinned to 3.11
```

## Development Environment

### Setup

```powershell
uv sync
```

For CPU-only PyTorch (recommended for 16 GB RAM):

```powershell
uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio
uv sync
```

### Run the CLI

```powershell
uv run podtran --help
uv run podtran path\to\podcast.mp3
uv run podtran resume
uv run podtran tasks
uv run podtran status
```

### Run Tests

```powershell
uv run pytest
```

Tests in `tests/` are pure-Python unit tests. They mock external services
(WhisperX, DashScope API, ffmpeg) and use `tmp_path` for filesystem isolation.

### Lint

```powershell
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

## Architecture & Key Patterns

### Execution Model

- **`podtran <audio>`** → creates a new task under `~/.podtran/artifacts/tasks/<task_id>/`,
  then runs the full pipeline `transcribe → merge → translate → synthesize → compose`.
- **`podtran resume [task]`** → loads an existing task (defaults to latest) and re-runs the pipeline
  from where it left off. Completed stages are skipped; interrupted translations resume from partial results.
- **`podtran tasks`** → lists recent task instances.
- **`podtran status [task]`** → shows the current facts for a task; defaults to the latest task.
- **Single-stage commands** (`transcribe`, `translate`, `synthesize`, `compose`)
  require `TASK` as a positional argument and execute only that stage.
- **`podtran cache clean`** → removes shared cache entries; no other cache inspection commands are exposed.

### Task Isolation

Each task gets its own directory:

```text
~/.podtran/artifacts/tasks/<task_id>/
  task.json          # TaskManifest (audio path, config snapshot, status)
  transcript.json   # WhisperX output
  segments.json     # Merged blocks
  translated.json   # Translations + TTS status per segment
  preview.wav       # Preview mode extracted audio clip
  voices.json       # Resolved voice profiles (clone mode)
  refs/             # Reference audio clips (clone mode)
  tts/              # Per-segment synthesized WAV files
  final/            # Final composed MP3
  manifests/        # Per-stage StageManifest JSON files
  tmp/              # Temporary build files (compose chunks, voice refs)
```

### Content-Addressable Cache

Shared cache lives in `~/.podtran/artifacts/cache/<stage>/<cache_key>/`.
Cache keys are SHA-256 hashes of `{stage, stage_version, input_fingerprints, config_fingerprint}`.

- **Fingerprinting**: `FingerprintService` hashes audio files, JSON structures, and config subsets.
- **Stage versions**: `stage_versions.py` constants; bump to force cache invalidation when stage logic changes.
- **Cache flow**: `lookup → restore` on hit; `publish` after successful execution.
- Shared cache is user-visible only through `podtran cache clean`; list/inspect operations are intentionally not part of the simplified CLI.

### Stage Lifecycle (`StageExecutor`)

Each stage follows: `is_current? → cache hit? → start → <execute> → complete/fail/interrupt`.
`StageManifest` records `input_fingerprints`, `config_fingerprint`, and `output_refs`
for deterministic staleness detection.

- **`KeyboardInterrupt` (Ctrl+C)** is caught per-stage and calls `executor.interrupt(manifest)`,
  setting status to `"interrupted"` before exiting. The pipeline prints a resume hint.
- **Partial result preservation**: `_can_resume_partial()` checks the old manifest's status,
  `stage_version`, input fingerprints, and config fingerprint to decide whether to keep
  existing partial output (e.g. `translated.json`) on resume.

### Status Reporting

`status` reports the current task state from on-disk manifests and output presence.
It does not perform dry-run planning or predict reruns.

### Config Structure

`AppConfig` (Pydantic) is loaded from `~/.podtran/config.toml`:
- `[providers.dashscope]` → `ProviderConfig` (API key)
- `[translation]` → `TranslationConfig`
- `[tts]` → `TTSConfig` (supports `preset` and `clone` voice modes)
- `[asr]` → `ASRConfig`
- `[compose]` → `ComposeConfig`
- All config sub-models use `extra="ignore"` for forward-compatible TOML parsing.
- Use `podtran init` to generate a template config interactively.

### TTS Modes

| Mode | Model | Behavior |
|---|---|---|
| `preset` | `qwen3-tts-flash` or provider-specific preset model | Uses `voice_map` + `fallback_voices`; supported by DashScope and explicit `openai_compatible` TTS providers |
| `clone` | Provider-specific clone model | Extracts reference audio, resolves a provider-specific clone asset, then caches the resolved voice profile |

### TTS Provider Routing

- **TTS provider routing is explicit** — `tts.provider = "dashscope"` selects DashScope; `tts.provider = "openai_compatible"` selects the OpenAI-compatible backend.
- **Unknown TTS provider names are rejected** — non-DashScope names no longer implicitly fall back to the OpenAI-compatible backend.
- **Clone support is backend-specific** — qwen-local and DashScope support production `clone` mode today.
- **Internal clone asset kinds** use `VoiceSpec` variants:
  - `preset`
  - `provider_clone` — provider-managed reusable clone asset (used by DashScope enrollment today)
  - `reference_clone` — reserved for backends that can synthesize directly from reference audio/text

### Clone Persistence Compatibility

- Clone-related cache validity depends on the serialized `VoiceSpec` shape and stage versions.
- When changing clone asset semantics or serialized `VoiceSpec` structure, bump the relevant stage versions in `stage_versions.py`.

### Memory Management

- WhisperX models are loaded sequentially and explicitly `del`-ed + `gc.collect()`-ed between stages
  to stay within 16 GB RAM.
- Default `asr.batch_size = 4` keeps memory usage low on CPU.

## Coding Conventions

1. **Type hints everywhere** — use `from __future__ import annotations` for deferred evaluation.
2. **Pydantic v2 models** — all domain data is typed via `BaseModel`.
   Use `model_dump()`, `model_copy(update={...})`, never raw dicts for domain objects.
3. **Atomic writes** — all JSON output uses `atomic_write_text` / `atomic_write_bytes`
   to prevent partial writes on crash.
4. **Imports** — heavy dependencies (`whisperx`) are lazy-imported inside functions
   to avoid loading CUDA/PyTorch at CLI parse time. `openai` and `httpx` are imported at module level.
5. **Config keys** — when adding a config field that affects a stage's output,
   add its dotted key to the corresponding `*_CONFIG_KEYS` list in `fingerprints.py`
   and bump the stage version in `stage_versions.py`.
6. **Error handling** — stages catch exceptions via `StageExecutor.fail()` which
   records the error in the manifest and updates task status. `KeyboardInterrupt` is
   caught separately via `StageExecutor.interrupt()` to set `"interrupted"` status.
   Individual segment-level failures (translate, TTS) are recorded per-segment in `SegmentRecord.error`.
7. **CLI pattern** — commands that need runtime state load it via `_load_runtime()` or `_load_task_context()`.
8. **No globals / singletons** — all state flows through function args or dataclass instances.

## Important Gotchas

- **`config.toml` lives at `~/.podtran/config.toml`** — outside the project repo. It contains API keys. Use `podtran init` to generate it.
- **`--workdir` overrides all paths** — config, artifacts, tasks, and cache all resolve relative to the workdir. Use for testing only.
- **`merge` is not a cached stage** — it re-derives `segments.json` from `transcript.json`
  on every run before `translate`, so changes to merge config take effect immediately.
- **`translate` only runs when `transcript.json` exists** — stage commands do not auto-run prerequisites.
- **`synthesize` does not use shared cache at the stage level** — only per-segment TTS
  results are cached in `artifacts/cache/tts/`.
- **Clone cache compatibility is versioned by serialized `VoiceSpec` shape** — renaming clone kinds or changing clone payload structure invalidates prior TTS / voice-clone cache entries.
- **`compose` only runs when there is at least one completed TTS output** — it does not auto-run synthesize.
- **WhisperX requires HuggingFace token** — you must accept the `speaker-diarization-community-1`
  model agreement on HuggingFace before diarization will work.
- **ffmpeg and ffprobe must be on PATH** — the CLI checks this before any audio operations.
- **Audio longer than 1 hour** — WhisperX loads the entire file into memory;
  consider pre-splitting with ffmpeg.


