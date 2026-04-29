# AGENTS.md - podtran

## Project Overview

`podtran` is a staged podcast translation CLI for low-memory laptops. It turns
English podcast audio into Chinese audio with resumable stages and reusable
content-addressed artifacts.

Current pipeline:

```text
transcribe -> translate -> synthesize -> compose
```

`merge` is not a CLI stage. Transcript blocks are rebuilt from `transcript.json`
inside translate preparation and written to `segments.json`.

The execution model uses task-isolated directories under a workdir plus a shared
cache under the same workdir. By default the workdir is `~/.podtran`; `--workdir`
moves config, tasks, cache, and indexes together.

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 recommended, package supports `>=3.10,<3.13` |
| Package manager | `uv` |
| Build backend | Hatchling, src-layout |
| CLI | Typer + Rich |
| Config | TOML via `tomllib` / `tomli`, Pydantic v2 models |
| ASR | WhisperX with alignment and speaker diarization |
| Translation | `google-free` by default; `openai-compatible` for LLM endpoints |
| TTS | `qwen-local` by default; also `dashscope`, `openai-compatible`, `vllm-omni` |
| Audio | external `ffmpeg` / `ffprobe` |
| HTTP / SDK | `httpx`, `openai` SDK |
| Retry | `tenacity` |
| Tests / lint | `pytest`, `ruff` |

`qwen-local` is optional in packaging (`podtran[qwen-local]`) but is the default
runtime TTS provider. Development tests do not require real WhisperX, DashScope,
Google, vLLM, ffmpeg, or qwen-local service calls.

## Repository Layout

```text
podtran/
├── src/podtran/
│   ├── cli.py              # Typer commands, root AUDIO shortcut, pipeline orchestration
│   ├── config.py           # Pydantic config models, TOML loading/rendering, legacy detection
│   ├── models.py           # Task, stage, segment, voice, and voice-spec models
│   ├── asr.py              # WhisperX transcription
│   ├── merge.py            # Transcript -> block SegmentRecord aggregation
│   ├── translate.py        # google-free and OpenAI-compatible translation backends
│   ├── tts.py              # TTS backends, segment synthesis, per-segment TTS cache
│   ├── voices.py           # Clone reference selection, voice resolver, voice profile cache
│   ├── compose.py          # Final audio composition
│   ├── artifacts.py        # Artifact paths, JSON I/O, atomic writes, copy/remove helpers
│   ├── tasks.py            # TaskStore under artifacts/tasks/<task_id>/
│   ├── cache_store.py      # Shared content-addressed cache
│   ├── fingerprints.py     # Stage/config hashing and config key lists
│   ├── stage_executor.py   # Stage lifecycle and manifest persistence
│   ├── stage_versions.py   # Cache invalidation version constants
│   ├── audio.py            # ffmpeg/ffprobe helpers
│   ├── checks.py           # Preflight checks
│   ├── __init__.py         # Package version
│   └── __main__.py         # python -m podtran entry
├── tests/                  # Unit tests with mocked external services
├── scripts/                # Local helper scripts
├── docs/                   # Local design docs, gitignored
├── main.py                 # Dev entry point that adds src/ to sys.path
├── pyproject.toml
└── uv.lock
```

## Development Commands

Install normal dev dependencies:

```powershell
uv sync
```

Install default local TTS dependencies for actual `qwen-local` runs:

```powershell
uv sync --extra qwen-local
```

Run the CLI:

```powershell
uv run podtran --help
uv run podtran run path\to\episode.mp3 --preview
uv run podtran path\to\episode.mp3
uv run podtran resume
uv run podtran tasks
uv run podtran status
uv run podtran version
```

Run tests and lint:

```powershell
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```

CI uses Python 3.11, installs the package with minimal non-heavy dependencies,
runs `uv run ruff check src tests`, then `uv run pytest -q`.

## CLI Behavior

- `podtran run AUDIO` is the explicit entry point.
- `podtran AUDIO` is a shortcut: `main()` rewrites root audio invocations to
  `run` when the first non-option token is not a known command.
- `--preview` extracts the first 300 seconds to `preview.wav` and creates a
  preview task.
- `--min_speakers` and `--max_speakers` default to 2 and 5 and are passed to
  WhisperX diarization.
- `resume [TASK]` loads a task id or unique prefix, defaulting to the latest
  task.
- Single-stage commands are `transcribe`, `translate`, `synthesize`, and
  `compose`; each requires an existing task and does not auto-run prerequisites.
- `cache clean [--before ...]` is the only public cache maintenance command.
  Do not add cache list/inspect commands unless the CLI scope intentionally
  changes.

## Artifact Model

Each task lives under:

```text
<workdir>/artifacts/tasks/<task_id>/
  task.json
  transcript.json
  segments.json
  translated.json
  preview.wav
  voices.json
  refs/
  tts/
  final/
  manifests/
  tmp/
```

Shared cache lives under:

```text
<workdir>/artifacts/cache/
  _indexes/audio_hashes.json
  transcribe/<cache_key>/
  translate/<cache_key>/
  voice_clone/<cache_key>/
  tts/<cache_key>/
```

`TaskStore` resolves exact task ids and unique prefixes. Task ids are UTC
timestamps plus the first 6 chars of the source audio hash.

## Stage And Cache Rules

`StageExecutor` owns stage manifests and task status. Stages follow this pattern:

```text
is_current? -> shared cache lookup where applicable -> start -> execute -> complete/fail/interrupt
```

Important boundaries:

- Transcribe and translate use shared stage-level cache entries.
- Synthesize does not publish one stage-level cache entry; it reuses voice-clone
  cache entries and per-segment TTS cache entries.
- Compose records a stage manifest for current/stale checks, but it does not
  publish a shared compose cache entry.
- `segments.json` is intentionally regenerated before translate so merge config
  changes take effect.
- Translate and synthesize can preserve compatible partial results after
  `failed` or `interrupted` manifests when stage version, input fingerprints,
  and config fingerprints still match.
- Unknown speaker segments in clone mode are skipped with a recorded segment
  failure and warning; compose is allowed to continue when required TTS output is
  otherwise complete.

When changing output-affecting behavior:

- Add or update the relevant dotted config keys in `fingerprints.py`.
- Bump the affected constant in `stage_versions.py`.
- If serialized `VoiceSpec` or clone profile semantics change, bump the clone
  and TTS-related versions.
- Do not include API keys or pure scheduling knobs in fingerprints unless they
  actually change deterministic output. Current tests assert that provider API
  keys and TTS concurrency/batch scheduling do not affect TTS fingerprints.

## Config Model

Default config path:

```text
~/.podtran/config.toml
```

`AppConfig` contains:

- top-level `hf_token`
- `[providers.dashscope]`
- `[providers.openai_compatible]`
- `[providers.vllm_omni]`
- `[providers.qwen_local]`
- `[asr]`
- `[translation]`
- `[tts]`, `[tts.preset]`, `[tts.preset.voice_map]`, `[tts.clone]`
- `[compose]`

Defaults:

- Translation provider: `google-free`
- OpenAI-compatible translation default base URL: DashScope compatible-mode
- OpenAI-compatible translation default model: `qwen-flash`
- TTS provider: `qwen-local`
- TTS mode: `auto`
- `auto` mode resolves to `preset` only for `openai-compatible`; it resolves to
  `clone` for `qwen-local`, `dashscope`, and `vllm-omni`.
- ASR defaults: `model = "medium"`, `device = "cpu"`, `compute_type = "int8"`,
  `batch_size = 4`.

Legacy config is intentionally rejected by `load_config()`. `podtran init` can
rebuild legacy TTS config, write `config.toml.bak`, and preserve `hf_token` plus
provider API keys. Old translation fields such as `translation.base_url`,
`translation.model`, and `translation.provider = "dashscope"` are legacy; use
`translation.provider = "openai-compatible"` plus
`[providers.openai_compatible]` instead.

Never commit real `config.toml` contents or API keys.

## Provider Notes

Translation providers:

- `google-free`: default, no API key, uses an unofficial Google Translate web
  endpoint and can be affected by network/rate limits.
- `openai-compatible`: uses Chat Completions via the OpenAI SDK. DashScope
  translation is configured through this provider and its compatible-mode URL.

TTS providers:

- `qwen-local`: default, supports `preset` and `clone`, uses local Qwen3-TTS
  models through the optional `qwen-local` extra, single worker, batches matching
  same-voice work items.
- `dashscope`: supports `preset` and server-managed `provider_clone` voices.
- `openai-compatible`: supports `preset` only.
- `vllm-omni`: supports `preset` and `reference_clone` through
  `/audio/speech`.

Internal voice spec kinds:

- `preset`
- `provider_clone`
- `reference_clone`

`qwen-local` and `vllm-omni` use `reference_clone`; `dashscope` uses
`provider_clone`.

## Coding Conventions

1. Use `from __future__ import annotations` in Python modules.
2. Keep domain state in Pydantic v2 models. Prefer `model_dump()` and
   `model_copy(update={...})` over raw dict mutation for domain data.
3. Write JSON and binary artifacts atomically through helpers in `artifacts.py`.
4. Lazy-import heavy optional/runtime dependencies inside functions or backend
   methods. `whisperx`, `qwen_tts`, `torch`, and `soundfile` should not be loaded
   by CLI help.
5. Route runtime state through function args, `TaskStore`, `CacheStore`,
   `FingerprintService`, and `ArtifactPaths`. Do not introduce global mutable
   state or singletons.
6. Let `StageExecutor.fail()` and `StageExecutor.interrupt()` record stage
   errors. Segment-level translation/TTS failures belong on `SegmentRecord.error`.
7. Preserve resumability: do not delete `translated.json`, `tts/`, `refs/`, or
   `voices.json` unless the current implementation has determined the old partial
   output is not resumable.
8. Keep CLI commands using `_load_runtime()` or `_load_task_context()` for
   consistent config/workdir/task/cache setup.
9. Prefer structured parsing and Pydantic validation over ad hoc string handling.
10. Keep tests pure Python and isolated with `tmp_path`; mock external services,
    ffmpeg/ffprobe, model downloads, and network calls.

## Testing Guidance

Add or update focused tests when changing:

- CLI help, command routing, stage prerequisites, resume behavior, or status
  output: `tests/test_cli.py`
- Config shape, default values, legacy detection, config rendering, fingerprints:
  `tests/test_config.py`
- Translation backends and response parsing: `tests/test_translate.py`
- TTS backends, cache behavior, concurrency, qwen-local batching, provider
  routing: `tests/test_tts.py`
- Voice clone resolution, reference selection, profile reuse: `tests/test_voices.py`
- Audio helpers and compose behavior: `tests/test_audio.py`,
  `tests/test_compose.py`
- Task and cache store behavior: `tests/test_tasks.py`,
  `tests/test_cache_store.py`

Run the smallest relevant pytest file first, then `uv run pytest -q` before
finishing broad changes.

## Release Notes

The user-facing changelog is `CHANGELOG.md` and is written in Chinese. For a
release, update the changelog, bump `src/podtran/__init__.py`, run lint/tests,
commit, and tag. There is a local `podtran-release` Codex skill for this
workflow.

## Gotchas

- `config.toml` is outside the repo by default and can contain secrets.
- `--workdir` changes where config, artifacts, tasks, cache, and cache indexes
  live; use it heavily in tests.
- WhisperX diarization needs a Hugging Face token and acceptance of the
  `pyannote/speaker-diarization-community-1` model terms.
- `ffmpeg` and `ffprobe` must be on `PATH` before preview, synthesize validation,
  or compose operations.
- `translate` requires `transcript.json`; `synthesize` requires complete
  translations; `compose` requires complete required TTS output.
- `qwen-local` model sizes are currently `0.6B` and `1.7B`.
- For long audio, WhisperX can load the whole file into memory; pre-splitting is
  still the practical fallback on constrained machines.
- `docs/` and `.codex/` are gitignored local working areas.
