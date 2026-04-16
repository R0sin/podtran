from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Protocol

import httpx
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from podtran.artifacts import ArtifactPaths, atomic_write_bytes, copy_path, read_model_list, write_json
from podtran.audio import FFPROBE_COMMAND, probe_duration
from podtran.cache_store import CacheStore
from podtran.config import AppConfig, DEFAULT_TTS_PRESET_MODEL
from podtran.fingerprints import FingerprintService, TTS_CONFIG_KEYS, normalize_text
from podtran.models import ResolvedVoiceTarget, SegmentRecord, StageManifest, StageProgressCallback
from podtran.stage_versions import TTS_STAGE_VERSION
from podtran.voices import VoiceProfileManager, build_preset_targets

TTS_RETRY_ATTEMPTS = 3


class TTSBackend(Protocol):
    def synthesize(self, text: str, target: ResolvedVoiceTarget, output_path: Path) -> None:
        ...


class DashScopeTTSBackend:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = httpx.Client(timeout=config.tts.timeout_seconds)

    @retry(
        reraise=True,
        stop=stop_after_attempt(TTS_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((RuntimeError, httpx.HTTPError)),
    )
    def synthesize(self, text: str, target: ResolvedVoiceTarget, output_path: Path) -> None:
        response = self.client.post(
            f"{self.config.tts.resolved_base_url()}/services/aigc/multimodal-generation/generation",
            headers={"Authorization": f"Bearer {_resolve_tts_key(self.config)}"},
            json={
                "model": _resolve_tts_model(self.config, target),
                "input": {
                    "text": text,
                    "voice": target.voice,
                    "language_type": self.config.tts.language_type,
                },
            },
        )
        response.raise_for_status()
        payload = response.json()
        audio_url = payload.get("output", {}).get("audio", {}).get("url", "")
        if not audio_url:
            raise RuntimeError(f"DashScope TTS returned no audio URL: {json.dumps(payload, ensure_ascii=False)}")
        audio_response = self.client.get(audio_url)
        audio_response.raise_for_status()
        atomic_write_bytes(output_path, audio_response.content)


class OpenAICompatibleTTSBackend:
    def __init__(self, config: AppConfig) -> None:
        self.client = OpenAI(
            api_key=_resolve_tts_key(config),
            base_url=config.tts.resolved_base_url(),
            timeout=config.tts.timeout_seconds,
        )
        self.config = config

    @retry(
        reraise=True,
        stop=stop_after_attempt(TTS_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(RuntimeError),
    )
    def synthesize(self, text: str, target: ResolvedVoiceTarget, output_path: Path) -> None:
        response = self.client.audio.speech.create(
            model=_resolve_tts_model(self.config, target),
            voice=target.voice,
            input=text,
            response_format="wav",
        )
        payload = _read_binary_response(response)
        atomic_write_bytes(output_path, payload)


@dataclass(slots=True)
class _PendingSegment:
    segment_index: int
    audio_path: Path


@dataclass(slots=True)
class _SynthesisWorkItem:
    text_zh: str
    target: ResolvedVoiceTarget
    output_path: Path
    cache_key: str | None
    segments: list[_PendingSegment] = field(default_factory=list)


@dataclass(slots=True)
class _SynthesisResult:
    output_path: Path
    duration_ms: int


def synthesize_segments(
    input_path: Path,
    output_path: Path,
    config: AppConfig,
    paths: ArtifactPaths,
    source_audio: Path | None = None,
    source_audio_fingerprint: str | None = None,
    cache_store: CacheStore | None = None,
    fingerprints: FingerprintService | None = None,
    progress_callback: StageProgressCallback | None = None,
) -> list[SegmentRecord]:
    segments = _load_resume_segments(input_path, output_path)
    paths.tts_dir.mkdir(parents=True, exist_ok=True)

    speaker_units = _speaker_progress_units(config, segments)
    segment_units = len(segments)
    stage_total = max(speaker_units + segment_units, 1)
    segment_offset = speaker_units

    if progress_callback is not None:
        progress_callback(0, stage_total, "Resolving voices")

    voice_targets = _resolve_voice_targets(
        config,
        paths,
        segments,
        source_audio,
        source_audio_fingerprint,
        cache_store,
        fingerprints,
        progress_callback=(
            lambda completed, total, message: progress_callback(min(completed, speaker_units), stage_total, message)
            if progress_callback is not None
            else None
        ),
    )
    if progress_callback is not None and speaker_units == 1 and config.tts.voice_mode.strip().lower() != "clone":
        progress_callback(1, stage_total, "Using preset voices")

    tts_config_fingerprint = fingerprints.hash_config_subset(config, TTS_CONFIG_KEYS) if fingerprints else ""
    processed_segments = 0
    work_items: dict[str, _SynthesisWorkItem] = {}

    for index, segment in enumerate(segments):
        audio_path = _segment_audio_path(paths, segment)
        if segment.tts_audio_path and Path(segment.tts_audio_path).exists() and segment.status == "completed":
            processed_segments += 1
            _emit_segment_progress(progress_callback, segment_offset + processed_segments, stage_total, processed_segments, segment_units, "Reusing audio")
            continue
        if audio_path.exists() and audio_path.stat().st_size > 0:
            _mark_segment_completed(segment, audio_path, int(probe_duration(FFPROBE_COMMAND, audio_path) * 1000))
            write_json(output_path, segments)
            processed_segments += 1
            _emit_segment_progress(progress_callback, segment_offset + processed_segments, stage_total, processed_segments, segment_units, "Reusing audio")
            continue
        if not segment.text_zh.strip():
            segment.status = "failed"
            segment.error = "Missing translated text."
            write_json(output_path, segments)
            processed_segments += 1
            _emit_segment_progress(progress_callback, segment_offset + processed_segments, stage_total, processed_segments, segment_units, "Skipping segment")
            continue

        target = voice_targets.get(segment.speaker)
        if target is None:
            if config.tts.voice_mode.strip().lower() == "clone":
                raise RuntimeError(f"Missing resolved clone voice target for {segment.speaker}.")
            target = ResolvedVoiceTarget(
                speaker=segment.speaker,
                mode="preset",
                voice=segment.voice,
            )

        cache_key = _tts_cache_key(segment, target, config, fingerprints)
        if cache_key and cache_store:
            entry = cache_store.lookup("tts", cache_key)
            if entry is not None:
                cache_store.restore(entry, {"audio": audio_path})
                _mark_segment_completed(segment, audio_path, int(probe_duration(FFPROBE_COMMAND, audio_path) * 1000))
                write_json(output_path, segments)
                processed_segments += 1
                _emit_segment_progress(progress_callback, segment_offset + processed_segments, stage_total, processed_segments, segment_units, "Restored cached audio")
                continue

        work_key = _tts_work_key(segment, target, config)
        work_item = work_items.get(work_key)
        if work_item is None:
            work_item = _SynthesisWorkItem(
                text_zh=segment.text_zh,
                target=target,
                output_path=audio_path,
                cache_key=cache_key,
            )
            work_items[work_key] = work_item
        work_item.segments.append(_PendingSegment(segment_index=index, audio_path=audio_path))

    if work_items:
        max_workers = max(1, config.tts.max_concurrency)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_item = {
                executor.submit(_synthesize_work_item, config, item): item
                for item in work_items.values()
            }
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    if item.cache_key and cache_store and fingerprints:
                        cache_store.publish(
                            "tts",
                            item.cache_key,
                            {"audio": result.output_path},
                            _build_tts_manifest(item.text_zh, item.target, item.cache_key, tts_config_fingerprint, fingerprints),
                        )
                    for pending in item.segments:
                        destination = pending.audio_path
                        if destination.resolve() != result.output_path.resolve():
                            copy_path(result.output_path, destination)
                        _mark_segment_completed(segments[pending.segment_index], destination, result.duration_ms)
                        processed_segments += 1
                except Exception as exc:
                    for pending in item.segments:
                        segment = segments[pending.segment_index]
                        segment.status = "failed"
                        segment.error = str(exc)
                        processed_segments += 1
                write_json(output_path, segments)
                _emit_segment_progress(
                    progress_callback,
                    segment_offset + processed_segments,
                    stage_total,
                    processed_segments,
                    segment_units,
                    "Synthesizing audio",
                )

    if progress_callback is not None:
        progress_callback(stage_total, stage_total, "Synthesis complete")
    return segments


def build_tts_backend(config: AppConfig) -> TTSBackend:
    backend = config.tts.resolved_backend()
    if config.tts.voice_mode.strip().lower() == "clone" and backend != "dashscope":
        raise RuntimeError(f"Clone mode is not supported for TTS provider: {config.tts.provider}")
    if backend == "dashscope":
        return DashScopeTTSBackend(config)
    if backend == "openai_compatible":
        return OpenAICompatibleTTSBackend(config)
    raise RuntimeError(f"Unsupported TTS provider: {config.tts.provider}")


def _load_resume_segments(input_path: Path, output_path: Path) -> list[SegmentRecord]:
    source = output_path if output_path.exists() else input_path
    return read_model_list(source, SegmentRecord)


def _resolve_voice_targets(
    config: AppConfig,
    paths: ArtifactPaths,
    segments: list[SegmentRecord],
    source_audio: Path | None,
    source_audio_fingerprint: str | None,
    cache_store: CacheStore | None,
    fingerprints: FingerprintService | None,
    progress_callback: StageProgressCallback | None = None,
) -> dict[str, ResolvedVoiceTarget]:
    if config.tts.voice_mode.strip().lower() != "clone":
        return build_preset_targets(segments)
    if source_audio is None:
        raise RuntimeError("Clone mode requires source audio.")
    resolved_source_audio = source_audio.resolve()
    if not resolved_source_audio.exists():
        raise RuntimeError(f"Source audio not found for clone mode: {resolved_source_audio}")
    return VoiceProfileManager(config, paths, cache_store=cache_store, fingerprints=fingerprints).resolve_voice_targets(
        segments,
        resolved_source_audio,
        source_audio_fingerprint=source_audio_fingerprint,
        progress_callback=progress_callback,
    )


def _speaker_progress_units(config: AppConfig, segments: list[SegmentRecord]) -> int:
    speakers = {segment.speaker for segment in segments}
    if not speakers:
        return 0
    if config.tts.voice_mode.strip().lower() == "clone":
        return len(speakers)
    return 1


def _emit_segment_progress(
    progress_callback: StageProgressCallback | None,
    completed: int,
    stage_total: int,
    processed_segments: int,
    segment_units: int,
    action: str,
) -> None:
    if progress_callback is None:
        return
    total_segments = max(segment_units, 1)
    progress_callback(completed, stage_total, f"{action} {processed_segments}/{total_segments}")


def _mark_segment_completed(segment: SegmentRecord, audio_path: Path, duration_ms: int) -> None:
    segment.tts_audio_path = str(audio_path.resolve())
    segment.tts_duration_ms = duration_ms
    segment.status = "completed"
    segment.error = None


def _segment_audio_path(paths: ArtifactPaths, segment: SegmentRecord) -> Path:
    return paths.tts_dir / f"{segment.segment_id}_{segment.speaker}.wav"


def _tts_work_key(segment: SegmentRecord, target: ResolvedVoiceTarget, config: AppConfig) -> str:
    return json.dumps(
        {
            "text_zh": normalize_text(segment.text_zh),
            "voice_target": target.model_dump(),
            "model": _resolve_tts_model(config, target),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _synthesize_work_item(config: AppConfig, item: _SynthesisWorkItem) -> _SynthesisResult:
    backend = build_tts_backend(config)
    backend.synthesize(item.text_zh, item.target, item.output_path)
    duration_ms = int(probe_duration(FFPROBE_COMMAND, item.output_path) * 1000)
    return _SynthesisResult(output_path=item.output_path, duration_ms=duration_ms)


def _build_tts_manifest(
    text_zh: str,
    target: ResolvedVoiceTarget,
    cache_key: str,
    config_fingerprint: str,
    fingerprints: FingerprintService,
) -> StageManifest:
    return StageManifest(
        stage="tts",
        status="completed",
        stage_version=TTS_STAGE_VERSION,
        cache_key=cache_key,
        input_fingerprints={
            "text_zh": fingerprints.hash_value(normalize_text(text_zh)),
            "voice_target": fingerprints.hash_value(target.model_dump()),
        },
        config_fingerprint=config_fingerprint,
        config_keys=TTS_CONFIG_KEYS,
    )


def _tts_cache_key(
    segment: SegmentRecord,
    target: ResolvedVoiceTarget,
    config: AppConfig,
    fingerprints: FingerprintService | None,
) -> str | None:
    if fingerprints is None:
        return None
    config_fingerprint = fingerprints.hash_config_subset(config, TTS_CONFIG_KEYS)
    return fingerprints.build_stage_cache_key(
        "tts",
        TTS_STAGE_VERSION,
        {
            "text_zh": fingerprints.hash_value(normalize_text(segment.text_zh)),
            "voice_target": fingerprints.hash_value(
                {
                    "target": target.model_dump(),
                    "model": _resolve_tts_model(config, target),
                }
            ),
        },
        config_fingerprint,
    )


def _read_binary_response(response: object) -> bytes:
    if hasattr(response, "read"):
        data = response.read()
        if isinstance(data, bytes):
            return data
    if hasattr(response, "content"):
        content = getattr(response, "content")
        if isinstance(content, bytes):
            return content
    raise RuntimeError("Unsupported binary response from TTS backend.")


def _resolve_tts_model(config: AppConfig, target: ResolvedVoiceTarget) -> str:
    if target.mode == "clone":
        return config.tts.resolved_model()
    if config.tts.voice_mode.strip().lower() == "clone":
        return DEFAULT_TTS_PRESET_MODEL
    return config.tts.resolved_model()


def _resolve_tts_key(config: AppConfig) -> str:
    resolved = config.resolve_provider_api_key(config.tts.provider)
    if resolved:
        return resolved
    if config.tts.provider.strip().lower() == "dashscope":
        return os.getenv("DASHSCOPE_API_KEY", "")
    return os.getenv("PODTRAN_TTS_API_KEY", "")
