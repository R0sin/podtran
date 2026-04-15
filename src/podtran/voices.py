from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from podtran.artifacts import ArtifactPaths, read_model, read_model_list, write_json
from podtran.audio import FFMPEG_COMMAND, concat_wav_chunks, extract_audio_chunk, reset_temp_dir
from podtran.cache_store import CacheStore
from podtran.config import AppConfig
from podtran.fingerprints import FingerprintService, VOICE_CLONE_CONFIG_KEYS
from podtran.models import ResolvedVoiceTarget, SegmentRecord, StageManifest, StageProgressCallback, VoiceProfile
from podtran.stage_versions import VOICE_CLONE_STAGE_VERSION

MAX_CLONE_REFERENCE_DURATION = 60.0
MAX_CLONE_REFERENCE_PAUSE = 2.0
MIN_CLONE_CONTIGUOUS_SPEECH = 3.0
VOICE_ENROLLMENT_RETRY_ATTEMPTS = 3


class VoiceCloningProvider(Protocol):
    def ensure_voice(self, reference_audio: Path, target_model: str, preferred_name: str) -> str:
        ...


class DashScopeVoiceCloningProvider:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = httpx.Client(timeout=config.tts.timeout_seconds)

    @retry(
        reraise=True,
        stop=stop_after_attempt(VOICE_ENROLLMENT_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((RuntimeError, httpx.HTTPError)),
    )
    def ensure_voice(self, reference_audio: Path, target_model: str, preferred_name: str) -> str:
        audio_bytes = reference_audio.read_bytes()
        mime_type = mimetypes.guess_type(reference_audio.name)[0] or "audio/wav"
        data_uri = f"data:{mime_type};base64,{base64.b64encode(audio_bytes).decode('utf-8')}"
        response = self.client.post(
            self.config.tts.resolved_customization_url(),
            headers={
                "Authorization": f"Bearer {self._resolve_api_key()}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.config.tts.enrollment_model,
                "input": {
                    "action": "create",
                    "target_model": target_model,
                    "preferred_name": preferred_name,
                    "audio": {"data": data_uri},
                },
            },
        )
        response.raise_for_status()
        payload = response.json()
        voice_token = str(payload.get("output", {}).get("voice", "")).strip()
        if not voice_token:
            raise RuntimeError(f"DashScope voice enrollment returned no voice token: {payload}")
        return voice_token

    def _resolve_api_key(self) -> str:
        resolved = self.config.resolve_provider_api_key(self.config.tts.provider)
        if resolved:
            return resolved
        import os as _os

        return _os.getenv("DASHSCOPE_API_KEY", "")


@dataclass(slots=True)
class ReferenceCandidate:
    start: float
    end: float
    text: str
    segments: list[SegmentRecord]

    @property
    def duration(self) -> float:
        return self.end - self.start


class VoiceProfileManager:
    def __init__(
        self,
        config: AppConfig,
        paths: ArtifactPaths,
        cache_store: CacheStore | None = None,
        fingerprints: FingerprintService | None = None,
    ) -> None:
        self.config = config
        self.paths = paths
        self.cache_store = cache_store
        self.fingerprints = fingerprints
        self._provider = self._build_provider()

    def resolve_voice_targets(
        self,
        segments: list[SegmentRecord],
        source_audio: Path,
        source_audio_fingerprint: str | None = None,
        progress_callback: StageProgressCallback | None = None,
    ) -> dict[str, ResolvedVoiceTarget]:
        source_path = source_audio.resolve()
        preset_targets = build_preset_targets(segments)
        if self.config.tts.voice_mode.strip().lower() != "clone":
            return preset_targets

        target_model = self.config.tts.resolved_model()
        profile_map = {profile.speaker: profile for profile in self._load_profiles()}
        build_dir = self.paths.temp_dir / "voice_refs"
        reset_temp_dir(build_dir, self.paths.task_dir)
        resolved: dict[str, ResolvedVoiceTarget] = {}
        failures: list[str] = []
        speakers = list(preset_targets)
        total_speakers = max(len(speakers), 1)
        completed = 0
        if progress_callback is not None:
            progress_callback(0, total_speakers, "Resolving cloned voices")

        for speaker in speakers:
            cached = profile_map.get(speaker)
            if self._is_profile_reusable(cached, source_path, target_model):
                resolved[speaker] = ResolvedVoiceTarget(
                    speaker=speaker,
                    provider=cached.provider,
                    mode="clone",
                    voice=cached.voice_token,
                )
                completed += 1
                self._emit_progress(progress_callback, completed, total_speakers, f"Reusing voice for {speaker}")
                continue

            candidate = select_reference_candidate(
                segments,
                speaker,
                pause_threshold=max(float(self.config.compose.block_pause_threshold), MAX_CLONE_REFERENCE_PAUSE),
                preferred_min_duration=float(self.config.tts.clone_min_ref_seconds),
                preferred_max_duration=float(self.config.tts.clone_max_ref_seconds),
                hard_max_duration=MAX_CLONE_REFERENCE_DURATION,
                min_continuous_speech=MIN_CLONE_CONTIGUOUS_SPEECH,
            )
            if candidate is None:
                message = (
                    f"No qualifying reference audio for {speaker}. "
                    f"Need at least {MIN_CLONE_CONTIGUOUS_SPEECH:g}s contiguous speech, "
                    f"short pauses <= {MAX_CLONE_REFERENCE_PAUSE:g}s, and total reference audio <= "
                    f"{MAX_CLONE_REFERENCE_DURATION:g}s "
                    f"(preferred {self.config.tts.clone_min_ref_seconds}s-"
                    f"{self.config.tts.clone_max_ref_seconds}s)."
                )
                profile_map[speaker] = VoiceProfile(
                    speaker=speaker,
                    provider=self.config.tts.provider,
                    target_model=target_model,
                    ref_audio_path="",
                    ref_text_path="",
                    source_audio_path=str(source_path),
                    status="failed",
                    error=message,
                )
                failures.append(f"{speaker}: {message}")
                completed += 1
                self._emit_progress(progress_callback, completed, total_speakers, f"Voice reference failed for {speaker}")
                continue

            ref_audio_path, ref_text_path = self._export_reference_audio(source_path, speaker, candidate, build_dir)
            cache_key = self._voice_cache_key(speaker, candidate, source_audio_fingerprint, target_model)
            cached_profile = self._load_cached_profile(cache_key)
            if cached_profile is not None:
                profile_map[speaker] = VoiceProfile(
                    speaker=speaker,
                    provider=cached_profile.provider,
                    target_model=target_model,
                    voice_token=cached_profile.voice_token,
                    ref_audio_path=str(ref_audio_path.resolve()),
                    ref_text_path=str(ref_text_path.resolve()),
                    source_audio_path=str(source_path),
                    status="completed",
                    error=None,
                )
                resolved[speaker] = ResolvedVoiceTarget(
                    speaker=speaker,
                    provider=cached_profile.provider,
                    mode="clone",
                    voice=cached_profile.voice_token,
                )
                completed += 1
                self._emit_progress(progress_callback, completed, total_speakers, f"Restored cached voice for {speaker}")
                continue

            try:
                voice_token = self._provider.ensure_voice(
                    ref_audio_path,
                    target_model,
                    preferred_name=self._preferred_name(speaker),
                )
                profile = VoiceProfile(
                    speaker=speaker,
                    provider=self.config.tts.provider,
                    target_model=target_model,
                    voice_token=voice_token,
                    ref_audio_path=str(ref_audio_path.resolve()),
                    ref_text_path=str(ref_text_path.resolve()),
                    source_audio_path=str(source_path),
                    status="completed",
                    error=None,
                )
                profile_map[speaker] = profile
                resolved[speaker] = ResolvedVoiceTarget(
                    speaker=speaker,
                    provider=self.config.tts.provider,
                    mode="clone",
                    voice=voice_token,
                )
                self._publish_cached_profile(cache_key, profile, source_audio_fingerprint, candidate)
                completed += 1
                self._emit_progress(progress_callback, completed, total_speakers, f"Enrolled voice for {speaker}")
            except Exception as exc:
                profile_map[speaker] = VoiceProfile(
                    speaker=speaker,
                    provider=self.config.tts.provider,
                    target_model=target_model,
                    ref_audio_path=str(ref_audio_path.resolve()),
                    ref_text_path=str(ref_text_path.resolve()),
                    source_audio_path=str(source_path),
                    status="failed",
                    error=str(exc),
                )
                failures.append(f"{speaker}: {exc}")
                completed += 1
                self._emit_progress(progress_callback, completed, total_speakers, f"Voice enrollment failed for {speaker}")

        write_json(self.paths.voices_json, list(profile_map.values()))
        if failures:
            raise RuntimeError(f"Voice clone failed for {len(failures)} speaker(s): {'; '.join(failures)}")
        self._emit_progress(progress_callback, total_speakers, total_speakers, "Voice resolution complete")
        return resolved

    def _build_provider(self) -> VoiceCloningProvider:
        backend = self.config.tts.resolved_backend()
        if backend == "dashscope":
            return DashScopeVoiceCloningProvider(self.config)
        raise RuntimeError(f"Clone mode is not supported for TTS provider: {self.config.tts.provider}")

    def _export_reference_audio(
        self,
        source_audio: Path,
        speaker: str,
        candidate: ReferenceCandidate,
        build_dir: Path,
    ) -> tuple[Path, Path]:
        ref_audio_path = self.paths.refs_dir / f"{speaker}.wav"
        ref_text_path = self.paths.refs_dir / f"{speaker}.txt"
        if len(candidate.segments) == 1:
            extract_audio_chunk(
                FFMPEG_COMMAND,
                source_audio,
                ref_audio_path,
                candidate.start,
                candidate.end,
            )
        else:
            chunk_paths: list[Path] = []
            for index, segment in enumerate(candidate.segments):
                chunk_path = build_dir / f"{speaker}_{index:02d}.wav"
                extract_audio_chunk(
                    FFMPEG_COMMAND,
                    source_audio,
                    chunk_path,
                    segment.start,
                    segment.end,
                )
                chunk_paths.append(chunk_path)
            concat_wav_chunks(FFMPEG_COMMAND, chunk_paths, ref_audio_path)
        ref_text_path.write_text(candidate.text, encoding="utf-8")
        return ref_audio_path, ref_text_path


    def _is_profile_reusable(self, profile: VoiceProfile | None, source_audio: Path, target_model: str) -> bool:
        if profile is None or profile.status != "completed":
            return False
        if profile.target_model != target_model:
            return False
        if profile.source_audio_path != str(source_audio):
            return False
        if not profile.voice_token.strip():
            return False
        if not profile.ref_audio_path or not Path(profile.ref_audio_path).exists():
            return False
        return True

    def _load_profiles(self) -> list[VoiceProfile]:
        if not self.paths.voices_json.exists():
            return []
        return read_model_list(self.paths.voices_json, VoiceProfile)

    def _preferred_name(self, speaker: str) -> str:
        normalized = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in speaker.lower())
        while "__" in normalized:
            normalized = normalized.replace("__", "_")
        compact = normalized.strip("_") or "speaker"
        return compact[:16]

    def _emit_progress(
        self,
        progress_callback: StageProgressCallback | None,
        completed: int,
        total: int,
        message: str,
    ) -> None:
        if progress_callback is None:
            return
        progress_callback(completed, total, message)
    def _voice_cache_key(
        self,
        speaker: str,
        candidate: ReferenceCandidate,
        source_audio_fingerprint: str | None,
        target_model: str,
    ) -> str | None:
        if self.cache_store is None or self.fingerprints is None or not source_audio_fingerprint:
            return None
        reference_fingerprint = self.fingerprints.hash_value(
            {
                "speaker": speaker,
                "source_audio": source_audio_fingerprint,
                "start": round(candidate.start, 3),
                "end": round(candidate.end, 3),
                "text": candidate.text.strip(),
            }
        )
        config_fingerprint = self.fingerprints.hash_config_subset(self.config, VOICE_CLONE_CONFIG_KEYS)
        return self.fingerprints.build_stage_cache_key(
            "voice_clone",
            VOICE_CLONE_STAGE_VERSION,
            {
                "source_audio": source_audio_fingerprint,
                "reference": reference_fingerprint,
                "speaker": speaker,
                "target_model": target_model,
            },
            config_fingerprint,
        )

    def _load_cached_profile(self, cache_key: str | None) -> VoiceProfile | None:
        if cache_key is None or self.cache_store is None:
            return None
        entry = self.cache_store.lookup("voice_clone", cache_key)
        if entry is None:
            return None
        return read_model(entry.output_path("voice_profile"), VoiceProfile)

    def _publish_cached_profile(
        self,
        cache_key: str | None,
        profile: VoiceProfile,
        source_audio_fingerprint: str | None,
        candidate: ReferenceCandidate,
    ) -> None:
        if cache_key is None or self.cache_store is None or self.fingerprints is None or not source_audio_fingerprint:
            return
        profile_json = self.paths.temp_dir / f"{profile.speaker}.voice_profile.json"
        write_json(profile_json, profile)
        reference_fingerprint = self.fingerprints.hash_value(
            {
                "speaker": profile.speaker,
                "source_audio": source_audio_fingerprint,
                "start": round(candidate.start, 3),
                "end": round(candidate.end, 3),
                "text": candidate.text.strip(),
            }
        )
        manifest = StageManifest(
            stage="voice_clone",
            status="completed",
            stage_version=VOICE_CLONE_STAGE_VERSION,
            cache_key=cache_key,
            input_fingerprints={
                "source_audio": source_audio_fingerprint,
                "reference": reference_fingerprint,
                "speaker": profile.speaker,
                "target_model": profile.target_model,
            },
            config_fingerprint=self.fingerprints.hash_config_subset(self.config, VOICE_CLONE_CONFIG_KEYS),
            config_keys=VOICE_CLONE_CONFIG_KEYS,
            started_at=_utc_now(),
            finished_at=_utc_now(),
            pid=os.getpid(),
        )
        self.cache_store.publish("voice_clone", cache_key, {"voice_profile": profile_json}, manifest)


def build_preset_targets(segments: list[SegmentRecord]) -> dict[str, ResolvedVoiceTarget]:
    targets: dict[str, ResolvedVoiceTarget] = {}
    for segment in segments:
        targets.setdefault(
            segment.speaker,
            ResolvedVoiceTarget(
                speaker=segment.speaker,
                provider="",
                mode="preset",
                voice=segment.voice,
            ),
        )
    return targets



def select_reference_candidate(
    segments: list[SegmentRecord],
    speaker: str,
    pause_threshold: float,
    preferred_min_duration: float,
    preferred_max_duration: float,
    hard_max_duration: float = MAX_CLONE_REFERENCE_DURATION,
    min_continuous_speech: float = MIN_CLONE_CONTIGUOUS_SPEECH,
) -> ReferenceCandidate | None:
    ordered = sorted(segments, key=lambda item: item.start)
    candidates: list[ReferenceCandidate] = []
    target_duration = (preferred_min_duration + preferred_max_duration) / 2.0

    for index, segment in enumerate(ordered):
        if segment.speaker != speaker:
            continue
        current_segments = [segment]
        current_texts = [segment.text.strip()]
        block_start = segment.start
        block_end = segment.end
        if _is_reference_worthy(
            current_segments,
            segment.text,
            block_end - block_start,
            hard_max_duration,
            min_continuous_speech,
        ):
            candidates.append(
                ReferenceCandidate(
                    start=block_start,
                    end=block_end,
                    text=" ".join(part for part in current_texts if part).strip(),
                    segments=list(current_segments),
                )
            )

        for next_segment in ordered[index + 1 :]:
            if next_segment.speaker != speaker:
                break
            gap = next_segment.start - block_end
            if gap > pause_threshold:
                break
            proposed_end = next_segment.end
            proposed_duration = proposed_end - block_start
            if proposed_duration > hard_max_duration:
                break
            current_segments.append(next_segment)
            current_texts.append(next_segment.text.strip())
            block_end = proposed_end
            if _is_reference_worthy(
                current_segments,
                " ".join(part for part in current_texts if part).strip(),
                proposed_duration,
                hard_max_duration,
                min_continuous_speech,
            ):
                candidates.append(
                    ReferenceCandidate(
                        start=block_start,
                        end=block_end,
                        text=" ".join(part for part in current_texts if part).strip(),
                        segments=list(current_segments),
                    )
                )

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda candidate: _candidate_score(
            candidate,
            target_duration,
            preferred_min_duration,
            preferred_max_duration,
        ),
    )


def _is_reference_worthy(
    segments: list[SegmentRecord],
    text: str,
    duration: float,
    hard_max_duration: float,
    min_continuous_speech: float,
) -> bool:
    normalized = text.strip()
    if duration <= 0 or duration > hard_max_duration:
        return False
    if not normalized:
        return False
    return any((segment.end - segment.start) >= min_continuous_speech for segment in segments)


def _candidate_score(
    candidate: ReferenceCandidate,
    target_duration: float,
    preferred_min_duration: float,
    preferred_max_duration: float,
) -> float:
    if preferred_min_duration <= candidate.duration <= preferred_max_duration:
        duration_score = 200.0 - abs(candidate.duration - target_duration) * 6.0
    elif candidate.duration < preferred_min_duration:
        duration_score = 100.0 + candidate.duration * 6.0
    else:
        duration_score = 100.0 - (candidate.duration - preferred_max_duration) * 2.0
    text_score = min(len(candidate.text), 240) / 6.0
    continuity_bonus = max(0, 20 - (len(candidate.segments) - 1) * 4)
    return duration_score + text_score + continuity_bonus


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()








