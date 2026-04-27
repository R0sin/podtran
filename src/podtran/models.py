from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


StageStatus = Literal["pending", "running", "completed", "failed"]
TaskStatus = Literal["pending", "running", "completed", "failed", "interrupted"]
VoiceMode = Literal["preset", "clone"]
VoiceSpecKind = Literal["preset", "provider_clone", "reference_clone"]
StageProgressCallback = Callable[[int, int, str], None]


class WordAlignment(BaseModel):
    word: str
    start: float | None = None
    end: float | None = None
    score: float | None = None
    speaker: str | None = None


class TranscriptSegment(BaseModel):
    segment_id: str
    start: float
    end: float
    text: str
    language: str | None = None
    speaker: str = "UNKNOWN"
    words: list[WordAlignment] = Field(default_factory=list)


class SegmentRecord(BaseModel):
    segment_id: str
    block_id: str
    start: float
    end: float
    text: str
    speaker: str
    voice: str
    words: list[WordAlignment] = Field(default_factory=list)
    text_zh: str = ""
    tts_audio_path: str = ""
    tts_duration_ms: int = 0
    status: StageStatus = "pending"
    error: str | None = None


class PresetVoiceSpec(BaseModel):
    kind: Literal["preset"] = "preset"
    identity: str
    voice_name: str


class ProviderClonePayload(BaseModel):
    voice_token: str


class ProviderCloneSpec(BaseModel):
    kind: Literal["provider_clone"] = "provider_clone"
    identity: str
    provider: str
    payload: ProviderClonePayload


class ReferenceClonePayload(BaseModel):
    reference_fingerprint: str
    reference_audio_path: str = ""
    reference_text_path: str = ""
    reference_text: str = ""


class ReferenceCloneSpec(BaseModel):
    """Voice clone by passing reference audio at synthesis time.

    This shape is reserved for backends that can consume reference audio
    directly during synthesis instead of requiring a separate enrollment step.
    """

    kind: Literal["reference_clone"] = "reference_clone"
    identity: str
    provider: str
    payload: ReferenceClonePayload


VoiceSpec = Annotated[
    PresetVoiceSpec | ProviderCloneSpec | ReferenceCloneSpec,
    Field(discriminator="kind"),
]
CloneVoiceSpec = ProviderCloneSpec | ReferenceCloneSpec


class VoiceProfile(BaseModel):
    """Persisted clone state for a speaker.

    `voice_spec` is the canonical source of clone asset metadata. The flattened
    top-level fields are retained for legacy JSON compatibility and quick access
    when rebuilding paths or rendering task state.
    """

    speaker: str
    provider: str = ""
    mode: VoiceMode = "clone"
    target_model: str
    asset_kind: str = ""
    asset_identity: str = ""
    voice_spec: CloneVoiceSpec | None = None
    reference_fingerprint: str = ""
    reference_audio_path: str = ""
    reference_text_path: str = ""
    source_audio_fingerprint: str = ""
    source_audio_path: str = ""
    status: StageStatus = "pending"
    error: str | None = None

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_profile(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        if "voice_spec" in data:
            return data

        voice_token = str(data.get("voice_token", "") or "").strip()
        provider = str(data.get("provider", "") or "").strip()
        if not voice_token:
            return data

        identity = f"{provider or 'unknown'}:provider_clone:{voice_token}"
        migrated = dict(data)
        migrated["voice_spec"] = {
            "kind": "provider_clone",
            "identity": identity,
            "provider": provider,
            "payload": {"voice_token": voice_token},
        }
        migrated.setdefault("asset_kind", "provider_clone")
        migrated.setdefault("asset_identity", identity)
        migrated.setdefault(
            "reference_audio_path", str(data.get("ref_audio_path", "") or "")
        )
        migrated.setdefault(
            "reference_text_path", str(data.get("ref_text_path", "") or "")
        )
        migrated.setdefault(
            "source_audio_path", str(data.get("source_audio_path", "") or "")
        )
        migrated.setdefault(
            "source_audio_fingerprint",
            str(data.get("source_audio_fingerprint", "") or ""),
        )
        migrated.setdefault(
            "reference_fingerprint", str(data.get("reference_fingerprint", "") or "")
        )
        return migrated

    @model_validator(mode="after")
    def sync_asset_fields(self) -> "VoiceProfile":
        if self.voice_spec is not None:
            self.asset_kind = self.voice_spec.kind
            self.asset_identity = self.voice_spec.identity
            if not self.provider.strip():
                self.provider = self.voice_spec.provider
        return self


class TaskManifest(BaseModel):
    task_id: str
    created_at: str
    updated_at: str
    source_audio_path: str
    source_audio_name: str
    source_audio_sha256: str
    preview: bool = False
    preview_start_seconds: float = 0.0
    preview_duration_seconds: float = 0.0
    processing_audio_path: str = ""
    processing_audio_sha256: str = ""
    entry_command: str
    config_hash: str
    config_snapshot: dict[str, object] = Field(default_factory=dict)
    current_stage: str = ""
    status: TaskStatus = "pending"

    @model_validator(mode="after")
    def apply_processing_defaults(self) -> "TaskManifest":
        if not self.processing_audio_path:
            self.processing_audio_path = self.source_audio_path
        if not self.processing_audio_sha256:
            self.processing_audio_sha256 = self.source_audio_sha256
        return self


class StageManifest(BaseModel):
    stage: str
    status: TaskStatus = "pending"
    stage_version: int = 1
    cache_key: str = ""
    input_fingerprints: dict[str, str] = Field(default_factory=dict)
    config_fingerprint: str = ""
    config_keys: list[str] = Field(default_factory=list)
    output_refs: dict[str, str] = Field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""
    pid: int = 0
    error: str | None = None
    stale_reason: str | None = None


class ResolvedVoiceTarget(BaseModel):
    speaker: str
    spec: VoiceSpec
    error: str | None = None
