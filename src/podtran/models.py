from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


StageStatus = Literal["pending", "running", "completed", "failed"]
TaskStatus = Literal["pending", "running", "completed", "failed", "interrupted"]
VoiceMode = Literal["preset", "clone"]
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


class VoiceProfile(BaseModel):
    speaker: str
    provider: str
    mode: VoiceMode = "clone"
    target_model: str
    voice_token: str = ""
    ref_audio_path: str = ""
    ref_text_path: str = ""
    source_audio_path: str = ""
    status: StageStatus = "pending"
    error: str | None = None


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
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
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
    provider: str = ""
    mode: VoiceMode = "preset"
    voice: str
    error: str | None = None
