from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from podtran.artifacts import ArtifactPaths, read_model, write_json
from podtran.config import AppConfig, model_dump
from podtran.fingerprints import FingerprintService
from podtran.models import TaskManifest


class TaskStore:
    def __init__(self, workdir: Path, fingerprints: FingerprintService) -> None:
        self.workdir = workdir.resolve()
        self.fingerprints = fingerprints
        self.artifacts_dir = self.workdir / "artifacts"
        self.tasks_dir = self.artifacts_dir / "tasks"
        self.tasks_dir.mkdir(parents=True, exist_ok=True)

    def create_task(self, audio: Path, config_snapshot: AppConfig, entry_command: str) -> TaskManifest:
        resolved_audio = audio.resolve()
        source_audio_sha256 = self.fingerprints.hash_audio(resolved_audio)
        task_id = self._build_unique_task_id(source_audio_sha256)
        return self.create_task_with_processing_audio(
            resolved_audio,
            config_snapshot,
            entry_command,
            task_id=task_id,
            source_audio_sha256=source_audio_sha256,
            processing_audio=resolved_audio,
            processing_audio_sha256=source_audio_sha256,
        )

    def reserve_task_id(self, audio: Path) -> tuple[str, str]:
        resolved_audio = audio.resolve()
        source_audio_sha256 = self.fingerprints.hash_audio(resolved_audio)
        return self._build_unique_task_id(source_audio_sha256), source_audio_sha256

    def create_task_with_processing_audio(
        self,
        audio: Path,
        config_snapshot: AppConfig,
        entry_command: str,
        *,
        task_id: str,
        source_audio_sha256: str,
        processing_audio: Path,
        processing_audio_sha256: str,
        preview: bool = False,
        preview_start_seconds: float = 0.0,
        preview_duration_seconds: float = 0.0,
    ) -> TaskManifest:
        resolved_audio = audio.resolve()
        resolved_processing_audio = processing_audio.resolve()
        paths = ArtifactPaths.from_task_id(self.workdir, task_id)
        paths.ensure()
        timestamp = _utc_now()
        manifest = TaskManifest(
            task_id=task_id,
            created_at=timestamp,
            updated_at=timestamp,
            source_audio_path=str(resolved_audio),
            source_audio_name=resolved_audio.name,
            source_audio_sha256=source_audio_sha256,
            preview=preview,
            preview_start_seconds=preview_start_seconds,
            preview_duration_seconds=preview_duration_seconds,
            processing_audio_path=str(resolved_processing_audio),
            processing_audio_sha256=processing_audio_sha256,
            entry_command=entry_command,
            config_hash=self.fingerprints.hash_value(model_dump(config_snapshot)),
            config_snapshot=model_dump(config_snapshot),
            current_stage="",
            status="pending",
        )
        write_json(paths.task_json, manifest)
        return manifest

    def save_task(self, manifest: TaskManifest) -> None:
        paths = ArtifactPaths.from_task_id(self.workdir, manifest.task_id)
        paths.ensure()
        write_json(paths.task_json, manifest)

    def load_task(self, id_or_prefix: str) -> TaskManifest:
        matches = self._resolve_matches(id_or_prefix)
        if not matches:
            raise FileNotFoundError(f"Task not found: {id_or_prefix}")
        if len(matches) > 1:
            raise ValueError(f"Task prefix is ambiguous: {id_or_prefix}")
        return read_model(matches[0] / "task.json", TaskManifest)

    def load_latest_task(self) -> TaskManifest:
        tasks = self.list_tasks(limit=1)
        if not tasks:
            raise FileNotFoundError("No tasks found.")
        return tasks[0]

    def list_tasks(self, limit: int = 20) -> list[TaskManifest]:
        manifests: list[TaskManifest] = []
        if not self.tasks_dir.exists():
            return manifests
        for child in self.tasks_dir.iterdir():
            if not child.is_dir():
                continue
            task_json = child / "task.json"
            if not task_json.exists():
                continue
            manifests.append(read_model(task_json, TaskManifest))
        manifests.sort(key=lambda item: (item.updated_at, item.task_id), reverse=True)
        return manifests[:limit]

    def paths_for(self, task_manifest: TaskManifest) -> ArtifactPaths:
        return ArtifactPaths.from_task_id(self.workdir, task_manifest.task_id)

    def _resolve_matches(self, id_or_prefix: str) -> list[Path]:
        exact = self.tasks_dir / id_or_prefix
        if exact.exists() and (exact / "task.json").exists():
            return [exact]
        if not self.tasks_dir.exists():
            return []
        return sorted([child for child in self.tasks_dir.iterdir() if child.is_dir() and child.name.startswith(id_or_prefix)])

    def _build_unique_task_id(self, source_audio_sha256: str) -> str:
        base_time = datetime.now(timezone.utc).replace(microsecond=0)
        suffix = source_audio_sha256[:6]
        for attempt in range(300):
            candidate_time = base_time + timedelta(seconds=attempt)
            task_id = f"{candidate_time.strftime('%Y%m%d-%H%M%S')}-{suffix}"
            if not (self.tasks_dir / task_id).exists():
                return task_id
        raise RuntimeError("Unable to allocate a unique task id.")



def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
