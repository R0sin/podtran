from __future__ import annotations

import os
from datetime import datetime, timezone

from podtran.artifacts import ArtifactPaths, output_refs_exist, read_model, write_json
from podtran.models import StageManifest, TaskManifest
from podtran.tasks import TaskStore


class StageExecutor:
    def __init__(
        self, task_store: TaskStore, task_manifest: TaskManifest, paths: ArtifactPaths
    ) -> None:
        self.task_store = task_store
        self.task_manifest = task_manifest
        self.paths = paths

    def load_manifest(self, stage: str) -> StageManifest | None:
        manifest_path = self.paths.manifest_path(stage)
        if not manifest_path.exists():
            return None
        return read_model(manifest_path, StageManifest)

    def is_current(
        self,
        stage: str,
        input_fingerprints: dict[str, str],
        config_fingerprint: str,
        output_refs: dict[str, str],
    ) -> tuple[bool, str | None]:
        manifest = self.load_manifest(stage)
        if manifest is None:
            return False, "stage has not run yet"
        if manifest.status == "running":
            return False, "previous execution interrupted"
        if manifest.status == "interrupted":
            return False, "previous execution interrupted by user"
        if manifest.status != "completed":
            return False, manifest.error or f"{stage} is {manifest.status}"
        if manifest.output_refs != output_refs:
            return False, "output refs changed"
        if not output_refs_exist(self.paths.task_dir, output_refs):
            return False, "output missing"
        for key, value in input_fingerprints.items():
            previous = manifest.input_fingerprints.get(key)
            if previous != value:
                return False, f"{key} fingerprint changed"
        if manifest.config_fingerprint != config_fingerprint:
            return False, f"{stage} config changed"
        return True, None

    def start(self, manifest: StageManifest) -> StageManifest:
        manifest.status = "running"
        manifest.started_at = _utc_now()
        manifest.finished_at = ""
        manifest.pid = os.getpid()
        manifest.error = None
        self._save_manifest(manifest)
        self._update_task(stage=manifest.stage, status="running")
        return manifest

    def complete(self, manifest: StageManifest) -> StageManifest:
        manifest.status = "completed"
        manifest.finished_at = _utc_now()
        manifest.error = None
        manifest.stale_reason = None
        self._save_manifest(manifest)
        next_status = "completed" if manifest.stage == "compose" else "running"
        self._update_task(stage=manifest.stage, status=next_status)
        return manifest

    def fail(self, manifest: StageManifest, exc: Exception) -> StageManifest:
        manifest.status = "failed"
        manifest.finished_at = _utc_now()
        manifest.error = str(exc).strip() or repr(exc)
        self._save_manifest(manifest)
        self._update_task(stage=manifest.stage, status="failed")
        return manifest

    def interrupt(self, manifest: StageManifest) -> StageManifest:
        manifest.status = "interrupted"
        manifest.finished_at = _utc_now()
        manifest.error = "Interrupted by user"
        self._save_manifest(manifest)
        self._update_task(stage=manifest.stage, status="interrupted")
        return manifest

    def save_completed(self, manifest: StageManifest) -> StageManifest:
        manifest.status = "completed"
        if not manifest.started_at:
            manifest.started_at = _utc_now()
        manifest.finished_at = _utc_now()
        manifest.error = None
        manifest.stale_reason = None
        self._save_manifest(manifest)
        next_status = "completed" if manifest.stage == "compose" else "running"
        self._update_task(stage=manifest.stage, status=next_status)
        return manifest

    def _save_manifest(self, manifest: StageManifest) -> None:
        write_json(self.paths.manifest_path(manifest.stage), manifest)

    def _update_task(self, stage: str, status: str) -> None:
        self.task_manifest.current_stage = stage
        self.task_manifest.status = status
        self.task_manifest.updated_at = _utc_now()
        self.task_store.save_task(self.task_manifest)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
