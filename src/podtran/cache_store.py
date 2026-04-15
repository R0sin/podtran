from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from podtran.artifacts import copy_path, output_refs_exist, read_model, remove_path, write_json
from podtran.models import StageManifest


@dataclass(slots=True)
class CacheEntry:
    stage: str
    cache_key: str
    entry_dir: Path
    manifest: StageManifest

    @property
    def manifest_path(self) -> Path:
        return self.entry_dir / "manifest.json"

    def output_path(self, name: str) -> Path:
        relative = self.manifest.output_refs[name]
        return self.entry_dir / relative


class CacheStore:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def lookup(self, stage: str, cache_key: str) -> CacheEntry | None:
        entry_dir = self._entry_dir(stage, cache_key)
        manifest_path = entry_dir / "manifest.json"
        if not manifest_path.exists():
            return None
        manifest = read_model(manifest_path, StageManifest)
        if manifest.status != "completed":
            return None
        if not output_refs_exist(entry_dir, manifest.output_refs):
            return None
        return CacheEntry(stage=stage, cache_key=cache_key, entry_dir=entry_dir, manifest=manifest)

    def publish(self, stage: str, cache_key: str, outputs: dict[str, Path], manifest: StageManifest) -> CacheEntry:
        entry_dir = self._entry_dir(stage, cache_key)
        if entry_dir.exists():
            shutil.rmtree(entry_dir)
        entry_dir.mkdir(parents=True, exist_ok=True)

        output_refs: dict[str, str] = {}
        for name, source in outputs.items():
            destination = entry_dir / source.name
            copy_path(source, destination)
            output_refs[name] = destination.name

        cache_manifest = manifest.model_copy(deep=True)
        cache_manifest.output_refs = output_refs
        write_json(entry_dir / "manifest.json", cache_manifest)
        return CacheEntry(stage=stage, cache_key=cache_key, entry_dir=entry_dir, manifest=cache_manifest)

    def restore(self, entry: CacheEntry, outputs: dict[str, Path]) -> None:
        for name, destination in outputs.items():
            source = entry.output_path(name)
            copy_path(source, destination)

    def list_entries(self, stage: str | None = None) -> list[CacheEntry]:
        stages = [stage] if stage else [child.name for child in self.cache_dir.iterdir() if child.is_dir() and not child.name.startswith("_")]
        entries: list[CacheEntry] = []
        for stage_name in stages:
            stage_dir = self.cache_dir / stage_name
            if not stage_dir.exists():
                continue
            for child in stage_dir.iterdir():
                if not child.is_dir():
                    continue
                manifest_path = child / "manifest.json"
                if not manifest_path.exists():
                    continue
                entries.append(
                    CacheEntry(
                        stage=stage_name,
                        cache_key=child.name,
                        entry_dir=child,
                        manifest=read_model(manifest_path, StageManifest),
                    )
                )
        entries.sort(key=lambda item: item.manifest.finished_at, reverse=True)
        return entries

    def clean(self, before: datetime | None = None) -> int:
        removed = 0
        normalized_before = _normalize_datetime(before) if before is not None else None
        for entry in self.list_entries():
            if normalized_before is not None:
                finished_at = entry.manifest.finished_at
                if not finished_at:
                    continue
                try:
                    finished_dt = _normalize_datetime(datetime.fromisoformat(finished_at))
                except ValueError:
                    continue
                if finished_dt >= normalized_before:
                    continue
            remove_path(entry.entry_dir)
            removed += 1
        return removed

    def _entry_dir(self, stage: str, cache_key: str) -> Path:
        return self.cache_dir / stage / cache_key


def _normalize_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
