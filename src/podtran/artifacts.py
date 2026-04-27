from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, TypeAdapter

from podtran.config import model_dump


ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(slots=True)
class ArtifactPaths:
    workdir: Path
    artifacts_dir: Path
    tasks_dir: Path
    cache_dir: Path
    cache_indexes_dir: Path
    task_id: str
    task_dir: Path
    task_json: Path
    manifests_dir: Path
    transcript_json: Path
    segments_json: Path
    translated_json: Path
    preview_audio_path: Path
    refs_dir: Path
    voices_json: Path
    tts_dir: Path
    final_dir: Path
    temp_dir: Path

    @classmethod
    def from_task_id(cls, workdir: Path, task_id: str) -> "ArtifactPaths":
        resolved_workdir = workdir.resolve()
        artifacts_dir = resolved_workdir / "artifacts"
        tasks_dir = artifacts_dir / "tasks"
        cache_dir = artifacts_dir / "cache"
        cache_indexes_dir = cache_dir / "_indexes"
        task_dir = tasks_dir / task_id
        return cls(
            workdir=resolved_workdir,
            artifacts_dir=artifacts_dir,
            tasks_dir=tasks_dir,
            cache_dir=cache_dir,
            cache_indexes_dir=cache_indexes_dir,
            task_id=task_id,
            task_dir=task_dir,
            task_json=task_dir / "task.json",
            manifests_dir=task_dir / "manifests",
            transcript_json=task_dir / "transcript.json",
            segments_json=task_dir / "segments.json",
            translated_json=task_dir / "translated.json",
            preview_audio_path=task_dir / "preview.wav",
            refs_dir=task_dir / "refs",
            voices_json=task_dir / "voices.json",
            tts_dir=task_dir / "tts",
            final_dir=task_dir / "final",
            temp_dir=task_dir / "tmp",
        )

    def ensure(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_indexes_dir.mkdir(parents=True, exist_ok=True)
        self.task_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        self.refs_dir.mkdir(parents=True, exist_ok=True)
        self.tts_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def manifest_path(self, stage: str) -> Path:
        return self.manifests_dir / f"{stage}.json"

    def relative_to_task(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.task_dir.resolve())).replace(
            "\\", "/"
        )


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        suffix=".tmp",
    ) as handle:
        handle.write(content)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=path.parent,
        delete=False,
        suffix=".tmp",
    ) as handle:
        handle.write(content)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def write_json(path: Path, data: BaseModel | list[BaseModel] | dict) -> None:
    payload = json.dumps(model_dump(data), ensure_ascii=False, indent=2)
    atomic_write_text(path, payload)


def read_json_data(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def read_model(path: Path, model_type: type[ModelT]) -> ModelT:
    adapter = TypeAdapter(model_type)
    return adapter.validate_python(read_json_data(path))


def read_model_list(path: Path, model_type: type[ModelT]) -> list[ModelT]:
    adapter = TypeAdapter(list[model_type])
    return adapter.validate_python(read_json_data(path))


def copy_path(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)
    return destination


def remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def output_refs_exist(base_dir: Path, output_refs: dict[str, str]) -> bool:
    for relative_path in output_refs.values():
        if not (base_dir / relative_path).exists():
            return False
    return True
