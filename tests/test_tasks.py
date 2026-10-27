from __future__ import annotations

from pathlib import Path

import pytest

from podtran.config import AppConfig
from podtran.fingerprints import FingerprintService
from podtran.tasks import TaskStore

pytestmark = pytest.mark.usefixtures("fake_audio_decoder")


def test_task_store_creates_unique_task_ids_and_loads_latest(tmp_path: Path) -> None:
    audio = tmp_path / "podcast.wav"
    audio.write_bytes(b"audio")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)

    first = store.create_task(audio, AppConfig(), "podtran podcast.wav")
    second = store.create_task(audio, AppConfig(), "podtran podcast.wav")

    assert first.task_id != second.task_id
    assert store.load_latest_task().task_id == second.task_id


def test_task_store_resolves_unique_prefix(tmp_path: Path) -> None:
    audio = tmp_path / "podcast.wav"
    audio.write_bytes(b"audio")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)

    created = store.create_task(audio, AppConfig(), "podtran podcast.wav")
    loaded = store.load_task(created.task_id[:10])

    assert loaded.task_id == created.task_id


def test_task_store_creates_preview_task_with_processing_audio_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "podcast.wav"
    preview = tmp_path / "preview.wav"
    source.write_bytes(b"source")
    preview.write_bytes(b"preview")
    fingerprints = FingerprintService(tmp_path / "artifacts" / "cache" / "_indexes")
    store = TaskStore(tmp_path, fingerprints)

    task = store.create_task_with_processing_audio(
        source,
        AppConfig(),
        "podtran --preview podcast.wav",
        task_id="20260408-120000-aaaaaa",
        source_audio_sha256=fingerprints.hash_audio(source),
        processing_audio=preview,
        processing_audio_sha256=fingerprints.hash_audio(preview),
        preview=True,
        preview_start_seconds=0.0,
        preview_duration_seconds=300.0,
    )

    assert task.preview is True
    assert task.preview_start_seconds == 0.0
    assert task.preview_duration_seconds == 300.0
    assert task.processing_audio_path == str(preview.resolve())
    assert task.processing_audio_sha256 != task.source_audio_sha256


def test_refresh_replaces_legacy_hashes_without_removing_task_outputs(tmp_path):
    source = tmp_path / "source.wav"
    preview = tmp_path / "preview.wav"
    source.write_bytes(b"source")
    preview.write_bytes(b"preview")
    fingerprints = FingerprintService(tmp_path / "indexes")
    store = TaskStore(tmp_path, fingerprints)
    task = store.create_task_with_processing_audio(
        source,
        AppConfig(),
        "podtran source.wav --preview",
        task_id="legacy",
        source_audio_sha256="old-source",
        processing_audio=preview,
        processing_audio_sha256="old-preview",
        preview=True,
    )
    paths = store.paths_for(task)
    output = paths.tts_dir / "existing.wav"
    output.write_bytes(b"existing tts")
    store.refresh_audio_fingerprints(task)
    reloaded = store.load_task(task.task_id)
    assert reloaded.source_audio_sha256 == fingerprints.hash_audio(source)
    assert reloaded.processing_audio_sha256 == fingerprints.hash_audio(preview)
    assert reloaded.processing_audio_sha256 != reloaded.source_audio_sha256
    assert output.read_bytes() == b"existing tts"


def test_refresh_does_not_partially_save_when_processing_audio_is_missing(tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    store = TaskStore(tmp_path, FingerprintService(tmp_path / "indexes"))
    task = store.create_task_with_processing_audio(
        source,
        AppConfig(),
        "podtran source.wav --preview",
        task_id="legacy",
        source_audio_sha256="old-source",
        processing_audio=tmp_path / "missing.wav",
        processing_audio_sha256="old-preview",
    )
    with pytest.raises(FileNotFoundError):
        store.refresh_audio_fingerprints(task)
    assert task.source_audio_sha256 == "old-source"
    assert store.load_task(task.task_id).source_audio_sha256 == "old-source"
