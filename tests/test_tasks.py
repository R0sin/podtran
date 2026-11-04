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


def test_refresh_updates_processing_identity_and_preserves_source_provenance(tmp_path):
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
    assert reloaded.source_audio_sha256 == "old-source"
    assert reloaded.processing_audio_sha256 == fingerprints.hash_audio(preview)
    assert reloaded.processing_audio_sha256 != reloaded.source_audio_sha256
    assert output.read_bytes() == b"existing tts"


def test_new_task_uses_separate_prepared_snapshot(tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    fingerprints = FingerprintService(tmp_path / "indexes")
    store = TaskStore(tmp_path, fingerprints)
    task = store.create_task(source, AppConfig(), "podtran source.wav")
    processing = Path(task.processing_audio_path)
    assert processing.name == "processing.flac"
    assert processing != source
    source_hash = task.source_audio_sha256
    source.write_bytes(b"changed after task creation")
    store.refresh_audio_fingerprints(task)
    assert processing.read_bytes() == b"source"
    assert task.source_audio_sha256 == source_hash
    assert task.processing_audio_sha256 == fingerprints.hash_audio(processing)


def test_legacy_task_requires_new_run_without_modifying_artifacts(tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    store = TaskStore(tmp_path, FingerprintService(tmp_path / "indexes"))
    task = store.create_task(source, AppConfig(), "podtran source.wav")
    task.audio_preparation_version = 0
    store.save_task(task)
    paths = store.paths_for(task)
    before = paths.task_json.read_bytes()
    with pytest.raises(RuntimeError, match="Start a new task"):
        store.refresh_audio_fingerprints(task)
    assert paths.task_json.read_bytes() == before
    assert Path(task.processing_audio_path).exists()


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


@pytest.mark.parametrize("preview", [False, True])
@pytest.mark.parametrize("failure_at", ["prepare", "fingerprint", "manifest"])
def test_task_creation_failure_cleans_partial_directory(
    tmp_path, monkeypatch, preview, failure_at
):
    source = tmp_path / "source.wav"
    source.write_bytes(b"original")
    fingerprints = FingerprintService(tmp_path / "indexes")
    store = TaskStore(tmp_path, fingerprints)

    def fail(*args, **kwargs):
        raise RuntimeError("creation failed")

    if failure_at == "prepare":
        monkeypatch.setattr("podtran.tasks.prepare_audio", fail)
    elif failure_at == "fingerprint":
        monkeypatch.setattr(fingerprints, "hash_audio", fail)
    else:
        monkeypatch.setattr("podtran.tasks.write_json", fail)
    with pytest.raises(RuntimeError, match="creation failed"):
        store.create_task(source, AppConfig(), "podtran source.wav", preview=preview)
    assert source.read_bytes() == b"original"
    assert list(store.tasks_dir.iterdir()) == []
