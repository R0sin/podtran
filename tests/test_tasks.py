from pathlib import Path

from podtran.config import AppConfig
from podtran.fingerprints import FingerprintService
from podtran.tasks import TaskStore



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



def test_task_store_creates_preview_task_with_processing_audio_metadata(tmp_path: Path) -> None:
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
