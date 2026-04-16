from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

from podtran.artifacts import copy_path as artifact_copy_path
from podtran.cache_store import CacheStore
from podtran.models import StageManifest



def test_cache_store_publish_lookup_and_restore(tmp_path: Path) -> None:
    cache = CacheStore(tmp_path / "cache")
    source = tmp_path / "source.json"
    source.write_text('{"ok":true}', encoding="utf-8")
    manifest = StageManifest(
        stage="translate",
        status="completed",
        stage_version=1,
        cache_key="abc123",
        input_fingerprints={"segments_json": "hash-1"},
        config_fingerprint="cfg-1",
        config_keys=["translation.model"],
    )

    cache.publish("translate", "abc123", {"translated_json": source}, manifest)
    entry = cache.lookup("translate", "abc123")
    assert entry is not None

    restored = tmp_path / "restored.json"
    cache.restore(entry, {"translated_json": restored})

    assert restored.read_text(encoding="utf-8") == '{"ok":true}'


def test_cache_store_publish_keeps_existing_completed_entry(tmp_path: Path) -> None:
    cache = CacheStore(tmp_path / "cache")
    original = tmp_path / "original.json"
    original.write_text('{"version":1}', encoding="utf-8")
    replacement = tmp_path / "replacement.json"
    replacement.write_text('{"version":2}', encoding="utf-8")
    manifest = StageManifest(
        stage="translate",
        status="completed",
        stage_version=1,
        cache_key="abc123",
        input_fingerprints={"segments_json": "hash-1"},
        config_fingerprint="cfg-1",
        config_keys=["translation.model"],
    )

    cache.publish("translate", "abc123", {"translated_json": original}, manifest)
    cache.publish("translate", "abc123", {"translated_json": replacement}, manifest)
    entry = cache.lookup("translate", "abc123")

    assert entry is not None
    assert entry.output_path("translated_json").read_text(encoding="utf-8") == '{"version":1}'


def test_cache_store_publish_handles_same_key_race(tmp_path: Path, monkeypatch) -> None:
    cache = CacheStore(tmp_path / "cache")
    source_a = tmp_path / "a.json"
    source_b = tmp_path / "b.json"
    source_a.write_text('{"source":"a"}', encoding="utf-8")
    source_b.write_text('{"source":"b"}', encoding="utf-8")
    manifest = StageManifest(
        stage="translate",
        status="completed",
        stage_version=1,
        cache_key="race-key",
        input_fingerprints={"segments_json": "hash-1"},
        config_fingerprint="cfg-1",
        config_keys=["translation.model"],
    )
    barrier = threading.Barrier(2)

    def synchronized_copy(source: Path, destination: Path) -> Path:
        barrier.wait(timeout=2)
        return artifact_copy_path(source, destination)

    monkeypatch.setattr("podtran.cache_store.copy_path", synchronized_copy)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(cache.publish, "translate", "race-key", {"translated_json": source_a}, manifest),
            executor.submit(cache.publish, "translate", "race-key", {"translated_json": source_b}, manifest),
        ]
        entries = [future.result() for future in futures]

    entry = cache.lookup("translate", "race-key")

    assert all(item.cache_key == "race-key" for item in entries)
    assert entry is not None
    assert entry.output_path("translated_json").read_text(encoding="utf-8") in {'{"source":"a"}', '{"source":"b"}'}
