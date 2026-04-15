from pathlib import Path

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
