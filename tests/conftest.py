from __future__ import annotations

import pytest
import hashlib


@pytest.fixture(autouse=True)
def isolate_platform_detection(monkeypatch):
    # Existing backend tests describe platform-independent behavior. Apple tests
    # opt in explicitly rather than depending on the machine running pytest.
    monkeypatch.setattr("podtran.macos.is_apple_silicon", lambda: False)


@pytest.fixture
def fake_audio_decoder(monkeypatch):
    # Task/CLI fixtures contain arbitrary bytes, not media. Keep the real
    # fingerprint index in use and replace only the external decoder boundary.
    monkeypatch.setattr(
        "podtran.fingerprints.hash_audio_content",
        lambda path: hashlib.sha256(b"decoded:" + path.read_bytes()).hexdigest(),
    )
