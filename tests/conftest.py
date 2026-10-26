from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_platform_detection(monkeypatch):
    # Existing backend tests describe platform-independent behavior. Apple tests
    # opt in explicitly rather than depending on the machine running pytest.
    monkeypatch.setattr("podtran.macos.is_apple_silicon", lambda: False)
