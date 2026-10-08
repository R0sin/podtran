from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from urllib.request import getproxies

import httpx
import pytest

from podtran.network import proxy_environment


@pytest.mark.parametrize("proxy", ["http://proxy.example:7890", "", None])
def test_proxy_environment_routes_and_restores(monkeypatch, proxy):
    for name in ("http_proxy", "https_proxy", "all_proxy", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://inherited.example:8080")
    monkeypatch.setenv("NO_PROXY", "existing.internal")
    constants = SimpleNamespace(
        HF_HUB_DISABLE_XET=False, HF_HUB_ENABLE_HF_TRANSFER=True
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
    before = dict(os.environ)
    with pytest.raises(RuntimeError, match="stage failed"):
        with proxy_environment(
            proxy, ["localhost", "127.0.0.1", "::1", "service.internal"]
        ):
            with httpx.Client() as client:
                direct = client._transport
                assert (
                    client._transport_for_url(httpx.URL("http://localhost:8000"))
                    is direct
                )
                assert (
                    client._transport_for_url(httpx.URL("http://[::1]:8000")) is direct
                )
                assert (
                    client._transport_for_url(httpx.URL("https://service.internal"))
                    is direct
                )
                external = client._transport_for_url(
                    httpx.URL("https://translate.google.com")
                )
                assert (external is direct) == (proxy == "")
            from yt_dlp.utils.networking import select_proxy

            expected = proxy if proxy is not None else "http://inherited.example:8080"
            assert select_proxy("https://www.youtube.com", getproxies()) == (
                expected or None
            )
            assert select_proxy("http://localhost:8000", getproxies()) is None
            if proxy is not None:
                assert constants.HF_HUB_DISABLE_XET is True
                assert constants.HF_HUB_ENABLE_HF_TRANSFER is False
            raise RuntimeError("stage failed")
    assert dict(os.environ) == before
    assert constants.HF_HUB_DISABLE_XET is False
    assert constants.HF_HUB_ENABLE_HF_TRANSFER is True
