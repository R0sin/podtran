from __future__ import annotations

from contextlib import contextmanager
import os
import sys
from urllib.parse import urlsplit


def validate_proxy(value: str) -> str:
    value = value.strip()
    if not value:
        return value
    message = (
        "Proxy must be http://host:port without credentials, path, query or fragment."
    )
    try:
        url = urlsplit(value)
        valid = (
            url.scheme == "http"
            and url.hostname
            and url.port is not None
            and url.port > 0
            and url.username is None
            and url.password is None
            and url.path in {"", "/"}
            and not url.query
            and not url.fragment
            and not any(char.isspace() for char in value)
        )
    except ValueError:
        raise ValueError(message) from None
    if not valid:
        raise ValueError(message)
    return value


@contextmanager
def proxy_environment(proxy: str | None, no_proxy: list[str]):
    """Scope process-wide SDK/download settings to one CLI execution.

    None inherits proxies; an empty string forces direct connections.
    """
    keys = ("http_proxy", "https_proxy", "all_proxy", "no_proxy")
    names = [name for key in keys for name in (key, key.upper())]
    names += ["HF_HUB_DISABLE_XET", "HF_HUB_ENABLE_HF_TRANSFER"]
    previous = {name: os.environ.get(name) for name in names}
    constants = sys.modules.get("huggingface_hub.constants")
    hf_previous = {}
    try:
        if proxy is not None:
            validate_proxy(proxy)
            for key in keys[:3]:
                for name in (key, key.upper()):
                    os.environ[name] = proxy
            # Use the standard HTTP path instead of native download accelerators.
            for name, value in (
                ("HF_HUB_DISABLE_XET", True),
                ("HF_HUB_ENABLE_HF_TRANSFER", False),
            ):
                os.environ[name] = "1" if value else "0"
                if constants is not None and hasattr(constants, name):
                    hf_previous[name] = getattr(constants, name)
                    setattr(constants, name, value)
        bypass = list(no_proxy)
        inherited = os.environ.get("no_proxy", os.environ.get("NO_PROXY", ""))
        if inherited:
            bypass.append(inherited)
        bypass_value = "*" if proxy == "" else ",".join(bypass)
        for name in ("no_proxy", "NO_PROXY"):
            os.environ[name] = bypass_value
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        constants = sys.modules.get("huggingface_hub.constants")
        if constants is not None and proxy is not None:
            for name in ("HF_HUB_DISABLE_XET", "HF_HUB_ENABLE_HF_TRANSFER"):
                if hasattr(constants, name):
                    restored = hf_previous.get(
                        name,
                        (previous[name] or "").upper() in {"1", "ON", "YES", "TRUE"},
                    )
                    setattr(constants, name, restored)
