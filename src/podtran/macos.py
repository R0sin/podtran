from __future__ import annotations

import platform
import sys
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from podtran.config import AppConfig, ASRConfig

APPLE_GPU_DEVICES = frozenset({"mlx", "mps", "gpu"})


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine().lower() == "arm64"


def apple_gpu_available() -> bool:
    if not is_apple_silicon():
        return False
    version = platform.mac_ver()[0].split(".")[0]
    if not version or int(version) < 14:
        return False
    try:
        import mlx.core as mx
    except ImportError:
        return False
    return mx.metal.is_available()


def apple_device(requested: str) -> str:
    """Resolve only Apple devices; leave existing CPU/CUDA/XPU handling alone."""
    value = requested.strip().lower()
    explicit = value in APPLE_GPU_DEVICES
    if not explicit and (value != "auto" or not is_apple_silicon()):
        return value
    if apple_gpu_available():
        return "mlx"
    if explicit:
        raise RuntimeError("Apple GPU requires Apple Silicon and macOS 14 or newer.")
    warnings.warn(
        "Apple GPU is unavailable; auto mode is using CPU.",
        RuntimeWarning,
        stacklevel=2,
    )
    return "cpu"


def resolve_asr_config(config: ASRConfig) -> ASRConfig:
    device = apple_device(config.device)
    return config.model_copy(update={"device": device})


def resolve_tts_config(config: AppConfig) -> AppConfig:
    if config.tts.provider != "qwen-local":
        return config
    device = apple_device(config.providers.qwen_local.device)
    if device == config.providers.qwen_local.device:
        return config
    resolved = config.model_copy(deep=True)
    resolved.providers.qwen_local.device = device
    return resolved
