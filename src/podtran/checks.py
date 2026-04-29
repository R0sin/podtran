from __future__ import annotations

import subprocess
from pathlib import Path


def ensure_command(command: str) -> None:
    result = subprocess.run(
        [command, "-version"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{command} not found in PATH or failed to execute.")


def ensure_audio_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")


def ensure_hf_token(token: str) -> None:
    if not token.strip():
        raise RuntimeError(
            "Missing Hugging Face token. Set hf_token in config.toml and accept the speaker-diarization-community-1 model terms first."
        )
