"""Resolve both Mac environments without installing packages or downloading models."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-mlx-lock", action="store_true")
    args = parser.parse_args()
    uv = shutil.which("uv")
    if uv is None:
        parser.error("uv must be on PATH")
    env = {**os.environ, "MACOSX_DEPLOYMENT_TARGET": "14.0"}
    with tempfile.TemporaryDirectory(prefix="podtran-macos-check-") as temporary:
        target = Path(temporary)
        common = [
            "--python-version",
            "3.11",
            "--python-platform",
            "aarch64-apple-darwin",
        ]
        exported = target / "main-locked.txt"
        subprocess.run(
            [
                uv,
                "export",
                "--frozen",
                "--no-dev",
                "--no-emit-project",
                "--no-hashes",
                "--output-file",
                str(exported),
            ],
            cwd=ROOT,
            env=env,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                uv,
                "pip",
                "compile",
                str(exported),
                *common,
                "--output-file",
                str(target / "main.txt"),
            ],
            cwd=ROOT,
            env=env,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        source = (
            "src/podtran/mlx-tts-requirements.in"
            if args.update_mlx_lock
            else "src/podtran/mlx-tts-requirements.txt"
        )
        destination = (
            ROOT / "src/podtran/mlx-tts-requirements.txt"
            if args.update_mlx_lock
            else target / "tts.txt"
        )
        subprocess.run(
            [
                uv,
                "pip",
                "compile",
                source,
                *common,
                "--output-file",
                str(destination.relative_to(ROOT))
                if args.update_mlx_lock
                else str(destination),
            ],
            cwd=ROOT,
            env=env,
            check=True,
            stdout=subprocess.DEVNULL,
        )
    print("Both macOS 14 / Python 3.11 dependency sets resolved successfully.")


if __name__ == "__main__":
    main()
