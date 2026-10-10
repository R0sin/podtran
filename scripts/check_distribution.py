from __future__ import annotations

import argparse
import os
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test a built wheel outside the source tree."
    )
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    root = Path(__file__).resolve().parents[1]
    version = runpy.run_path(str(root / "src/podtran/__init__.py"))["__version__"]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONUTF8"] = "1"
    env["NO_COLOR"] = "1"
    with tempfile.TemporaryDirectory(prefix="podtran-wheel-") as directory:
        cwd = Path(directory)
        venv = cwd / "venv"
        bin_dir = venv / ("Scripts" if os.name == "nt" else "bin")
        python = bin_dir / ("python.exe" if os.name == "nt" else "python")
        cli = bin_dir / ("podtran.exe" if os.name == "nt" else "podtran")

        def run(*command: str) -> str:
            result = subprocess.run(
                command,
                cwd=cwd,
                env=env,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            return result.stdout.strip()

        run("uv", "venv", "--python", sys.executable, str(venv))
        run(
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "-r",
            str(root / "scripts/ci-requirements.txt"),
        )
        run("uv", "pip", "install", "--python", str(python), "--no-deps", str(wheel))
        installed = run(
            str(python),
            "-c",
            "from importlib.metadata import version; print(version('podtran'))",
        )
        if installed != version:
            raise ValueError(
                f"Wheel version {installed} does not match source {version}."
            )
        if run(str(cli), "version") != f"podtran {version}":
            raise ValueError("Installed CLI reported an unexpected version.")
        run(str(cli), "--help")
        run(str(python), "-m", "podtran", "--help")
        print(f"Verified {wheel.name}: metadata, version, and CLI entry points.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(exc.stdout or "", file=sys.stderr)
        print(exc.stderr or "", file=sys.stderr)
        raise
