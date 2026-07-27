from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


def test_qwen_local_prevents_incompatible_numba_backtracking() -> None:
    pyproject_path = Path(__file__).parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as file:
        pyproject = tomllib.load(file)

    qwen_local = pyproject["project"]["optional-dependencies"]["qwen-local"]

    assert "numba>=0.60" in qwen_local
