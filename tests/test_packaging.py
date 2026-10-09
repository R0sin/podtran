from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


def test_default_install_includes_qwen_and_compatible_numba() -> None:
    pyproject_path = Path(__file__).parents[1] / "pyproject.toml"
    with pyproject_path.open("rb") as file:
        pyproject = tomllib.load(file)

    dependencies = pyproject["project"]["dependencies"]

    assert "numba>=0.60" in dependencies
    assert "qwen-tts>=0.1.1,<0.2" in dependencies
    assert pyproject["project"]["optional-dependencies"]["qwen-local"] == []
