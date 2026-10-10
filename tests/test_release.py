from __future__ import annotations

import runpy
from pathlib import Path

import pytest


release_notes = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/prepare_release.py")
)["release_notes"]


@pytest.fixture
def release_root(tmp_path: Path) -> Path:
    package = tmp_path / "src/podtran"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('__version__ = "1.2.3"\n', encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text(
        "# 更新日志\n\n## [1.2.3] - 2026-10-10\n\n### 新增\n\n- 中文说明。\n\n"
        "## [1.2.2] - 2026-10-09\n\n- 旧版本。\n",
        encoding="utf-8",
    )
    return tmp_path


def test_release_notes_selects_only_matching_version(release_root: Path) -> None:
    assert release_notes(release_root, "v1.2.3") == "### 新增\n\n- 中文说明。\n"


@pytest.mark.parametrize("tag", ["1.2.3", "v1.2.3rc1", "v01.2.3", "v1.2.4"])
def test_release_rejects_invalid_or_mismatched_tag(
    release_root: Path, tag: str
) -> None:
    with pytest.raises(ValueError):
        release_notes(release_root, tag)


@pytest.mark.parametrize(
    "changelog",
    [
        "## [1.2.2] - 2026-10-10\n- wrong version\n",
        "## [1.2.3] - 2026-10-10\n### 新增\n",
        "## [1.2.3] - 2026-10-10\n- one\n## [1.2.3] - 2026-10-10\n- two\n",
    ],
)
def test_release_rejects_missing_empty_or_duplicate_notes(
    release_root: Path, changelog: str
) -> None:
    (release_root / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
    with pytest.raises(ValueError):
        release_notes(release_root, "v1.2.3")
