from __future__ import annotations

import argparse
import re
import runpy
from pathlib import Path


def release_notes(root: Path, tag: str) -> str:
    if not re.fullmatch(r"v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", tag):
        raise ValueError("Release tags must use vX.Y.Z (stable releases only).")
    version = runpy.run_path(str(root / "src/podtran/__init__.py"))["__version__"]
    if tag != f"v{version}":
        raise ValueError(f"Tag {tag} does not match package version {version}.")
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    sections = re.split(r"^## ", changelog, flags=re.MULTILINE)[1:]
    matches = [
        section.partition("\n")[2].strip()
        for section in sections
        if re.fullmatch(
            rf"\[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}",
            section.partition("\n")[0].strip(),
        )
    ]
    if len(matches) != 1 or not re.search(r"(?m)^- \S", matches[0]):
        raise ValueError(
            f"CHANGELOG.md must contain one nonempty section for {version}."
        )
    return matches[0] + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a release and extract its notes."
    )
    parser.add_argument("tag")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    notes = release_notes(Path(__file__).resolve().parents[1], args.tag)
    args.output.write_text(notes, encoding="utf-8")


if __name__ == "__main__":
    main()
