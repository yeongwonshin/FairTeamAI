from __future__ import annotations

"""Create a clean contest-submission zip without local/cache artifacts.

Run from the project root:
    python scripts/make_submission_zip.py

The output excludes .git, __pycache__, .pytest_cache, .DS_Store, __MACOSX, and
other local artifacts that should not be submitted to judges.
"""

from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs" / "FairTeamAI_submission_clean.zip"
EXCLUDE_DIRS = {".git", "__pycache__", ".pytest_cache", "__MACOSX", ".mypy_cache", ".ruff_cache"}
EXCLUDE_NAMES = {".DS_Store"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


def should_skip(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in EXCLUDE_DIRS for part in rel.parts):
        return True
    if path.name in EXCLUDE_NAMES:
        return True
    if path.suffix in EXCLUDE_SUFFIXES:
        return True
    if path == OUT:
        return True
    return False


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(OUT, "w", ZIP_DEFLATED) as zf:
        for path in sorted(ROOT.rglob("*")):
            if path.is_file() and not should_skip(path):
                zf.write(path, path.relative_to(ROOT))
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
