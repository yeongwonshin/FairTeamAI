from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


def _read_csv(path_or_buffer) -> pd.DataFrame:
    return pd.read_csv(path_or_buffer)


def safe_read_csv(path_or_buffer, required_columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Read a CSV and add missing required columns as empty values.

    This keeps the demo robust when a user uploads a partial log.
    """
    df = _read_csv(path_or_buffer)
    if required_columns:
        for col in required_columns:
            if col not in df.columns:
                df[col] = 0
    return df


def read_text(path_or_buffer) -> str:
    if hasattr(path_or_buffer, "read"):
        data = path_or_buffer.read()
        if isinstance(data, bytes):
            return data.decode("utf-8", errors="ignore")
        return str(data)
    return Path(path_or_buffer).read_text(encoding="utf-8", errors="ignore")


REQUIRED_GITHUB_COLUMNS = [
    "member", "commits", "additions", "deletions", "files_changed",
    "issues_closed", "prs_merged", "reviews", "bugfix_commits", "test_commits",
]

REQUIRED_DOC_COLUMNS = [
    "member", "edits", "words_added", "comments_resolved", "sections_owned",
    "suggestions_accepted", "references_added",
]

REQUIRED_SLIDE_COLUMNS = [
    "member", "slides_edited", "visuals_created", "script_words", "presenter_minutes",
]

REQUIRED_ROLE_COLUMNS = [
    "member", "assigned_tasks", "completed_tasks", "late_tasks", "critical_tasks",
]

REQUIRED_SELF_EVAL_COLUMNS = [
    "member", "self_claim_percent", "claimed_main_work", "peer_comment",
]


def load_project_bundle(base_dir: Path) -> Dict[str, object]:
    """Load the sample project bundle from a directory."""
    base_dir = Path(base_dir)
    return {
        "meeting_notes": read_text(base_dir / "meeting_notes.txt"),
        "github_log": safe_read_csv(base_dir / "github_log.csv", REQUIRED_GITHUB_COLUMNS),
        "docs_revision": safe_read_csv(base_dir / "docs_revision.csv", REQUIRED_DOC_COLUMNS),
        "slides_revision": safe_read_csv(base_dir / "slides_revision.csv", REQUIRED_SLIDE_COLUMNS),
        "roles": safe_read_csv(base_dir / "roles.csv", REQUIRED_ROLE_COLUMNS),
        "self_eval": safe_read_csv(base_dir / "self_evaluation.csv", REQUIRED_SELF_EVAL_COLUMNS),
    }


def infer_members(*frames: pd.DataFrame, meeting_notes: str = "") -> List[str]:
    members = set()
    for frame in frames:
        if frame is not None and "member" in frame.columns:
            members.update(str(x).strip() for x in frame["member"].dropna().tolist())
    # Prefer structured logs. Meeting-note parsing is only a fallback when no CSV
    # contains a member column, because headings such as "결정:" or "갈등:" can
    # otherwise be mistaken for people.
    if not members:
        blocked = {"date", "agenda", "결정", "todo", "action", "회의", "참석", "갈등", "담당자"}
        for line in meeting_notes.splitlines():
            stripped = line.strip()
            if ":" in stripped:
                name = stripped.split(":", 1)[0].strip().strip("[]")
                if 1 <= len(name) <= 20 and not any(ch.isdigit() for ch in name):
                    if name.lower() not in blocked and name not in blocked:
                        members.add(name)
    return sorted(m for m in members if m)
