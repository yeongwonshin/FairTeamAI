from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


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

BUNDLE_FILE_SPECS = {
    "meeting_notes": {
        "label": "회의록 TXT/MD",
        "filename": "meeting_notes.txt",
        "type": "text",
        "required_columns": [],
        "help": "회의 내용, 담당자, TODO, 갈등/완료 문장을 포함한 txt 또는 md 파일",
    },
    "github_log": {
        "label": "GitHub 로그 CSV",
        "filename": "github_log.csv",
        "type": "csv",
        "required_columns": REQUIRED_GITHUB_COLUMNS,
        "help": "member, commits, additions, deletions, files_changed 등 코드 기여 로그",
    },
    "docs_revision": {
        "label": "문서 수정 기록 CSV",
        "filename": "docs_revision.csv",
        "type": "csv",
        "required_columns": REQUIRED_DOC_COLUMNS,
        "help": "보고서/문서 편집량, 추가 단어, 댓글 해결, 참고문헌 추가 등",
    },
    "slides_revision": {
        "label": "발표자료 수정 기록 CSV",
        "filename": "slides_revision.csv",
        "type": "csv",
        "required_columns": REQUIRED_SLIDE_COLUMNS,
        "help": "슬라이드 수정, 시각자료 생성, 발표 스크립트, 발표 시간 등",
    },
    "roles": {
        "label": "역할분담/업무완료 CSV",
        "filename": "roles.csv",
        "type": "csv",
        "required_columns": REQUIRED_ROLE_COLUMNS,
        "help": "배정 업무, 완료 업무, 지연 업무, 핵심 업무 수",
    },
    "self_eval": {
        "label": "자기평가 CSV",
        "filename": "self_evaluation.csv",
        "type": "csv",
        "required_columns": REQUIRED_SELF_EVAL_COLUMNS,
        "help": "자기 주장 기여율, 주요 작업 주장, 동료 코멘트",
    },
}

_TEXT_DEFAULTS = {
    "member": "",
    "claimed_main_work": "",
    "peer_comment": "",
    "commit_message": "",
    "message": "",
    "commit_messages": "",
    "timestamp": "",
    "created_at": "",
    "updated_at": "",
    "date": "",
    "time": "",
}


def _rewind_if_possible(path_or_buffer) -> None:
    if hasattr(path_or_buffer, "seek"):
        try:
            path_or_buffer.seek(0)
        except Exception:
            pass


def _read_csv(path_or_buffer) -> pd.DataFrame:
    """Read CSV robustly from paths, file handles, or Streamlit UploadedFile."""
    _rewind_if_possible(path_or_buffer)
    if hasattr(path_or_buffer, "getvalue"):
        data = path_or_buffer.getvalue()
        for encoding in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
            try:
                from io import BytesIO

                return pd.read_csv(BytesIO(data), encoding=encoding)
            except UnicodeDecodeError:
                continue
        from io import BytesIO

        return pd.read_csv(BytesIO(data), encoding="utf-8", encoding_errors="ignore")
    return pd.read_csv(path_or_buffer)


def safe_read_csv(path_or_buffer, required_columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Read a CSV and add missing required columns as safe empty values.

    This keeps the demo robust when a user uploads a partial log. Missing numeric
    columns become 0; text columns such as member/comment become empty strings.
    """
    df = _read_csv(path_or_buffer)
    df.columns = [str(c).strip() for c in df.columns]
    if required_columns:
        for col in required_columns:
            if col not in df.columns:
                df[col] = _TEXT_DEFAULTS.get(col, 0)
    if "member" in df.columns:
        df["member"] = df["member"].astype(str).str.strip()
        df = df[df["member"].astype(str).str.len() > 0].copy()
    return df


def read_text(path_or_buffer) -> str:
    _rewind_if_possible(path_or_buffer)
    if hasattr(path_or_buffer, "read"):
        data = path_or_buffer.read()
        if isinstance(data, bytes):
            return data.decode("utf-8", errors="ignore")
        return str(data)
    return Path(path_or_buffer).read_text(encoding="utf-8", errors="ignore")


def load_project_bundle(base_dir: Path) -> Dict[str, object]:
    """Load a FairTeam project bundle from a directory."""
    base_dir = Path(base_dir)
    return {
        "meeting_notes": read_text(base_dir / "meeting_notes.txt"),
        "github_log": safe_read_csv(base_dir / "github_log.csv", REQUIRED_GITHUB_COLUMNS),
        "docs_revision": safe_read_csv(base_dir / "docs_revision.csv", REQUIRED_DOC_COLUMNS),
        "slides_revision": safe_read_csv(base_dir / "slides_revision.csv", REQUIRED_SLIDE_COLUMNS),
        "roles": safe_read_csv(base_dir / "roles.csv", REQUIRED_ROLE_COLUMNS),
        "self_eval": safe_read_csv(base_dir / "self_evaluation.csv", REQUIRED_SELF_EVAL_COLUMNS),
    }


def template_dataframe(required_columns: Iterable[str]) -> pd.DataFrame:
    """Return a one-row CSV template with stable column order."""
    row = {}
    for col in required_columns:
        if col == "member":
            row[col] = "A"
        elif col == "self_claim_percent":
            row[col] = 25
        elif col in _TEXT_DEFAULTS:
            row[col] = ""
        else:
            row[col] = 0
    return pd.DataFrame([row])


def dataframe_health(df: pd.DataFrame, required_columns: Iterable[str]) -> Tuple[List[str], int, int]:
    """Return missing columns, row count, and distinct member count."""
    missing = [col for col in required_columns if col not in df.columns]
    row_count = 0 if df is None else len(df)
    member_count = 0
    if df is not None and "member" in df.columns:
        member_count = int(df["member"].dropna().astype(str).str.strip().replace("", pd.NA).dropna().nunique())
    return missing, row_count, member_count


def bundle_health_rows(bundle: Dict[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for key, spec in BUNDLE_FILE_SPECS.items():
        if spec["type"] == "text":
            value = str(bundle.get(key, "") or "")
            rows.append(
                {
                    "source": key,
                    "required_file": spec["filename"],
                    "rows_or_chars": len(value),
                    "members": "-",
                    "missing_columns": "-",
                    "status": "OK" if value.strip() else "EMPTY",
                }
            )
            continue
        df = bundle.get(key)
        if not isinstance(df, pd.DataFrame):
            rows.append(
                {
                    "source": key,
                    "required_file": spec["filename"],
                    "rows_or_chars": 0,
                    "members": 0,
                    "missing_columns": ", ".join(spec["required_columns"]),
                    "status": "MISSING",
                }
            )
            continue
        missing, row_count, member_count = dataframe_health(df, spec["required_columns"])
        rows.append(
            {
                "source": key,
                "required_file": spec["filename"],
                "rows_or_chars": row_count,
                "members": member_count,
                "missing_columns": ", ".join(missing) if missing else "-",
                "status": "OK" if not missing and row_count > 0 else ("PARTIAL" if row_count > 0 else "EMPTY"),
            }
        )
    return rows


def save_bundle_to_directory(bundle: Dict[str, object], target_dir: Path) -> None:
    """Persist the currently active dashboard inputs for reproducibility."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "meeting_notes.txt").write_text(str(bundle.get("meeting_notes", "")), encoding="utf-8")
    for key, filename in [
        ("github_log", "github_log.csv"),
        ("docs_revision", "docs_revision.csv"),
        ("slides_revision", "slides_revision.csv"),
        ("roles", "roles.csv"),
        ("self_eval", "self_evaluation.csv"),
    ]:
        value = bundle.get(key)
        if isinstance(value, pd.DataFrame):
            value.to_csv(target_dir / filename, index=False)


def infer_members(*frames: pd.DataFrame, meeting_notes: str = "") -> List[str]:
    members = set()
    for frame in frames:
        if frame is not None and isinstance(frame, pd.DataFrame) and "member" in frame.columns:
            members.update(str(x).strip() for x in frame["member"].dropna().tolist() if str(x).strip())
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
