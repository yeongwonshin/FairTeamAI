from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import pandas as pd


CONFLICT_KEYWORDS = [
    "안 했", "안함", "늦", "지각", "무임승차", "연락", "답장", "갈등", "불만",
    "혼자", "부담", "미완료", "누락", "failed", "late", "no response", "free rider",
    "conflict", "blocked", "unfair", "overload", "missing",
]

ACTION_PATTERNS = [
    re.compile(r"(?:담당|owner|assignee|담당자)\s*[:=]\s*([가-힣A-Za-z0-9_ -]{1,20})", re.I),
    re.compile(r"([가-힣A-Za-z0-9_ -]{1,20})\s*(?:님)?\s*(?:이|가)?\s*(?:담당|맡)", re.I),
    re.compile(r"@([A-Za-z0-9_가-힣-]{1,20})"),
]

DONE_KEYWORDS = ["완료", "done", "merged", "제출", "해결", "finished", "completed", "반영"]


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    low = text.lower()
    return any(k.lower() in low for k in keywords)


def analyze_meeting_notes(meeting_notes: str, members: List[str]) -> Tuple[pd.DataFrame, float, List[str]]:
    """Extract simple meeting participation/action-item/conflict signals.

    The goal is not to replace human judgment. The function produces auditable
    signals that are shown with the evidence text in the report.
    """
    rows = {
        m: {
            "member": m,
            "attendance_count": 0,
            "speaking_turns": 0,
            "action_items_assigned": 0,
            "action_items_completed": 0,
            "decision_mentions": 0,
        }
        for m in members
    }
    conflict_lines: List[str] = []
    total_lines = 0
    conflict_hits = 0

    for raw_line in meeting_notes.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        total_lines += 1
        lower = line.lower()
        if _contains_any(line, CONFLICT_KEYWORDS):
            conflict_hits += 1
            if len(conflict_lines) < 8:
                conflict_lines.append(line)
        for member in members:
            if member and re.search(rf"(?<!\w){re.escape(member)}(?!\w)", line):
                rows[member]["attendance_count"] += int(any(k in lower for k in ["참석", "attend", "present"]))
                rows[member]["speaking_turns"] += int(":" in line or "발언" in line or "said" in lower)
                rows[member]["decision_mentions"] += int(any(k in lower for k in ["결정", "decision", "확정", "agreed"]))
                if any(k in lower for k in ["todo", "action", "담당", "마감", "deadline", "해야"]):
                    rows[member]["action_items_assigned"] += 1
                    if _contains_any(line, DONE_KEYWORDS):
                        rows[member]["action_items_completed"] += 1
        for pattern in ACTION_PATTERNS:
            match = pattern.search(line)
            if match:
                name = match.group(1).strip()
                matched_member = next((m for m in members if m in name or name in m), None)
                if matched_member:
                    rows[matched_member]["action_items_assigned"] += 1
                    if _contains_any(line, DONE_KEYWORDS):
                        rows[matched_member]["action_items_completed"] += 1

    # Avoid zero attendance when notes state only speaker turns.
    for member in members:
        if rows[member]["attendance_count"] == 0 and rows[member]["speaking_turns"] > 0:
            rows[member]["attendance_count"] = 1
    conflict_risk = min(1.0, (conflict_hits / max(total_lines, 1)) * 5.0)
    return pd.DataFrame(rows.values()), conflict_risk, conflict_lines


def numeric_sum_by_member(df: pd.DataFrame, members: List[str], columns: List[str]) -> Dict[str, Dict[str, float]]:
    if df is None or df.empty or "member" not in df.columns:
        return {m: {col: 0.0 for col in columns} for m in members}
    out: Dict[str, Dict[str, float]] = {}
    clean = df.copy()
    clean["member"] = clean["member"].astype(str).str.strip()
    for col in columns:
        if col not in clean.columns:
            clean[col] = 0
        clean[col] = pd.to_numeric(clean[col], errors="coerce").fillna(0)
    grouped = clean.groupby("member")[columns].sum()
    for m in members:
        if m in grouped.index:
            out[m] = grouped.loc[m].to_dict()
        else:
            out[m] = {col: 0.0 for col in columns}
    return out
