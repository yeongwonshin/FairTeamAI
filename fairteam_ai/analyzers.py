from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import pandas as pd


# Conflict words are used only for team-level risk lines, not direct member
# penalties. Detailed member-level negative attribution is handled in
# meeting_ai.extract_meeting_insights.
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


def _member_regex(member: str) -> str:
    escaped = re.escape(member)
    # Team member IDs are often A/B/C or GitHub handles. They may be followed by
    # Korean particles such as "C는" or "A가", so ASCII IDs only require ASCII
    # identifier boundaries. Korean names still avoid matching inside longer names
    # while allowing common Korean particles after the name.
    if re.fullmatch(r"[A-Za-z0-9_.@-]+", member):
        return rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])"
    particles = "은는이가의을를과와님"
    return rf"(?<![A-Za-z0-9_가-힣]){escaped}(?=(?:[{particles}])|(?![A-Za-z0-9_가-힣]))"


def _mentioned_members(text: str, members: List[str]) -> List[str]:
    return [m for m in members if m and re.search(_member_regex(m), text)]


def _split_sentences(line: str) -> List[str]:
    line = re.sub(r"\s+", " ", str(line).strip())
    if not line:
        return []
    parts = re.split(r"(?<=[.!?。])\s+|[;；]+\s*", line)
    return [p.strip() for p in parts if p.strip()]


def _attendance_members(line: str, members: List[str]) -> List[str]:
    """Return only the members inside the explicit attendance segment.

    Example: "참석: A, B, D. C는 연락이 늦고 답장이 없음" returns A/B/D only.
    The old fallback scanned the whole line and incorrectly counted C as present.
    """
    match = re.search(r"(?:참석|present|attendance)\s*[:：]\s*([^.!?。;；\n]+)", line, flags=re.I)
    if not match:
        return []
    segment = match.group(1)
    return _mentioned_members(segment, members)


def _line_speaker(line: str, members: List[str]) -> str | None:
    match = re.match(r"^\s*\[?([^:\]]{1,30})\]?\s*[:：]", line)
    if not match:
        return None
    prefix = match.group(1).strip()
    return prefix if prefix in members else None


def _assignment_members(line: str, members: List[str]) -> List[str]:
    if any(k in line for k in ["대체 작성", " 대신 ", "took over", "reassigned", "substituted"]):
        original_owners: List[str] = []
        for member in members:
            if re.search(rf"{_member_regex(member)}\s*(?:은|는|이|가)?\s*[^.。;；]{{0,40}}맡은", line):
                original_owners.append(member)
        if original_owners:
            return list(dict.fromkeys(original_owners))
    found: List[str] = []
    for pattern in ACTION_PATTERNS:
        match = pattern.search(line)
        if not match:
            continue
        name = match.group(1).strip()
        for m in members:
            if re.search(_member_regex(m), name) or name == m:
                if m not in found:
                    found.append(m)
    # TODO 담당자: C ... / 담당자: A, deadline ...
    for regex in [
        r"(?:TODO\s*)?(?:담당자|담당|owner|assignee)\s*[:=]?\s*([^,.。;；\n]{1,80})",
        r"(?:action\s*item|todo)\s*[:=]?\s*([^,.。;；\n]{1,80})",
    ]:
        for match in re.finditer(regex, line, flags=re.I):
            for m in _mentioned_members(match.group(1), members):
                if m not in found:
                    found.append(m)
    return found


def analyze_meeting_notes(meeting_notes: str, members: List[str]) -> Tuple[pd.DataFrame, float, List[str]]:
    """Extract simple meeting participation/action-item/conflict signals.

    The goal is not to replace human judgment. The function produces auditable
    signals that are shown with the evidence text in the report. It avoids broad
    line-level attribution errors by parsing explicit attendance and assignment
    segments before member matching.
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

    for raw_line in str(meeting_notes).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        total_lines += 1
        lower = line.lower()

        if _contains_any(line, CONFLICT_KEYWORDS):
            conflict_hits += 1
            if len(conflict_lines) < 8:
                conflict_lines.append(line)

        for member in _attendance_members(line, members):
            rows[member]["attendance_count"] += 1

        for sentence in _split_sentences(line):
            sentence_lower = sentence.lower()
            speaker = _line_speaker(sentence, members)
            if speaker:
                rows[speaker]["speaking_turns"] += 1
                rows[speaker]["decision_mentions"] += int(any(k in sentence_lower for k in ["결정", "decision", "확정", "agreed"]))

            # Decision lines without a named speaker can mention a member, but they
            # should not become that member's speaking turn.
            for member in _mentioned_members(sentence, members):
                rows[member]["decision_mentions"] += int(any(k in sentence_lower for k in ["결정", "decision", "확정", "agreed"]))

            assigned = _assignment_members(sentence, members)
            for matched_member in assigned:
                rows[matched_member]["action_items_assigned"] += 1
                if _contains_any(sentence, DONE_KEYWORDS):
                    rows[matched_member]["action_items_completed"] += 1

            # "A: ... 완료" without an explicit 담당자 should count as A completing
            # an action/turn, but deadline-only assignment is not completion.
            if speaker and _contains_any(sentence, DONE_KEYWORDS):
                rows[speaker]["action_items_completed"] += 1

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
