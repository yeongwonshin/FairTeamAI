from __future__ import annotations

"""Meeting-note evidence structuring.

This module gives the project an AI-ready analysis layer without making the demo
fragile. If OPENAI_API_KEY is available and the optional openai package is
installed, `extract_meeting_insights(..., use_llm=True)` can ask an LLM to return
structured JSON. Otherwise it falls back to a deterministic Korean/English rule
extractor that produces the same schema, so Streamlit and CLI demos always run.
"""

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Iterable, List, Sequence

import pandas as pd


@dataclass
class MeetingInsight:
    member: str
    evidence_type: str
    polarity: str
    severity: float
    confidence: float
    source_sentence: str
    suggested_review_action: str


NEGATIVE_PATTERNS = {
    "missed_deadline": ["미완료", "누락", "deadline", "마감", "not done", "missing", "failed"],
    "no_response": ["연락", "답장", "no response", "unresponsive"],
    "late_attendance": ["지각", "late"],
    "conflict": ["갈등", "불만", "unfair", "conflict", "혼자", "부담", "free rider", "무임승차"],
    "low_reflection": ["반영량은 적음", "부족", "low contribution"],
}

POSITIVE_PATTERNS = {
    "completed_work": ["완료", "done", "finished", "completed", "submitted", "제출", "해결"],
    "review_activity": ["review", "리뷰", "comments resolved", "comment resolved"],
    "merged_work": ["merged", "pr merged", "병합"],
    "shared_evidence": ["공유", "링크", "uploaded", "첨부"],
    "decision_leadership": ["결정", "확정", "agreed", "정리"],
}

ASSIGNMENT_PATTERNS = ["담당자", "담당", "owner", "assignee", "맡음", "todo", "action"]
OVERCLAIM_PATTERNS = ["오프라인 기여", "본인은", "주장", "claim", "self"]
SUBSTITUTION_PATTERNS = ["대체 작성", " 대신 ", "took over", "reassigned"]


_RE_LINE_SPLIT = re.compile(r"[\n\r]+")


def _clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip(" -\t"))


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    low = text.lower()
    return any(k.lower() in low for k in keywords)


def _mentioned_members(line: str, members: Sequence[str]) -> List[str]:
    found: List[str] = []
    for member in members:
        if not member:
            continue
        # Handles names like A/B/C/D as well as Korean names. Avoids matching A in words.
        if re.search(rf"(?<![A-Za-z0-9_가-힣]){re.escape(member)}(?![A-Za-z0-9_가-힣])", line):
            found.append(member)
    return found


def _mk(member: str, evidence_type: str, polarity: str, severity: float, confidence: float, line: str) -> MeetingInsight:
    action = {
        "positive": "해당 긍정 기여가 실제 산출물 로그와 연결되는지 확인",
        "negative": "해당 위험 신호가 일시적 상황인지 반복 패턴인지 원자료 재검토",
        "neutral": "역할 배정과 완료 로그가 일치하는지 확인",
    }.get(polarity, "원자료 재검토")
    return MeetingInsight(
        member=member,
        evidence_type=evidence_type,
        polarity=polarity,
        severity=float(max(0.0, min(1.0, severity))),
        confidence=float(max(0.0, min(1.0, confidence))),
        source_sentence=line,
        suggested_review_action=action,
    )


def _rule_based_extract(meeting_notes: str, members: Sequence[str]) -> List[MeetingInsight]:
    insights: List[MeetingInsight] = []
    for raw in _RE_LINE_SPLIT.split(str(meeting_notes)):
        line = _clean_line(raw)
        if not line or line.startswith("#"):
            continue
        mentioned = _mentioned_members(line, members)
        if not mentioned:
            continue

        negative_hit = None
        for evidence_type, patterns in NEGATIVE_PATTERNS.items():
            if _contains_any(line, patterns):
                negative_hit = evidence_type
                break

        positive_hit = None
        for evidence_type, patterns in POSITIVE_PATTERNS.items():
            if _contains_any(line, patterns):
                positive_hit = evidence_type
                break

        for member in mentioned:
            if negative_hit:
                sev = 0.72 if negative_hit in {"missed_deadline", "no_response", "conflict"} else 0.55
                insights.append(_mk(member, negative_hit, "negative", sev, 0.78, line))
            if positive_hit:
                sev = 0.50 if positive_hit in {"completed_work", "merged_work"} else 0.38
                insights.append(_mk(member, positive_hit, "positive", sev, 0.74, line))
            if _contains_any(line, ASSIGNMENT_PATTERNS):
                insights.append(_mk(member, "assigned_action_item", "neutral", 0.35, 0.70, line))
            if _contains_any(line, OVERCLAIM_PATTERNS):
                insights.append(_mk(member, "self_claim_needs_evidence", "negative", 0.42, 0.66, line))
            if _contains_any(line, SUBSTITUTION_PATTERNS):
                insights.append(_mk(member, "task_substitution", "negative", 0.50, 0.68, line))

    # Deduplicate exact repeated signals so one sentence does not dominate the report.
    unique = {}
    for item in insights:
        key = (item.member, item.evidence_type, item.source_sentence)
        if key not in unique or item.confidence > unique[key].confidence:
            unique[key] = item
    return list(unique.values())


def _try_openai_extract(meeting_notes: str, members: Sequence[str]) -> List[MeetingInsight] | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return None

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    schema_hint = {
        "member": "exact member name from provided members",
        "evidence_type": "completed_work | missed_deadline | no_response | conflict | assignment | review_activity | self_claim_needs_evidence | task_substitution | other",
        "polarity": "positive | negative | neutral",
        "severity": "0.0-1.0",
        "confidence": "0.0-1.0",
        "source_sentence": "short source sentence copied from notes",
        "suggested_review_action": "one sentence for professor review",
    }
    prompt = (
        "You are an audit assistant for university team projects. Extract auditable contribution and fairness signals "
        "from meeting notes. Do not decide grades. Return only JSON array.\n"
        f"Members: {list(members)}\n"
        f"Schema: {json.dumps(schema_hint, ensure_ascii=False)}\n"
        f"Meeting notes:\n{meeting_notes}"
    )
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content or "{}"
        loaded = json.loads(text)
        records = loaded.get("insights", loaded if isinstance(loaded, list) else [])
        insights: List[MeetingInsight] = []
        for rec in records:
            member = str(rec.get("member", "")).strip()
            if member not in members:
                continue
            insights.append(
                MeetingInsight(
                    member=member,
                    evidence_type=str(rec.get("evidence_type", "other")),
                    polarity=str(rec.get("polarity", "neutral")),
                    severity=float(rec.get("severity", 0.3)),
                    confidence=float(rec.get("confidence", 0.5)),
                    source_sentence=str(rec.get("source_sentence", ""))[:500],
                    suggested_review_action=str(rec.get("suggested_review_action", "원자료 재검토"))[:300],
                )
            )
        return insights or None
    except Exception:
        return None


def extract_meeting_insights(meeting_notes: str, members: Sequence[str], *, use_llm: bool = False) -> pd.DataFrame:
    """Return structured meeting evidence as a DataFrame.

    Columns are stable for both LLM and fallback modes, allowing the scorer,
    reports, and dashboard to treat the result as a first-class evidence source.
    """
    members = [str(m).strip() for m in members if str(m).strip()]
    insights = _try_openai_extract(meeting_notes, members) if use_llm else None
    if insights is None:
        insights = _rule_based_extract(meeting_notes, members)
    if not insights:
        return pd.DataFrame(
            columns=["member", "evidence_type", "polarity", "severity", "confidence", "source_sentence", "suggested_review_action"]
        )
    return pd.DataFrame([asdict(item) for item in insights])


def summarize_insights_by_member(insights_df: pd.DataFrame, members: Sequence[str]) -> pd.DataFrame:
    rows = []
    if insights_df is None or insights_df.empty:
        for m in members:
            rows.append({"member": m, "positive_severity": 0.0, "negative_severity": 0.0, "neutral_count": 0, "insight_count": 0})
        return pd.DataFrame(rows)
    tmp = insights_df.copy()
    tmp["severity"] = pd.to_numeric(tmp.get("severity", 0), errors="coerce").fillna(0.0)
    for m in members:
        cur = tmp[tmp["member"].astype(str).str.strip() == str(m).strip()]
        rows.append(
            {
                "member": m,
                "positive_severity": float(cur.loc[cur["polarity"] == "positive", "severity"].sum()),
                "negative_severity": float(cur.loc[cur["polarity"] == "negative", "severity"].sum()),
                "neutral_count": int((cur["polarity"] == "neutral").sum()),
                "insight_count": int(len(cur)),
            }
        )
    return pd.DataFrame(rows)
