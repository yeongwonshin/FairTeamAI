from __future__ import annotations

"""Meeting-note evidence structuring.

This module gives the project an AI-ready analysis layer without making the demo
fragile. If an OpenAI API key is provided explicitly, or OPENAI_API_KEY is
available in the environment, and the optional openai package is installed,
`extract_meeting_insights(..., use_llm=True)` can ask an LLM to return structured
JSON. Otherwise it falls back to a deterministic Korean/English rule extractor
that produces the same schema, so Streamlit and CLI demos always run.

The rule fallback is intentionally conservative. It does *targeted attribution*:
when a sentence says "참석: A, B, D. C는 연락이 늦고 답장이 없음", the negative
signal is assigned to C only, not to every member mentioned in the full line.
"deadline 05-03" is treated as an assignment/deadline record, not a missed
deadline unless the sentence also contains explicit failure/late wording.
"B가 C의 일을 대체 작성" becomes a positive recovery signal for B and a separate
review-needed substitution signal for C.
"""

import re
from dataclasses import asdict, dataclass
from typing import Iterable, List, Literal, Sequence

import pandas as pd
from pydantic import BaseModel, Field

from .settings import get_openai_settings


@dataclass
class MeetingInsight:
    member: str
    evidence_type: str
    polarity: str
    severity: float
    confidence: float
    source_sentence: str
    suggested_review_action: str
    analysis_engine: str = "rule_fallback"


INSIGHT_COLUMNS = [
    "member",
    "evidence_type",
    "polarity",
    "severity",
    "confidence",
    "source_sentence",
    "suggested_review_action",
    "analysis_engine",
]

# Conservative positive rules: these can safely be credited to the sentence's
# explicit speaker/actor.
POSITIVE_PATTERNS = {
    "completed_work": ["완료", "done", "finished", "completed", "submitted", "제출", "해결"],
    "review_activity": ["review", "리뷰", "comments resolved", "comment resolved", "코멘트 해결"],
    "merged_work": ["merged", "pr merged", "병합"],
    "shared_evidence": ["공유", "링크", "uploaded", "첨부"],
    "decision_leadership": ["결정", "확정", "agreed", "정리"],
}

# The word "deadline" or "마감" alone is intentionally NOT a missed-deadline
# signal. It is often used in normal task assignment notes.
NEGATIVE_REGEX = {
    "missed_deadline": [
        r"미\s*완료",
        r"누락",
        r"마감\s*(?:지남|초과|놓침|못|실패|늦)",
        r"(?:missed|past|overdue)\s+(?:the\s+)?deadline",
        r"deadline\s+(?:missed|past|overdue|failed)",
        r"not\s+done",
        r"missing",
        r"failed\s+to\s+(?:submit|finish|complete)",
        r"제출\s*(?:안|못|누락)",
    ],
    "no_response": [
        r"답장\s*(?:은|는|이|가)?\s*(?:없|안|늦|지연)",
        r"연락\s*(?:은|는|이|가)?\s*(?:없|안|두절|늦|지연)",
        r"no\s+response",
        r"unresponsive",
        r"not\s+respond(?:ing|ed)?",
    ],
    "late_attendance": [
        r"지각",
        r"late\s+(?:attendance|arrival)",
        r"arrived\s+late",
    ],
    "conflict": [
        r"갈등",
        r"불만",
        r"unfair",
        r"conflict",
        r"free\s+rider",
        r"무임승차",
    ],
    "low_reflection": [
        r"반영량\s*(?:은\s*)?적",
        r"기여\s*(?:부족|낮)",
        r"low\s+contribution",
    ],
}

ASSIGNMENT_PATTERNS = ["담당자", "담당", "owner", "assignee", "맡음", "todo", "action"]
OVERCLAIM_PATTERNS = ["오프라인 기여", "본인은", "주장", "claim", "self"]
SUBSTITUTION_PATTERNS = ["대체 작성", " 대신 ", "took over", "reassigned", "substituted"]
OVERLOAD_PATTERNS = ["혼자", "부담", "overload", "과중"]

_RE_LINE_SPLIT = re.compile(r"[\n\r]+")
_RE_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。])\s+|[;；]+\s*")


def _clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip(" -\t"))


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    low = text.lower()
    return any(k.lower() in low for k in keywords)


def _matches_any_regex(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text, flags=re.I) for pattern in patterns)


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


def _mentioned_members(text: str, members: Sequence[str]) -> List[str]:
    found: List[str] = []
    for member in members:
        if member and re.search(_member_regex(member), text):
            found.append(member)
    return found


def _split_sentences(raw_line: str) -> List[str]:
    """Split a note line into attribution-safe sentences/clauses.

    We split after sentence punctuation but do not split on commas, because Korean
    attendance lists such as "참석: A, B, D" use commas inside the same clause.
    """
    line = _clean_line(raw_line)
    if not line:
        return []
    parts = [_clean_line(p) for p in _RE_SENTENCE_SPLIT.split(line) if _clean_line(p)]
    # If a Korean/English period is followed by a new actor without whitespace in
    # odd copy-paste cases, add a second lightweight split.
    out: List[str] = []
    for part in parts:
        subparts = re.split(r"(?<=[.!?。])(?=[가-힣A-Za-z0-9_])", part)
        out.extend(_clean_line(x) for x in subparts if _clean_line(x))
    return out or [line]


def _sentence_prefix_speaker(sentence: str, members: Sequence[str]) -> str | None:
    # A: ..., 홍길동: ..., [A]: ...
    match = re.match(r"^\s*\[?([^:\]]{1,30})\]?\s*[:：]", sentence)
    if not match:
        return None
    prefix = match.group(1).strip()
    for member in members:
        if prefix == member:
            return member
    return None


def _members_before_phrase(sentence: str, phrase_regex: str, members: Sequence[str]) -> List[str]:
    match = re.search(phrase_regex, sentence, flags=re.I)
    if not match:
        return []
    before = sentence[: match.start()]
    return _mentioned_members(before, members)


def _explicit_subject_members(sentence: str, members: Sequence[str]) -> List[str]:
    """Return members that are grammatical actors/targets in the sentence.

    This handles Korean particles (A는/A가/A의), English @mentions, and prefix
    speakers. It deliberately avoids returning every member in a pure attendance
    list.
    """
    subjects: List[str] = []
    for member in members:
        if not member:
            continue
        escaped = re.escape(member)
        patterns = [
            rf"(?<![A-Za-z0-9_가-힣]){escaped}\s*(?:은|는|이|가|의|님은|님는|님이|님가)",
            rf"@{escaped}(?![A-Za-z0-9_가-힣])",
            rf"(?:by|from|for)\s+{escaped}\b",
        ]
        if any(re.search(p, sentence, flags=re.I) for p in patterns):
            subjects.append(member)
    speaker = _sentence_prefix_speaker(sentence, members)
    if speaker and speaker not in subjects:
        subjects.insert(0, speaker)
    return subjects


def _assignment_members(sentence: str, members: Sequence[str]) -> List[str]:
    """Extract explicit owners from assignment/TODO sentences."""
    # In substitution sentences, the assignment owner is usually the person whose
    # task was taken over, not the helper who performed the recovery action.
    if _contains_any(sentence, SUBSTITUTION_PATTERNS):
        original_owners: List[str] = []
        for member in members:
            if re.search(rf"{_member_regex(member)}\s*(?:은|는|이|가)?\s*[^.。;；]{{0,40}}맡은", sentence):
                original_owners.append(member)
        if original_owners:
            return list(dict.fromkeys(original_owners))
    found: List[str] = []
    for regex in [
        r"(?:담당자|담당|owner|assignee)\s*[:=]?\s*([^,.。;；\n]{1,80})",
        r"TODO\s*(?:담당자)?\s*[:=]?\s*([^,.。;；\n]{1,80})",
        r"action\s*(?:item)?\s*[:=]?\s*([^,.。;；\n]{1,80})",
    ]:
        for match in re.finditer(regex, sentence, flags=re.I):
            fragment = match.group(1)
            for m in _mentioned_members(fragment, members):
                if m not in found:
                    found.append(m)
    # "C가 자료조사를 맡음" / "B가 담당"
    for member in members:
        if re.search(rf"{_member_regex(member)}\s*(?:은|는|이|가)?\s*[^.。;；]{{0,40}}(?:맡|담당)", sentence):
            if member not in found:
                found.append(member)
    return found


def _targeted_members(sentence: str, members: Sequence[str]) -> List[str]:
    """Best-effort actor/target selection for one already-split sentence."""
    subjects = _explicit_subject_members(sentence, members)
    if subjects:
        return subjects
    assigned = _assignment_members(sentence, members)
    if assigned:
        return assigned
    mentioned = _mentioned_members(sentence, members)
    # Avoid pure attendance list attribution: "참석: A, B, D" is evidence of
    # attendance, not a positive/negative contribution insight.
    if re.match(r"^\s*(참석|present|attendance)\s*[:：]", sentence, flags=re.I):
        return []
    return mentioned


def _classify_negative(sentence: str) -> str | None:
    # "혼자 코드 부담" is a team-risk signal, but should not be treated as poor
    # contribution by the overloaded member even if the line is prefixed with
    # "갈등:".
    if _contains_any(sentence, OVERLOAD_PATTERNS):
        return "workload_overload"
    for evidence_type, patterns in NEGATIVE_REGEX.items():
        if _matches_any_regex(sentence, patterns):
            return evidence_type
    return None


def _classify_positive(sentence: str) -> str | None:
    # Avoid false positives where "완료" appears inside negative phrases such as
    # "미완료" or "완료 기록 부족".
    if re.search(r"(?<!미)완료(?!\s*(?:기록\s*)?부족)|\bdone\b|\bfinished\b|\bcompleted\b|\bsubmitted\b|제출|해결", sentence, flags=re.I):
        return "completed_work"
    for evidence_type, patterns in POSITIVE_PATTERNS.items():
        if evidence_type == "completed_work":
            continue
        if _contains_any(sentence, patterns):
            return evidence_type
    return None


def _substitution_pairs(sentence: str, members: Sequence[str]) -> tuple[List[str], List[str]]:
    """Return (recovery_members, substituted_task_members)."""
    if not _contains_any(sentence, SUBSTITUTION_PATTERNS):
        return [], []

    recovery: List[str] = []
    substituted: List[str] = []

    # "C가 맡은 ... B가 대체 작성" => C substituted, B recovery.
    for member in members:
        if re.search(rf"{_member_regex(member)}\s*(?:은|는|이|가)?\s*[^.。;；]{{0,40}}맡은", sentence):
            substituted.append(member)
        if re.search(rf"{_member_regex(member)}\s*(?:은|는|이|가)?\s*대체\s*작성", sentence):
            recovery.append(member)

    # "B가 C 대신 작성" / "B wrote instead of C". For recovery, require the
    # completion verb to be close to the same actor so "C가 맡은 ... B가 대체 작성"
    # does not incorrectly credit C.
    for member in members:
        if re.search(rf"{_member_regex(member)}\s*대신", sentence):
            substituted.append(member)
        if re.search(rf"{_member_regex(member)}\s*(?:은|는|이|가)?\s*(?:대체\s*)?(?:작성|완료|took\s+over)", sentence, flags=re.I):
            recovery.append(member)

    # Fallback: speaker/subject is recovery, owner/assigned member is substituted.
    if not recovery:
        speaker = _sentence_prefix_speaker(sentence, members)
        if speaker:
            recovery.append(speaker)
    if not substituted:
        substituted = [m for m in _assignment_members(sentence, members) if m not in recovery]

    return list(dict.fromkeys(recovery)), list(dict.fromkeys(substituted))


def _mk(
    member: str,
    evidence_type: str,
    polarity: str,
    severity: float,
    confidence: float,
    line: str,
    *,
    engine: str = "rule_fallback",
) -> MeetingInsight:
    action = {
        "positive": "해당 긍정 기여가 실제 산출물 로그와 연결되는지 확인",
        "negative": "해당 위험 신호가 일시적 상황인지 반복 패턴인지 원자료 재검토",
        "neutral": "역할 배정과 완료 로그가 일치하는지 확인",
    }.get(polarity, "원자료 재검토")
    if evidence_type == "workload_overload":
        action = "업무 과중/업무 배분 실패 여부를 확인하고 감점 신호로 단정하지 않기"
    if evidence_type == "task_substitution":
        action = "대체 작성이 발생한 원인과 원래 담당자의 실제 기여 증거를 함께 검토"
    return MeetingInsight(
        member=member,
        evidence_type=evidence_type,
        polarity=polarity,
        severity=float(max(0.0, min(1.0, severity))),
        confidence=float(max(0.0, min(1.0, confidence))),
        source_sentence=line,
        suggested_review_action=action,
        analysis_engine=engine,
    )


def _rule_based_extract(meeting_notes: str, members: Sequence[str]) -> List[MeetingInsight]:
    insights: List[MeetingInsight] = []
    for raw in _RE_LINE_SPLIT.split(str(meeting_notes)):
        for sentence in _split_sentences(raw):
            if not sentence or sentence.startswith("#"):
                continue
            mentioned = _mentioned_members(sentence, members)
            if not mentioned:
                continue

            # Assignment is neutral and should go only to explicit owners.
            assigned_members = _assignment_members(sentence, members)
            if assigned_members and _contains_any(sentence, ASSIGNMENT_PATTERNS):
                for member in assigned_members:
                    insights.append(_mk(member, "assigned_action_item", "neutral", 0.35, 0.76, sentence))

            # Substitution is handled before generic positive/negative matching so
            # "B가 C의 일을 대체 작성함" does not penalize B or credit C.
            recovery_members, substituted_members = _substitution_pairs(sentence, members)
            for member in recovery_members:
                insights.append(_mk(member, "task_recovery", "positive", 0.42, 0.76, sentence))
            for member in substituted_members:
                insights.append(_mk(member, "task_substitution", "negative", 0.50, 0.76, sentence))

            negative_hit = _classify_negative(sentence)
            positive_hit = _classify_positive(sentence)
            targets = _targeted_members(sentence, members)

            if negative_hit:
                # Workload overload is a team-risk review signal; keep it neutral so
                # the member who reports overload is not automatically penalized.
                polarity = "neutral" if negative_hit == "workload_overload" else "negative"
                if negative_hit in {"missed_deadline", "no_response", "conflict"}:
                    sev = 0.72
                elif negative_hit == "workload_overload":
                    sev = 0.44
                else:
                    sev = 0.55
                for member in targets:
                    insights.append(_mk(member, negative_hit, polarity, sev, 0.82, sentence))

            if positive_hit:
                sev = 0.50 if positive_hit in {"completed_work", "merged_work"} else 0.38
                # Prefer explicit speaker/subject over all mentioned members.
                positive_targets = targets or mentioned
                for member in positive_targets:
                    insights.append(_mk(member, positive_hit, "positive", sev, 0.78, sentence))

            if _contains_any(sentence, OVERCLAIM_PATTERNS):
                # Self-claim language should only attach to the claimant, not to
                # every person mentioned in the same sentence.
                claim_targets = _explicit_subject_members(sentence, members) or targets
                for member in claim_targets:
                    insights.append(_mk(member, "self_claim_needs_evidence", "negative", 0.42, 0.70, sentence))

    # Deduplicate exact repeated signals so one sentence does not dominate the report.
    unique = {}
    for item in insights:
        key = (item.member, item.evidence_type, item.polarity, item.source_sentence)
        if key not in unique or item.confidence > unique[key].confidence:
            unique[key] = item
    return list(unique.values())


class _OpenAIMeetingInsight(BaseModel):
    member: str
    evidence_type: Literal[
        "completed_work",
        "missed_deadline",
        "no_response",
        "conflict",
        "assigned_action_item",
        "review_activity",
        "self_claim_needs_evidence",
        "task_substitution",
        "task_recovery",
        "workload_overload",
        "merged_work",
        "shared_evidence",
        "decision_leadership",
        "other",
    ]
    polarity: Literal["positive", "negative", "neutral"]
    severity: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    source_sentence: str
    suggested_review_action: str


class _OpenAIMeetingPayload(BaseModel):
    insights: list[_OpenAIMeetingInsight] = Field(default_factory=list)


def _try_openai_extract(
    meeting_notes: str,
    members: Sequence[str],
    *,
    api_key: str | None = None,
    model: str | None = None,
) -> List[MeetingInsight] | None:
    settings = get_openai_settings(api_key_override=api_key, model_override=model)
    if not settings.configured:
        return None
    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        return None

    system_prompt = (
        "You are a neutral evidence-structuring assistant for team projects. "
        "Extract auditable contribution and fairness signals from meeting notes. "
        "Do not decide grades or accuse anyone of misconduct. Attribute each signal only to its actual actor or target. "
        "Attendance lists are not negative evidence, a deadline date alone is not a missed deadline, and workload overload "
        "must remain a neutral review signal rather than a penalty."
    )
    user_prompt = (
        f"Allowed member names: {list(members)}\n"
        "Every member field must exactly match one allowed name. Copy a short supporting sentence from the notes. "
        "Return no signal when the evidence is ambiguous.\n\n"
        f"Meeting notes:\n{meeting_notes}"
    )
    try:
        client = OpenAI(
            api_key=settings.api_key,
            timeout=settings.timeout_seconds,
            max_retries=settings.max_retries,
        )
        response = client.responses.parse(
            model=settings.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text_format=_OpenAIMeetingPayload,
        )
        parsed = response.output_parsed
        if parsed is None:
            return None
        member_set = set(members)
        insights: List[MeetingInsight] = []
        for rec in parsed.insights:
            if rec.member not in member_set:
                continue
            insights.append(
                MeetingInsight(
                    member=rec.member,
                    evidence_type=rec.evidence_type,
                    polarity=rec.polarity,
                    severity=float(rec.severity),
                    confidence=float(rec.confidence),
                    source_sentence=rec.source_sentence[:500],
                    suggested_review_action=rec.suggested_review_action[:300],
                    analysis_engine="openai_responses",
                )
            )
        return insights or None
    except Exception:
        return None

def extract_meeting_insights(
    meeting_notes: str,
    members: Sequence[str],
    *,
    use_llm: bool = False,
    openai_api_key: str | None = None,
    openai_model: str | None = None,
    llm_input_text: str | None = None,
) -> pd.DataFrame:
    """Return structured meeting evidence as a DataFrame.

    Columns are stable for both LLM and fallback modes, allowing the scorer,
    reports, and dashboard to treat the result as a first-class evidence source.
    """
    members = [str(m).strip() for m in members if str(m).strip()]
    llm_notes = meeting_notes if llm_input_text is None else llm_input_text
    insights = (
        _try_openai_extract(llm_notes, members, api_key=openai_api_key, model=openai_model)
        if use_llm
        else None
    )
    if insights is None:
        insights = _rule_based_extract(meeting_notes, members)
    if not insights:
        return pd.DataFrame(columns=INSIGHT_COLUMNS)
    df = pd.DataFrame([asdict(item) for item in insights])
    for col in INSIGHT_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col not in {"severity", "confidence"} else 0.0
    return df[INSIGHT_COLUMNS]


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
