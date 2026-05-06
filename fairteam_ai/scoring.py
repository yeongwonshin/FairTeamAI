from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .analyzers import analyze_meeting_notes, numeric_sum_by_member
from .config import DEFAULT_WEIGHTS, RiskThresholds
from .models import FairnessReport, TeamMemberEvidence
from .reporting import build_professor_report, build_team_report, build_summary


def _safe_sqrt(x: float) -> float:
    return float(np.sqrt(max(float(x), 0.0)))


def _normalize(raw: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(v, 0.0) for v in raw.values())
    if total <= 0:
        n = max(len(raw), 1)
        return {k: 1.0 / n for k in raw}
    return {k: max(v, 0.0) / total for k, v in raw.items()}


def _gini(values: List[float]) -> float:
    arr = np.array(values, dtype=float)
    if len(arr) == 0 or np.sum(arr) == 0:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    cumulative = np.cumsum(arr)
    return float((n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n)


def _percent(x: float) -> str:
    return f"{x * 100:.1f}%"


def compute_fairness_report(
    *,
    members: List[str],
    meeting_notes: str,
    github_log: pd.DataFrame,
    docs_revision: pd.DataFrame,
    slides_revision: pd.DataFrame,
    roles: pd.DataFrame,
    self_eval: pd.DataFrame,
    project_type: str = "development",
    custom_weights: Dict[str, float] | None = None,
    thresholds: RiskThresholds | None = None,
) -> FairnessReport:
    """Compute a transparent contribution and fairness report.

    The score is intentionally evidence-based and auditable, not a hidden black box.
    Each category is normalized across the team first, then combined using project
    weights. This makes the final percentages comparable even when logs have
    different units such as commits, words, edits, and meeting turns.
    """
    members = sorted(set(str(m).strip() for m in members if str(m).strip()))
    if not members:
        raise ValueError("No team members found. Provide at least one member in the logs or member list.")

    thresholds = thresholds or RiskThresholds()
    weights = custom_weights or DEFAULT_WEIGHTS.get(project_type, DEFAULT_WEIGHTS["balanced"])
    total_w = sum(weights.values()) or 1.0
    weights = {k: v / total_w for k, v in weights.items()}

    meeting_df, conflict_risk, conflict_lines = analyze_meeting_notes(meeting_notes, members)

    code_data = numeric_sum_by_member(
        github_log,
        members,
        [
            "commits", "additions", "deletions", "files_changed", "issues_closed",
            "prs_merged", "reviews", "bugfix_commits", "test_commits",
        ],
    )
    doc_data = numeric_sum_by_member(
        docs_revision,
        members,
        ["edits", "words_added", "comments_resolved", "sections_owned", "suggestions_accepted", "references_added"],
    )
    slide_data = numeric_sum_by_member(
        slides_revision,
        members,
        ["slides_edited", "visuals_created", "script_words", "presenter_minutes"],
    )
    role_data = numeric_sum_by_member(
        roles,
        members,
        ["assigned_tasks", "completed_tasks", "late_tasks", "critical_tasks"],
    )
    meeting_data = {
        row["member"]: {
            "attendance_count": float(row["attendance_count"]),
            "speaking_turns": float(row["speaking_turns"]),
            "action_items_assigned": float(row["action_items_assigned"]),
            "action_items_completed": float(row["action_items_completed"]),
            "decision_mentions": float(row["decision_mentions"]),
        }
        for _, row in meeting_df.iterrows()
    }

    raw_code = {}
    raw_doc = {}
    raw_slide = {}
    raw_meeting = {}
    raw_role = {}

    for m in members:
        c = code_data[m]
        raw_code[m] = (
            4.0 * c["commits"]
            + 0.55 * _safe_sqrt(c["additions"] + c["deletions"])
            + 1.8 * c["files_changed"]
            + 4.5 * c["issues_closed"]
            + 5.0 * c["prs_merged"]
            + 2.5 * c["reviews"]
            + 4.0 * c["bugfix_commits"]
            + 3.2 * c["test_commits"]
        )
        d = doc_data[m]
        raw_doc[m] = (
            2.0 * d["edits"]
            + 0.35 * _safe_sqrt(d["words_added"])
            + 3.0 * d["comments_resolved"]
            + 7.0 * d["sections_owned"]
            + 3.5 * d["suggestions_accepted"]
            + 3.0 * d["references_added"]
        )
        s = slide_data[m]
        raw_slide[m] = (
            4.0 * s["slides_edited"]
            + 5.0 * s["visuals_created"]
            + 0.25 * _safe_sqrt(s["script_words"])
            + 3.0 * s["presenter_minutes"]
        )
        mt = meeting_data.get(m, {})
        raw_meeting[m] = (
            8.0 * mt.get("attendance_count", 0.0)
            + 2.0 * mt.get("speaking_turns", 0.0)
            + 4.0 * mt.get("action_items_assigned", 0.0)
            + 6.0 * mt.get("action_items_completed", 0.0)
            + 3.0 * mt.get("decision_mentions", 0.0)
        )
        r = role_data[m]
        completion_rate = r["completed_tasks"] / max(r["assigned_tasks"], 1.0)
        late_penalty = 0.6 * r["late_tasks"]
        raw_role[m] = max(0.0, 12.0 * completion_rate + 3.0 * r["completed_tasks"] + 3.5 * r["critical_tasks"] - late_penalty)

    code_norm = _normalize(raw_code)
    doc_norm = _normalize(raw_doc)
    slide_norm = _normalize(raw_slide)
    meeting_norm = _normalize(raw_meeting)
    role_norm = _normalize(raw_role)

    combined = {}
    for m in members:
        combined[m] = (
            weights.get("code", 0.0) * code_norm[m]
            + weights.get("document", 0.0) * doc_norm[m]
            + weights.get("slide", 0.0) * slide_norm[m]
            + weights.get("meeting", 0.0) * meeting_norm[m]
            + weights.get("role", 0.0) * role_norm[m]
        )
    # Normalize once more to remove rounding/weight leakage.
    share = _normalize(combined)

    self_claims = {}
    if self_eval is not None and not self_eval.empty and "member" in self_eval.columns:
        tmp = self_eval.copy()
        tmp["member"] = tmp["member"].astype(str).str.strip()
        if "self_claim_percent" in tmp.columns:
            tmp["self_claim_percent"] = pd.to_numeric(tmp["self_claim_percent"], errors="coerce") / 100.0
            self_claims = tmp.groupby("member")["self_claim_percent"].mean().dropna().to_dict()

    reports: List[TeamMemberEvidence] = []
    for m in members:
        ev = TeamMemberEvidence(
            name=m,
            code_points=code_norm[m],
            document_points=doc_norm[m],
            slide_points=slide_norm[m],
            meeting_points=meeting_norm[m],
            role_points=role_norm[m],
            total_points=combined[m],
            contribution_share=share[m],
            self_claim_share=self_claims.get(m),
            completed_action_items=int(meeting_data.get(m, {}).get("action_items_completed", 0)),
            assigned_action_items=int(meeting_data.get(m, {}).get("action_items_assigned", 0)),
            attendance_count=int(meeting_data.get(m, {}).get("attendance_count", 0)),
            speaking_turns=int(meeting_data.get(m, {}).get("speaking_turns", 0)),
        )
        if ev.self_claim_share is not None:
            ev.overclaim_gap = ev.self_claim_share - ev.contribution_share
        ev.evidence = _build_member_evidence(m, share[m], code_data[m], doc_data[m], slide_data[m], role_data[m], meeting_data.get(m, {}))
        ev.risk_tags = _build_risk_tags(ev, thresholds)
        reports.append(ev)

    values = [m.contribution_share for m in reports]
    gini = _gini(values)
    imbalance_ratio = max(values) / max(min(values), 1e-6)
    summary = build_summary(reports, gini, imbalance_ratio, conflict_risk)
    professor_report = build_professor_report(reports, gini, imbalance_ratio, conflict_risk, conflict_lines, project_type, weights)
    team_report = build_team_report(reports, conflict_risk, conflict_lines)
    return FairnessReport(
        project_type=project_type,
        weights=weights,
        members=reports,
        gini=gini,
        imbalance_ratio=imbalance_ratio,
        conflict_risk_score=conflict_risk,
        conflict_evidence=conflict_lines,
        summary=summary,
        professor_report_md=professor_report,
        team_report_md=team_report,
    )


def _build_member_evidence(member: str, share: float, code: Dict[str, float], doc: Dict[str, float], slide: Dict[str, float], role: Dict[str, float], meeting: Dict[str, float]) -> List[str]:
    evidence = [f"총 기여도 추정치: {_percent(share)}"]
    if code.get("commits", 0) or code.get("additions", 0) or code.get("prs_merged", 0):
        evidence.append(
            f"코드 로그: 커밋 {int(code.get('commits', 0))}건, PR 병합 {int(code.get('prs_merged', 0))}건, 변경 라인 {int(code.get('additions', 0) + code.get('deletions', 0))}줄"
        )
    if doc.get("edits", 0) or doc.get("words_added", 0):
        evidence.append(
            f"문서 로그: 편집 {int(doc.get('edits', 0))}회, 추가 단어 {int(doc.get('words_added', 0))}개, 담당 섹션 {int(doc.get('sections_owned', 0))}개"
        )
    if slide.get("slides_edited", 0) or slide.get("presenter_minutes", 0):
        evidence.append(
            f"발표자료 로그: 슬라이드 수정 {int(slide.get('slides_edited', 0))}장, 발표 담당 {float(slide.get('presenter_minutes', 0)):.1f}분"
        )
    if meeting.get("speaking_turns", 0) or meeting.get("action_items_assigned", 0):
        evidence.append(
            f"회의 로그: 발언 {int(meeting.get('speaking_turns', 0))}회, 액션아이템 {int(meeting.get('action_items_completed', 0))}/{int(meeting.get('action_items_assigned', 0))} 완료"
        )
    if role.get("assigned_tasks", 0):
        evidence.append(
            f"역할 이행: 배정 업무 {int(role.get('assigned_tasks', 0))}개 중 {int(role.get('completed_tasks', 0))}개 완료, 지연 {int(role.get('late_tasks', 0))}개"
        )
    if len(evidence) == 1:
        evidence.append("확인 가능한 작업 로그가 부족합니다. 원자료 추가 검토가 필요합니다.")
    return evidence


def _build_risk_tags(ev: TeamMemberEvidence, thresholds: RiskThresholds) -> List[str]:
    tags: List[str] = []
    if ev.contribution_share < thresholds.very_low_contribution_share:
        tags.append("고위험: 기여 로그 매우 부족")
    elif ev.contribution_share < thresholds.low_contribution_share:
        tags.append("주의: 기여 로그 부족")
    if ev.overclaim_gap is not None and ev.overclaim_gap > thresholds.high_overclaim_gap:
        tags.append("자기평가-실제로그 불일치 큼")
    if ev.contribution_share > thresholds.overload_share:
        tags.append("업무 과중 위험")
    if ev.assigned_action_items > 0 and ev.completed_action_items / max(ev.assigned_action_items, 1) < 0.5:
        tags.append("회의 액션아이템 완료율 낮음")
    return tags
