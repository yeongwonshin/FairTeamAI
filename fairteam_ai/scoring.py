from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .analyzers import analyze_meeting_notes, numeric_sum_by_member
from .config import DEFAULT_WEIGHTS, RiskThresholds
from .interventions import build_intervention_plan
from .meeting_ai import extract_meeting_insights, summarize_insights_by_member
from .models import FairnessReport, TeamMemberEvidence
from .quality import build_quality_audit, audit_rows_to_dataframe
from .reporting import build_professor_report, build_team_report, build_summary
from .scoring_policy import (
    build_scoring_policy_markdown,
    score_code_signal,
    score_document_signal,
    score_meeting_signal,
    score_role_signal,
    score_slide_signal,
)


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


def _insight_lines(insights_df: pd.DataFrame, limit: int = 8) -> List[str]:
    if insights_df is None or insights_df.empty:
        return []
    tmp = insights_df.copy()
    tmp["severity"] = pd.to_numeric(tmp.get("severity", 0), errors="coerce").fillna(0)
    negative = tmp[tmp["polarity"].astype(str) == "negative"].sort_values("severity", ascending=False)
    lines = []
    for _, row in negative.head(limit).iterrows():
        sent = str(row.get("source_sentence", "")).strip()
        if sent and sent not in lines:
            lines.append(sent)
    return lines


def _insight_map(summary_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    if summary_df is None or summary_df.empty:
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for _, row in summary_df.iterrows():
        out[str(row["member"])] = {
            "positive_severity": float(row.get("positive_severity", 0.0)),
            "negative_severity": float(row.get("negative_severity", 0.0)),
            "neutral_count": float(row.get("neutral_count", 0.0)),
            "insight_count": float(row.get("insight_count", 0.0)),
        }
    return out


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
    use_llm_meeting_analysis: bool = False,
    openai_api_key: str | None = None,
    openai_model: str | None = None,
    llm_meeting_notes: str | None = None,
    meeting_insights: pd.DataFrame | None = None,
) -> FairnessReport:
    """Compute a transparent contribution and fairness report.

    The score is evidence-based and auditable, not a hidden black box. Each
    category is normalized across the team, then combined with project weights.
    Meeting notes are additionally converted into structured review signals via
    an optional LLM path with deterministic fallback.
    """
    members = sorted(set(str(m).strip() for m in members if str(m).strip()))
    if not members:
        raise ValueError("No team members found. Provide at least one member in the logs or member list.")

    thresholds = thresholds or RiskThresholds()
    weights = custom_weights or DEFAULT_WEIGHTS.get(project_type, DEFAULT_WEIGHTS["balanced"])
    total_w = sum(weights.values()) or 1.0
    weights = {k: v / total_w for k, v in weights.items()}

    meeting_df, rule_conflict_risk, conflict_lines = analyze_meeting_notes(meeting_notes, members)
    if meeting_insights is None:
        meeting_insights = extract_meeting_insights(
            meeting_notes,
            members,
            use_llm=use_llm_meeting_analysis,
            openai_api_key=openai_api_key,
            openai_model=openai_model,
            llm_input_text=llm_meeting_notes,
        )
    insight_summary = summarize_insights_by_member(meeting_insights, members)
    insight_data = _insight_map(insight_summary)

    negative_severity_total = 0.0
    if meeting_insights is not None and not meeting_insights.empty:
        tmp = meeting_insights.copy()
        tmp["severity"] = pd.to_numeric(tmp.get("severity", 0), errors="coerce").fillna(0.0)
        negative_severity_total = float(tmp.loc[tmp["polarity"] == "negative", "severity"].sum())
    insight_conflict_risk = min(1.0, negative_severity_total / max(len(members) * 2.0, 1.0))
    conflict_risk = max(rule_conflict_risk, insight_conflict_risk)
    conflict_lines = list(dict.fromkeys(conflict_lines + _insight_lines(meeting_insights)))[:12]

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
        raw_code[m] = score_code_signal(code_data[m])
        raw_doc[m] = score_document_signal(doc_data[m])
        raw_slide[m] = score_slide_signal(slide_data[m])
        raw_meeting[m] = score_meeting_signal(meeting_data.get(m, {}), insight_data.get(m, {}))
        raw_role[m] = score_role_signal(role_data[m])

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
    raw_share = _normalize(combined)

    quality_audits = build_quality_audit(
        members=members,
        github_log=github_log,
        docs_revision=docs_revision,
        slides_revision=slides_revision,
        roles=roles,
        self_eval=self_eval,
        meeting_insights=meeting_insights,
    )
    adjusted_base = {m: raw_share[m] * (0.60 + 0.40 * quality_audits[m].quality_score) for m in members}
    share = _normalize(adjusted_base)

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
            raw_contribution_share=raw_share[m],
            quality_adjusted_share=share[m],
            confidence_score=quality_audits[m].confidence_score,
            quality_score=quality_audits[m].quality_score,
            anti_gaming_score=quality_audits[m].anti_gaming_score,
            self_claim_share=self_claims.get(m),
            audit_flags=list(quality_audits[m].flags),
            source_coverage=dict(quality_audits[m].source_coverage),
            completed_action_items=int(meeting_data.get(m, {}).get("action_items_completed", 0)),
            assigned_action_items=int(meeting_data.get(m, {}).get("action_items_assigned", 0)),
            attendance_count=int(meeting_data.get(m, {}).get("attendance_count", 0)),
            speaking_turns=int(meeting_data.get(m, {}).get("speaking_turns", 0)),
        )
        if ev.self_claim_share is not None:
            ev.overclaim_gap = ev.self_claim_share - ev.contribution_share
        ev.evidence = _build_member_evidence(
            m,
            share[m],
            code_data[m],
            doc_data[m],
            slide_data[m],
            role_data[m],
            meeting_data.get(m, {}),
            insight_data=insight_data.get(m, {}),
            raw_share=raw_share[m],
            quality_score=quality_audits[m].quality_score,
            confidence_score=quality_audits[m].confidence_score,
        )
        ev.risk_tags = _build_risk_tags(ev, thresholds, insight_data.get(m, {}))
        if quality_audits[m].flags:
            ev.risk_tags.extend([f"검토: {flag}" for flag in quality_audits[m].flags])
        reports.append(ev)

    values = [m.contribution_share for m in reports]
    gini = _gini(values)
    imbalance_ratio = max(values) / max(min(values), 1e-6)
    summary = build_summary(reports, gini, imbalance_ratio, conflict_risk)
    intervention_plan = build_intervention_plan(reports, conflict_risk, imbalance_ratio)
    audit_rows = audit_rows_to_dataframe(quality_audits).to_dict(orient="records")
    score_policy_md = build_scoring_policy_markdown()
    professor_report = build_professor_report(
        reports,
        gini,
        imbalance_ratio,
        conflict_risk,
        conflict_lines,
        project_type,
        weights,
        intervention_plan,
        meeting_insights=meeting_insights,
        score_policy_md=score_policy_md,
    )
    team_report = build_team_report(reports, conflict_risk, conflict_lines, intervention_plan)
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
        audit_rows=audit_rows,
        intervention_plan=intervention_plan,
        meeting_insights=[] if meeting_insights is None else meeting_insights.to_dict(orient="records"),
        score_policy_md=score_policy_md,
    )


def _build_member_evidence(
    member: str,
    share: float,
    code: Dict[str, float],
    doc: Dict[str, float],
    slide: Dict[str, float],
    role: Dict[str, float],
    meeting: Dict[str, float],
    *,
    insight_data: Dict[str, float] | None = None,
    raw_share: float | None = None,
    quality_score: float | None = None,
    confidence_score: float | None = None,
) -> List[str]:
    insight_data = insight_data or {}
    evidence = [f"품질 보정 후 기여도 추정치: {_percent(share)}"]
    if raw_share is not None and abs(raw_share - share) >= 0.005:
        evidence.append(f"원점수 기준 기여도: {_percent(raw_share)} / 품질·조작 신호 반영 후: {_percent(share)}")
    if quality_score is not None or confidence_score is not None:
        evidence.append(f"근거 품질 점수: {_percent(quality_score or 0.0)}, 산출 신뢰도: {_percent(confidence_score or 0.0)}")
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
    if insight_data.get("insight_count", 0):
        evidence.append(
            f"회의 AI 구조화 신호: 긍정 severity {insight_data.get('positive_severity', 0.0):.2f}, 부정/검토 severity {insight_data.get('negative_severity', 0.0):.2f}, 총 {int(insight_data.get('insight_count', 0))}건"
        )
    if role.get("assigned_tasks", 0):
        evidence.append(
            f"역할 이행: 배정 업무 {int(role.get('assigned_tasks', 0))}개 중 {int(role.get('completed_tasks', 0))}개 완료, 지연 {int(role.get('late_tasks', 0))}개"
        )
    if len(evidence) == 1:
        evidence.append("확인 가능한 작업 로그가 부족합니다. 원자료 추가 검토가 필요합니다.")
    return evidence


def _build_risk_tags(ev: TeamMemberEvidence, thresholds: RiskThresholds, insight_data: Dict[str, float] | None = None) -> List[str]:
    insight_data = insight_data or {}
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
    if insight_data.get("negative_severity", 0.0) >= 1.2:
        tags.append("회의록 구조화 분석상 반복 검토 신호")
    return tags
