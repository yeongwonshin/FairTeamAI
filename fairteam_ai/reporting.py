from __future__ import annotations

from typing import Dict, Iterable, List

from .models import TeamMemberEvidence


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def build_summary(members: List[TeamMemberEvidence], gini: float, imbalance_ratio: float, conflict_risk: float) -> str:
    if not members:
        return "팀원 로그가 없습니다."
    sorted_members = sorted(members, key=lambda x: x.contribution_share, reverse=True)
    top = sorted_members[0]
    low = sorted_members[-1]
    risk_count = sum(1 for m in members if m.risk_tags)
    avg_conf = sum(m.confidence_score for m in members) / max(len(members), 1)
    avg_quality = sum(m.quality_score for m in members) / max(len(members), 1)
    return (
        f"최고 기여자는 {top.name}({pct(top.contribution_share)})이며, "
        f"최저 로그 기여자는 {low.name}({pct(low.contribution_share)})입니다. "
        f"기여 불균형 Gini={gini:.3f}, 최대/최소 비율={imbalance_ratio:.2f}, "
        f"갈등 위험={pct(conflict_risk)}로 추정됩니다. "
        f"평균 근거 품질={pct(avg_quality)}, 평균 산출 신뢰도={pct(avg_conf)}이며, "
        f"위험 태그가 붙은 팀원은 {risk_count}명입니다."
    )


def build_professor_report(
    members: List[TeamMemberEvidence],
    gini: float,
    imbalance_ratio: float,
    conflict_risk: float,
    conflict_lines: List[str],
    project_type: str,
    weights: Dict[str, float],
    intervention_plan: List[str] | None = None,
    meeting_insights=None,
    score_policy_md: str = "",
) -> str:
    intervention_plan = intervention_plan or []
    lines: List[str] = []
    lines.append("# 교수자용 공정평가 근거 리포트")
    lines.append("")
    lines.append("## 1. 요약 판단")
    lines.append(build_summary(members, gini, imbalance_ratio, conflict_risk))
    lines.append("")
    lines.append("## 2. 평가 설정")
    lines.append(f"- 프로젝트 유형: `{project_type}`")
    lines.append("- 가중치: " + ", ".join(f"{k}={pct(v)}" for k, v in weights.items()))
    lines.append("- 해석 원칙: 본 점수는 최종 성적이 아니라 Git/문서/회의/역할/자기평가 로그 기반의 검토 보조 지표입니다.")
    lines.append("- 보정 원칙: 원점수는 보존하되, 조작 의심·단일 출처 의존·근거 부족 신호가 있으면 품질 보정 점수를 함께 표시합니다.")
    lines.append("")
    if score_policy_md:
        lines.append("## 3. 산식 및 보정 근거")
        lines.extend(score_policy_md.replace("## 점수 산식 근거", "").strip().splitlines())
        lines.append("")
    lines.append("## 4. 팀원별 기여도 추정")
    lines.append("| 팀원 | 품질보정 기여도 | 원점수 기여도 | 신뢰도 | 근거품질 | 조작방어 | 코드 | 문서 | 발표 | 회의 | 역할 | 위험 태그 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for m in sorted(members, key=lambda x: x.contribution_share, reverse=True):
        tags = "; ".join(m.risk_tags) if m.risk_tags else "-"
        lines.append(
            f"| {m.name} | {pct(m.contribution_share)} | {pct(m.raw_contribution_share or m.contribution_share)} | "
            f"{pct(m.confidence_score)} | {pct(m.quality_score)} | {pct(m.anti_gaming_score)} | "
            f"{pct(m.code_points)} | {pct(m.document_points)} | {pct(m.slide_points)} | "
            f"{pct(m.meeting_points)} | {pct(m.role_points)} | {tags} |"
        )
    lines.append("")
    lines.append("## 5. 근거 로그")
    for m in sorted(members, key=lambda x: x.contribution_share, reverse=True):
        lines.append(f"### {m.name}")
        for item in m.evidence:
            lines.append(f"- {item}")
        if m.self_claim_share is not None:
            gap = (m.self_claim_share - m.contribution_share) * 100
            lines.append(f"- 자기평가 주장: {pct(m.self_claim_share)} / 보정 추정과의 차이: {gap:+.1f}%p")
        if m.source_coverage:
            coverage = ", ".join(k for k, ok in m.source_coverage.items() if ok) or "확인된 출처 없음"
            lines.append(f"- 확인된 로그 출처: {coverage}")
        if m.audit_flags:
            lines.append("- 조작/품질 검토 신호: " + "; ".join(m.audit_flags))
        if m.risk_tags:
            lines.append("- 검토 필요: " + "; ".join(m.risk_tags))
        lines.append("")
    lines.append("## 6. 갈등·무임승차 조기 신호")
    lines.append(f"- 갈등 위험 점수: {pct(conflict_risk)}")
    if conflict_lines:
        lines.append("- 감지된 문장 예시:")
        for line in conflict_lines[:6]:
            lines.append(f"  - {line}")
    else:
        lines.append("- 회의록에서 명시적 갈등 문장은 크게 감지되지 않았습니다.")
    lines.append("")
    if meeting_insights is not None and hasattr(meeting_insights, "empty") and not meeting_insights.empty:
        lines.append("## 7. 회의록 AI 구조화 신호")
        lines.append("| 팀원 | 유형 | 방향 | 심각도 | 신뢰도 | 근거 문장 |")
        lines.append("|---|---|---|---:|---:|---|")
        tmp = meeting_insights.copy()
        try:
            tmp["severity"] = tmp["severity"].astype(float)
            tmp = tmp.sort_values(["severity", "confidence"], ascending=False).head(12)
        except Exception:
            tmp = tmp.head(12)
        for _, row in tmp.iterrows():
            src = str(row.get("source_sentence", "")).replace("|", "/")[:140]
            lines.append(
                f"| {row.get('member', '')} | {row.get('evidence_type', '')} | {row.get('polarity', '')} | "
                f"{float(row.get('severity', 0)):.2f} | {float(row.get('confidence', 0)):.2f} | {src} |"
            )
        lines.append("")
    lines.append("## 8. 권장 조치")
    if intervention_plan:
        for item in intervention_plan:
            lines.append(f"- {item}")
    else:
        lines.append("- 로그 부족 팀원은 실제 오프라인 기여 증거를 추가 확인하십시오.")
        lines.append("- 자기평가와 로그 기반 추정 차이가 큰 경우, 팀원별 역할 산출물 원본을 함께 검토하십시오.")
    lines.append("- 업무 과중 팀원이 있으면 감점보다 업무 배분 실패 여부를 함께 검토하십시오.")
    lines.append("- 본 리포트는 자동 산출 근거 자료이며, 최종 평가는 교수자 판단으로 확정해야 합니다.")
    return "\n".join(lines)


def build_team_report(
    members: List[TeamMemberEvidence],
    conflict_risk: float,
    conflict_lines: List[str],
    intervention_plan: List[str] | None = None,
) -> str:
    intervention_plan = intervention_plan or []
    lines = ["# 팀원 공유용 협업 건강도 리포트", ""]
    lines.append("## 팀 상태 요약")
    if conflict_risk >= 0.5:
        lines.append("갈등 위험 신호가 높습니다. 역할 재조정 회의를 권장합니다.")
    elif conflict_risk >= 0.25:
        lines.append("일부 갈등 또는 지연 신호가 있습니다. 담당자와 마감일을 명확히 재확인하는 것이 좋습니다.")
    else:
        lines.append("명시적 갈등 신호는 낮습니다. 다만 로그 기반 기여 불균형은 주기적으로 확인하십시오.")
    lines.append("")
    lines.append("## 팀원별 피드백")
    for m in sorted(members, key=lambda x: x.contribution_share, reverse=True):
        lines.append(f"### {m.name}")
        lines.append(f"- 품질 보정 기여도 추정: {pct(m.contribution_share)}")
        lines.append(f"- 원점수 기여도 추정: {pct(m.raw_contribution_share or m.contribution_share)}")
        lines.append(f"- 산출 신뢰도: {pct(m.confidence_score)} / 근거 품질: {pct(m.quality_score)}")
        if m.risk_tags:
            lines.append(f"- 주의 신호: {'; '.join(m.risk_tags)}")
        else:
            lines.append("- 주요 위험 신호 없음")
        lines.append("- 다음 액션: 본인이 수행한 오프라인 기여가 누락되었다면 근거 파일이나 회의록을 추가하십시오.")
        lines.append("")
    if intervention_plan:
        lines.append("## 다음 회의 전 권장 액션")
        for item in intervention_plan:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines)


def members_to_rows(members: Iterable[TeamMemberEvidence]) -> List[dict]:
    rows = []
    for m in members:
        rows.append(
            {
                "member": m.name,
                "contribution_share": round(m.contribution_share, 4),
                "raw_contribution_share": round(m.raw_contribution_share or m.contribution_share, 4),
                "quality_adjusted_share": round(m.quality_adjusted_share or m.contribution_share, 4),
                "confidence_score": round(m.confidence_score, 4),
                "quality_score": round(m.quality_score, 4),
                "anti_gaming_score": round(m.anti_gaming_score, 4),
                "code_share": round(m.code_points, 4),
                "document_share": round(m.document_points, 4),
                "slide_share": round(m.slide_points, 4),
                "meeting_share": round(m.meeting_points, 4),
                "role_share": round(m.role_points, 4),
                "self_claim_share": None if m.self_claim_share is None else round(m.self_claim_share, 4),
                "overclaim_gap": None if m.overclaim_gap is None else round(m.overclaim_gap, 4),
                "source_coverage": ", ".join(k for k, ok in m.source_coverage.items() if ok),
                "audit_flags": "; ".join(m.audit_flags),
                "risk_tags": "; ".join(m.risk_tags),
            }
        )
    return rows
