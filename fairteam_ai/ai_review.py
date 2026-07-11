from __future__ import annotations

import json
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from .models import FairnessReport
from .settings import get_openai_settings


class ReviewFinding(BaseModel):
    title: str
    severity: Literal["low", "medium", "high"]
    why_it_matters: str
    evidence_refs: list[str] = Field(default_factory=list)
    recommended_action: str


class AIReviewPayload(BaseModel):
    executive_summary: str
    decision_readiness: Literal["ready", "needs_more_evidence", "insufficient"]
    priority_findings: list[ReviewFinding] = Field(default_factory=list)
    questions_for_team: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class AIReviewBrief(AIReviewPayload):
    analysis_engine: str = "deterministic"
    model: str = ""

    def to_markdown(self) -> str:
        lines = ["# FairTeam AI Review Brief", "", self.executive_summary, ""]
        lines.append(f"**Decision readiness:** `{self.decision_readiness}`")
        lines.append("")
        lines.append("## Priority findings")
        if not self.priority_findings:
            lines.append("- No priority findings were generated.")
        for finding in self.priority_findings:
            lines.append(f"### {finding.title} ({finding.severity})")
            lines.append(f"- Why it matters: {finding.why_it_matters}")
            if finding.evidence_refs:
                lines.append("- Evidence: " + "; ".join(finding.evidence_refs))
            lines.append(f"- Recommended action: {finding.recommended_action}")
            lines.append("")
        lines.append("## Questions for the team")
        lines.extend(f"- {item}" for item in self.questions_for_team)
        lines.append("")
        lines.append("## Next steps")
        lines.extend(f"- {item}" for item in self.next_steps)
        lines.append("")
        lines.append("## Caveats")
        lines.extend(f"- {item}" for item in self.caveats)
        lines.append("")
        lines.append(f"_Generated with: {self.analysis_engine}{f' / {self.model}' if self.model else ''}_")
        return "\n".join(lines)


def _deterministic_brief(report: FairnessReport, readiness_status: str) -> AIReviewBrief:
    findings: list[ReviewFinding] = []
    sorted_members = sorted(report.members, key=lambda m: m.contribution_share)
    if sorted_members:
        lowest = sorted_members[0]
        if lowest.risk_tags:
            findings.append(
                ReviewFinding(
                    title=f"Review evidence for {lowest.name}",
                    severity="high" if lowest.contribution_share < 0.1 else "medium",
                    why_it_matters="The member has a low evidence-adjusted contribution share and one or more review tags.",
                    evidence_refs=lowest.risk_tags[:4],
                    recommended_action="Ask for missing offline evidence and compare it with the original project artifacts before making a decision.",
                )
            )
    if report.conflict_risk_score >= 0.25:
        findings.append(
            ReviewFinding(
                title="Resolve collaboration risk before final evaluation",
                severity="high" if report.conflict_risk_score >= 0.5 else "medium",
                why_it_matters="Meeting evidence contains delay, communication, conflict, or workload imbalance signals.",
                evidence_refs=report.conflict_evidence[:3],
                recommended_action="Run a short evidence-based review meeting with owners, deadlines, and disputed work listed explicitly.",
            )
        )
    flagged = [m for m in report.members if m.audit_flags]
    if flagged:
        findings.append(
            ReviewFinding(
                title="Verify quality and anti-gaming flags",
                severity="medium",
                why_it_matters="Volume-based activity can overstate contribution unless it is connected to reviewed outputs.",
                evidence_refs=[f"{m.name}: {', '.join(m.audit_flags[:2])}" for m in flagged[:3]],
                recommended_action="Open the referenced commits, documents, and revision history and verify substance rather than count alone.",
            )
        )

    readiness_map = {
        "Ready for human review": "ready",
        "Needs more evidence": "needs_more_evidence",
        "Insufficient for decision": "insufficient",
    }
    return AIReviewBrief(
        executive_summary=report.summary,
        decision_readiness=readiness_map.get(readiness_status, "needs_more_evidence"),
        priority_findings=findings,
        questions_for_team=[
            "Which important offline tasks are not represented in the uploaded evidence?",
            "Were responsibilities and deadlines agreed before the disputed work occurred?",
            "Can each claimed contribution be linked to a file, commit, review, decision, or completed action item?",
        ],
        next_steps=report.intervention_plan[:5] or ["Collect missing evidence and repeat the analysis."],
        caveats=[
            "This brief is decision support, not an automatic grade.",
            "Low activity in one source does not prove low contribution when work happened offline.",
            "Reviewers should inspect original artifacts for every material dispute.",
        ],
        analysis_engine="deterministic",
    )


def generate_ai_review_brief(
    *,
    report: FairnessReport,
    readiness_status: str,
    scores: pd.DataFrame,
    audit: pd.DataFrame,
    use_llm: bool = False,
    api_key: str | None = None,
    model: str | None = None,
) -> AIReviewBrief:
    fallback = _deterministic_brief(report, readiness_status)
    settings = get_openai_settings(api_key_override=api_key, model_override=model)
    if not use_llm or not settings.configured:
        return fallback

    compact_scores = scores[
        [c for c in ["member", "contribution_share", "confidence_score", "quality_score", "risk_tags"] if c in scores.columns]
    ].to_dict(orient="records")
    compact_audit = audit.head(20).to_dict(orient="records") if audit is not None and not audit.empty else []
    payload = {
        "summary": report.summary,
        "readiness_status": readiness_status,
        "project_type": report.project_type,
        "gini": report.gini,
        "imbalance_ratio": report.imbalance_ratio,
        "conflict_risk_score": report.conflict_risk_score,
        "member_scores": compact_scores,
        "quality_audit": compact_audit,
        "recommended_interventions": report.intervention_plan,
    }
    system = (
        "You are an evidence-review assistant for team projects. Produce a concise, neutral reviewer brief. "
        "Never assign a grade, accuse a person of misconduct, or treat missing digital evidence as proof of no contribution. "
        "Use only the supplied evidence. Prioritize verification questions and reversible interventions."
    )
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(
            api_key=settings.api_key,
            timeout=settings.timeout_seconds,
            max_retries=settings.max_retries,
        )
        response = client.responses.parse(
            model=settings.model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
            ],
            text_format=AIReviewPayload,
        )
        parsed = response.output_parsed
        if parsed is None:
            return fallback
        return AIReviewBrief(**parsed.model_dump(), analysis_engine="openai_responses", model=settings.model)
    except Exception:
        return fallback
