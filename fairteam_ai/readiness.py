from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Mapping

from .models import FairnessReport


@dataclass(frozen=True)
class ReviewReadiness:
    score: float
    status: str
    evidence_coverage: float
    average_confidence: float
    unresolved_appeals: int
    critical_flags: int
    blockers: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def calculate_review_readiness(
    *,
    report: FairnessReport,
    bundle_health: Iterable[Mapping[str, object]],
    unresolved_appeals: int = 0,
) -> ReviewReadiness:
    health = list(bundle_health)
    if health:
        usable = sum(1 for row in health if str(row.get("status", "")).upper() in {"OK", "PARTIAL"})
        evidence_coverage = usable / len(health)
    else:
        evidence_coverage = 0.0

    average_confidence = (
        sum(float(member.confidence_score) for member in report.members) / max(len(report.members), 1)
    )
    critical_flags = sum(
        1
        for member in report.members
        for flag in member.audit_flags
        if any(token in flag for token in ["대량", "조작", "과장", "반복", "부족"])
    )
    high_risk_members = sum(1 for member in report.members if member.risk_tags)

    score = (
        evidence_coverage * 0.42
        + average_confidence * 0.43
        + max(0.0, 1.0 - report.conflict_risk_score) * 0.15
    )
    score -= min(0.18, unresolved_appeals * 0.04)
    score -= min(0.15, critical_flags * 0.025)
    score = max(0.0, min(1.0, score))

    blockers: list[str] = []
    strengths: list[str] = []
    if evidence_coverage < 0.75:
        blockers.append("One or more core evidence sources are missing or empty.")
    else:
        strengths.append("Most core evidence sources are available.")
    if average_confidence < 0.55:
        blockers.append("Average scoring confidence is below the recommended review threshold.")
    else:
        strengths.append("The analysis has usable cross-source confidence.")
    if unresolved_appeals:
        blockers.append(f"{unresolved_appeals} appeal(s) still require a reviewer decision.")
    if critical_flags:
        blockers.append(f"{critical_flags} quality or anti-gaming flag(s) require source verification.")
    if high_risk_members == 0:
        strengths.append("No member currently has a high-risk review tag.")

    if score >= 0.78 and not unresolved_appeals and evidence_coverage >= 0.8:
        status = "Ready for human review"
    elif score >= 0.52:
        status = "Needs more evidence"
    else:
        status = "Insufficient for decision"

    return ReviewReadiness(
        score=round(score, 4),
        status=status,
        evidence_coverage=round(evidence_coverage, 4),
        average_confidence=round(average_confidence, 4),
        unresolved_appeals=int(unresolved_appeals),
        critical_flags=int(critical_flags),
        blockers=blockers,
        strengths=strengths,
    )
