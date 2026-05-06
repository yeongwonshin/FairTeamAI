from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TeamMemberEvidence:
    name: str
    code_points: float = 0.0
    document_points: float = 0.0
    slide_points: float = 0.0
    meeting_points: float = 0.0
    role_points: float = 0.0
    total_points: float = 0.0
    contribution_share: float = 0.0
    raw_contribution_share: float = 0.0
    quality_adjusted_share: float = 0.0
    confidence_score: float = 0.0
    quality_score: float = 1.0
    anti_gaming_score: float = 1.0
    self_claim_share: Optional[float] = None
    overclaim_gap: Optional[float] = None
    completed_action_items: int = 0
    assigned_action_items: int = 0
    attendance_count: int = 0
    speaking_turns: int = 0
    risk_tags: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    audit_flags: List[str] = field(default_factory=list)
    source_coverage: Dict[str, bool] = field(default_factory=dict)


@dataclass
class FairnessReport:
    project_type: str
    weights: Dict[str, float]
    members: List[TeamMemberEvidence]
    gini: float
    imbalance_ratio: float
    conflict_risk_score: float
    conflict_evidence: List[str]
    summary: str
    professor_report_md: str
    team_report_md: str
    audit_rows: List[Dict[str, Any]] = field(default_factory=list)
    intervention_plan: List[str] = field(default_factory=list)
