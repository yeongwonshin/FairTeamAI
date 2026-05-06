from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from .loaders import infer_members, load_project_bundle
from .reporting import members_to_rows
from .scoring import compute_fairness_report


def analyze_team_directory(team_dir: str | Path, project_type: str = "balanced") -> Dict[str, object]:
    team_dir = Path(team_dir)
    bundle = load_project_bundle(team_dir)
    members = infer_members(
        bundle["github_log"],
        bundle["docs_revision"],
        bundle["slides_revision"],
        bundle["roles"],
        bundle["self_eval"],
        meeting_notes=bundle["meeting_notes"],
    )
    report = compute_fairness_report(members=members, project_type=project_type, **bundle)
    return {"team": team_dir.name, "report": report, "rows": pd.DataFrame(members_to_rows(report.members))}


def analyze_many_teams(root_dir: str | Path, project_type: str = "balanced") -> pd.DataFrame:
    """Scan subdirectories that contain a FairTeam input bundle and compare teams."""
    root = Path(root_dir)
    rows: List[dict] = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        required = ["meeting_notes.txt", "github_log.csv", "docs_revision.csv", "slides_revision.csv", "roles.csv", "self_evaluation.csv"]
        if not all((child / name).exists() for name in required):
            continue
        result = analyze_team_directory(child, project_type=project_type)
        report = result["report"]
        rows.append(
            {
                "team": child.name,
                "members": len(report.members),
                "gini": round(report.gini, 4),
                "imbalance_ratio": round(report.imbalance_ratio, 4),
                "conflict_risk_score": round(report.conflict_risk_score, 4),
                "high_risk_members": sum(1 for m in report.members if m.risk_tags),
                "avg_confidence_score": round(sum(m.confidence_score for m in report.members) / max(len(report.members), 1), 4),
            }
        )
    return pd.DataFrame(rows)
