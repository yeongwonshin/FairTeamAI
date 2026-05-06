from __future__ import annotations

from pathlib import Path

import pandas as pd

from fairteam_ai.loaders import infer_members, load_project_bundle
from fairteam_ai.reporting import members_to_rows
from fairteam_ai.scoring import compute_fairness_report

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "sample_data"
OUT_DIR = ROOT / "outputs"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    bundle = load_project_bundle(SAMPLE_DIR)
    members = infer_members(
        bundle["github_log"],
        bundle["docs_revision"],
        bundle["slides_revision"],
        bundle["roles"],
        bundle["self_eval"],
        meeting_notes=bundle["meeting_notes"],
    )
    report = compute_fairness_report(members=members, project_type="development", **bundle)
    pd.DataFrame(members_to_rows(report.members)).to_csv(OUT_DIR / "fairteam_member_scores.csv", index=False)
    pd.DataFrame(report.audit_rows).to_csv(OUT_DIR / "fairteam_quality_audit.csv", index=False)
    (OUT_DIR / "professor_report.md").write_text(report.professor_report_md, encoding="utf-8")
    (OUT_DIR / "team_report.md").write_text(report.team_report_md, encoding="utf-8")


if __name__ == "__main__":
    main()
