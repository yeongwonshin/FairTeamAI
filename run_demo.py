from __future__ import annotations

from pathlib import Path

import pandas as pd

from fairteam_ai.loaders import infer_members, load_project_bundle
from fairteam_ai.reporting import members_to_rows
from fairteam_ai.scoring import compute_fairness_report

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "sample_data"
OUT_DIR = ROOT / "outputs"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = load_project_bundle(SAMPLE_DIR)
    members = infer_members(
        bundle["github_log"],
        bundle["docs_revision"],
        bundle["slides_revision"],
        bundle["roles"],
        bundle["self_eval"],
        meeting_notes=bundle["meeting_notes"],
    )
    report = compute_fairness_report(members=members, project_type="development", use_llm_meeting_analysis=False, **bundle)
    scores = pd.DataFrame(members_to_rows(report.members))
    audit = pd.DataFrame(report.audit_rows)
    insights = pd.DataFrame(report.meeting_insights)

    scores.to_csv(OUT_DIR / "fairteam_member_scores.csv", index=False)
    audit.to_csv(OUT_DIR / "fairteam_quality_audit.csv", index=False)
    insights.to_csv(OUT_DIR / "fairteam_meeting_insights.csv", index=False)
    (OUT_DIR / "professor_report.md").write_text(report.professor_report_md, encoding="utf-8")
    (OUT_DIR / "team_report.md").write_text(report.team_report_md, encoding="utf-8")
    (OUT_DIR / "scoring_policy.md").write_text(report.score_policy_md, encoding="utf-8")

    print(report.summary)
    print("\nRecommended interventions:")
    for i, action in enumerate(report.intervention_plan, start=1):
        print(f"{i}. {action}")
    print(f"Saved: {OUT_DIR / 'fairteam_member_scores.csv'}")
    print(f"Saved: {OUT_DIR / 'fairteam_quality_audit.csv'}")
    print(f"Saved: {OUT_DIR / 'fairteam_meeting_insights.csv'}")
    print(f"Saved: {OUT_DIR / 'professor_report.md'}")
    print(f"Saved: {OUT_DIR / 'scoring_policy.md'}")


if __name__ == "__main__":
    main()
