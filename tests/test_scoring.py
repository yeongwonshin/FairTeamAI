from pathlib import Path

from fairteam_ai.loaders import infer_members, load_project_bundle
from fairteam_ai.scoring import compute_fairness_report


def test_sample_scoring_runs():
    root = Path(__file__).resolve().parents[1]
    bundle = load_project_bundle(root / "sample_data")
    members = infer_members(
        bundle["github_log"],
        bundle["docs_revision"],
        bundle["slides_revision"],
        bundle["roles"],
        bundle["self_eval"],
        meeting_notes=bundle["meeting_notes"],
    )
    report = compute_fairness_report(members=members, project_type="development", **bundle)
    assert len(report.members) == 4
    assert abs(sum(m.contribution_share for m in report.members) - 1.0) < 1e-6
    assert report.meeting_insights
    assert report.score_policy_md
    assert any("기여 로그" in ";".join(m.risk_tags) or "자기평가" in ";".join(m.risk_tags) for m in report.members)
