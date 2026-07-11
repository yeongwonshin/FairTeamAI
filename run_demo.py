from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fairteam_ai.ai_review import generate_ai_review_brief
from fairteam_ai.loaders import bundle_health_rows, infer_members, load_project_bundle
from fairteam_ai.readiness import calculate_review_readiness
from fairteam_ai.reporting import members_to_rows
from fairteam_ai.scoring import compute_fairness_report
from fairteam_ai.settings import env_flag, get_openai_settings
from fairteam_ai.workspace import (
    SnapshotStore,
    build_analysis_manifest,
    build_review_package,
    bundle_fingerprint,
)

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
    openai_settings = get_openai_settings()
    use_llm = env_flag("FAIRTEAM_USE_LLM", False) and openai_settings.configured
    fingerprint = bundle_fingerprint(
        bundle,
        {
            "project_type": "development",
            "use_llm": use_llm,
            "model": openai_settings.model,
        },
    )
    report = compute_fairness_report(
        members=members,
        project_type="development",
        use_llm_meeting_analysis=use_llm,
        openai_api_key=openai_settings.api_key,
        openai_model=openai_settings.model,
        **bundle,
    )
    scores = pd.DataFrame(members_to_rows(report.members))
    audit = pd.DataFrame(report.audit_rows)
    insights = pd.DataFrame(report.meeting_insights)
    readiness = calculate_review_readiness(
        report=report,
        bundle_health=bundle_health_rows(bundle),
        unresolved_appeals=0,
    )
    brief = generate_ai_review_brief(
        report=report,
        readiness_status=readiness.status,
        scores=scores,
        audit=audit,
        use_llm=use_llm,
        api_key=openai_settings.api_key,
        model=openai_settings.model,
    )
    manifest = build_analysis_manifest(
        workspace_name="FairTeam AI Demo",
        reviewer_name="CLI",
        fingerprint=fingerprint,
        project_type=report.project_type,
        weights=report.weights,
        members=members,
        source_label="sample_data",
        llm_enabled=use_llm,
        llm_model=openai_settings.model,
        llm_redacted=False,
        readiness=readiness.to_dict(),
    )
    package = build_review_package(
        scores=scores,
        audit=audit,
        meeting_insights=insights,
        professor_report_md=report.professor_report_md,
        team_report_md=report.team_report_md,
        scoring_policy_md=report.score_policy_md,
        ai_review_brief_md=brief.to_markdown(),
        manifest=manifest,
    )

    scores.to_csv(OUT_DIR / "fairteam_member_scores.csv", index=False)
    audit.to_csv(OUT_DIR / "fairteam_quality_audit.csv", index=False)
    insights.to_csv(OUT_DIR / "fairteam_meeting_insights.csv", index=False)
    (OUT_DIR / "professor_report.md").write_text(report.professor_report_md, encoding="utf-8")
    (OUT_DIR / "team_report.md").write_text(report.team_report_md, encoding="utf-8")
    (OUT_DIR / "ai_review_brief.md").write_text(brief.to_markdown(), encoding="utf-8")
    (OUT_DIR / "scoring_policy.md").write_text(report.score_policy_md, encoding="utf-8")
    (OUT_DIR / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "fairteam_review_package.zip").write_bytes(package)
    snapshot_path = SnapshotStore(OUT_DIR / "snapshots").save(manifest)

    print(report.summary)
    print(f"Review readiness: {readiness.status} ({readiness.score * 100:.0f}%)")
    print(f"AI brief engine: {brief.analysis_engine}")
    print(f"Saved review package: {OUT_DIR / 'fairteam_review_package.zip'}")
    print(f"Saved snapshot: {snapshot_path}")


if __name__ == "__main__":
    main()
