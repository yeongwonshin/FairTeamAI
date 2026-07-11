from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

from fairteam_ai.ai_review import generate_ai_review_brief
from fairteam_ai.loaders import bundle_health_rows, infer_members, load_project_bundle
from fairteam_ai.readiness import calculate_review_readiness
from fairteam_ai.reporting import members_to_rows
from fairteam_ai.scoring import compute_fairness_report
from fairteam_ai.settings import get_openai_settings
from fairteam_ai.workspace import build_analysis_manifest, build_review_package, bundle_fingerprint


def _sample_report():
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
    return bundle, members, report


def test_openai_settings_support_env_and_override(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_MODEL", "env-model")
    from_env = get_openai_settings()
    assert from_env.api_key == "env-key"
    assert from_env.model == "env-model"
    overridden = get_openai_settings(api_key_override="session-key", model_override="session-model")
    assert overridden.api_key == "session-key"
    assert overridden.model == "session-model"


def test_bundle_fingerprint_is_stable_and_sensitive():
    bundle = {
        "meeting_notes": "A completed the task.",
        "github_log": pd.DataFrame([{"member": "A", "commits": 1}]),
    }
    first = bundle_fingerprint(bundle, {"project_type": "development"})
    second = bundle_fingerprint(bundle, {"project_type": "development"})
    changed = bundle_fingerprint({**bundle, "meeting_notes": "A completed two tasks."}, {"project_type": "development"})
    assert first == second
    assert first != changed


def test_readiness_brief_and_review_package():
    bundle, members, report = _sample_report()
    scores = pd.DataFrame(members_to_rows(report.members))
    audit = pd.DataFrame(report.audit_rows)
    insights = pd.DataFrame(report.meeting_insights)
    readiness = calculate_review_readiness(
        report=report,
        bundle_health=bundle_health_rows(bundle),
        unresolved_appeals=1,
    )
    assert 0.0 <= readiness.score <= 1.0
    assert readiness.unresolved_appeals == 1

    brief = generate_ai_review_brief(
        report=report,
        readiness_status=readiness.status,
        scores=scores,
        audit=audit,
        use_llm=False,
    )
    assert brief.analysis_engine == "deterministic"
    assert "FairTeam AI Review Brief" in brief.to_markdown()

    fingerprint = bundle_fingerprint(bundle, {"project_type": report.project_type})
    manifest = build_analysis_manifest(
        workspace_name="Test workspace",
        reviewer_name="Reviewer",
        fingerprint=fingerprint,
        project_type=report.project_type,
        weights=report.weights,
        members=members,
        source_label="test",
        llm_enabled=False,
        llm_model="gpt-5.6",
        llm_redacted=True,
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
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        names = set(archive.namelist())
    assert "manifest.json" in names
    assert "reports/ai_review_brief.md" in names
    assert "data/member_scores.csv" in names
