from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def _stable_value(value: object) -> bytes:
    if isinstance(value, pd.DataFrame):
        frame = value.copy()
        frame = frame.reindex(sorted(frame.columns), axis=1)
        if len(frame.columns):
            frame = frame.sort_values(list(frame.columns), kind="stable", na_position="last")
        return frame.to_csv(index=False).encode("utf-8")
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return str(value or "").encode("utf-8")


def bundle_fingerprint(bundle: Mapping[str, object], options: Mapping[str, object] | None = None) -> str:
    digest = hashlib.sha256()
    for key in sorted(bundle):
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_stable_value(bundle[key]))
        digest.update(b"\0")
    if options:
        digest.update(_stable_value(dict(options)))
    return digest.hexdigest()[:16]


def build_analysis_manifest(
    *,
    workspace_name: str,
    reviewer_name: str,
    fingerprint: str,
    project_type: str,
    weights: Mapping[str, float],
    members: list[str],
    source_label: str,
    llm_enabled: bool,
    llm_model: str,
    llm_redacted: bool,
    readiness: Mapping[str, object],
) -> dict[str, Any]:
    return {
        "schema_version": "2.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workspace_name": workspace_name.strip() or "Untitled workspace",
        "reviewer_name": reviewer_name.strip(),
        "analysis_fingerprint": fingerprint,
        "project_type": project_type,
        "weights": {k: round(float(v), 6) for k, v in weights.items()},
        "members": members,
        "source_label": source_label,
        "llm": {
            "enabled": bool(llm_enabled),
            "model": llm_model,
            "input_redacted": bool(llm_redacted),
            "api_key_persisted": False,
        },
        "review_readiness": dict(readiness),
        "decision_notice": "This package supports human review and must not be used as an automatic final grade.",
    }


class SnapshotStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, manifest: Mapping[str, Any]) -> Path:
        created = str(manifest.get("created_at", "")).replace(":", "-").replace("+", "_")
        created = created[:19] or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        fingerprint = str(manifest.get("analysis_fingerprint", "snapshot"))
        path = self.directory / f"{created}_{fingerprint}.json"
        path.write_text(json.dumps(dict(manifest), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return path


def _to_jsonable(value: object) -> object:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def build_review_package(
    *,
    scores: pd.DataFrame,
    audit: pd.DataFrame,
    meeting_insights: pd.DataFrame,
    professor_report_md: str,
    team_report_md: str,
    scoring_policy_md: str,
    ai_review_brief_md: str,
    manifest: Mapping[str, Any],
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("reports/professor_report.md", professor_report_md)
        zf.writestr("reports/team_report.md", team_report_md)
        zf.writestr("reports/ai_review_brief.md", ai_review_brief_md)
        zf.writestr("reports/scoring_policy.md", scoring_policy_md)
        zf.writestr("data/member_scores.csv", scores.to_csv(index=False))
        zf.writestr("data/quality_audit.csv", audit.to_csv(index=False))
        zf.writestr("data/meeting_insights.csv", meeting_insights.to_csv(index=False))
        zf.writestr("manifest.json", json.dumps(_to_jsonable(dict(manifest)), ensure_ascii=False, indent=2, default=str))
    return buffer.getvalue()
