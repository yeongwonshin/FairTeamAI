from __future__ import annotations

"""Generate multi-team validation scenarios for FairTeam AI.

This script creates 20 synthetic but auditable team-project cases and runs the
same scoring engine used by the dashboard. It is meant for contest demos: instead
of showing one handcrafted example, you can show that the system handles normal
collaboration, free riding, commit padding, document-heavy contribution,
offline-evidence claims, and task substitution cases.

Run:
    python scripts/generate_validation_scenarios.py

Output:
    outputs/fairteam_scenario_validation.csv
"""

from pathlib import Path
import sys
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fairteam_ai.scoring import compute_fairness_report

OUT_DIR = ROOT / "outputs"
MEMBERS = ["A", "B", "C", "D"]


def _base_frames() -> Dict[str, pd.DataFrame]:
    github = pd.DataFrame(
        [
            {"member": "A", "commits": 12, "additions": 900, "deletions": 220, "files_changed": 18, "issues_closed": 3, "prs_merged": 4, "reviews": 2, "bugfix_commits": 2, "test_commits": 2},
            {"member": "B", "commits": 9, "additions": 520, "deletions": 120, "files_changed": 12, "issues_closed": 2, "prs_merged": 2, "reviews": 4, "bugfix_commits": 1, "test_commits": 1},
            {"member": "C", "commits": 8, "additions": 480, "deletions": 90, "files_changed": 10, "issues_closed": 2, "prs_merged": 2, "reviews": 2, "bugfix_commits": 1, "test_commits": 1},
            {"member": "D", "commits": 10, "additions": 620, "deletions": 160, "files_changed": 14, "issues_closed": 2, "prs_merged": 2, "reviews": 3, "bugfix_commits": 2, "test_commits": 2},
        ]
    )
    docs = pd.DataFrame(
        [
            {"member": "A", "edits": 12, "words_added": 900, "comments_resolved": 3, "sections_owned": 2, "suggestions_accepted": 2, "references_added": 2},
            {"member": "B", "edits": 15, "words_added": 1200, "comments_resolved": 6, "sections_owned": 3, "suggestions_accepted": 4, "references_added": 5},
            {"member": "C", "edits": 10, "words_added": 750, "comments_resolved": 3, "sections_owned": 2, "suggestions_accepted": 2, "references_added": 3},
            {"member": "D", "edits": 8, "words_added": 620, "comments_resolved": 2, "sections_owned": 1, "suggestions_accepted": 2, "references_added": 1},
        ]
    )
    slides = pd.DataFrame(
        [
            {"member": "A", "slides_edited": 4, "visuals_created": 2, "script_words": 220, "presenter_minutes": 3},
            {"member": "B", "slides_edited": 5, "visuals_created": 2, "script_words": 300, "presenter_minutes": 4},
            {"member": "C", "slides_edited": 4, "visuals_created": 2, "script_words": 250, "presenter_minutes": 3},
            {"member": "D", "slides_edited": 6, "visuals_created": 4, "script_words": 350, "presenter_minutes": 5},
        ]
    )
    roles = pd.DataFrame(
        [
            {"member": "A", "assigned_tasks": 5, "completed_tasks": 5, "late_tasks": 0, "critical_tasks": 2},
            {"member": "B", "assigned_tasks": 5, "completed_tasks": 5, "late_tasks": 0, "critical_tasks": 2},
            {"member": "C", "assigned_tasks": 5, "completed_tasks": 5, "late_tasks": 0, "critical_tasks": 1},
            {"member": "D", "assigned_tasks": 5, "completed_tasks": 5, "late_tasks": 0, "critical_tasks": 2},
        ]
    )
    self_eval = pd.DataFrame(
        [
            {"member": "A", "self_claim_percent": 26, "claimed_main_work": "backend", "peer_comment": "정상"},
            {"member": "B", "self_claim_percent": 26, "claimed_main_work": "report", "peer_comment": "정상"},
            {"member": "C", "self_claim_percent": 24, "claimed_main_work": "research", "peer_comment": "정상"},
            {"member": "D", "self_claim_percent": 24, "claimed_main_work": "demo", "peer_comment": "정상"},
        ]
    )
    return {"github_log": github, "docs_revision": docs, "slides_revision": slides, "roles": roles, "self_eval": self_eval}


def _notes(case_type: str, weak_member: str) -> str:
    if case_type == "normal":
        return "\n".join([
            "참석: A, B, C, D",
            "A: API 연동 완료.",
            "B: 보고서 초안 완료.",
            "C: 자료조사 링크 공유 및 문서 반영 완료.",
            "D: 테스트와 발표자료 완료.",
        ])
    if case_type == "free_rider":
        return f"참석: A, B, D. {weak_member}는 연락이 늦고 답장이 없음.\nTODO 담당자: {weak_member} 자료조사 문서 업로드 미완료.\n갈등 위험: 역할 분담표 대비 {weak_member}의 완료 기록 부족."
    if case_type == "substitution":
        helper = "B" if weak_member != "B" else "A"
        return f"참석: A, B, D. {weak_member}는 지각.\nTODO: {weak_member}가 맡은 남은 자료조사 요약은 {helper}가 대체 작성함."
    if case_type == "overclaim":
        return f"참석: A, B, C, D\n{weak_member}: 초반 자료조사 링크 1개를 공유했으나 보고서 반영량은 적음. 본인은 오프라인 기여가 많다고 주장."
    if case_type == "overload":
        return "참석: A, B, C, D\n갈등: A가 마감 직전 혼자 코드 부담이 크다고 언급.\nA: scoring.py 버그 수정 완료."
    return "참석: A, B, C, D"


def _mutate(frames: Dict[str, pd.DataFrame], case_type: str, weak_member: str, idx: int) -> Dict[str, pd.DataFrame]:
    out = {k: v.copy() for k, v in frames.items()}
    if case_type in {"free_rider", "substitution"}:
        out["github_log"].loc[out["github_log"]["member"] == weak_member, ["commits", "additions", "deletions", "files_changed", "issues_closed", "prs_merged", "reviews", "bugfix_commits", "test_commits"]] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        out["roles"].loc[out["roles"]["member"] == weak_member, ["completed_tasks", "late_tasks"]] = [1, 3]
        out["self_eval"].loc[out["self_eval"]["member"] == weak_member, "self_claim_percent"] = 35
    elif case_type == "commit_padding":
        out["github_log"].loc[out["github_log"]["member"] == weak_member, ["commits", "additions", "deletions", "files_changed"]] = [35, 80, 10, 2]
        out["github_log"].loc[out["github_log"]["member"] == weak_member, ["commit_messages", "unique_message_ratio", "dominant_file_ratio"]] = ["update | update | update", 0.08, 0.95]
    elif case_type == "doc_heavy":
        out["github_log"].loc[out["github_log"]["member"] == weak_member, ["commits", "additions", "files_changed"]] = [1, 40, 1]
        out["docs_revision"].loc[out["docs_revision"]["member"] == weak_member, ["edits", "words_added", "comments_resolved", "sections_owned", "references_added"]] = [30, 2500, 10, 5, 12]
    elif case_type == "overclaim":
        out["self_eval"].loc[out["self_eval"]["member"] == weak_member, "self_claim_percent"] = 45
        out["github_log"].loc[out["github_log"]["member"] == weak_member, ["commits", "additions", "files_changed"]] = [1, 35, 1]
        out["docs_revision"].loc[out["docs_revision"]["member"] == weak_member, ["edits", "words_added"]] = [2, 120]
    # deterministic variation so the 20 rows are not identical
    out["github_log"]["additions"] = out["github_log"]["additions"] + idx * 3
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    case_types = ["normal", "free_rider", "substitution", "commit_padding", "doc_heavy", "overclaim", "overload"]
    rows: List[dict] = []
    base = _base_frames()

    for idx in range(20):
        case_type = case_types[idx % len(case_types)]
        weak_member = MEMBERS[idx % len(MEMBERS)]
        frames = _mutate(base, case_type, weak_member, idx)
        report = compute_fairness_report(
            members=MEMBERS,
            meeting_notes=_notes(case_type, weak_member),
            project_type="development",
            use_llm_meeting_analysis=False,
            **frames,
        )
        member_rows = {m.name: m for m in report.members}
        detected = member_rows[weak_member]
        risk_text = "; ".join(detected.risk_tags + detected.audit_flags)
        should_flag = case_type in {"free_rider", "substitution", "commit_padding", "overclaim"}
        passed = (not should_flag) or bool(risk_text.strip()) or detected.contribution_share < 0.15
        rows.append(
            {
                "scenario_id": f"team_{idx + 1:02d}",
                "case_type": case_type,
                "expected_focus_member": weak_member,
                "expected_flag": should_flag,
                "focus_member_share": round(detected.contribution_share, 4),
                "focus_member_risk_or_audit": risk_text,
                "top_contributor": max(report.members, key=lambda m: m.contribution_share).name,
                "conflict_risk": round(report.conflict_risk_score, 4),
                "passed_expected_check": passed,
            }
        )

    out = pd.DataFrame(rows)
    path = OUT_DIR / "fairteam_scenario_validation.csv"
    out.to_csv(path, index=False)
    print(f"Saved: {path}")
    print(out[["scenario_id", "case_type", "expected_focus_member", "passed_expected_check"]].to_string(index=False))


if __name__ == "__main__":
    main()
