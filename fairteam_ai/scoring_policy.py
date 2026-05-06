from __future__ import annotations

from typing import Dict

import numpy as np

CODE_SCORE_WEIGHTS: Dict[str, float] = {
    "commits": 4.0,
    "sqrt_changed_lines": 0.55,
    "files_changed": 1.8,
    "issues_closed": 4.5,
    "prs_merged": 5.0,
    "reviews": 2.5,
    "bugfix_commits": 4.0,
    "test_commits": 3.2,
}

DOCUMENT_SCORE_WEIGHTS: Dict[str, float] = {
    "edits": 2.0,
    "sqrt_words_added": 0.35,
    "comments_resolved": 3.0,
    "sections_owned": 7.0,
    "suggestions_accepted": 3.5,
    "references_added": 3.0,
}

SLIDE_SCORE_WEIGHTS: Dict[str, float] = {
    "slides_edited": 4.0,
    "visuals_created": 5.0,
    "sqrt_script_words": 0.25,
    "presenter_minutes": 3.0,
}

MEETING_SCORE_WEIGHTS: Dict[str, float] = {
    "attendance_count": 8.0,
    "speaking_turns": 2.0,
    "action_items_assigned": 4.0,
    "action_items_completed": 6.0,
    "decision_mentions": 3.0,
    "structured_positive_severity": 3.0,
    "structured_negative_penalty": -2.2,
}

ROLE_SCORE_WEIGHTS: Dict[str, float] = {
    "completion_rate": 12.0,
    "completed_tasks": 3.0,
    "critical_tasks": 3.5,
    "late_task_penalty": -0.6,
}


def safe_sqrt(x: float) -> float:
    return float(np.sqrt(max(float(x), 0.0)))


def score_code_signal(c: Dict[str, float]) -> float:
    return (
        CODE_SCORE_WEIGHTS["commits"] * c.get("commits", 0.0)
        + CODE_SCORE_WEIGHTS["sqrt_changed_lines"] * safe_sqrt(c.get("additions", 0.0) + c.get("deletions", 0.0))
        + CODE_SCORE_WEIGHTS["files_changed"] * c.get("files_changed", 0.0)
        + CODE_SCORE_WEIGHTS["issues_closed"] * c.get("issues_closed", 0.0)
        + CODE_SCORE_WEIGHTS["prs_merged"] * c.get("prs_merged", 0.0)
        + CODE_SCORE_WEIGHTS["reviews"] * c.get("reviews", 0.0)
        + CODE_SCORE_WEIGHTS["bugfix_commits"] * c.get("bugfix_commits", 0.0)
        + CODE_SCORE_WEIGHTS["test_commits"] * c.get("test_commits", 0.0)
    )


def score_document_signal(d: Dict[str, float]) -> float:
    return (
        DOCUMENT_SCORE_WEIGHTS["edits"] * d.get("edits", 0.0)
        + DOCUMENT_SCORE_WEIGHTS["sqrt_words_added"] * safe_sqrt(d.get("words_added", 0.0))
        + DOCUMENT_SCORE_WEIGHTS["comments_resolved"] * d.get("comments_resolved", 0.0)
        + DOCUMENT_SCORE_WEIGHTS["sections_owned"] * d.get("sections_owned", 0.0)
        + DOCUMENT_SCORE_WEIGHTS["suggestions_accepted"] * d.get("suggestions_accepted", 0.0)
        + DOCUMENT_SCORE_WEIGHTS["references_added"] * d.get("references_added", 0.0)
    )


def score_slide_signal(s: Dict[str, float]) -> float:
    return (
        SLIDE_SCORE_WEIGHTS["slides_edited"] * s.get("slides_edited", 0.0)
        + SLIDE_SCORE_WEIGHTS["visuals_created"] * s.get("visuals_created", 0.0)
        + SLIDE_SCORE_WEIGHTS["sqrt_script_words"] * safe_sqrt(s.get("script_words", 0.0))
        + SLIDE_SCORE_WEIGHTS["presenter_minutes"] * s.get("presenter_minutes", 0.0)
    )


def score_meeting_signal(mt: Dict[str, float], insight: Dict[str, float] | None = None) -> float:
    insight = insight or {}
    score = (
        MEETING_SCORE_WEIGHTS["attendance_count"] * mt.get("attendance_count", 0.0)
        + MEETING_SCORE_WEIGHTS["speaking_turns"] * mt.get("speaking_turns", 0.0)
        + MEETING_SCORE_WEIGHTS["action_items_assigned"] * mt.get("action_items_assigned", 0.0)
        + MEETING_SCORE_WEIGHTS["action_items_completed"] * mt.get("action_items_completed", 0.0)
        + MEETING_SCORE_WEIGHTS["decision_mentions"] * mt.get("decision_mentions", 0.0)
        + MEETING_SCORE_WEIGHTS["structured_positive_severity"] * insight.get("positive_severity", 0.0)
        + MEETING_SCORE_WEIGHTS["structured_negative_penalty"] * insight.get("negative_severity", 0.0)
    )
    return max(0.0, score)


def score_role_signal(r: Dict[str, float]) -> float:
    completion_rate = r.get("completed_tasks", 0.0) / max(r.get("assigned_tasks", 0.0), 1.0)
    return max(
        0.0,
        ROLE_SCORE_WEIGHTS["completion_rate"] * completion_rate
        + ROLE_SCORE_WEIGHTS["completed_tasks"] * r.get("completed_tasks", 0.0)
        + ROLE_SCORE_WEIGHTS["critical_tasks"] * r.get("critical_tasks", 0.0)
        + ROLE_SCORE_WEIGHTS["late_task_penalty"] * r.get("late_tasks", 0.0),
    )


def build_scoring_policy_markdown() -> str:
    return """## 점수 산식 근거

- 모든 원자료는 먼저 `코드/문서/발표/회의/역할` 범주별 원점수로 환산한 뒤, 팀 내 상대 비중으로 정규화합니다.
- 라인 수·단어 수·스크립트 단어 수는 대량 붙여넣기와 자동 생성물 과대평가를 막기 위해 `sqrt` 압축을 적용합니다.
- PR 병합, 이슈 종료, 리뷰, 테스트 커밋은 검증 가능한 협업 품질 신호이므로 단순 커밋 수보다 높은 의미를 부여합니다.
- 품질 감사에서 커밋 쪼개기, 대량 붙여넣기, 마감 직전 집중, 단일 출처 의존, 자기평가 과장 신호가 발견되면 원점수를 보존하되 품질 보정 기여도를 별도로 산출합니다.
- 최종 값은 성적 확정값이 아니라 교수자가 원자료를 우선 검토할 수 있도록 정렬한 공정평가 보조 지표입니다.
"""
