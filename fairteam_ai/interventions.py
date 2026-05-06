from __future__ import annotations

from typing import List

from .models import TeamMemberEvidence


def build_intervention_plan(members: List[TeamMemberEvidence], conflict_risk: float, imbalance_ratio: float) -> List[str]:
    """Generate action-oriented recommendations before the project collapses."""
    actions: List[str] = []
    if conflict_risk >= 0.45:
        actions.append("24시간 내 역할 재조정 회의: 갈등 문장에 등장한 업무를 기준으로 담당자·마감일·검수자를 재확정하십시오.")
    elif conflict_risk >= 0.25:
        actions.append("다음 회의 시작 10분을 리스크 점검에 배정하고, 지연 업무의 차단 요인을 회의록에 명시하십시오.")

    if imbalance_ratio >= 3.0:
        actions.append("기여도 최대/최소 격차가 큽니다. 고기여자에게 몰린 작업을 1개 이상 분리해 저기여자에게 검수 가능한 단위로 재배정하십시오.")
    elif imbalance_ratio >= 2.0:
        actions.append("역할 불균형이 관찰됩니다. 남은 작업을 코드·문서·발표·검수 단위로 쪼개 재분배하십시오.")

    for m in sorted(members, key=lambda x: x.contribution_share):
        tag_text = " ".join(m.risk_tags + m.audit_flags)
        if "기여 로그" in tag_text or "단일 출처" in tag_text:
            actions.append(f"{m.name}: 오프라인 기여가 있다면 원본 파일·회의 발언·검수 기록을 추가 제출하도록 요청하십시오.")
        if "자기평가" in tag_text:
            actions.append(f"{m.name}: 자기평가 주장과 로그 차이가 큽니다. 주장한 핵심 작업의 산출물 링크/파일명을 요구하십시오.")
        if "업무 과중" in tag_text:
            actions.append(f"{m.name}: 업무 과중 가능성이 있으므로 감점보다 팀 운영 실패 여부를 함께 검토하십시오.")
        if "커밋 쪼개기" in tag_text or "대량 붙여넣기" in tag_text:
            actions.append(f"{m.name}: 코드 변경량보다 PR 리뷰, 이슈 해결, 테스트 추가 여부를 우선 검토하십시오.")

    if not actions:
        actions.append("현재 자동 탐지 기준에서는 큰 위험 신호가 낮습니다. 다만 최종 평가는 원자료와 팀원 이의제기 내용을 함께 검토하십시오.")
    # De-duplicate while preserving order.
    seen = set()
    unique: List[str] = []
    for action in actions:
        if action not in seen:
            seen.add(action)
            unique.append(action)
    return unique[:10]
