from fairteam_ai.analyzers import analyze_meeting_notes
from fairteam_ai.meeting_ai import extract_meeting_insights


MEMBERS = ["A", "B", "C", "D"]


def test_no_response_is_attributed_only_to_actual_target():
    notes = "참석: A, B, D. C는 연락이 늦고 답장이 없음."
    df = extract_meeting_insights(notes, MEMBERS, use_llm=False)
    negatives = df[(df["polarity"] == "negative") & (df["evidence_type"] == "no_response")]

    assert set(negatives["member"]) == {"C"}
    assert not set(negatives["member"]).intersection({"A", "B", "D"})


def test_deadline_assignment_is_not_missed_deadline():
    notes = "A: GitHub repository와 기본 Streamlit 구조를 만들기로 함. 담당자: A, deadline 05-03."
    df = extract_meeting_insights(notes, MEMBERS, use_llm=False)

    assert "assigned_action_item" in set(df["evidence_type"])
    assert "missed_deadline" not in set(df["evidence_type"])


def test_substitution_splits_original_owner_and_recovery_member():
    notes = "TODO: C가 맡은 남은 자료조사 요약은 B가 대체 작성함."
    df = extract_meeting_insights(notes, MEMBERS, use_llm=False)

    c_negative = df[(df["member"] == "C") & (df["evidence_type"] == "task_substitution") & (df["polarity"] == "negative")]
    b_positive = df[(df["member"] == "B") & (df["evidence_type"] == "task_recovery") & (df["polarity"] == "positive")]
    c_recovery = df[(df["member"] == "C") & (df["evidence_type"] == "task_recovery")]

    assert len(c_negative) == 1
    assert len(b_positive) == 1
    assert c_recovery.empty


def test_attendance_parser_does_not_count_absent_member_from_later_clause():
    notes = "참석: A, B, D. C는 연락이 늦고 답장이 없음."
    meeting_df, _, _ = analyze_meeting_notes(notes, MEMBERS)
    attendance = dict(zip(meeting_df["member"], meeting_df["attendance_count"]))

    assert attendance["A"] == 1
    assert attendance["B"] == 1
    assert attendance["D"] == 1
    assert attendance["C"] == 0


def test_negative_completion_words_are_not_positive_completed_work():
    notes = "TODO 담당자: C 자료조사 문서 업로드 미완료.\n갈등 위험: 역할 분담표 대비 C의 완료 기록 부족."
    df = extract_meeting_insights(notes, MEMBERS, use_llm=False)
    matched = df[df["source_sentence"].str.contains("미완료|완료 기록 부족", regex=True, na=False)]

    assert "completed_work" not in set(matched["evidence_type"])
