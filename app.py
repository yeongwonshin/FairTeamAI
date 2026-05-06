from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from fairteam_ai.appeals import AppealStore
from fairteam_ai.config import DEFAULT_WEIGHTS
from fairteam_ai.loaders import (
    REQUIRED_DOC_COLUMNS,
    REQUIRED_GITHUB_COLUMNS,
    REQUIRED_ROLE_COLUMNS,
    REQUIRED_SELF_EVAL_COLUMNS,
    REQUIRED_SLIDE_COLUMNS,
    infer_members,
    load_project_bundle,
    read_text,
    safe_read_csv,
)
from fairteam_ai.privacy import redact_dataframe, redact_text
from fairteam_ai.reporting import members_to_rows
from fairteam_ai.scoring import compute_fairness_report

st.set_page_config(page_title="FairTeam AI", page_icon="⚖️", layout="wide")

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "sample_data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

st.title("⚖️ FairTeam AI")
st.caption("팀 프로젝트 기여도 추적 · 조작 방어 · 갈등 예방 · 교수자용 공정평가 리포트")

with st.sidebar:
    st.header("입력 설정")
    use_sample = st.toggle("샘플 데이터로 데모 실행", value=True)
    redact_export = st.toggle("개인정보 자동 마스킹", value=True)
    project_type = st.selectbox(
        "프로젝트 유형",
        options=list(DEFAULT_WEIGHTS.keys()),
        index=list(DEFAULT_WEIGHTS.keys()).index("development"),
    )
    st.markdown("### 가중치")
    base_weights = DEFAULT_WEIGHTS[project_type]
    code_w = st.slider("코드", 0.0, 1.0, float(base_weights["code"]), 0.01)
    doc_w = st.slider("보고서/문서", 0.0, 1.0, float(base_weights["document"]), 0.01)
    slide_w = st.slider("발표자료", 0.0, 1.0, float(base_weights["slide"]), 0.01)
    meeting_w = st.slider("회의/관리", 0.0, 1.0, float(base_weights["meeting"]), 0.01)
    role_w = st.slider("역할 이행", 0.0, 1.0, float(base_weights["role"]), 0.01)
    custom_weights = {
        "code": code_w,
        "document": doc_w,
        "slide": slide_w,
        "meeting": meeting_w,
        "role": role_w,
    }

if use_sample:
    bundle = load_project_bundle(SAMPLE_DIR)
else:
    st.subheader("원자료 업로드")
    c1, c2, c3 = st.columns(3)
    with c1:
        meeting_file = st.file_uploader("회의록 TXT/MD", type=["txt", "md"])
        github_file = st.file_uploader("GitHub 로그 CSV", type=["csv"])
    with c2:
        docs_file = st.file_uploader("문서 수정 기록 CSV", type=["csv"])
        slides_file = st.file_uploader("발표자료 수정 기록 CSV", type=["csv"])
    with c3:
        roles_file = st.file_uploader("역할분담/업무완료 CSV", type=["csv"])
        self_eval_file = st.file_uploader("자기평가 CSV", type=["csv"])

    if not all([meeting_file, github_file, docs_file, slides_file, roles_file, self_eval_file]):
        st.info("모든 파일을 업로드하거나, 사이드바에서 샘플 데이터 데모를 켜세요.")
        st.stop()

    bundle = {
        "meeting_notes": read_text(meeting_file),
        "github_log": safe_read_csv(github_file, REQUIRED_GITHUB_COLUMNS),
        "docs_revision": safe_read_csv(docs_file, REQUIRED_DOC_COLUMNS),
        "slides_revision": safe_read_csv(slides_file, REQUIRED_SLIDE_COLUMNS),
        "roles": safe_read_csv(roles_file, REQUIRED_ROLE_COLUMNS),
        "self_eval": safe_read_csv(self_eval_file, REQUIRED_SELF_EVAL_COLUMNS),
    }

members = infer_members(
    bundle["github_log"],
    bundle["docs_revision"],
    bundle["slides_revision"],
    bundle["roles"],
    bundle["self_eval"],
    meeting_notes=str(bundle["meeting_notes"]),
)

report = compute_fairness_report(
    members=members,
    project_type=project_type,
    custom_weights=custom_weights,
    **bundle,
)

rows = pd.DataFrame(members_to_rows(report.members))
rows_display = rows.copy()
for col in [
    "contribution_share",
    "raw_contribution_share",
    "quality_adjusted_share",
    "confidence_score",
    "quality_score",
    "anti_gaming_score",
    "code_share",
    "document_share",
    "slide_share",
    "meeting_share",
    "role_share",
    "self_claim_share",
    "overclaim_gap",
]:
    if col in rows_display.columns:
        rows_display[col] = rows_display[col].apply(lambda x: "" if pd.isna(x) else f"{x*100:.1f}%")

professor_report_md = report.professor_report_md
team_report_md = report.team_report_md
if redact_export:
    professor_report_md = redact_text(professor_report_md)
    team_report_md = redact_text(team_report_md)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["대시보드", "근거 로그", "품질/조작 감사", "갈등/위험", "개입/이의제기", "교수자 리포트", "원자료"]
)

with tab1:
    st.subheader("핵심 요약")
    st.write(report.summary)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("기여 불균형 Gini", f"{report.gini:.3f}")
    m2.metric("최대/최소 기여 비율", f"{report.imbalance_ratio:.2f}x")
    m3.metric("갈등 위험", f"{report.conflict_risk_score * 100:.1f}%")
    m4.metric("평균 산출 신뢰도", f"{rows['confidence_score'].mean() * 100:.1f}%")

    chart_df = rows.sort_values("contribution_share", ascending=False)
    fig = px.bar(
        chart_df,
        x="member",
        y=["raw_contribution_share", "quality_adjusted_share"],
        barmode="group",
        title="원점수 vs 품질 보정 기여도",
    )
    fig.update_layout(yaxis_tickformat=".0%", yaxis_title="기여도", xaxis_title="팀원")
    st.plotly_chart(fig, use_container_width=True)

    stack = rows.melt(
        id_vars=["member"],
        value_vars=["code_share", "document_share", "slide_share", "meeting_share", "role_share"],
        var_name="category",
        value_name="category_share",
    )
    fig2 = px.bar(stack, x="member", y="category_share", color="category", title="카테고리별 상대 기여 신호")
    fig2.update_layout(yaxis_tickformat=".0%", yaxis_title="카테고리 내 상대 비중", xaxis_title="팀원")
    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(rows_display, use_container_width=True)

with tab2:
    st.subheader("팀원별 근거")
    for member in sorted(report.members, key=lambda x: x.contribution_share, reverse=True):
        with st.expander(f"{member.name} · {member.contribution_share * 100:.1f}%"):
            for item in member.evidence:
                st.write("- " + item)
            if member.audit_flags:
                st.warning("조작/품질 검토 신호: " + " / ".join(member.audit_flags))
            if member.risk_tags:
                st.error("위험 태그: " + " / ".join(member.risk_tags))

with tab3:
    st.subheader("품질/조작 감사")
    audit_df = pd.DataFrame(report.audit_rows)
    st.dataframe(audit_df, use_container_width=True)

    qfig = px.scatter(
        rows,
        x="confidence_score",
        y="quality_score",
        size="contribution_share",
        hover_name="member",
        title="산출 신뢰도 × 근거 품질 매트릭스",
    )
    qfig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    st.plotly_chart(qfig, use_container_width=True)

    st.info(
        "이 탭은 구성원을 처벌하기 위한 자동 판정이 아니라, 커밋 쪼개기·대량 붙여넣기·마감 직전 집중·단일 출처 의존처럼 교수자가 원자료를 다시 볼 지점을 표시합니다."
    )

with tab4:
    st.subheader("갈등 위험 및 무임승차 위험")
    risk_rows = rows[rows["risk_tags"].astype(str).str.len() > 0]
    if risk_rows.empty:
        st.success("위험 태그가 붙은 팀원이 없습니다.")
    else:
        st.dataframe(
            risk_rows[["member", "contribution_share", "self_claim_share", "overclaim_gap", "risk_tags"]],
            use_container_width=True,
        )
    st.markdown("### 회의록에서 감지된 갈등 문장")
    if report.conflict_evidence:
        for line in report.conflict_evidence:
            st.write("- " + (redact_text(line) if redact_export else line))
    else:
        st.write("명시적 갈등 문장이 크게 감지되지 않았습니다.")

with tab5:
    st.subheader("개입 권장안")
    for i, action in enumerate(report.intervention_plan, start=1):
        st.write(f"{i}. {action}")

    st.markdown("---")
    st.subheader("팀원 이의제기 접수")
    appeal_store = AppealStore(OUT_DIR / "appeals.jsonl")
    with st.form("appeal_form"):
        c1, c2 = st.columns(2)
        with c1:
            appeal_member = st.selectbox("팀원", options=members)
            appeal_category = st.selectbox("분류", options=["코드", "문서", "발표", "회의", "역할", "기타"])
        with c2:
            appeal_ref = st.text_input("증거 링크/파일명", placeholder="예: PR #12, 보고서 2장 초안, 회의록 2026-05-01")
        appeal_claim = st.text_area("이의제기 내용", placeholder="누락된 오프라인 기여나 잘못 해석된 로그를 구체적으로 적으세요.")
        submitted = st.form_submit_button("이의제기 제출")
        if submitted:
            created = appeal_store.submit(appeal_member, appeal_category, appeal_claim, appeal_ref)
            st.success(f"이의제기가 접수되었습니다: {created.appeal_id}")

    appeals = appeal_store.list()
    if appeals:
        st.markdown("### 접수된 이의제기")
        st.dataframe(pd.DataFrame([a.__dict__ for a in appeals]), use_container_width=True)
    else:
        st.caption("아직 접수된 이의제기가 없습니다.")

with tab6:
    st.subheader("교수자용 공정평가 리포트")
    st.markdown(professor_report_md)
    st.download_button(
        "교수자 리포트 Markdown 다운로드",
        data=professor_report_md.encode("utf-8"),
        file_name="fairteam_professor_report.md",
        mime="text/markdown",
    )
    st.download_button(
        "팀원별 점수 CSV 다운로드",
        data=rows.to_csv(index=False).encode("utf-8"),
        file_name="fairteam_member_scores.csv",
        mime="text/csv",
    )
    st.download_button(
        "조작/품질 감사 CSV 다운로드",
        data=pd.DataFrame(report.audit_rows).to_csv(index=False).encode("utf-8"),
        file_name="fairteam_quality_audit.csv",
        mime="text/csv",
    )

with tab7:
    st.subheader("입력 원자료 미리보기")
    st.markdown("#### 회의록")
    preview_notes = redact_text(bundle["meeting_notes"]) if redact_export else str(bundle["meeting_notes"])
    st.text_area("meeting_notes", preview_notes, height=220)
    for key in ["github_log", "docs_revision", "slides_revision", "roles", "self_eval"]:
        st.markdown(f"#### {key}")
        frame = redact_dataframe(bundle[key]) if redact_export else bundle[key]
        st.dataframe(frame, use_container_width=True)
