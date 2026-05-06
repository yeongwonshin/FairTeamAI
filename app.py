from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

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
from fairteam_ai.reporting import members_to_rows
from fairteam_ai.scoring import compute_fairness_report

st.set_page_config(page_title="FairTeam AI", page_icon="⚖️", layout="wide")

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "sample_data"

st.title("⚖️ FairTeam AI")
st.caption("팀 프로젝트 기여도 추적 · 갈등 예방 · 교수자용 공정평가 리포트")

with st.sidebar:
    st.header("입력 설정")
    use_sample = st.toggle("샘플 데이터로 데모 실행", value=True)
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
for col in ["contribution_share", "code_share", "document_share", "slide_share", "meeting_share", "role_share", "self_claim_share", "overclaim_gap"]:
    if col in rows_display.columns:
        rows_display[col] = rows_display[col].apply(lambda x: "" if pd.isna(x) else f"{x*100:.1f}%")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["대시보드", "근거 로그", "갈등/위험", "교수자 리포트", "원자료"])

with tab1:
    st.subheader("핵심 요약")
    st.write(report.summary)
    m1, m2, m3 = st.columns(3)
    m1.metric("기여 불균형 Gini", f"{report.gini:.3f}")
    m2.metric("최대/최소 기여 비율", f"{report.imbalance_ratio:.2f}x")
    m3.metric("갈등 위험", f"{report.conflict_risk_score * 100:.1f}%")

    chart_df = rows.sort_values("contribution_share", ascending=False)
    fig = px.bar(chart_df, x="member", y="contribution_share", text="contribution_share", title="팀원별 총 기여도 추정치")
    fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
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
            if member.risk_tags:
                st.warning(" / ".join(member.risk_tags))

with tab3:
    st.subheader("갈등 위험 및 무임승차 위험")
    risk_rows = rows[rows["risk_tags"].astype(str).str.len() > 0]
    if risk_rows.empty:
        st.success("위험 태그가 붙은 팀원이 없습니다.")
    else:
        st.dataframe(risk_rows[["member", "contribution_share", "self_claim_share", "overclaim_gap", "risk_tags"]], use_container_width=True)
    st.markdown("### 회의록에서 감지된 갈등 문장")
    if report.conflict_evidence:
        for line in report.conflict_evidence:
            st.write("- " + line)
    else:
        st.write("명시적 갈등 문장이 크게 감지되지 않았습니다.")

with tab4:
    st.subheader("교수자용 공정평가 리포트")
    st.markdown(report.professor_report_md)
    st.download_button(
        "교수자 리포트 Markdown 다운로드",
        data=report.professor_report_md.encode("utf-8"),
        file_name="fairteam_professor_report.md",
        mime="text/markdown",
    )
    st.download_button(
        "팀원별 점수 CSV 다운로드",
        data=rows.to_csv(index=False).encode("utf-8"),
        file_name="fairteam_member_scores.csv",
        mime="text/csv",
    )

with tab5:
    st.subheader("입력 원자료 미리보기")
    st.markdown("#### 회의록")
    st.text_area("meeting_notes", str(bundle["meeting_notes"]), height=220)
    for key in ["github_log", "docs_revision", "slides_revision", "roles", "self_eval"]:
        st.markdown(f"#### {key}")
        st.dataframe(bundle[key], use_container_width=True)
