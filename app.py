from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

from fairteam_ai.appeals import AppealStore
from fairteam_ai.batch import analyze_many_teams
from fairteam_ai.config import DEFAULT_WEIGHTS
from fairteam_ai.github_ingest import GitHubIngestConfig, fetch_github_log
from fairteam_ai.loaders import (
    BUNDLE_FILE_SPECS,
    REQUIRED_DOC_COLUMNS,
    REQUIRED_GITHUB_COLUMNS,
    REQUIRED_ROLE_COLUMNS,
    REQUIRED_SELF_EVAL_COLUMNS,
    REQUIRED_SLIDE_COLUMNS,
    bundle_health_rows,
    infer_members,
    load_project_bundle,
    read_text,
    safe_read_csv,
    save_bundle_to_directory,
    template_dataframe,
)
from fairteam_ai.privacy import redact_dataframe, redact_text
from fairteam_ai.reporting import members_to_rows
from fairteam_ai.scoring import compute_fairness_report

st.set_page_config(page_title="FairTeam AI", page_icon="⚖️", layout="wide")

ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "sample_data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
          --ft-bg: #f6f8fb;
          --ft-card: #ffffff;
          --ft-ink: #102033;
          --ft-muted: #667085;
          --ft-line: rgba(16, 32, 51, .10);
          --ft-accent: #5967ff;
          --ft-accent-2: #14b8a6;
          --ft-danger: #ef4444;
          --ft-warn: #f59e0b;
        }
        .stApp { background: linear-gradient(180deg, #f7f9ff 0%, #f6f8fb 40%, #ffffff 100%); }
        div[data-testid="stSidebar"] { background: #0f172a; }
        div[data-testid="stSidebar"] * { color: #e5e7eb !important; }
        div[data-testid="stSidebar"] input, div[data-testid="stSidebar"] textarea { color: #111827 !important; }
        div[data-testid="stSidebar"] .stSelectbox div, div[data-testid="stSidebar"] .stNumberInput div { color: #111827 !important; }
        .hero {
            padding: 28px 32px;
            border-radius: 28px;
            background: radial-gradient(circle at top left, rgba(89,103,255,.25), transparent 28%),
                        linear-gradient(135deg, #111827 0%, #26345f 52%, #4f46e5 100%);
            color: white;
            box-shadow: 0 24px 70px rgba(17, 24, 39, .22);
            margin-bottom: 20px;
        }
        .hero h1 { margin: 0; font-size: 2.35rem; letter-spacing: -.04em; }
        .hero p { margin: 10px 0 0 0; max-width: 980px; color: rgba(255,255,255,.82); font-size: 1.02rem; }
        .pill-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }
        .pill {
            border: 1px solid rgba(255,255,255,.22);
            background: rgba(255,255,255,.10);
            color: white;
            padding: 7px 11px;
            border-radius: 999px;
            font-size: .86rem;
        }
        .soft-card {
            background: var(--ft-card);
            border: 1px solid var(--ft-line);
            border-radius: 22px;
            padding: 18px 20px;
            box-shadow: 0 18px 45px rgba(16, 32, 51, .07);
            margin-bottom: 14px;
        }
        .status-card {
            background: white;
            border: 1px solid var(--ft-line);
            border-radius: 18px;
            padding: 16px 18px;
            box-shadow: 0 10px 30px rgba(16, 32, 51, .06);
            height: 100%;
        }
        .status-card .label { color: #667085; font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; }
        .status-card .value { color: #111827; font-size: 1.34rem; font-weight: 750; margin-top: 4px; }
        .status-card .note { color: #667085; font-size: .84rem; margin-top: 5px; }
        .upload-guide {
            border: 1px dashed rgba(89, 103, 255, .35);
            background: rgba(89, 103, 255, .055);
            border-radius: 18px;
            padding: 14px 16px;
            color: #26345f;
        }
        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid rgba(16,32,51,.08);
            border-radius: 18px;
            padding: 14px 16px;
            box-shadow: 0 10px 28px rgba(16,32,51,.06);
        }
        .block-container { padding-top: 1.7rem; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            background: white;
            border: 1px solid rgba(16,32,51,.10);
            padding: 10px 16px;
        }
        .stTabs [aria-selected="true"] { background: #eef2ff !important; color: #3730a3 !important; }
        button[kind="primary"] { border-radius: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>⚖️ FairTeam AI</h1>
          <p>팀 프로젝트 기여도 추적 · OpenAI 회의록 구조화 · 조작 방어 · 교수자용 공정평가 리포트</p>
          <div class="pill-row">
            <span class="pill">CSV/TXT 직접 업로드</span>
            <span class="pill">API Key 선택 입력</span>
            <span class="pill">근거 기반 점수</span>
            <span class="pill">다팀 비교</span>
            <span class="pill">감사 가능한 리포트</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="status-card">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
          <div class="note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def percent_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
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
        if col in out.columns:
            out[col] = out[col].apply(lambda x: "" if pd.isna(x) else f"{x*100:.1f}%")
    return out


def template_downloads() -> None:
    with st.expander("CSV 템플릿 다운로드", expanded=False):
        c1, c2, c3, c4, c5 = st.columns(5)
        templates = [
            (c1, "github_log", REQUIRED_GITHUB_COLUMNS),
            (c2, "docs_revision", REQUIRED_DOC_COLUMNS),
            (c3, "slides_revision", REQUIRED_SLIDE_COLUMNS),
            (c4, "roles", REQUIRED_ROLE_COLUMNS),
            (c5, "self_evaluation", REQUIRED_SELF_EVAL_COLUMNS),
        ]
        for col, name, required in templates:
            with col:
                df = template_dataframe(required)
                st.download_button(
                    f"{name}.csv",
                    data=df.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"{name}_template.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


def data_upload_panel(sample_bundle: Dict[str, object]) -> Tuple[Dict[str, object] | None, str, pd.DataFrame | None]:
    """Build active input bundle from dashboard uploads.

    Returns bundle, source label, and a health table. If required files are still
    missing and sample fallback is disabled, returns (None, label, health_df).
    """
    st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
    st.subheader("입력 데이터 업로드")
    st.markdown(
        "<div class='upload-guide'>샘플 데이터를 끄면, 아래 업로드 파일들이 즉시 현재 분석 입력으로 사용됩니다. 업로드 파일을 바꾸면 모든 탭의 차트·리포트·원자료 미리보기가 새 데이터 기준으로 다시 계산됩니다.</div>",
        unsafe_allow_html=True,
    )
    st.write("")
    template_downloads()

    fill_missing_from_sample = st.toggle(
        "누락된 파일은 샘플 데이터로 임시 보완",
        value=False,
        help="실제 제출/시연 전에는 끄는 것을 권장합니다. 켜면 일부 파일만 업로드해도 전체 화면을 확인할 수 있습니다.",
    )

    up1, up2, up3 = st.columns(3)
    with up1:
        meeting_file = st.file_uploader("회의록 TXT/MD", type=["txt", "md"], help=BUNDLE_FILE_SPECS["meeting_notes"]["help"])
        github_file = st.file_uploader("GitHub 로그 CSV", type=["csv"], help=BUNDLE_FILE_SPECS["github_log"]["help"])
    with up2:
        docs_file = st.file_uploader("문서 수정 기록 CSV", type=["csv"], help=BUNDLE_FILE_SPECS["docs_revision"]["help"])
        slides_file = st.file_uploader("발표자료 수정 기록 CSV", type=["csv"], help=BUNDLE_FILE_SPECS["slides_revision"]["help"])
    with up3:
        roles_file = st.file_uploader("역할분담/업무완료 CSV", type=["csv"], help=BUNDLE_FILE_SPECS["roles"]["help"])
        self_eval_file = st.file_uploader("자기평가 CSV", type=["csv"], help=BUNDLE_FILE_SPECS["self_eval"]["help"])

    st.markdown("#### GitHub API로 `github_log.csv` 자동 생성")
    g1, g2, g3, g4 = st.columns([2.2, 1, 1, 1.4])
    with g1:
        repo_url = st.text_input("GitHub repo", placeholder="owner/repo 또는 https://github.com/owner/repo")
    with g2:
        branch = st.text_input("branch/SHA", value="")
    with g3:
        max_commits = st.number_input("max commits", min_value=1, max_value=300, value=120, step=10)
    with g4:
        github_token = st.text_input("GitHub token", value="", type="password", help="비공개 저장소 또는 rate limit 회피용. 입력값은 저장하지 않습니다.")
    if st.button("GitHub 로그 가져오기", disabled=not bool(repo_url.strip()), use_container_width=True):
        try:
            fetched = fetch_github_log(
                GitHubIngestConfig(
                    repo=repo_url,
                    token=github_token or None,
                    branch=branch or None,
                    max_commits=int(max_commits),
                )
            )
            st.session_state["fetched_github_log"] = fetched
            st.success(f"GitHub 로그를 가져왔습니다: {len(fetched)}명")
        except Exception as exc:
            st.error(f"GitHub API 수집 실패: {exc}")

    fetched_github = st.session_state.get("fetched_github_log")

    uploaded_sources = {
        "meeting_notes": meeting_file,
        "github_log": github_file,
        "docs_revision": docs_file,
        "slides_revision": slides_file,
        "roles": roles_file,
        "self_eval": self_eval_file,
    }
    bundle: Dict[str, object] = {}
    missing_files = []

    try:
        bundle["meeting_notes"] = read_text(meeting_file) if meeting_file is not None else sample_bundle["meeting_notes"]
        if meeting_file is None and not fill_missing_from_sample:
            missing_files.append("meeting_notes.txt")

        if github_file is not None:
            bundle["github_log"] = safe_read_csv(github_file, REQUIRED_GITHUB_COLUMNS)
        elif isinstance(fetched_github, pd.DataFrame):
            bundle["github_log"] = fetched_github
        else:
            bundle["github_log"] = sample_bundle["github_log"]
            if not fill_missing_from_sample:
                missing_files.append("github_log.csv 또는 GitHub API 수집")

        csv_specs = [
            ("docs_revision", docs_file, REQUIRED_DOC_COLUMNS, "docs_revision.csv"),
            ("slides_revision", slides_file, REQUIRED_SLIDE_COLUMNS, "slides_revision.csv"),
            ("roles", roles_file, REQUIRED_ROLE_COLUMNS, "roles.csv"),
            ("self_eval", self_eval_file, REQUIRED_SELF_EVAL_COLUMNS, "self_evaluation.csv"),
        ]
        for key, uploaded, required, filename in csv_specs:
            if uploaded is not None:
                bundle[key] = safe_read_csv(uploaded, required)
            else:
                bundle[key] = sample_bundle[key]
                if not fill_missing_from_sample:
                    missing_files.append(filename)
    except Exception as exc:
        st.error(f"업로드 파일을 읽는 중 오류가 발생했습니다: {exc}")
        st.stop()

    health_df = pd.DataFrame(bundle_health_rows(bundle))
    st.markdown("#### 현재 입력 상태")
    st.dataframe(health_df, use_container_width=True)

    if missing_files and not fill_missing_from_sample:
        st.warning("분석을 시작하려면 다음 파일이 필요합니다: " + ", ".join(missing_files))
        st.markdown("</div>", unsafe_allow_html=True)
        return None, "업로드 대기", health_df

    uploaded_count = sum(1 for x in uploaded_sources.values() if x is not None) + (1 if isinstance(fetched_github, pd.DataFrame) and github_file is None else 0)
    source_label = f"업로드 데이터 {uploaded_count}개" + (" + 샘플 보완" if fill_missing_from_sample else "")
    st.markdown("</div>", unsafe_allow_html=True)
    return bundle, source_label, health_df


def save_outputs(
    *,
    bundle: Dict[str, object],
    report,
    rows: pd.DataFrame,
    audit_df: pd.DataFrame,
    meeting_insights_df: pd.DataFrame,
    professor_report_md: str,
    team_report_md: str,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows.to_csv(OUT_DIR / "fairteam_member_scores.csv", index=False)
    audit_df.to_csv(OUT_DIR / "fairteam_quality_audit.csv", index=False)
    meeting_insights_df.to_csv(OUT_DIR / "fairteam_meeting_insights.csv", index=False)
    (OUT_DIR / "professor_report.md").write_text(professor_report_md, encoding="utf-8")
    (OUT_DIR / "team_report.md").write_text(team_report_md, encoding="utf-8")
    (OUT_DIR / "scoring_policy.md").write_text(report.score_policy_md, encoding="utf-8")
    save_bundle_to_directory(bundle, OUT_DIR / "last_dashboard_inputs")


def main() -> None:
    inject_css()
    hero()

    sample_bundle = load_project_bundle(SAMPLE_DIR)

    with st.sidebar:
        st.header("실행 설정")
        data_mode = st.radio("데이터 소스", options=["샘플 데이터", "업로드 데이터"], index=0)
        redact_export = st.toggle("개인정보 자동 마스킹", value=True)

        st.markdown("---")
        st.subheader("OpenAI API")
        api_key_input = st.text_input(
            "OpenAI API Key",
            value="",
            type="password",
            placeholder="sk-...",
            help="회의록 구조화에만 사용합니다. 입력값은 파일로 저장하지 않습니다.",
        )
        env_key_exists = bool(os.getenv("OPENAI_API_KEY"))
        default_llm = bool(api_key_input.strip() or env_key_exists)
        use_openai_llm = st.toggle(
            "OpenAI로 회의록 구조화",
            value=default_llm,
            help="키가 없거나 openai 패키지가 없거나 호출에 실패하면 규칙 기반 fallback을 자동 사용합니다.",
        )
        if api_key_input.strip():
            st.success("입력한 API Key를 현재 세션에서 사용")
        elif env_key_exists:
            st.info("환경변수 OPENAI_API_KEY 감지")
        else:
            st.caption("키가 없으면 fallback 분석 사용")

        st.markdown("---")
        st.subheader("가중치")
        project_type = st.selectbox("프로젝트 유형", options=list(DEFAULT_WEIGHTS.keys()), index=list(DEFAULT_WEIGHTS.keys()).index("development"))
        base_weights = DEFAULT_WEIGHTS[project_type]
        code_w = st.slider("코드", 0.0, 1.0, float(base_weights["code"]), 0.01)
        doc_w = st.slider("보고서/문서", 0.0, 1.0, float(base_weights["document"]), 0.01)
        slide_w = st.slider("발표자료", 0.0, 1.0, float(base_weights["slide"]), 0.01)
        meeting_w = st.slider("회의/관리", 0.0, 1.0, float(base_weights["meeting"]), 0.01)
        role_w = st.slider("역할 이행", 0.0, 1.0, float(base_weights["role"]), 0.01)
        custom_weights = {"code": code_w, "document": doc_w, "slide": slide_w, "meeting": meeting_w, "role": role_w}

        st.markdown("---")
        st.subheader("다팀 비교")
        team_root_path = st.text_input("팀별 입력 폴더 루트", value="", placeholder="예: ./class_teams")

    if data_mode == "샘플 데이터":
        bundle = sample_bundle
        source_label = "샘플 데이터"
        health_df = pd.DataFrame(bundle_health_rows(bundle))
        with st.expander("샘플 입력 상태 보기", expanded=False):
            st.dataframe(health_df, use_container_width=True)
    else:
        bundle, source_label, health_df = data_upload_panel(sample_bundle)
        if bundle is None:
            st.stop()

    members = infer_members(
        bundle["github_log"],
        bundle["docs_revision"],
        bundle["slides_revision"],
        bundle["roles"],
        bundle["self_eval"],
        meeting_notes=str(bundle["meeting_notes"]),
    )
    if not members:
        st.error("팀원을 찾지 못했습니다. 업로드 CSV의 member 컬럼 또는 회의록의 이름 표기를 확인하세요.")
        st.stop()

    report = compute_fairness_report(
        members=members,
        project_type=project_type,
        custom_weights=custom_weights,
        use_llm_meeting_analysis=use_openai_llm,
        openai_api_key=api_key_input.strip() or None,
        **bundle,
    )

    rows = pd.DataFrame(members_to_rows(report.members))
    rows_display = percent_columns(rows)
    professor_report_md = report.professor_report_md
    team_report_md = report.team_report_md
    if redact_export:
        professor_report_md = redact_text(professor_report_md)
        team_report_md = redact_text(team_report_md)

    meeting_insights_df = pd.DataFrame(report.meeting_insights)
    audit_df = pd.DataFrame(report.audit_rows)
    engine_label = "fallback"
    if not meeting_insights_df.empty and "analysis_engine" in meeting_insights_df.columns:
        engines = sorted(set(str(x) for x in meeting_insights_df["analysis_engine"].dropna().tolist()))
        engine_label = ", ".join(engines) if engines else "fallback"
    elif use_openai_llm and (api_key_input.strip() or os.getenv("OPENAI_API_KEY")):
        engine_label = "openai 시도 후 fallback/empty"

    st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
    top1, top2, top3, top4 = st.columns(4)
    with top1:
        status_card("현재 데이터", source_label, "화면 전체가 이 입력 기준으로 계산됩니다")
    with top2:
        status_card("팀원 수", str(len(members)), ", ".join(members[:5]) + ("..." if len(members) > 5 else ""))
    with top3:
        status_card("회의록 분석", engine_label, "OpenAI 실패/미설정 시 fallback")
    with top4:
        status_card("출력 위치", "outputs/", "저장 버튼 클릭 시 현재 결과 저장")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("현재 분석 결과를 outputs 폴더에 저장", type="primary", use_container_width=True):
        save_outputs(
            bundle=bundle,
            report=report,
            rows=rows,
            audit_df=audit_df,
            meeting_insights_df=meeting_insights_df,
            professor_report_md=professor_report_md,
            team_report_md=team_report_md,
        )
        st.success("현재 대시보드 입력과 분석 결과를 outputs/에 저장했습니다.")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
        ["대시보드", "근거 로그", "AI 회의록 구조화", "품질/조작 감사", "갈등/위험", "다팀 비교", "개입/이의제기", "교수자 리포트", "원자료"]
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
        fig.update_layout(yaxis_tickformat=".0%", yaxis_title="기여도", xaxis_title="팀원", legend_title_text="기준")
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
        st.subheader("AI 회의록 구조화 신호")
        st.caption("OpenAI API Key를 입력하고 토글을 켜면 LLM 구조화를 시도합니다. 키가 없거나 호출에 실패하면 같은 스키마의 규칙 기반 fallback으로 자동 전환됩니다.")
        if meeting_insights_df.empty:
            st.info("회의록에서 구조화할 신호가 크게 감지되지 않았습니다.")
        else:
            show_df = meeting_insights_df.copy()
            if redact_export and "source_sentence" in show_df.columns:
                show_df["source_sentence"] = show_df["source_sentence"].astype(str).apply(redact_text)
            st.dataframe(show_df, use_container_width=True)
            polarity_counts = meeting_insights_df.groupby(["member", "polarity"]).size().reset_index(name="count")
            fig = px.bar(polarity_counts, x="member", y="count", color="polarity", title="팀원별 회의록 구조화 신호 수")
            st.plotly_chart(fig, use_container_width=True)
        st.download_button(
            "회의록 구조화 JSON 다운로드",
            data=meeting_insights_df.to_json(orient="records", force_ascii=False, indent=2).encode("utf-8"),
            file_name="fairteam_meeting_insights.json",
            mime="application/json",
        )

    with tab4:
        st.subheader("품질/조작 감사")
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
        with st.expander("점수 산식과 보정 원칙"):
            st.markdown(report.score_policy_md)
        st.info("이 탭은 자동 판정이 아니라 교수자가 원자료를 다시 볼 지점을 표시합니다. 커밋 쪼개기·대량 붙여넣기·반복 커밋 메시지·동일 파일 반복 수정·마감 직전 집중·단일 출처 의존 등을 검토 신호로 다룹니다.")

    with tab5:
        st.subheader("갈등 위험 및 무임승차 위험")
        risk_rows = rows[rows["risk_tags"].astype(str).str.len() > 0]
        if risk_rows.empty:
            st.success("위험 태그가 붙은 팀원이 없습니다.")
        else:
            st.dataframe(percent_columns(risk_rows[["member", "contribution_share", "self_claim_share", "overclaim_gap", "risk_tags"]]), use_container_width=True)
        st.markdown("### 회의록에서 감지된 갈등 문장")
        if report.conflict_evidence:
            for line in report.conflict_evidence:
                st.write("- " + (redact_text(line) if redact_export else line))
        else:
            st.write("명시적 갈등 문장이 크게 감지되지 않았습니다.")

    with tab6:
        st.subheader("교수자용 다팀 비교")
        current_row = pd.DataFrame(
            [
                {
                    "team": "current_dashboard_input",
                    "members": len(report.members),
                    "gini": round(report.gini, 4),
                    "imbalance_ratio": round(report.imbalance_ratio, 4),
                    "conflict_risk_score": round(report.conflict_risk_score, 4),
                    "high_risk_members": sum(1 for m in report.members if m.risk_tags),
                    "avg_confidence_score": round(sum(m.confidence_score for m in report.members) / max(len(report.members), 1), 4),
                }
            ]
        )
        comparison = current_row
        if team_root_path.strip():
            try:
                extra = analyze_many_teams(team_root_path.strip(), project_type=project_type)
                if not extra.empty:
                    comparison = pd.concat([current_row, extra], ignore_index=True)
            except Exception as exc:
                st.warning(f"다팀 폴더 분석 실패: {exc}")
        st.dataframe(comparison, use_container_width=True)
        if len(comparison) > 1:
            fig = px.scatter(comparison, x="gini", y="conflict_risk_score", size="high_risk_members", hover_name="team", title="팀별 불균형 × 갈등 위험")
            fig.update_layout(xaxis_title="Gini", yaxis_title="갈등 위험")
            st.plotly_chart(fig, use_container_width=True)
        st.caption("팀별 폴더는 meeting_notes.txt, github_log.csv, docs_revision.csv, slides_revision.csv, roles.csv, self_evaluation.csv를 포함해야 합니다.")

    with tab7:
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

    with tab8:
        st.subheader("교수자용 공정평가 리포트")
        st.markdown(professor_report_md)
        st.download_button("교수자 리포트 Markdown 다운로드", data=professor_report_md.encode("utf-8"), file_name="fairteam_professor_report.md", mime="text/markdown")
        st.download_button("팀원별 점수 CSV 다운로드", data=rows.to_csv(index=False).encode("utf-8"), file_name="fairteam_member_scores.csv", mime="text/csv")
        st.download_button("조작/품질 감사 CSV 다운로드", data=audit_df.to_csv(index=False).encode("utf-8"), file_name="fairteam_quality_audit.csv", mime="text/csv")
        st.download_button("회의록 구조화 CSV 다운로드", data=meeting_insights_df.to_csv(index=False).encode("utf-8"), file_name="fairteam_meeting_insights.csv", mime="text/csv")

    with tab9:
        st.subheader("입력 원자료 미리보기")
        st.markdown("#### 입력 상태")
        st.dataframe(health_df, use_container_width=True)
        st.markdown("#### 회의록")
        preview_notes = redact_text(bundle["meeting_notes"]) if redact_export else str(bundle["meeting_notes"])
        st.text_area("meeting_notes", preview_notes, height=220)
        for key in ["github_log", "docs_revision", "slides_revision", "roles", "self_eval"]:
            st.markdown(f"#### {key}")
            frame = redact_dataframe(bundle[key]) if redact_export else bundle[key]
            st.dataframe(frame, use_container_width=True)


if __name__ == "__main__":
    main()
