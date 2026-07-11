from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

from fairteam_ai.ai_review import generate_ai_review_brief
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
from fairteam_ai.readiness import calculate_review_readiness
from fairteam_ai.reporting import members_to_rows
from fairteam_ai.scoring import compute_fairness_report
from fairteam_ai.settings import get_openai_settings
from fairteam_ai.workspace import (
    SnapshotStore,
    build_analysis_manifest,
    build_review_package,
    bundle_fingerprint,
)

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
    ai_review_brief_md: str,
    manifest: Dict[str, object],
    review_package: bytes,
) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows.to_csv(OUT_DIR / "fairteam_member_scores.csv", index=False)
    audit_df.to_csv(OUT_DIR / "fairteam_quality_audit.csv", index=False)
    meeting_insights_df.to_csv(OUT_DIR / "fairteam_meeting_insights.csv", index=False)
    (OUT_DIR / "professor_report.md").write_text(professor_report_md, encoding="utf-8")
    (OUT_DIR / "team_report.md").write_text(team_report_md, encoding="utf-8")
    (OUT_DIR / "ai_review_brief.md").write_text(ai_review_brief_md, encoding="utf-8")
    (OUT_DIR / "scoring_policy.md").write_text(report.score_policy_md, encoding="utf-8")
    (OUT_DIR / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (OUT_DIR / "fairteam_review_package.zip").write_bytes(review_package)
    save_bundle_to_directory(bundle, OUT_DIR / "last_dashboard_inputs")
    return SnapshotStore(OUT_DIR / "snapshots").save(manifest)


def main() -> None:
    inject_css()
    hero()

    sample_bundle = load_project_bundle(SAMPLE_DIR)
    env_openai = get_openai_settings()

    with st.sidebar:
        st.header("실행 설정")
        workspace_name = st.text_input("워크스페이스 이름", value="FairTeam Review")
        reviewer_name = st.text_input("검토자 이름", value="", placeholder="선택 입력")
        data_mode = st.radio("데이터 소스", options=["샘플 데이터", "업로드 데이터"], index=0)
        redact_export = st.toggle("리포트 개인정보 자동 마스킹", value=True)
        redact_before_llm = st.toggle(
            "OpenAI 전송 전 개인정보 마스킹",
            value=True,
            help="이메일, 전화번호, 학번, URL 토큰을 마스킹한 회의록만 OpenAI에 전송합니다.",
        )

        st.markdown("---")
        st.subheader("OpenAI API")
        api_key_input = st.text_input(
            "OpenAI API Key",
            value="",
            type="password",
            placeholder=".env에 입력하면 자동 인식",
            help="현재 세션에서만 사용하며 파일이나 분석 결과에 저장하지 않습니다.",
        )
        openai_model = st.text_input("OpenAI model", value=env_openai.model)
        use_openai_llm = st.toggle(
            "OpenAI 회의록 구조화 사용",
            value=env_openai.configured,
            help="호출 실패 또는 키 미설정 시 규칙 기반 분석으로 자동 전환합니다.",
        )
        resolved_openai = get_openai_settings(
            api_key_override=api_key_input.strip() or None,
            model_override=openai_model.strip() or None,
        )
        if api_key_input.strip():
            st.success("세션 API Key 사용 중")
        elif resolved_openai.configured:
            st.info(f".env API Key 감지 · {resolved_openai.model}")
        else:
            st.caption("키가 없으면 모든 핵심 기능이 deterministic fallback으로 동작합니다.")

        st.markdown("---")
        st.subheader("가중치")
        project_type = st.selectbox(
            "프로젝트 유형",
            options=list(DEFAULT_WEIGHTS.keys()),
            index=list(DEFAULT_WEIGHTS.keys()).index("development"),
        )
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

    llm_meeting_notes = (
        redact_text(bundle["meeting_notes"]) if redact_before_llm else str(bundle["meeting_notes"])
    )
    key_fingerprint = hashlib.sha256((resolved_openai.api_key or "").encode("utf-8")).hexdigest()[:8]
    analysis_options = {
        "project_type": project_type,
        "weights": custom_weights,
        "use_openai": use_openai_llm,
        "openai_model": resolved_openai.model,
        "openai_key_fingerprint": key_fingerprint,
        "redact_before_llm": redact_before_llm,
    }
    analysis_fingerprint = bundle_fingerprint(bundle, analysis_options)
    refresh_requested = st.button("분석 실행 / 새로고침", type="primary", use_container_width=True)
    cached_analysis = st.session_state.get("fairteam_analysis")
    if (
        refresh_requested
        or not cached_analysis
        or cached_analysis.get("fingerprint") != analysis_fingerprint
    ):
        with st.spinner("근거를 교차 검증하고 분석 리포트를 생성하는 중입니다..."):
            report = compute_fairness_report(
                members=members,
                project_type=project_type,
                custom_weights=custom_weights,
                use_llm_meeting_analysis=use_openai_llm,
                openai_api_key=resolved_openai.api_key,
                openai_model=resolved_openai.model,
                llm_meeting_notes=llm_meeting_notes,
                **bundle,
            )
        st.session_state["fairteam_analysis"] = {
            "fingerprint": analysis_fingerprint,
            "report": report,
        }
    else:
        report = cached_analysis["report"]

    rows = pd.DataFrame(members_to_rows(report.members))
    rows_display = percent_columns(rows)
    professor_report_md = report.professor_report_md
    team_report_md = report.team_report_md
    if redact_export:
        professor_report_md = redact_text(professor_report_md)
        team_report_md = redact_text(team_report_md)

    meeting_insights_df = pd.DataFrame(report.meeting_insights)
    audit_df = pd.DataFrame(report.audit_rows)
    appeal_store = AppealStore(OUT_DIR / "appeals.jsonl")
    appeals = appeal_store.list()
    unresolved_appeals = sum(1 for appeal in appeals if appeal.status in {"submitted", "under_review"})
    readiness = calculate_review_readiness(
        report=report,
        bundle_health=health_df.to_dict(orient="records"),
        unresolved_appeals=unresolved_appeals,
    )

    brief_key = f"{analysis_fingerprint}:{readiness.score}:{unresolved_appeals}"
    cached_brief = st.session_state.get("fairteam_ai_brief")
    if cached_brief and cached_brief.get("key") == brief_key:
        ai_review_brief = cached_brief["brief"]
    else:
        ai_review_brief = generate_ai_review_brief(
            report=report,
            readiness_status=readiness.status,
            scores=rows,
            audit=audit_df,
            use_llm=False,
        )
    ai_review_brief_md = ai_review_brief.to_markdown()

    manifest = build_analysis_manifest(
        workspace_name=workspace_name,
        reviewer_name=reviewer_name,
        fingerprint=analysis_fingerprint,
        project_type=project_type,
        weights=report.weights,
        members=members,
        source_label=source_label,
        llm_enabled=use_openai_llm and resolved_openai.configured,
        llm_model=resolved_openai.model,
        llm_redacted=redact_before_llm,
        readiness=readiness.to_dict(),
    )
    review_package = build_review_package(
        scores=rows,
        audit=audit_df,
        meeting_insights=meeting_insights_df,
        professor_report_md=professor_report_md,
        team_report_md=team_report_md,
        scoring_policy_md=report.score_policy_md,
        ai_review_brief_md=ai_review_brief_md,
        manifest=manifest,
    )

    engine_label = "rule_fallback"
    if not meeting_insights_df.empty and "analysis_engine" in meeting_insights_df.columns:
        engines = sorted(set(str(x) for x in meeting_insights_df["analysis_engine"].dropna().tolist()))
        engine_label = ", ".join(engines) if engines else "rule_fallback"
    elif use_openai_llm and resolved_openai.configured:
        engine_label = "OpenAI attempted → fallback/empty"

    st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
    top1, top2, top3, top4, top5 = st.columns(5)
    with top1:
        status_card("현재 데이터", source_label, "활성 입력 기준")
    with top2:
        status_card("검토 준비도", f"{readiness.score * 100:.0f}%", readiness.status)
    with top3:
        status_card("팀원 수", str(len(members)), ", ".join(members[:4]) + ("..." if len(members) > 4 else ""))
    with top4:
        status_card("회의록 분석", engine_label, resolved_openai.model if use_openai_llm else "deterministic")
    with top5:
        status_card("분석 ID", analysis_fingerprint, "재현 가능한 스냅샷")
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("현재 분석 결과와 검토 패키지를 outputs 폴더에 저장", use_container_width=True):
        snapshot_path = save_outputs(
            bundle=bundle,
            report=report,
            rows=rows,
            audit_df=audit_df,
            meeting_insights_df=meeting_insights_df,
            professor_report_md=professor_report_md,
            team_report_md=team_report_md,
            ai_review_brief_md=ai_review_brief_md,
            manifest=manifest,
            review_package=review_package,
        )
        st.success(f"저장 완료: outputs/ · 스냅샷 {snapshot_path.name}")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(
        [
            "대시보드",
            "근거 로그",
            "AI 회의록 구조화",
            "AI 검토 브리프",
            "품질/조작 감사",
            "갈등/위험",
            "다팀 비교",
            "개입/이의제기",
            "교수자 리포트",
            "원자료",
        ]
    )

    with tab1:
        st.subheader("핵심 요약")
        st.write(report.summary)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("기여 불균형 Gini", f"{report.gini:.3f}")
        m2.metric("최대/최소 기여 비율", f"{report.imbalance_ratio:.2f}x")
        m3.metric("갈등 위험", f"{report.conflict_risk_score * 100:.1f}%")
        m4.metric("평균 산출 신뢰도", f"{rows['confidence_score'].mean() * 100:.1f}%")

        with st.expander("검토 준비도 상세", expanded=readiness.score < 0.78):
            r1, r2, r3 = st.columns(3)
            r1.metric("근거 소스 커버리지", f"{readiness.evidence_coverage * 100:.0f}%")
            r2.metric("미해결 이의제기", str(readiness.unresolved_appeals))
            r3.metric("핵심 감사 플래그", str(readiness.critical_flags))
            if readiness.blockers:
                st.warning("검토 전 보완: " + " / ".join(readiness.blockers))
            if readiness.strengths:
                st.success("현재 강점: " + " / ".join(readiness.strengths))

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
        st.subheader("AI 검토 브리프")
        st.caption(
            "점수 확정이 아니라, 검토 우선순위·확인 질문·다음 조치를 정리하는 보조 브리프입니다. "
            "OpenAI를 사용하지 않아도 deterministic 브리프가 제공됩니다."
        )
        if st.button(
            "OpenAI로 검토 브리프 다시 생성",
            disabled=not resolved_openai.configured,
            use_container_width=True,
        ):
            with st.spinner("OpenAI가 검토 브리프를 구조화하는 중입니다..."):
                generated_brief = generate_ai_review_brief(
                    report=report,
                    readiness_status=readiness.status,
                    scores=rows,
                    audit=audit_df,
                    use_llm=True,
                    api_key=resolved_openai.api_key,
                    model=resolved_openai.model,
                )
            st.session_state["fairteam_ai_brief"] = {"key": brief_key, "brief": generated_brief}
            st.rerun()

        b1, b2, b3 = st.columns(3)
        b1.metric("검토 준비 상태", readiness.status)
        b2.metric("준비도 점수", f"{readiness.score * 100:.0f}%")
        b3.metric("브리프 엔진", ai_review_brief.analysis_engine)
        st.markdown(ai_review_brief_md)
        st.download_button(
            "AI 검토 브리프 다운로드",
            data=ai_review_brief_md.encode("utf-8"),
            file_name="fairteam_ai_review_brief.md",
            mime="text/markdown",
        )

    with tab5:
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

    with tab6:
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

    with tab7:
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

    with tab8:
        st.subheader("개입 권장안")
        for i, action in enumerate(report.intervention_plan, start=1):
            st.write(f"{i}. {action}")

        st.markdown("---")
        st.subheader("팀원 이의제기 접수")
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
                if not appeal_claim.strip():
                    st.error("이의제기 내용을 입력하세요.")
                else:
                    created = appeal_store.submit(appeal_member, appeal_category, appeal_claim, appeal_ref)
                    st.success(f"이의제기가 접수되었습니다: {created.appeal_id}")
                    st.rerun()

        appeals = appeal_store.list()
        if appeals:
            st.markdown("### 접수된 이의제기")
            st.dataframe(pd.DataFrame([a.__dict__ for a in appeals]), use_container_width=True)
            with st.expander("검토자 상태 업데이트", expanded=False):
                selected_appeal = st.selectbox(
                    "이의제기 선택",
                    options=[a.appeal_id for a in appeals],
                    format_func=lambda appeal_id: next(
                        f"{a.appeal_id} · {a.member} · {a.status}" for a in appeals if a.appeal_id == appeal_id
                    ),
                )
                selected_status = st.selectbox(
                    "검토 상태",
                    options=["submitted", "under_review", "accepted", "rejected"],
                )
                reviewer_note = st.text_area("검토 메모", placeholder="판단 근거와 추가 확인 사항을 기록하세요.")
                if st.button("이의제기 상태 저장", use_container_width=True):
                    updated = appeal_store.update(selected_appeal, selected_status, reviewer_note)
                    if updated:
                        st.success(f"{updated.appeal_id} 상태를 {updated.status}(으)로 변경했습니다.")
                        st.rerun()
        else:
            st.caption("아직 접수된 이의제기가 없습니다.")

    with tab9:
        st.subheader("교수자용 공정평가 리포트")
        st.markdown(professor_report_md)
        st.download_button("교수자 리포트 Markdown 다운로드", data=professor_report_md.encode("utf-8"), file_name="fairteam_professor_report.md", mime="text/markdown")
        st.download_button("팀원별 점수 CSV 다운로드", data=rows.to_csv(index=False).encode("utf-8"), file_name="fairteam_member_scores.csv", mime="text/csv")
        st.download_button("조작/품질 감사 CSV 다운로드", data=audit_df.to_csv(index=False).encode("utf-8"), file_name="fairteam_quality_audit.csv", mime="text/csv")
        st.download_button("회의록 구조화 CSV 다운로드", data=meeting_insights_df.to_csv(index=False).encode("utf-8"), file_name="fairteam_meeting_insights.csv", mime="text/csv")
        st.download_button(
            "전체 검토 패키지 ZIP 다운로드",
            data=review_package,
            file_name=f"fairteam_review_{analysis_fingerprint}.zip",
            mime="application/zip",
            use_container_width=True,
        )
        st.download_button(
            "분석 manifest JSON 다운로드",
            data=json.dumps(manifest, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
            file_name="fairteam_analysis_manifest.json",
            mime="application/json",
        )

    with tab10:
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
