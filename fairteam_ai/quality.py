from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


@dataclass
class MemberQualityAudit:
    member: str
    quality_score: float = 1.0
    anti_gaming_score: float = 1.0
    confidence_score: float = 0.0
    source_coverage: Dict[str, bool] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)
    audit_rows: List[Dict[str, Any]] = field(default_factory=list)


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if df is None or df.empty or col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0)


def _member_rows(df: pd.DataFrame, member: str) -> pd.DataFrame:
    if df is None or df.empty or "member" not in df.columns:
        return pd.DataFrame()
    tmp = df.copy()
    tmp["member"] = tmp["member"].astype(str).str.strip()
    return tmp[tmp["member"] == member]


def _sum_for(df: pd.DataFrame, member: str, cols: Iterable[str]) -> Dict[str, float]:
    rows = _member_rows(df, member)
    return {col: float(_num(rows, col).sum()) if not rows.empty else 0.0 for col in cols}


def _has_positive(df: pd.DataFrame, member: str, cols: Iterable[str]) -> bool:
    rows = _member_rows(df, member)
    return any(float(_num(rows, col).sum()) > 0 for col in cols) if not rows.empty else False


def _parse_time_column(df: pd.DataFrame) -> pd.Series | None:
    if df is None or df.empty:
        return None
    for col in ["timestamp", "created_at", "updated_at", "date", "time"]:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() >= 2:
                return parsed
    return None


def _last_minute_ratio(df: pd.DataFrame, member: str, effort_cols: Iterable[str]) -> float | None:
    rows = _member_rows(df, member)
    if rows.empty:
        return None
    ts = _parse_time_column(rows)
    if ts is None or ts.notna().sum() < 2:
        return None
    effort = sum((_num(rows, col) for col in effort_cols), start=pd.Series(0.0, index=rows.index))
    if float(effort.sum()) <= 0:
        effort = pd.Series(1.0, index=rows.index)
    valid = rows.assign(_ts=ts, _effort=effort).dropna(subset=["_ts"])
    if len(valid) < 2:
        return None
    start, end = valid["_ts"].min(), valid["_ts"].max()
    cutoff = start + (end - start) * 0.8
    return float(valid.loc[valid["_ts"] >= cutoff, "_effort"].sum() / max(valid["_effort"].sum(), 1e-9))


def _add_flag(audit: MemberQualityAudit, flag: str, severity: float, source: str, evidence: str) -> None:
    if flag not in audit.flags:
        audit.flags.append(flag)
    audit.quality_score = max(0.0, audit.quality_score - severity)
    audit.anti_gaming_score = max(0.0, audit.anti_gaming_score - severity * 1.25)
    audit.audit_rows.append(
        {
            "member": audit.member,
            "source": source,
            "flag": flag,
            "severity": round(float(severity), 3),
            "evidence": evidence,
        }
    )


def _text_values(rows: pd.DataFrame, candidates: Iterable[str]) -> List[str]:
    values: List[str] = []
    if rows is None or rows.empty:
        return values
    for col in candidates:
        if col in rows.columns:
            values.extend(str(x) for x in rows[col].dropna().tolist() if str(x).strip())
    return values


def _ratio_from_optional_column(rows: pd.DataFrame, col: str) -> float | None:
    if rows is None or rows.empty or col not in rows.columns:
        return None
    series = pd.to_numeric(rows[col], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.mean())


def _meeting_negative_severity(meeting_insights: pd.DataFrame | None, member: str) -> float:
    if meeting_insights is None or meeting_insights.empty:
        return 0.0
    tmp = meeting_insights.copy()
    if "member" not in tmp.columns:
        return 0.0
    tmp["member"] = tmp["member"].astype(str).str.strip()
    tmp["severity"] = pd.to_numeric(tmp.get("severity", 0), errors="coerce").fillna(0.0)
    return float(tmp.loc[(tmp["member"] == member) & (tmp.get("polarity", "") == "negative"), "severity"].sum())


def build_quality_audit(
    *,
    members: List[str],
    github_log: pd.DataFrame,
    docs_revision: pd.DataFrame,
    slides_revision: pd.DataFrame,
    roles: pd.DataFrame,
    self_eval: pd.DataFrame,
    meeting_insights: pd.DataFrame | None = None,
) -> Dict[str, MemberQualityAudit]:
    """Detect weak evidence, gaming-like patterns, and confidence per member.

    This is deliberately transparent. It never accuses a member; it marks records
    that deserve human review before a professor uses the score.
    """
    out: Dict[str, MemberQualityAudit] = {}
    source_cols = {
        "code": ["commits", "additions", "deletions", "files_changed", "issues_closed", "prs_merged", "reviews"],
        "document": ["edits", "words_added", "comments_resolved", "sections_owned", "suggestions_accepted", "references_added"],
        "slide": ["slides_edited", "visuals_created", "script_words", "presenter_minutes"],
        "role": ["assigned_tasks", "completed_tasks", "late_tasks", "critical_tasks"],
        "self_eval": ["self_claim_percent"],
    }
    frames = {"code": github_log, "document": docs_revision, "slide": slides_revision, "role": roles, "self_eval": self_eval}

    for member in members:
        audit = MemberQualityAudit(member=member)
        coverage = {src: _has_positive(frames[src], member, cols) for src, cols in source_cols.items()}
        audit.source_coverage = coverage
        covered_count = sum(coverage.values())

        code_rows = _member_rows(github_log, member)
        doc_rows = _member_rows(docs_revision, member)
        slide_rows = _member_rows(slides_revision, member)
        self_rows = _member_rows(self_eval, member)

        code = _sum_for(github_log, member, source_cols["code"] + ["bugfix_commits", "test_commits", "generated_files"])
        doc = _sum_for(docs_revision, member, source_cols["document"])
        slide = _sum_for(slides_revision, member, source_cols["slide"])
        role = _sum_for(roles, member, source_cols["role"])

        commits = code.get("commits", 0.0)
        changed_lines = code.get("additions", 0.0) + code.get("deletions", 0.0)
        if commits >= 8 and changed_lines / max(commits, 1.0) < 6 and code.get("prs_merged", 0.0) + code.get("issues_closed", 0.0) <= 1:
            _add_flag(
                audit,
                "커밋 쪼개기/저품질 커밋 의심",
                0.16,
                "github_log",
                f"commits={commits:.0f}, changed_lines_per_commit={changed_lines / max(commits, 1.0):.1f}, pr_or_issue={code.get('prs_merged', 0.0)+code.get('issues_closed', 0.0):.0f}",
            )

        if code.get("additions", 0.0) >= 2500 and code.get("files_changed", 0.0) <= 2:
            _add_flag(
                audit,
                "대량 붙여넣기성 코드 변경 의심",
                0.12,
                "github_log",
                f"additions={code.get('additions', 0.0):.0f}, files_changed={code.get('files_changed', 0.0):.0f}",
            )

        generated_ratio = code.get("generated_files", 0.0) / max(code.get("files_changed", 0.0), 1.0)
        if code.get("generated_files", 0.0) >= 3 and generated_ratio >= 0.35:
            _add_flag(audit, "자동 생성/빌드 산출물 기여 과대평가 의심", 0.10, "github_log", f"generated_file_ratio={generated_ratio:.2f}")

        unique_message_ratio = _ratio_from_optional_column(code_rows, "unique_message_ratio")
        if unique_message_ratio is not None and commits >= 5 and unique_message_ratio <= 0.35:
            _add_flag(audit, "반복 커밋 메시지 패턴", 0.08, "github_log", f"unique_message_ratio={unique_message_ratio:.2f}")

        dominant_file_ratio = _ratio_from_optional_column(code_rows, "dominant_file_ratio")
        if dominant_file_ratio is not None and commits >= 5 and dominant_file_ratio >= 0.70:
            _add_flag(audit, "동일 파일 반복 수정 집중", 0.08, "github_log", f"dominant_file_ratio={dominant_file_ratio:.2f}")

        messages = " | ".join(_text_values(code_rows, ["commit_message", "message", "commit_messages"])).lower()
        vague_tokens = ["update", "fix", "final", "asdf", "misc", "wip", "수정", "최종"]
        if commits >= 6 and messages:
            vague_count = sum(messages.count(tok) for tok in vague_tokens)
            if vague_count >= max(4, commits * 0.45):
                _add_flag(audit, "의미가 약한 커밋 메시지 과다", 0.07, "github_log", f"vague_message_hits={vague_count}")

        if doc.get("words_added", 0.0) >= 3000 and doc.get("sections_owned", 0.0) <= 1 and doc.get("comments_resolved", 0.0) <= 1:
            _add_flag(
                audit,
                "문서 대량 추가 대비 검토/담당 근거 부족",
                0.12,
                "docs_revision",
                f"words_added={doc.get('words_added', 0.0):.0f}, sections_owned={doc.get('sections_owned', 0.0):.0f}, comments_resolved={doc.get('comments_resolved', 0.0):.0f}",
            )

        if doc.get("words_added", 0.0) >= 1200 and doc.get("references_added", 0.0) == 0 and doc.get("comments_resolved", 0.0) <= 1:
            _add_flag(audit, "문서 분량 대비 검증 흔적 부족", 0.08, "docs_revision", f"words_added={doc.get('words_added', 0.0):.0f}, references_added=0")

        if slide.get("presenter_minutes", 0.0) >= 5 and slide.get("slides_edited", 0.0) == 0 and slide.get("script_words", 0.0) == 0:
            _add_flag(audit, "발표 담당 시간 대비 산출물 로그 부족", 0.08, "slides_revision", f"presenter_minutes={slide.get('presenter_minutes', 0.0):.1f}")

        if role.get("assigned_tasks", 0.0) >= 3:
            completion_rate = role.get("completed_tasks", 0.0) / max(role.get("assigned_tasks", 0.0), 1.0)
            late_rate = role.get("late_tasks", 0.0) / max(role.get("assigned_tasks", 0.0), 1.0)
            if completion_rate < 0.5 or late_rate >= 0.5:
                _add_flag(audit, "업무 지연/미완료 비율 높음", 0.14, "roles", f"completion_rate={completion_rate:.2f}, late_rate={late_rate:.2f}")

        self_claim = float(_num(self_rows, "self_claim_percent").mean()) if not self_rows.empty and "self_claim_percent" in self_rows.columns else 0.0
        if self_claim >= 30.0 and covered_count <= 2:
            _add_flag(audit, "자기평가 주장 대비 확인 출처 부족", 0.10, "self_eval", f"self_claim_percent={self_claim:.1f}, covered_sources={covered_count}")

        negative_sev = _meeting_negative_severity(meeting_insights, member)
        if negative_sev >= 1.2:
            _add_flag(audit, "회의록 구조화 분석상 반복 부정 신호", 0.10, "meeting_insights", f"negative_severity_sum={negative_sev:.2f}")

        for src, df, effort_cols in [
            ("github_log", github_log, ["commits", "additions", "deletions", "files_changed"]),
            ("docs_revision", docs_revision, ["edits", "words_added", "comments_resolved"]),
            ("slides_revision", slides_revision, ["slides_edited", "visuals_created", "script_words"]),
        ]:
            ratio = _last_minute_ratio(df, member, effort_cols)
            if ratio is not None and ratio >= 0.75:
                _add_flag(audit, "마감 직전 기여 집중", 0.08, src, f"last_20pct_time_effort_ratio={ratio:.2f}")

        if covered_count <= 1:
            _add_flag(audit, "단일 출처에 과도하게 의존", 0.08, "source_coverage", f"covered_sources={covered_count}")

        raw_volume = (
            commits
            + np.sqrt(max(changed_lines, 0.0))
            + doc.get("edits", 0.0)
            + np.sqrt(max(doc.get("words_added", 0.0), 0.0))
            + slide.get("slides_edited", 0.0)
            + role.get("completed_tasks", 0.0)
        )
        coverage_bonus = min(0.55, 0.11 * covered_count)
        volume_bonus = min(0.35, np.log1p(raw_volume) / np.log1p(120.0) * 0.35)
        flag_penalty = min(0.35, 0.08 * len(audit.flags))
        audit.confidence_score = float(np.clip(0.12 + coverage_bonus + volume_bonus - flag_penalty, 0.0, 1.0))
        out[member] = audit
    return out


def audit_rows_to_dataframe(audits: Dict[str, MemberQualityAudit]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for audit in audits.values():
        if audit.audit_rows:
            rows.extend(audit.audit_rows)
        else:
            rows.append(
                {
                    "member": audit.member,
                    "source": "all",
                    "flag": "특이 조작 신호 없음",
                    "severity": 0.0,
                    "evidence": "여러 로그 출처와 기본 품질 규칙에서 큰 이상치를 찾지 못했습니다.",
                }
            )
    return pd.DataFrame(rows)
