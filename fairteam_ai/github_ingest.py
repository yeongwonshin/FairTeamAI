from __future__ import annotations

"""GitHub REST API ingestion for FairTeam AI.

The dashboard can still accept CSV uploads, but this module lets a user provide a
GitHub repository URL and generate a compatible `github_log.csv` automatically.
It intentionally uses the Python standard library so the project has no hard
runtime dependency on requests or PyGithub.

Upgrade notes:
- Commits are still collected with additions/deletions/files_changed.
- Closed issues are now collected from the Issues API and attributed primarily to
  the `closed_by` user, falling back to assignees/author when needed.
- Merged PRs are now collected from the Pulls API and attributed to the merge
  actor when available, falling back to the PR author.
- PR reviews and review comments are collected per PR and counted for reviewers.
- Anti-gaming metadata such as repeated messages, dominant files, generated files,
  bugfix commits, and test commits remains compatible with quality.py.
"""

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import pandas as pd


@dataclass
class GitHubIngestConfig:
    repo: str
    token: str | None = None
    branch: str | None = None
    max_commits: int = 120
    max_issues: int = 100
    max_prs: int = 80
    sleep_seconds: float = 0.05


def parse_repo_slug(repo_url_or_slug: str) -> Tuple[str, str]:
    value = str(repo_url_or_slug or "").strip().rstrip("/")
    if value.startswith("git@github.com:"):
        value = value.split(":", 1)[1]
    if value.startswith("http://") or value.startswith("https://"):
        parsed = urllib.parse.urlparse(value)
        parts = [p for p in parsed.path.strip("/").split("/") if p]
    else:
        parts = [p for p in value.split("/") if p]
    if len(parts) < 2:
        raise ValueError("GitHub repo는 owner/repo 또는 https://github.com/owner/repo 형식이어야 합니다.")
    owner, repo = parts[0], parts[1].replace(".git", "")
    return owner, repo


def _request_json(url: str, token: str | None = None) -> object:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "FairTeamAI/1.0",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")[:500]
        raise RuntimeError(f"GitHub API 오류 HTTP {e.code}: {detail}") from e


def _api(owner: str, repo: str, path: str, **query: object) -> str:
    cleaned = {k: v for k, v in query.items() if v is not None and v != ""}
    qs = urllib.parse.urlencode(cleaned)
    suffix = f"?{qs}" if qs else ""
    return f"https://api.github.com/repos/{owner}/{repo}/{path.lstrip('/')}{suffix}"


def _login(obj: object) -> str | None:
    if isinstance(obj, dict):
        val = obj.get("login") or obj.get("name") or obj.get("email")
        if val:
            return str(val).strip()
    return None


def _name_from_commit(commit: Dict) -> str:
    author = commit.get("author") or {}
    login = _login(author)
    if login:
        return login
    commit_obj = commit.get("commit") or {}
    c_author = commit_obj.get("author") or {}
    return str(c_author.get("name") or c_author.get("email") or "unknown").strip()


def _is_bugfix(message: str) -> bool:
    return bool(re.search(r"\b(fix|bug|hotfix|error|crash|issue|defect)\b|버그|수정|오류", message, re.I))


def _is_test(message: str, files: Iterable[str]) -> bool:
    if re.search(r"\b(test|pytest|unittest|spec|ci|coverage)\b|테스트", message, re.I):
        return True
    return any("test" in f.lower() or "spec" in f.lower() or ".github/workflows" in f.lower() for f in files)


def _is_generated_file(path: str) -> bool:
    low = path.lower()
    return any(
        marker in low
        for marker in [
            "dist/",
            "build/",
            "node_modules/",
            "__pycache__",
            ".min.js",
            "package-lock",
            "yarn.lock",
            ".generated",
            "generated/",
        ]
    )


def _default_member_bucket() -> Dict[str, object]:
    return {
        "commits": 0,
        "additions": 0,
        "deletions": 0,
        "files_changed": 0,
        "issues_closed": 0,
        "prs_merged": 0,
        "reviews": 0,
        "review_comments": 0,
        "bugfix_commits": 0,
        "test_commits": 0,
        "commit_messages": [],
        "touched_files": [],
        "generated_files": 0,
        "closed_issue_numbers": [],
        "merged_pr_numbers": [],
        "reviewed_pr_numbers": [],
    }


def _bump(acc: Dict[str, Dict[str, object]], member: str | None, field: str, amount: int = 1) -> None:
    if not member:
        return
    member = str(member).strip() or "unknown"
    acc[member][field] = int(acc[member].get(field, 0) or 0) + int(amount)


def _append(acc: Dict[str, Dict[str, object]], member: str | None, field: str, value: object) -> None:
    if not member:
        return
    member = str(member).strip() or "unknown"
    target = acc[member].setdefault(field, [])
    if isinstance(target, list):
        target.append(value)


def _issue_actor(issue: Dict) -> str | None:
    actor = _login(issue.get("closed_by"))
    if actor:
        return actor
    assignees = issue.get("assignees") or []
    if isinstance(assignees, list) and assignees:
        actor = _login(assignees[0])
        if actor:
            return actor
    return _login(issue.get("user"))


def _pr_merge_actor(pr_detail: Dict, pr_summary: Dict) -> str | None:
    return _login(pr_detail.get("merged_by")) or _login(pr_summary.get("user")) or _login(pr_detail.get("user"))


def _safe_list(payload: object) -> List[Dict]:
    if not isinstance(payload, list):
        return []
    return [x for x in payload if isinstance(x, dict)]


def fetch_github_log(config: GitHubIngestConfig) -> pd.DataFrame:
    owner, repo = parse_repo_slug(config.repo)
    acc: Dict[str, Dict[str, object]] = defaultdict(_default_member_bucket)

    # 1) Commit-level engineering activity.
    commit_query = {"per_page": min(max(config.max_commits, 1), 100)}
    if config.branch:
        commit_query["sha"] = config.branch
    commits = _safe_list(_request_json(_api(owner, repo, "commits", **commit_query), config.token))

    for item in commits[: config.max_commits]:
        detail_url = item.get("url")
        detail = _request_json(str(detail_url), config.token) if detail_url else item
        if config.sleep_seconds:
            time.sleep(config.sleep_seconds)
        if not isinstance(detail, dict):
            continue
        member = _name_from_commit(detail)
        commit_obj = detail.get("commit") or item.get("commit") or {}
        message = str(commit_obj.get("message", "")).splitlines()[0][:160]
        stats = detail.get("stats", {}) if isinstance(detail, dict) else {}
        files = detail.get("files", []) if isinstance(detail, dict) else []
        filenames = [str(f.get("filename", "")) for f in files if isinstance(f, dict)]

        _bump(acc, member, "commits")
        _bump(acc, member, "additions", int(stats.get("additions", 0) or 0))
        _bump(acc, member, "deletions", int(stats.get("deletions", 0) or 0))
        _bump(acc, member, "files_changed", len(filenames))
        _bump(acc, member, "bugfix_commits", int(_is_bugfix(message)))
        _bump(acc, member, "test_commits", int(_is_test(message, filenames)))
        _bump(acc, member, "generated_files", sum(1 for f in filenames if _is_generated_file(f)))
        _append(acc, member, "commit_messages", message)
        for filename in filenames:
            _append(acc, member, "touched_files", filename)

    # 2) Closed issue ownership. The issues endpoint includes PRs, so filter those out.
    issues = _safe_list(
        _request_json(
            _api(owner, repo, "issues", state="closed", per_page=min(max(config.max_issues, 1), 100)),
            config.token,
        )
    )
    for issue in issues[: config.max_issues]:
        if "pull_request" in issue:
            continue
        actor = _issue_actor(issue)
        _bump(acc, actor, "issues_closed")
        if issue.get("number") is not None:
            _append(acc, actor, "closed_issue_numbers", issue.get("number"))

    # 3) Merged PRs and reviews/review comments.
    pulls = _safe_list(
        _request_json(
            _api(owner, repo, "pulls", state="closed", per_page=min(max(config.max_prs, 1), 100)),
            config.token,
        )
    )
    for pr in pulls[: config.max_prs]:
        number = pr.get("number")
        pr_detail = pr
        if number is not None:
            try:
                pr_detail_payload = _request_json(_api(owner, repo, f"pulls/{number}"), config.token)
                if isinstance(pr_detail_payload, dict):
                    pr_detail = pr_detail_payload
            except Exception:
                pr_detail = pr
            if config.sleep_seconds:
                time.sleep(config.sleep_seconds)

        merged_at = pr_detail.get("merged_at") or pr.get("merged_at")
        if merged_at:
            actor = _pr_merge_actor(pr_detail, pr)
            _bump(acc, actor, "prs_merged")
            _append(acc, actor, "merged_pr_numbers", number)

        if number is None:
            continue
        try:
            reviews = _safe_list(_request_json(_api(owner, repo, f"pulls/{number}/reviews", per_page=100), config.token))
        except Exception:
            reviews = []
        for review in reviews:
            reviewer = _login(review.get("user"))
            _bump(acc, reviewer, "reviews")
            _append(acc, reviewer, "reviewed_pr_numbers", number)
        if config.sleep_seconds and reviews:
            time.sleep(config.sleep_seconds)

        try:
            comments = _safe_list(_request_json(_api(owner, repo, f"pulls/{number}/comments", per_page=100), config.token))
        except Exception:
            comments = []
        for comment in comments:
            reviewer = _login(comment.get("user"))
            _bump(acc, reviewer, "reviews")
            _bump(acc, reviewer, "review_comments")
            _append(acc, reviewer, "reviewed_pr_numbers", number)
        if config.sleep_seconds and comments:
            time.sleep(config.sleep_seconds)

    rows: List[dict] = []
    for member, data in sorted(acc.items()):
        files = list(data.get("touched_files", []))
        file_counts: Dict[str, int] = defaultdict(int)
        for f in files:
            file_counts[str(f)] += 1
        dominant_file_ratio = max(file_counts.values()) / max(len(files), 1) if files else 0.0
        messages = [str(x) for x in data.get("commit_messages", [])]
        unique_message_ratio = len(set(messages)) / max(len(messages), 1) if messages else 0.0
        rows.append(
            {
                "member": member,
                "commits": int(data.get("commits", 0) or 0),
                "additions": int(data.get("additions", 0) or 0),
                "deletions": int(data.get("deletions", 0) or 0),
                "files_changed": int(data.get("files_changed", 0) or 0),
                "issues_closed": int(data.get("issues_closed", 0) or 0),
                "prs_merged": int(data.get("prs_merged", 0) or 0),
                "reviews": int(data.get("reviews", 0) or 0),
                "review_comments": int(data.get("review_comments", 0) or 0),
                "bugfix_commits": int(data.get("bugfix_commits", 0) or 0),
                "test_commits": int(data.get("test_commits", 0) or 0),
                "commit_messages": " | ".join(messages[:30]),
                "unique_message_ratio": round(unique_message_ratio, 4),
                "dominant_file_ratio": round(dominant_file_ratio, 4),
                "generated_files": int(data.get("generated_files", 0) or 0),
                "closed_issue_numbers": ",".join(str(x) for x in data.get("closed_issue_numbers", [])[:30]),
                "merged_pr_numbers": ",".join(str(x) for x in data.get("merged_pr_numbers", [])[:30]),
                "reviewed_pr_numbers": ",".join(str(x) for x in data.get("reviewed_pr_numbers", [])[:30]),
            }
        )
    return pd.DataFrame(rows)
