from __future__ import annotations

"""GitHub REST API ingestion for FairTeam AI.

The dashboard can still accept CSV uploads, but this module lets a user provide a
GitHub repository URL and generate a compatible `github_log.csv` automatically.
It intentionally uses the Python standard library so the project has no hard
runtime dependency on requests or PyGithub.
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
    sleep_seconds: float = 0.05


def parse_repo_slug(repo_url_or_slug: str) -> Tuple[str, str]:
    value = repo_url_or_slug.strip().rstrip("/")
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


def _name_from_commit(commit: Dict) -> str:
    author = commit.get("author") or {}
    if isinstance(author, dict) and author.get("login"):
        return str(author["login"])
    commit_obj = commit.get("commit") or {}
    c_author = commit_obj.get("author") or {}
    return str(c_author.get("name") or c_author.get("email") or "unknown").strip()


def _is_bugfix(message: str) -> bool:
    return bool(re.search(r"\b(fix|bug|hotfix|error|crash|issue)\b|버그|수정", message, re.I))


def _is_test(message: str, files: Iterable[str]) -> bool:
    if re.search(r"\b(test|pytest|unittest|spec)\b|테스트", message, re.I):
        return True
    return any("test" in f.lower() or "spec" in f.lower() for f in files)


def _is_generated_file(path: str) -> bool:
    low = path.lower()
    return any(
        marker in low
        for marker in ["dist/", "build/", "node_modules/", "__pycache__", ".min.js", "package-lock", "yarn.lock", ".generated"]
    )


def fetch_github_log(config: GitHubIngestConfig) -> pd.DataFrame:
    owner, repo = parse_repo_slug(config.repo)
    query = {"per_page": min(max(config.max_commits, 1), 100)}
    if config.branch:
        query["sha"] = config.branch
    url = f"https://api.github.com/repos/{owner}/{repo}/commits?{urllib.parse.urlencode(query)}"
    commits = _request_json(url, config.token)
    if not isinstance(commits, list):
        raise RuntimeError("GitHub commits 응답 형식이 예상과 다릅니다.")

    acc: Dict[str, Dict[str, object]] = defaultdict(
        lambda: {
            "commits": 0,
            "additions": 0,
            "deletions": 0,
            "files_changed": 0,
            "issues_closed": 0,
            "prs_merged": 0,
            "reviews": 0,
            "bugfix_commits": 0,
            "test_commits": 0,
            "commit_messages": [],
            "touched_files": [],
            "generated_files": 0,
        }
    )

    for item in commits[: config.max_commits]:
        detail_url = item.get("url")
        detail = _request_json(detail_url, config.token) if detail_url else item
        if config.sleep_seconds:
            time.sleep(config.sleep_seconds)
        member = _name_from_commit(detail if isinstance(detail, dict) else item)
        commit_obj = (detail.get("commit") if isinstance(detail, dict) else item.get("commit")) or {}
        message = str(commit_obj.get("message", "")).splitlines()[0][:160]
        stats = detail.get("stats", {}) if isinstance(detail, dict) else {}
        files = detail.get("files", []) if isinstance(detail, dict) else []
        filenames = [str(f.get("filename", "")) for f in files if isinstance(f, dict)]
        target = acc[member]
        target["commits"] = int(target["commits"]) + 1
        target["additions"] = int(target["additions"]) + int(stats.get("additions", 0) or 0)
        target["deletions"] = int(target["deletions"]) + int(stats.get("deletions", 0) or 0)
        target["files_changed"] = int(target["files_changed"]) + len(filenames)
        target["bugfix_commits"] = int(target["bugfix_commits"]) + int(_is_bugfix(message))
        target["test_commits"] = int(target["test_commits"]) + int(_is_test(message, filenames))
        target["generated_files"] = int(target["generated_files"]) + sum(1 for f in filenames if _is_generated_file(f))
        target["commit_messages"].append(message)
        target["touched_files"].extend(filenames)

    rows: List[dict] = []
    for member, data in sorted(acc.items()):
        files = list(data["touched_files"])
        file_counts: Dict[str, int] = defaultdict(int)
        for f in files:
            file_counts[f] += 1
        dominant_file_ratio = max(file_counts.values()) / max(len(files), 1) if files else 0.0
        messages = list(data["commit_messages"])
        unique_message_ratio = len(set(messages)) / max(len(messages), 1) if messages else 0.0
        rows.append(
            {
                "member": member,
                "commits": data["commits"],
                "additions": data["additions"],
                "deletions": data["deletions"],
                "files_changed": data["files_changed"],
                "issues_closed": data["issues_closed"],
                "prs_merged": data["prs_merged"],
                "reviews": data["reviews"],
                "bugfix_commits": data["bugfix_commits"],
                "test_commits": data["test_commits"],
                "commit_messages": " | ".join(messages[:30]),
                "unique_message_ratio": round(unique_message_ratio, 4),
                "dominant_file_ratio": round(dominant_file_ratio, 4),
                "generated_files": data["generated_files"],
            }
        )
    return pd.DataFrame(rows)
