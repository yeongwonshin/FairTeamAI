from fairteam_ai.github_ingest import GitHubIngestConfig, fetch_github_log, parse_repo_slug


def test_parse_repo_slug_variants():
    assert parse_repo_slug("owner/repo") == ("owner", "repo")
    assert parse_repo_slug("https://github.com/owner/repo.git") == ("owner", "repo")
    assert parse_repo_slug("git@github.com:owner/repo.git") == ("owner", "repo")


def test_fetch_github_log_collects_issues_prs_and_reviews(monkeypatch):
    def fake_request_json(url, token=None):
        if url.endswith("/commits?per_page=10"):
            return [{"url": "https://api.github.com/repos/o/r/commits/abc"}]
        if url.endswith("/commits/abc"):
            return {
                "author": {"login": "alice"},
                "commit": {"message": "fix scoring test"},
                "stats": {"additions": 10, "deletions": 2},
                "files": [{"filename": "fairteam_ai/scoring.py"}, {"filename": "tests/test_scoring.py"}],
            }
        if "/issues?" in url:
            return [
                {"number": 3, "closed_by": {"login": "bob"}, "user": {"login": "alice"}},
                {"number": 4, "pull_request": {}, "closed_by": {"login": "ignored"}},
            ]
        if url.endswith("/pulls?state=closed&per_page=10"):
            return [{"number": 7, "merged_at": "2026-05-01T00:00:00Z", "user": {"login": "alice"}}]
        if url.endswith("/pulls/7"):
            return {"number": 7, "merged_at": "2026-05-01T00:00:00Z", "merged_by": {"login": "bob"}, "user": {"login": "alice"}}
        if url.endswith("/pulls/7/reviews?per_page=100"):
            return [{"user": {"login": "carol"}}]
        if url.endswith("/pulls/7/comments?per_page=100"):
            return [{"user": {"login": "dave"}}]
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr("fairteam_ai.github_ingest._request_json", fake_request_json)
    df = fetch_github_log(GitHubIngestConfig(repo="o/r", max_commits=10, max_issues=10, max_prs=10, sleep_seconds=0))
    rows = {row["member"]: row for row in df.to_dict("records")}

    assert rows["alice"]["commits"] == 1
    assert rows["alice"]["bugfix_commits"] == 1
    assert rows["alice"]["test_commits"] == 1
    assert rows["bob"]["issues_closed"] == 1
    assert rows["bob"]["prs_merged"] == 1
    assert rows["carol"]["reviews"] == 1
    assert rows["dave"]["reviews"] == 1
    assert rows["dave"]["review_comments"] == 1
