from __future__ import annotations

import argparse
from pathlib import Path

from fairteam_ai.github_ingest import GitHubIngestConfig, fetch_github_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch GitHub commits and generate a FairTeam-compatible github_log.csv")
    parser.add_argument("--repo", required=True, help="owner/repo or https://github.com/owner/repo")
    parser.add_argument("--token", default=None, help="GitHub token. Optional, but recommended for private repos/rate limits.")
    parser.add_argument("--branch", default=None, help="Branch or SHA to inspect")
    parser.add_argument("--max-commits", type=int, default=120)
    parser.add_argument("--out", default="sample_data/github_log.csv")
    args = parser.parse_args()

    df = fetch_github_log(
        GitHubIngestConfig(repo=args.repo, token=args.token, branch=args.branch, max_commits=args.max_commits)
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} member rows to {out}")


if __name__ == "__main__":
    main()
