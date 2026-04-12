"""GitHub integration — issue fetch, repo clone, branch/PR management."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import git
from github import Github, GithubException
from github.Issue import Issue as GHIssue
from github.Repository import Repository

from src.config import settings
from src.schema import GitHubIssue, PRResult, ReviewResult
from src.utils.logger import get_logger
from src.utils.rate_limiter import call_with_backoff

log = get_logger(__name__)


class GitHubClient:
    def __init__(self) -> None:
        self._gh = Github(settings.github_token)
        self._repo: Repository = self._gh.get_repo(settings.github_repo)

    # ── Issue ──────────────────────────────────────────────────────────────────

    def fetch_issue(self, number: int) -> GitHubIssue:
        def _fetch() -> GHIssue:
            return self._repo.get_issue(number)

        raw = call_with_backoff(_fetch)
        comments = [c.body for c in raw.get_comments()]
        labels   = [l.name for l in raw.labels]
        linked   = self._extract_linked_prs(raw.body or "")

        log.info("issue_fetched", number=number, title=raw.title, comments=len(comments))
        return GitHubIssue(
            number=number,
            title=raw.title,
            body=raw.body or "",
            labels=labels,
            comments=comments,
            repo_full_name=self._repo.full_name,
            html_url=raw.html_url,
            created_at=raw.created_at.isoformat() if raw.created_at else "",
            linked_prs=linked,
        )

    # ── Clone ─────────────────────────────────────────────────────────────────

    def clone_repo(self, target_dir: str | None = None) -> str:
        if target_dir is None:
            target_dir = tempfile.mkdtemp(prefix="resolver_")
        url = self._repo.clone_url.replace(
            "https://", f"https://x-access-token:{settings.github_token}@"
        )
        log.info("cloning_repo", repo=self._repo.full_name, target=target_dir)
        git.Repo.clone_from(url, target_dir, depth=1)
        return target_dir

    # ── PR ────────────────────────────────────────────────────────────────────

    def create_pull_request(
        self,
        issue: GitHubIssue,
        patch_files: dict[str, str],
        review: ReviewResult,
        repo_local_path: str,
    ) -> PRResult:
        branch = f"fix/issue-{issue.number}"
        default = self._repo.default_branch

        local = git.Repo(repo_local_path)
        new_branch = local.create_head(branch)
        new_branch.checkout()

        for rel_path, content in patch_files.items():
            abs_path = Path(repo_local_path) / rel_path
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text(content, encoding="utf-8")

        local.git.add(A=True)
        commit = local.index.commit(
            f"fix: resolve issue #{issue.number} — {issue.title}",
            author=git.Actor("github-issue-resolver[bot]", "bot@resolver.local"),
        )

        origin = local.remote("origin")
        origin.push(refspec=f"{branch}:{branch}")
        log.info("branch_pushed", branch=branch)

        def _open_pr():
            pr = self._repo.create_pull(
                title=f"fix(#{issue.number}): {issue.title}",
                body=review.pr_body,
                head=branch,
                base=default,
            )
            if review.suggested_labels:
                pr.set_labels(*review.suggested_labels)
            raw_issue: GHIssue = self._repo.get_issue(issue.number)
            raw_issue.create_comment(
                f"Automated fix proposed in PR #{pr.number}: {pr.html_url}\n\n"
                f"_Resolved by [GitHub Issue Resolver](https://github.com) pipeline._"
            )
            return pr

        pr = call_with_backoff(_open_pr)
        log.info("pr_created", pr_number=pr.number, url=pr.html_url)
        return PRResult(
            pr_number=pr.number,
            pr_url=pr.html_url,
            branch_name=branch,
            commit_sha=commit.hexsha,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_linked_prs(body: str) -> list[int]:
        import re
        return [int(m) for m in re.findall(r"(?:closes?|fixes?|resolves?)\s+#(\d+)", body, re.I)]

    @property
    def default_branch(self) -> str:
        return self._repo.default_branch
