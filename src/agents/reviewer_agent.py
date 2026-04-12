"""
Reviewer Agent — final quality gate before code touches GitHub.

Steps:
  1. Static style checks (no LLM — cheap and fast)
  2. Security scan (SQL injection, hardcoded secrets, dangerous calls)
  3. LLM-powered review → structured ReviewResult + PR body
  4. PR quality score (1-5)
  5. If approved: create GitHub PR via GitHubClient
"""
from __future__ import annotations

import re
from pathlib import Path

import anthropic

from src.config import settings
from src.github_client import GitHubClient
from src.schema import (
    FeedbackPayload, GitHubIssue, PatchPayload,
    PRResult, ReviewResult, SystemState,
)
from src.utils.logger import get_logger
from src.utils.rate_limiter import call_with_backoff

log = get_logger(__name__)

_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
_github = GitHubClient()

REVIEW_TOOL: dict = {
    "name": "submit_review",
    "description": "Submit the final code review and PR description.",
    "input_schema": {
        "type": "object",
        "properties": {
            "approved":    {"type": "boolean"},
            "quality_score": {
                "type": "number",
                "description": "PR quality 1–5 (5=excellent, 1=reject)",
            },
            "comments":    {"type": "array", "items": {"type": "string"}},
            "pr_body":     {"type": "string", "description": "Full PR body in Markdown"},
            "suggested_labels": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["approved", "quality_score", "comments", "pr_body", "suggested_labels"],
    },
}

# ── Security patterns ──────────────────────────────────────────────────────────
_SECURITY_PATTERNS: list[tuple[str, str]] = [
    (r"^\+.*execute\s*\(\s*['\"].*%.*['\"]", "Possible SQL injection (% string format)"),
    (r"^\+.*eval\s*\(",                       "Use of eval() is dangerous"),
    (r"^\+.*exec\s*\(",                       "Use of exec() is dangerous"),
    (r"^\+.*(password|secret|api_key)\s*=\s*['\"].+['\"]",
                                               "Hardcoded credential detected"),
    (r"^\+.*subprocess\..*shell\s*=\s*True",  "subprocess with shell=True is risky"),
    (r"^\+.*pickle\.loads?\(",                "pickle.load is unsafe with untrusted data"),
    (r"^\+.*yaml\.load\(",                    "Use yaml.safe_load instead of yaml.load"),
]


class ReviewerAgent:
    def run(self, state: SystemState) -> SystemState:
        issue:    GitHubIssue     = state["issue"]
        patch:    PatchPayload    = state["patch"]
        feedback: FeedbackPayload = state["feedback"]
        repo:     str             = state.get("repo_local_path", "")
        tokens:   int             = state.get("cumulative_tokens", 0)

        log.info("reviewer_agent_start", issue=issue.number, attempt=patch.attempt)

        # Step 1: Static checks
        style_issues   = self._static_style_check(patch.unified_diff)
        security_issues = self._security_scan(patch.unified_diff)

        # Step 2: LLM review
        review, used = self._llm_review(issue, patch, feedback, style_issues, security_issues)
        tokens += used

        log.info("review_complete",
                 approved=review.approved,
                 quality=review.pr_quality_score,
                 comments=len(review.comments))

        if not review.approved:
            log.warning("reviewer_rejected", comments=review.comments[:2])
            return {**state, "review": review, "cumulative_tokens": tokens}  # type: ignore[return-value]

        # Step 3: Create GitHub PR
        patch_files = self._extract_patched_files(patch, repo)
        try:
            pr: PRResult = _github.create_pull_request(
                issue=issue,
                patch_files=patch_files,
                review=review,
                repo_local_path=repo,
            )
            log.info("pr_created", pr_number=pr.pr_number, url=pr.pr_url)
        except Exception as exc:
            log.error("pr_creation_failed", error=str(exc))
            return {
                **state,  # type: ignore[return-value]
                "review": review,
                "cumulative_tokens": tokens,
                "error_message": str(exc),
            }

        return {
            **state,  # type: ignore[return-value]
            "review": review,
            "pr": pr,
            "cumulative_tokens": tokens,
        }

    # ── Static checks ─────────────────────────────────────────────────────────

    @staticmethod
    def _static_style_check(diff: str) -> list[str]:
        issues: list[str] = []
        for line in diff.splitlines():
            if not line.startswith("+") or line.startswith("+++"):
                continue
            c = line[1:]
            if re.search(r"\bprint\s*\(", c) and "# noqa" not in c:
                issues.append(f"Debug print(): {c.strip()!r}")
            if re.match(r"\s*#\s*\S", c) and not re.match(r"\s*#\s*(type:|noqa|pylint)", c):
                issues.append(f"Commented-out code: {c.strip()!r}")
            if re.search(r"\b(TODO|FIXME|HACK|XXX)\b", c):
                issues.append(f"TODO/FIXME marker: {c.strip()!r}")
            if len(c.rstrip()) > 120:
                issues.append(f"Line too long ({len(c.rstrip())} chars)")
        return issues

    @staticmethod
    def _security_scan(diff: str) -> list[str]:
        issues: list[str] = []
        for pattern, message in _SECURITY_PATTERNS:
            if re.search(pattern, diff, re.MULTILINE):
                issues.append(f"SECURITY: {message}")
        return issues

    # ── LLM review ────────────────────────────────────────────────────────────

    def _llm_review(
        self,
        issue: GitHubIssue,
        patch: PatchPayload,
        feedback: FeedbackPayload,
        style_issues: list[str],
        security_issues: list[str],
    ) -> tuple[ReviewResult, int]:
        patches_md = "\n\n".join(
            f"**{p.file_path}** — {p.explanation}\n```diff\n"
            + "\n".join(
                f"- {l}" if i % 2 == 0 else f"+ {l}"
                for i, l in enumerate([p.original_snippet, p.patched_snippet])
            ) + "\n```"
            for p in patch.patches
        )
        test_summary = (
            f"✅ All tests passed in {feedback.test_result.duration_ms}ms "
            f"(runner: {feedback.test_result.test_runner})"
            if feedback.test_result.exit_code == 0
            else f"❌ Tests failed: {', '.join(feedback.test_result.failed_tests[:5])}"
        )
        style_md  = "\n".join(f"- {s}" for s in style_issues)  or "None"
        sec_md    = "\n".join(f"- {s}" for s in security_issues) or "None"

        prompt = f"""You are a senior engineer performing a final code review before merging an automated fix.

## Issue #{issue.number}: {issue.title}
{issue.body[:800] if issue.body else ""}

## Changes (attempt {patch.attempt})
Root cause: {patch.overall_explanation}

{patches_md}

## Automated style checks
{style_md}

## Security scan
{sec_md}

## Test results
{test_summary}

## Review criteria
- Is the fix correct and minimal?
- Does it address the stated root cause?
- Are there any regressions risks?
- Is the code style consistent?

Call `submit_review` with:
- `approved`: true if safe and correct to merge
- `quality_score`: 1–5 rating
- `comments`: list of concerns (empty if none)
- `pr_body`: Markdown PR description with:
    ## Summary
    ## Root cause
    ## Changes made
    ## Testing evidence
    ## Risk assessment
- `suggested_labels`: ["automated-fix", "tested", "needs-review"] + extras as appropriate
"""
        def _call():
            return _client.messages.create(
                model=settings.anthropic_model,
                max_tokens=2048,
                tools=[REVIEW_TOOL],          # type: ignore[list-item]
                tool_choice={"type": "any"},
                messages=[{"role": "user", "content": prompt}],
            )

        response = call_with_backoff(_call)
        used = response.usage.input_tokens + response.usage.output_tokens

        tool_block = next((b for b in response.content if b.type == "tool_use"), None)
        if tool_block is None:
            return ReviewResult(
                approved=False, comments=["No review generated."],
                pr_body="", suggested_labels=["automated-fix", "needs-review"],
            ), used

        raw: dict = tool_block.input  # type: ignore[attr-defined]

        # Force reject if security issues found
        approved = raw.get("approved", False)
        if security_issues:
            approved = False
            raw.setdefault("comments", []).insert(0,
                f"Auto-rejected due to security issues: {security_issues[0]}")

        return ReviewResult(
            approved=approved,
            comments=raw.get("comments", []),
            pr_body=raw.get("pr_body", ""),
            suggested_labels=raw.get("suggested_labels", ["automated-fix", "needs-review"]),
            pr_quality_score=float(raw.get("quality_score", 0.0)),
        ), used

    # ── File extraction ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_patched_files(patch: PatchPayload, repo_root: str) -> dict[str, str]:
        result: dict[str, str] = {}
        for fp in patch.patches:
            abs_path = Path(repo_root) / fp.file_path
            if fp.is_new_file:
                result[fp.file_path] = fp.patched_snippet
                continue
            try:
                original = abs_path.read_text(encoding="utf-8")
                result[fp.file_path] = original.replace(fp.original_snippet, fp.patched_snippet, 1)
            except FileNotFoundError:
                result[fp.file_path] = fp.patched_snippet
        return result


def reviewer_agent_node(state: SystemState) -> SystemState:
    return ReviewerAgent().run(state)
