"""
Developer Agent — generates file-level patches via Claude tool_use.

Features:
  - Chain-of-thought reasoning before patching
  - Structured output via tool_use (no free-form text parsing)
  - Unified diff generation with DiffApplier
  - Dedup guard: identical patch hash → clears context to force re-fetch
  - Retry prompt incorporates structured FeedbackPayload
  - Diff validation before returning patch
"""
from __future__ import annotations

import difflib
import hashlib

import anthropic

from src.config import settings
from src.schema import (
    ContextPayload, FeedbackPayload, FilePatch,
    GitHubIssue, PatchPayload, SystemState,
)
from src.utils.diff_validator import validate_diff
from src.utils.logger import get_logger
from src.utils.rate_limiter import call_with_backoff

log = get_logger(__name__)

_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

PATCH_TOOL: dict = {
    "name": "submit_patch",
    "description": "Submit ALL file patches that together resolve the GitHub issue.",
    "input_schema": {
        "type": "object",
        "properties": {
            "root_cause": {
                "type": "string",
                "description": "2-3 sentence root cause analysis"
            },
            "patches": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file_path":        {"type": "string"},
                        "original_snippet": {"type": "string", "description": "Exact verbatim lines to replace"},
                        "patched_snippet":  {"type": "string", "description": "Replacement lines"},
                        "explanation":      {"type": "string"},
                        "is_new_file":      {"type": "boolean", "default": False},
                    },
                    "required": ["file_path", "original_snippet", "patched_snippet", "explanation"],
                },
            },
        },
        "required": ["root_cause", "patches"],
    },
}


class DeveloperAgent:
    def run(self, state: SystemState) -> SystemState:
        issue:    GitHubIssue      = state["issue"]
        context:  ContextPayload   = state["context"]
        feedback: FeedbackPayload | None = state.get("feedback")
        retry:    int              = state.get("retry_count", 0)
        seen:     list[str]        = state.get("seen_patch_hashes", [])
        tokens:   int              = state.get("cumulative_tokens", 0)

        log.info("developer_agent_start", issue=issue.number, attempt=retry + 1)

        prompt = self._build_prompt(issue, context, feedback, retry)

        def _call():
            return _client.messages.create(
                model=settings.anthropic_model,
                max_tokens=4096,
                tools=[PATCH_TOOL],            # type: ignore[list-item]
                tool_choice={"type": "any"},
                messages=[{"role": "user", "content": prompt}],
            )

        response = call_with_backoff(_call)
        tokens += response.usage.input_tokens + response.usage.output_tokens

        tool_block = next((b for b in response.content if b.type == "tool_use"), None)
        if tool_block is None:
            log.error("developer_no_tool_call")
            return {**state, "retry_count": retry + 1, "cumulative_tokens": tokens}  # type: ignore[return-value]

        raw: dict = tool_block.input  # type: ignore[attr-defined]
        patches = self._build_patches(raw.get("patches", []))

        if not patches:
            log.warning("developer_empty_patches")
            return {**state, "retry_count": retry + 1, "cumulative_tokens": tokens}  # type: ignore[return-value]

        unified = self._make_unified_diff(patches)
        patch_hash = hashlib.sha256(unified.encode()).hexdigest()[:16]

        # ── Diff validation ────────────────────────────────────────────────────
        validation = validate_diff(unified)
        if not validation.valid:
            log.warning("diff_invalid", errors=validation.errors)
            return {**state, "retry_count": retry + 1, "cumulative_tokens": tokens}  # type: ignore[return-value]

        # ── Dedup guard ────────────────────────────────────────────────────────
        if patch_hash in seen:
            log.warning("duplicate_patch_detected", hash=patch_hash)
            return {
                **state,  # type: ignore[return-value]
                "context": None,
                "feedback": None,
                "patch": None,
                "retry_count": retry + 1,
                "cumulative_tokens": tokens,
            }

        log.info("developer_agent_complete",
                 files=len(patches), hash=patch_hash, attempt=retry + 1,
                 lines_added=validation.lines_added, lines_removed=validation.lines_removed)

        return {
            **state,  # type: ignore[return-value]
            "patch": PatchPayload(
                patches=patches,
                unified_diff=unified,
                patch_hash=patch_hash,
                attempt=retry + 1,
                overall_explanation=raw.get("root_cause", ""),
            ),
            "feedback": None,
            "seen_patch_hashes": seen + [patch_hash],
            "retry_count": retry + 1,
            "cumulative_tokens": tokens,
        }

    # ── Prompt ────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_prompt(
        issue: GitHubIssue,
        context: ContextPayload,
        feedback: FeedbackPayload | None,
        retry: int,
    ) -> str:
        chunks_md = "\n\n".join(
            f"### {c.file_path}:{c.start_line}-{c.end_line}"
            f" ({c.chunk_type}, score={c.relevance_score:.2f})\n"
            f"```{c.language}\n{c.source}\n```"
            for c in context.chunks
        )

        retry_section = ""
        if feedback and retry > 0:
            retry_section = f"""
## ⚠️ Previous attempt {retry} FAILED — fix the specific error below

**Error category:** `{feedback.error_category.value}`
**Affected lines:** {", ".join(feedback.affected_lines) or "see traceback"}
**Root cause analysis:** {feedback.root_cause_analysis}
**Suggested fix:** {feedback.suggested_fix_hint}

**Test stderr:**
```
{feedback.test_result.stderr[-3000:]}
```
**Failed tests:**
{chr(10).join(f"  - {t}" for t in feedback.test_result.failed_tests) or "  (none named)"}

Do NOT repeat the same change. Address the specific failure above.
"""

        missing = context.symbols_missing[:10]
        coverage_note = (
            f"\n> ⚠️ These symbols from the issue were NOT found in retrieved context: "
            f"{', '.join(f'`{s}`' for s in missing)}\n> They may be in files not yet retrieved."
            if missing else ""
        )

        return f"""You are a senior software engineer resolving a production GitHub issue.

## Issue #{issue.number}: {issue.title}
**Labels:** {', '.join(issue.labels) or 'none'}

{issue.body or '(no body)'}

{"### User comments\n" + chr(10).join(f"- {c[:300]}" for c in issue.comments[:3]) if issue.comments else ""}

## Relevant codebase context
{coverage_note}
{chunks_md}
{retry_section}
## Instructions

Think step by step:
1. **Identify** the root cause in the retrieved code
2. **Determine** the minimal set of changes needed
3. **Call** `submit_patch` with:
   - `root_cause`: 2-3 sentence explanation
   - `patches`: list of file-level changes
     - `original_snippet`: EXACT verbatim lines from the source (copy-paste, no paraphrasing)
     - `patched_snippet`: corrected replacement
     - `explanation`: one sentence per patch
     - `is_new_file`: true only for new files

**Rules:**
- Only change what is necessary — minimal diff
- No `print()` debug statements
- No commented-out code  
- Preserve existing code style and naming conventions
- Ensure syntactic validity
- `original_snippet` must be an exact substring of the file — do not paraphrase
"""

    # ── Patch helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _build_patches(raw: list[dict]) -> list[FilePatch]:
        out: list[FilePatch] = []
        for p in raw:
            fp   = (p.get("file_path") or "").strip()
            orig = p.get("original_snippet", "")
            new  = p.get("patched_snippet", "")
            if fp and orig != new:
                out.append(FilePatch(
                    file_path=fp,
                    original_snippet=orig,
                    patched_snippet=new,
                    explanation=p.get("explanation", ""),
                    is_new_file=p.get("is_new_file", False),
                ))
        return out

    @staticmethod
    def _make_unified_diff(patches: list[FilePatch]) -> str:
        parts: list[str] = []
        for p in patches:
            diff = difflib.unified_diff(
                p.original_snippet.splitlines(keepends=True),
                p.patched_snippet.splitlines(keepends=True),
                fromfile=f"a/{p.file_path}",
                tofile=f"b/{p.file_path}",
                lineterm="",
            )
            parts.append("\n".join(diff))
        return "\n".join(parts)


def developer_agent_node(state: SystemState) -> SystemState:
    return DeveloperAgent().run(state)
