"""
Tester / Execution Agent — applies the patch in a Docker sandbox and runs tests.

On failure: Claude performs structured failure analysis returning a
FeedbackPayload with error_category, affected_lines, root_cause_analysis,
and a targeted suggested_fix_hint for the Developer Agent.
"""
from __future__ import annotations

import anthropic

from src.config import settings
from src.sandbox.manager import SandboxManager
from src.schema import (
    ErrorCategory, FeedbackPayload, PatchPayload,
    SystemState, TestResult, TestVerdict,
)
from src.utils.logger import get_logger
from src.utils.rate_limiter import call_with_backoff

log = get_logger(__name__)

_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

FAILURE_TOOL: dict = {
    "name": "classify_failure",
    "description": "Classify the test failure and give targeted guidance to the Developer Agent.",
    "input_schema": {
        "type": "object",
        "properties": {
            "error_category": {
                "type": "string",
                "enum": ["syntax", "logic", "missing_import", "test_expectation", "unknown"],
            },
            "root_cause_analysis": {
                "type": "string",
                "description": "2-3 sentence explanation of WHY the tests failed",
            },
            "affected_lines": {
                "type": "array",
                "items": {"type": "string"},
                "description": "file:line references from the traceback",
            },
            "suggested_fix_hint": {
                "type": "string",
                "description": "ONE specific, actionable fix instruction for the Developer Agent",
            },
        },
        "required": ["error_category", "root_cause_analysis", "affected_lines", "suggested_fix_hint"],
    },
}

_sandbox: SandboxManager | None = None


class TesterAgent:
    def __init__(self) -> None:
        global _sandbox
        if _sandbox is None:
            _sandbox = SandboxManager()
        self._sandbox = _sandbox

    def run(self, state: SystemState) -> SystemState:
        patch: PatchPayload = state["patch"]
        repo_path: str = state.get("repo_local_path", "")
        tokens: int = state.get("cumulative_tokens", 0)

        log.info("tester_agent_start", attempt=patch.attempt, hash=patch.patch_hash)

        test_result: TestResult = self._sandbox.apply_patch_and_test(
            repo_local_path=repo_path,
            unified_diff=patch.unified_diff,
        )

        log.info("test_result",
                 verdict=test_result.verdict.value,
                 exit_code=test_result.exit_code,
                 failures=len(test_result.failed_tests),
                 duration_ms=test_result.duration_ms,
                 runner=test_result.test_runner)

        # Pass — no analysis needed
        if test_result.verdict == TestVerdict.PASS:
            return {
                **state,  # type: ignore[return-value]
                "feedback": FeedbackPayload(
                    error_category=ErrorCategory.UNKNOWN,
                    affected_lines=[],
                    suggested_fix_hint="",
                    test_result=test_result,
                    root_cause_analysis="Tests passed.",
                ),
            }

        # Fail — analyse with Claude
        feedback, used = self._analyse(patch, test_result)
        tokens += used

        log.info("failure_analysis_complete",
                 category=feedback.error_category.value,
                 hint=feedback.suggested_fix_hint[:80])

        return {
            **state,  # type: ignore[return-value]
            "feedback": feedback,
            "cumulative_tokens": tokens,
        }

    def _analyse(self, patch: PatchPayload, result: TestResult) -> tuple[FeedbackPayload, int]:
        stderr_tail = (result.stderr or result.stdout)[-3_000:]
        failed_list = "\n".join(f"- {t}" for t in result.failed_tests) or "  (not extracted)"

        prompt = f"""You are a senior engineer analysing a failed automated test run.

## Patch applied (attempt {patch.attempt})
Overall explanation: {patch.overall_explanation or "(none)"}

```diff
{patch.unified_diff[:3000]}
```

## Test failures
{failed_list}

## Test output
```
{stderr_tail}
```

Call `classify_failure` with a precise analysis that will help the Developer Agent fix the issue on the next attempt.
Focus on WHY the test failed, not just what failed.
"""
        def _call():
            return _client.messages.create(
                model=settings.anthropic_model,
                max_tokens=512,
                tools=[FAILURE_TOOL],         # type: ignore[list-item]
                tool_choice={"type": "any"},
                messages=[{"role": "user", "content": prompt}],
            )

        response = call_with_backoff(_call)
        used = response.usage.input_tokens + response.usage.output_tokens

        tool_block = next((b for b in response.content if b.type == "tool_use"), None)
        if tool_block is None:
            return FeedbackPayload(
                error_category=ErrorCategory.UNKNOWN,
                affected_lines=[],
                suggested_fix_hint="Review the test output and fix accordingly.",
                test_result=result,
                root_cause_analysis="Failure analysis unavailable.",
            ), used

        raw: dict = tool_block.input  # type: ignore[attr-defined]
        return FeedbackPayload(
            error_category=ErrorCategory(raw.get("error_category", "unknown")),
            affected_lines=raw.get("affected_lines", []),
            suggested_fix_hint=raw.get("suggested_fix_hint", ""),
            test_result=result,
            root_cause_analysis=raw.get("root_cause_analysis", ""),
        ), used


def tester_agent_node(state: SystemState) -> SystemState:
    return TesterAgent().run(state)
