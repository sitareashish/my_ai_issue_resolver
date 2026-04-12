"""
Complete test suite — Sprint 1-4 acceptance criteria.

Coverage targets:
  - Unit: schema, config, diff_validator, token_counter, rate_limiter
  - Integration: all 5 agents (mocked external services)
  - Pipeline: orchestrator routing, retry loop, budget abort
  - API: FastAPI endpoints

Run: pytest src/tests/ -v --cov=src
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from src.schema import (
    CodeChunk, ContextPayload, ErrorCategory, FeedbackPayload,
    FilePatch, GitHubIssue, PatchPayload, PipelineStatus,
    TestResult, TestVerdict,
)

# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def issue() -> GitHubIssue:
    return GitHubIssue(
        number=42,
        title="Bug: divide_by_zero not handled in calculate()",
        body="When calling `calculate(x, 0)` the function raises ZeroDivisionError instead of returning None.",
        labels=["bug", "good-first-issue"],
        comments=["Reproduced on v2.1.0"],
        repo_full_name="owner/myrepo",
        html_url="https://github.com/owner/myrepo/issues/42",
    )


@pytest.fixture
def chunk() -> CodeChunk:
    return CodeChunk(
        file_path="myrepo/math_utils.py",
        start_line=10, end_line=14,
        language="python", chunk_type="function",
        source="def calculate(x, y):\n    return x / y\n",
        relevance_score=0.93,
    )


@pytest.fixture
def context(chunk) -> ContextPayload:
    return ContextPayload(chunks=[chunk], confidence=0.88,
                          symbols_found=["calculate"], symbols_missing=[])


@pytest.fixture
def patch_payload() -> PatchPayload:
    return PatchPayload(
        patches=[FilePatch(
            file_path="myrepo/math_utils.py",
            original_snippet="    return x / y",
            patched_snippet="    return x / y if y != 0 else None",
            explanation="Guard against ZeroDivisionError",
        )],
        unified_diff=(
            "--- a/myrepo/math_utils.py\n+++ b/myrepo/math_utils.py\n"
            "@@ -2 +2 @@\n-    return x / y\n+    return x / y if y != 0 else None\n"
        ),
        patch_hash="abc123456789abcd",
        attempt=1,
        overall_explanation="Added zero-division guard.",
    )


@pytest.fixture
def passing_feedback(patch_payload) -> FeedbackPayload:
    return FeedbackPayload(
        error_category=ErrorCategory.UNKNOWN,
        affected_lines=[],
        suggested_fix_hint="",
        test_result=TestResult(
            verdict=TestVerdict.PASS, failed_tests=[],
            stdout="3 passed in 0.12s", stderr="",
            duration_ms=120, exit_code=0, test_runner="pytest",
        ),
        root_cause_analysis="Tests passed.",
    )


@pytest.fixture
def failing_feedback(patch_payload) -> FeedbackPayload:
    return FeedbackPayload(
        error_category=ErrorCategory.LOGIC,
        affected_lines=["myrepo/math_utils.py:11"],
        suggested_fix_hint="Return None when y == 0 instead of dividing.",
        test_result=TestResult(
            verdict=TestVerdict.FAIL,
            failed_tests=["test_calculate_zero_division"],
            stdout="", stderr="ZeroDivisionError: division by zero\n  File math_utils.py:11",
            duration_ms=80, exit_code=1, test_runner="pytest",
        ),
        root_cause_analysis="y==0 still reaches the division operator.",
    )


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA & CONFIG TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestSchema:
    def test_pipeline_status_values(self):
        assert PipelineStatus.PASSED.value == "passed"
        assert PipelineStatus.FAILED.value == "failed"

    def test_test_verdict_values(self):
        assert TestVerdict.PASS.value  == "pass"
        assert TestVerdict.FAIL.value  == "fail"
        assert TestVerdict.ERROR.value == "error"

    def test_code_chunk_defaults(self, chunk):
        assert chunk.bm25_score == 0.0
        assert chunk.hybrid_score == 0.0

    def test_patch_payload_immutable_hash(self, patch_payload):
        assert len(patch_payload.patch_hash) == 16

    def test_context_payload_fields(self, context):
        assert context.confidence == 0.88
        assert context.search_strategy == "hybrid"


# ══════════════════════════════════════════════════════════════════════════════
# UTILS TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestDiffValidator:
    def test_valid_diff_passes(self):
        from src.utils.diff_validator import validate_diff
        diff = (
            "--- a/foo.py\n+++ b/foo.py\n"
            "@@ -1,2 +1,2 @@\n def f():\n-    return 1\n+    return 2\n"
        )
        r = validate_diff(diff)
        assert r.valid
        assert r.lines_added == 1
        assert r.lines_removed == 1
        assert "foo.py" in r.files_changed

    def test_empty_diff_fails(self):
        from src.utils.diff_validator import validate_diff
        r = validate_diff("")
        assert not r.valid
        assert any("empty" in e.lower() for e in r.errors)

    def test_debug_print_warning(self):
        from src.utils.diff_validator import validate_diff
        diff = (
            "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n"
            "-old\n+print('debug')\n"
        )
        r = validate_diff(diff)
        assert any("print" in w for w in r.warnings)

    def test_no_op_diff_fails(self):
        from src.utils.diff_validator import validate_diff
        diff = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n unchanged\n"
        r = validate_diff(diff)
        assert not r.valid

    def test_file_path_extraction(self):
        from src.utils.diff_validator import extract_file_paths_from_diff
        diff = "--- a/src/foo.py\n+++ b/src/foo.py\n@@ -1 +1 @@\n-x\n+y\n"
        paths = extract_file_paths_from_diff(diff)
        assert "src/foo.py" in paths


class TestTokenCounter:
    def test_count_tokens_nonempty(self):
        from src.utils.token_counter import count_tokens
        n = count_tokens("Hello, world!")
        assert n > 0

    def test_budget_warn_then_abort(self):
        from src.utils.token_counter import BudgetExceededError, TokenBudget
        budget = TokenBudget(hard_limit=1000, warn_limit=500)
        budget.add(300, 100)        # 400 — below warn
        budget.add(50,  100)        # 550 — triggers warn
        assert budget.used == 550
        with pytest.raises(BudgetExceededError):
            budget.add(400, 200)    # 1150 — exceeds hard limit

    def test_budget_remaining(self):
        from src.utils.token_counter import TokenBudget
        b = TokenBudget(hard_limit=10_000, warn_limit=5_000)
        b.add(1000, 500)
        assert b.remaining == 8_500

    def test_count_messages(self):
        from src.utils.token_counter import count_messages_tokens
        msgs = [{"role": "user", "content": "Hello world"}]
        assert count_messages_tokens(msgs) > 0


class TestRateLimiter:
    def test_success_on_first_call(self):
        from src.utils.rate_limiter import call_with_backoff
        result = call_with_backoff(lambda: 42)
        assert result == 42

    def test_retries_on_transient_error(self):
        import anthropic
        from src.utils.rate_limiter import call_with_backoff
        calls = []
        def fn():
            calls.append(1)
            if len(calls) < 3:
                raise anthropic.APIConnectionError(request=MagicMock())
            return "ok"
        result = call_with_backoff(fn, base_delay=0.001)
        assert result == "ok"
        assert len(calls) == 3

    def test_raises_after_max_attempts(self):
        import anthropic
        from src.utils.rate_limiter import call_with_backoff
        def fn():
            raise anthropic.RateLimitError(
                message="rate limited", response=MagicMock(), body={}
            )
        with pytest.raises(anthropic.RateLimitError):
            call_with_backoff(fn, max_attempts=2, base_delay=0.001)


# ══════════════════════════════════════════════════════════════════════════════
# AST CHUNKER TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestASTChunker:
    def test_chunks_python_functions(self, tmp_path: Path):
        code = textwrap.dedent("""\
            def foo(x):
                return x + 1

            def bar(y):
                return y * 2
        """)
        (tmp_path / "sample.py").write_text(code)
        from src.indexer.ast_chunker import ASTChunker
        chunks = ASTChunker().chunk_repository(str(tmp_path))
        sources = [c.source for c in chunks]
        assert any("def foo" in s for s in sources)
        assert any("def bar" in s for s in sources)

    def test_no_mid_function_split(self, tmp_path: Path):
        lines = ["def big_fn():\n"] + [f"    x_{i} = {i}\n" for i in range(60)] + ["    return x_0\n"]
        (tmp_path / "big.py").write_text("".join(lines))
        from src.indexer.ast_chunker import ASTChunker
        chunks = ASTChunker().chunk_repository(str(tmp_path))
        big = [c for c in chunks if "big_fn" in c.source]
        assert len(big) == 1, "big_fn must not be split"

    def test_skips_binary_files(self, tmp_path: Path):
        (tmp_path / "img.png").write_bytes(b"\x89PNG\r\n")
        (tmp_path / "mod.py").write_text("def ok(): pass\n")
        from src.indexer.ast_chunker import ASTChunker
        chunks = ASTChunker().chunk_repository(str(tmp_path))
        assert all(c.file_path.endswith(".py") for c in chunks)

    def test_sliding_window_fallback(self):
        from src.indexer.ast_chunker import ASTChunker
        chunker = ASTChunker()
        long_code = "x = 1\n" * 300
        chunks = chunker._sliding_window(long_code, "file.rb", "ruby")
        assert len(chunks) > 1
        # Ensure no chunk is longer than window
        for c in chunks:
            assert len(c.source) <= 500 * 4 + 50

    def test_chunk_id_is_deterministic(self, tmp_path: Path):
        code = "def fn(): pass\n"
        (tmp_path / "a.py").write_text(code)
        from src.indexer.ast_chunker import ASTChunker
        c1 = ASTChunker().chunk_repository(str(tmp_path))
        c2 = ASTChunker().chunk_repository(str(tmp_path))
        assert [c.chunk_id for c in c1] == [c.chunk_id for c in c2]


# ══════════════════════════════════════════════════════════════════════════════
# HYBRID SEARCH TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestHybridSearch:
    def _make_raw(self, source: str, path: str = "f.py") -> object:
        rc = MagicMock()
        rc.source = source
        rc.file_path = path
        rc.start_line = 1
        rc.end_line = 5
        rc.language = "python"
        rc.chunk_type = "function"
        return rc

    def test_bm25_ranks_exact_match_higher(self):
        from src.indexer.hybrid_search import HybridSearchIndex
        raw = [
            self._make_raw("def calculate(x, y): return x / y"),
            self._make_raw("def greet(name): return 'Hello ' + name"),
            self._make_raw("class Unrelated: pass"),
        ]
        idx = HybridSearchIndex(raw)  # type: ignore[arg-type]
        results = idx.search("calculate divide", top_k=3)
        top_source = results[0][0].source
        assert "calculate" in top_source

    def test_fuse_results_combines_both(self, chunk):
        from src.indexer.hybrid_search import fuse_results
        raw = MagicMock()
        raw.file_path = "other.py"
        raw.start_line = 1
        raw.end_line = 3
        raw.language = "python"
        raw.chunk_type = "function"
        raw.source = "def other(): pass"

        dense   = [chunk]
        bm25    = [(raw, 0.8)]
        fused   = fuse_results(dense, bm25, top_k=5, bm25_weight=0.3)
        paths   = [c.file_path for c in fused]
        assert chunk.file_path in paths
        assert "other.py" in paths


# ══════════════════════════════════════════════════════════════════════════════
# AGENT TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestContextAgent:
    @patch("src.agents.context_agent._pipeline")
    def test_returns_context_payload(self, mock_pipe, issue, chunk):
        mock_pipe.search.return_value = [chunk]
        mock_pipe.scroll_all.return_value = []
        from src.agents.context_agent import ContextAgent
        agent = ContextAgent(pipeline=mock_pipe, bm25_index=None)
        result = agent.run({"issue": issue, "repo_local_path": ""})  # type: ignore[arg-type]
        assert "context" in result
        assert result["context"].confidence > 0

    def test_confidence_high_when_symbols_present(self, issue, chunk):
        from src.agents.context_agent import ContextAgent
        score, found, _ = ContextAgent._compute_confidence(issue, [chunk])
        # "calculate" is in both issue body and chunk source
        assert score > 0.2
        assert "calculate" in found

    def test_query_truncated_to_2000_chars(self, issue):
        from src.agents.context_agent import ContextAgent
        issue.body = "x" * 5000
        q = ContextAgent._build_query(issue)
        assert len(q) <= 2000


class TestDeveloperAgent:
    def _mock_response(self, patches: list[dict], root_cause: str = "Root cause.") -> MagicMock:
        resp = MagicMock()
        resp.usage.input_tokens  = 400
        resp.usage.output_tokens = 200
        resp.content = [MagicMock(
            type="tool_use",
            input={"root_cause": root_cause, "patches": patches},
        )]
        return resp

    @patch("src.agents.developer_agent._client")
    def test_generates_valid_patch(self, mock_client, issue, context):
        mock_client.messages.create.return_value = self._mock_response([{
            "file_path": "myrepo/math_utils.py",
            "original_snippet": "    return x / y",
            "patched_snippet":  "    return x / y if y != 0 else None",
            "explanation": "Guard zero division",
        }])
        from src.agents.developer_agent import DeveloperAgent
        result = DeveloperAgent().run({  # type: ignore[arg-type]
            "issue": issue, "context": context,
            "retry_count": 0, "seen_patch_hashes": [], "cumulative_tokens": 0,
        })
        assert "patch" in result
        assert result["patch"].patch_hash
        assert result["cumulative_tokens"] == 600

    @patch("src.agents.developer_agent._client")
    def test_dedup_guard_clears_context(self, mock_client, issue, context):
        import difflib, hashlib
        orig = "    return x / y"
        new  = "    return x / y if y != 0 else None"
        diff = "\n".join(difflib.unified_diff(
            orig.splitlines(keepends=True), new.splitlines(keepends=True),
            fromfile="a/myrepo/math_utils.py", tofile="b/myrepo/math_utils.py", lineterm="",
        ))
        known_hash = hashlib.sha256(diff.encode()).hexdigest()[:16]

        mock_client.messages.create.return_value = self._mock_response([{
            "file_path": "myrepo/math_utils.py",
            "original_snippet": orig, "patched_snippet": new,
            "explanation": "same patch",
        }])
        from src.agents.developer_agent import DeveloperAgent
        result = DeveloperAgent().run({  # type: ignore[arg-type]
            "issue": issue, "context": context,
            "retry_count": 1, "seen_patch_hashes": [known_hash], "cumulative_tokens": 0,
        })
        assert result.get("context") is None   # cleared by dedup guard

    @patch("src.agents.developer_agent._client")
    def test_invalid_diff_increments_retry(self, mock_client, issue, context):
        mock_client.messages.create.return_value = self._mock_response([{
            "file_path": "x.py",
            "original_snippet": "same",
            "patched_snippet":  "same",   # identical → empty diff → invalid
            "explanation": "no-op",
        }])
        from src.agents.developer_agent import DeveloperAgent
        result = DeveloperAgent().run({  # type: ignore[arg-type]
            "issue": issue, "context": context,
            "retry_count": 0, "seen_patch_hashes": [], "cumulative_tokens": 0,
        })
        assert result.get("patch") is None
        assert result["retry_count"] == 1


class TestTesterAgent:
    @patch("src.agents.tester_agent._client")
    @patch("src.agents.tester_agent.SandboxManager")
    def test_pass_no_analysis_call(self, MockSandbox, mock_client, patch_payload, passing_feedback):
        MockSandbox.return_value.apply_patch_and_test.return_value = passing_feedback.test_result
        from src.agents.tester_agent import TesterAgent
        TesterAgent.__init__ = lambda self: setattr(self, "_sandbox", MockSandbox())  # type: ignore
        agent = TesterAgent()
        result = agent.run({"patch": patch_payload, "repo_local_path": "", "cumulative_tokens": 0})  # type: ignore[arg-type]
        assert result["feedback"].test_result.verdict == TestVerdict.PASS
        mock_client.messages.create.assert_not_called()

    @patch("src.agents.tester_agent._client")
    @patch("src.agents.tester_agent.SandboxManager")
    def test_fail_triggers_analysis(self, MockSandbox, mock_client, patch_payload, failing_feedback):
        MockSandbox.return_value.apply_patch_and_test.return_value = failing_feedback.test_result
        mock_response = MagicMock()
        mock_response.usage.input_tokens  = 200
        mock_response.usage.output_tokens = 80
        mock_response.content = [MagicMock(
            type="tool_use",
            input={
                "error_category": "logic",
                "root_cause_analysis": "y==0 still reaches division.",
                "affected_lines": ["math_utils.py:11"],
                "suggested_fix_hint": "Check y != 0 before dividing.",
            }
        )]
        mock_client.messages.create.return_value = mock_response
        from src.agents.tester_agent import TesterAgent
        TesterAgent.__init__ = lambda self: setattr(self, "_sandbox", MockSandbox())  # type: ignore
        agent = TesterAgent()
        result = agent.run({"patch": patch_payload, "repo_local_path": "", "cumulative_tokens": 0})  # type: ignore[arg-type]
        assert result["feedback"].error_category == ErrorCategory.LOGIC
        assert "zero" in result["feedback"].suggested_fix_hint.lower() or "divid" in result["feedback"].suggested_fix_hint.lower()
        assert result["cumulative_tokens"] == 280


class TestReviewerAgent:
    @patch("src.agents.reviewer_agent._github")
    @patch("src.agents.reviewer_agent._client")
    def test_security_rejection(self, mock_client, mock_github, issue, patch_payload, passing_feedback):
        # Inject a SQL injection pattern into the diff
        patch_payload.unified_diff += '\n+    cursor.execute("SELECT * FROM users WHERE id = %s" % user_id)'
        from src.agents.reviewer_agent import ReviewerAgent
        security_issues = ReviewerAgent._security_scan(patch_payload.unified_diff)
        assert any("SQL injection" in s for s in security_issues)

    def test_style_check_detects_print(self):
        from src.agents.reviewer_agent import ReviewerAgent
        diff = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+print('debug')\n"
        issues = ReviewerAgent._static_style_check(diff)
        assert any("print" in i for i in issues)

    def test_style_check_clean_diff(self):
        from src.agents.reviewer_agent import ReviewerAgent
        diff = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new_value = compute()\n"
        issues = ReviewerAgent._static_style_check(diff)
        assert len(issues) == 0


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR ROUTING TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestOrchestratorRouting:
    def _state(self, **kw) -> dict:
        base = {
            "retry_count": 0,
            "cumulative_tokens": 0,
            "status": PipelineStatus.RUNNING,
            "messages": [],
        }
        base.update(kw)
        return base

    def test_no_context_routes_to_context(self, issue):
        from src.agents.orchestrator import route_from_orchestrator, CONTEXT
        assert route_from_orchestrator(self._state(issue=issue)) == CONTEXT  # type: ignore[arg-type]

    def test_low_confidence_rerequests_context(self, issue):
        from src.agents.orchestrator import route_from_orchestrator, CONTEXT
        low_ctx = ContextPayload(chunks=[], confidence=0.3)
        assert route_from_orchestrator(self._state(issue=issue, context=low_ctx)) == CONTEXT  # type: ignore[arg-type]

    def test_good_context_routes_to_developer(self, issue, context):
        from src.agents.orchestrator import route_from_orchestrator, DEVELOPER
        assert route_from_orchestrator(self._state(issue=issue, context=context)) == DEVELOPER  # type: ignore[arg-type]

    def test_max_retries_routes_to_failed(self, issue, context, patch_payload, failing_feedback):
        from src.agents.orchestrator import route_from_orchestrator, FAILED
        s = self._state(issue=issue, context=context,
                        patch=patch_payload, feedback=failing_feedback, retry_count=5)
        assert route_from_orchestrator(s) == FAILED  # type: ignore[arg-type]

    def test_token_budget_aborts(self, issue):
        from src.agents.orchestrator import orchestrator_node
        result = orchestrator_node(self._state(issue=issue, cumulative_tokens=200_000))  # type: ignore[arg-type]
        assert result["status"] == PipelineStatus.ABORTED

    def test_pass_routes_to_reviewer(self, issue, context, patch_payload, passing_feedback):
        from src.agents.orchestrator import route_from_orchestrator, REVIEWER
        s = self._state(issue=issue, context=context,
                        patch=patch_payload, feedback=passing_feedback)
        assert route_from_orchestrator(s) == REVIEWER  # type: ignore[arg-type]

    def test_after_tester_pass_routes_reviewer(self, passing_feedback):
        from src.agents.orchestrator import route_after_tester, REVIEWER
        assert route_after_tester({"feedback": passing_feedback, "retry_count": 0}) == REVIEWER  # type: ignore[arg-type]

    def test_after_tester_fail_routes_developer(self, failing_feedback):
        from src.agents.orchestrator import route_after_tester, DEVELOPER
        assert route_after_tester({"feedback": failing_feedback, "retry_count": 1}) == DEVELOPER  # type: ignore[arg-type]

    def test_after_tester_max_retries_fails(self, failing_feedback):
        from src.agents.orchestrator import route_after_tester, FAILED
        assert route_after_tester({"feedback": failing_feedback, "retry_count": 5}) == FAILED  # type: ignore[arg-type]


# ══════════════════════════════════════════════════════════════════════════════
# API TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestAPIServer:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from src.api.server import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_jobs_empty(self, client):
        r = client.get("/jobs")
        assert r.status_code == 200
        assert "jobs" in r.json()

    def test_status_404_unknown_job(self, client):
        r = client.get("/status/nonexistent")
        assert r.status_code == 404

    @patch("src.api.server._resolver")
    def test_resolve_returns_job_id(self, mock_resolver, client):
        mock_resolver.resolve.return_value = {
            "status": PipelineStatus.PASSED,
            "retry_count": 1, "cumulative_tokens": 5000,
            "pr": MagicMock(pr_number=99, pr_url="https://github.com/pr/99"),
        }
        r = client.post("/resolve", json={"issue_number": 42})
        assert r.status_code == 202
        data = r.json()
        assert "job_id" in data
        assert data["status"] == "queued"
