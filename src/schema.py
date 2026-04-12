"""All shared data models. Every agent reads/writes SystemState."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Annotated, Any

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class PipelineStatus(str, Enum):
    RUNNING  = "running"
    PASSED   = "passed"
    FAILED   = "failed"
    ABORTED  = "aborted"


class ErrorCategory(str, Enum):
    SYNTAX           = "syntax"
    LOGIC            = "logic"
    MISSING_IMPORT   = "missing_import"
    TEST_EXPECTATION = "test_expectation"
    UNKNOWN          = "unknown"


class TestVerdict(str, Enum):
    PASS  = "pass"
    FAIL  = "fail"
    ERROR = "error"


class ChunkType(str, Enum):
    FUNCTION = "function"
    CLASS    = "class"
    MODULE   = "module"
    METHOD   = "method"


# ── Core domain objects ───────────────────────────────────────────────────────

@dataclass
class GitHubIssue:
    number: int
    title: str
    body: str
    labels: list[str]
    comments: list[str]
    repo_full_name: str
    html_url: str
    created_at: str = ""
    linked_prs: list[int] = field(default_factory=list)


@dataclass
class CodeChunk:
    file_path: str
    start_line: int
    end_line: int
    language: str
    chunk_type: str
    source: str
    relevance_score: float = 0.0
    bm25_score: float = 0.0
    hybrid_score: float = 0.0


@dataclass
class ContextPayload:
    chunks: list[CodeChunk]
    confidence: float
    symbols_found: list[str] = field(default_factory=list)
    symbols_missing: list[str] = field(default_factory=list)
    search_strategy: str = "hybrid"   # "hybrid" | "dense" | "bm25" | "grep"


@dataclass
class FilePatch:
    file_path: str
    original_snippet: str
    patched_snippet: str
    explanation: str
    is_new_file: bool = False


@dataclass
class PatchPayload:
    patches: list[FilePatch]
    unified_diff: str
    patch_hash: str
    attempt: int = 1
    overall_explanation: str = ""


@dataclass
class TestResult:
    verdict: TestVerdict
    failed_tests: list[str]
    stdout: str
    stderr: str
    duration_ms: int
    exit_code: int
    test_runner: str = "unknown"


@dataclass
class FeedbackPayload:
    error_category: ErrorCategory
    affected_lines: list[str]
    suggested_fix_hint: str
    test_result: TestResult
    root_cause_analysis: str = ""


@dataclass
class ReviewResult:
    approved: bool
    comments: list[str]
    pr_body: str
    suggested_labels: list[str]
    pr_quality_score: float = 0.0    # 0-5


@dataclass
class PRResult:
    pr_number: int
    pr_url: str
    branch_name: str
    commit_sha: str = ""


@dataclass
class PipelineMetrics:
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    finished_at: str = ""
    total_tokens: int = 0
    total_retries: int = 0
    context_calls: int = 0
    developer_calls: int = 0
    tester_calls: int = 0
    reviewer_calls: int = 0
    time_to_pr_seconds: float = 0.0
    chunks_retrieved: int = 0
    search_strategy_used: str = ""


# ── LangGraph SystemState ─────────────────────────────────────────────────────

class SystemState(TypedDict, total=False):
    issue:             GitHubIssue
    repo_local_path:   str
    context:           ContextPayload
    patch:             PatchPayload
    feedback:          FeedbackPayload
    review:            ReviewResult
    pr:                PRResult
    metrics:           PipelineMetrics

    retry_count:        int
    seen_patch_hashes:  list[str]
    cumulative_tokens:  int
    status:             PipelineStatus
    error_message:      str

    # LangGraph append-only message history
    messages: Annotated[list[Any], add_messages]
