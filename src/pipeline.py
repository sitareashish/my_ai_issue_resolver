"""LangGraph pipeline — wires all 5 agents, adds checkpointing, exposes IssueResolver."""
from __future__ import annotations

import sqlite3
import time
from datetime import datetime

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from src.agents.context_agent import context_agent_node
from src.agents.developer_agent import developer_agent_node
from src.agents.orchestrator import (
    CONTEXT, DEVELOPER, FAILED, ORCHESTRATOR, REVIEWER, TESTER, _END,
    failed_node, orchestrator_node,
    route_after_tester, route_from_orchestrator,
)
from src.agents.reviewer_agent import reviewer_agent_node
from src.agents.tester_agent import tester_agent_node
from src.config import settings
from src.github_client import GitHubClient
from src.indexer.ast_chunker import ASTChunker
from src.indexer.embedding_pipeline import EmbeddingPipeline
from src.observability.tracer import (
    bind_run_context, clear_run_context, record_pipeline_result,
)
from src.schema import GitHubIssue, PipelineMetrics, PipelineStatus, SystemState
from src.utils.logger import get_logger

log = get_logger(__name__)


def build_graph() -> StateGraph:
    g = StateGraph(SystemState)
    g.add_node(ORCHESTRATOR, orchestrator_node)
    g.add_node(CONTEXT,      context_agent_node)
    g.add_node(DEVELOPER,    developer_agent_node)
    g.add_node(TESTER,       tester_agent_node)
    g.add_node(REVIEWER,     reviewer_agent_node)
    g.add_node(FAILED,       failed_node)

    g.set_entry_point(ORCHESTRATOR)

    g.add_conditional_edges(ORCHESTRATOR, route_from_orchestrator, {
        CONTEXT: CONTEXT, DEVELOPER: DEVELOPER,
        REVIEWER: REVIEWER, FAILED: FAILED, _END: END,
    })
    g.add_edge(CONTEXT,   ORCHESTRATOR)
    g.add_edge(DEVELOPER, TESTER)
    g.add_conditional_edges(TESTER, route_after_tester, {
        REVIEWER: REVIEWER, DEVELOPER: DEVELOPER, FAILED: FAILED,
    })
    g.add_edge(REVIEWER, END)
    g.add_edge(FAILED,   END)
    return g


def compile_pipeline(checkpoint_db: str | None = None):
    db = checkpoint_db or settings.checkpoint_db
    conn = sqlite3.connect(db, check_same_thread=False)
    return build_graph().compile(checkpointer=SqliteSaver(conn))


# ── High-level runner ─────────────────────────────────────────────────────────

class IssueResolver:
    def __init__(self) -> None:
        self._github   = GitHubClient()
        self._chunker  = ASTChunker()
        self._embedder = EmbeddingPipeline()
        self._pipeline = compile_pipeline()

    def index_repo(self, repo_path: str, drop: bool = False) -> int:
        if drop:
            self._embedder.drop_collection()
        chunks = self._chunker.chunk_repository(repo_path)
        self._embedder.index_chunks(chunks)
        count = self._embedder.count()
        log.info("repo_indexed", chunks=len(chunks), vectors=count, path=repo_path)
        return count

    def resolve(
        self,
        issue_number: int,
        skip_indexing: bool = False,
        repo_path: str | None = None,
    ) -> SystemState:
        bind_run_context(issue_number, settings.github_repo)
        t0 = time.monotonic()

        issue: GitHubIssue = self._github.fetch_issue(issue_number)
        log.info("resolving_issue", number=issue_number, title=issue.title)

        if repo_path is None:
            repo_path = self._github.clone_repo()

        if not skip_indexing:
            self.index_repo(repo_path)

        initial: SystemState = {
            "issue":             issue,
            "repo_local_path":   repo_path,
            "retry_count":       0,
            "seen_patch_hashes": [],
            "cumulative_tokens": 0,
            "status":            PipelineStatus.RUNNING,
            "messages":          [],
            "metrics": PipelineMetrics(
                started_at=datetime.utcnow().isoformat()
            ),
        }  # type: ignore[typeddict-item]

        config = {"configurable": {"thread_id": f"issue-{issue_number}"}}
        log.info("pipeline_starting")
        final: SystemState = self._pipeline.invoke(initial, config=config)

        elapsed  = time.monotonic() - t0
        status   = final.get("status", PipelineStatus.RUNNING)
        retries  = final.get("retry_count", 0)
        tokens   = final.get("cumulative_tokens", 0)
        pr       = final.get("pr")

        # Update metrics
        if "metrics" in final:
            final["metrics"].finished_at        = datetime.utcnow().isoformat()
            final["metrics"].time_to_pr_seconds = round(elapsed, 2)
            final["metrics"].total_tokens       = tokens
            final["metrics"].total_retries      = retries

        record_pipeline_result(
            status=status.value,
            duration_seconds=elapsed,
            retries=retries,
            tokens=tokens,
        )

        log.info("pipeline_complete",
                 status=status.value, elapsed=round(elapsed, 1),
                 retries=retries, tokens=tokens,
                 pr=pr.pr_number if pr else None)

        clear_run_context()
        return final
