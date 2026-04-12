"""
Observability layer:
  - LangSmith automatic tracing (via LANGCHAIN_TRACING_V2 env var)
  - Prometheus metrics exposed at /metrics
  - Structured span context for each pipeline run
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Generator

import structlog

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    _prometheus_available = True
except ImportError:
    _prometheus_available = False

log = structlog.get_logger(__name__)

# ── Prometheus metrics ─────────────────────────────────────────────────────────

if _prometheus_available:
    PIPELINE_RUNS = Counter(
        "resolver_pipeline_runs_total",
        "Total pipeline runs",
        ["status"],
    )
    PIPELINE_DURATION = Histogram(
        "resolver_pipeline_duration_seconds",
        "Pipeline wall-clock time",
        buckets=[30, 60, 120, 180, 300, 600],
    )
    RETRIES_HIST = Histogram(
        "resolver_retries_per_issue",
        "Number of Developer-Tester retries per resolved issue",
        buckets=[0, 1, 2, 3, 4, 5],
    )
    TOKEN_USAGE = Histogram(
        "resolver_tokens_per_issue",
        "Total tokens consumed per issue",
        buckets=[5_000, 15_000, 30_000, 60_000, 100_000, 150_000],
    )
    ACTIVE_SANDBOXES = Gauge(
        "resolver_active_sandboxes",
        "Currently running Docker sandbox containers",
    )


def start_metrics_server(port: int = 9090) -> None:
    if _prometheus_available:
        start_http_server(port)
        log.info("prometheus_metrics_server_started", port=port)
    else:
        log.warning("prometheus_not_installed_metrics_disabled")


def record_pipeline_result(
    status: str,
    duration_seconds: float,
    retries: int,
    tokens: int,
) -> None:
    if not _prometheus_available:
        return
    PIPELINE_RUNS.labels(status=status).inc()
    PIPELINE_DURATION.observe(duration_seconds)
    RETRIES_HIST.observe(retries)
    TOKEN_USAGE.observe(tokens)


@contextmanager
def sandbox_active() -> Generator[None, None, None]:
    if _prometheus_available:
        ACTIVE_SANDBOXES.inc()
    try:
        yield
    finally:
        if _prometheus_available:
            ACTIVE_SANDBOXES.dec()


# ── LangSmith context binding ──────────────────────────────────────────────────

def bind_run_context(issue_number: int, repo: str) -> None:
    """Bind structured context vars so every log line carries issue metadata."""
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        issue_number=issue_number,
        repo=repo,
        langsmith_project=os.environ.get("LANGCHAIN_PROJECT", "resolver"),
    )


def clear_run_context() -> None:
    structlog.contextvars.clear_contextvars()
