"""
FastAPI REST server — exposes the pipeline over HTTP.

Endpoints:
  POST /resolve          — trigger a pipeline run (async, returns job_id)
  GET  /status/{job_id}  — poll job status + final result
  GET  /jobs             — list recent jobs
  GET  /health           — liveness probe
  GET  /metrics          — Prometheus metrics (text/plain)
  WS   /ws/{job_id}      — WebSocket live log stream
"""
from __future__ import annotations

import asyncio
import threading
import uuid
from datetime import datetime
from typing import Any

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.config import settings
from src.observability.tracer import start_metrics_server
from src.pipeline import IssueResolver
from src.schema import PipelineStatus
from src.utils.logger import configure_logging, get_logger

configure_logging()
log = get_logger(__name__)

app = FastAPI(
    title="GitHub Issue Resolver API",
    version="1.0.0",
    description="Autonomous multi-agent pipeline: Issue → Verified PR",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ── In-memory job store ────────────────────────────────────────────────────────

_jobs: dict[str, dict[str, Any]] = {}
_log_queues: dict[str, asyncio.Queue] = {}
_resolver = IssueResolver()


# ── Request/response models ────────────────────────────────────────────────────

class ResolveRequest(BaseModel):
    issue_number: int
    skip_indexing: bool = False
    repo_path: str | None = None


class ResolveResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    issue_number: int
    started_at: str
    finished_at: str | None
    pr_number: int | None
    pr_url: str | None
    retry_count: int
    tokens_used: int
    time_seconds: float | None
    error: str | None


# ── Background job runner ──────────────────────────────────────────────────────

def _run_job(job_id: str, req: ResolveRequest) -> None:
    job = _jobs[job_id]
    job["status"] = PipelineStatus.RUNNING.value

    try:
        final = _resolver.resolve(
            issue_number=req.issue_number,
            skip_indexing=req.skip_indexing,
            repo_path=req.repo_path,
        )
        status  = final.get("status", PipelineStatus.RUNNING)
        pr      = final.get("pr")
        metrics = final.get("metrics")

        job.update({
            "status":       status.value,
            "pr_number":    pr.pr_number if pr else None,
            "pr_url":       pr.pr_url    if pr else None,
            "retry_count":  final.get("retry_count", 0),
            "tokens_used":  final.get("cumulative_tokens", 0),
            "finished_at":  datetime.utcnow().isoformat(),
            "time_seconds": metrics.time_to_pr_seconds if metrics else None,
            "error":        final.get("error_message"),
        })

    except Exception as exc:
        log.error("job_crashed", job_id=job_id, error=str(exc))
        job.update({
            "status":      PipelineStatus.FAILED.value,
            "finished_at": datetime.utcnow().isoformat(),
            "error":       str(exc),
        })


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.post("/resolve", response_model=ResolveResponse, status_code=202)
async def resolve(req: ResolveRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {
        "job_id":       job_id,
        "status":       "queued",
        "issue_number": req.issue_number,
        "started_at":   datetime.utcnow().isoformat(),
        "finished_at":  None,
        "pr_number":    None,
        "pr_url":       None,
        "retry_count":  0,
        "tokens_used":  0,
        "time_seconds": None,
        "error":        None,
    }
    log.info("job_queued", job_id=job_id, issue=req.issue_number)

    # Run in thread so FastAPI stays responsive
    t = threading.Thread(target=_run_job, args=(job_id, req), daemon=True)
    t.start()

    return ResolveResponse(
        job_id=job_id,
        status="queued",
        message=f"Pipeline started. Poll GET /status/{job_id} for updates.",
    )


@app.get("/status/{job_id}", response_model=JobStatus)
async def status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return JobStatus(**job)


@app.get("/jobs")
async def list_jobs(limit: int = 20):
    recent = sorted(_jobs.values(), key=lambda j: j["started_at"], reverse=True)[:limit]
    return {"jobs": recent, "total": len(_jobs)}


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0", "timestamp": datetime.utcnow().isoformat()}


@app.get("/metrics")
async def metrics():
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        from fastapi.responses import Response
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        return {"error": "prometheus_client not installed"}


@app.websocket("/ws/{job_id}")
async def ws_logs(websocket: WebSocket, job_id: str):
    """Stream live log events for a running job via WebSocket."""
    await websocket.accept()
    if job_id not in _jobs:
        await websocket.send_json({"error": "job not found"})
        await websocket.close()
        return

    try:
        while True:
            job = _jobs.get(job_id, {})
            await websocket.send_json({
                "job_id":  job_id,
                "status":  job.get("status"),
                "tokens":  job.get("tokens_used", 0),
                "retries": job.get("retry_count", 0),
            })
            if job.get("status") not in ("queued", PipelineStatus.RUNNING.value):
                break
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        pass


# ── Entry point ────────────────────────────────────────────────────────────────

def run() -> None:
    if settings.metrics_enabled:
        start_metrics_server(port=9090)
    uvicorn.run(
        "src.api.server:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        log_level="info",
    )
