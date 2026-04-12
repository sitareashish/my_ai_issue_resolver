"""Orchestrator — LangGraph FSM routing decisions."""
from __future__ import annotations

from src.config import settings
from src.schema import PipelineStatus, SystemState, TestVerdict
from src.utils.logger import get_logger

log = get_logger(__name__)

ORCHESTRATOR = "orchestrator"
CONTEXT      = "context_agent"
DEVELOPER    = "developer_agent"
TESTER       = "tester_agent"
REVIEWER     = "reviewer_agent"
FAILED       = "failed"
_END         = "__end__"


def orchestrator_node(state: SystemState) -> SystemState:
    retry  = state.get("retry_count", 0)
    tokens = state.get("cumulative_tokens", 0)

    if tokens >= settings.max_tokens_per_issue:
        log.warning("budget_exceeded", tokens=tokens, limit=settings.max_tokens_per_issue)
        return {**state, "status": PipelineStatus.ABORTED}  # type: ignore[return-value]

    if tokens >= settings.token_warn_threshold:
        log.warning("budget_warning", tokens=tokens, warn_limit=settings.token_warn_threshold)

    if retry >= settings.max_retries:
        log.error("max_retries_reached", retries=retry, max=settings.max_retries)
        return {**state, "status": PipelineStatus.FAILED}  # type: ignore[return-value]

    log.info("orchestrator_tick", retry=retry, tokens=tokens,
             status=state.get("status", PipelineStatus.RUNNING).value)
    return state  # type: ignore[return-value]


def route_from_orchestrator(state: SystemState) -> str:
    status = state.get("status", PipelineStatus.RUNNING)
    if status in (PipelineStatus.ABORTED, PipelineStatus.FAILED):
        return FAILED

    context  = state.get("context")
    patch    = state.get("patch")
    feedback = state.get("feedback")
    review   = state.get("review")

    if context is None:
        return CONTEXT
    if context.confidence < settings.confidence_threshold and patch is None:
        log.info("low_confidence_requery", confidence=context.confidence)
        return CONTEXT
    if patch is None:
        return DEVELOPER
    if feedback is None:
        return TESTER
    if feedback.test_result.verdict != TestVerdict.PASS:
        return DEVELOPER
    if review is None:
        return REVIEWER
    return _END


def route_after_tester(state: SystemState) -> str:
    feedback = state.get("feedback")
    retry    = state.get("retry_count", 0)

    if feedback is None:
        return TESTER

    if feedback.test_result.verdict == TestVerdict.PASS:
        log.info("tests_passed_routing_to_reviewer")
        return REVIEWER

    if retry >= settings.max_retries:
        log.error("max_retries_after_tester", retries=retry)
        return FAILED

    log.info("test_failed_retrying",
             category=feedback.error_category.value,
             attempt=retry, max=settings.max_retries)
    return DEVELOPER


def failed_node(state: SystemState) -> SystemState:
    status = state.get("status", PipelineStatus.FAILED)
    log.error("pipeline_terminal",
              status=status.value,
              retries=state.get("retry_count", 0),
              tokens=state.get("cumulative_tokens", 0),
              error=state.get("error_message", ""))
    return {**state, "status": status}  # type: ignore[return-value]
