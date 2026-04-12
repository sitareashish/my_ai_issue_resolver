"""
Rate-limit-aware API call wrapper with exponential backoff + full jitter.

Usage:
    result = call_with_backoff(lambda: client.messages.create(...))
"""
from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

import anthropic
import openai

from src.utils.logger import get_logger

log = get_logger(__name__)

T = TypeVar("T")

_RETRYABLE_ANTHROPIC = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
)
_RETRYABLE_OPENAI = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


def call_with_backoff(
    fn: Callable[[], T],
    max_attempts: int = 6,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
) -> T:
    """
    Call fn() with full-jitter exponential backoff on rate limit / transient errors.
    Raises on non-retryable errors immediately.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except (*_RETRYABLE_ANTHROPIC, *_RETRYABLE_OPENAI) as exc:  # type: ignore[misc]
            if attempt == max_attempts:
                log.error("api_call_failed_permanently", attempts=attempt, error=str(exc))
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            if jitter:
                delay = random.uniform(0, delay)
            log.warning("api_rate_limited",
                        attempt=attempt, retry_in=round(delay, 2), error=str(exc))
            time.sleep(delay)

    raise RuntimeError("call_with_backoff: unreachable")  # pragma: no cover
