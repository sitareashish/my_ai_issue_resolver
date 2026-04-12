"""Token counting and budget enforcement utilities."""
from __future__ import annotations

from dataclasses import dataclass, field

import tiktoken

from src.utils.logger import get_logger

log = get_logger(__name__)

_ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Approximate token count using cl100k_base (Claude/GPT-4 compatible)."""
    return len(_ENC.encode(text, disallowed_special=()))


def count_messages_tokens(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total += count_tokens(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    total += count_tokens(block["text"])
    return total


@dataclass
class TokenBudget:
    """
    Tracks cumulative token usage across an entire pipeline run.
    Raises BudgetExceededError when the hard limit is hit.
    """
    hard_limit: int
    warn_limit: int
    _used: int = field(default=0, init=False)
    _warned: bool = field(default=False, init=False)

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self._used += input_tokens + output_tokens
        if not self._warned and self._used >= self.warn_limit:
            log.warning("token_budget_warning",
                        used=self._used, warn_limit=self.warn_limit, hard_limit=self.hard_limit)
            self._warned = True
        if self._used >= self.hard_limit:
            raise BudgetExceededError(self._used, self.hard_limit)

    @property
    def used(self) -> int:
        return self._used

    @property
    def remaining(self) -> int:
        return max(0, self.hard_limit - self._used)

    def to_dict(self) -> dict:
        return {
            "used": self._used,
            "hard_limit": self.hard_limit,
            "remaining": self.remaining,
            "pct_used": round(self._used / self.hard_limit * 100, 1),
        }


class BudgetExceededError(Exception):
    def __init__(self, used: int, limit: int) -> None:
        super().__init__(f"Token budget exceeded: {used:,} / {limit:,}")
        self.used = used
        self.limit = limit
