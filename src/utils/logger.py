"""Structured logging setup. Call configure_logging() once at startup."""
from __future__ import annotations

import logging
import sys

import structlog
from rich.console import Console
from rich.logging import RichHandler

_console = Console(stderr=True)


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    """
    Set up structlog + stdlib logging.
    - In development: colourised rich output
    - In production (json_output=True): JSON lines for log aggregators
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if json_output:
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
        handler: logging.Handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
    else:
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]
        handler = RichHandler(console=_console, rich_tracebacks=True, markup=True)

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib so third-party libs use same handler
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "anthropic", "urllib3", "docker"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str = __name__) -> structlog.BoundLogger:
    return structlog.get_logger(name)
