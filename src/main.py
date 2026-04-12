"""CLI entry point: python -m src.main --issue 42"""
from __future__ import annotations

import argparse
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.utils.logger import configure_logging

console = Console()


def _banner() -> None:
    console.print(Panel.fit(
        "[bold purple]GitHub Issue Resolver[/bold purple]  v1.0.0\n"
        "[dim]Autonomous multi-agent pipeline · Issue → Verified PR[/dim]",
        border_style="dim purple",
    ))


def _summary(state: dict, elapsed: float) -> None:
    from src.schema import PipelineStatus
    status  = state.get("status", PipelineStatus.RUNNING)
    pr      = state.get("pr")
    metrics = state.get("metrics")
    retries = state.get("retry_count", 0)
    tokens  = state.get("cumulative_tokens", 0)

    color = {
        PipelineStatus.PASSED:  "green",
        PipelineStatus.FAILED:  "red",
        PipelineStatus.ABORTED: "yellow",
        PipelineStatus.RUNNING: "blue",
    }.get(status, "white")

    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column("k", style="dim")
    t.add_column("v", style="bold")
    t.add_row("Status",  f"[{color}]{status.value.upper()}[/{color}]")
    t.add_row("Retries", str(retries))
    t.add_row("Tokens",  f"{tokens:,}")
    t.add_row("Time",    f"{elapsed:.1f}s")
    if pr:
        t.add_row("PR",    f"#{pr.pr_number}")
        t.add_row("URL",   pr.pr_url)
    if state.get("error_message"):
        t.add_row("Error", f"[red]{state['error_message'][:80]}[/red]")
    if metrics and metrics.search_strategy_used:
        t.add_row("Search", metrics.search_strategy_used)

    console.print(Panel(t, title="[bold]Pipeline summary[/bold]", border_style=color))


def main() -> None:
    p = argparse.ArgumentParser(description="Autonomous GitHub Issue Resolver")
    p.add_argument("--issue",          type=int, required=True)
    p.add_argument("--skip-indexing",  action="store_true")
    p.add_argument("--repo-path",      type=str, default=None)
    p.add_argument("--repo",           type=str, default=None,
                   help="Override GITHUB_REPO env var (owner/repo)")
    p.add_argument("--drop-index",     action="store_true",
                   help="Drop and rebuild Qdrant collection before running")
    p.add_argument("--json-logs",      action="store_true",
                   help="Emit JSON log lines (for log aggregators)")
    p.add_argument("--verbose", "-v",  action="store_true")
    args = p.parse_args()

    configure_logging(
        level="DEBUG" if args.verbose else "INFO",
        json_output=args.json_logs,
    )

    if args.repo:
        import os
        os.environ["GITHUB_REPO"] = args.repo

    _banner()

    from src.pipeline import IssueResolver
    resolver = IssueResolver()

    if args.drop_index and args.repo_path:
        console.print("[yellow]Dropping and rebuilding index…[/yellow]")
        resolver.index_repo(args.repo_path, drop=True)
        console.print("[green]Index rebuilt.[/green]")

    t0 = time.monotonic()
    try:
        final = resolver.resolve(
            issue_number=args.issue,
            skip_indexing=args.skip_indexing,
            repo_path=args.repo_path,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(1)
    except Exception:
        console.print_exception()
        sys.exit(1)

    _summary(final, time.monotonic() - t0)  # type: ignore[arg-type]

    from src.schema import PipelineStatus
    sys.exit(0 if final.get("status") == PipelineStatus.PASSED else 1)


if __name__ == "__main__":
    main()
