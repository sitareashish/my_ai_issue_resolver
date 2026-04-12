"""Benchmark evaluation suite — runs pipeline on N issues and writes evaluation_report.json."""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def run_evaluation(issues_file: str, skip_indexing: bool = False, verbose: bool = False) -> dict:
    from src.pipeline import IssueResolver
    from src.schema import PipelineStatus

    with open(issues_file) as f:
        benchmark: list[dict] = json.load(f)

    resolver = IssueResolver()
    results: list[dict] = []

    for entry in benchmark:
        num        = entry["issue_number"]
        difficulty = entry.get("difficulty", "medium")
        expected   = entry.get("expected_changed_files", [])

        console.print(f"\n[bold cyan]Issue #{num}[/bold cyan] [{difficulty}]")
        t0 = time.monotonic()

        try:
            state    = resolver.resolve(num, skip_indexing=skip_indexing)
            elapsed  = time.monotonic() - t0
            status   = state.get("status", PipelineStatus.RUNNING)
            pr       = state.get("pr")
            patch    = state.get("patch")
            feedback = state.get("feedback")
            metrics  = state.get("metrics")

            resolved     = pr is not None
            first_attempt = state.get("retry_count", 0) <= 1 and resolved
            files_touched = [p.file_path for p in patch.patches] if patch else []
            correct_files = all(f in files_touched for f in expected)

            record = {
                "issue_number":        num,
                "difficulty":          difficulty,
                "status":              status.value,
                "resolved":            resolved,
                "first_attempt":       first_attempt,
                "retry_count":         state.get("retry_count", 0),
                "tokens":              state.get("cumulative_tokens", 0),
                "time_to_pr_seconds":  round(elapsed, 1),
                "pr_number":           pr.pr_number if pr else None,
                "pr_url":              pr.pr_url    if pr else None,
                "pr_quality_score":    state.get("review", {}).pr_quality_score if state.get("review") else None,
                "files_touched":       files_touched,
                "correct_files":       correct_files,
                "test_passed":         feedback.test_result.exit_code == 0 if feedback else False,
                "search_strategy":     state.get("context", {}).search_strategy if state.get("context") else None,
                "error":               state.get("error_message"),
            }
        except Exception as exc:
            elapsed = time.monotonic() - t0
            logging.exception(f"Issue #{num} crashed")
            record = {
                "issue_number": num, "difficulty": difficulty,
                "status": "crashed", "resolved": False, "first_attempt": False,
                "retry_count": 0, "tokens": 0,
                "time_to_pr_seconds": round(elapsed, 1),
                "pr_number": None, "pr_url": None, "pr_quality_score": None,
                "files_touched": [], "correct_files": False, "test_passed": False,
                "search_strategy": None, "error": str(exc),
            }

        results.append(record)
        _print_record(record)
        _save(results)

    summary = _summarise(results)
    _print_summary(summary)
    _save(results, summary)
    return {"results": results, "summary": summary}


def _summarise(results: list[dict]) -> dict:
    n = len(results)
    if not n:
        return {}
    resolved      = [r for r in results if r["resolved"]]
    first         = [r for r in resolved if r["first_attempt"]]
    tokens_list   = [r["tokens"]              for r in resolved] or [0]
    retries_list  = [r["retry_count"]         for r in resolved] or [0]
    time_list     = [r["time_to_pr_seconds"]  for r in resolved] or [0]
    quality_list  = [r["pr_quality_score"] for r in resolved if r["pr_quality_score"]] or [0]

    avg = lambda lst: round(sum(lst) / len(lst), 2)

    return {
        "total":                n,
        "resolved":             len(resolved),
        "resolution_rate_pct":  round(len(resolved) / n * 100, 1),
        "first_attempt_count":  len(first),
        "first_attempt_pct":    round(len(first) / n * 100, 1),
        "avg_retries":          avg(retries_list),
        "avg_tokens":           int(avg(tokens_list)),
        "avg_time_seconds":     avg(time_list),
        "avg_quality_score":    avg(quality_list),
        "by_difficulty": {
            diff: {
                "total":    len(grp := [r for r in results if r["difficulty"] == diff]),
                "resolved": len([r for r in grp if r["resolved"]]),
            }
            for diff in ("easy", "medium", "hard")
        },
        "targets": {
            "resolution_≥70%":     len(resolved) / n >= 0.70,
            "first_attempt_≥40%":  len(first)    / n >= 0.40,
            "avg_retries_≤2.5":    avg(retries_list) <= 2.5,
            "avg_tokens_≤60k":     avg(tokens_list)  <= 60_000,
            "avg_time_≤300s":      avg(time_list)    <= 300,
        },
    }


def _print_record(r: dict) -> None:
    color = "green" if r["resolved"] else "red"
    console.print(
        f"  [{color}]{r['status']}[/{color}]  "
        f"retries={r['retry_count']}  tokens={r['tokens']:,}  "
        f"time={r['time_to_pr_seconds']}s  "
        f"files={r['files_touched']}"
    )


def _print_summary(s: dict) -> None:
    console.print("\n[bold]Evaluation summary[/bold]")
    t = Table(show_header=True, box=None, padding=(0, 2))
    t.add_column("Metric",   style="dim", width=32)
    t.add_column("Result",   style="bold")
    t.add_column("Target",   style="dim")
    t.add_column("",         width=3)

    def row(label, val, target, key):
        ok = s.get("targets", {}).get(key, False)
        t.add_row(label, str(val), target, "[green]✓[/green]" if ok else "[red]✗[/red]")

    row("Resolution rate",     f"{s.get('resolution_rate_pct', 0)}%",  "≥ 70%",     "resolution_≥70%")
    row("First-attempt rate",  f"{s.get('first_attempt_pct', 0)}%",   "≥ 40%",     "first_attempt_≥40%")
    row("Avg retries",         s.get("avg_retries", 0),                "≤ 2.5",     "avg_retries_≤2.5")
    row("Avg tokens / issue",  f"{s.get('avg_tokens', 0):,}",          "≤ 60,000",  "avg_tokens_≤60k")
    row("Avg time to PR",      f"{s.get('avg_time_seconds', 0)}s",     "≤ 300s",    "avg_time_≤300s")
    console.print(t)

    console.print("\n[bold]By difficulty[/bold]")
    for diff, data in s.get("by_difficulty", {}).items():
        pct = round(data["resolved"] / max(data["total"], 1) * 100, 1)
        console.print(f"  {diff:6s}  {data['resolved']}/{data['total']}  ({pct}%)")


def _save(results: list[dict], summary: dict | None = None) -> None:
    out = {"results": results}
    if summary:
        out["summary"] = summary
    Path("evaluation_report.json").write_text(json.dumps(out, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--issues-file",   default="scripts/benchmark_issues.json")
    p.add_argument("--skip-indexing", action="store_true")
    p.add_argument("--verbose",       action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    run_evaluation(args.issues_file, args.skip_indexing, args.verbose)


if __name__ == "__main__":
    main()
