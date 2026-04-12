"""
Unified diff validation utilities.

Catches bad diffs BEFORE they hit the Docker sandbox, saving ~30s per failure.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.utils.logger import get_logger

log = get_logger(__name__)

_DIFF_HEADER = re.compile(r"^--- .+\n\+\+\+ .+", re.MULTILINE)
_HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", re.MULTILINE)

# Style issues we flag (but don't reject)
_DEBUG_PRINT  = re.compile(r"^\+.*print\s*\(", re.MULTILINE)
_COMMENTED_OUT = re.compile(r"^\+\s*#.*\S", re.MULTILINE)
_TODO_FIXME   = re.compile(r"^\+.*\b(TODO|FIXME|HACK|XXX)\b", re.MULTILINE)


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]
    warnings: list[str]
    files_changed: list[str]
    lines_added: int
    lines_removed: int


def validate_diff(unified_diff: str) -> ValidationResult:
    """
    Validates a unified diff string.
    Returns a ValidationResult with errors (blocking) and warnings (informational).
    """
    errors: list[str] = []
    warnings: list[str] = []
    files_changed: list[str] = []
    lines_added = 0
    lines_removed = 0

    if not unified_diff or not unified_diff.strip():
        errors.append("Diff is empty")
        return ValidationResult(False, errors, warnings, [], 0, 0)

    # Must have at least one file header
    if not _DIFF_HEADER.search(unified_diff):
        errors.append("No valid diff header found (--- / +++ lines missing)")

    # Must have at least one hunk
    if not _HUNK_HEADER.search(unified_diff):
        errors.append("No hunk header found (@@ lines missing)")

    # Parse files and line counts
    for line in unified_diff.splitlines():
        if line.startswith("+++ b/"):
            fname = line[6:].strip()
            if fname and fname != "/dev/null":
                files_changed.append(fname)
        elif line.startswith("+") and not line.startswith("+++"):
            lines_added += 1
        elif line.startswith("-") and not line.startswith("---"):
            lines_removed += 1

    if not files_changed:
        errors.append("Could not identify any changed files from diff")

    # Style warnings
    if _DEBUG_PRINT.search(unified_diff):
        warnings.append("Diff adds print() statements — remove before merging")
    if _TODO_FIXME.search(unified_diff):
        warnings.append("Diff adds TODO/FIXME comments")

    # Sanity: no-op diff
    if lines_added == 0 and lines_removed == 0:
        errors.append("Diff makes no actual changes (no +/- lines)")

    valid = len(errors) == 0
    if not valid:
        log.warning("diff_validation_failed", errors=errors)
    elif warnings:
        log.warning("diff_validation_warnings", warnings=warnings)

    return ValidationResult(
        valid=valid,
        errors=errors,
        warnings=warnings,
        files_changed=files_changed,
        lines_added=lines_added,
        lines_removed=lines_removed,
    )


def extract_file_paths_from_diff(unified_diff: str) -> list[str]:
    """Quick helper — returns list of relative file paths touched by the diff."""
    paths: list[str] = []
    for line in unified_diff.splitlines():
        if line.startswith("+++ b/"):
            p = line[6:].strip()
            if p and p != "/dev/null":
                paths.append(p)
    return paths
