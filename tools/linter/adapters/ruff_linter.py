# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "ruff==0.9.4",
# ]
# ///
"""
Ruff formatter adapter for lintrunner.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from enum import Enum
from pathlib import Path
from typing import NamedTuple


IS_WINDOWS: bool = os.name == "nt"


class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


class LintMessage(NamedTuple):
    path: str | None
    line: int | None
    char: int | None
    code: str
    severity: LintSeverity
    name: str
    original: str | None
    replacement: str | None
    description: str | None


def as_posix(name: str) -> str:
    return name.replace("\\", "/") if IS_WINDOWS else name


def run_command(
    args: list[str],
    stdin_input: str | None = None,
) -> subprocess.CompletedProcess[str]:
    logging.debug("$ %s", " ".join(args))
    start_time = time.monotonic()
    try:
        return subprocess.run(
            args,
            input=stdin_input,
            capture_output=True,
            check=False,
            encoding="utf-8",
        )
    finally:
        end_time = time.monotonic()
        logging.debug("took %dms", (end_time - start_time) * 1000)


def check_file(filename: str) -> list[LintMessage]:
    try:
        with open(filename, encoding="utf-8") as f:
            original = f.read()
    except Exception as err:
        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code="RUFF",
                severity=LintSeverity.ERROR,
                name="read-failed",
                original=None,
                replacement=None,
                description=f"Failed to read file: {err}",
            )
        ]

    try:
        proc = run_command(
            [sys.executable, "-mruff", "format", "--stdin-filename", filename, "-"],
            stdin_input=original,
        )
    except OSError as err:
        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code="RUFF",
                severity=LintSeverity.ERROR,
                name="command-failed",
                original=None,
                replacement=None,
                description=f"Failed to run ruff format: {err}",
            )
        ]

    if proc.returncode != 0:
        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code="RUFF",
                severity=LintSeverity.ERROR,
                name="format-error",
                original=None,
                replacement=None,
                description=f"Ruff failed with exit code {proc.returncode}: {proc.stderr.strip()}",
            )
        ]

    replacement = proc.stdout

    if original == replacement:
        return []

    return [
        LintMessage(
            path=filename,
            line=1,
            char=1,
            code="RUFF",
            severity=LintSeverity.WARNING,
            name="format",
            original=original,
            replacement=replacement,
            description="Run `lintrunner -a` to apply formatting changes.",
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ruff formatter wrapper for lintrunner.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose logging",
    )
    parser.add_argument(
        "filenames",
        nargs="+",
        help="paths to lint",
    )
    args = parser.parse_args()

    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.NOTSET if args.verbose else logging.DEBUG,
        stream=sys.stderr,
    )

    lint_messages: list[LintMessage] = []
    for filename in args.filenames:
        lint_messages.extend(check_file(filename))

    for lint_message in lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)


if __name__ == "__main__":
    main()
