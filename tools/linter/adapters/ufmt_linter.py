# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "ufmt==2.8.0",
#   "usort==1.1.0",
#   "ruff-api==0.2.1",
# ]
# ///
#
# ufmt adapter for lintrunner - combines usort (import sorting) + ruff (formatting)
# to match internal fbcode pyfmt behavior.
#
"""
ufmt formatter adapter for lintrunner.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from enum import Enum
from pathlib import Path
from typing import NamedTuple


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


def check_file(filename: str) -> list[LintMessage]:
    from ufmt import Result, ufmt_file

    path = Path(filename).resolve()

    try:
        original = path.read_text(encoding="utf-8")
    except Exception as err:
        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code="UFMT",
                severity=LintSeverity.ERROR,
                name="read-failed",
                original=None,
                replacement=None,
                description=f"Failed to read file: {err}",
            )
        ]

    try:
        # Run ufmt (ruff isort + ruff format) with dry_run to not modify file
        # return_content=True to get formatted content
        result: Result = ufmt_file(
            path,
            dry_run=True,
            return_content=True,
        )
        if result.error:
            raise result.error
        replacement = result.after.decode("utf-8") if result.after else original
    except Exception as err:
        return [
            LintMessage(
                path=filename,
                line=None,
                char=None,
                code="UFMT",
                severity=LintSeverity.ERROR,
                name="format-error",
                original=None,
                replacement=None,
                description=f"ufmt failed: {err}",
            )
        ]

    if original == replacement:
        return []

    return [
        LintMessage(
            path=filename,
            line=1,
            char=1,
            code="UFMT",
            severity=LintSeverity.WARNING,
            name="format",
            original=original,
            replacement=replacement,
            description="Run `lintrunner -a` to apply formatting changes.",
        )
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ufmt (usort + formatter) wrapper for lintrunner.",
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
        level=logging.DEBUG if args.verbose else logging.WARNING,
        stream=sys.stderr,
    )

    lint_messages: list[LintMessage] = []
    for filename in args.filenames:
        lint_messages.extend(check_file(filename))

    for lint_message in lint_messages:
        print(json.dumps(lint_message._asdict()), flush=True)


if __name__ == "__main__":
    main()
