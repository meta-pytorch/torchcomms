# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Retire the preceding unpublished automation diff after its replacement exists."""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
from pathlib import Path

logger: logging.Logger = logging.getLogger(__name__)


def read_diff_command(action: str, *args: str) -> object:
    annotation = json.loads(
        subprocess.check_output(
            [
                "meta",
                "metacli.command",
                "annotations",
                "--command",
                f"phabricator.diff {action}",
                "-o",
                "json",
            ],
            text=True,
        )
    )
    if annotation["control"] != "AUTONOMOUS":
        raise ValueError(f"Diff {action} requires interactive authorization")
    return json.loads(
        subprocess.check_output(
            [
                "meta",
                "phabricator.diff",
                action,
                *args,
                "-o",
                "json",
            ],
            text=True,
        )
    )


def can_retire(diff: dict, service_user: str, marker: str) -> bool:
    return (
        diff.get("status") == "Unpublished"
        and str(diff.get("author_fbid")) == service_user
        and diff.get("published_epoch") is None
        and marker in diff.get("summary", "")
    )


def retire(new_diff: str, service_user: str, marker: str) -> None:
    if not re.fullmatch(r"D[0-9]+", new_diff) or not service_user.isdecimal():
        raise ValueError("Expected a submitted D-number and numeric bot identity")
    candidates = read_diff_command(
        "list",
        f"--author-is={service_user}",
        "--status-is=UNPUBLISHED",
        f"--summary-has-the-phrase={marker}",
        f"--diff-number-less-than={new_diff[1:]}",
        "--sort-by=number",
        "--sort-direction=desc",
        "--limit=1",
        "--columns=number",
    )
    if not isinstance(candidates, list) or not candidates:
        return
    previous = candidates[0]["number"]
    if not re.fullmatch(r"D[0-9]+", previous):
        raise ValueError("Invalid predecessor diff number")
    diff = read_diff_command("describe", f"--number={previous}")
    if not isinstance(diff, dict) or not can_retire(diff, service_user, marker):
        logger.info(
            "Predecessor %s is published or no longer our draft; leaving it", previous
        )
        return
    subprocess.run(["jf", "help", "arcrc"], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(
        ["jf", "arcrc", "--cats-path", "/tmp/jf-home/.phabricator_cats"], check=True
    )
    subprocess.run(["jf", "help", "action"], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(
        [
            "jf",
            "action",
            "--abandon",
            "--diff",
            previous,
            "--message",
            f"Replaced by the combined wheel bump {new_diff}.",
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-diff", required=True)
    parser.add_argument("--service-user-id", required=True)
    parser.add_argument("--provenance", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        marker = json.loads(args.provenance.read_text())["automation"]
        if not isinstance(marker, str) or not marker.startswith(
            "Comms wheel bump automation: "
        ):
            raise ValueError("Missing automation marker")
        retire(args.new_diff, args.service_user_id, marker)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError):
        # Submission already succeeded. Do not turn cleanup failures into a
        # failed run that could be retried and submit a duplicate replacement.
        logger.exception("Draft was submitted; predecessor cleanup needs attention")


if __name__ == "__main__":
    main()
