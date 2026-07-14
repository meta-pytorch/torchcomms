# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import sys

try:
    from comms.uniflow.benchmarks.disagg_accuracy_test import main as accuracy_main
except ImportError:
    from disagg_accuracy_test import main as accuracy_main


def main(argv: list[str] | None = None) -> None:
    # Keep the historical entry point name while sharing the maintained runner.
    accuracy_main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
