#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import unittest


def skip_unless_pytorch_version(
    min_version: str,
    reason: str = "Requires newer PyTorch",
    _current_version: str | None = None,
):
    """Decorator to skip tests if PyTorch version is below min_version.

    When applied to a test class and the version requirement is not met,
    replaces the class with a placeholder that has one skipped test.
    This ensures pytest doesn't exit with code 5 (no tests collected).

    Usage:
        @skip_unless_pytorch_version("2.6", "Requires PyTorch 2.6+ hotfixes")
        def test_something(self):
            ...

    Args:
        min_version: Minimum required PyTorch version (e.g., "2.12")
        reason: Reason for the version requirement
        _current_version: For testing only - override the detected PyTorch version
    """
    from packaging import version

    if _current_version is None:
        import torch

        _current_version = torch.__version__

    meets_requirement = (
        version.parse(_current_version) >= version.parse(min_version)
        or os.environ.get("TORCHCOMMS_TEST_IGNORE_PYTORCH_VERSION_REQUIREMENT") == "1"
    )

    skip_message = (
        f"Requires PyTorch {min_version} or higher. "
        "To override, set TORCHCOMMS_TEST_IGNORE_PYTORCH_VERSION_REQUIREMENT=1"
    )

    def decorator(test_item):
        if meets_requirement:
            return test_item
        return unittest.skip(skip_message)(test_item)

    return decorator
