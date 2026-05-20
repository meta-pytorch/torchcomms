#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import unittest

from torchcomms.functional import is_torch_compile_supported_and_enabled

# pyre-fixme[5]: Global annotation for skip decorator.
skip_if_ncclx = unittest.skipIf(
    os.getenv("TEST_BACKEND") == "ncclx", "Skipping tests for NCCLX backend."
)


def skip_if_torch_compile_not_supported_or_enabled(
    _current_version: str | None = None,
):
    def decorator(test_item):
        if is_torch_compile_supported_and_enabled(_current_version=_current_version):
            return test_item
        return unittest.skip("is_torch_compile_supported_and_enabled() is False.")(
            test_item
        )

    return decorator
