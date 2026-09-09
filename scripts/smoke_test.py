# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
DO NOT DELETE

This file is used to test the build of the torchcomms package in CI.
"""

import importlib.util

import torchcomms  # noqa: F401

# uniflow ships only in USE_UNIFLOW builds (not ROCm); when present it must
# import (a stub-only package was the failure mode of #3168).
if importlib.util.find_spec("uniflow") is not None:
    import uniflow  # noqa: F401
