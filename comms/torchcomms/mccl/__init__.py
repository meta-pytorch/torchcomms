#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""TorchComms-owned registration hooks for the MCCL backend plugin."""

import importlib
from pathlib import Path

from ._identity import validate_installed_build_info
from ._registration import register_c10d_backend


_BUILD_INFO_PATH = Path(__file__).with_name("_build_info.json")
_BACKEND_LOADED = False


def _ensure_backend_loaded() -> None:
    global _BACKEND_LOADED
    if _BACKEND_LOADED:
        return
    if _BUILD_INFO_PATH.is_file():
        validate_installed_build_info(_BUILD_INFO_PATH)
    importlib.import_module("torchcomms._comms_mccl")
    _BACKEND_LOADED = True


def _register_c10d_backend() -> None:
    """Register the TorchComms MCCL implementation as a CUDA c10d backend."""
    register_c10d_backend(_ensure_backend_loaded)


# The companion wheel points TorchComms discovery at this module. Its generated
# build identity is also the marker that distinguishes that wheel from internal
# source-tree imports, whose existing lazy registration path remains unchanged.
if _BUILD_INFO_PATH.is_file():
    _ensure_backend_loaded()
