#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import subprocess
import sys
import unittest


def can_import_torchcomms() -> bool:
    """Check if torchcomms can be imported in a subprocess.

    Inherits the current environment, so set TORCHCOMMS_PATCH_FOR_COMPILE
    before calling if you want to test compile support.
    """
    code = """
try:
    import torchcomms
    print("SUCCESS")
except ImportError:
    print("FAIL")
"""

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
            env=os.environ.copy(),
        )
        return result.stdout.strip() == "SUCCESS"
    except Exception:
        return False


def skip_if_pytorch_version_unsupported():
    if not can_import_torchcomms():
        raise unittest.SkipTest("PyTorch version does not support torch.compile")
