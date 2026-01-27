#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from torchcomms.tests.unit.py.test_helpers import (
    can_import_torchcomms,
    skip_if_pytorch_version_unsupported,
)


class TestCanImportTorchcomms(unittest.TestCase):
    def test_can_import_without_compile_flag(self):
        env_backup = os.environ.pop("TORCHCOMMS_PATCH_FOR_COMPILE", None)
        try:
            result = can_import_torchcomms()
            self.assertTrue(result)
        finally:
            if env_backup is not None:
                os.environ["TORCHCOMMS_PATCH_FOR_COMPILE"] = env_backup

    def test_can_import_with_compile_flag_on_supported_pytorch(self):
        os.environ["TORCHCOMMS_PATCH_FOR_COMPILE"] = "1"
        try:
            result = can_import_torchcomms()
            self.assertIsInstance(result, bool)
        finally:
            del os.environ["TORCHCOMMS_PATCH_FOR_COMPILE"]

    def test_fails_when_opaque_base_import_fails(self):
        """Test that torchcomms import fails when torch._opaque_base is unavailable."""
        code = """
import sys
from unittest.mock import MagicMock

# Mock torch._opaque_base so accessing OpaqueBaseMeta raises ImportError
class FailingModule:
    @property
    def OpaqueBaseMeta(self):
        raise ImportError("No module named 'torch._opaque_base'")

sys.modules["torch._opaque_base"] = FailingModule()

import os
os.environ["TORCHCOMMS_PATCH_FOR_COMPILE"] = "1"

try:
    import torchcomms
    print("SUCCESS")
except ImportError:
    print("FAIL")
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.stdout.strip(), "FAIL")


class TestSkipIfPytorchVersionUnsupported(unittest.TestCase):
    def test_does_not_skip_without_compile_flag(self):
        env_backup = os.environ.pop("TORCHCOMMS_PATCH_FOR_COMPILE", None)
        try:
            skip_if_pytorch_version_unsupported()
        finally:
            if env_backup is not None:
                os.environ["TORCHCOMMS_PATCH_FOR_COMPILE"] = env_backup

    def test_skips_when_can_import_returns_false(self):
        """Test that skip_if_pytorch_version_unsupported raises SkipTest when import fails."""
        with patch(
            "torchcomms.tests.unit.py.test_helpers.can_import_torchcomms",
            return_value=False,
        ):
            with self.assertRaises(unittest.SkipTest):
                skip_if_pytorch_version_unsupported()


if __name__ == "__main__":
    unittest.main()
