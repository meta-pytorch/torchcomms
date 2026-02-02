#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import unittest

from torchcomms.tests.helpers.py.test_helpers import skip_unless_pytorch_version


class TestXfailUnlessPytorchVersion(unittest.TestCase):
    def test_passes_when_version_meets_requirement(self):
        """Test that decorated function runs normally when version is sufficient."""

        @skip_unless_pytorch_version("2.12", _current_version="2.12.0")
        def dummy_test(self):
            return "success"

        result = dummy_test(self)
        self.assertEqual(result, "success")
        self.assertFalse(hasattr(dummy_test, "__unittest_expecting_failure__"))

    def test_xfails_when_version_below_requirement(self):
        """Test that decorated function is marked as expectedFailure when version is too low."""

        @skip_unless_pytorch_version("2.12", _current_version="2.10.0")
        def dummy_test(self):
            pass

        self.assertTrue(getattr(dummy_test, "__unittest_expecting_failure__", False))

    def test_handles_version_with_cuda_suffix(self):
        """Test that version parsing handles CUDA suffixes like 2.12.0+cu118."""

        @skip_unless_pytorch_version("2.12", _current_version="2.12.0+cu118")
        def dummy_test(self):
            return "success"

        result = dummy_test(self)
        self.assertEqual(result, "success")

    def test_handles_dev_version(self):
        """Test that version parsing handles dev versions like 2.12.0.dev20240101."""

        @skip_unless_pytorch_version("2.12", _current_version="2.12.0.dev20240101")
        def dummy_test(self):
            return "success"

        result = dummy_test(self)
        self.assertEqual(result, "success")

    def test_compares_minor_version_correctly(self):
        """Test that 2.11 < 2.12."""

        @skip_unless_pytorch_version("2.12", _current_version="2.11.0")
        def dummy_test(self):
            pass

        self.assertTrue(getattr(dummy_test, "__unittest_expecting_failure__", False))

    def test_compares_major_version_correctly(self):
        """Test that 3.0 > 2.12."""

        @skip_unless_pytorch_version("2.12", _current_version="3.0.0")
        def dummy_test(self):
            return "success"

        result = dummy_test(self)
        self.assertEqual(result, "success")


if __name__ == "__main__":
    unittest.main()
