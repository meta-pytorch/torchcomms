#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import unittest

os.environ["TORCHCOMMS_PATCH_FOR_COMPILE"] = "1"

from torchcomms.functional import inductor_lowering, registry
from torchcomms.tests.helpers.py.test_helpers import (
    skip_if_torch_compile_not_supported_or_enabled,
)


class _ProcessKernelResult:
    def __init__(
        self,
        example_output,
        tensor_args,
        non_tensor_args,
        unflatten_args,
        unbacked_bindings,
    ):
        self.example_output = example_output
        self.tensor_args = tensor_args
        self.non_tensor_args = non_tensor_args
        self.unflatten_args = unflatten_args
        self.unbacked_bindings = unbacked_bindings


@skip_if_torch_compile_not_supported_or_enabled()
class TestProcessKernelCompatibility(unittest.TestCase):
    def setUp(self):
        self.expected = ("example", [1], [2], lambda _: _, {"sym": "binding"})

    def test_registry_unpack_accepts_structured_result(self):
        result = _ProcessKernelResult(*self.expected)
        self.assertEqual(
            registry._unpack_process_kernel_result(result),
            self.expected,
        )

    def test_registry_unpack_accepts_legacy_tuple(self):
        self.assertEqual(
            registry._unpack_process_kernel_result(self.expected),
            self.expected,
        )

    def test_inductor_unpack_accepts_structured_result(self):
        if not hasattr(inductor_lowering, "_unpack_process_kernel_result"):
            self.skipTest("inductor helpers unavailable")
        result = _ProcessKernelResult(*self.expected)
        self.assertEqual(
            inductor_lowering._unpack_process_kernel_result(result),
            self.expected,
        )

    def test_inductor_unpack_accepts_legacy_tuple(self):
        if not hasattr(inductor_lowering, "_unpack_process_kernel_result"):
            self.skipTest("inductor helpers unavailable")
        self.assertEqual(
            inductor_lowering._unpack_process_kernel_result(self.expected),
            self.expected,
        )
