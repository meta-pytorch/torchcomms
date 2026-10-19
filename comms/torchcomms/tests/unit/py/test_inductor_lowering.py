#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import unittest
from unittest import mock


os.environ["TORCHCOMMS_PATCH_FOR_COMPILE"] = "1"

import torch

# Registers the compile ops that register_torchcomms_lowerings() lowers.
import torchcomms  # noqa: F401
from torch._inductor import config as inductor_config
from torchcomms.functional.inductor_lowering import register_torchcomms_lowerings
from torchcomms.tests.helpers.py.test_helpers import (
    skip_if_torch_compile_not_supported_or_enabled,
)


@skip_if_torch_compile_not_supported_or_enabled()
class RegisterTorchcommsLoweringsTest(unittest.TestCase):
    def _simple_overlap_after_registration(self, torch_version: str) -> bool:
        with (
            mock.patch.object(torch, "__version__", torch_version),
            inductor_config.patch(
                {"aten_distributed_optimizations.enable_simple_overlap": True}
            ),
        ):
            register_torchcomms_lowerings()
            return inductor_config.aten_distributed_optimizations.enable_simple_overlap

    def test_disables_simple_overlap_on_torch_2_14(self) -> None:
        for torch_version in ["2.14.0+cu130", "2.14.1", "2.14.0rc1"]:
            with self.subTest(torch_version=torch_version):
                self.assertFalse(self._simple_overlap_after_registration(torch_version))

    def test_keeps_simple_overlap_on_other_versions(self) -> None:
        for torch_version in [
            "2.13.1+cu130",
            "2.15.0",
            "2.15.0.dev20261019+cu130",
            "2.16.0a0",
        ]:
            with self.subTest(torch_version=torch_version):
                self.assertTrue(self._simple_overlap_after_registration(torch_version))
