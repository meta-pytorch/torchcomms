#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import unittest

import torch
from torchcomms.tests.integration.helpers.TorchCommTestHelpers import (
    TorchCommTestWrapper,
)


class AllGatherSingleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.wrapper = TorchCommTestWrapper()
        self.torchcomm = self.wrapper.get_torchcomm()
        self.device = self.torchcomm.get_device()
        self.num_ranks = self.torchcomm.get_size()

    def tearDown(self) -> None:
        del self.torchcomm
        del self.wrapper

    def test_cuda_all_gather_single_is_unsupported(self) -> None:
        input_tensor = torch.ones(4, dtype=torch.float, device=self.device)
        output_tensor = torch.zeros(
            4 * self.num_ranks, dtype=torch.float, device=self.device
        )

        for async_op in (False, True):
            with self.subTest(async_op=async_op):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "^CUDA all_gather_single is not supported by MCCL$",
                ):
                    self.torchcomm.all_gather_single(
                        output_tensor, input_tensor, async_op
                    )
