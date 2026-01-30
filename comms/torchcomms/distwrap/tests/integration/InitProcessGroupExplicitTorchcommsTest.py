#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Test init_process_group with use_torchcomms explicitly set to True."""

import unittest

import torch
from torchcomms import distwrap as dist
from torchcomms.distwrap.tests.integration.test_helpers import (
    get_backend,
    get_device,
    get_rank_and_size,
)


class Test(unittest.TestCase):
    def test(self) -> None:
        rank, num_ranks = get_rank_and_size()
        device = get_device(rank)

        dist.init_process_group(
            backend=get_backend(),
            use_torchcomms=True,
        )

        if device.type == "cuda":
            torch.cuda.set_device(device)

        try:
            self.assertTrue(dist.is_initialized())

            input_tensor = torch.ones(1024, dtype=torch.float, device=device) * (
                rank + 1
            )
            dist.all_reduce(input_tensor, dist.ReduceOp.SUM, async_op=False)

            expected = num_ranks * (num_ranks + 1) // 2
            expected_tensor = torch.full_like(input_tensor.cpu(), expected)
            torch.testing.assert_close(input_tensor.cpu(), expected_tensor)
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
