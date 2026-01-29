#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Integration tests for broadcast collective."""

import unittest

import torch
from torchcomms import distwrap as dist
from torchcomms.distwrap.tests.integration.test_helpers import (
    get_backend,
    get_device,
    get_rank_and_size,
    use_torchcomms,
)


class BroadcastTest(unittest.TestCase):
    """Test class for broadcast operations using distwrap."""

    @classmethod
    def setUpClass(cls) -> None:
        """Initialize distwrap once for all tests."""
        rank, _ = get_rank_and_size()
        device = get_device(rank)
        backend = get_backend()

        dist.init_process_group(
            backend=backend,
            use_torchcomms=use_torchcomms(),
        )

        if device.type == "cuda":
            torch.cuda.set_device(device)

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up distwrap after all tests."""
        dist.destroy_process_group()

    def test_sync_broadcast(self) -> None:
        """Test synchronous broadcast from rank 0."""
        rank = dist.get_rank()
        device = get_device(rank)

        if rank == 0:
            tensor = torch.ones(1024, dtype=torch.float, device=device) * 42
        else:
            tensor = torch.zeros(1024, dtype=torch.float, device=device)

        dist.broadcast(tensor, src=0, async_op=False)

        expected_tensor = torch.full_like(tensor.cpu(), 42)
        torch.testing.assert_close(tensor.cpu(), expected_tensor)

    def test_async_broadcast(self) -> None:
        """Test asynchronous broadcast from rank 0."""
        rank = dist.get_rank()
        device = get_device(rank)

        if rank == 0:
            tensor = torch.ones(1024, dtype=torch.float, device=device) * 99
        else:
            tensor = torch.zeros(1024, dtype=torch.float, device=device)

        work = dist.broadcast(tensor, src=0, async_op=True)
        if work is None:
            raise AssertionError("work is None")
        work.wait()

        expected_tensor = torch.full_like(tensor.cpu(), 99)
        torch.testing.assert_close(tensor.cpu(), expected_tensor)


if __name__ == "__main__":
    unittest.main()
