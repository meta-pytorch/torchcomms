#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Integration tests for split_group.

Tests various parameter combinations for split_group:
- split_ranks with single group, individual groups
- Collectives on split groups
- Explicit backend parameter
- Timeout parameter
- Group description parameter
- Explicit parent group parameter
"""

import unittest

import torch
from torchcomms import distwrap as dist
from torchcomms.distwrap.tests.integration.test_helpers import (
    get_backend,
    get_device,
    get_pg_options,
    get_rank_and_size,
    use_torchcomms,
)


class SplitGroupTest(unittest.TestCase):
    """Test class for split_group operations using distwrap."""

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
        if dist.is_initialized():
            dist.destroy_process_group()

    def _verify_collective(self, pg: dist.ProcessGroup) -> None:
        """Verify the process group works with a basic all_reduce."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size(pg)
        device = get_device(rank)

        input_tensor = torch.ones(1024, dtype=torch.float, device=device) * (rank + 1)
        dist.all_reduce(input_tensor, dist.ReduceOp.SUM, group=pg, async_op=False)

        expected = num_ranks * (num_ranks + 1) // 2
        expected_tensor = torch.full_like(input_tensor.cpu(), expected)
        torch.testing.assert_close(input_tensor.cpu(), expected_tensor)

    def test_split_group_single_group(self) -> None:
        """Test split_group with all ranks in one group."""
        num_ranks = dist.get_world_size()
        all_ranks = list(range(num_ranks))

        new_pg = dist.split_group(split_ranks=[all_ranks])

        self.assertIsNotNone(new_pg)
        self.assertEqual(
            dist.get_world_size(new_pg),
            num_ranks,
        )

        self._verify_collective(new_pg)

        dist.destroy_process_group(new_pg)

    def test_split_group_individual_groups(self) -> None:
        """Test split_group with each rank in its own group."""
        num_ranks = dist.get_world_size()
        split_ranks = [[i] for i in range(num_ranks)]

        new_pg = dist.split_group(split_ranks=split_ranks)

        self.assertIsNotNone(new_pg)
        self.assertEqual(dist.get_world_size(new_pg), 1)

        # Can't do allreduce with group size 1, just verify it was created
        # self._verify_collective(new_pg)

        dist.destroy_process_group(new_pg)

    def test_split_group_collective(self) -> None:
        """Test that collectives work on split groups."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        device = get_device(rank)

        # Create a single group with all ranks
        all_ranks = list(range(num_ranks))
        new_pg = dist.split_group(split_ranks=[all_ranks])

        # Test all_reduce on the new group
        input_tensor = torch.ones(1024, dtype=torch.float, device=device) * (rank + 1)
        dist.all_reduce(input_tensor, dist.ReduceOp.SUM, group=new_pg, async_op=False)

        expected = num_ranks * (num_ranks + 1) // 2
        expected_tensor = torch.full_like(input_tensor.cpu(), expected)
        torch.testing.assert_close(input_tensor.cpu(), expected_tensor)

        dist.destroy_process_group(new_pg)

    def test_split_group_with_backend(self) -> None:
        """Test split_group with explicit backend parameter."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        device = get_device(rank)

        # Create a single group with all ranks and explicit backend
        all_ranks = list(range(num_ranks))
        backend = get_backend()
        new_pg = dist.split_group(split_ranks=[all_ranks], backend=backend)

        self.assertIsNotNone(new_pg)
        self.assertEqual(
            dist.get_world_size(new_pg),
            num_ranks,
        )

        # Test all_reduce on the new group with explicit backend
        input_tensor = torch.ones(1024, dtype=torch.float, device=device) * (rank + 1)
        dist.all_reduce(input_tensor, dist.ReduceOp.SUM, group=new_pg, async_op=False)

        expected = num_ranks * (num_ranks + 1) // 2
        expected_tensor = torch.full_like(input_tensor.cpu(), expected)
        torch.testing.assert_close(input_tensor.cpu(), expected_tensor)

        dist.destroy_process_group(new_pg)

    def test_split_group_with_timeout(self) -> None:
        """Test split_group with explicit timeout parameter."""
        from datetime import timedelta

        num_ranks = dist.get_world_size()
        all_ranks = list(range(num_ranks))

        new_pg = dist.split_group(
            split_ranks=[all_ranks],
            timeout=timedelta(seconds=60),
        )

        self.assertIsNotNone(new_pg)
        self.assertEqual(
            dist.get_world_size(new_pg),
            num_ranks,
        )

        self._verify_collective(new_pg)

        dist.destroy_process_group(new_pg)

    def test_split_group_with_group_desc(self) -> None:
        """Test split_group with group_desc parameter."""
        num_ranks = dist.get_world_size()
        all_ranks = list(range(num_ranks))

        new_pg = dist.split_group(
            split_ranks=[all_ranks],
            group_desc="test_split_group",
        )

        self.assertIsNotNone(new_pg)
        self.assertEqual(
            dist.get_world_size(new_pg),
            num_ranks,
        )

        self._verify_collective(new_pg)

        dist.destroy_process_group(new_pg)

    def test_split_group_with_parent_group(self) -> None:
        """Test split_group with explicit parent group parameter."""
        num_ranks = dist.get_world_size()
        all_ranks = list(range(num_ranks))

        # First create a split group from WORLD
        parent_pg = dist.split_group(split_ranks=[all_ranks])

        # Then split from that parent group
        new_pg = dist.split_group(
            parent_pg=parent_pg,
            split_ranks=[all_ranks],
        )

        self.assertIsNotNone(new_pg)
        self.assertEqual(
            dist.get_world_size(new_pg),
            num_ranks,
        )

        self._verify_collective(new_pg)

        # Destroy child group first, then parent
        dist.destroy_process_group(new_pg)
        dist.destroy_process_group(parent_pg)

    def test_split_group_with_pg_options(self) -> None:
        """Test split_group with pg_options parameter."""
        num_ranks = dist.get_world_size()
        all_ranks = list(range(num_ranks))

        backend = get_backend()
        pg_options = get_pg_options(backend)
        if pg_options is None:
            self.skipTest(f"pg_options test not supported for backend {backend}")

        new_pg = dist.split_group(
            split_ranks=[all_ranks],
            pg_options=pg_options,
        )

        self.assertIsNotNone(new_pg)
        self.assertEqual(
            dist.get_world_size(new_pg),
            num_ranks,
        )

        self._verify_collective(new_pg)

        dist.destroy_process_group(new_pg)


if __name__ == "__main__":
    unittest.main()
