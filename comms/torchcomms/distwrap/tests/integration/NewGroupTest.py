#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Integration tests for new_group.

Tests various parameter combinations for new_group:
- Explicit ranks and default ranks
- Backend parameter (required)
- Timeout parameter
- Group description parameter
- use_local_synchronization parameter
- device_id parameter
- Collectives on new groups
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


class NewGroupTest(unittest.TestCase):
    """Test class for new_group operations using distwrap."""

    def setUp(self) -> None:
        """Initialize distwrap before each test."""
        rank, size = get_rank_and_size()
        device = get_device(rank)
        backend = get_backend()

        dist.init_process_group(
            backend=backend,
            use_torchcomms=use_torchcomms(),
        )

        if device.type == "cuda":
            torch.cuda.set_device(device)

    def tearDown(self) -> None:
        """Clean up distwrap after each test."""
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

    def test_new_group_all_ranks(self) -> None:
        """Test new_group with all ranks."""
        num_ranks = dist.get_world_size()
        all_ranks = list(range(num_ranks))
        backend = get_backend()

        if use_torchcomms():
            # new_group is not supported with torchcomms
            with self.assertRaises(AssertionError) as context:
                dist.new_group(ranks=all_ranks, backend=backend)
            self.assertIn(
                "new_group is not supported with torchcomms",
                str(context.exception),
            )
            return

        new_pg = dist.new_group(ranks=all_ranks, backend=backend)

        self.assertIsNotNone(new_pg)
        self.assertEqual(
            dist.get_world_size(new_pg),
            num_ranks,
        )

        self._verify_collective(new_pg)

        dist.destroy_process_group(new_pg)

    def test_new_group_default_ranks(self) -> None:
        """Test new_group with default ranks (None = all ranks)."""
        num_ranks = dist.get_world_size()
        backend = get_backend()

        if use_torchcomms():
            with self.assertRaises(AssertionError) as context:
                dist.new_group(backend=backend)
            self.assertIn(
                "new_group is not supported with torchcomms",
                str(context.exception),
            )
            return

        new_pg = dist.new_group(backend=backend)

        self.assertIsNotNone(new_pg)
        self.assertEqual(
            dist.get_world_size(new_pg),
            num_ranks,
        )

        self._verify_collective(new_pg)

        dist.destroy_process_group(new_pg)

    def test_new_group_collective(self) -> None:
        """Test that collectives work on new groups."""
        num_ranks = dist.get_world_size()
        backend = get_backend()

        # Create a group with all ranks
        all_ranks = list(range(num_ranks))

        if use_torchcomms():
            # new_group is not supported with torchcomms
            with self.assertRaises(AssertionError) as context:
                dist.new_group(ranks=all_ranks, backend=backend)
            self.assertIn(
                "new_group is not supported with torchcomms",
                str(context.exception),
            )
            return

        new_pg = dist.new_group(ranks=all_ranks, backend=backend)

        self._verify_collective(new_pg)

        dist.destroy_process_group(new_pg)

    def test_new_group_with_timeout(self) -> None:
        """Test new_group with explicit timeout parameter."""
        from datetime import timedelta

        num_ranks = dist.get_world_size()
        all_ranks = list(range(num_ranks))
        backend = get_backend()

        if use_torchcomms():
            with self.assertRaises(AssertionError):
                dist.new_group(
                    ranks=all_ranks,
                    backend=backend,
                    timeout=timedelta(seconds=60),
                )
            return

        new_pg = dist.new_group(
            ranks=all_ranks,
            backend=backend,
            timeout=timedelta(seconds=60),
        )

        self.assertIsNotNone(new_pg)
        self.assertEqual(
            dist.get_world_size(new_pg),
            num_ranks,
        )

        self._verify_collective(new_pg)

        dist.destroy_process_group(new_pg)

    def test_new_group_with_group_desc(self) -> None:
        """Test new_group with group_desc parameter."""
        num_ranks = dist.get_world_size()
        all_ranks = list(range(num_ranks))
        backend = get_backend()

        if use_torchcomms():
            with self.assertRaises(AssertionError):
                dist.new_group(
                    ranks=all_ranks,
                    backend=backend,
                    group_desc="test_new_group",
                )
            return

        new_pg = dist.new_group(
            ranks=all_ranks,
            backend=backend,
            group_desc="test_new_group",
        )

        self.assertIsNotNone(new_pg)
        self.assertEqual(
            dist.get_world_size(new_pg),
            num_ranks,
        )

        self._verify_collective(new_pg)

        dist.destroy_process_group(new_pg)

    def test_new_group_with_local_synchronization(self) -> None:
        """Test new_group with use_local_synchronization parameter."""
        num_ranks = dist.get_world_size()
        all_ranks = list(range(num_ranks))
        backend = get_backend()

        if use_torchcomms():
            with self.assertRaises(AssertionError):
                dist.new_group(
                    ranks=all_ranks,
                    backend=backend,
                    use_local_synchronization=True,
                )
            return

        new_pg = dist.new_group(
            ranks=all_ranks,
            backend=backend,
            use_local_synchronization=True,
        )

        self.assertIsNotNone(new_pg)
        self.assertEqual(
            dist.get_world_size(new_pg),
            num_ranks,
        )

        self._verify_collective(new_pg)

        dist.destroy_process_group(new_pg)

    def test_new_group_with_device_id(self) -> None:
        """Test new_group with device_id parameter."""
        rank = dist.get_rank()
        device = get_device(rank)

        # Only test with CUDA if available
        if device.type != "cuda":
            self.skipTest("device_id test requires CUDA")

        num_ranks = dist.get_world_size()
        all_ranks = list(range(num_ranks))
        backend = get_backend()

        if use_torchcomms():
            with self.assertRaises(AssertionError):
                dist.new_group(
                    ranks=all_ranks,
                    backend=backend,
                    device_id=device,
                )
            return

        new_pg = dist.new_group(
            ranks=all_ranks,
            backend=backend,
            device_id=device,
        )

        self.assertIsNotNone(new_pg)
        self.assertEqual(
            dist.get_world_size(new_pg),
            num_ranks,
        )

        self._verify_collective(new_pg)

        dist.destroy_process_group(new_pg)

    def test_new_group_backend_required(self) -> None:
        """Test that new_group raises error when backend is None."""
        num_ranks = dist.get_world_size()
        all_ranks = list(range(num_ranks))

        if use_torchcomms():
            # new_group is not supported with torchcomms
            with self.assertRaises(AssertionError) as context:
                dist.new_group(ranks=all_ranks, backend=None)
            self.assertIn(
                "new_group is not supported with torchcomms",
                str(context.exception),
            )
            return

        with self.assertRaises(ValueError) as context:
            dist.new_group(ranks=all_ranks, backend=None)

        self.assertIn("backend must be specified", str(context.exception))

    def test_new_group_with_pg_options(self) -> None:
        """Test new_group with pg_options parameter."""
        backend = get_backend()
        num_ranks = dist.get_world_size()
        all_ranks = list(range(num_ranks))

        pg_options = get_pg_options(backend)
        if pg_options is None:
            self.skipTest(f"pg_options test not supported for backend {backend}")

        if use_torchcomms():
            with self.assertRaises(AssertionError):
                dist.new_group(
                    ranks=all_ranks,
                    backend=backend,
                    pg_options=pg_options,
                )
            return

        new_pg = dist.new_group(
            ranks=all_ranks,
            backend=backend,
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
