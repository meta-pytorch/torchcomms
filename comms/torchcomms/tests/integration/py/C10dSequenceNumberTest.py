# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-unsafe

"""Integration tests for ProcessGroup._get_sequence_number_for_group() routed
through BackendWrapper.

BackendWrapper mirrors c10d::ProcessGroupNCCL::seqCollective_: a per-PG
monotonic counter bumped once per collective (never for P2P send/recv), exposed
to Python via ProcessGroup._get_sequence_number_for_group(). These tests pin
that contract -- each collective advances the counter by exactly one, and P2P
leaves it untouched. They assert on deltas rather than an absolute starting
value because init_process_group may issue its own collectives.
"""

import os
import unittest

import torch
import torch.distributed as dist
from torchcomms.tests.integration.helpers.TorchCommTestHelpers import (
    get_device,
    get_rank_and_size,
)


class TestC10dSequenceNumber(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dist.config.use_torchcomms = True
        rank, world_size = get_rank_and_size()
        dist.init_process_group(
            backend=os.environ["TEST_BACKEND"], rank=rank, world_size=world_size
        )
        device = get_device(os.environ["TEST_BACKEND"], dist.get_rank())
        torch.set_default_device(device)

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()

    def setUp(self):
        self.pg = dist.group.WORLD
        self.rank = dist.get_rank()
        self.world = dist.get_world_size()

    def _seq(self) -> int:
        return self.pg._get_sequence_number_for_group()

    def test_all_reduce_increments_by_one_each(self):
        """Each all_reduce bumps the collective counter by exactly one, matching
        the c10d ProcessGroupNCCL contract."""
        base = self._seq()
        num_collectives = 5
        for _ in range(num_collectives):
            dist.all_reduce(torch.ones(4))
        self.assertEqual(self._seq(), base + num_collectives)

    def test_distinct_collectives_each_increment_by_one(self):
        """Different collective types all funnel through the same counter, one
        bump apiece."""
        base = self._seq()
        tensor = torch.ones(4)
        dist.all_reduce(tensor)
        dist.broadcast(tensor, src=0)
        dist.reduce(tensor, dst=0)
        dist.barrier()
        self.assertEqual(self._seq(), base + 4)

    def test_all_gather_increments_once(self):
        """A single all_gather is one collective regardless of world size."""
        base = self._seq()
        output = [torch.empty(2) for _ in range(self.world)]
        dist.all_gather(output, torch.ones(2) * self.rank)
        self.assertEqual(self._seq(), base + 1)

    def test_p2p_does_not_increment(self):
        """send/recv are P2P, not collectives, so they must leave the collective
        sequence number untouched (getSequenceNumberForGroup returns
        seqCollective_ only, never seqP2P)."""
        if self.world < 2:
            self.skipTest("P2P test requires world_size >= 2")
        peer_next = (self.rank + 1) % self.world
        peer_prev = (self.rank - 1 + self.world) % self.world
        base = self._seq()
        send_tensor = torch.tensor([self.rank], dtype=torch.float32)
        recv_tensor = torch.empty(1, dtype=torch.float32)
        if self.rank % 2 == 0:
            dist.send(send_tensor, dst=peer_next)
            dist.recv(recv_tensor, src=peer_prev)
        else:
            dist.recv(recv_tensor, src=peer_prev)
            dist.send(send_tensor, dst=peer_next)
        self.assertEqual(self._seq(), base)


if __name__ == "__main__":
    unittest.main()
