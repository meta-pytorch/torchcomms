#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import itertools
import unittest

import torch
from torchcomms import new_comm
from torchcomms.tests.integration.helpers.TorchCommTestHelpers import (
    create_store,
    get_device,
    get_dtype_name,
    get_rank_and_size,
)


class LazyP2PTest(unittest.TestCase):
    """Test NCCL lazy pair-comm creation via the 'nccl-lazy' backend.

    The 'nccl-lazy' backend wraps TorchCommNCCL in LazyBackend, which builds a
    dedicated 2-rank sibling communicator the first time a rank does a P2P op
    with a given peer (TorchCommNCCL::createPairComm). That bootstrap uses the
    store preserved from init (bootstrap_store_), wrapped in a PrefixStore, to
    exchange the ncclUniqueId -- instead of creating a fresh TCPStore from
    MASTER_ADDR/MASTER_PORT. This test exercises that path by creating an
    'nccl-lazy' comm with an explicit store and running P2P between disjoint
    rank pairs, which forces createPairComm to run on every rank.
    """

    counts = [4]
    dtypes = [torch.float, torch.int, torch.int8]

    def setUp(self):
        self.rank, self.num_ranks = get_rank_and_size()
        self.device = get_device("nccl", self.rank)
        # A store MUST be supplied: createPairComm bootstraps each pair comm
        # from the store preserved at init (bootstrap_store_).
        self.torchcomm = new_comm(
            "nccl-lazy",
            self.device,
            store=create_store(),
            name="lazy_p2p_test",
        )

    def tearDown(self):
        if self.torchcomm is not None:
            self.torchcomm.finalize()
            self.torchcomm = None

    def _lazy_send_recv(self, count, dtype):
        """Exchange a tensor with the XOR-partner rank.

        Disjoint pairs (0<->1, 2<->3, ...) keep pair-comm bootstrap and the
        transfer deadlock-free: the lower rank sends first, the higher rank
        receives first.
        """
        print(
            f"Testing lazy pair-comm send/recv with count={count} and dtype={get_dtype_name(dtype)}"
        )

        partner = self.rank ^ 1
        if partner >= self.num_ranks:
            # Unpaired rank (odd world size): nothing to exchange.
            return

        send_tensor = torch.ones(count, dtype=dtype, device=self.device) * int(
            self.rank + 1
        )
        recv_tensor = torch.zeros(count, dtype=dtype, device=self.device)

        if self.rank < partner:
            send_work = self.torchcomm.send(send_tensor, partner, True)
            recv_work = self.torchcomm.recv(recv_tensor, partner, True)
        else:
            recv_work = self.torchcomm.recv(recv_tensor, partner, True)
            send_work = self.torchcomm.send(send_tensor, partner, True)

        send_work.wait()
        recv_work.wait()

        expected = torch.ones(count, dtype=dtype) * int(partner + 1)
        description = f"lazy recv from partner rank {partner}"
        if dtype == torch.float:
            self.assertTrue(
                torch.allclose(recv_tensor.cpu(), expected),
                f"Tensors not close enough for {description}",
            )
        else:
            self.assertTrue(
                torch.equal(recv_tensor.cpu(), expected),
                f"Tensors not equal for {description}",
            )

    def test_lazy_send_recv(self):
        """P2P over lazily-created pair comms across disjoint rank pairs."""
        for count, dtype in itertools.product(self.counts, self.dtypes):
            with self.subTest(count=count, dtype=dtype):
                self._lazy_send_recv(count, dtype)


if __name__ == "__main__":
    unittest.main()
