#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Integration tests for send, recv, isend, and irecv point-to-point operations."""

import unittest

import torch
from torchcomms import distwrap as dist
from torchcomms.distwrap.tests.integration.test_helpers import (
    get_backend,
    get_device,
    get_rank_and_size,
    use_torchcomms,
)


class SendRecvTest(unittest.TestCase):
    """Test class for send/recv operations using distwrap."""

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

    def tearDown(self) -> None:
        """Synchronize all ranks after each test."""
        dist.barrier()

    def _get_pair_peer(self) -> int:
        """Return this rank's peer for world-wide pairwise P2P tests."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()

        if num_ranks < 2:
            self.skipTest("Need at least 2 ranks for send/recv test")
        if num_ranks % 2 != 0:
            self.skipTest("Need an even number of ranks for pairwise send/recv test")

        return rank + 1 if rank % 2 == 0 else rank - 1

    def test_send_recv(self) -> None:
        """Test synchronous send and recv between paired ranks."""
        rank = dist.get_rank()
        peer = self._get_pair_peer()
        device = get_device(rank)

        if rank % 2 == 0:
            tensor = torch.ones(1024, dtype=torch.float, device=device) * (rank + 42)
            dist.send(tensor, dst=peer)
        else:
            tensor = torch.zeros(1024, dtype=torch.float, device=device)
            dist.recv(tensor, src=peer)
            expected = torch.full_like(tensor.cpu(), peer + 42)
            torch.testing.assert_close(tensor.cpu(), expected)

    def test_isend_irecv(self) -> None:
        """Test asynchronous isend and irecv between paired ranks."""
        rank = dist.get_rank()
        peer = self._get_pair_peer()
        device = get_device(rank)

        if rank % 2 == 0:
            tensor = torch.ones(1024, dtype=torch.float, device=device) * (rank + 99)
            work = dist.isend(tensor, dst=peer)
            work.wait()
        else:
            tensor = torch.zeros(1024, dtype=torch.float, device=device)
            work = dist.irecv(tensor, src=peer)
            work.wait()
            expected = torch.full_like(tensor.cpu(), peer + 99)
            torch.testing.assert_close(tensor.cpu(), expected)

    def test_bidirectional_send_recv(self) -> None:
        """Test bidirectional send/recv between paired ranks."""
        rank = dist.get_rank()
        peer = self._get_pair_peer()
        device = get_device(rank)

        send_tensor = torch.ones(1024, dtype=torch.float, device=device) * (rank + 10)
        recv_tensor = torch.zeros(1024, dtype=torch.float, device=device)

        if rank % 2 == 0:
            send_work = dist.isend(send_tensor, dst=peer)
            recv_work = dist.irecv(recv_tensor, src=peer)
        else:
            recv_work = dist.irecv(recv_tensor, src=peer)
            send_work = dist.isend(send_tensor, dst=peer)

        send_work.wait()
        recv_work.wait()

        expected = torch.full_like(recv_tensor.cpu(), peer + 10)
        torch.testing.assert_close(recv_tensor.cpu(), expected)

    def test_batch_isend_irecv(self) -> None:
        """Test batch_isend_irecv for batched point-to-point operations."""
        rank = dist.get_rank()
        peer = self._get_pair_peer()
        device = get_device(rank)

        send_tensor = torch.ones(1024, dtype=torch.float, device=device) * (rank + 77)
        recv_tensor = torch.zeros(1024, dtype=torch.float, device=device)

        if rank % 2 == 0:
            p2p_ops = [
                dist.P2POp(dist.isend, send_tensor, peer=peer),
                dist.P2POp(dist.irecv, recv_tensor, peer=peer),
            ]
        else:
            p2p_ops = [
                dist.P2POp(dist.irecv, recv_tensor, peer=peer),
                dist.P2POp(dist.isend, send_tensor, peer=peer),
            ]

        works = dist.batch_isend_irecv(p2p_ops)
        for work in works:
            work.wait()

        expected = torch.full_like(recv_tensor.cpu(), peer + 77)
        torch.testing.assert_close(recv_tensor.cpu(), expected)

    def test_batch_isend_irecv_multiple_ops(self) -> None:
        """Test batch_isend_irecv with multiple send/recv operations."""
        rank = dist.get_rank()
        peer = self._get_pair_peer()
        device = get_device(rank)

        send_tensor1 = torch.ones(512, dtype=torch.float, device=device) * (rank + 11)
        send_tensor2 = torch.ones(512, dtype=torch.float, device=device) * (rank + 22)
        recv_tensor1 = torch.zeros(512, dtype=torch.float, device=device)
        recv_tensor2 = torch.zeros(512, dtype=torch.float, device=device)

        if rank % 2 == 0:
            p2p_ops = [
                dist.P2POp(dist.isend, send_tensor1, peer=peer),
                dist.P2POp(dist.isend, send_tensor2, peer=peer),
                dist.P2POp(dist.irecv, recv_tensor1, peer=peer),
                dist.P2POp(dist.irecv, recv_tensor2, peer=peer),
            ]
        else:
            p2p_ops = [
                dist.P2POp(dist.irecv, recv_tensor1, peer=peer),
                dist.P2POp(dist.irecv, recv_tensor2, peer=peer),
                dist.P2POp(dist.isend, send_tensor1, peer=peer),
                dist.P2POp(dist.isend, send_tensor2, peer=peer),
            ]

        works = dist.batch_isend_irecv(p2p_ops)
        for work in works:
            work.wait()

        expected1 = torch.full_like(recv_tensor1.cpu(), peer + 11)
        expected2 = torch.full_like(recv_tensor2.cpu(), peer + 22)
        torch.testing.assert_close(recv_tensor1.cpu(), expected1)
        torch.testing.assert_close(recv_tensor2.cpu(), expected2)


if __name__ == "__main__":
    unittest.main()
