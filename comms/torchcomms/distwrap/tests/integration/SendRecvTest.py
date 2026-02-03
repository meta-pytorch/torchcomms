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

    def test_send_recv(self) -> None:
        """Test synchronous send and recv between rank 0 and rank 1."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        device = get_device(rank)

        if num_ranks < 2:
            self.skipTest("Need at least 2 ranks for send/recv test")

        if rank == 0:
            tensor = torch.ones(1024, dtype=torch.float, device=device) * 42
            dist.send(tensor, dst=1)
        elif rank == 1:
            tensor = torch.zeros(1024, dtype=torch.float, device=device)
            dist.recv(tensor, src=0)
            expected = torch.full_like(tensor.cpu(), 42)
            torch.testing.assert_close(tensor.cpu(), expected)

    def test_isend_irecv(self) -> None:
        """Test asynchronous isend and irecv between rank 0 and rank 1."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        device = get_device(rank)

        if num_ranks < 2:
            self.skipTest("Need at least 2 ranks for isend/irecv test")

        if rank == 0:
            tensor = torch.ones(1024, dtype=torch.float, device=device) * 99
            work = dist.isend(tensor, dst=1)
            work.wait()
        elif rank == 1:
            tensor = torch.zeros(1024, dtype=torch.float, device=device)
            work = dist.irecv(tensor, src=0)
            work.wait()
            expected = torch.full_like(tensor.cpu(), 99)
            torch.testing.assert_close(tensor.cpu(), expected)

    def test_bidirectional_send_recv(self) -> None:
        """Test bidirectional send/recv between rank 0 and rank 1."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        device = get_device(rank)

        if num_ranks < 2:
            self.skipTest("Need at least 2 ranks for bidirectional test")

        if rank == 0:
            send_tensor = torch.ones(1024, dtype=torch.float, device=device) * 10
            recv_tensor = torch.zeros(1024, dtype=torch.float, device=device)

            send_work = dist.isend(send_tensor, dst=1)
            recv_work = dist.irecv(recv_tensor, src=1)

            send_work.wait()
            recv_work.wait()

            expected = torch.full_like(recv_tensor.cpu(), 20)
            torch.testing.assert_close(recv_tensor.cpu(), expected)

        elif rank == 1:
            send_tensor = torch.ones(1024, dtype=torch.float, device=device) * 20
            recv_tensor = torch.zeros(1024, dtype=torch.float, device=device)

            recv_work = dist.irecv(recv_tensor, src=0)
            send_work = dist.isend(send_tensor, dst=0)

            recv_work.wait()
            send_work.wait()

            expected = torch.full_like(recv_tensor.cpu(), 10)
            torch.testing.assert_close(recv_tensor.cpu(), expected)

    def test_batch_isend_irecv(self) -> None:
        """Test batch_isend_irecv for batched point-to-point operations."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        device = get_device(rank)

        if num_ranks < 2:
            self.skipTest("Need at least 2 ranks for batch_isend_irecv test")

        if rank == 0:
            send_tensor = torch.ones(1024, dtype=torch.float, device=device) * 77
            recv_tensor = torch.zeros(1024, dtype=torch.float, device=device)

            p2p_ops = [
                dist.P2POp(dist.isend, send_tensor, peer=1),
                dist.P2POp(dist.irecv, recv_tensor, peer=1),
            ]

            works = dist.batch_isend_irecv(p2p_ops)
            for work in works:
                work.wait()

            expected = torch.full_like(recv_tensor.cpu(), 88)
            torch.testing.assert_close(recv_tensor.cpu(), expected)

        elif rank == 1:
            send_tensor = torch.ones(1024, dtype=torch.float, device=device) * 88
            recv_tensor = torch.zeros(1024, dtype=torch.float, device=device)

            p2p_ops = [
                dist.P2POp(dist.irecv, recv_tensor, peer=0),
                dist.P2POp(dist.isend, send_tensor, peer=0),
            ]

            works = dist.batch_isend_irecv(p2p_ops)
            for work in works:
                work.wait()

            expected = torch.full_like(recv_tensor.cpu(), 77)
            torch.testing.assert_close(recv_tensor.cpu(), expected)

    def test_batch_isend_irecv_multiple_ops(self) -> None:
        """Test batch_isend_irecv with multiple send/recv operations."""
        rank = dist.get_rank()
        num_ranks = dist.get_world_size()
        device = get_device(rank)

        if num_ranks < 2:
            self.skipTest("Need at least 2 ranks for batch_isend_irecv test")

        if rank == 0:
            send_tensor1 = torch.ones(512, dtype=torch.float, device=device) * 11
            send_tensor2 = torch.ones(512, dtype=torch.float, device=device) * 22
            recv_tensor1 = torch.zeros(512, dtype=torch.float, device=device)
            recv_tensor2 = torch.zeros(512, dtype=torch.float, device=device)

            p2p_ops = [
                dist.P2POp(dist.isend, send_tensor1, peer=1),
                dist.P2POp(dist.isend, send_tensor2, peer=1),
                dist.P2POp(dist.irecv, recv_tensor1, peer=1),
                dist.P2POp(dist.irecv, recv_tensor2, peer=1),
            ]

            works = dist.batch_isend_irecv(p2p_ops)
            for work in works:
                work.wait()

            expected1 = torch.full_like(recv_tensor1.cpu(), 33)
            expected2 = torch.full_like(recv_tensor2.cpu(), 44)
            torch.testing.assert_close(recv_tensor1.cpu(), expected1)
            torch.testing.assert_close(recv_tensor2.cpu(), expected2)

        elif rank == 1:
            send_tensor1 = torch.ones(512, dtype=torch.float, device=device) * 33
            send_tensor2 = torch.ones(512, dtype=torch.float, device=device) * 44
            recv_tensor1 = torch.zeros(512, dtype=torch.float, device=device)
            recv_tensor2 = torch.zeros(512, dtype=torch.float, device=device)

            p2p_ops = [
                dist.P2POp(dist.irecv, recv_tensor1, peer=0),
                dist.P2POp(dist.irecv, recv_tensor2, peer=0),
                dist.P2POp(dist.isend, send_tensor1, peer=0),
                dist.P2POp(dist.isend, send_tensor2, peer=0),
            ]

            works = dist.batch_isend_irecv(p2p_ops)
            for work in works:
                work.wait()

            expected1 = torch.full_like(recv_tensor1.cpu(), 11)
            expected2 = torch.full_like(recv_tensor2.cpu(), 22)
            torch.testing.assert_close(recv_tensor1.cpu(), expected1)
            torch.testing.assert_close(recv_tensor2.cpu(), expected2)


if __name__ == "__main__":
    unittest.main()
