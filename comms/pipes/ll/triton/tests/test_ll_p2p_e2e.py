# Copyright (c) Meta Platforms, Inc. and affiliates.
# Confidential and proprietary.
# pyre-unsafe
"""
End-to-end integration tests for Triton LL P2P send/recv/forward.

Requires 8 NVLink-connected GPUs. Run with:
    buck2 test -c nccl.enable_pipes=false \
        -c fbcode.enable_gpu_sections=true \
        -c fbcode.platform010_cuda_version=12.8 \
        -c fbcode.nvcc_arch=h100a \
        -c hpc_comms.use_ncclx=stable \
        //comms/pipes/ll/triton/tests:test_ll_p2p_e2e
"""

from __future__ import annotations

import gc
import os
import time
import unittest

import torch
from torch.utils._triton import has_triton
from torchcomms.tests.integration.py.TorchCommTestHelpers import TorchCommTestWrapper


TRITON_AVAILABLE = has_triton()
RUN_DEVICE_API_TEST = os.environ.get("RUN_DEVICE_API_TEST", "false").lower() == "true"


def _skip_if_not_ready() -> bool:
    return TRITON_AVAILABLE and torch.cuda.is_available() and RUN_DEVICE_API_TEST


class LlP2pTestBase(unittest.TestCase):
    """Base class that sets up LlP2pOp once per test class."""

    wrapper = None
    op = None
    MAX_NBYTES = 65536

    @classmethod
    def setUpClass(cls) -> None:
        if not _skip_if_not_ready():
            raise unittest.SkipTest("E2E test environment not ready")
        from comms.pipes.ll.triton.ll_p2p_op import LlP2pOp

        torch.cuda.synchronize()
        cls.wrapper = TorchCommTestWrapper()
        cls.torchcomm = cls.wrapper.get_torchcomm()
        cls.rank = cls.torchcomm.get_rank()
        cls.world_size = cls.torchcomm.get_size()
        cls.device = cls.torchcomm.get_device()

        cls.op = LlP2pOp(
            comm=cls.torchcomm,
            max_nbytes=cls.MAX_NBYTES,
            device=cls.device,
        )
        cls.op.setup()

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.op is not None:
            cls.op.teardown()
            cls.op = None
        cls.torchcomm = None
        cls.wrapper = None
        gc.collect()
        torch.cuda.synchronize()
        time.sleep(2)


class TestBasicSendRecv(LlP2pTestBase):
    """Basic send/recv at various message sizes."""

    def _run_send_recv(self, nbytes: int) -> None:
        num_elements = nbytes // 4  # int32
        peer = 1 if self.rank == 0 else 0

        if self.rank == 0:
            src = torch.arange(num_elements, dtype=torch.int32, device=self.device)
            dst = torch.zeros(num_elements, dtype=torch.int32, device=self.device)
            self.op.send(peer=peer, src_tensor=src, nbytes=nbytes)
            torch.cuda.synchronize()
        elif self.rank == 1:
            src = torch.zeros(num_elements, dtype=torch.int32, device=self.device)
            dst = torch.zeros(num_elements, dtype=torch.int32, device=self.device)
            self.op.recv(peer=peer, dst_tensor=dst, nbytes=nbytes)
            torch.cuda.synchronize()

            expected = torch.arange(num_elements, dtype=torch.int32, device=self.device)
            torch.testing.assert_close(
                dst,
                expected,
                msg=f"Rank {self.rank}: Data mismatch for {nbytes}B send/recv",
            )
        else:
            return

    def test_8b(self) -> None:
        self._run_send_recv(8)

    def test_64b(self) -> None:
        self._run_send_recv(64)

    def test_256b(self) -> None:
        self._run_send_recv(256)

    def test_1kb(self) -> None:
        self._run_send_recv(1024)

    def test_4kb(self) -> None:
        self._run_send_recv(4096)

    def test_16kb(self) -> None:
        self._run_send_recv(16384)

    def test_64kb(self) -> None:
        self._run_send_recv(65536)


class TestBidirectional(LlP2pTestBase):
    """Both rank 0 and rank 1 send/recv simultaneously."""

    def test_bidirectional_1kb(self) -> None:
        if self.rank > 1:
            return

        nbytes = 1024
        num_elements = nbytes // 4
        peer = 1 if self.rank == 0 else 0

        src = torch.full(
            (num_elements,),
            self.rank + 1,
            dtype=torch.int32,
            device=self.device,
        )
        dst = torch.zeros(num_elements, dtype=torch.int32, device=self.device)

        self.op.send(peer=peer, src_tensor=src, nbytes=nbytes)
        self.op.recv(peer=peer, dst_tensor=dst, nbytes=nbytes)
        torch.cuda.synchronize()

        expected = torch.full(
            (num_elements,),
            peer + 1,
            dtype=torch.int32,
            device=self.device,
        )
        torch.testing.assert_close(
            dst,
            expected,
            msg=f"Rank {self.rank}: Bidirectional data mismatch",
        )


class TestMultiStep(LlP2pTestBase):
    """Multiple consecutive send/recv on the same buffer (ABA prevention)."""

    def test_10_steps(self) -> None:
        if self.rank > 1:
            return

        nbytes = 256
        num_elements = nbytes // 4
        peer = 1 if self.rank == 0 else 0

        for step in range(10):
            fill_value = (self.rank + 1) * 100 + step

            if self.rank == 0:
                src = torch.full(
                    (num_elements,),
                    fill_value,
                    dtype=torch.int32,
                    device=self.device,
                )
                self.op.send(peer=peer, src_tensor=src, nbytes=nbytes)
            else:
                dst = torch.zeros(num_elements, dtype=torch.int32, device=self.device)
                self.op.recv(peer=peer, dst_tensor=dst, nbytes=nbytes)
                torch.cuda.synchronize()

                expected_value = (peer + 1) * 100 + step
                expected = torch.full(
                    (num_elements,),
                    expected_value,
                    dtype=torch.int32,
                    device=self.device,
                )
                torch.testing.assert_close(
                    dst,
                    expected,
                    msg=f"Rank {self.rank}: Multi-step mismatch at step {step}",
                )


class TestVaryingData(LlP2pTestBase):
    """Different data patterns to catch corruption."""

    def _send_recv_pattern(self, src_data: torch.Tensor) -> None:
        nbytes = src_data.nbytes
        peer = 1 if self.rank == 0 else 0

        if self.rank == 0:
            self.op.send(peer=peer, src_tensor=src_data, nbytes=nbytes)
        elif self.rank == 1:
            dst = torch.zeros_like(src_data)
            self.op.recv(peer=peer, dst_tensor=dst, nbytes=nbytes)
            torch.cuda.synchronize()
            torch.testing.assert_close(dst, src_data.to(self.device))
        else:
            return

    def test_all_zeros(self) -> None:
        self._send_recv_pattern(torch.zeros(64, dtype=torch.int32, device=self.device))

    def test_all_ones(self) -> None:
        self._send_recv_pattern(torch.ones(64, dtype=torch.int32, device=self.device))

    def test_sequential(self) -> None:
        self._send_recv_pattern(
            torch.arange(128, dtype=torch.int32, device=self.device)
        )

    def test_large_values(self) -> None:
        self._send_recv_pattern(
            torch.full((64,), 0x7FFFFFFF, dtype=torch.int32, device=self.device)
        )


class TestZeroBytes(LlP2pTestBase):
    """Edge case: zero-byte send/recv should be a no-op."""

    def test_zero_bytes(self) -> None:
        if self.rank > 1:
            return

        peer = 1 if self.rank == 0 else 0
        src = torch.empty(0, dtype=torch.int32, device=self.device)
        dst = torch.empty(0, dtype=torch.int32, device=self.device)

        if self.rank == 0:
            self.op.send(peer=peer, src_tensor=src, nbytes=0)
        else:
            self.op.recv(peer=peer, dst_tensor=dst, nbytes=0)

        torch.cuda.synchronize()


class TestMultiPeer(LlP2pTestBase):
    """Rank 0 sends to all other ranks simultaneously."""

    def test_multi_peer_1kb(self) -> None:
        nbytes = 1024
        num_elements = nbytes // 4

        if self.rank == 0:
            for peer in range(1, self.world_size):
                src = torch.full(
                    (num_elements,),
                    peer * 10,
                    dtype=torch.int32,
                    device=self.device,
                )
                self.op.send(peer=peer, src_tensor=src, nbytes=nbytes)
            torch.cuda.synchronize()
        else:
            dst = torch.zeros(num_elements, dtype=torch.int32, device=self.device)
            self.op.recv(peer=0, dst_tensor=dst, nbytes=nbytes)
            torch.cuda.synchronize()

            expected = torch.full(
                (num_elements,),
                self.rank * 10,
                dtype=torch.int32,
                device=self.device,
            )
            torch.testing.assert_close(
                dst,
                expected,
                msg=f"Rank {self.rank}: Multi-peer data mismatch",
            )


@unittest.skipUnless(
    _skip_if_not_ready(),
    "Forward test requires E2E environment",
)
class TestForward(unittest.TestCase):
    """Forward test: rank 0 -> rank 1 -> rank 2 pipeline."""

    wrapper = None
    op = None

    @classmethod
    def setUpClass(cls) -> None:
        from comms.pipes.ll.triton.ll_p2p_op import LlP2pOp

        torch.cuda.synchronize()
        cls.wrapper = TorchCommTestWrapper()
        cls.torchcomm = cls.wrapper.get_torchcomm()
        cls.rank = cls.torchcomm.get_rank()
        cls.world_size = cls.torchcomm.get_size()
        cls.device = cls.torchcomm.get_device()

        if cls.world_size < 3:
            raise unittest.SkipTest("Forward test requires at least 3 GPUs")

        cls.op = LlP2pOp(
            comm=cls.torchcomm,
            max_nbytes=4096,
            device=cls.device,
        )
        cls.op.setup()

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.op is not None:
            cls.op.teardown()
            cls.op = None
        cls.torchcomm = None
        cls.wrapper = None
        gc.collect()
        torch.cuda.synchronize()
        time.sleep(2)

    def test_forward_1kb(self) -> None:
        nbytes = 1024
        num_elements = nbytes // 4

        if self.rank == 0:
            src = torch.arange(num_elements, dtype=torch.int32, device=self.device)
            self.op.send(peer=1, src_tensor=src, nbytes=nbytes)
            torch.cuda.synchronize()

        elif self.rank == 1:
            dst = torch.zeros(num_elements, dtype=torch.int32, device=self.device)
            self.op.forward(
                predecessor=0,
                successor=2,
                dst_tensor=dst,
                nbytes=nbytes,
            )
            torch.cuda.synchronize()

            expected = torch.arange(num_elements, dtype=torch.int32, device=self.device)
            torch.testing.assert_close(
                dst,
                expected,
                msg="Rank 1: Forward local copy mismatch",
            )

        elif self.rank == 2:
            dst = torch.zeros(num_elements, dtype=torch.int32, device=self.device)
            self.op.recv(peer=1, dst_tensor=dst, nbytes=nbytes)
            torch.cuda.synchronize()

            expected = torch.arange(num_elements, dtype=torch.int32, device=self.device)
            torch.testing.assert_close(
                dst,
                expected,
                msg="Rank 2: Forward receive mismatch",
            )
        else:
            return


class TestFusedSendRecv(LlP2pTestBase):
    """Fused sendrecv correctness tests."""

    def test_bidirectional_1kb(self) -> None:
        if self.rank > 1:
            return

        nbytes = 1024
        num_elements = nbytes // 4
        peer = 1 if self.rank == 0 else 0

        src = torch.full(
            (num_elements,),
            self.rank + 1,
            dtype=torch.int32,
            device=self.device,
        )
        dst = torch.zeros(num_elements, dtype=torch.int32, device=self.device)

        self.op.sendrecv(peer=peer, src_tensor=src, dst_tensor=dst, nbytes=nbytes)
        torch.cuda.synchronize()

        expected = torch.full(
            (num_elements,),
            peer + 1,
            dtype=torch.int32,
            device=self.device,
        )
        torch.testing.assert_close(
            dst,
            expected,
            msg=f"Rank {self.rank}: Fused sendrecv data mismatch",
        )

    def test_multiple_sizes(self) -> None:
        if self.rank > 1:
            return

        peer = 1 if self.rank == 0 else 0

        for nbytes in [8, 64, 256, 1024, 16384, 65536]:
            num_elements = nbytes // 4
            src = (
                torch.arange(num_elements, dtype=torch.int32, device=self.device)
                + self.rank * 1000
            )
            dst = torch.zeros(num_elements, dtype=torch.int32, device=self.device)

            self.op.sendrecv(peer=peer, src_tensor=src, dst_tensor=dst, nbytes=nbytes)
            torch.cuda.synchronize()

            expected = (
                torch.arange(num_elements, dtype=torch.int32, device=self.device)
                + peer * 1000
            )
            torch.testing.assert_close(
                dst,
                expected,
                msg=f"Rank {self.rank}: Fused sendrecv mismatch at {nbytes}B",
            )
