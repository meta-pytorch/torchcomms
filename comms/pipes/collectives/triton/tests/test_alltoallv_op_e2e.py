# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-unsafe
"""
End-to-end integration tests for AlltoallvOp (high-level MSL-compatible API).

Tests the simplified token-level API covering:
- Copy-in mode with uniform splits (alltoallv)
- Zero-copy mode via get_send_buffer + alltoallv_from_buffer
- Repeated calls with auto-incrementing iteration counter
- get_or_create() caching factory
- Error handling (double setup, alltoallv without setup, oversized input)

NOTE: Only uniform distribution is supported. Non-uniform distributions will
raise a ValueError.

Each test class contains a SINGLE test to avoid P2P mapping races
between tests sharing the same NCCL allocator.  register_local_buffer
maps GPU memory via ibv_reg_mr_iova2, and deregister_local_buffer's
unmap can race with the next test's torch.zeros (cudaMemset).

Run with:
    buck2 test @fbcode//mode/opt \\
        -c fbcode.enable_gpu_sections=true \\
        -c fbcode.platform010_cuda_version=12.8 \\
        -c fbcode.nvcc_arch=h100a \\
        -c hpc_comms.use_ncclx=stable \\
        fbcode//comms/torchcomms/triton/fb/tests:test_alltoallv_op_e2e
"""

import gc
import os
import sys
import time
import unittest
from typing import Optional

import torch
from torch.utils._triton import has_triton
from torchcomms.tests.integration.py.TorchCommTestHelpers import TorchCommTestWrapper


TRITON_AVAILABLE = has_triton()
RUN_DEVICE_API_TEST = os.environ.get("RUN_DEVICE_API_TEST", "false").lower() == "true"


def _skip_if_not_ready() -> bool:
    return TRITON_AVAILABLE and torch.cuda.is_available() and RUN_DEVICE_API_TEST


# =============================================================================
# Base Test Class
# =============================================================================


class _OpTestBase(unittest.TestCase):
    """Base class providing common helpers for AlltoallvOp tests."""

    wrapper: Optional[TorchCommTestWrapper] = None

    @classmethod
    def setUpClass(cls) -> None:
        if not _skip_if_not_ready():
            raise unittest.SkipTest("E2E test environment not ready")

        torch.cuda.synchronize()
        cls.wrapper = TorchCommTestWrapper()
        cls.torchcomm = cls.wrapper.get_torchcomm()
        cls.rank = cls.torchcomm.get_rank()
        cls.world_size = cls.torchcomm.get_size()
        cls.device = cls.torchcomm.get_device()
        cls.dtype = torch.float32
        cls.backend = cls.torchcomm.get_backend()

    @classmethod
    def tearDownClass(cls) -> None:
        from comms.pipes.collectives.triton import AlltoallvOp

        AlltoallvOp.clear_cache()
        if cls.torchcomm is not None:
            cls.torchcomm.barrier(False)
            cls.torchcomm.finalize()
        cls.torchcomm = None
        cls.wrapper = None
        gc.collect()
        torch.cuda.synchronize()
        time.sleep(2)

    def _verify_packed_data(
        self,
        output: torch.Tensor,
        D: int,
        output_split_sizes: torch.Tensor,
        test_name: str = "",
    ) -> None:
        """Verify data in packed contiguous layout.

        The output is contiguous: [peer_0_data, peer_1_data, ..., peer_{W-1}_data].
        Each element should be peer * 1000 + rank.
        """
        offset = 0
        for peer in range(self.world_size):
            count = int(output_split_sizes[peer].item())
            if count == 0:
                continue
            actual = output[offset : offset + count, :].cpu()
            expected_value = float(peer * 1000 + self.rank)
            expected = torch.full_like(actual, expected_value)
            torch.testing.assert_close(
                actual,
                expected,
                msg=(
                    f"[{test_name}] Rank {self.rank}: Data from peer {peer} at "
                    f"offset {offset} is incorrect. Expected {expected_value}, "
                    f"got {actual[0, :5].tolist()}..."
                ),
            )
            offset += count


# =============================================================================
# Test: Copy-In Mode with Uniform Splits
# =============================================================================


class TestOpUniformSplitsCopyIn(_OpTestBase):
    """Test alltoallv() copy-in path with uniform per-peer token counts."""

    def test_uniform_splits_copy_in(self) -> None:
        from comms.pipes.collectives.triton import AlltoallvOp

        tokens_per_peer = 64
        D = 16
        max_input_tokens = tokens_per_peer * self.world_size

        # Build the input tensor: tokens_for_peer_0 ++ tokens_for_peer_1 ++ …
        input_tensor = torch.empty(
            max_input_tokens, D, dtype=self.dtype, device=self.device
        )
        for peer in range(self.world_size):
            start = peer * tokens_per_peer
            input_tensor[start : start + tokens_per_peer] = float(
                self.rank * 1000 + peer
            )

        input_split_sizes = torch.full(
            (self.world_size,), tokens_per_peer, dtype=torch.int64, device=self.device
        )
        # Uniform: everyone sends the same amount → everyone receives the same.
        output_split_sizes = input_split_sizes.clone()

        op = AlltoallvOp(
            self.torchcomm,
            max_input_tokens,
            D,
            self.dtype,
            self.device,
            max_recv_tokens_per_peer=tokens_per_peer,
            sync_buffer=True,
        )
        with op:
            output = op.alltoallv(input_tensor, output_split_sizes, input_split_sizes)

        torch.cuda.synchronize()
        total_tokens = tokens_per_peer * self.world_size
        self.assertEqual(
            output.shape,
            (total_tokens, D),
        )
        self._verify_packed_data(output, D, output_split_sizes, "uniform_copy_in")


# =============================================================================
# Test: Zero-Copy Mode via get_send_buffer
# =============================================================================


class TestOpZeroCopy(_OpTestBase):
    """Test get_send_buffer() + alltoallv_from_buffer() zero-copy path."""

    def test_zero_copy_send(self) -> None:
        from comms.pipes.collectives.triton import AlltoallvOp

        tokens_per_peer = 64
        D = 16
        max_input_tokens = tokens_per_peer * self.world_size
        total_tokens = tokens_per_peer * self.world_size

        input_split_sizes = torch.full(
            (self.world_size,), tokens_per_peer, dtype=torch.int64, device=self.device
        )
        output_split_sizes = input_split_sizes.clone()

        op = AlltoallvOp(
            self.torchcomm,
            max_input_tokens,
            D,
            self.dtype,
            self.device,
            max_recv_tokens_per_peer=tokens_per_peer,
        )

        # Fill send buffer BEFORE setup — GIN blocks regular CUDA fill ops.
        send_buf = op.get_send_buffer(total_tokens)
        self.assertEqual(send_buf.shape, (total_tokens, D))
        for peer in range(self.world_size):
            start = peer * tokens_per_peer
            send_buf[start : start + tokens_per_peer] = float(self.rank * 1000 + peer)

        with op:
            output = op.alltoallv_from_buffer(
                output_split_sizes,
                input_split_sizes,
                num_input_tokens=total_tokens,
            )

        torch.cuda.synchronize()
        self.assertEqual(output.shape, (total_tokens, D))
        self._verify_packed_data(output, D, output_split_sizes, "zero_copy")


# =============================================================================
# Test: Repeated Calls (Iteration Counter)
# =============================================================================


class TestOpRepeatedCalls(_OpTestBase):
    """Test multiple consecutive alltoallv calls with iteration tracking."""

    def test_repeated_calls(self) -> None:
        from comms.pipes.collectives.triton import AlltoallvOp

        tokens_per_peer = 64
        D = 16
        max_input_tokens = tokens_per_peer * self.world_size
        num_iterations = 5

        input_tensor = torch.empty(
            max_input_tokens, D, dtype=self.dtype, device=self.device
        )
        for peer in range(self.world_size):
            start = peer * tokens_per_peer
            input_tensor[start : start + tokens_per_peer] = float(
                self.rank * 1000 + peer
            )

        input_split_sizes = torch.full(
            (self.world_size,), tokens_per_peer, dtype=torch.int64, device=self.device
        )
        output_split_sizes = input_split_sizes.clone()

        op = AlltoallvOp(
            self.torchcomm,
            max_input_tokens,
            D,
            self.dtype,
            self.device,
            max_recv_tokens_per_peer=tokens_per_peer,
        )
        with op:
            output = None
            for _ in range(num_iterations):
                output = op.alltoallv(
                    input_tensor, output_split_sizes, input_split_sizes
                )
            torch.cuda.synchronize()

        # Verify final output (teardown already happened via __exit__).
        self.torchcomm.barrier(False)
        total_tokens = tokens_per_peer * self.world_size
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (total_tokens, D))
        self._verify_packed_data(output, D, output_split_sizes, "repeated_calls")


# =============================================================================
# Test: get_or_create() Caching Factory
# =============================================================================


class TestOpGetOrCreate(_OpTestBase):
    """Test that get_or_create() returns a cached, ready-to-use op."""

    def test_get_or_create_returns_same_instance(self) -> None:
        from comms.pipes.collectives.triton import AlltoallvOp

        tokens_per_peer = 32
        D = 16
        max_input_tokens = tokens_per_peer * self.world_size

        # Create and fill input tensor BEFORE get_or_create (which calls
        # setup and activates GIN, blocking regular CUDA fill ops).
        input_tensor = torch.empty(
            max_input_tokens, D, dtype=self.dtype, device=self.device
        )
        for peer in range(self.world_size):
            start = peer * tokens_per_peer
            input_tensor[start : start + tokens_per_peer] = float(
                self.rank * 1000 + peer
            )

        input_split_sizes = torch.full(
            (self.world_size,), tokens_per_peer, dtype=torch.int64, device=self.device
        )
        output_split_sizes = input_split_sizes.clone()

        # First call creates and sets up the op.
        op1 = AlltoallvOp.get_or_create(
            self.torchcomm,
            max_input_tokens,
            D,
            self.dtype,
            self.device,
            max_recv_tokens_per_peer=tokens_per_peer,
        )
        # Second call should return the exact same object.
        op2 = AlltoallvOp.get_or_create(
            self.torchcomm,
            max_input_tokens,
            D,
            self.dtype,
            self.device,
            max_recv_tokens_per_peer=tokens_per_peer,
        )
        self.assertIs(op1, op2)

        # The cached op should be ready to use (no explicit setup needed).
        # alltoallv uses a Triton copy kernel internally (GIN-safe).
        output = op1.alltoallv(input_tensor, output_split_sizes, input_split_sizes)
        torch.cuda.synchronize()

        total_tokens = tokens_per_peer * self.world_size
        self.assertEqual(output.shape, (total_tokens, D))
        self._verify_packed_data(output, D, output_split_sizes, "get_or_create")

        # Cleanup: clear cache so tearDownClass can proceed cleanly.
        AlltoallvOp.clear_cache()


# =============================================================================
# Test: Error Handling
# =============================================================================


class TestOpDoubleSetupRaises(_OpTestBase):
    """Test that calling setup() twice without teardown() raises."""

    def test_double_setup_raises(self) -> None:
        from comms.pipes.collectives.triton import AlltoallvOp

        D = 16
        max_input_tokens = 64
        tokens_per_peer = max_input_tokens // self.world_size

        op = AlltoallvOp(
            self.torchcomm,
            max_input_tokens,
            D,
            self.dtype,
            self.device,
            max_recv_tokens_per_peer=tokens_per_peer,
        )
        op.setup()
        with self.assertRaises(RuntimeError):
            op.setup()
        # Sync all ranks BEFORE teardown to avoid race conditions where one
        # rank tears down while another is still using shared NCCL resources.
        torch.cuda.synchronize()
        self.torchcomm.barrier(False)
        op.teardown()


class TestOpAlltoallvWithoutSetupRaises(_OpTestBase):
    """Test that calling alltoallv() without setup() raises."""

    def test_alltoallv_without_setup_raises(self) -> None:
        from comms.pipes.collectives.triton import AlltoallvOp

        D = 16
        max_input_tokens = 64
        tokens_per_peer = max_input_tokens // self.world_size

        op = AlltoallvOp(
            self.torchcomm,
            max_input_tokens,
            D,
            self.dtype,
            self.device,
            max_recv_tokens_per_peer=tokens_per_peer,
        )
        input_tensor = torch.empty(
            max_input_tokens, D, dtype=self.dtype, device=self.device
        )
        split_sizes = torch.full(
            (self.world_size,),
            max_input_tokens // self.world_size,
            dtype=torch.int64,
            device=self.device,
        )

        with self.assertRaises(RuntimeError):
            op.alltoallv(input_tensor, split_sizes, split_sizes)

        # Sync all ranks BEFORE teardown to avoid race conditions where one
        # rank tears down while another is still using shared NCCL resources.
        torch.cuda.synchronize()
        self.torchcomm.barrier(False)
        op.teardown()


class TestOpOversizedInputRaises(_OpTestBase):
    """Test that passing an input exceeding max_input_tokens raises."""

    def test_oversized_input_raises(self) -> None:
        from comms.pipes.collectives.triton import AlltoallvOp

        D = 16
        max_input_tokens = 128
        tokens_per_peer = max_input_tokens // self.world_size

        # Oversized by 1 token
        oversized_input = torch.empty(
            max_input_tokens + 1, D, dtype=self.dtype, device=self.device
        )
        split_sizes = torch.full(
            (self.world_size,),
            tokens_per_peer,
            dtype=torch.int64,
            device=self.device,
        )

        op = AlltoallvOp(
            self.torchcomm,
            max_input_tokens,
            D,
            self.dtype,
            self.device,
            max_recv_tokens_per_peer=tokens_per_peer,
        )
        op.setup()

        with self.assertRaises(ValueError):
            op.alltoallv(oversized_input, split_sizes, split_sizes)

        # Sync all ranks BEFORE teardown to avoid race conditions where one
        # rank tears down while another is still using shared NCCL resources.
        torch.cuda.synchronize()
        self.torchcomm.barrier(False)
        op.teardown()


class TestOpOversizedGetSendBufferRaises(_OpTestBase):
    """Test that get_send_buffer() with too many tokens raises."""

    def test_oversized_get_send_buffer_raises(self) -> None:
        from comms.pipes.collectives.triton import AlltoallvOp

        D = 16
        max_input_tokens = 64
        tokens_per_peer = max_input_tokens // self.world_size

        op = AlltoallvOp(
            self.torchcomm,
            max_input_tokens,
            D,
            self.dtype,
            self.device,
            max_recv_tokens_per_peer=tokens_per_peer,
        )
        with self.assertRaises(ValueError):
            op.get_send_buffer(max_input_tokens + 1)

        # Sync all ranks BEFORE teardown to avoid race conditions where one
        # rank tears down while another is still using shared NCCL resources.
        # Even though setup() was not called, the op still holds buffers
        # that must be released cleanly.
        torch.cuda.synchronize()
        self.torchcomm.barrier(False)
        op.teardown()


class TestOpTeardownIdempotent(_OpTestBase):
    """Test that teardown() can be called multiple times safely."""

    def test_teardown_idempotent(self) -> None:
        from comms.pipes.collectives.triton import AlltoallvOp

        D = 16
        max_input_tokens = 64
        tokens_per_peer = max_input_tokens // self.world_size

        op = AlltoallvOp(
            self.torchcomm,
            max_input_tokens,
            D,
            self.dtype,
            self.device,
            max_recv_tokens_per_peer=tokens_per_peer,
        )
        op.setup()
        op.teardown()
        op.teardown()  # Should not raise
        # Sync all ranks before test ends to avoid cleanup race conditions.
        torch.cuda.synchronize()
        self.torchcomm.barrier(False)


# =============================================================================
# Test: Packed Output Mode - Uniform Splits
# =============================================================================


class TestOpPackedOutputUniform(_OpTestBase):
    """Test alltoallv() with packed_output mode (now the default) and uniform splits."""

    def test_packed_output_uniform(self) -> None:
        from comms.pipes.collectives.triton import AlltoallvOp

        tokens_per_peer = 64
        D = 16
        max_input_tokens = tokens_per_peer * self.world_size

        # Build input tensor
        input_tensor = torch.empty(
            max_input_tokens, D, dtype=self.dtype, device=self.device
        )
        for peer in range(self.world_size):
            start = peer * tokens_per_peer
            input_tensor[start : start + tokens_per_peer] = float(
                self.rank * 1000 + peer
            )

        input_split_sizes = torch.full(
            (self.world_size,), tokens_per_peer, dtype=torch.int64, device=self.device
        )
        output_split_sizes = input_split_sizes.clone()

        # Pre-compute total tokens to avoid .item() call when GIN is active
        packed_output_tokens = tokens_per_peer * self.world_size

        # Use packed_output mode (now the default) for contiguous output
        op = AlltoallvOp(
            self.torchcomm,
            max_input_tokens,
            D,
            self.dtype,
            self.device,
            max_recv_tokens_per_peer=tokens_per_peer,
        )
        with op:
            output = op.alltoallv(
                input_tensor,
                output_split_sizes,
                input_split_sizes,
                packed_output_tokens=packed_output_tokens,
            )

        torch.cuda.synchronize()

        # Verify packed output shape
        self.assertEqual(output.shape, (packed_output_tokens, D))

        # Verify packed output data
        self._verify_packed_data(output, D, output_split_sizes, "packed_uniform")

    def _verify_packed_data(
        self,
        output: torch.Tensor,
        D: int,
        output_split_sizes: torch.Tensor,
        test_name: str = "",
    ) -> None:
        """Verify data in packed contiguous layout."""
        offset = 0
        for peer in range(self.world_size):
            count = int(output_split_sizes[peer].item())
            if count == 0:
                continue
            actual = output[offset : offset + count, :].cpu()
            expected_value = float(peer * 1000 + self.rank)
            expected = torch.full_like(actual, expected_value)
            torch.testing.assert_close(
                actual,
                expected,
                msg=(
                    f"[{test_name}] Rank {self.rank}: Data from peer {peer} at "
                    f"offset {offset} is incorrect. Expected {expected_value}, "
                    f"got {actual[0, :5].tolist()}..."
                ),
            )
            offset += count


# =============================================================================
# Test: fill_send_buffer() GIN-Safe Buffer Update
# =============================================================================


class TestOpFillSendBufferBasic(_OpTestBase):
    """Test fill_send_buffer() basic functionality."""

    def test_fill_send_buffer_basic(self) -> None:
        """Test that fill_send_buffer works after GIN is active."""
        from comms.pipes.collectives.triton import AlltoallvOp

        tokens_per_peer = 64
        D = 16
        max_input_tokens = tokens_per_peer * self.world_size
        total_tokens = tokens_per_peer * self.world_size

        # Pre-allocate input tensor BEFORE context (GIN blocks torch.empty)
        input_tensor = torch.empty(
            total_tokens, D, dtype=self.dtype, device=self.device
        )
        for peer in range(self.world_size):
            start = peer * tokens_per_peer
            input_tensor[start : start + tokens_per_peer] = float(
                self.rank * 1000 + peer
            )

        input_split_sizes = torch.full(
            (self.world_size,), tokens_per_peer, dtype=torch.int64, device=self.device
        )
        output_split_sizes = input_split_sizes.clone()

        op = AlltoallvOp(
            self.torchcomm,
            max_input_tokens,
            D,
            self.dtype,
            self.device,
            max_recv_tokens_per_peer=tokens_per_peer,
        )

        with op:
            # Use fill_send_buffer (GIN-safe) instead of direct assignment
            send_view = op.fill_send_buffer(input_tensor)
            self.assertEqual(send_view.shape, (total_tokens, D))

            output = op.alltoallv_from_buffer(
                output_split_sizes,
                input_split_sizes,
                num_input_tokens=total_tokens,
            )

        torch.cuda.synchronize()
        self.assertEqual(output.shape, (total_tokens, D))
        self._verify_packed_data(
            output, D, output_split_sizes, "fill_send_buffer_basic"
        )


class TestOpFillSendBufferBackToBack(_OpTestBase):
    """Test fill_send_buffer() for back-to-back iterations."""

    def test_fill_send_buffer_back_to_back(self) -> None:
        """Test multiple iterations with fill_send_buffer inside 'with op:'."""
        from comms.pipes.collectives.triton import AlltoallvOp

        tokens_per_peer = 32
        D = 16
        max_input_tokens = tokens_per_peer * self.world_size
        total_tokens = tokens_per_peer * self.world_size
        num_iterations = 5

        input_split_sizes = torch.full(
            (self.world_size,), tokens_per_peer, dtype=torch.int64, device=self.device
        )
        output_split_sizes = input_split_sizes.clone()

        # Pre-allocate input tensor BEFORE entering context
        input_tensor = torch.empty(
            total_tokens, D, dtype=self.dtype, device=self.device
        )
        for peer in range(self.world_size):
            start = peer * tokens_per_peer
            input_tensor[start : start + tokens_per_peer] = float(
                self.rank * 1000 + peer
            )

        op = AlltoallvOp(
            self.torchcomm,
            max_input_tokens,
            D,
            self.dtype,
            self.device,
            max_recv_tokens_per_peer=tokens_per_peer,
        )

        output = None
        with op:
            for _iteration in range(num_iterations):
                # Use fill_send_buffer to copy into registered buffer (GIN-safe)
                op.fill_send_buffer(input_tensor, num_tokens=total_tokens)

                output = op.alltoallv_from_buffer(
                    output_split_sizes,
                    input_split_sizes,
                    num_input_tokens=total_tokens,
                )
        assert output is not None

        torch.cuda.synchronize()
        self.assertEqual(output.shape, (total_tokens, D))
        self._verify_packed_data(
            output,
            D,
            output_split_sizes,
            "fill_send_buffer_back_to_back",
        )


class TestOpFillSendBufferOversizedRaises(_OpTestBase):
    """Test that fill_send_buffer raises for oversized input."""

    def test_fill_send_buffer_oversized_raises(self) -> None:
        from comms.pipes.collectives.triton import AlltoallvOp

        D = 16
        max_input_tokens = 64
        tokens_per_peer = max_input_tokens // self.world_size

        # Create oversized input BEFORE context
        oversized_input = torch.ones(
            max_input_tokens + 10, D, dtype=self.dtype, device=self.device
        )

        op = AlltoallvOp(
            self.torchcomm,
            max_input_tokens,
            D,
            self.dtype,
            self.device,
            max_recv_tokens_per_peer=tokens_per_peer,
        )

        with op:
            with self.assertRaises(ValueError):
                op.fill_send_buffer(oversized_input)


class TestOpFillSendBufferWithExplicitNumTokens(_OpTestBase):
    """Test fill_send_buffer with explicit num_tokens parameter."""

    def test_fill_send_buffer_explicit_num_tokens(self) -> None:
        """Test fill_send_buffer with smaller num_tokens than input size."""
        from comms.pipes.collectives.triton import AlltoallvOp

        tokens_per_peer = 32
        D = 16
        max_input_tokens = tokens_per_peer * self.world_size
        # Use fewer tokens than allocated
        actual_tokens = tokens_per_peer * self.world_size // 2

        # Pre-allocate larger buffer
        input_tensor = torch.empty(
            max_input_tokens, D, dtype=self.dtype, device=self.device
        )
        # Fill only the portion we'll use
        half_tokens_per_peer = tokens_per_peer // 2
        for peer in range(self.world_size):
            start = peer * half_tokens_per_peer
            input_tensor[start : start + half_tokens_per_peer] = float(
                self.rank * 1000 + peer
            )

        input_split_sizes = torch.full(
            (self.world_size,),
            half_tokens_per_peer,
            dtype=torch.int64,
            device=self.device,
        )
        output_split_sizes = input_split_sizes.clone()

        op = AlltoallvOp(
            self.torchcomm,
            max_input_tokens,
            D,
            self.dtype,
            self.device,
            max_recv_tokens_per_peer=half_tokens_per_peer,
        )

        with op:
            # Explicitly specify num_tokens to copy less than full input
            send_view = op.fill_send_buffer(input_tensor, num_tokens=actual_tokens)
            self.assertEqual(send_view.shape, (actual_tokens, D))

            output = op.alltoallv_from_buffer(
                output_split_sizes,
                input_split_sizes,
                num_input_tokens=actual_tokens,
            )

        torch.cuda.synchronize()
        actual_recv_tokens = actual_tokens  # Same as sent in uniform distribution
        self.assertEqual(output.shape, (actual_recv_tokens, D))
        self._verify_packed_data(
            output,
            D,
            output_split_sizes,
            "fill_send_buffer_explicit_num_tokens",
        )


# =============================================================================
# Main
# =============================================================================

# =============================================================================
# Test: Multi-Iteration with Different Input Content (Graph & Non-Graph)
# =============================================================================


class TestOpMultiIterDifferentContentPackedNonGraph(_OpTestBase):
    """Test zero-copy mode with varying content across iterations (packed output).

    Similar to TestOpMultiIterDifferentContentPackedGraph but WITHOUT graph capture.
    Uses op.get_send_buffer() to get the internal send buffer, then fills it
    with different data for each iteration. Packed output mode.
    Cross-rank synchronization is used.
    """

    def test_multi_iter_different_content_packed_non_graph(self) -> None:
        from comms.pipes.collectives.triton import AlltoallvOp

        tokens_per_peer = 32
        D = 16
        max_input_tokens = tokens_per_peer * self.world_size
        num_iterations = 3
        packed_output_tokens = tokens_per_peer * self.world_size

        input_split_sizes = torch.full(
            (self.world_size,), tokens_per_peer, dtype=torch.int64, device=self.device
        )
        output_split_sizes = input_split_sizes.clone()

        # Use a dedicated stream for operations (consistent with graph mode tests)
        op_stream = torch.cuda.Stream()

        # proper cross-rank synchronization via the BUFFER_READY signal protocol.
        op = AlltoallvOp(
            self.torchcomm,
            max_input_tokens,
            D,
            self.dtype,
            self.device,
            max_recv_tokens_per_peer=tokens_per_peer,
            sync_buffer=True,  # Required for multi-iteration
        )

        with op:
            # Get send buffer for zero-copy writes
            send_buf = op.get_send_buffer(max_input_tokens)

            # Pre-allocate buffers to capture output state after each iteration.
            # This avoids validation racing with the next iteration's buffer updates.
            validation_buffers = []

            for iteration in range(num_iterations):
                with torch.cuda.stream(op_stream):
                    # Fill send buffer with iteration-specific content
                    for peer in range(self.world_size):
                        start = peer * tokens_per_peer
                        value = float(iteration * 10000 + self.rank * 1000 + peer)
                        send_buf[start : start + tokens_per_peer] = value

                    output = op.alltoallv_from_buffer(
                        output_split_sizes,
                        input_split_sizes,
                        num_input_tokens=max_input_tokens,
                        packed_output_tokens=packed_output_tokens,
                    )

                    # Clone output on op_stream to capture this iteration's data
                    # before the next iteration overwrites it
                    validation_buffers.append(output.clone())

            # Wait for all operations to complete
            op_stream.synchronize()

            # Now validate all iterations on CPU (safe since all GPU work is done)
            for iteration in range(num_iterations):
                output_snapshot = validation_buffers[iteration]

                # Verify packed output shape
                self.assertEqual(output_snapshot.shape, (packed_output_tokens, D))

                # Verify packed output data
                offset = 0
                for peer in range(self.world_size):
                    count = int(output_split_sizes[peer].item())
                    actual = output_snapshot[offset : offset + count, :].cpu()
                    expected_value = float(iteration * 10000 + peer * 1000 + self.rank)
                    expected = torch.full_like(actual, expected_value)
                    torch.testing.assert_close(
                        actual,
                        expected,
                        msg=(
                            f"Zero-copy packed iter {iteration}, Rank {self.rank}: Data "
                            f"from peer {peer} incorrect. Expected {expected_value}, "
                            f"got {actual[0, 0].item()}"
                        ),
                    )
                    offset += count


class TestOpMultiIterDifferentContentPackedNonGraphCopyIn(_OpTestBase):
    """Test multiple iterations with different input content per iteration (packed, non-graph)."""

    def test_multi_iter_different_content_packed_non_graph(self) -> None:
        """Test packed output with different input content per iteration (non-graph mode).

        This test verifies that packed output mode correctly handles different
        data per iteration. Cross-rank synchronization ensures proper
        cross-rank synchronization between iterations.
        """
        from comms.pipes.collectives.triton import AlltoallvOp

        tokens_per_peer = 32
        D = 16
        max_input_tokens = tokens_per_peer * self.world_size
        num_iterations = 3
        packed_output_tokens = tokens_per_peer * self.world_size

        input_split_sizes = torch.full(
            (self.world_size,), tokens_per_peer, dtype=torch.int64, device=self.device
        )
        output_split_sizes = input_split_sizes.clone()

        # Use a dedicated stream for operations (consistent with graph mode tests)
        op_stream = torch.cuda.Stream()

        # proper cross-rank synchronization via the BUFFER_READY signal protocol.
        op = AlltoallvOp(
            self.torchcomm,
            max_input_tokens,
            D,
            self.dtype,
            self.device,
            max_recv_tokens_per_peer=tokens_per_peer,
            sync_buffer=True,  # Required for multi-iteration
        )

        with op:
            # Pre-allocate buffers to capture output state after each iteration.
            # This avoids validation racing with the next iteration's buffer updates.
            validation_buffers = []

            for iteration in range(num_iterations):
                with torch.cuda.stream(op_stream):
                    # Build unique input tensor for this iteration
                    input_tensor = torch.empty(
                        max_input_tokens, D, dtype=self.dtype, device=self.device
                    )
                    for peer in range(self.world_size):
                        start = peer * tokens_per_peer
                        value = float(iteration * 10000 + self.rank * 1000 + peer)
                        input_tensor[start : start + tokens_per_peer] = value

                    output = op.alltoallv(
                        input_tensor,
                        output_split_sizes,
                        input_split_sizes,
                        packed_output_tokens=packed_output_tokens,
                    )

                    # Clone output on op_stream to capture this iteration's data
                    # before the next iteration overwrites it
                    validation_buffers.append(output.clone())

            # Wait for all operations to complete
            op_stream.synchronize()

            # Now validate all iterations on CPU (safe since all GPU work is done)
            for iteration in range(num_iterations):
                output_snapshot = validation_buffers[iteration]

                # Verify packed output shape
                self.assertEqual(output_snapshot.shape, (packed_output_tokens, D))

                # Verify packed output data
                offset = 0
                for peer in range(self.world_size):
                    count = int(output_split_sizes[peer].item())
                    actual = output_snapshot[offset : offset + count, :].cpu()
                    expected_value = float(iteration * 10000 + peer * 1000 + self.rank)
                    expected = torch.full_like(actual, expected_value)
                    torch.testing.assert_close(
                        actual,
                        expected,
                        msg=(
                            f"Packed iter {iteration}, Rank {self.rank}: Data from peer "
                            f"{peer} incorrect. Expected {expected_value}, got "
                            f"{actual[0, 0].item()}"
                        ),
                    )
                    offset += count


class TestOpMultiIterDifferentContentPackedGraph(_OpTestBase):
    """Test CUDA graph replay with varying input data per iteration (packed output).

    This test verifies the vLLM production use case (piecewise CUDA graphs):
    - Capture a SINGLE graph with one alltoallv call
    - Update input buffer OUTSIDE the graph before each replay
    - Replay the same graph multiple times with different data
    - Validate only the final iteration's output (CUDA graph internal
      synchronization for intermediate snapshots has known limitations)
    """

    def test_multi_iter_different_content_packed_graph(self) -> None:
        """Test packed output with single graph replayed with different data.

        vLLM-style approach:
        1. Pre-stage input data for all iterations
        2. Capture a SINGLE graph with one alltoallv call
        3. Before each replay: copy staged input to send buffer (outside graph)
        4. Replay the same graph - it sees the updated buffer contents
        5. Verify final output data matches expected values
        """
        from comms.pipes.collectives.triton import AlltoallvOp

        tokens_per_peer = 32
        D = 16
        max_input_tokens = tokens_per_peer * self.world_size
        num_iterations = 3
        packed_output_tokens = tokens_per_peer * self.world_size

        input_split_sizes = torch.full(
            (self.world_size,), tokens_per_peer, dtype=torch.int64, device=self.device
        )
        output_split_sizes = input_split_sizes.clone()

        op = AlltoallvOp(
            self.torchcomm,
            max_input_tokens,
            D,
            self.dtype,
            self.device,
            max_recv_tokens_per_peer=tokens_per_peer,
            sync_buffer=True,  # Required for multi-iteration CUDA graph replays
        )

        with op:
            # Get send buffer for zero-copy writes
            send_buf = op.get_send_buffer(max_input_tokens)

            # Pre-stage input data for ALL iterations before graph capture
            staged_inputs = []
            for iteration in range(num_iterations):
                staged_input = torch.empty(
                    max_input_tokens, D, dtype=self.dtype, device=self.device
                )
                for peer in range(self.world_size):
                    start = peer * tokens_per_peer
                    value = float(iteration * 10000 + self.rank * 1000 + peer)
                    staged_input[start : start + tokens_per_peer] = value
                staged_inputs.append(staged_input)

            # Initialize send buffer with first iteration's data for warmup
            send_buf.copy_(staged_inputs[0])

            # Warmup (compile Triton kernels before graph capture)
            op.alltoallv_from_buffer(
                output_split_sizes,
                input_split_sizes,
                num_input_tokens=max_input_tokens,
                packed_output_tokens=packed_output_tokens,
            )
            torch.cuda.synchronize()

            # Capture a SINGLE graph with alltoallv
            graph_stream = torch.cuda.Stream()
            with torch.cuda.stream(graph_stream):
                graph = torch.cuda.CUDAGraph()
                # pyre-fixme[6]: Pyre doesn't recognize pool ID as valid _POOL_HANDLE
                with torch.cuda.graph(graph, pool=op.get_graph_pool_id()):  # type: ignore[arg-type]
                    graph_output = op.alltoallv_from_buffer(
                        output_split_sizes,
                        input_split_sizes,
                        num_input_tokens=max_input_tokens,
                        packed_output_tokens=packed_output_tokens,
                    )
            torch.cuda.synchronize()

            # Replay the SAME graph multiple times with different input data
            # and validate each iteration
            for iteration in range(num_iterations):
                with torch.cuda.stream(graph_stream):
                    send_buf.copy_(staged_inputs[iteration])
                    graph.replay()

                # Wait for this iteration to complete before validating
                graph_stream.synchronize()

                # Clone output for this iteration
                iter_output = graph_output.clone()

                # Verify packed output shape
                self.assertEqual(iter_output.shape, (packed_output_tokens, D))

                # Verify packed data for this iteration
                offset = 0
                for peer in range(self.world_size):
                    count = int(output_split_sizes[peer].item())
                    actual = iter_output[offset : offset + count, :].cpu()
                    expected_value = float(iteration * 10000 + peer * 1000 + self.rank)
                    expected = torch.full_like(actual, expected_value)
                    torch.testing.assert_close(
                        actual,
                        expected,
                        msg=(
                            f"Graph iteration {iteration}, Rank {self.rank}: "
                            f"Packed data from peer {peer} incorrect. "
                            f"Expected {expected_value}, got {actual[0, 0].item()}"
                        ),
                    )
                    offset += count


class TestOpMultiIterDifferentContentPackedGraphCopyIn(_OpTestBase):
    """Test CUDA graph replay with varying input data per iteration (packed, copy-in).

    This test verifies the vLLM production use case (piecewise CUDA graphs):
    - Capture a SINGLE graph with one alltoallv call
    - Update input tensor OUTSIDE the graph before each replay
    - Replay the same graph multiple times with different data
    - Validate only the final iteration's output (CUDA graph internal
      synchronization for intermediate snapshots has known limitations)
    """

    def test_multi_iter_different_content_packed_graph_copy_in(self) -> None:
        """Test packed output with single graph replayed with different data (copy-in).

        vLLM-style approach with copy-in API:
        1. Pre-stage input data for all iterations
        2. Capture a SINGLE graph with one alltoallv call
        3. Before each replay: copy staged input to input tensor (outside graph)
        4. Replay the same graph - it sees the updated tensor contents
        5. Verify final output data matches expected values
        """
        from comms.pipes.collectives.triton import AlltoallvOp

        tokens_per_peer = 32
        D = 16
        max_input_tokens = tokens_per_peer * self.world_size
        num_iterations = 3
        packed_output_tokens = tokens_per_peer * self.world_size

        input_split_sizes = torch.full(
            (self.world_size,), tokens_per_peer, dtype=torch.int64, device=self.device
        )
        output_split_sizes = input_split_sizes.clone()

        op = AlltoallvOp(
            self.torchcomm,
            max_input_tokens,
            D,
            self.dtype,
            self.device,
            max_recv_tokens_per_peer=tokens_per_peer,
            sync_buffer=True,  # Required for multi-iteration CUDA graph replays
        )

        with op:
            # Create a persistent input tensor that will be reused
            input_tensor = torch.empty(
                max_input_tokens, D, dtype=self.dtype, device=self.device
            )

            # Pre-stage input data for ALL iterations before graph capture
            staged_inputs = []
            for iteration in range(num_iterations):
                staged_input = torch.empty(
                    max_input_tokens, D, dtype=self.dtype, device=self.device
                )
                for peer in range(self.world_size):
                    start = peer * tokens_per_peer
                    value = float(iteration * 10000 + self.rank * 1000 + peer)
                    staged_input[start : start + tokens_per_peer] = value
                staged_inputs.append(staged_input)

            # Initialize input tensor with first iteration's data for warmup
            input_tensor.copy_(staged_inputs[0])

            # Warmup (compile Triton kernels before graph capture)
            op.alltoallv(
                input_tensor,
                output_split_sizes,
                input_split_sizes,
                packed_output_tokens=packed_output_tokens,
            )
            torch.cuda.synchronize()

            # Capture a SINGLE graph with alltoallv
            graph_stream = torch.cuda.Stream()
            with torch.cuda.stream(graph_stream):
                graph = torch.cuda.CUDAGraph()
                # pyre-fixme[6]: Pyre doesn't recognize pool ID as valid _POOL_HANDLE
                with torch.cuda.graph(graph, pool=op.get_graph_pool_id()):  # type: ignore[arg-type]
                    graph_output = op.alltoallv(
                        input_tensor,
                        output_split_sizes,
                        input_split_sizes,
                        packed_output_tokens=packed_output_tokens,
                    )
            torch.cuda.synchronize()

            # Replay the SAME graph multiple times with different input data
            # and validate each iteration
            for iteration in range(num_iterations):
                with torch.cuda.stream(graph_stream):
                    input_tensor.copy_(staged_inputs[iteration])
                    graph.replay()

                # Wait for this iteration to complete before validating
                graph_stream.synchronize()

                # Clone output for this iteration
                iter_output = graph_output.clone()

                # Verify packed output shape
                self.assertEqual(iter_output.shape, (packed_output_tokens, D))

                # Verify packed data for this iteration
                offset = 0
                for peer in range(self.world_size):
                    count = int(output_split_sizes[peer].item())
                    actual = iter_output[offset : offset + count, :].cpu()
                    expected_value = float(iteration * 10000 + peer * 1000 + self.rank)
                    expected = torch.full_like(actual, expected_value)
                    torch.testing.assert_close(
                        actual,
                        expected,
                        msg=(
                            f"Graph iteration {iteration}, Rank {self.rank}: "
                            f"Packed data from peer {peer} incorrect. "
                            f"Expected {expected_value}, got {actual[0, 0].item()}"
                        ),
                    )
                    offset += count


def _update_send_buffer_kernel(
    send_buf: torch.Tensor,
    loop_index: torch.Tensor,
    base_value: float,
    tokens_per_peer: int,
    world_size: int,
) -> None:
    """Simple kernel to update send buffer with loop-iteration-specific values.

    This kernel is captured in the CUDA graph along with the alltoallv call.
    On each iteration of the captured loop, it reads the loop_index tensor
    (which was filled with a different constant for each iteration during capture)
    and updates the send buffer accordingly.

    IMPORTANT: This function must be CUDA graph-compatible. We cannot use
    .item() or any host-CPU sync operations inside graph capture. Instead,
    we use pure tensor operations that stay on the GPU.

    Args:
        send_buf: The send buffer to update (shape: [max_tokens, D])
        loop_index: Tensor containing the current loop iteration index
        base_value: Base value (typically rank * 1000)
        tokens_per_peer: Number of tokens per peer
        world_size: Number of peers
    """
    # Compute iteration-specific offset using GPU tensor ops (no .item()!)
    # iter_offset = loop_index * 10000 (stays on GPU)
    iter_offset = loop_index * 10000

    # Update buffer with iteration-specific values using broadcasting
    # Value format: loop_iter * 10000 + rank * 1000 + peer
    # We use tensor operations that are CUDA graph-compatible
    for peer in range(world_size):
        start = peer * tokens_per_peer
        end = start + tokens_per_peer
        # Use iter_offset tensor (on GPU) + scalars for base_value and peer
        # The tensor + scalar operations are graph-compatible
        value = iter_offset.float() + base_value + peer
        send_buf[start:end, :] = value


class TestOpMultiIterDifferentContentPackedGraphLoop(_OpTestBase):
    """Test multi-iteration with loop captured in graph (packed output).

    Validates only the final iteration's output per replay (CUDA graph internal
    synchronization for intermediate snapshots has known limitations).
    """

    def test_multi_iter_different_content_packed_graph_loop(self) -> None:
        """Test packed output with iteration loop captured in CUDA graph.

        Unlike the regular graph test where we capture one iteration and replay
        multiple times, here we capture the entire iteration loop in the graph
        and replay the loop as a whole.

        Each iteration within the captured loop sees DIFFERENT data because we
        include a compute kernel that updates the send buffer based on a
        loop_index tensor. The loop_index.fill_(i) operations are captured
        with different constant values for each iteration.

        This tests the scenario where multiple alltoallv calls with varying
        data are captured in a single CUDA graph with packed output mode.
        Only validates the final loop iteration's output per replay.
        """
        from comms.pipes.collectives.triton import AlltoallvOp

        tokens_per_peer = 32
        D = 16
        max_input_tokens = tokens_per_peer * self.world_size
        num_iterations_in_loop = 3  # Number of iterations captured in the loop
        packed_output_tokens = tokens_per_peer * self.world_size

        input_split_sizes = torch.full(
            (self.world_size,), tokens_per_peer, dtype=torch.int64, device=self.device
        )
        output_split_sizes = input_split_sizes.clone()

        op = AlltoallvOp(
            self.torchcomm,
            max_input_tokens,
            D,
            self.dtype,
            self.device,
            max_recv_tokens_per_peer=tokens_per_peer,
            sync_buffer=True,  # Required for multi-iteration CUDA graph replays
        )

        with op:
            # Get send buffer for zero-copy writes
            send_buf = op.get_send_buffer(max_input_tokens)

            # Tensor to hold the current loop index (updated inside the captured loop)
            loop_index = torch.zeros(1, dtype=torch.int64, device=self.device)

            # Base value for this rank (rank * 1000)
            base_value = float(self.rank * 1000)

            # Warmup (compile Triton kernels before graph capture)
            _update_send_buffer_kernel(
                send_buf, loop_index, base_value, tokens_per_peer, self.world_size
            )
            op.alltoallv_from_buffer(
                output_split_sizes,
                input_split_sizes,
                num_input_tokens=max_input_tokens,
                packed_output_tokens=packed_output_tokens,
            )
            torch.cuda.synchronize()

            # Capture the entire iteration loop in a single graph.
            # We only keep the final iteration's output for validation.
            graph_stream = torch.cuda.Stream()

            with torch.cuda.stream(graph_stream):
                graph = torch.cuda.CUDAGraph()
                # pyre-fixme[6]: Pyre doesn't recognize pool ID as valid _POOL_HANDLE
                with torch.cuda.graph(graph, pool=op.get_graph_pool_id()):  # type: ignore[arg-type]
                    for loop_iter in range(num_iterations_in_loop):
                        # Update loop index - this is captured with constant loop_iter
                        loop_index.fill_(loop_iter)

                        # Update send buffer based on loop index
                        _update_send_buffer_kernel(
                            send_buf,
                            loop_index,
                            base_value,
                            tokens_per_peer,
                            self.world_size,
                        )

                        # Perform alltoallv
                        graph_output = op.alltoallv_from_buffer(
                            output_split_sizes,
                            input_split_sizes,
                            num_input_tokens=max_input_tokens,
                            packed_output_tokens=packed_output_tokens,
                        )
            torch.cuda.synchronize()

            # Replay the captured loop multiple times and validate after each replay.
            num_replays = 2

            for replay in range(num_replays):
                with torch.cuda.stream(graph_stream):
                    graph.replay()

                # Wait for this replay to complete before validating
                graph_stream.synchronize()

                # Clone output after this replay
                replay_output = graph_output.clone()

                # Verify packed output shape
                self.assertEqual(replay_output.shape, (packed_output_tokens, D))

                # Validate the final loop iteration's output for this replay.
                # Each replay produces the same final loop iteration output.
                final_loop_iter = num_iterations_in_loop - 1

                # Verify packed output data for final loop iteration
                offset = 0
                for peer in range(self.world_size):
                    count = int(output_split_sizes[peer].item())
                    actual = replay_output[offset : offset + count, :].cpu()
                    # Expected: final_loop_iter * 10000 + peer * 1000 + self.rank
                    expected_value = float(
                        final_loop_iter * 10000 + peer * 1000 + self.rank
                    )
                    expected = torch.full_like(actual, expected_value)
                    torch.testing.assert_close(
                        actual,
                        expected,
                        msg=(
                            f"Replay {replay}, final loop iteration {final_loop_iter}, "
                            f"Rank {self.rank}: Data from peer {peer} incorrect. "
                            f"Expected {expected_value}, got {actual[0, 0].item()}"
                        ),
                    )
                    offset += count


def _update_input_tensor_kernel(
    input_tensor: torch.Tensor,
    loop_index: torch.Tensor,
    base_value: float,
    tokens_per_peer: int,
    world_size: int,
) -> None:
    """Simple kernel to update input tensor with loop-iteration-specific values.

    Similar to _update_send_buffer_kernel but for copy-in mode where we update
    the user's input tensor instead of the send buffer.

    IMPORTANT: This function must be CUDA graph-compatible. We cannot use
    .item() or any host-CPU sync operations inside graph capture. Instead,
    we use pure tensor operations that stay on the GPU.

    Args:
        input_tensor: The input tensor to update (shape: [max_tokens, D])
        loop_index: Tensor containing the current loop iteration index
        base_value: Base value (typically rank * 1000)
        tokens_per_peer: Number of tokens per peer
        world_size: Number of peers
    """
    # Compute iteration-specific offset using GPU tensor ops (no .item()!)
    iter_offset = loop_index * 10000

    for peer in range(world_size):
        start = peer * tokens_per_peer
        end = start + tokens_per_peer
        # Use iter_offset tensor (on GPU) + scalars for base_value and peer
        value = iter_offset.float() + base_value + peer
        input_tensor[start:end, :] = value


class TestOpMultiIterDifferentContentPackedGraphLoopCopyIn(_OpTestBase):
    """Test multi-iteration with loop captured in graph (packed, copy-in).

    Validates only the final iteration's output per replay (CUDA graph internal
    synchronization for intermediate snapshots has known limitations).
    """

    def test_multi_iter_different_content_packed_graph_loop_copy_in(self) -> None:
        """Test packed output with iteration loop captured in CUDA graph (copy-in).

        Only validates the final loop iteration's output per replay.
        """
        from comms.pipes.collectives.triton import AlltoallvOp

        tokens_per_peer = 32
        D = 16
        max_input_tokens = tokens_per_peer * self.world_size
        num_iterations_in_loop = 3
        packed_output_tokens = tokens_per_peer * self.world_size

        input_split_sizes = torch.full(
            (self.world_size,), tokens_per_peer, dtype=torch.int64, device=self.device
        )
        output_split_sizes = input_split_sizes.clone()

        op = AlltoallvOp(
            self.torchcomm,
            max_input_tokens,
            D,
            self.dtype,
            self.device,
            max_recv_tokens_per_peer=tokens_per_peer,
            sync_buffer=True,  # Required for multi-iteration CUDA graph replays
        )

        with op:
            input_tensor = torch.empty(
                max_input_tokens, D, dtype=self.dtype, device=self.device
            )
            loop_index = torch.zeros(1, dtype=torch.int64, device=self.device)

            base_value = float(self.rank * 1000)

            # Warmup
            _update_input_tensor_kernel(
                input_tensor, loop_index, base_value, tokens_per_peer, self.world_size
            )
            op.alltoallv(
                input_tensor,
                output_split_sizes,
                input_split_sizes,
                packed_output_tokens=packed_output_tokens,
            )
            torch.cuda.synchronize()

            # Capture the iteration loop
            graph_stream = torch.cuda.Stream()

            with torch.cuda.stream(graph_stream):
                graph = torch.cuda.CUDAGraph()
                # pyre-fixme[6]: Pyre doesn't recognize pool ID as valid _POOL_HANDLE
                with torch.cuda.graph(graph, pool=op.get_graph_pool_id()):  # type: ignore[arg-type]
                    for loop_iter in range(num_iterations_in_loop):
                        loop_index.fill_(loop_iter)
                        _update_input_tensor_kernel(
                            input_tensor,
                            loop_index,
                            base_value,
                            tokens_per_peer,
                            self.world_size,
                        )
                        graph_output = op.alltoallv(
                            input_tensor,
                            output_split_sizes,
                            input_split_sizes,
                            packed_output_tokens=packed_output_tokens,
                        )
            torch.cuda.synchronize()

            # Replay the captured loop multiple times and validate after each replay.
            num_replays = 2

            for replay in range(num_replays):
                with torch.cuda.stream(graph_stream):
                    graph.replay()

                # Wait for this replay to complete before validating
                graph_stream.synchronize()

                # Clone output after this replay
                replay_output = graph_output.clone()

                # Verify packed output shape
                self.assertEqual(replay_output.shape, (packed_output_tokens, D))

                # Validate the final loop iteration's output for this replay.
                # Each replay produces the same final loop iteration output.
                final_loop_iter = num_iterations_in_loop - 1

                # Verify packed output data for final loop iteration
                offset = 0
                for peer in range(self.world_size):
                    count = int(output_split_sizes[peer].item())
                    actual = replay_output[offset : offset + count, :].cpu()
                    # Expected: final_loop_iter * 10000 + peer * 1000 + self.rank
                    expected_value = float(
                        final_loop_iter * 10000 + peer * 1000 + self.rank
                    )
                    expected = torch.full_like(actual, expected_value)
                    torch.testing.assert_close(
                        actual,
                        expected,
                        msg=(
                            f"Replay {replay}, final loop iteration {final_loop_iter}, "
                            f"Rank {self.rank}: Data from peer {peer} incorrect. "
                            f"Expected {expected_value}, got {actual[0, 0].item()}"
                        ),
                    )
                    offset += count


# =============================================================================
# Sync Buffer Mode Tests
# =============================================================================


class TestOpSyncBufferBasic(_OpTestBase):
    """Test AlltoallvOp with sync_buffer=True basic functionality."""

    def test_sync_buffer_single_call(self) -> None:
        from comms.pipes.collectives.triton import AlltoallvOp

        tokens_per_peer = 64
        D = 16
        max_input_tokens = tokens_per_peer * self.world_size

        input_tensor = torch.empty(
            max_input_tokens, D, dtype=self.dtype, device=self.device
        )
        for peer in range(self.world_size):
            start = peer * tokens_per_peer
            input_tensor[start : start + tokens_per_peer] = float(
                self.rank * 1000 + peer
            )

        input_split_sizes = torch.full(
            (self.world_size,), tokens_per_peer, dtype=torch.int64, device=self.device
        )
        output_split_sizes = input_split_sizes.clone()

        # Create op with sync_buffer=True
        op = AlltoallvOp(
            self.torchcomm,
            max_input_tokens,
            D,
            self.dtype,
            self.device,
            max_recv_tokens_per_peer=tokens_per_peer,
            sync_buffer=True,  # Enable buffer-ready synchronization
        )
        with op:
            output = op.alltoallv(input_tensor, output_split_sizes, input_split_sizes)

        torch.cuda.synchronize()
        total_tokens = tokens_per_peer * self.world_size
        self.assertEqual(output.shape, (total_tokens, D))
        self._verify_packed_data(output, D, output_split_sizes, "sync_buffer_single")


class TestOpSyncBufferRepeatedCalls(_OpTestBase):
    """Test AlltoallvOp with sync_buffer=True and multiple iterations.

    This is the critical test for sync_buffer mode - verifies buffer-ready
    synchronization prevents race conditions across iterations.
    """

    def test_sync_buffer_repeated_calls(self) -> None:
        from comms.pipes.collectives.triton import AlltoallvOp

        tokens_per_peer = 64
        D = 16
        max_input_tokens = tokens_per_peer * self.world_size
        num_iterations = 5

        input_tensor = torch.empty(
            max_input_tokens, D, dtype=self.dtype, device=self.device
        )
        for peer in range(self.world_size):
            start = peer * tokens_per_peer
            input_tensor[start : start + tokens_per_peer] = float(
                self.rank * 1000 + peer
            )

        input_split_sizes = torch.full(
            (self.world_size,), tokens_per_peer, dtype=torch.int64, device=self.device
        )
        output_split_sizes = input_split_sizes.clone()

        op = AlltoallvOp(
            self.torchcomm,
            max_input_tokens,
            D,
            self.dtype,
            self.device,
            max_recv_tokens_per_peer=tokens_per_peer,
            sync_buffer=True,
        )
        with op:
            output = None
            for _iteration in range(num_iterations):
                output = op.alltoallv(
                    input_tensor, output_split_sizes, input_split_sizes
                )
            torch.cuda.synchronize()

            assert output is not None
            total_tokens = tokens_per_peer * self.world_size
            self.assertEqual(output.shape, (total_tokens, D))
            self._verify_packed_data(
                output, D, output_split_sizes, "sync_buffer_repeated"
            )


class TestOpSyncBufferZeroCopy(_OpTestBase):
    """Test AlltoallvOp sync_buffer with zero-copy send buffer."""

    def test_sync_buffer_zero_copy(self) -> None:
        from comms.pipes.collectives.triton import AlltoallvOp

        tokens_per_peer = 64
        D = 16
        max_input_tokens = tokens_per_peer * self.world_size
        total_tokens = tokens_per_peer * self.world_size

        input_split_sizes = torch.full(
            (self.world_size,), tokens_per_peer, dtype=torch.int64, device=self.device
        )
        output_split_sizes = input_split_sizes.clone()

        op = AlltoallvOp(
            self.torchcomm,
            max_input_tokens,
            D,
            self.dtype,
            self.device,
            max_recv_tokens_per_peer=tokens_per_peer,
            sync_buffer=True,
        )

        # Fill send buffer BEFORE setup
        send_buf = op.get_send_buffer(total_tokens)
        for peer in range(self.world_size):
            start = peer * tokens_per_peer
            send_buf[start : start + tokens_per_peer] = float(self.rank * 1000 + peer)

        with op:
            output = op.alltoallv_from_buffer(
                output_split_sizes,
                input_split_sizes,
                num_input_tokens=total_tokens,
            )

        torch.cuda.synchronize()
        self.assertEqual(output.shape, (total_tokens, D))
        self._verify_packed_data(output, D, output_split_sizes, "sync_buffer_zero_copy")


class TestOpSyncBufferPackedOutput(_OpTestBase):
    """Test AlltoallvOp sync_buffer with packed output."""

    def test_sync_buffer_packed_output(self) -> None:
        from comms.pipes.collectives.triton import AlltoallvOp

        tokens_per_peer = 64
        D = 16
        max_input_tokens = tokens_per_peer * self.world_size

        input_tensor = torch.empty(
            max_input_tokens, D, dtype=self.dtype, device=self.device
        )
        for peer in range(self.world_size):
            start = peer * tokens_per_peer
            input_tensor[start : start + tokens_per_peer] = float(
                self.rank * 1000 + peer
            )

        input_split_sizes = torch.full(
            (self.world_size,), tokens_per_peer, dtype=torch.int64, device=self.device
        )
        output_split_sizes = input_split_sizes.clone()
        total_output_tokens = int(output_split_sizes.sum().item())

        # Create op with both sync_buffer and packed_output
        op = AlltoallvOp(
            self.torchcomm,
            max_input_tokens,
            D,
            self.dtype,
            self.device,
            max_recv_tokens_per_peer=tokens_per_peer,
            sync_buffer=True,
        )
        with op:
            # Pass packed_output_tokens to enable CUDA graph compatibility
            # (avoids .item() call during graph capture)
            output = op.alltoallv(
                input_tensor,
                output_split_sizes,
                input_split_sizes,
                packed_output_tokens=total_output_tokens,
            )

        torch.cuda.synchronize()
        self.assertEqual(output.shape, (total_output_tokens, D))


class TestOpSyncBufferAttribute(_OpTestBase):
    """Test that sync_buffer is correctly stored as an attribute."""

    def test_sync_buffer_attribute_stored(self) -> None:
        from comms.pipes.collectives.triton import AlltoallvOp

        tokens_per_peer = 64
        D = 16
        max_input_tokens = tokens_per_peer * self.world_size

        # Create op with sync_buffer=True
        op_sync = AlltoallvOp(
            self.torchcomm,
            max_input_tokens,
            D,
            self.dtype,
            self.device,
            max_recv_tokens_per_peer=tokens_per_peer,
            sync_buffer=True,
        )
        self.assertTrue(op_sync.sync_buffer)

        # Create op with sync_buffer=False (default)
        op_non_sync = AlltoallvOp(
            self.torchcomm,
            max_input_tokens,
            D,
            self.dtype,
            self.device,
            max_recv_tokens_per_peer=tokens_per_peer,
            sync_buffer=False,
        )
        self.assertFalse(op_non_sync.sync_buffer)


# =============================================================================
# Test: Intentional Race Condition for Debugging
# =============================================================================


@unittest.skipIf(not _skip_if_not_ready(), "Skipping without device API test flag")
class TestOpRaceConditionDebug(_OpTestBase):
    """
    Test that intentionally creates a race condition by delaying graph completion
    on some ranks using a spin loop kernel. This helps identify protocol violations
    by comparing failing runs with passing runs.

    The race condition occurs when:
    1. Some ranks finish their graph early and start send_buf.copy_() for next iteration
    2. Other ranks are still executing the previous graph and reading from send buffers
    3. The early ranks overwrite their send buffers while other ranks are still reading

    Run with:
        TEST_FILTER=TestOpRaceConditionDebug$ buck2 run ...
    """

    def test_race_condition_debug(self) -> None:
        """
        Create intentional race by delaying graph completion on odd ranks.
        Even ranks finish quickly, odd ranks spin for extra cycles.
        This creates a window where even ranks may overwrite send buffers
        while odd ranks are still reading.
        """
        import triton
        import triton.language as tl
        from comms.pipes.collectives.triton import AlltoallvOp

        # Spin loop kernel to delay execution on specific ranks
        @triton.jit
        def spin_loop_kernel(
            spin_cycles: tl.constexpr,
        ):
            """Busy-wait spin loop to delay GPU execution."""
            # Simple spin loop - each iteration takes a few cycles
            for _ in range(spin_cycles):
                # Use a volatile memory operation to prevent optimization
                tl.debug_barrier()

        tokens_per_peer = 32
        D = 16
        max_input_tokens = tokens_per_peer * self.world_size
        num_iterations = 5  # More iterations to increase chance of hitting race
        packed_output_tokens = tokens_per_peer * self.world_size

        # Spin cycles for odd ranks to create timing skew
        # Higher value = more delay = higher chance of race
        SPIN_CYCLES_ODD_RANKS = 100000  # Tune this to create race window

        input_split_sizes = torch.full(
            (self.world_size,), tokens_per_peer, dtype=torch.int64, device=self.device
        )
        output_split_sizes = input_split_sizes.clone()

        op = AlltoallvOp(
            self.torchcomm,
            max_input_tokens,
            D,
            self.dtype,
            self.device,
            max_recv_tokens_per_peer=tokens_per_peer,
            sync_buffer=True,
        )

        with op:
            send_buf = op.get_send_buffer(max_input_tokens)

            # Pre-stage all iteration inputs to verify data integrity
            # Value encoding: iteration * 10000 + my_rank * 1000 + dest_rank
            staged_inputs = []
            for iteration in range(num_iterations):
                staged_input = torch.zeros_like(send_buf)
                for dest_rank in range(self.world_size):
                    start_idx = dest_rank * tokens_per_peer
                    end_idx = start_idx + tokens_per_peer
                    value = float(iteration * 10000 + self.rank * 1000 + dest_rank)
                    staged_input[start_idx:end_idx, :] = value
                staged_inputs.append(staged_input)

            # Warmup iteration to establish baseline signals
            send_buf.copy_(staged_inputs[0])
            op.alltoallv_from_buffer(
                output_split_sizes,
                input_split_sizes,
                num_input_tokens=max_input_tokens,
                packed_output_tokens=packed_output_tokens,
            )
            torch.cuda.synchronize()

            # Create a staging buffer for graph capture
            # The staging buffer holds the input data that will be copied to send_buf
            # inside the graph, AFTER the BUFFER_READY wait
            staging_buffer = torch.zeros_like(send_buf)

            # Capture graph with copy INSIDE the graph
            # This ensures the copy happens on the GPU stream, synchronized with
            # the BUFFER_READY wait in the alltoallv kernel
            graph_stream = torch.cuda.Stream()
            with torch.cuda.stream(graph_stream):
                graph = torch.cuda.CUDAGraph()
                # pyre-fixme[6]: Pyre doesn't recognize pool ID as valid _POOL_HANDLE
                with torch.cuda.graph(graph, pool=op.get_graph_pool_id()):  # type: ignore[arg-type]
                    # Copy from staging to send_buf INSIDE the graph
                    # This copy will be captured and replayed
                    send_buf.copy_(staging_buffer)
                    # Then run alltoallv
                    graph_output = op.alltoallv_from_buffer(
                        output_split_sizes,
                        input_split_sizes,
                        num_input_tokens=max_input_tokens,
                        packed_output_tokens=packed_output_tokens,
                    )
                    # Add spin loop AFTER alltoallv for ODD ranks only
                    # This delays graph completion, creating a race window
                    if self.rank % 2 == 1:
                        # pyre-fixme[6]: Triton constexpr not recognized
                        spin_loop_kernel[(1,)](SPIN_CYCLES_ODD_RANKS)  # type: ignore[arg-type]
            torch.cuda.synchronize()

            # Track outputs per iteration
            outputs_per_iter = []

            for iteration in range(num_iterations):
                with torch.cuda.stream(graph_stream):
                    staging_buffer.copy_(staged_inputs[iteration])
                    graph.replay()

                # Wait for graph to complete before cloning
                graph_stream.synchronize()

                # Clone output on graph_stream to ensure it completes BEFORE
                # the next iteration starts (which signals BUFFER_READY)
                with torch.cuda.stream(graph_stream):
                    output_snapshot = graph_output.clone()

                # Synchronize to ensure clone is complete before next iteration
                graph_stream.synchronize()

                outputs_per_iter.append(output_snapshot)

            # Final synchronize
            torch.cuda.synchronize()

            # Validate all peer communications
            errors = []
            for iteration in range(num_iterations):
                output = outputs_per_iter[iteration]
                for peer in range(self.world_size):
                    if peer == self.rank:
                        continue
                    expected_val = float(iteration * 10000 + peer * 1000 + self.rank)
                    start_idx = peer * tokens_per_peer
                    actual_val = output[start_idx, 0].item()
                    if abs(actual_val - expected_val) > 0.1:
                        errors.append(
                            f"Iter {iteration}, from peer {peer}: expected {expected_val:.0f}, got {actual_val:.0f}"
                        )

            if errors:
                print(
                    f"\n[Rank {self.rank}] RACE CONDITION DETECTED! {len(errors)} errors found.",
                    flush=True,
                )
                for error in errors[:10]:
                    print(f"[Rank {self.rank}] ERROR: {error}", file=sys.stderr)
                self.fail(
                    f"Race condition detected on rank {self.rank}: {len(errors)} mismatches\n"
                    + "\n".join(errors[:10])
                )
            else:
                print(
                    f"\n[Rank {self.rank}] All {num_iterations} iterations validated successfully.",
                    flush=True,
                )


@unittest.skipIf(not _skip_if_not_ready(), "Skipping without device API test flag")
class TestOpRaceConditionDebugMultiBlock(_OpTestBase):
    """
    Test race conditions with BLOCKS_PER_PEER > 1 code paths.

    Similar to TestOpRaceConditionDebug but uses larger message sizes to trigger
    multi-block code paths (BLOCKS_PER_PEER = 8 for 64KB-256KB messages).

    The multi-block path uses atomic completion counters and different signaling
    logic that needs to be tested separately.

    Run with:
        TEST_FILTER=TestOpRaceConditionDebugMultiBlock$ buck2 run ...
    """

    def test_race_condition_debug_multi_block(self) -> None:
        """
        Create intentional race by delaying graph completion on odd ranks.
        Uses large message sizes to trigger BLOCKS_PER_PEER > 1.
        """
        import triton
        import triton.language as tl
        from comms.pipes.collectives.triton import AlltoallvOp

        # Spin loop kernel to delay execution on specific ranks
        @triton.jit
        def spin_loop_kernel(
            spin_cycles: tl.constexpr,
        ):
            """Busy-wait spin loop to delay GPU execution."""
            for _ in range(spin_cycles):
                tl.debug_barrier()

        # Use larger tokens_per_peer and D to get > 64KB per peer
        # With bfloat16 (2 bytes): 2048 * 32 * 2 = 131072 bytes = 128KB per peer
        # This triggers BLOCKS_PER_PEER = 8 (for 64KB-256KB range)
        tokens_per_peer = 2048
        D = 32
        max_input_tokens = tokens_per_peer * self.world_size
        num_iterations = 5
        packed_output_tokens = tokens_per_peer * self.world_size

        # Calculate expected per-peer message size
        elem_bytes = 2  # bfloat16
        per_peer_bytes = tokens_per_peer * D * elem_bytes
        print(
            f"[Rank {self.rank}] Per-peer message size: {per_peer_bytes} bytes ({per_peer_bytes / 1024:.1f} KB)",
            flush=True,
        )

        # Spin cycles for odd ranks to create timing skew
        SPIN_CYCLES_ODD_RANKS = 100000

        input_split_sizes = torch.full(
            (self.world_size,), tokens_per_peer, dtype=torch.int64, device=self.device
        )
        output_split_sizes = input_split_sizes.clone()

        op = AlltoallvOp(
            self.torchcomm,
            max_input_tokens,
            D,
            self.dtype,
            self.device,
            max_recv_tokens_per_peer=tokens_per_peer,
            sync_buffer=True,
        )

        with op:
            send_buf = op.get_send_buffer(max_input_tokens)

            # Pre-stage all iteration inputs to verify data integrity
            # Value encoding: iteration * 10000 + my_rank * 1000 + dest_rank
            staged_inputs = []
            for iteration in range(num_iterations):
                staged_input = torch.zeros_like(send_buf)
                for dest_rank in range(self.world_size):
                    start_idx = dest_rank * tokens_per_peer
                    end_idx = start_idx + tokens_per_peer
                    value = float(iteration * 10000 + self.rank * 1000 + dest_rank)
                    staged_input[start_idx:end_idx, :] = value
                staged_inputs.append(staged_input)

            # Warmup iteration to establish baseline signals
            send_buf.copy_(staged_inputs[0])
            op.alltoallv_from_buffer(
                output_split_sizes,
                input_split_sizes,
                num_input_tokens=max_input_tokens,
                packed_output_tokens=packed_output_tokens,
            )
            torch.cuda.synchronize()

            # Create a staging buffer for graph capture
            staging_buffer = torch.zeros_like(send_buf)

            # Capture graph with copy INSIDE the graph
            graph_stream = torch.cuda.Stream()
            with torch.cuda.stream(graph_stream):
                graph = torch.cuda.CUDAGraph()
                # pyre-fixme[6]: Pyre doesn't recognize pool ID as valid _POOL_HANDLE
                with torch.cuda.graph(graph, pool=op.get_graph_pool_id()):  # type: ignore[arg-type]
                    send_buf.copy_(staging_buffer)
                    graph_output = op.alltoallv_from_buffer(
                        output_split_sizes,
                        input_split_sizes,
                        num_input_tokens=max_input_tokens,
                        packed_output_tokens=packed_output_tokens,
                    )
                    # Add spin loop AFTER alltoallv for ODD ranks only
                    if self.rank % 2 == 1:
                        # pyre-fixme[6]: Triton constexpr not recognized
                        spin_loop_kernel[(1,)](SPIN_CYCLES_ODD_RANKS)  # type: ignore[arg-type]
            torch.cuda.synchronize()

            # Track outputs per iteration
            outputs_per_iter = []

            for iteration in range(num_iterations):
                with torch.cuda.stream(graph_stream):
                    staging_buffer.copy_(staged_inputs[iteration])
                    graph.replay()

                # Wait for graph to complete before cloning
                graph_stream.synchronize()

                # Clone output on graph_stream
                with torch.cuda.stream(graph_stream):
                    output_snapshot = graph_output.clone()

                graph_stream.synchronize()
                outputs_per_iter.append(output_snapshot)

            torch.cuda.synchronize()

            # Validate all peer communications
            errors = []
            for iteration in range(num_iterations):
                output = outputs_per_iter[iteration]
                for peer in range(self.world_size):
                    if peer == self.rank:
                        continue
                    expected_val = float(iteration * 10000 + peer * 1000 + self.rank)
                    start_idx = peer * tokens_per_peer
                    actual_val = output[start_idx, 0].item()
                    if abs(actual_val - expected_val) > 0.1:
                        errors.append(
                            f"Iter {iteration}, from peer {peer}: expected {expected_val:.0f}, got {actual_val:.0f}"
                        )

            if errors:
                print(
                    f"\n[Rank {self.rank}] RACE CONDITION DETECTED! {len(errors)} errors found.",
                    flush=True,
                )
                for error in errors[:10]:
                    print(f"[Rank {self.rank}] ERROR: {error}", file=sys.stderr)
                self.fail(
                    f"Race condition detected on rank {self.rank}: {len(errors)} mismatches\n"
                    + "\n".join(errors[:10])
                )
            else:
                print(
                    f"\n[Rank {self.rank}] All {num_iterations} iterations validated successfully.",
                    flush=True,
                )


# =============================================================================
# Main Registry
# =============================================================================

ALL_TEST_CLASSES = [
    TestOpUniformSplitsCopyIn,
    TestOpZeroCopy,
    TestOpRepeatedCalls,
    TestOpGetOrCreate,
    TestOpDoubleSetupRaises,
    TestOpAlltoallvWithoutSetupRaises,
    TestOpOversizedInputRaises,
    TestOpOversizedGetSendBufferRaises,
    TestOpTeardownIdempotent,
    TestOpPackedOutputUniform,
    TestOpFillSendBufferBasic,
    # Multi-iteration tests (different content per iteration)
    TestOpFillSendBufferBackToBack,
    TestOpFillSendBufferOversizedRaises,
    TestOpFillSendBufferWithExplicitNumTokens,
    # Sync buffer mode tests
    TestOpSyncBufferBasic,
    TestOpSyncBufferRepeatedCalls,
    TestOpSyncBufferZeroCopy,
    TestOpSyncBufferPackedOutput,
    TestOpSyncBufferAttribute,
    # Multi-iteration tests (different content per iteration)
    # Non-graph: zero-copy API
    TestOpMultiIterDifferentContentPackedNonGraph,
    # Non-graph: copy-in API
    TestOpMultiIterDifferentContentPackedNonGraphCopyIn,
    # Graph: zero-copy API
    TestOpMultiIterDifferentContentPackedGraph,
    # Graph: copy-in API
    TestOpMultiIterDifferentContentPackedGraphCopyIn,
    # GraphLoop: zero-copy API (loop captured in graph)
    TestOpMultiIterDifferentContentPackedGraphLoop,
    # GraphLoop: copy-in API (loop captured in graph)
    TestOpMultiIterDifferentContentPackedGraphLoopCopyIn,
    # Debug: intentional race condition test (BLOCKS_PER_PEER == 1)
    TestOpRaceConditionDebug,
    # Debug: intentional race condition test (BLOCKS_PER_PEER > 1)
    TestOpRaceConditionDebugMultiBlock,
]


def main() -> int:
    import re

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    test_filter = os.environ.get("TEST_FILTER", "")

    if test_filter:
        # Compile regex pattern for matching
        try:
            pattern = re.compile(test_filter)
        except re.error:
            # If invalid regex, treat as literal substring
            pattern = re.compile(re.escape(test_filter))

        for cls in ALL_TEST_CLASSES:
            class_name = cls.__name__
            # Check if class name matches the pattern
            if pattern.search(class_name):
                suite.addTests(loader.loadTestsFromTestCase(cls))
            else:
                # Check individual test methods
                for name in loader.getTestCaseNames(cls):
                    full_name = f"{class_name}.{name}"
                    if pattern.search(full_name) or pattern.search(name):
                        suite.addTest(cls(name))

        if suite.countTestCases() == 0:
            print(
                f"WARNING: TEST_FILTER='{test_filter}' matched no tests. "
                f"Running all {len(ALL_TEST_CLASSES)} test classes.",
                file=sys.stderr,
            )
            for cls in ALL_TEST_CLASSES:
                suite.addTests(loader.loadTestsFromTestCase(cls))
    else:
        for cls in ALL_TEST_CLASSES:
            suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
