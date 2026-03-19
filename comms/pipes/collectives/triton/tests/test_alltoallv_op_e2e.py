# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-unsafe
"""
End-to-end integration tests for AlltoallvOp (high-level MSL-compatible API).

Tests the simplified token-level API with zero-copy buffer ownership covering:
- Zero-copy mode with mandatory input_tensor and output_tensor
- Repeated calls with in-place input tensor updates
- get_or_create() caching factory
- Error handling (double setup, alltoallv without setup, oversized input)
- CUDA graph capture and replay

Zero-Copy Architecture:
    User provides both input_tensor and output_tensor (MANDATORY).
    These tensors become GIN-registered buffers on first alltoallv() call.
    User updates input_tensor contents in-place between calls.
    Data arrives directly in output_tensor.

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
from typing import Any

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
    """Common setUp / tearDown for AlltoallvOp tests."""

    # Class-level shared state for the test session
    wrapper: TorchCommTestWrapper | None = None
    torchcomm: Any = None
    rank: int = 0
    world_size: int = 1
    device: Any = None
    dtype: torch.dtype = torch.float32
    D: int = 16  # Unused but kept for backward compatibility

    @classmethod
    def setUpClass(cls) -> None:
        cls.wrapper = TorchCommTestWrapper()
        cls.torchcomm = cls.wrapper.get_torchcomm()
        cls.rank = cls.torchcomm.get_rank()
        cls.world_size = cls.torchcomm.get_size()
        cls.device = cls.torchcomm.get_device()
        cls.dtype = torch.float32

    @classmethod
    def tearDownClass(cls) -> None:
        from comms.pipes.collectives.triton import AlltoallvOp

        AlltoallvOp.clear_cache()
        if cls.torchcomm is not None:
            try:
                cls.torchcomm.barrier(False)
            except RuntimeError:
                pass  # Ignore if communicator already torn down
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
# Test: Zero-Copy Mode with Uniform Splits
# =============================================================================


class TestOpUniformSplitsZeroCopy(_OpTestBase):
    """Test basic alltoallv with user-owned zero-copy buffers (uniform splits)."""

    def test_uniform_splits_zero_copy(self) -> None:
        from comms.pipes.collectives.triton import AlltoallvOp

        tokens_per_peer = 64
        D = 16
        max_input_tokens = tokens_per_peer * self.world_size
        total_output_tokens = tokens_per_peer * self.world_size

        # Create op
        op = AlltoallvOp(
            self.torchcomm,
            max_input_tokens,
            D,
            self.dtype,
            self.device,
            max_recv_tokens_per_peer=tokens_per_peer,
        )

        input_split_sizes = torch.full(
            (self.world_size,), tokens_per_peer, dtype=torch.int64, device=self.device
        )
        output_split_sizes = input_split_sizes.clone()

        with op:
            # Allocate GIN-compatible tensors using the op's internal pool
            # Must be done AFTER setup() (inside the with block)
            input_tensor = op.alloc_buffer((max_input_tokens, D))
            output_tensor = op.alloc_buffer((total_output_tokens, D))

            # Fill input tensor
            for peer in range(self.world_size):
                start = peer * tokens_per_peer
                input_tensor[start : start + tokens_per_peer] = float(
                    self.rank * 1000 + peer
                )

            output = op.alltoallv(
                input_tensor, output_tensor, output_split_sizes, input_split_sizes
            )

        torch.cuda.synchronize()
        total_tokens = tokens_per_peer * self.world_size
        self.assertEqual(
            output.shape,
            (total_tokens, D),
        )
        self._verify_packed_data(output, D, output_split_sizes, "uniform_zero_copy")


# =============================================================================
# Test: Repeated Calls with In-Place Updates
# =============================================================================


class TestOpRepeatedCalls(_OpTestBase):
    """Test calling alltoallv multiple times with same buffers (zero-copy pattern)."""

    def test_repeated_calls_same_buffers(self) -> None:
        from comms.pipes.collectives.triton import AlltoallvOp

        tokens_per_peer = 64
        D = 16
        max_input_tokens = tokens_per_peer * self.world_size
        total_output_tokens = tokens_per_peer * self.world_size
        num_iterations = 5

        # Create op
        op = AlltoallvOp(
            self.torchcomm,
            max_input_tokens,
            D,
            self.dtype,
            self.device,
            max_recv_tokens_per_peer=tokens_per_peer,
        )

        input_split_sizes = torch.full(
            (self.world_size,), tokens_per_peer, dtype=torch.int64, device=self.device
        )
        output_split_sizes = input_split_sizes.clone()

        with op:
            # Allocate GIN-compatible tensors using the op's internal pool
            input_tensor = op.alloc_buffer((max_input_tokens, D))
            output_tensor = op.alloc_buffer((total_output_tokens, D))

            # Fill input tensor
            for peer in range(self.world_size):
                start = peer * tokens_per_peer
                input_tensor[start : start + tokens_per_peer] = float(
                    self.rank * 1000 + peer
                )

            output = None
            for _ in range(num_iterations):
                # Use same tensors, update input in-place if needed
                output = op.alltoallv(
                    input_tensor, output_tensor, output_split_sizes, input_split_sizes
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
        total_output_tokens = tokens_per_peer * self.world_size

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

        # Allocate GIN-compatible tensors AFTER get_or_create (which calls setup)
        input_tensor = op1.alloc_buffer((max_input_tokens, D))
        output_tensor = op1.alloc_buffer((total_output_tokens, D))

        # Fill input tensor
        for peer in range(self.world_size):
            start = peer * tokens_per_peer
            input_tensor[start : start + tokens_per_peer] = float(
                self.rank * 1000 + peer
            )

        # The cached op should be ready to use (no explicit setup needed).
        output = op1.alltoallv(
            input_tensor, output_tensor, output_split_sizes, input_split_sizes
        )
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

        # Try to call alloc_buffer without setup - should raise
        with self.assertRaises(RuntimeError):
            op.alloc_buffer((max_input_tokens, D))

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
        total_output_tokens = tokens_per_peer * self.world_size

        op = AlltoallvOp(
            self.torchcomm,
            max_input_tokens,
            D,
            self.dtype,
            self.device,
            max_recv_tokens_per_peer=tokens_per_peer,
        )

        split_sizes = torch.full(
            (self.world_size,),
            tokens_per_peer,
            dtype=torch.int64,
            device=self.device,
        )

        with op:
            # Allocate GIN-compatible tensors
            # Oversized by 1 token - allocate max_input_tokens + 1
            oversized_input = op.alloc_buffer((max_input_tokens + 1, D))
            output_tensor = op.alloc_buffer((total_output_tokens, D))

            with self.assertRaises(ValueError):
                op.alltoallv(oversized_input, output_tensor, split_sizes, split_sizes)

        torch.cuda.synchronize()
        self.torchcomm.barrier(False)


class TestOpTensorIdentityEnforced(_OpTestBase):
    """Test that different tensors after first call raises RuntimeError."""

    def test_tensor_identity_enforced(self) -> None:
        from comms.pipes.collectives.triton import AlltoallvOp

        D = 16
        max_input_tokens = 64
        tokens_per_peer = max_input_tokens // self.world_size
        total_output_tokens = tokens_per_peer * self.world_size

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

        with op:
            # Allocate GIN-compatible tensors
            input_tensor = op.alloc_buffer((max_input_tokens, D))
            output_tensor = op.alloc_buffer((total_output_tokens, D))

            # Fill input tensor
            for peer in range(self.world_size):
                start = peer * tokens_per_peer
                input_tensor[start : start + tokens_per_peer] = float(
                    self.rank * 1000 + peer
                )

            # First call takes ownership
            op.alltoallv(input_tensor, output_tensor, split_sizes, split_sizes)

            # Second call with different input tensor should raise
            different_input = op.alloc_buffer((max_input_tokens, D))
            with self.assertRaises(RuntimeError):
                op.alltoallv(different_input, output_tensor, split_sizes, split_sizes)

            # Second call with different output tensor should raise
            different_output = op.alloc_buffer((total_output_tokens, D))
            with self.assertRaises(RuntimeError):
                op.alltoallv(input_tensor, different_output, split_sizes, split_sizes)


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
    """Test packed output logic with uniform splits and GIN-compatible tensors."""

    def test_packed_output_uniform(self) -> None:
        from comms.pipes.collectives.triton import AlltoallvOp

        tokens_per_peer = 32
        D = 16
        max_input_tokens = tokens_per_peer * self.world_size
        total_output_tokens = tokens_per_peer * self.world_size

        input_split_sizes = torch.full(
            (self.world_size,), tokens_per_peer, dtype=torch.int64, device=self.device
        )
        output_split_sizes = input_split_sizes.clone()

        # Pre-compute total tokens
        packed_output_tokens = tokens_per_peer * self.world_size

        op = AlltoallvOp(
            self.torchcomm,
            max_input_tokens,
            D,
            self.dtype,
            self.device,
            max_recv_tokens_per_peer=tokens_per_peer,
        )
        with op:
            # Allocate GIN-compatible tensors
            input_tensor = op.alloc_buffer((max_input_tokens, D))
            output_tensor = op.alloc_buffer((total_output_tokens, D))

            # Build input tensor
            for peer in range(self.world_size):
                start = peer * tokens_per_peer
                input_tensor[start : start + tokens_per_peer] = float(
                    self.rank * 1000 + peer
                )

            output = op.alltoallv(
                input_tensor,
                output_tensor,
                output_split_sizes,
                input_split_sizes,
                packed_output_tokens=packed_output_tokens,
            )

        torch.cuda.synchronize()

        # Verify packed output shape
        self.assertEqual(output.shape, (packed_output_tokens, D))

        # Verify packed output data
        self._verify_packed_data(output, D, output_split_sizes, "packed_uniform")


# =============================================================================
# Test: Multi-Iteration with Different Input Content (Non-Graph)
# =============================================================================


class TestOpMultiIterDifferentContentNonGraph(_OpTestBase):
    """Test repeated alltoallv calls with varying input data per iteration (no CUDA graph)."""

    def test_multi_iter_different_content_non_graph(self) -> None:
        from comms.pipes.collectives.triton import AlltoallvOp

        tokens_per_peer = 32
        D = 16
        max_input_tokens = tokens_per_peer * self.world_size
        total_output_tokens = tokens_per_peer * self.world_size
        num_iterations = 3
        packed_output_tokens = tokens_per_peer * self.world_size

        input_split_sizes = torch.full(
            (self.world_size,), tokens_per_peer, dtype=torch.int64, device=self.device
        )
        output_split_sizes = input_split_sizes.clone()

        # Use a dedicated stream for operations
        op_stream = torch.cuda.Stream()

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
            # Allocate GIN-compatible tensors inside the with block
            input_tensor = op.alloc_buffer((max_input_tokens, D))
            output_tensor = op.alloc_buffer((total_output_tokens, D))

            # Pre-allocate buffers to capture output state after each iteration.
            validation_buffers = []

            for iteration in range(num_iterations):
                with torch.cuda.stream(op_stream):
                    # Update input tensor in-place with iteration-specific content
                    for peer in range(self.world_size):
                        start = peer * tokens_per_peer
                        value = float(iteration * 10000 + self.rank * 1000 + peer)
                        input_tensor[start : start + tokens_per_peer] = value

                    output = op.alltoallv(
                        input_tensor,
                        output_tensor,
                        output_split_sizes,
                        input_split_sizes,
                        packed_output_tokens=packed_output_tokens,
                    )

                    # Clone output on op_stream to capture this iteration's data
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
                            f"Zero-copy iter {iteration}, Rank {self.rank}: Data "
                            f"from peer {peer} incorrect. Expected {expected_value}, "
                            f"got {actual[0, 0].item()}"
                        ),
                    )
                    offset += count


# =============================================================================
# Test: Multi-Iteration with CUDA Graph
# =============================================================================


class TestOpMultiIterDifferentContentGraph(_OpTestBase):
    """Test CUDA graph replay with varying input data per iteration.

    This test verifies the vLLM production use case (piecewise CUDA graphs):
    - Capture a SINGLE graph with one alltoallv call
    - Update input tensor OUTSIDE the graph before each replay
    - Replay the same graph multiple times with different data
    - Validate output data per iteration
    """

    def test_multi_iter_different_content_graph(self) -> None:
        from comms.pipes.collectives.triton import AlltoallvOp

        tokens_per_peer = 32
        D = 16
        max_input_tokens = tokens_per_peer * self.world_size
        total_output_tokens = tokens_per_peer * self.world_size
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
            # Allocate GIN-compatible tensors inside the with block
            input_tensor = op.alloc_buffer((max_input_tokens, D))
            output_tensor = op.alloc_buffer((total_output_tokens, D))

            # Pre-stage input data for ALL iterations before graph capture
            staged_inputs = []
            for iteration in range(num_iterations):
                staged_input = op.alloc_buffer((max_input_tokens, D))
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
                output_tensor,
                output_split_sizes,
                input_split_sizes,
                packed_output_tokens=packed_output_tokens,
            )
            torch.cuda.synchronize()

            # Capture a SINGLE graph with alltoallv
            graph_stream = torch.cuda.Stream()
            with torch.cuda.stream(graph_stream):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graph_output = op.alltoallv(
                        input_tensor,
                        output_tensor,
                        output_split_sizes,
                        input_split_sizes,
                        packed_output_tokens=packed_output_tokens,
                    )
            torch.cuda.synchronize()

            # Replay the SAME graph multiple times with different input data
            for iteration in range(num_iterations):
                with torch.cuda.stream(graph_stream):
                    # Update input tensor in-place before replay
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


def _update_input_tensor_kernel(
    input_tensor: torch.Tensor,
    loop_index: torch.Tensor,
    base_value: float,
    tokens_per_peer: int,
    world_size: int,
) -> None:
    """Simple kernel to update input tensor with loop-iteration-specific values.

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


class TestOpMultiIterDifferentContentGraphLoop(_OpTestBase):
    """Test CUDA graph replay with a loop of alltoallv calls captured in the graph.

    This test verifies capturing multiple iterations inside a single graph:
    - Capture a SINGLE graph containing a LOOP of multiple alltoallv calls
    - Update input tensor INSIDE the graph using GPU-compatible operations
    - Replay the same graph multiple times
    - Validate output data for each iteration per replay
    """

    def test_multi_iter_different_content_graph_loop(self) -> None:
        from comms.pipes.collectives.triton import AlltoallvOp

        tokens_per_peer = 32
        D = 16
        max_input_tokens = tokens_per_peer * self.world_size
        total_output_tokens = tokens_per_peer * self.world_size
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
            # Allocate GIN-compatible tensors inside the with block
            input_tensor = op.alloc_buffer((max_input_tokens, D))
            output_tensor = op.alloc_buffer((total_output_tokens, D))
            loop_index = torch.zeros(1, dtype=torch.int64, device=self.device)
            base_value = float(self.rank * 1000)

            # Allocate separate storage tensors for each iteration's output
            iter_outputs = [
                op.alloc_buffer((packed_output_tokens, D))
                for _ in range(num_iterations_in_loop)
            ]

            # Warmup
            _update_input_tensor_kernel(
                input_tensor, loop_index, base_value, tokens_per_peer, self.world_size
            )
            # First call takes ownership of tensors
            op.alltoallv(
                input_tensor,
                output_tensor,
                output_split_sizes,
                input_split_sizes,
                packed_output_tokens=packed_output_tokens,
            )
            torch.cuda.synchronize()

            # Capture the iteration loop
            graph_stream = torch.cuda.Stream()

            with torch.cuda.stream(graph_stream):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
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
                            output_tensor,
                            output_split_sizes,
                            input_split_sizes,
                            packed_output_tokens=packed_output_tokens,
                        )
                        # Copy output to iteration-specific storage
                        iter_outputs[loop_iter].copy_(graph_output)
            torch.cuda.synchronize()

            # Replay the captured loop multiple times and validate after each replay.
            num_replays = 2

            for replay in range(num_replays):
                with torch.cuda.stream(graph_stream):
                    graph.replay()

                # Wait for this replay to complete before validating
                graph_stream.synchronize()

                # Validate output for each loop iteration
                for loop_iter in range(num_iterations_in_loop):
                    iter_output = iter_outputs[loop_iter].clone()

                    # Verify packed output shape
                    self.assertEqual(iter_output.shape, (packed_output_tokens, D))

                    # Verify packed output data for this iteration
                    offset = 0
                    for peer in range(self.world_size):
                        count = int(output_split_sizes[peer].item())
                        actual = iter_output[offset : offset + count, :].cpu()
                        expected_value = float(
                            loop_iter * 10000 + peer * 1000 + self.rank
                        )
                        expected = torch.full_like(actual, expected_value)
                        torch.testing.assert_close(
                            actual,
                            expected,
                            msg=(
                                f"Replay {replay}, loop iteration {loop_iter}, "
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
        total_output_tokens = tokens_per_peer * self.world_size

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
            # Allocate GIN-compatible tensors
            input_tensor = op.alloc_buffer((max_input_tokens, D))
            output_tensor = op.alloc_buffer((total_output_tokens, D))

            for peer in range(self.world_size):
                start = peer * tokens_per_peer
                input_tensor[start : start + tokens_per_peer] = float(
                    self.rank * 1000 + peer
                )

            output = op.alltoallv(
                input_tensor, output_tensor, output_split_sizes, input_split_sizes
            )

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
        total_output_tokens = tokens_per_peer * self.world_size
        num_iterations = 5

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
            # Allocate GIN-compatible tensors
            input_tensor = op.alloc_buffer((max_input_tokens, D))
            output_tensor = op.alloc_buffer((total_output_tokens, D))

            for peer in range(self.world_size):
                start = peer * tokens_per_peer
                input_tensor[start : start + tokens_per_peer] = float(
                    self.rank * 1000 + peer
                )
            output = None
            for _iteration in range(num_iterations):
                output = op.alltoallv(
                    input_tensor, output_tensor, output_split_sizes, input_split_sizes
                )
            torch.cuda.synchronize()

            assert output is not None
            total_tokens = tokens_per_peer * self.world_size
            self.assertEqual(output.shape, (total_tokens, D))
            self._verify_packed_data(
                output, D, output_split_sizes, "sync_buffer_repeated"
            )


class TestOpSyncBufferPackedOutput(_OpTestBase):
    """Test AlltoallvOp sync_buffer with packed output."""

    def test_sync_buffer_packed_output(self) -> None:
        from comms.pipes.collectives.triton import AlltoallvOp

        tokens_per_peer = 64
        D = 16
        max_input_tokens = tokens_per_peer * self.world_size
        total_output_tokens = tokens_per_peer * self.world_size

        input_split_sizes = torch.full(
            (self.world_size,), tokens_per_peer, dtype=torch.int64, device=self.device
        )
        output_split_sizes = input_split_sizes.clone()
        total_packed_tokens = int(output_split_sizes.sum().item())

        # Create op with sync_buffer
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
            # Allocate GIN-compatible tensors
            input_tensor = op.alloc_buffer((max_input_tokens, D))
            output_tensor = op.alloc_buffer((total_output_tokens, D))

            for peer in range(self.world_size):
                start = peer * tokens_per_peer
                input_tensor[start : start + tokens_per_peer] = float(
                    self.rank * 1000 + peer
                )

            # Pass packed_output_tokens to enable CUDA graph compatibility
            output = op.alltoallv(
                input_tensor,
                output_tensor,
                output_split_sizes,
                input_split_sizes,
                packed_output_tokens=total_packed_tokens,
            )

        torch.cuda.synchronize()
        self.assertEqual(output.shape, (total_packed_tokens, D))


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

        # Create op with sync_buffer=False
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
# Test: Full Lifecycle Graph Capture
# =============================================================================


class TestOpFullLifecycleGraph(_OpTestBase):
    """Test the complete AlltoallvOp lifecycle captured inside a CUDA graph.

    This test intentionally captures the **entire op lifecycle** inside
    ``torch.cuda.graph()`` to validate that AlltoallvOp's setup path
    (GIN registration, window creation, buffer allocation) works correctly
    under CUDA graph capture:

        1. Warmup   – A separate throwaway op compiles Triton kernels eagerly
                      (JIT compilation cannot happen during graph capture).
        2. Capture  – Inside ``torch.cuda.graph()``:
                      a. AlltoallvOp(...)       – creation
                      b. op.setup()             – comms setup + memory pool
                      c. op.alloc_buffer(...)   – GIN-compatible tensor allocation
                      d. input fill             – populate send data
                      e. op.alltoallv(...)      – first call (triggers GIN registration)
                      f. op.alltoallv(...)      – second call captured for replay
        3. Replay   – graph.replay() multiple times, verifying correctness
        4. Teardown – op.teardown() after graph usage is complete

    NOTE: This test exercises a code path that requires GIN window
    registration and buffer setup to be graph-capture-aware.  If the
    underlying C++ layer does not yet support these operations during
    capture the test will fail with an error such as
    ``RuntimeError: Window not initialized``.
    """

    def test_full_lifecycle_graph(self) -> None:
        import traceback

        from comms.pipes.collectives.triton import AlltoallvOp

        tokens_per_peer = 32
        D = 16
        max_input_tokens = tokens_per_peer * self.world_size
        total_output_tokens = tokens_per_peer * self.world_size
        packed_output_tokens = tokens_per_peer * self.world_size
        num_replays = 3

        input_split_sizes = torch.full(
            (self.world_size,), tokens_per_peer, dtype=torch.int64, device=self.device
        )
        output_split_sizes = input_split_sizes.clone()

        # ── Warmup phase ─────────────────────────────────────────────────
        # A separate op instance compiles the Triton kernels eagerly.
        # This is required because Triton JIT compilation cannot happen
        # during CUDA graph capture.
        warmup_op = AlltoallvOp(
            self.torchcomm,
            max_input_tokens,
            D,
            self.dtype,
            self.device,
            max_recv_tokens_per_peer=tokens_per_peer,
            sync_buffer=True,
        )
        with warmup_op:
            w_in = warmup_op.alloc_buffer((max_input_tokens, D))
            w_out = warmup_op.alloc_buffer((total_output_tokens, D))
            w_in.fill_(1.0)
            warmup_op.alltoallv(
                w_in,
                w_out,
                output_split_sizes,
                input_split_sizes,
                packed_output_tokens=packed_output_tokens,
            )
            torch.cuda.synchronize()

        # ── Graph capture: full lifecycle ────────────────────────────────
        graph_stream = torch.cuda.Stream()
        op = None
        graph = None
        graph_output = None

        try:
            with torch.cuda.stream(graph_stream):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    # (a) Creation
                    op = AlltoallvOp(
                        self.torchcomm,
                        max_input_tokens,
                        D,
                        self.dtype,
                        self.device,
                        max_recv_tokens_per_peer=tokens_per_peer,
                        sync_buffer=True,
                    )

                    # (b) Setup (memory pool + completion counters + offsets)
                    op.setup()

                    # (c) Buffer allocation from GIN-compatible pool
                    input_tensor = op.alloc_buffer((max_input_tokens, D))
                    output_tensor = op.alloc_buffer((total_output_tokens, D))

                    # (d) Fill input: value = rank * 1000 + dest_peer
                    for peer in range(self.world_size):
                        start = peer * tokens_per_peer
                        input_tensor[start : start + tokens_per_peer] = float(
                            self.rank * 1000 + peer
                        )

                    # (e) First alltoallv – triggers GIN registration
                    op.alltoallv(
                        input_tensor,
                        output_tensor,
                        output_split_sizes,
                        input_split_sizes,
                        packed_output_tokens=packed_output_tokens,
                    )

                    # (f) Second alltoallv – the one that will be replayed
                    graph_output = op.alltoallv(
                        input_tensor,
                        output_tensor,
                        output_split_sizes,
                        input_split_sizes,
                        packed_output_tokens=packed_output_tokens,
                    )
        except Exception:
            # Print immediately so the traceback is visible in MPI output
            # even if the process hangs during cleanup.
            print(
                f"\n{'=' * 60}\n[Rank {self.rank}] Graph capture failed:\n{'=' * 60}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()

            # Neutralise NCCL state on the op so that neither teardown()
            # nor the GC destructor attempt collective/GIN operations that
            # would hang (the window was never fully initialised during
            # capture).
            if op is not None:
                op._window = None
                op._src_info = None
                op._dev_win_ptr = None
                op._buffer_pool = None
                op._setup_done = False
                op._buffers_owned = False

            # Drop the partially-captured graph before it can be GC'd by
            # tearDownClass' gc.collect() → CUDAGraph.__del__ would block
            # on captured-but-never-executed NCCL ops.
            graph = None

            # Force-exit because torch.cuda.synchronize() in tearDownClass
            # will hang on the NCCL ops that were captured into the graph
            # stream but never executed.
            os._exit(1)

        torch.cuda.synchronize()

        # ── Replay and validate ──────────────────────────────────────────
        for replay in range(num_replays):
            with torch.cuda.stream(graph_stream):
                graph.replay()
            graph_stream.synchronize()

            replay_output = graph_output.clone()

            self.assertEqual(
                replay_output.shape,
                (packed_output_tokens, D),
                f"Replay {replay}: unexpected output shape",
            )

            offset = 0
            for peer in range(self.world_size):
                count = int(output_split_sizes[peer].item())
                actual = replay_output[offset : offset + count, :].cpu()
                expected_value = float(peer * 1000 + self.rank)
                expected = torch.full_like(actual, expected_value)
                torch.testing.assert_close(
                    actual,
                    expected,
                    msg=(
                        f"Replay {replay}, Rank {self.rank}: Data from "
                        f"peer {peer} incorrect. Expected {expected_value}, "
                        f"got {actual[0, 0].item()}"
                    ),
                )
                offset += count

        # ── Teardown ─────────────────────────────────────────────────────
        op.teardown()
        torch.cuda.synchronize()
        self.torchcomm.barrier(False)


# =============================================================================
# Test: release_buffers() API
# =============================================================================


class TestOpReleaseBuffers(_OpTestBase):
    """Test release_buffers() allows switching to new tensors."""

    def test_release_buffers_allows_new_tensors(self) -> None:
        from comms.pipes.collectives.triton import AlltoallvOp

        tokens_per_peer = 32
        D = 16
        max_input_tokens = tokens_per_peer * self.world_size
        total_output_tokens = tokens_per_peer * self.world_size

        split_sizes = torch.full(
            (self.world_size,), tokens_per_peer, dtype=torch.int64, device=self.device
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
            # Allocate first set of GIN-compatible tensors
            input_tensor1 = op.alloc_buffer((max_input_tokens, D))
            output_tensor1 = op.alloc_buffer((total_output_tokens, D))

            for peer in range(self.world_size):
                start = peer * tokens_per_peer
                input_tensor1[start : start + tokens_per_peer] = float(
                    self.rank * 1000 + peer
                )

            # First call takes ownership of tensor1 set
            output1 = op.alltoallv(
                input_tensor1, output_tensor1, split_sizes, split_sizes
            )
            torch.cuda.synchronize()

            # Release buffers
            op.release_buffers()

            # Allocate second set of GIN-compatible tensors
            input_tensor2 = op.alloc_buffer((max_input_tokens, D))
            output_tensor2 = op.alloc_buffer((total_output_tokens, D))

            for peer in range(self.world_size):
                start = peer * tokens_per_peer
                input_tensor2[start : start + tokens_per_peer] = float(
                    self.rank * 1000 + peer + 100
                )

            # Should work with new tensors after release
            output2 = op.alltoallv(
                input_tensor2, output_tensor2, split_sizes, split_sizes
            )
            torch.cuda.synchronize()

            # Verify output shapes
            total_tokens = tokens_per_peer * self.world_size
            self.assertEqual(output1.shape, (total_tokens, D))
            self.assertEqual(output2.shape, (total_tokens, D))


# =============================================================================
# Test: Intentional Race Condition for Debugging
# =============================================================================


class TestOpRaceConditionDebug(_OpTestBase):
    """
    Test that intentionally creates a race condition by delaying graph completion
    on some ranks using a spin loop kernel. This helps identify protocol violations
    by comparing failing runs with passing runs.

    The race condition occurs when:
    1. Some ranks finish their graph early and start input_tensor updates for next iteration
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
        total_output_tokens = tokens_per_peer * self.world_size
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
            # Allocate GIN-compatible tensors inside the with block
            input_tensor = op.alloc_buffer((max_input_tokens, D))
            output_tensor = op.alloc_buffer((total_output_tokens, D))

            # Pre-stage all iteration inputs to verify data integrity
            # Value encoding: iteration * 10000 + my_rank * 1000 + dest_rank
            staged_inputs = []
            for iteration in range(num_iterations):
                staged_input = op.alloc_buffer((max_input_tokens, D))
                for dest_rank in range(self.world_size):
                    start_idx = dest_rank * tokens_per_peer
                    end_idx = start_idx + tokens_per_peer
                    value = float(iteration * 10000 + self.rank * 1000 + dest_rank)
                    staged_input[start_idx:end_idx, :] = value
                staged_inputs.append(staged_input)

            # Warmup iteration to establish baseline signals
            # First call takes ownership of input_tensor and output_tensor
            input_tensor.copy_(staged_inputs[0])
            op.alltoallv(
                input_tensor,
                output_tensor,
                output_split_sizes,
                input_split_sizes,
                packed_output_tokens=packed_output_tokens,
            )
            torch.cuda.synchronize()

            # Create a staging buffer for graph capture
            # The staging buffer holds the input data that will be copied to input_tensor
            # inside the graph, AFTER the BUFFER_READY wait
            staging_buffer = op.alloc_buffer((max_input_tokens, D))

            # Capture graph with copy INSIDE the graph
            # This ensures the copy happens on the GPU stream, synchronized with
            # the BUFFER_READY wait in the alltoallv kernel
            graph_stream = torch.cuda.Stream()
            with torch.cuda.stream(graph_stream):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    # Copy from staging to input_tensor INSIDE the graph
                    # This copy will be captured and replayed
                    input_tensor.copy_(staging_buffer)
                    # Then run alltoallv
                    graph_output = op.alltoallv(
                        input_tensor,
                        output_tensor,
                        output_split_sizes,
                        input_split_sizes,
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


class TestOpRaceConditionDebugMultiBlock(_OpTestBase):
    """Test race conditions with BLOCKS_PER_PEER > 1 code paths.

    Similar to TestOpRaceConditionDebug but uses larger message sizes to trigger
    multi-block code paths (BLOCKS_PER_PEER = 8 for 64KB to 256KB messages).

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
        total_output_tokens = tokens_per_peer * self.world_size
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
            # Allocate GIN-compatible tensors inside the with block
            input_tensor = op.alloc_buffer((max_input_tokens, D))
            output_tensor = op.alloc_buffer((total_output_tokens, D))

            # Pre-stage all iteration inputs to verify data integrity
            # Value encoding: iteration * 10000 + my_rank * 1000 + dest_rank
            staged_inputs = []
            for iteration in range(num_iterations):
                staged_input = op.alloc_buffer((max_input_tokens, D))
                for dest_rank in range(self.world_size):
                    start_idx = dest_rank * tokens_per_peer
                    end_idx = start_idx + tokens_per_peer
                    value = float(iteration * 10000 + self.rank * 1000 + dest_rank)
                    staged_input[start_idx:end_idx, :] = value
                staged_inputs.append(staged_input)

            # Warmup iteration to establish baseline signals
            # First call takes ownership of input_tensor and output_tensor
            input_tensor.copy_(staged_inputs[0])
            op.alltoallv(
                input_tensor,
                output_tensor,
                output_split_sizes,
                input_split_sizes,
                packed_output_tokens=packed_output_tokens,
            )
            torch.cuda.synchronize()

            # Create a staging buffer for graph capture
            staging_buffer = op.alloc_buffer((max_input_tokens, D))

            # Capture graph with copy INSIDE the graph
            graph_stream = torch.cuda.Stream()
            with torch.cuda.stream(graph_stream):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    input_tensor.copy_(staging_buffer)
                    graph_output = op.alltoallv(
                        input_tensor,
                        output_tensor,
                        output_split_sizes,
                        input_split_sizes,
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
    TestOpUniformSplitsZeroCopy,
    TestOpRepeatedCalls,
    TestOpGetOrCreate,
    TestOpDoubleSetupRaises,
    TestOpAlltoallvWithoutSetupRaises,
    TestOpOversizedInputRaises,
    TestOpTensorIdentityEnforced,
    TestOpTeardownIdempotent,
    TestOpPackedOutputUniform,
    TestOpReleaseBuffers,
    # Sync buffer mode tests
    TestOpSyncBufferBasic,
    TestOpSyncBufferRepeatedCalls,
    TestOpSyncBufferPackedOutput,
    TestOpSyncBufferAttribute,
    # Multi-iteration tests
    TestOpMultiIterDifferentContentNonGraph,
    TestOpMultiIterDifferentContentGraph,
    TestOpMultiIterDifferentContentGraphLoop,
    # Release buffers test
    TestOpReleaseBuffers,
    # Full lifecycle graph test
    TestOpFullLifecycleGraph,
    # Race condition debug tests
    TestOpRaceConditionDebug,
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
