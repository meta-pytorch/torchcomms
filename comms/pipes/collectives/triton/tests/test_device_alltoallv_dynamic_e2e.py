# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-unsafe
"""
End-to-end integration tests for device_alltoallv_dynamic.

Each test class contains a SINGLE test to avoid P2P mapping races
between tests sharing the same NCCL allocator.  register_local_buffer
maps GPU memory via ibv_reg_mr_iova2, and deregister_local_buffer's
unmap can race with the next test's torch.zeros (cudaMemset).

Run with:
    buck2 run @fbcode//mode/opt \
        -c hpc_comms.use_ncclx=stable \
        fbcode//comms/torchcomms/triton/fb/tests:test_device_alltoallv_dynamic_e2e
"""

import gc
import os
import sys
import time
import unittest
from typing import Optional

import torch
import torchcomms
from torch.utils._triton import has_triton
from torchcomms.tests.integration.py.TorchCommTestHelpers import TorchCommTestWrapper


TRITON_AVAILABLE = has_triton()
RUN_DEVICE_API_TEST = os.environ.get("RUN_DEVICE_API_TEST", "false").lower() == "true"


def _skip_if_not_ready() -> bool:
    return TRITON_AVAILABLE and torch.cuda.is_available() and RUN_DEVICE_API_TEST


# =============================================================================
# Test Base Class
# =============================================================================


class _SingleTestBase(unittest.TestCase):
    """Allocates pool + window once, runs one test, tears down."""

    wrapper: Optional[TorchCommTestWrapper] = None
    recv_pool: Optional["torch.cuda.MemPool"] = None
    send_pool: Optional["torch.cuda.MemPool"] = None

    @classmethod
    def setUpClass(cls) -> None:
        if not _skip_if_not_ready():
            raise unittest.SkipTest("E2E test environment not ready")
        from comms.pipes.collectives.triton import prewarm_completion_counters

        torch.cuda.synchronize()
        cls.wrapper = TorchCommTestWrapper()
        cls.torchcomm = cls.wrapper.get_torchcomm()
        cls.rank = cls.torchcomm.get_rank()
        cls.world_size = cls.torchcomm.get_size()
        cls.device = cls.torchcomm.get_device()
        cls.allocator = torchcomms.get_mem_allocator(cls.torchcomm.get_backend())
        # 4MiB default pool capacity, matching NCCL DEFAULT_BUFFSIZE.
        cls.pool_capacity = 4 * 1024 * 1024

        cls.dtype = torch.float32
        alloc_elems = (
            cls.pool_capacity // torch.tensor([], dtype=cls.dtype).element_size()
        )

        # Allocate GIN-compatible buffers using torchcomms memory allocator.
        # These low-level tests don't use AlltoallvOp, so we allocate directly.
        cls.recv_pool = torch.cuda.MemPool(cls.allocator)
        with torch.cuda.use_mem_pool(cls.recv_pool):
            cls.recv_buf = torch.zeros(alloc_elems, dtype=cls.dtype, device=cls.device)

        cls.send_pool = torch.cuda.MemPool(cls.allocator)
        with torch.cuda.use_mem_pool(cls.send_pool):
            cls.send_buf = torch.zeros(alloc_elems, dtype=cls.dtype, device=cls.device)

        # Pre-allocate completion counters BEFORE GIN activation.
        # GIN (GPU-Initiated NCCL) blocks regular CUDA allocations after
        # get_device_window() is called.  The counters must be allocated
        # here while regular CUDA operations still work.
        prewarm_completion_counters(cls.world_size, cls.device)

        cls.torchcomm.barrier(False)
        cls.window = cls.torchcomm.new_window()
        cls.window.tensor_register(cls.recv_buf)
        cls.torchcomm.barrier(False)

        # Cached setup values for the lean collective API
        cls.my_rank = cls.rank
        cls.src_info = None
        cls.dev_win_ptr = None

    def setUp(self) -> None:
        """Register send buffer for one-sided operations after test data has been filled."""
        # Subclass tests call _fill_uniform / manual fill first, then register.
        # We don't register here because data isn't filled yet.
        pass

    def _register_send_buf(self) -> None:
        """Register send_buf for one-sided operations and cache dev_win_ptr."""
        if self.src_info is None:
            # Use signal_count = world_size * 2 for BUFFER_READY and DATA_COMPLETE signals
            self.dev_win_ptr = self.window.get_device_window(
                signal_count=self.world_size * 2
            )
            self.src_info = self.window.register_local_buffer(self.send_buf)
            # Reset iteration counter to match fresh signal memory state.
            # Without this, cross-test hangs occur because the iteration counter
            # persists across tests but signal memory is recreated per-window.
            from comms.pipes.collectives.triton.device_alltoallv_dynamic import (
                _reset_iteration_counter,
            )

            _reset_iteration_counter(self.world_size, self.device)

    def _deregister_send_buf(self) -> None:
        """Deregister send_buf after collective, before next data fill."""
        if self.src_info is not None:
            self.window.deregister_local_buffer(*self.src_info)
            self.src_info = None

    def tearDown(self) -> None:
        """Deregister send buffer after each test."""
        self._deregister_send_buf()

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.torchcomm is not None:
            cls.torchcomm.barrier(False)
        if hasattr(cls, "src_info") and cls.src_info is not None:
            cls.window.deregister_local_buffer(*cls.src_info)
            cls.src_info = None
        if hasattr(cls, "window") and cls.window is not None:
            cls.window.tensor_deregister()
            cls.window = None
        cls.recv_buf = None
        cls.send_buf = None
        cls.recv_pool = None
        cls.send_pool = None
        cls.allocator = None
        cls.torchcomm = None
        cls.wrapper = None
        gc.collect()
        torch.cuda.synchronize()
        time.sleep(2)

    # ---- helpers ----

    def _verify_received_data(
        self,
        recv_buf: torch.Tensor,
        local_recv_slot_offsets: torch.Tensor,
        recv_sizes: torch.Tensor,
        test_name: str = "",
    ) -> None:
        elem_size = recv_buf.element_size()
        for peer in range(self.world_size):
            recv_size = recv_sizes[peer].item()
            if recv_size == 0:
                continue
            recv_offset = local_recv_slot_offsets[peer].item()
            num_elements = recv_size // elem_size
            start_idx = recv_offset // elem_size
            actual = recv_buf[start_idx : start_idx + num_elements].cpu()
            expected_value = float(peer * 1000 + self.rank)
            expected = torch.full_like(actual, expected_value)
            torch.testing.assert_close(
                actual,
                expected,
                msg=(
                    f"[{test_name}] Rank {self.rank}: Data from peer {peer} is "
                    f"incorrect. Expected all values to be {expected_value}, "
                    f"got {actual[:5].tolist()}..."
                ),
            )

    def _fill_uniform(self, msg_size: int) -> tuple:
        """Fill shared buffers with uniform-size identifiable patterns.

        Returns (send_sizes, send_offsets, recv_sizes, local_recv_slot_offsets, remote_write_offsets).
        """
        from comms.pipes.collectives.triton import exchange_offsets

        elem_size = self.send_buf.element_size()
        num_elements_per_peer = msg_size // elem_size
        self.send_buf.zero_()
        self.recv_buf.zero_()
        for peer in range(self.world_size):
            start = peer * num_elements_per_peer
            self.send_buf[start : start + num_elements_per_peer] = float(
                self.rank * 1000 + peer
            )
        send_sizes = torch.full(
            (self.world_size,), msg_size, dtype=torch.int64, device=self.device
        )
        send_offsets = (
            torch.arange(self.world_size, dtype=torch.int64, device=self.device)
            * msg_size
        )
        recv_sizes = send_sizes.clone()
        local_recv_slot_offsets = send_offsets.clone()
        remote_write_offsets = exchange_offsets(local_recv_slot_offsets, self.torchcomm)
        return (
            send_sizes,
            send_offsets,
            recv_sizes,
            local_recv_slot_offsets,
            remote_write_offsets,
        )


# =============================================================================
# Non-Pipelined Tests
# =============================================================================


class TestUniformSizesBasic(_SingleTestBase):
    """Test basic alltoallv with uniform message sizes."""

    def test_uniform_sizes_basic(self) -> None:
        from comms.pipes.collectives.triton import device_alltoallv_dynamic

        (
            send_sizes,
            send_offsets,
            recv_sizes,
            local_recv_slot_offsets,
            remote_write_offsets,
        ) = self._fill_uniform(1024)
        self._register_send_buf()
        device_alltoallv_dynamic(
            self.send_buf,
            self.recv_buf,
            send_sizes,
            send_offsets,
            recv_sizes,
            local_recv_slot_offsets,
            remote_write_offsets,
            self.dev_win_ptr,
            self.src_info,
            self.my_rank,
            self.world_size,
        )
        torch.cuda.synchronize()
        self._verify_received_data(
            self.recv_buf, send_offsets, send_sizes, "uniform_sizes_basic"
        )


class TestLargeMessages(_SingleTestBase):
    """Test alltoallv with large messages that fill the pool."""

    def test_large_messages(self) -> None:
        from comms.pipes.collectives.triton import device_alltoallv_dynamic

        elem_size = self.send_buf.element_size()
        msg_size = (self.pool_capacity // self.world_size // elem_size) * elem_size
        (
            send_sizes,
            send_offsets,
            recv_sizes,
            local_recv_slot_offsets,
            remote_write_offsets,
        ) = self._fill_uniform(msg_size)
        self._register_send_buf()
        device_alltoallv_dynamic(
            self.send_buf,
            self.recv_buf,
            send_sizes,
            send_offsets,
            recv_sizes,
            local_recv_slot_offsets,
            remote_write_offsets,
            self.dev_win_ptr,
            self.src_info,
            self.my_rank,
            self.world_size,
        )
        torch.cuda.synchronize()
        self._verify_received_data(
            self.recv_buf, send_offsets, send_sizes, "large_messages"
        )


# =============================================================================
# Edge-Case Tests
# =============================================================================


class TestMinimumMessageSize(_SingleTestBase):
    """Test with minimum viable message size (4 bytes = 1 float32)."""

    def test_minimum_message_size(self) -> None:
        from comms.pipes.collectives.triton import device_alltoallv_dynamic

        (
            send_sizes,
            send_offsets,
            recv_sizes,
            local_recv_slot_offsets,
            remote_write_offsets,
        ) = self._fill_uniform(4)
        self._register_send_buf()
        device_alltoallv_dynamic(
            self.send_buf,
            self.recv_buf,
            send_sizes,
            send_offsets,
            recv_sizes,
            local_recv_slot_offsets,
            remote_write_offsets,
            self.dev_win_ptr,
            self.src_info,
            self.my_rank,
            self.world_size,
        )
        torch.cuda.synchronize()
        self._verify_received_data(
            self.recv_buf, send_offsets, send_sizes, "minimum_message_size"
        )


class TestRepeatedCalls(_SingleTestBase):
    """Test multiple consecutive alltoallv calls for correctness."""

    def test_repeated_calls(self) -> None:
        from comms.pipes.collectives.triton import device_alltoallv_dynamic

        msg_size = 1024
        for iteration in range(5):
            self._deregister_send_buf()
            (
                send_sizes,
                send_offsets,
                recv_sizes,
                local_recv_slot_offsets,
                remote_write_offsets,
            ) = self._fill_uniform(msg_size)
            self._register_send_buf()
            device_alltoallv_dynamic(
                self.send_buf,
                self.recv_buf,
                send_sizes,
                send_offsets,
                recv_sizes,
                local_recv_slot_offsets,
                remote_write_offsets,
                self.dev_win_ptr,
                self.src_info,
                self.my_rank,
                self.world_size,
            )
            # Must sync all ranks before next iteration's deregister +
            # exchange_offsets (a collective requiring all ranks).
            torch.cuda.synchronize()
            self.torchcomm.barrier(False)
            self._verify_received_data(
                self.recv_buf,
                send_offsets,
                send_sizes,
                f"repeated_calls_iter_{iteration}",
            )


# =============================================================================
# Num Warps Tests (Phase 1: warp parallelism tuning)
# =============================================================================


class TestNumWarps4(_SingleTestBase):
    """Test non-pipelined alltoallv with num_warps=4 (minimum, old default)."""

    def test_num_warps_4(self) -> None:
        from comms.pipes.collectives.triton import device_alltoallv_dynamic

        (
            send_sizes,
            send_offsets,
            recv_sizes,
            local_recv_slot_offsets,
            remote_write_offsets,
        ) = self._fill_uniform(1024)
        self._register_send_buf()
        device_alltoallv_dynamic(
            self.send_buf,
            self.recv_buf,
            send_sizes,
            send_offsets,
            recv_sizes,
            local_recv_slot_offsets,
            remote_write_offsets,
            self.dev_win_ptr,
            self.src_info,
            self.my_rank,
            self.world_size,
            num_warps=4,
        )
        torch.cuda.synchronize()
        self._verify_received_data(
            self.recv_buf, send_offsets, send_sizes, "num_warps_4"
        )


class TestNumWarps32(_SingleTestBase):
    """Test non-pipelined alltoallv with num_warps=32 (maximum, 1024 threads)."""

    def test_num_warps_32(self) -> None:
        from comms.pipes.collectives.triton import device_alltoallv_dynamic

        (
            send_sizes,
            send_offsets,
            recv_sizes,
            local_recv_slot_offsets,
            remote_write_offsets,
        ) = self._fill_uniform(1024)
        self._register_send_buf()
        device_alltoallv_dynamic(
            self.send_buf,
            self.recv_buf,
            send_sizes,
            send_offsets,
            recv_sizes,
            local_recv_slot_offsets,
            remote_write_offsets,
            self.dev_win_ptr,
            self.src_info,
            self.my_rank,
            self.world_size,
            num_warps=32,
        )
        torch.cuda.synchronize()
        self._verify_received_data(
            self.recv_buf, send_offsets, send_sizes, "num_warps_32"
        )


# =============================================================================
# Blocks-Per-Peer Tests (Phase 2: block parallelism tuning)
# =============================================================================


class TestBlocksPerPeer2Uniform(_SingleTestBase):
    """Test non-pipelined alltoallv with blocks_per_peer=2 and uniform sizes."""

    def test_blocks_per_peer_2_uniform(self) -> None:
        from comms.pipes.collectives.triton import device_alltoallv_dynamic

        (
            send_sizes,
            send_offsets,
            recv_sizes,
            local_recv_slot_offsets,
            remote_write_offsets,
        ) = self._fill_uniform(1024)
        self._register_send_buf()
        device_alltoallv_dynamic(
            self.send_buf,
            self.recv_buf,
            send_sizes,
            send_offsets,
            recv_sizes,
            local_recv_slot_offsets,
            remote_write_offsets,
            self.dev_win_ptr,
            self.src_info,
            self.my_rank,
            self.world_size,
            blocks_per_peer=2,
        )
        torch.cuda.synchronize()
        self._verify_received_data(
            self.recv_buf, send_offsets, send_sizes, "blocks_per_peer_2_uniform"
        )


class TestBlocksPerPeer4Uniform(_SingleTestBase):
    """Test non-pipelined alltoallv with blocks_per_peer=4 and uniform sizes."""

    def test_blocks_per_peer_4_uniform(self) -> None:
        from comms.pipes.collectives.triton import device_alltoallv_dynamic

        (
            send_sizes,
            send_offsets,
            recv_sizes,
            local_recv_slot_offsets,
            remote_write_offsets,
        ) = self._fill_uniform(1024)
        self._register_send_buf()
        device_alltoallv_dynamic(
            self.send_buf,
            self.recv_buf,
            send_sizes,
            send_offsets,
            recv_sizes,
            local_recv_slot_offsets,
            remote_write_offsets,
            self.dev_win_ptr,
            self.src_info,
            self.my_rank,
            self.world_size,
            blocks_per_peer=4,
        )
        torch.cuda.synchronize()
        self._verify_received_data(
            self.recv_buf, send_offsets, send_sizes, "blocks_per_peer_4_uniform"
        )


class TestBlocksPerPeer4Large(_SingleTestBase):
    """Test non-pipelined alltoallv with blocks_per_peer=4 and large messages.

    Large messages are the primary use case for blocks_per_peer > 1:
    multiple blocks issue parallel put_block calls, each copying a chunk,
    which increases aggregate NVLink bandwidth utilization.
    """

    def test_blocks_per_peer_4_large(self) -> None:
        from comms.pipes.collectives.triton import device_alltoallv_dynamic

        elem_size = self.send_buf.element_size()
        msg_size = (self.pool_capacity // self.world_size // elem_size) * elem_size
        (
            send_sizes,
            send_offsets,
            recv_sizes,
            local_recv_slot_offsets,
            remote_write_offsets,
        ) = self._fill_uniform(msg_size)
        self._register_send_buf()
        device_alltoallv_dynamic(
            self.send_buf,
            self.recv_buf,
            send_sizes,
            send_offsets,
            recv_sizes,
            local_recv_slot_offsets,
            remote_write_offsets,
            self.dev_win_ptr,
            self.src_info,
            self.my_rank,
            self.world_size,
            blocks_per_peer=4,
        )
        torch.cuda.synchronize()
        self._verify_received_data(
            self.recv_buf, send_offsets, send_sizes, "blocks_per_peer_4_large"
        )


class TestBlocksPerPeerConsistency(_SingleTestBase):
    """Verify blocks_per_peer=1 and blocks_per_peer=4 produce identical results.

    The multi-block path splits each peer's data into BLOCKS_PER_PEER
    chunks and reassembles via independent put_block calls.  This test
    confirms that chunked transfer produces byte-identical recv_buf
    contents as the single-block baseline.
    """

    def test_blocks_per_peer_consistency(self) -> None:
        from comms.pipes.collectives.triton import device_alltoallv_dynamic

        msg_size = 1024
        (
            send_sizes,
            send_offsets,
            recv_sizes,
            local_recv_slot_offsets,
            remote_write_offsets,
        ) = self._fill_uniform(msg_size)

        # Run with blocks_per_peer=1 (iteration 0)
        self._register_send_buf()
        device_alltoallv_dynamic(
            self.send_buf,
            self.recv_buf,
            send_sizes,
            send_offsets,
            recv_sizes,
            local_recv_slot_offsets,
            remote_write_offsets,
            self.dev_win_ptr,
            self.src_info,
            self.my_rank,
            self.world_size,
            blocks_per_peer=1,
        )
        torch.cuda.synchronize()
        # Deregister before clone() — GIN prevents regular CUDA allocations
        self._deregister_send_buf()
        result_single_block = self.recv_buf.clone()

        # Re-fill and run with blocks_per_peer=4 (iteration 1)
        (
            send_sizes,
            send_offsets,
            recv_sizes,
            local_recv_slot_offsets,
            remote_write_offsets,
        ) = self._fill_uniform(msg_size)
        self._register_send_buf()
        device_alltoallv_dynamic(
            self.send_buf,
            self.recv_buf,
            send_sizes,
            send_offsets,
            recv_sizes,
            local_recv_slot_offsets,
            remote_write_offsets,
            self.dev_win_ptr,
            self.src_info,
            self.my_rank,
            self.world_size,
            blocks_per_peer=4,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(
            result_single_block,
            self.recv_buf,
            msg=(
                f"Rank {self.rank}: blocks_per_peer=1 and blocks_per_peer=4 "
                f"results differ!"
            ),
        )


class TestRepeatedCallsMultiBlock(_SingleTestBase):
    """Test multiple iterations with blocks_per_peer > 1.

    This test validates the monotonic signal counter fix for the race
    condition identified in review comment #6 (line 257).  With the old
    code, block scheduling non-determinism could cause:
        block1(inc→1) → block0(reset→0) → signal(0) ← WRONG!

    The fix uses monotonic signal values:
        iteration 0 signals BLOCKS_PER_PEER * 1
        iteration 1 signals BLOCKS_PER_PEER * 2
        etc.

    This test verifies that multiple iterations with blocks_per_peer > 1
    complete successfully without races or deadlocks.
    """

    def test_repeated_calls_multi_block(self) -> None:
        from comms.pipes.collectives.triton import device_alltoallv_dynamic

        msg_size = 2048  # Larger message to exercise chunking
        num_iterations = 5
        blocks_per_peer = 4

        for iteration in range(num_iterations):
            self._deregister_send_buf()
            (
                send_sizes,
                send_offsets,
                recv_sizes,
                local_recv_slot_offsets,
                remote_write_offsets,
            ) = self._fill_uniform(msg_size)
            self._register_send_buf()
            device_alltoallv_dynamic(
                self.send_buf,
                self.recv_buf,
                send_sizes,
                send_offsets,
                recv_sizes,
                local_recv_slot_offsets,
                remote_write_offsets,
                self.dev_win_ptr,
                self.src_info,
                self.my_rank,
                self.world_size,
                blocks_per_peer=blocks_per_peer,
            )
            # Must sync all ranks before next iteration's deregister +
            # exchange_offsets (a collective requiring all ranks).
            torch.cuda.synchronize()
            self.torchcomm.barrier(False)
            self._verify_received_data(
                self.recv_buf,
                send_offsets,
                send_sizes,
                f"repeated_calls_multi_block_iter_{iteration}",
            )


class TestRepeatedCallsVaryingBlocksPerPeer(_SingleTestBase):
    """Test iterations with varying blocks_per_peer values.

    This test validates that the monotonic signal counter works correctly
    when blocks_per_peer changes between iterations.  The signal value
    is BLOCKS_PER_PEER * (iteration + 1), which must be correctly
    computed for each call regardless of previous calls' BLOCKS_PER_PEER.
    """

    def test_repeated_calls_varying_blocks_per_peer(self) -> None:
        from comms.pipes.collectives.triton import device_alltoallv_dynamic

        msg_size = 2048
        # Test with varying blocks_per_peer across iterations
        blocks_per_peer_sequence = [1, 2, 4, 2, 1, 8]

        for iteration, bpp in enumerate(blocks_per_peer_sequence):
            self._deregister_send_buf()
            (
                send_sizes,
                send_offsets,
                recv_sizes,
                local_recv_slot_offsets,
                remote_write_offsets,
            ) = self._fill_uniform(msg_size)
            self._register_send_buf()
            device_alltoallv_dynamic(
                self.send_buf,
                self.recv_buf,
                send_sizes,
                send_offsets,
                recv_sizes,
                local_recv_slot_offsets,
                remote_write_offsets,
                self.dev_win_ptr,
                self.src_info,
                self.my_rank,
                self.world_size,
                blocks_per_peer=bpp,
            )
            torch.cuda.synchronize()
            self.torchcomm.barrier(False)
            self._verify_received_data(
                self.recv_buf,
                send_offsets,
                send_sizes,
                f"varying_bpp_iter_{iteration}_bpp_{bpp}",
            )


# =============================================================================
# Sync Buffer Mode Tests
# =============================================================================


class TestSyncBufferBasic(_SingleTestBase):
    """Test sync_buffer=True with basic uniform message sizes."""

    def test_sync_buffer_uniform_single_iteration(self) -> None:
        """Test sync_buffer=True works correctly with single iteration."""
        from comms.pipes.collectives.triton import device_alltoallv_dynamic

        (
            send_sizes,
            send_offsets,
            recv_sizes,
            local_recv_slot_offsets,
            remote_write_offsets,
        ) = self._fill_uniform(1024)
        self._register_send_buf()
        device_alltoallv_dynamic(
            self.send_buf,
            self.recv_buf,
            send_sizes,
            send_offsets,
            recv_sizes,
            local_recv_slot_offsets,
            remote_write_offsets,
            self.dev_win_ptr,
            self.src_info,
            self.my_rank,
            self.world_size,
            sync_buffer=True,  # Enable buffer-ready synchronization
        )
        torch.cuda.synchronize()
        self._verify_received_data(
            self.recv_buf, send_offsets, send_sizes, "sync_buffer_single_iter"
        )


class TestSyncBufferRepeatedCalls(_SingleTestBase):
    """Test sync_buffer=True with repeated calls (critical for buffer-ready sync)."""

    def test_sync_buffer_repeated_calls(self) -> None:
        """Test sync_buffer=True works correctly with multiple iterations.

        This is the key test for sync_buffer mode - it verifies that buffer-ready
        synchronization correctly prevents sender from overwriting data that
        receiver hasn't consumed yet.

        IMPORTANT: This test does NOT refill buffers between iterations because:
        1. get_device_window() enables GIN, which blocks regular CUDA ops like zero_()
        2. Signal buffers must persist across iterations for sync_buffer mode to work

        The test verifies correctness by running multiple iterations with the same
        data and checking that no hangs or data corruption occur.
        """
        from comms.pipes.collectives.triton import device_alltoallv_dynamic

        msg_size = 1024
        num_iterations = 5

        # Setup buffers ONCE before all iterations (before GIN is activated)
        (
            send_sizes,
            send_offsets,
            recv_sizes,
            local_recv_slot_offsets,
            remote_write_offsets,
        ) = self._fill_uniform(msg_size)

        # Register send buffer ONCE (this activates GIN)
        self._register_send_buf()

        for _iteration in range(num_iterations):
            device_alltoallv_dynamic(
                self.send_buf,
                self.recv_buf,
                send_sizes,
                send_offsets,
                recv_sizes,
                local_recv_slot_offsets,
                remote_write_offsets,
                self.dev_win_ptr,
                self.src_info,
                self.my_rank,
                self.world_size,
                sync_buffer=True,  # Enable buffer-ready synchronization
            )
            torch.cuda.synchronize()
            self.torchcomm.barrier(False)

        # Verify final data after all iterations complete
        self._verify_received_data(
            self.recv_buf,
            send_offsets,
            send_sizes,
            "sync_buffer_repeated_calls",
        )


class TestSyncBufferMultiBlock(_SingleTestBase):
    """Test sync_buffer=True with multi-block per peer configuration."""

    def test_sync_buffer_multi_block(self) -> None:
        """Test sync_buffer=True with blocks_per_peer > 1."""
        from comms.pipes.collectives.triton import device_alltoallv_dynamic

        msg_size = 2048  # Larger message for multi-block
        blocks_per_peer = 4

        (
            send_sizes,
            send_offsets,
            recv_sizes,
            local_recv_slot_offsets,
            remote_write_offsets,
        ) = self._fill_uniform(msg_size)
        self._register_send_buf()
        device_alltoallv_dynamic(
            self.send_buf,
            self.recv_buf,
            send_sizes,
            send_offsets,
            recv_sizes,
            local_recv_slot_offsets,
            remote_write_offsets,
            self.dev_win_ptr,
            self.src_info,
            self.my_rank,
            self.world_size,
            blocks_per_peer=blocks_per_peer,
            sync_buffer=True,
        )
        torch.cuda.synchronize()
        self._verify_received_data(
            self.recv_buf, send_offsets, send_sizes, "sync_buffer_multi_block"
        )


# =============================================================================
# Test Registry
# =============================================================================

ALL_TEST_CLASSES = [
    TestUniformSizesBasic,
    TestLargeMessages,
    TestMinimumMessageSize,
    TestRepeatedCalls,
    TestNumWarps4,
    TestNumWarps32,
    TestBlocksPerPeer2Uniform,
    TestBlocksPerPeer4Uniform,
    TestBlocksPerPeer4Large,
    TestBlocksPerPeerConsistency,
    TestRepeatedCallsMultiBlock,
    TestRepeatedCallsVaryingBlocksPerPeer,
    # Sync buffer mode tests
    TestSyncBufferBasic,
    TestSyncBufferRepeatedCalls,
    TestSyncBufferMultiBlock,
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

        # If the filter matched nothing, fall back to running all tests.
        # This handles the case where TEST_FILTER is set to the buck target
        # name (e.g. "TestDeviceAlltoallvDynamicE2E") rather than a specific
        # test class or method name.
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
