#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import unittest

import torch
import torchcomms
from torchcomms.tests.helpers.py.cuda_graph_test_helpers import (
    analyze_cuda_graph,
    CudaGraphTestBase,
    probe_tensor_addr,
    skip_unless_ncclx,
)


# stolen from WindowRmaTest.py
def _should_skip_rma_test() -> tuple[bool, str]:
    """Check if RMA tests should be skipped.

    RMA window ops require the ncclx backend with CTran enabled.
    Returns (should_skip, reason).
    """
    if os.getenv("TEST_BACKEND", "").lower() != "ncclx":
        return True, "RMA window ops require ncclx backend"
    if os.getenv("NCCL_CTRAN_ENABLE", "").lower() not in (
        "1",
        "y",
        "yes",
        "t",
        "true",
    ):
        return True, "RMA window ops require ctran (NCCL_CTRAN_ENABLE not set)"
    return False, ""


_rma_skip: bool
_rma_skip_reason: str
_rma_skip, _rma_skip_reason = _should_skip_rma_test()

_gin_skip: bool = os.getenv("NCCL_GIN_ENABLE", "0") not in (
    "1",
    "y",
    "yes",
    "t",
    "true",
)
_gin_skip_reason: str = "Device-side window tests require GIN (NCCL_GIN_ENABLE not set)"


class TestWindowGraphCapture(CudaGraphTestBase):
    """Tests that window registration (winRegister/commRegister) works
    correctly during CUDA graph capture."""

    NUM_REPLAYS = 2
    ELEM_COUNT = 1024

    @skip_unless_ncclx
    @unittest.skipIf(_rma_skip, _rma_skip_reason)
    def test_window_register_and_reuse_during_capture(self) -> None:
        """Verify NCCL pool memory reuse, then register the reused buffer
        during CUDA graph capture and run captured RMA ops.

        Pool reuse is validated before capture (alloc/free/realloc from the
        NCCL MemPool).  Registration happens during capture (relaxed mode).
        RMA put/signal/wait_signal are captured and replayed with correctness
        verification."""
        with self.create_comms(1) as comms:
            comm = comms[0]
            rank = comm.get_rank()
            size = comm.get_size()
            count = self.ELEM_COUNT
            buf_numel = count * size

            dst_rank = (rank + 1) % size
            src_rank = (rank - 1 + size) % size

            src_data = torch.ones(count, dtype=torch.float32, device=self.device) * rank

            put_stream = torch.cuda.Stream()
            wait_stream = torch.cuda.Stream()

            win = comm.new_window()
            comm.barrier(False)

            addr_probe = torch.zeros(1, dtype=torch.int64, device=self.device)
            addr_probe_2 = torch.zeros(1, dtype=torch.int64, device=self.device)

            received_snapshot = torch.zeros(
                count, dtype=torch.float32, device=self.device
            )

            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph):
                win_buf = torch.zeros(
                    buf_numel, dtype=torch.float32, device=self.device
                )

                win.tensor_register(win_buf)

                initial_stream = torch.cuda.current_stream()
                put_stream.wait_stream(initial_stream)
                with torch.cuda.stream(put_stream):
                    win.put(src_data, dst_rank, dst_rank * count, False)
                    win.signal(dst_rank, False)

                wait_stream.wait_stream(put_stream)
                with torch.cuda.stream(wait_stream):
                    win.wait_signal(src_rank, False)

                initial_stream.wait_stream(wait_stream)

                received_snapshot.copy_(win_buf[rank * count : (rank + 1) * count])

                probe_tensor_addr(win_buf, addr_probe)

                win.tensor_deregister()
                del win_buf

                new_buf = torch.zeros(
                    buf_numel, dtype=torch.float32, device=self.device
                )

                probe_tensor_addr(new_buf, addr_probe_2)

            info = analyze_cuda_graph(graph)
            self.assertGreater(
                info.num_kernel_nodes, 0, "Expected kernel nodes in graph"
            )
            memcpy_nodes = info.memcpy_nodes()
            self.assertGreater(len(memcpy_nodes), 0, "Expected MEMCPY node for NVL put")
            signal_kernels = info.kernels_with_name("ncclKernelSignal")
            self.assertEqual(
                len(signal_kernels), 1, "Expected one ncclKernelSignal kernel"
            )

            graph.instantiate()
            comm.barrier(False)
            for replay in range(self.NUM_REPLAYS):
                addr_probe.zero_()
                addr_probe_2.zero_()

                torch.cuda.synchronize()
                comm.barrier(False)

                graph.replay()

                torch.cuda.synchronize()
                comm.barrier(False)

                expected = (
                    torch.ones(count, dtype=torch.float32, device=self.device)
                    * src_rank
                )
                torch.testing.assert_close(
                    received_snapshot,
                    expected,
                    rtol=1e-5,
                    atol=1e-5,
                    msg=(
                        f"Replay {replay}: rank {rank} expected data from "
                        f"rank {src_rank}, got {received_snapshot[:8].tolist()}"
                    ),
                )

                graph_ptr_1 = addr_probe.item()
                graph_ptr_2 = addr_probe_2.item()
                self.assertNotEqual(graph_ptr_1, 0, "Expected non-zero buffer address")
                self.assertNotEqual(graph_ptr_2, 0, "Expected non-zero buffer address")
                self.assertEqual(
                    graph_ptr_1,
                    graph_ptr_2,
                    f"Expected graph to reuse buffer address: "
                    f"win_buf=0x{graph_ptr_1:x}, new_buf=0x{graph_ptr_2:x}",
                )

            del win
            torch.cuda.synchronize()

    @skip_unless_ncclx
    @unittest.skipIf(_rma_skip, _rma_skip_reason)
    def test_window_rma_ops_during_capture(self) -> None:
        """Minimal reproducer for RMA ops during CUDA graph capture.

        Registration is done BEFORE capture (known to work).  Only
        put/signal/wait_signal are inside the capture context.  This isolates
        the GPE capture-safety question from the registration question."""
        with self.create_comms(1) as comms:
            comm = comms[0]
            rank = comm.get_rank()
            size = comm.get_size()
            count = self.ELEM_COUNT
            buf_numel = count * size

            dst_rank = (rank + 1) % size
            src_rank = (rank - 1 + size) % size

            allocator = torchcomms.get_mem_allocator(comm.get_backend())
            pool = torch.cuda.MemPool(allocator)
            with torch.cuda.use_mem_pool(pool):
                win_buf = torch.zeros(
                    buf_numel, dtype=torch.float32, device=self.device
                )

            src_data = torch.ones(count, dtype=torch.float32, device=self.device) * rank

            win = comm.new_window()
            win.tensor_register(win_buf)
            comm.barrier(False)

            put_stream = torch.cuda.Stream()
            wait_stream = torch.cuda.Stream()

            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph):
                initial_stream = torch.cuda.current_stream()
                put_stream.wait_stream(initial_stream)
                with torch.cuda.stream(put_stream):
                    win.put(src_data, dst_rank, dst_rank * count, False)
                    win.signal(dst_rank, False)

                wait_stream.wait_stream(put_stream)
                with torch.cuda.stream(wait_stream):
                    win.wait_signal(src_rank, False)

                initial_stream.wait_stream(wait_stream)

            info = analyze_cuda_graph(graph)
            self.assertGreater(
                info.num_kernel_nodes, 0, "Expected RMA kernel nodes in graph"
            )

            graph.instantiate()
            comm.barrier(False)
            for replay in range(self.NUM_REPLAYS):
                win_buf.zero_()
                torch.cuda.synchronize()
                comm.barrier(False)

                graph.replay()
                torch.cuda.synchronize()
                comm.barrier(False)

                local_tensor = win.map_remote_tensor(rank)
                received = local_tensor[rank * count : (rank + 1) * count]
                expected = (
                    torch.ones(count, dtype=torch.float32, device=self.device)
                    * src_rank
                )
                torch.testing.assert_close(
                    received,
                    expected,
                    rtol=1e-5,
                    atol=1e-5,
                    msg=(
                        f"Replay {replay}: rank {rank} expected data from "
                        f"rank {src_rank}, got {received[:8]}..."
                    ),
                )

            win.tensor_deregister()
            del win
            graph.reset()
            del pool
            torch.cuda.synchronize()

    @skip_unless_ncclx
    @unittest.skipIf(_rma_skip, _rma_skip_reason)
    @unittest.skipIf(_gin_skip, _gin_skip_reason)
    def test_device_window_memory_reuse_during_capture(self) -> None:
        """Verify device window creation during CUDA graph capture with pool
        memory reuse.

        Tests that:
        1. get_device_window() works during graph capture (uses
           StreamCaptureModeGuard internally to execute eagerly).
        2. register_local_buffer() works during graph capture.
        3. Pool memory is reused after deregister+free because the device
           window stores only a raw pointer (buf_data_ptr_) instead of
           holding a tensor reference (buf_tensor_).
        """
        with self.create_comms(1) as comms:
            comm = comms[0]
            rank = comm.get_rank()
            size = comm.get_size()
            count = self.ELEM_COUNT
            buf_numel = count * size

            dst_rank = (rank + 1) % size
            src_rank = (rank - 1 + size) % size

            src_data = torch.ones(count, dtype=torch.float32, device=self.device) * rank

            # Use NCCL memory allocator so buffers are VMM-allocated
            # (cuMemCreate). This is required for register_extra_window
            # (NCCL_WIN_DEVICE_API) which calls cuMemRetainAllocationHandle
            # — only works on VMM memory, not cudaMallocAsync pool memory.
            allocator = torchcomms.get_mem_allocator(comm.get_backend())
            pool = torch.cuda.MemPool(allocator)

            put_stream = torch.cuda.Stream()
            wait_stream = torch.cuda.Stream()

            win = comm.new_window()
            comm.barrier(False)

            addr_probe = torch.zeros(1, dtype=torch.int64, device=self.device)
            addr_probe_2 = torch.zeros(1, dtype=torch.int64, device=self.device)

            received_snapshot = torch.zeros(
                count, dtype=torch.float32, device=self.device
            )

            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph):
                with torch.cuda.use_mem_pool(pool):
                    win_buf = torch.zeros(
                        buf_numel, dtype=torch.float32, device=self.device
                    )

                win.tensor_register(win_buf)

                # Device window creation during capture — the C++ layer uses
                # StreamCaptureModeGuard so devCommCreate/cudaMalloc/cudaMemcpy
                # execute eagerly rather than being captured.
                has_device_api = hasattr(win, "get_device_window")
                if has_device_api:
                    # pyre-ignore[16]: device API methods on NCCLX subclass
                    dev_win_ptr = win.get_device_window(signal_count=size)

                    # Local buffer registration during capture — also uses
                    # StreamCaptureModeGuard for commWindowRegister.
                    # pyre-ignore[16]: device API methods on NCCLX subclass
                    src_buf_info = win.register_local_buffer(src_data)

                # RMA ops captured into the graph
                initial_stream = torch.cuda.current_stream()
                put_stream.wait_stream(initial_stream)
                with torch.cuda.stream(put_stream):
                    win.put(src_data, dst_rank, dst_rank * count, False)
                    win.signal(dst_rank, False)

                wait_stream.wait_stream(put_stream)
                with torch.cuda.stream(wait_stream):
                    win.wait_signal(src_rank, False)

                initial_stream.wait_stream(wait_stream)

                received_snapshot.copy_(win_buf[rank * count : (rank + 1) * count])

                probe_tensor_addr(win_buf, addr_probe)

                # tensor_deregister is a no-op during capture; the window
                # keeps the raw data pointer via buf_data_ptr_.
                win.tensor_deregister()
                del win_buf

                with torch.cuda.use_mem_pool(pool):
                    new_buf = torch.zeros(
                        buf_numel, dtype=torch.float32, device=self.device
                    )

                probe_tensor_addr(new_buf, addr_probe_2)

            if has_device_api:
                self.assertIsInstance(dev_win_ptr, int)  # pyre-ignore[61]
                self.assertNotEqual(
                    dev_win_ptr,  # pyre-ignore[61]
                    0,
                    "Device window pointer should be non-zero",
                )

            info = analyze_cuda_graph(graph)
            self.assertGreater(
                info.num_kernel_nodes, 0, "Expected kernel nodes in graph"
            )
            memcpy_nodes = info.memcpy_nodes()
            self.assertGreater(len(memcpy_nodes), 0, "Expected MEMCPY node for NVL put")

            graph.instantiate()
            comm.barrier(False)
            for replay in range(self.NUM_REPLAYS):
                addr_probe.zero_()
                addr_probe_2.zero_()

                torch.cuda.synchronize()
                comm.barrier(False)

                graph.replay()

                torch.cuda.synchronize()
                comm.barrier(False)

                expected = (
                    torch.ones(count, dtype=torch.float32, device=self.device)
                    * src_rank
                )
                torch.testing.assert_close(
                    received_snapshot,
                    expected,
                    rtol=1e-5,
                    atol=1e-5,
                    msg=(
                        f"Replay {replay}: rank {rank} expected data from "
                        f"rank {src_rank}, got {received_snapshot[:8].tolist()}"
                    ),
                )

                graph_ptr_1 = addr_probe.item()
                graph_ptr_2 = addr_probe_2.item()
                self.assertNotEqual(graph_ptr_1, 0, "Expected non-zero buffer address")
                self.assertNotEqual(graph_ptr_2, 0, "Expected non-zero buffer address")
                self.assertEqual(
                    graph_ptr_1,
                    graph_ptr_2,
                    f"Expected graph to reuse buffer address: "
                    f"win_buf=0x{graph_ptr_1:x}, new_buf=0x{graph_ptr_2:x}",
                )

            if has_device_api:
                # pyre-ignore[16, 61]: device API on NCCLX subclass
                win.deregister_local_buffer(*src_buf_info)
            del win
            torch.cuda.synchronize()

    @skip_unless_ncclx
    @unittest.skipIf(_rma_skip, _rma_skip_reason)
    def test_window_nonowning_register_during_capture(self) -> None:
        """Verify non-owning tensor_register during CUDA graph capture.

        The buffer is pre-allocated from the NCCL pool before capture,
        then tensor_register is called during capture. In graph capture
        mode the window stores only a raw data pointer (buf_data_ptr_)
        rather than holding a tensor reference (buf_tensor_), so the
        registration is non-owning. The caller is responsible for
        keeping the buffer alive for the lifetime of the window.

        RMA ops are captured and replayed with correctness verification.
        """
        with self.create_comms(1) as comms:
            comm = comms[0]
            rank = comm.get_rank()
            size = comm.get_size()
            count = self.ELEM_COUNT
            buf_numel = count * size

            dst_rank = (rank + 1) % size
            src_rank = (rank - 1 + size) % size

            allocator = torchcomms.get_mem_allocator(comm.get_backend())
            pool = torch.cuda.MemPool(allocator)
            with torch.cuda.use_mem_pool(pool):
                win_buf = torch.zeros(
                    buf_numel, dtype=torch.float32, device=self.device
                )

            src_data = torch.ones(count, dtype=torch.float32, device=self.device) * rank

            win = comm.new_window()
            comm.barrier(False)

            put_stream = torch.cuda.Stream()
            wait_stream = torch.cuda.Stream()

            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph):
                # Non-owning registration: buffer is pre-allocated, window
                # stores only buf_data_ptr_ (not buf_tensor_).
                win.tensor_register(win_buf)

                initial_stream = torch.cuda.current_stream()
                put_stream.wait_stream(initial_stream)
                with torch.cuda.stream(put_stream):
                    win.put(src_data, dst_rank, dst_rank * count, False)
                    win.signal(dst_rank, False)

                wait_stream.wait_stream(put_stream)
                with torch.cuda.stream(wait_stream):
                    win.wait_signal(src_rank, False)

                initial_stream.wait_stream(wait_stream)

            info = analyze_cuda_graph(graph)
            self.assertGreater(
                info.num_kernel_nodes, 0, "Expected RMA kernel nodes in graph"
            )

            graph.instantiate()
            comm.barrier(False)
            for replay in range(self.NUM_REPLAYS):
                win_buf.zero_()
                torch.cuda.synchronize()
                comm.barrier(False)

                graph.replay()
                torch.cuda.synchronize()
                comm.barrier(False)

                local_tensor = win.map_remote_tensor(rank)
                received = local_tensor[rank * count : (rank + 1) * count]
                expected = (
                    torch.ones(count, dtype=torch.float32, device=self.device)
                    * src_rank
                )
                torch.testing.assert_close(
                    received,
                    expected,
                    rtol=1e-5,
                    atol=1e-5,
                    msg=(
                        f"Replay {replay}: rank {rank} expected data from "
                        f"rank {src_rank}, got {received[:8]}..."
                    ),
                )

            win.tensor_deregister()
            del win
            graph.reset()
            del pool
            torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
