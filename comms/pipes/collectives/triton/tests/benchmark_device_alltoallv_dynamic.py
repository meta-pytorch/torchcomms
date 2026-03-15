# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-unsafe
"""
Benchmark tests comparing Triton device_alltoallv_dynamic against NCCL alltoallv.

IMPORTANT: Understanding the Comparison
========================================

This benchmark compares two fundamentally different approaches:

1. NCCL alltoallv (CPU-initiated):
   - Counts/sizes are Python lists (CPU/host memory)
   - CPU calls NCCL API with these values
   - In a real dynamic workload, this REQUIRES a D2H copy if counts
     were computed on GPU
   - This is standard NCCL - there is NO "NCCL alltoallv_dynamic"

2. Triton device_alltoallv_dynamic (GPU-initiated):
   - Counts/sizes are GPU tensors (device memory)
   - Kernel reads counts directly from GPU memory via tl.load()
   - NO D2H copy needed - counts stay on GPU
   - The "dynamic" refers to GPU-resident counts (NCCL doesn't have this)

The key insight: NCCL does NOT have a GPU-resident counts API.
Therefore, comparing against CPU-initiated NCCL alltoallv IS the correct
baseline, as it represents what users would do today.

The performance benefit of device_alltoallv_dynamic:
- Eliminating CPU synchronization point
- Eliminating D2H copy of counts (in real workflows)
- Enabling fused compute+communicate patterns

Run with:
    buck2 run @fbcode//mode/opt \
        -c fbcode.enable_gpu_sections=true \
        -c fbcode.platform010_cuda_version=12.8 \
        -c fbcode.nvcc_arch=h100a \
        -c hpc_comms.use_ncclx=stable \
        fbcode//comms/torchcomms/triton/fb/tests:benchmark_device_alltoallv_dynamic
"""

import argparse
import gc
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

import torch
from torch.utils._triton import has_triton

if TYPE_CHECKING:
    from torchcomms import TorchComm


TRITON_AVAILABLE = has_triton()
RUN_DEVICE_API_TEST = os.environ.get("RUN_DEVICE_API_TEST", "false").lower() == "true"


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    name: str
    msg_size_bytes: int
    num_ranks: int
    latency_us: float  # average latency
    bandwidth_gbps: float  # bandwidth computed from average latency
    iterations: int
    min_latency_us: float = 0.0
    max_latency_us: float = 0.0
    avg_latency_us: float = 0.0
    p99_latency_us: float = 0.0
    per_iter_latencies_us: List[float] = field(default_factory=list)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    warmup_iters: int = 10
    bench_iters: int = 100
    msg_sizes: Optional[List[int]] = None

    def __post_init__(self) -> None:
        if self.msg_sizes is None:
            # Default message sizes (bytes per peer, equal for all peers).
            # Matches the Pipes AllToAllvBenchmark sizing convention.
            # Total buffer per rank = msg_size * num_ranks; must fit in pool.
            self.msg_sizes = [
                1024,  # 1KB
                2 * 1024,  # 2KB
                4 * 1024,  # 4KB
                8 * 1024,  # 8KB
                16 * 1024,  # 16KB
                32 * 1024,  # 32KB
                64 * 1024,  # 64KB
                128 * 1024,  # 128KB
                256 * 1024,  # 256KB
                512 * 1024,  # 512KB
                1024 * 1024,  # 1MB
                2 * 1024 * 1024,  # 2MB
                4 * 1024 * 1024,  # 4MB
                8 * 1024 * 1024,  # 8MB
                16 * 1024 * 1024,  # 16MB
            ]


class AlltoallvDynamicBenchmark:
    """
    Benchmark suite for device_alltoallv_dynamic vs NCCL comparison.

    This benchmark measures the performance difference between:
    - NCCL alltoallv: CPU-initiated, counts on host (Python lists)
    - Triton device_alltoallv_dynamic: GPU-initiated, counts on device (GPU tensors)

    Note: NCCL does not support GPU-resident counts. The comparison is fair
    because it measures what users would do TODAY without device_alltoallv_dynamic.
    """

    def __init__(self, comm: "TorchComm", max_msg_size: int = 16 * 1024 * 1024) -> None:
        import torchcomms
        from comms.pipes.collectives.triton import alloc_comms_buffer

        self.comm = comm
        self.rank = comm.get_rank()
        self.num_ranks = comm.get_size()
        self.device = comm.get_device()
        self.allocator = torchcomms.get_mem_allocator(comm.get_backend())
        # Pool capacity must hold the largest possible buffer:
        # total_buf = max_msg_size * num_ranks (uniform benchmark).
        # Derived from --max-size so the user never hits SKIPPED messages.
        self.pool_capacity = max_msg_size * self.num_ranks

        # Pre-allocate fixed-size buffers from the NCCL allocator via
        # alloc_comms_buffer, which handles transport-compatible allocation.
        alloc_elems = self.pool_capacity // 4
        self.recv_buf, self.recv_pool = alloc_comms_buffer(
            alloc_elems, torch.float32, self.device, comm.get_backend()
        )
        self.send_buf, self.send_pool = alloc_comms_buffer(
            alloc_elems, torch.float32, self.device, comm.get_backend()
        )

        # Register the NCCL window once, reuse across all Triton benchmarks.
        self.comm.barrier(False)
        self.window = self.comm.new_window()
        self.window.tensor_register(self.recv_buf)

        # get_device_window must be called before register_local_buffer to enable GIN
        self.dev_win_ptr = self.window.get_device_window(signal_count=self.num_ranks)
        self.src_info = self.window.register_local_buffer(self.send_buf)
        self.comm.barrier(False)

        # Storage for raw kernel benchmark graph (uses class-level pools).
        # Must be cleaned up before pool cleanup to avoid MemPool conflicts.
        self._current_raw_kernel_graph = None

    def _deregister_src(self) -> None:
        """Deregister send buffer so it can be modified."""
        if self.src_info is not None:
            self.window.deregister_local_buffer(*self.src_info)
            self.src_info = None

    def _register_src(self) -> None:
        """Re-register send buffer after modification."""
        if self.src_info is None:
            self.window.get_device_window(signal_count=self.num_ranks)
            self.src_info = self.window.register_local_buffer(self.send_buf)

    def cleanup(self) -> None:
        """Deregister window and release pools (mirrors e2e tearDownClass)."""
        # Clean up ALL graphs FIRST, then ops, then pools.
        # Graphs hold references to memory allocated from MemPools.
        # If MemPools are destroyed while graphs still reference them,
        # we get `captures_underway.empty() INTERNAL ASSERT FAILED`.

        # 1. Clean up raw kernel benchmark graph (uses class-level pools)
        if self._current_raw_kernel_graph is not None:
            del self._current_raw_kernel_graph
            self._current_raw_kernel_graph = None
            torch.cuda.synchronize()

        # 2. Clean up class-level window and pools
        self.comm.barrier(False)
        if self.src_info is not None:
            self.window.deregister_local_buffer(*self.src_info)
            self.src_info = None
        if self.window is not None:
            self.window.tensor_deregister()
            self.window = None

        # 4. Clean up class-level pools LAST (after all graphs are destroyed)
        self.recv_buf = None
        self.send_buf = None
        self.recv_pool = None  # pyre-ignore[8]: Intentional cleanup
        self.send_pool = None  # pyre-ignore[8]: Intentional cleanup
        self.allocator = None
        gc.collect()
        torch.cuda.synchronize()

    def benchmark_nccl_alltoallv(
        self,
        msg_size: int,
        config: BenchmarkConfig,
    ) -> BenchmarkResult:
        """
        Benchmark standard NCCL alltoallv (CPU-initiated).

        This is standard NCCL alltoallv - NOT "alltoallv_dynamic". NCCL does not
        have a GPU-resident counts API.

        IMPORTANT: This uses CPU-resident counts (Python lists) because that's
        how NCCL's alltoallv API works.

        In a real dynamic workload where counts are computed on GPU, using NCCL
        would ALSO require a D2H copy to get counts to CPU - which is NOT
        included in this benchmark timing (making NCCL appear faster than it
        would be in practice).

        Equal sizes per peer: msg_size bytes per peer (matches Pipes benchmark).
        Reuses the shared pre-allocated buffers (same pattern as e2e tests).
        """
        # Equal sizes per peer (matching Pipes AllToAllvBenchmark)
        dtype = torch.float32
        element_size = dtype.itemsize  # 4 bytes for float32
        elems_per_peer = msg_size // element_size
        # NOTE: These are Python lists (CPU memory) - this is NCCL's API requirement
        # Sizes are in number of elements (not bytes) to match all_to_all_v_single API
        send_sizes = [elems_per_peer] * self.num_ranks
        recv_sizes = [elems_per_peer] * self.num_ranks
        total_send = elems_per_peer * self.num_ranks
        total_recv = elems_per_peer * self.num_ranks

        # Deregister before buffer modifications, re-register after
        self._deregister_src()
        self.send_buf.zero_()
        self.recv_buf.zero_()
        self.send_buf[:total_send].normal_()
        self._register_src()

        # Warmup
        for _ in range(config.warmup_iters):
            self.comm.all_to_all_v_single(
                self.recv_buf, self.send_buf, recv_sizes, send_sizes, async_op=False
            )

        # Benchmark — single event pair wrapping all iterations (matches Pipes
        # AllToAllvBenchmark methodology: avoids per-iteration event overhead
        # that inflates small-message latencies by ~1-2us per event pair).
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        for _ in range(config.bench_iters):
            self.comm.all_to_all_v_single(
                self.recv_buf, self.send_buf, recv_sizes, send_sizes, async_op=False
            )
        end_ev.record()
        torch.cuda.synchronize()

        total_ms = start_ev.elapsed_time(end_ev)
        avg_us = total_ms * 1e3 / config.bench_iters
        total_bytes = (total_send + total_recv) * element_size
        bandwidth_gbps = total_bytes / (avg_us * 1e-6) / 1e9

        return BenchmarkResult(
            name="nccl_alltoallv",
            msg_size_bytes=msg_size,
            num_ranks=self.num_ranks,
            latency_us=avg_us,
            bandwidth_gbps=bandwidth_gbps,
            iterations=config.bench_iters,
            avg_latency_us=avg_us,
        )

    def benchmark_nccl_alltoallv_graph(
        self,
        msg_size: int,
        config: BenchmarkConfig,
    ) -> BenchmarkResult:
        """
        Benchmark NCCL alltoallv using CUDA graph capture + replay.

        Captures ALL iterations inside a single CUDA graph, then replays
        once.  This eliminates all per-iteration CPU overhead (mutexes,
        event creation, heap allocations, cudaGraphLaunch) and isolates
        the pure GPU-side NCCL transfer time.
        """
        dtype = torch.float32
        element_size = dtype.itemsize
        elems_per_peer = msg_size // element_size
        send_sizes = [elems_per_peer] * self.num_ranks
        recv_sizes = [elems_per_peer] * self.num_ranks
        total_send = elems_per_peer * self.num_ranks
        total_recv = elems_per_peer * self.num_ranks

        self._deregister_src()
        self.send_buf.zero_()
        self.recv_buf.zero_()
        self.send_buf[:total_send].normal_()
        self._register_src()

        # Warmup (eager, to initialize NCCL internals)
        for _ in range(config.warmup_iters):
            self.comm.all_to_all_v_single(
                self.recv_buf, self.send_buf, recv_sizes, send_sizes, async_op=False
            )
        torch.cuda.synchronize()

        # Capture bench_iters iterations in a single CUDA graph
        graph_stream = torch.cuda.Stream()
        with torch.cuda.stream(graph_stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(config.bench_iters):
                    self.comm.all_to_all_v_single(
                        self.recv_buf,
                        self.send_buf,
                        recv_sizes,
                        send_sizes,
                        async_op=False,
                    )

        # Warmup graph replay.
        # NOTE: Signal counters accumulate across replays (ADD-based).
        # After the first replay, subsequent replays see pre-satisfied
        # waits because the baked-in iteration thresholds are already
        # exceeded.  This matches NCCL graph replay behavior.
        with torch.cuda.stream(graph_stream):
            for _ in range(config.warmup_iters):
                graph.replay()
        torch.cuda.synchronize()

        # Benchmark: single graph replay = bench_iters iterations
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(graph_stream):
            start_ev.record()
            graph.replay()
            end_ev.record()
        torch.cuda.synchronize()

        total_ms = start_ev.elapsed_time(end_ev)
        avg_us = total_ms * 1e3 / config.bench_iters
        total_bytes = (total_send + total_recv) * element_size
        bandwidth_gbps = total_bytes / (avg_us * 1e-6) / 1e9

        del graph

        return BenchmarkResult(
            name="nccl_graph",
            msg_size_bytes=msg_size,
            num_ranks=self.num_ranks,
            latency_us=avg_us,
            bandwidth_gbps=bandwidth_gbps,
            iterations=config.bench_iters,
            avg_latency_us=avg_us,
        )

    def benchmark_triton_alltoallv_dynamic(
        self,
        msg_size: int,
        config: BenchmarkConfig,
        num_warps: int = 16,
        blocks_per_peer: int = 1,
        chunk_size: int = 256 * 1024,
    ) -> BenchmarkResult:
        """
        Benchmark Triton device_alltoallv_dynamic (GPU-initiated).

        This uses GPU-resident counts (torch tensors on CUDA). The kernel reads
        counts directly from GPU memory via tl.load() - no CPU involvement.

        Uses the pre-allocated shared buffers and window (same pattern as e2e tests).
        """
        from comms.pipes.collectives.triton import (
            compute_offsets_from_sizes,
            device_alltoallv_dynamic,
            exchange_offsets,
        )

        # Equal sizes per peer in BYTES (matches Pipes AllToAllvBenchmark)
        send_sizes_list = [msg_size] * self.num_ranks
        recv_sizes_list = [msg_size] * self.num_ranks
        total_send = msg_size * self.num_ranks
        total_recv = msg_size * self.num_ranks

        # Deregister before buffer modifications, re-register after
        self._deregister_src()
        self.send_buf.zero_()
        self.recv_buf.zero_()
        self.send_buf[: total_send // 4].normal_()

        # NOTE: These are GPU tensors — create before GIN is active
        send_sizes = torch.tensor(
            send_sizes_list, dtype=torch.int64, device=self.device
        )
        send_offsets = torch.zeros_like(send_sizes)
        compute_offsets_from_sizes(send_sizes, send_offsets)

        recv_sizes = torch.tensor(
            recv_sizes_list, dtype=torch.int64, device=self.device
        )
        recv_offsets = torch.zeros_like(recv_sizes)
        compute_offsets_from_sizes(recv_sizes, recv_offsets)

        # Pre-compute dst_offsets once (offset exchange) before registration
        dst_offsets = exchange_offsets(recv_offsets, self.comm)

        self._register_src()
        self.comm.barrier(False)

        # Warmup
        for _ in range(config.warmup_iters):
            device_alltoallv_dynamic(
                self.send_buf,
                self.recv_buf,
                send_sizes,
                send_offsets,
                recv_sizes,
                recv_offsets,
                dst_offsets,
                self.dev_win_ptr,
                self.src_info,
                self.rank,
                self.num_ranks,
                num_warps=num_warps,
                blocks_per_peer=blocks_per_peer,
                chunk_size=chunk_size,
                sync_buffer=False,  # Benchmark: measure raw kernel throughput
            )
        torch.cuda.synchronize()

        self.comm.barrier(False)

        # Benchmark — single event pair wrapping all iterations (matches Pipes
        # AllToAllvBenchmark methodology).
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        start_ev.record()
        for _ in range(config.bench_iters):
            device_alltoallv_dynamic(
                self.send_buf,
                self.recv_buf,
                send_sizes,
                send_offsets,
                recv_sizes,
                recv_offsets,
                dst_offsets,
                self.dev_win_ptr,
                self.src_info,
                self.rank,
                self.num_ranks,
                num_warps=num_warps,
                blocks_per_peer=blocks_per_peer,
                chunk_size=chunk_size,
                sync_buffer=False,  # Benchmark: measure raw kernel throughput
            )
        end_ev.record()
        torch.cuda.synchronize()

        total_ms = start_ev.elapsed_time(end_ev)
        avg_us = total_ms * 1e3 / config.bench_iters
        total_bytes = total_send + total_recv
        bandwidth_gbps = total_bytes / (avg_us * 1e-6) / 1e9

        return BenchmarkResult(
            name=f"triton_dynamic_w{num_warps} (GPU counts)",
            msg_size_bytes=msg_size,
            num_ranks=self.num_ranks,
            latency_us=avg_us,
            bandwidth_gbps=bandwidth_gbps,
            iterations=config.bench_iters,
            avg_latency_us=avg_us,
        )

    def benchmark_triton_alltoallv_dynamic_graph(
        self,
        msg_size: int,
        config: BenchmarkConfig,
        num_warps: int = 16,
        blocks_per_peer: int = 1,
        chunk_size: int = 256 * 1024,
    ) -> BenchmarkResult:
        """
        Benchmark Triton device_alltoallv_dynamic using CUDA graph capture + replay.

        Captures ALL iterations inside a single CUDA graph, then replays
        once.  This eliminates per-iteration Python/CPU overhead and isolates
        the pure GPU-side Triton kernel time.
        """
        from comms.pipes.collectives.triton import (
            compute_offsets_from_sizes,
            device_alltoallv_dynamic,
            exchange_offsets,
        )

        send_sizes_list = [msg_size] * self.num_ranks
        recv_sizes_list = [msg_size] * self.num_ranks
        total_send = msg_size * self.num_ranks
        total_recv = msg_size * self.num_ranks

        self._deregister_src()
        self.send_buf.zero_()
        self.recv_buf.zero_()
        self.send_buf[: total_send // 4].normal_()

        send_sizes = torch.tensor(
            send_sizes_list, dtype=torch.int64, device=self.device
        )
        send_offsets = torch.zeros_like(send_sizes)
        compute_offsets_from_sizes(send_sizes, send_offsets)

        recv_sizes = torch.tensor(
            recv_sizes_list, dtype=torch.int64, device=self.device
        )
        recv_offsets = torch.zeros_like(recv_sizes)
        compute_offsets_from_sizes(recv_sizes, recv_offsets)

        dst_offsets = exchange_offsets(recv_offsets, self.comm)

        self._register_src()
        self.comm.barrier(False)

        # Warmup (eager, to compile Triton kernel and initialize state)
        for _ in range(config.warmup_iters):
            device_alltoallv_dynamic(
                self.send_buf,
                self.recv_buf,
                send_sizes,
                send_offsets,
                recv_sizes,
                recv_offsets,
                dst_offsets,
                self.dev_win_ptr,
                self.src_info,
                self.rank,
                self.num_ranks,
                num_warps=num_warps,
                blocks_per_peer=blocks_per_peer,
                chunk_size=chunk_size,
                sync_buffer=False,  # Benchmark: measure raw kernel throughput
            )
        torch.cuda.synchronize()

        self.comm.barrier(False)

        # Capture bench_iters iterations in a single CUDA graph.
        # The iteration tensor is incremented via add_(1) inside the captured
        # graph, ensuring each iteration sees a monotonically increasing value.
        # This enables correct wait_signal_from behavior during graph replay.
        #
        # Store graph at class level to control cleanup order. The graph
        # uses class-level pools (recv_pool, send_pool) and must be deleted
        # BEFORE those pools to avoid MemPool cleanup conflicts.
        if self._current_raw_kernel_graph is not None:
            del self._current_raw_kernel_graph
            self._current_raw_kernel_graph = None
            torch.cuda.synchronize()

        graph_stream = torch.cuda.Stream()
        with torch.cuda.stream(graph_stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(config.bench_iters):
                    device_alltoallv_dynamic(
                        self.send_buf,
                        self.recv_buf,
                        send_sizes,
                        send_offsets,
                        recv_sizes,
                        recv_offsets,
                        dst_offsets,
                        self.dev_win_ptr,
                        self.src_info,
                        self.rank,
                        self.num_ranks,
                        num_warps=num_warps,
                        blocks_per_peer=blocks_per_peer,
                        chunk_size=chunk_size,
                        sync_buffer=False,  # Benchmark: measure raw kernel throughput
                    )

        # Store graph at class level to prevent cleanup during function return
        self._current_raw_kernel_graph = graph

        # Warmup graph replay
        with torch.cuda.stream(graph_stream):
            for _ in range(config.warmup_iters):
                graph.replay()
        torch.cuda.synchronize()

        # Benchmark: single graph replay = bench_iters iterations
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(graph_stream):
            start_ev.record()
            graph.replay()
            end_ev.record()
        torch.cuda.synchronize()

        total_ms = start_ev.elapsed_time(end_ev)
        avg_us = total_ms * 1e3 / config.bench_iters
        total_bytes = total_send + total_recv
        bandwidth_gbps = total_bytes / (avg_us * 1e-6) / 1e9

        del graph

        return BenchmarkResult(
            name=f"triton_graph_w{num_warps}",
            msg_size_bytes=msg_size,
            num_ranks=self.num_ranks,
            latency_us=avg_us,
            bandwidth_gbps=bandwidth_gbps,
            iterations=config.bench_iters,
            avg_latency_us=avg_us,
        )

    def benchmark_alltoallv_op(
        self,
        msg_size: int,
        config: BenchmarkConfig,
    ) -> BenchmarkResult:
        """
        Benchmark AlltoallvOp (packed output is now the default).

        Uses CUDA graph capture/replay for accurate latency measurement.
        Returns contiguous packed output for MSL API compatibility.

        NOTE: Only uniform distribution is supported. For uniform distribution
        (benchmark case), packed output is a zero-copy view of the internal
        data since slots are back-to-back.
        """
        from comms.pipes.collectives.triton import AlltoallvOp

        dtype = torch.float32
        element_size = dtype.itemsize
        D = 128  # Hidden dimension for benchmark
        elems_per_peer = msg_size // element_size // D
        if elems_per_peer == 0:
            elems_per_peer = 1
        # max_input_tokens is the total tokens we send (to all peers combined).
        # max_recv_tokens_per_peer is the max tokens received from any single peer.
        # For uniform distribution, each peer sends elems_per_peer to each other peer.
        max_input_tokens = elems_per_peer * self.num_ranks
        max_recv_tokens_per_peer = (
            elems_per_peer  # uniform: each peer sends same amount
        )

        # Create AlltoallvOp with sync_buffer=False for raw kernel overhead measurement.
        # NOTE: This is NOT production-safe. Use sync_buffer=True (the default) for
        # production code involving CUDA graphs or buffer reuse across iterations.
        op = AlltoallvOp(
            self.comm,
            max_input_tokens=max_input_tokens,
            D=D,
            dtype=dtype,
            device=self.device,
            max_recv_tokens_per_peer=max_recv_tokens_per_peer,
            sync_buffer=False,
        )

        input_split_sizes = torch.full(
            (self.num_ranks,), elems_per_peer, dtype=torch.int64, device=self.device
        )
        output_split_sizes = input_split_sizes.clone()

        # Pre-compute packed_output_tokens for CUDA graph capture.
        # For uniform split sizes, this is simply elems_per_peer * num_ranks.
        packed_output_tokens = elems_per_peer * self.num_ranks

        # Zero-copy path: fill send buffer BEFORE setup (GIN blocks regular fills)
        send_buf = op.get_send_buffer(max_input_tokens)
        send_buf.normal_()  # Fill with random data once

        # Setup (enables GIN, registers buffers)
        op.setup()

        # Warmup (eager, to compile Triton kernels).
        # First call auto-runs prep kernel; subsequent calls auto-skip.
        for _ in range(config.warmup_iters):
            _ = op.alltoallv_from_buffer(
                output_split_sizes,
                input_split_sizes,
                num_input_tokens=max_input_tokens,
                packed_output_tokens=packed_output_tokens,
            )
        torch.cuda.synchronize()
        self.comm.barrier(False)

        # Capture bench_iters iterations in the graph (as users would do).
        # Pass pool=op.get_graph_pool_id() to ensure allocations use the
        # same transport-compatible pool as AlltoallvOp's buffers.
        # Pass packed_output_tokens to avoid .item() call during capture.
        # Prep kernel is auto-skipped since it ran during warmup.
        graph_stream = torch.cuda.Stream()
        with torch.cuda.stream(graph_stream):
            graph = torch.cuda.CUDAGraph()
            # pyre-fixme[6]: Pyre doesn't recognize pool ID as valid _POOL_HANDLE
            with torch.cuda.graph(graph, pool=op.get_graph_pool_id()):  # type: ignore[arg-type]
                for _ in range(config.bench_iters):
                    _ = op.alltoallv_from_buffer(
                        output_split_sizes,
                        input_split_sizes,
                        num_input_tokens=max_input_tokens,
                        packed_output_tokens=packed_output_tokens,
                    )

        # Warmup graph replay (like the raw kernel benchmark does)
        with torch.cuda.stream(graph_stream):
            for _ in range(config.warmup_iters):
                graph.replay()
        torch.cuda.synchronize()

        # Benchmark: single graph replay = bench_iters iterations
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(graph_stream):
            start_ev.record()
            graph.replay()
            end_ev.record()
        torch.cuda.synchronize()

        total_ms = start_ev.elapsed_time(end_ev)
        avg_us = total_ms * 1e3 / config.bench_iters
        total_bytes = msg_size * self.num_ranks * 2  # send + recv
        bandwidth_gbps = total_bytes / (avg_us * 1e-6) / 1e9

        # Clean up graph and op after benchmark.
        del graph
        op.teardown()
        del op
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        return BenchmarkResult(
            name="op_packed_graph",
            msg_size_bytes=msg_size,
            num_ranks=self.num_ranks,
            latency_us=avg_us,
            bandwidth_gbps=bandwidth_gbps,
            iterations=config.bench_iters,
            avg_latency_us=avg_us,
        )

    def benchmark_alltoallv_op_sync_buffer(
        self,
        msg_size: int,
        config: BenchmarkConfig,
    ) -> BenchmarkResult:
        """
        Benchmark AlltoallvOp with sync_buffer=True (buffer-ready synchronization).

        Uses CUDA graph capture/replay for accurate latency measurement.
        This measures the overhead of buffer-ready synchronization, which is
        required for safe buffer reuse in fused compute+communication kernels.

        The sync_buffer mode adds ~1-3us per-peer latency overhead due to additional
        BUFFER_READY signal exchanges. This overhead is acceptable for fused
        kernels where buffer safety is required, but unnecessary for standalone
        alltoallv calls.

        Best for: Persistent/mega-kernels, fused MoE dispatch+alltoallv+combine.
        """
        from comms.pipes.collectives.triton import AlltoallvOp

        dtype = torch.float32
        element_size = dtype.itemsize
        D = 128  # Hidden dimension for benchmark
        elems_per_peer = msg_size // element_size // D
        if elems_per_peer == 0:
            elems_per_peer = 1
        # max_input_tokens is the total tokens we send (to all peers combined).
        # max_recv_tokens_per_peer is the max tokens received from any single peer.
        # For uniform distribution, each peer sends elems_per_peer to each other peer.
        max_input_tokens = elems_per_peer * self.num_ranks
        max_recv_tokens_per_peer = (
            elems_per_peer  # uniform: each peer sends same amount
        )

        # Create AlltoallvOp with sync_buffer=True (the default) for production-safe
        # buffer-ready synchronization across iterations.
        op = AlltoallvOp(
            self.comm,
            max_input_tokens=max_input_tokens,
            D=D,
            dtype=dtype,
            device=self.device,
            max_recv_tokens_per_peer=max_recv_tokens_per_peer,
            # sync_buffer=True is the default, explicit for clarity
        )

        input_split_sizes = torch.full(
            (self.num_ranks,), elems_per_peer, dtype=torch.int64, device=self.device
        )
        output_split_sizes = input_split_sizes.clone()

        # Pre-compute packed_output_tokens for CUDA graph capture.
        # For uniform split sizes, this is simply elems_per_peer * num_ranks.
        # This avoids .item() call during graph capture which would cause errors.
        packed_output_tokens = elems_per_peer * self.num_ranks

        # Zero-copy path: fill send buffer BEFORE setup (GIN blocks regular fills)
        send_buf = op.get_send_buffer(max_input_tokens)
        send_buf.normal_()  # Fill with random data once

        # Setup (enables GIN, registers buffers)
        op.setup()

        # Warmup (eager, to compile Triton kernels).
        # First call auto-runs prep kernel; subsequent calls auto-skip.
        for _ in range(config.warmup_iters):
            _ = op.alltoallv_from_buffer(
                output_split_sizes,
                input_split_sizes,
                num_input_tokens=max_input_tokens,
                packed_output_tokens=packed_output_tokens,
            )
        torch.cuda.synchronize()
        self.comm.barrier(False)

        # Capture bench_iters iterations in the graph (as users would do).
        # Pass pool=op.get_graph_pool_id() to ensure allocations use the
        # same transport-compatible pool as AlltoallvOp's buffers.
        # Pass packed_output_tokens to avoid .item() call during capture.
        # Prep kernel is auto-skipped since it ran during warmup.
        graph_stream = torch.cuda.Stream()
        with torch.cuda.stream(graph_stream):
            graph = torch.cuda.CUDAGraph()
            # pyre-fixme[6]: Pyre doesn't recognize pool ID as valid _POOL_HANDLE
            with torch.cuda.graph(graph, pool=op.get_graph_pool_id()):  # type: ignore[arg-type]
                for _ in range(config.bench_iters):
                    _ = op.alltoallv_from_buffer(
                        output_split_sizes,
                        input_split_sizes,
                        num_input_tokens=max_input_tokens,
                        packed_output_tokens=packed_output_tokens,
                    )

        # Warmup graph replay (like the raw kernel benchmark does)
        with torch.cuda.stream(graph_stream):
            for _ in range(config.warmup_iters):
                graph.replay()
        torch.cuda.synchronize()

        # Benchmark: single graph replay = bench_iters iterations
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(graph_stream):
            start_ev.record()
            graph.replay()
            end_ev.record()
        torch.cuda.synchronize()

        total_ms = start_ev.elapsed_time(end_ev)
        avg_us = total_ms * 1e3 / config.bench_iters
        total_bytes = msg_size * self.num_ranks * 2  # send + recv
        bandwidth_gbps = total_bytes / (avg_us * 1e-6) / 1e9

        # Clean up graph and op after benchmark.
        del graph
        op.teardown()
        del op
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        return BenchmarkResult(
            name="op_fused_graph",
            msg_size_bytes=msg_size,
            num_ranks=self.num_ranks,
            latency_us=avg_us,
            bandwidth_gbps=bandwidth_gbps,
            iterations=config.bench_iters,
            avg_latency_us=avg_us,
        )

    def run_comparison(
        self,
        config: BenchmarkConfig,
        test_filter: str = "",
        include_op_benchmarks: bool = False,
    ) -> List[BenchmarkResult]:
        """
        Run full comparison benchmark suite.

        Compares:
        - NCCL alltoallv (CPU-initiated, counts in Python lists)
        - Triton device_alltoallv_dynamic (GPU-initiated, counts in GPU tensors)
        - (Optional) AlltoallvOp benchmarks for API overhead comparison

        All tests use equal sizes per peer: msg_size bytes per peer
        (matches Pipes AllToAllvBenchmark convention).

        Args:
            test_filter: If non-empty, only run benchmarks whose name contains
                this substring.  Supported names: "nccl", "triton".
                Matches are case-insensitive.  An empty string runs everything.
            include_op_benchmarks: If True, also run AlltoallvOp
                benchmarks for API overhead comparison.
        """
        # Determine which benchmarks to run based on filter
        run_nccl = not test_filter or "nccl" in test_filter.lower()
        run_triton = (
            not test_filter
            or "triton" in test_filter.lower()
            or "alltoallv" in test_filter.lower()
        )
        if not test_filter:
            run_nccl = run_triton = True

        all_results = []
        table_width = 0

        if self.rank == 0:
            # Build header first to determine table width
            if include_op_benchmarks:
                header = f"{'Size':>12} | {'NCCLgr(us)':>12} | {'TRTgr(us)':>12} | {'TRTop(us)':>12} | {'TRTsync(us)':>12} | {'Sync Ovhd':>10} | {'Speedup':>10}"
            else:
                header = f"{'Size':>12} | {'NCCLgr(us)':>12} | {'TRTgr(us)':>12} | {'Speedup':>10}"
            table_width = len(header)

            print(f"\n{'=' * table_width}")
            print(f"AlltoAllv Dynamic Benchmark: {self.num_ranks} ranks")
            print("=" * table_width)
            print("Comparison:")
            print(
                "  - NCCL alltoallv:       CPU-initiated, counts in Python lists (host memory)"
            )
            print(
                "  - Triton dynamic:       GPU-initiated, counts in GPU tensors (device memory)"
            )
            if include_op_benchmarks:
                print("  - TRTop:                AlltoallvOp benchmark (API overhead)")
                print(
                    "  - TRTsync:              AlltoallvOp with sync_buffer=True (buffer-ready sync)"
                )
            print()
            print(
                "NOTE: NCCL does NOT support GPU-resident counts. In real dynamic workloads"
            )
            print(
                "      where counts are computed on GPU, NCCL would also require D2H copy."
            )
            print("=" * table_width)
            print("Equal sizes per peer: msg_size bytes per peer")
            print(f"Warmup: {config.warmup_iters}, Iterations: {config.bench_iters}")
            print("=" * table_width)

            print(header)
            print("-" * table_width)

        assert config.msg_sizes is not None
        for msg_size in config.msg_sizes:  # pyre-ignore[16]
            # Validate that total buffer fits within pool capacity.
            # Equal sizes: total = msg_size * num_ranks.
            total_buf = msg_size * self.num_ranks
            if total_buf > self.pool_capacity:
                if self.rank == 0:
                    size_str = self._format_size(msg_size)
                    print(f"{size_str:>12} | {'SKIPPED - exceeds pool capacity':>30}")
                continue

            nccl_result = None
            triton_result = None
            op_result = None
            sync_buffer_result = None

            # Run NCCL via CUDA graph (baseline)
            if run_nccl:
                nccl_result = self.benchmark_nccl_alltoallv_graph(msg_size, config)
                all_results.append(nccl_result)

            # Run Triton raw kernel via CUDA graph (auto-tuned parameters)
            if run_triton:
                from comms.pipes.collectives.triton.device_alltoallv_dynamic import (
                    auto_tune_alltoallv_params,
                )

                auto_params = auto_tune_alltoallv_params(msg_size)
                triton_result = self.benchmark_triton_alltoallv_dynamic_graph(
                    msg_size,
                    config,
                    num_warps=auto_params["num_warps"],
                    blocks_per_peer=auto_params["blocks_per_peer"],
                    chunk_size=auto_params["chunk_size"],
                )
                all_results.append(triton_result)

            # Optionally run AlltoallvOp benchmarks for API overhead comparison
            if run_triton and include_op_benchmarks:
                op_result = self.benchmark_alltoallv_op(msg_size, config)
                all_results.append(op_result)
                # Ensure full cleanup before running next benchmark to avoid
                # graph pool conflicts between different AlltoallvOp instances.
                # The barrier + sleep ensures all ranks have completed cleanup
                # before creating new CUDA resources.
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                self.comm.barrier(False)
                time.sleep(0.1)  # Allow background threads to settle
                sync_buffer_result = self.benchmark_alltoallv_op_sync_buffer(
                    msg_size, config
                )
                all_results.append(sync_buffer_result)

            if self.rank == 0:
                size_str = self._format_size(msg_size)
                nccl_str = (
                    f"{nccl_result.latency_us:>12.2f}"
                    if nccl_result
                    else f"{'skip':>12}"
                )
                triton_str = (
                    f"{triton_result.latency_us:>12.2f}"
                    if triton_result
                    else f"{'skip':>12}"
                )

                if include_op_benchmarks:
                    op_str = (
                        f"{op_result.latency_us:>12.2f}"
                        if op_result
                        else f"{'skip':>12}"
                    )
                    sync_str = (
                        f"{sync_buffer_result.latency_us:>12.2f}"
                        if sync_buffer_result
                        else f"{'skip':>12}"
                    )
                    # Calculate sync_buffer overhead compared to op_result (baseline)
                    if sync_buffer_result and op_result:
                        overhead_us = (
                            sync_buffer_result.latency_us - op_result.latency_us
                        )
                        overhead_per_peer = overhead_us / max(1, self.num_ranks - 1)
                        overhead_str = (
                            f"+{overhead_us:>4.1f}({overhead_per_peer:.1f}/p)"
                        )
                    else:
                        overhead_str = f"{'-':>10}"
                    row = f"{size_str:>12} | {nccl_str} | {triton_str} | {op_str} | {sync_str} | {overhead_str:>10}"
                else:
                    row = f"{size_str:>12} | {nccl_str} | {triton_str}"

                # Compute speedup vs NCCL (using raw kernel for fair comparison)
                if nccl_result and triton_result:
                    speedup = nccl_result.latency_us / triton_result.latency_us
                    row += f" | {speedup:>10.2f}x"
                else:
                    row += f" | {'-':>10}"

                print(row)

        if self.rank == 0:
            print("=" * table_width)

        return all_results

    def run_num_warps_sweep(
        self,
        config: BenchmarkConfig,
        warp_counts: Optional[List[int]] = None,
        block_counts: Optional[List[int]] = None,
        custom_configs: Optional[List[dict]] = None,
    ) -> List[BenchmarkResult]:
        """
        Sweep num_warps and blocks_per_peer for eager and graph kernels.

        Produces a table showing how latency varies with warp count and
        blocks_per_peer across all message sizes for all modes.

        Column groups (left to right):
        - NCCL(us), NCCLgr(us): baselines
        - w{N}(us): eager, 1 block/peer, sweeping warps
        - Gw{N}(us): CUDA graph, 1 block/peer, sweeping warps
        - B{B}(us): eager, {B} blocks/peer, fixed 16 warps
        - GB{B}(us): CUDA graph, {B} blocks/peer, fixed 16 warps

        Args:
            config: Benchmark configuration with message sizes, warmup, etc.
            warp_counts: List of num_warps values to test (default: [4, 8, 16, 32]).
            block_counts: List of blocks_per_peer values to test
                          (default: [2, 4, 8, 16, 32]).  Only applied to
                          eager and graph variants at 16 warps/block.
        """
        if warp_counts is None:
            warp_counts = [4, 8, 16, 32]
        if block_counts is None:
            block_counts = [2, 4, 8, 16, 32]

        # Build human-readable labels for custom configs
        custom_labels = []
        if custom_configs:
            for cc in custom_configs:
                b, w, c = cc["blocks_per_peer"], cc["num_warps"], cc["chunk_size"]
                if b == 1:
                    custom_labels.append(f"Gw{w}")
                else:
                    custom_labels.append(f"B{b}w{w}c{c // 1024}")

        all_results = []
        table_width = 0

        if self.rank == 0:
            # Build header — use compact widths to fit ~240-char terminal
            header = f"{'Size':>10} | {'NCCLgr':>9}"
            for w in warp_counts:
                header += f" | {'Gw' + str(w):>8}"
            for b in block_counts:
                header += f" | {'GB' + str(b):>8}"
            header += f" | {'GB16c64':>9}"
            for lbl in custom_labels:
                header += f" | {lbl:>10}"
            header += f" | {'Auto':>8}"
            header += f" | {'AutoCfg':>10}"
            header += f" | {'Best':>14}"
            table_width = len(header)

            print(f"\n{'=' * table_width}")
            print(f"Num Warps + Blocks/Peer Sweep: {self.num_ranks} ranks")
            print("=" * table_width)
            print("Compares eager and CUDA-graph kernels.")
            print("Higher warps = more threads for NVLink memcpy in put_block.")
            print(
                "Higher blocks/peer = more parallel put_block calls "
                "per peer (splits data into chunks)."
            )
            print(
                f"Warp counts: {warp_counts}  "
                f"(threads per block: {[w * 32 for w in warp_counts]})"
            )
            print(
                f"Blocks/peer: {block_counts}  (at 16 warps/block = 512 threads/block)"
            )
            print("All values in microseconds (us).")
            print("=" * table_width)

            print(header)
            print("-" * table_width)

        assert config.msg_sizes is not None
        for msg_size in config.msg_sizes:  # pyre-ignore[16]
            total_buf = msg_size * self.num_ranks
            if total_buf > self.pool_capacity:
                if self.rank == 0:
                    size_str = self._format_size(msg_size)
                    print(f"{size_str:>10} | {'SKIPPED - exceeds pool capacity':>30}")
                continue

            # NCCL via CUDA graph
            nccl_graph_result = self.benchmark_nccl_alltoallv_graph(msg_size, config)
            all_results.append(nccl_graph_result)

            # Graph warp sweep (1 block/peer)
            g_results = {}
            for w in warp_counts:
                r = self.benchmark_triton_alltoallv_dynamic_graph(
                    msg_size, config, num_warps=w
                )
                g_results[w] = r
                all_results.append(r)

            # Graph blocks_per_peer sweep (16 warps)
            gbp_results = {}
            for b in block_counts:
                r = self.benchmark_triton_alltoallv_dynamic_graph(
                    msg_size, config, num_warps=16, blocks_per_peer=b
                )
                gbp_results[b] = r
                all_results.append(r)

            # Multi-SM config: 16 blocks/peer, 8 warps/block, 64KB chunks
            msm_graph_result = self.benchmark_triton_alltoallv_dynamic_graph(
                msg_size,
                config,
                num_warps=8,
                blocks_per_peer=16,
                chunk_size=64 * 1024,
            )
            all_results.append(msm_graph_result)

            # Custom configs
            cc_results = {}
            if custom_configs:
                for i, cc in enumerate(custom_configs):
                    r = self.benchmark_triton_alltoallv_dynamic_graph(
                        msg_size,
                        config,
                        num_warps=cc["num_warps"],
                        blocks_per_peer=cc["blocks_per_peer"],
                        chunk_size=cc["chunk_size"],
                    )
                    cc_results[i] = r
                    all_results.append(r)

            # Auto-tuned

            # Auto-tuned: select optimal parameters based on msg_size
            from comms.pipes.collectives.triton.device_alltoallv_dynamic import (
                auto_tune_alltoallv_params,
            )

            auto_params = auto_tune_alltoallv_params(msg_size)
            auto_result = self.benchmark_triton_alltoallv_dynamic_graph(
                msg_size,
                config,
                num_warps=auto_params["num_warps"],
                blocks_per_peer=auto_params["blocks_per_peer"],
                chunk_size=auto_params["chunk_size"],
            )
            all_results.append(auto_result)

            # Derive a human-readable label for the auto-tuned config
            # so the report shows which specific config Auto maps to.
            ap_bpp = auto_params["blocks_per_peer"]
            ap_nw = auto_params["num_warps"]
            ap_cs = auto_params["chunk_size"]
            if ap_bpp == 1:
                auto_config_label = f"Gw{ap_nw}"
            else:
                chunk_kb = ap_cs // 1024
                auto_config_label = f"GB{ap_bpp}c{chunk_kb}"

            if self.rank == 0:
                size_str = self._format_size(msg_size)
                row = f"{size_str:>10} | {nccl_graph_result.latency_us:>9.2f}"
                for w in warp_counts:
                    row += f" | {g_results[w].latency_us:>8.2f}"
                for b in block_counts:
                    row += f" | {gbp_results[b].latency_us:>8.2f}"
                row += f" | {msm_graph_result.latency_us:>9.2f}"
                for i in range(len(custom_labels)):
                    row += f" | {cc_results[i].latency_us:>10.2f}"
                row += f" | {auto_result.latency_us:>8.2f}"
                row += f" | {auto_config_label:>10}"

                # Find overall best across all Triton graph variants.
                # When Auto ties the best latency, prefer Auto — it is the
                # recommended default and should be reported as best when it
                # matches any specific configuration.
                all_triton: List[tuple] = []
                for w, r in g_results.items():
                    all_triton.append((f"Gw{w}", r))
                for b, r in gbp_results.items():
                    all_triton.append((f"GB{b}", r))
                all_triton.append(("GB16c64", msm_graph_result))
                for i, lbl in enumerate(custom_labels):
                    all_triton.append((lbl, cc_results[i]))
                all_triton.append(("Auto", auto_result))
                best_name, best_result = min(all_triton, key=lambda x: x[1].latency_us)
                if auto_result.latency_us <= best_result.latency_us:
                    best_name = "Auto"
                    best_result = auto_result
                speedup = nccl_graph_result.latency_us / best_result.latency_us
                speedup_str = f"{best_name} ({speedup:.2f}x)"
                row += f" | {speedup_str:>14}"
                print(row)

            gc.collect()
            torch.cuda.synchronize()

        if self.rank == 0:
            print("=" * table_width)
            print()

        return all_results

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """Format byte size as human-readable string."""
        if size_bytes >= 1024**3:
            return f"{size_bytes / 1024**3:.1f}GB"
        elif size_bytes >= 1024**2:
            return f"{size_bytes / 1024**2:.1f}MB"
        elif size_bytes >= 1024:
            return f"{size_bytes / 1024:.1f}KB"
        return f"{size_bytes}B"


def main() -> int:
    """Run benchmark suite."""
    parser = argparse.ArgumentParser(
        description="AlltoAllv Dynamic Benchmark: Triton vs NCCL"
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    parser.add_argument(
        "--min-size",
        type=int,
        default=1024,
        help="Minimum per-peer message size (bytes)",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=16 * 1024 * 1024,
        help="Maximum per-peer message size (bytes, default: 16MB)",
    )
    parser.add_argument(
        "--sweep-warps",
        action="store_true",
        help="Run num_warps sweep (4, 8, 16, 32) across all message sizes",
    )
    parser.add_argument(
        "--warp-counts",
        type=str,
        default="4,8,16,32",
        help="Comma-separated list of num_warps values for --sweep-warps (default: 4,8,16,32)",
    )
    parser.add_argument(
        "--block-counts",
        type=str,
        default="2,4,8,16,32",
        help="Comma-separated list of blocks_per_peer values for --sweep-warps (default: 2,4,8,16,32)",
    )
    parser.add_argument(
        "--custom-configs",
        type=str,
        default="",
        help=(
            "Comma-separated list of custom configs to benchmark in the sweep. "
            "Each config is BxWxC where B=blocks_per_peer, W=num_warps, "
            "C=chunk_size in KB. Examples: '16x8x64' (GB16 w8 64KB chunks), "
            "'8x16x32,16x4x8' (two custom configs). "
            "These appear as additional columns in the sweep table."
        ),
    )
    parser.add_argument(
        "--test-filter",
        type=str,
        default="",
        help=(
            "Only run benchmarks whose name contains this substring "
            "(case-insensitive). Supported: 'nccl', 'triton'. "
            "Can also be set via TEST_FILTER env var."
        ),
    )
    parser.add_argument(
        "--include-op-benchmarks",
        action="store_true",
        help=(
            "Include AlltoallvOp benchmarks in addition to "
            "the raw kernel benchmark. Useful for comparing API overhead. Sync mode "
            "measures buffer-ready synchronization overhead (adds ~1-3us per-peer)."
        ),
    )
    args = parser.parse_args()

    # TEST_FILTER: CLI flag takes precedence over env var (matching e2e test pattern)
    test_filter = args.test_filter or os.environ.get("TEST_FILTER", "")

    if not RUN_DEVICE_API_TEST:
        print("Set RUN_DEVICE_API_TEST=true to run benchmarks")
        return 1

    if not TRITON_AVAILABLE:
        print("Triton not available")
        return 1

    # Parse warp counts and block counts
    warp_counts = [int(w) for w in args.warp_counts.split(",")]
    block_counts = [int(b) for b in args.block_counts.split(",")]

    # Parse custom configs: each is BxWxC (blocks x warps x chunk_kb)
    custom_configs = []
    if args.custom_configs:
        for spec in args.custom_configs.split(","):
            parts = spec.strip().split("x")
            if len(parts) != 3:
                print(f"Invalid custom config '{spec}': expected BxWxC (e.g. 16x8x64)")
                return 1
            b, w, c_kb = int(parts[0]), int(parts[1]), int(parts[2])
            if w <= 0 or (w & (w - 1)) != 0:
                print(
                    f"Skipping custom config '{spec}': "
                    f"num_warps={w} must be a power of 2"
                )
                continue
            if b <= 0:
                print(
                    f"Skipping custom config '{spec}': blocks_per_peer={b} must be > 0"
                )
                continue
            custom_configs.append(
                {"blocks_per_peer": b, "num_warps": w, "chunk_size": c_kb * 1024}
            )

    # Initialize TorchComm
    from torchcomms.tests.integration.py.TorchCommTestHelpers import (
        TorchCommTestWrapper,
    )

    wrapper = TorchCommTestWrapper()
    comm = wrapper.get_torchcomm()

    # Build message sizes
    msg_sizes = []
    size = args.min_size
    while size <= args.max_size:
        msg_sizes.append(size)
        size *= 2

    config = BenchmarkConfig(
        warmup_iters=args.warmup,
        bench_iters=args.iters,
        msg_sizes=msg_sizes,
    )

    benchmark = AlltoallvDynamicBenchmark(comm, max_msg_size=args.max_size)

    try:
        if args.sweep_warps:
            # Run num_warps sweep across all message sizes
            benchmark.run_num_warps_sweep(
                config,
                warp_counts=warp_counts,
                block_counts=block_counts,
                custom_configs=custom_configs if custom_configs else None,
            )
        else:
            # Run full comparison
            benchmark.run_comparison(
                config,
                test_filter=test_filter,
                include_op_benchmarks=args.include_op_benchmarks,
            )
    finally:
        benchmark.cleanup()
        # Release comm and wrapper (mirrors e2e tearDownClass ordering).
        # time.sleep(2) allows folly background threads to settle before
        # process exit, avoiding std::system_error on thread creation.
        comm = None
        wrapper = None
        gc.collect()
        torch.cuda.synchronize()
        time.sleep(2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
