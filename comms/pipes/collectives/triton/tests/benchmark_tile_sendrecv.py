# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-unsafe
"""
Benchmark: Triton tile send/recv vs TorchComm batched send/recv.

Compares bidirectional point-to-point transfer between adjacent NVLink peers.
Uses the same block counts as the C++ Pipes P2pNvlSendRecvBenchmark:
  16 blocks for sizes <= 256MB, 32 blocks for 512MB+.

Staging config: NCCL_CTRAN_P2P_NVL_SHARED_DEVBUF_SIZE=64MB, pipelineDepth=2
  => dataBufferSize=32MB per slot, perBlockSlotSize=2MB (16 blocks).

Run with:
    buck2 run @fbcode//mode/opt \
        -c comms.hosts=localhost \
        -c fbcode.enable_gpu_sections=true \
        -c fbcode.platform010_cuda_version=12.8 \
        -c fbcode.nvcc_arch=h100a \
        -c hpc_comms.use_ncclx=stable \
        fbcode//comms/pipes/collectives/triton/tests:benchmark_tile_sendrecv
"""

import argparse
import gc
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, List, Optional

import torch
from torch.utils._triton import has_triton

TRITON_AVAILABLE = has_triton()
RUN_DEVICE_API_TEST = os.environ.get("RUN_DEVICE_API_TEST", "false").lower() == "true"

if TRITON_AVAILABLE:
    import triton
    import triton.language as tl
    from torchcomms.triton.fb import requires_torchcomms, transport


@dataclass
class BenchmarkConfig:
    warmup_iters: int = 10
    bench_iters: int = 100
    msg_sizes: Optional[List[int]] = None

    def __post_init__(self) -> None:
        if self.msg_sizes is None:
            self.msg_sizes = [
                8 * 1024,
                16 * 1024,
                64 * 1024,
                128 * 1024,
                256 * 1024,
                512 * 1024,
                1024 * 1024,
                2 * 1024 * 1024,
                4 * 1024 * 1024,
                8 * 1024 * 1024,
                16 * 1024 * 1024,
                32 * 1024 * 1024,
                64 * 1024 * 1024,
                128 * 1024 * 1024,
                256 * 1024 * 1024,
                512 * 1024 * 1024,
                1024 * 1024 * 1024,
            ]


if TRITON_AVAILABLE:

    @requires_torchcomms
    @triton.jit
    def tile_sendrecv_kernel(
        transport_ptr,
        send_ptr,
        recv_ptr,
        nbytes,
        max_signal_bytes,
        num_send_blocks: tl.constexpr,
        peer: tl.constexpr,
    ):
        # Contiguous layout (matches C++ partition(2)):
        # pid 0..N-1 = senders, pid N..2N-1 = receivers.
        # Keeps senders on one set of SMs, receivers on another,
        # avoiding L1/LD-ST contention from co-located send+recv.
        pid = tl.program_id(axis=0)
        is_sender = pid < num_send_blocks
        block_id = pid if is_sender else pid - num_send_blocks

        # Partition data evenly across blocks (like TiledBuffer in C++).
        tile_bytes = nbytes // num_send_blocks
        tile_bytes = tile_bytes - (tile_bytes % 16)
        tile_off = block_id * tile_bytes
        if block_id == num_send_blocks - 1:
            tile_bytes = nbytes - tile_off

        if is_sender:
            transport.send(
                transport_ptr,
                peer,
                send_ptr + tile_off,
                tile_bytes,
                num_send_blocks,
                max_signal_bytes,
            )
        else:
            transport.recv(
                transport_ptr,
                peer,
                recv_ptr + tile_off,
                tile_bytes,
                num_send_blocks,
                max_signal_bytes,
            )


def _format_size(nbytes: int) -> str:
    if nbytes >= 1024 * 1024 * 1024:
        return f"{nbytes / (1024 * 1024 * 1024):.0f}GB"
    if nbytes >= 1024 * 1024:
        return f"{nbytes / (1024 * 1024):.0f}MB"
    if nbytes >= 1024:
        return f"{nbytes / 1024:.0f}KB"
    return f"{nbytes}B"


def _pipes_block_count(msg_size: int) -> int:
    if msg_size >= 512 * 1024 * 1024:
        return 32
    return 16


class TileSendRecvBenchmark:
    def __init__(self, comm: Any) -> None:
        import torchcomms

        self.comm = comm
        self.rank = comm.get_rank()
        self.num_ranks = comm.get_size()
        self.device = comm.get_device()
        self.allocator = torchcomms.get_mem_allocator(comm.get_backend())

        self.peer = self.rank ^ 1
        assert self.peer < self.num_ranks, "Need at least 2 ranks"

    def _get_transport_handle(self) -> int:
        try:
            handle = self.comm.get_device_transport()
        except RuntimeError as e:
            raise RuntimeError(f"Pipes transport not available: {e}") from e
        return handle

    def benchmark_tile_sendrecv_graph(
        self,
        msg_size: int,
        config: BenchmarkConfig,
        num_send_blocks: int = 16,
        max_signal_bytes: int = 0,
    ) -> float:
        transport_handle = self._get_transport_handle()

        send_buf = torch.zeros(msg_size // 4, dtype=torch.float32, device=self.device)
        recv_buf = torch.zeros(msg_size // 4, dtype=torch.float32, device=self.device)

        total_blocks = 2 * num_send_blocks

        for _ in range(config.warmup_iters):
            self.comm.barrier(False)
            tile_sendrecv_kernel[(total_blocks,)](
                transport_handle,
                send_buf.data_ptr(),
                recv_buf.data_ptr(),
                msg_size,
                max_signal_bytes,
                num_send_blocks=num_send_blocks,
                peer=self.peer,
                num_warps=16,
                num_ctas=4,
            )
            torch.cuda.synchronize()

        self.comm.barrier(False)

        graph_stream = torch.cuda.Stream()
        with torch.cuda.stream(graph_stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(config.bench_iters):
                    tile_sendrecv_kernel[(total_blocks,)](
                        transport_handle,
                        send_buf.data_ptr(),
                        recv_buf.data_ptr(),
                        msg_size,
                        max_signal_bytes,
                        num_send_blocks=num_send_blocks,
                        peer=self.peer,
                        num_warps=16,
                        num_ctas=4,
                    )

        with torch.cuda.stream(graph_stream):
            for _ in range(config.warmup_iters):
                graph.replay()
        torch.cuda.synchronize()

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(graph_stream):
            start_ev.record()
            graph.replay()
            end_ev.record()
        torch.cuda.synchronize()

        total_ms = start_ev.elapsed_time(end_ev)
        avg_us = total_ms * 1e3 / config.bench_iters

        del graph, send_buf, recv_buf
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return avg_us

    def benchmark_batched_sendrecv_graph(
        self,
        msg_size: int,
        config: BenchmarkConfig,
    ) -> float:
        send_buf = torch.zeros(msg_size // 4, dtype=torch.float32, device=self.device)
        recv_buf = torch.zeros(msg_size // 4, dtype=torch.float32, device=self.device)

        for _ in range(config.warmup_iters):
            batch = self.comm.batch_op_create()
            batch.send(send_buf, self.peer)
            batch.recv(recv_buf, self.peer)
            batch.issue(False)
        torch.cuda.synchronize()
        self.comm.barrier(False)

        graph_stream = torch.cuda.Stream()
        with torch.cuda.stream(graph_stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(config.bench_iters):
                    batch = self.comm.batch_op_create()
                    batch.send(send_buf, self.peer)
                    batch.recv(recv_buf, self.peer)
                    batch.issue(False)

        with torch.cuda.stream(graph_stream):
            for _ in range(config.warmup_iters):
                graph.replay()
        torch.cuda.synchronize()

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(graph_stream):
            start_ev.record()
            graph.replay()
            end_ev.record()
        torch.cuda.synchronize()

        total_ms = start_ev.elapsed_time(end_ev)
        avg_us = total_ms * 1e3 / config.bench_iters

        del graph, send_buf, recv_buf
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return avg_us

    def run_comparison(self, config: BenchmarkConfig) -> None:
        header = f"{'Size':>10} | {'Blk':>4} | {'NCCL(us)':>10}"
        for nb in [1, 2, 4, 8, 16, 32]:
            header += f" | {'B' + str(nb) + '(us)':>10}"
        header += f" | {'Best':>14} | {'Speedup':>8}"
        table_width = len(header)

        if self.rank == 0:
            print(f"\n{'=' * table_width}")
            print(f"Tile SendRecv Benchmark: rank {self.rank} <-> rank {self.peer}")
            print("=" * table_width)
            print("  - NCCL:     TorchComm batch_op send+recv (CUDA graph)")
            print("  - sXXX:     Triton send+recv, max_signal_bytes=XXX")
            print("  - Blk:      send blocks (matches C++ Pipes benchmark)")
            print("  - Staging:  32MB/slot, pipelineDepth=2")
            print(f"Warmup: {config.warmup_iters}, Iterations: {config.bench_iters}")
            print("=" * table_width)
            print(header)
            print("-" * table_width)

        msg_sizes = config.msg_sizes
        assert msg_sizes is not None
        for msg_size in msg_sizes:
            nblocks = _pipes_block_count(msg_size)

            nccl_us = self.benchmark_batched_sendrecv_graph(msg_size, config)
            self.comm.barrier(False)

            blk_results = {}
            for nb in [1, 2, 4, 8, 16, 32]:
                blk_results[nb] = self.benchmark_tile_sendrecv_graph(
                    msg_size,
                    config,
                    num_send_blocks=nb,
                )
                self.comm.barrier(False)

            best_us = min(blk_results.values())
            best_nb = min(blk_results, key=lambda k: blk_results[k])
            speedup = nccl_us / best_us if best_us > 0 else 0

            if self.rank == 0:
                size_str = _format_size(msg_size)
                row = f"{size_str:>10} | {nblocks:>4} | {nccl_us:>10.2f}"
                for nb in [1, 2, 4, 8, 16, 32]:
                    row += f" | {blk_results[nb]:>10.2f}"
                row += f" | {'B' + str(best_nb) + '=' + f'{best_us:.1f}':>14}"
                row += f" | {speedup:>7.2f}x"
                print(row)

        if self.rank == 0:
            print("=" * table_width)


def main() -> int:
    parser = argparse.ArgumentParser(description="Tile SendRecv Benchmark")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--min-size", type=int, default=8 * 1024)
    parser.add_argument("--max-size", type=int, default=1024 * 1024 * 1024)
    args = parser.parse_args()

    if not RUN_DEVICE_API_TEST:
        print("Set RUN_DEVICE_API_TEST=true to run benchmarks")
        return 1

    if not TRITON_AVAILABLE:
        print("Triton not available")
        return 1

    from torchcomms.tests.integration.py.TorchCommTestHelpers import (
        TorchCommTestWrapper,
    )

    wrapper = TorchCommTestWrapper()
    comm = wrapper.get_torchcomm()

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

    benchmark = TileSendRecvBenchmark(comm)

    try:
        benchmark.run_comparison(config)
    finally:
        comm = None
        wrapper = None
        gc.collect()
        torch.cuda.synchronize()
        time.sleep(2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
