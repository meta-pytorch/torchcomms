# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-strict
"""
IB auto-tune sweep for the Triton alltoallv kernel.

Sweeps IB-specific kernel parameters (blocks_per_peer, num_warps, chunk_size)
across message sizes to find optimal configs for inter-node IB transport.
Compares against NCCL alltoallv baseline and outputs a lookup table for
_tune_for_ib() in device_alltoallv_dynamic.py.

Run on multi-node H100 via the triton test launcher.
"""

from __future__ import annotations

import gc
import os
import sys
import unittest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch
from torch.utils._triton import has_triton

if TYPE_CHECKING:
    from torchcomms import TorchComm

TRITON_AVAILABLE: bool = has_triton()
RUN_DEVICE_API_TEST: bool = (
    os.environ.get("RUN_DEVICE_API_TEST", "false").lower() == "true"
)


@dataclass
class SweepResult:
    msg_size: int
    blocks_per_peer: int
    num_warps: int
    chunk_size: int
    latency_us: float
    bandwidth_gbps: float


_DEFAULT_MSG_SIZES: List[int] = [
    1024,
    4 * 1024,
    16 * 1024,
    64 * 1024,
    256 * 1024,
    512 * 1024,
    1024 * 1024,
    2 * 1024 * 1024,
    4 * 1024 * 1024,
    8 * 1024 * 1024,
    16 * 1024 * 1024,
]
_DEFAULT_BPP: List[int] = [1, 2, 4, 8]
_DEFAULT_WARPS: List[int] = [4, 8, 16, 32]
_DEFAULT_CHUNKS: List[int] = [64 * 1024, 128 * 1024, 256 * 1024, 512 * 1024]


@dataclass
class SweepConfig:
    warmup_iters: int = 5
    bench_iters: int = 50
    msg_sizes: List[int] = None  # pyre-ignore[8]
    blocks_per_peer_values: List[int] = None  # pyre-ignore[8]
    num_warps_values: List[int] = None  # pyre-ignore[8]
    chunk_size_values: List[int] = None  # pyre-ignore[8]

    def __post_init__(self) -> None:
        if self.msg_sizes is None:
            self.msg_sizes = list(_DEFAULT_MSG_SIZES)
        if self.blocks_per_peer_values is None:
            self.blocks_per_peer_values = list(_DEFAULT_BPP)
        if self.num_warps_values is None:
            self.num_warps_values = list(_DEFAULT_WARPS)
        if self.chunk_size_values is None:
            self.chunk_size_values = list(_DEFAULT_CHUNKS)


class IBAutoTuneSweep:
    """Sweep IB kernel parameters and compare against NCCL baseline."""

    comm: TorchComm
    rank: int
    num_ranks: int
    device: torch.device
    pool_capacity: int
    recv_buf: Optional[torch.Tensor]
    recv_pool: Optional[torch.cuda.MemPool]
    send_buf: Optional[torch.Tensor]
    send_pool: Optional[torch.cuda.MemPool]
    window: Any
    dev_win_ptr: int
    src_info: Optional[int]

    def __init__(self, comm: TorchComm, max_msg_size: int = 16 * 1024 * 1024) -> None:
        from comms.pipes.collectives.triton import alloc_comms_buffer

        self.comm = comm
        self.rank = comm.get_rank()
        self.num_ranks = comm.get_size()
        self.device = comm.get_device()
        self.pool_capacity = max_msg_size * self.num_ranks

        alloc_elems = self.pool_capacity // 4
        self.recv_buf, self.recv_pool = alloc_comms_buffer(
            alloc_elems, torch.float32, self.device, comm.get_backend()
        )
        self.send_buf, self.send_pool = alloc_comms_buffer(
            alloc_elems, torch.float32, self.device, comm.get_backend()
        )

        self.comm.barrier(False)
        self.window = self.comm.new_window()
        self.window.tensor_register(self.recv_buf)
        self.dev_win_ptr = self.window.get_device_window(
            signal_count=self.num_ranks * 2
        )
        self.src_info = self.window.register_local_buffer(self.send_buf)
        self.comm.barrier(False)

    def cleanup(self) -> None:
        self.comm.barrier(False)
        if self.src_info is not None:
            self.window.deregister_local_buffer(self.src_info)
            self.src_info = None
        if self.window is not None:
            self.window.tensor_deregister()
            self.window = None
        self.recv_buf = None
        self.send_buf = None
        self.recv_pool = None
        self.send_pool = None
        gc.collect()
        torch.cuda.synchronize()

    def _deregister_src(self) -> None:
        if self.src_info is not None:
            self.window.deregister_local_buffer(self.src_info)
            self.src_info = None

    def _register_src(self) -> None:
        if self.src_info is None:
            self.window.get_device_window(signal_count=self.num_ranks * 2)
            self.src_info = self.window.register_local_buffer(self.send_buf)

    def benchmark_nccl(self, msg_size: int, config: SweepConfig) -> float:
        """Return NCCL graph latency in us."""
        dtype = torch.float32
        elems_per_peer = msg_size // dtype.itemsize
        send_sizes = [elems_per_peer] * self.num_ranks
        recv_sizes = [elems_per_peer] * self.num_ranks
        total = elems_per_peer * self.num_ranks

        self._deregister_src()
        assert self.send_buf is not None
        assert self.recv_buf is not None
        send_buf = self.send_buf
        recv_buf = self.recv_buf
        send_buf.zero_()
        recv_buf.zero_()
        send_buf[:total].normal_()
        self._register_src()

        for _ in range(config.warmup_iters):
            self.comm.all_to_all_v_single(
                self.recv_buf, self.send_buf, recv_sizes, send_sizes, async_op=False
            )
        torch.cuda.synchronize()

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
        del graph
        return total_ms * 1e3 / config.bench_iters

    def benchmark_triton(
        self,
        msg_size: int,
        config: SweepConfig,
        blocks_per_peer: int,
        num_warps: int,
        chunk_size: int,
    ) -> float:
        """Return Triton graph latency in us."""
        from comms.pipes.collectives.triton import (
            compute_offsets_from_sizes,
            device_alltoallv_dynamic,
            exchange_offsets,
        )
        from comms.pipes.collectives.triton.device_alltoallv_dynamic import (
            _reset_iteration_counter,
        )

        send_sizes = torch.full(
            (self.num_ranks,), msg_size, dtype=torch.int64, device=self.device
        )
        send_offsets = torch.zeros_like(send_sizes)
        compute_offsets_from_sizes(send_sizes, send_offsets)
        recv_sizes = send_sizes.clone()
        recv_offsets = torch.zeros_like(recv_sizes)
        compute_offsets_from_sizes(recv_sizes, recv_offsets)
        dst_offsets = exchange_offsets(recv_offsets, self.comm)

        total = msg_size * self.num_ranks
        self._deregister_src()
        assert self.send_buf is not None
        assert self.recv_buf is not None
        send_buf = self.send_buf
        recv_buf = self.recv_buf
        send_buf.zero_()
        recv_buf.zero_()
        send_buf[: total // 4].normal_()
        self._register_src()
        self.comm.barrier(False)

        _reset_iteration_counter(self.num_ranks, self.device)

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
                sync_buffer=False,
            )
        torch.cuda.synchronize()
        self.comm.barrier(False)

        _reset_iteration_counter(self.num_ranks, self.device)

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
                        sync_buffer=False,
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
        del graph
        _reset_iteration_counter(self.num_ranks, self.device)
        return total_ms * 1e3 / config.bench_iters

    def _sweep_msg_size(
        self,
        msg_size: int,
        config: SweepConfig,
    ) -> tuple[float, tuple[int, int, int] | None]:
        """Sweep all param combos for one message size. Returns (nccl_us, best_config)."""
        nccl_us = self.benchmark_nccl(msg_size, config)
        best_lat = float("inf")
        best_config: tuple[int, int, int] | None = None

        for bpp in config.blocks_per_peer_values:
            for nw in config.num_warps_values:
                for cs in config.chunk_size_values:
                    try:
                        lat = self.benchmark_triton(msg_size, config, bpp, nw, cs)
                    except Exception as e:
                        if self.rank == 0:
                            print(
                                f"  WARN: bpp={bpp} nw={nw} cs={cs} failed: {e}",
                                file=sys.stderr,
                            )
                        lat = float("inf")

                    if lat < best_lat:
                        best_lat = lat
                        best_config = (bpp, nw, cs)

        return nccl_us, best_config

    def run_sweep(self, config: SweepConfig) -> None:
        """Run the full sweep and print results."""
        best_per_size: Dict[int, Dict[str, object]] = {}

        for msg_size in config.msg_sizes:
            if msg_size * self.num_ranks > self.pool_capacity:
                if self.rank == 0:
                    print(f"SKIPPED {msg_size} (exceeds pool)", file=sys.stderr)
                continue

            nccl_us, best_config = self._sweep_msg_size(msg_size, config)

            if best_config is not None:
                bpp, nw, cs = best_config
                best_lat = self.benchmark_triton(msg_size, config, bpp, nw, cs)
                speedup = nccl_us / best_lat if best_lat > 0 else 0
                best_per_size[msg_size] = {
                    "blocks_per_peer": bpp,
                    "num_warps": nw,
                    "chunk_size": cs,
                    "latency_us": best_lat,
                    "nccl_us": nccl_us,
                    "speedup": speedup,
                }

                if self.rank == 0:
                    size_str = _format_size(msg_size)
                    cs_str = _format_size(cs)
                    print(
                        f"  {size_str:>8s} | bpp={bpp:2d} nw={nw:2d} cs={cs_str:>5s} | "
                        f"NCCL={nccl_us:8.1f}us  Triton={best_lat:8.1f}us  "
                        f"Speedup={speedup:.2f}x",
                        file=sys.stderr,
                    )

        if self.rank == 0:
            print("\n=== IB Lookup Table (for _tune_for_ib) ===", file=sys.stderr)
            print(
                "def _tune_for_ib(max_msg_size_bytes: int) -> dict:",
                file=sys.stderr,
            )
            sorted_sizes = sorted(best_per_size.keys())
            for i, sz in enumerate(sorted_sizes):
                cfg = best_per_size[sz]
                size_str = _format_size(sz)
                kw = "if" if i == 0 else "elif"
                print(
                    f"    {kw} max_msg_size_bytes <= {sz}:  "
                    f"# {size_str} ({cfg['speedup']:.2f}x vs NCCL)",
                    file=sys.stderr,
                )
                print(
                    f'        return {{"blocks_per_peer": {cfg["blocks_per_peer"]}, '
                    f'"num_warps": {cfg["num_warps"]}, '
                    f'"chunk_size": {cfg["chunk_size"]}}}',
                    file=sys.stderr,
                )
            if sorted_sizes:
                last = best_per_size[sorted_sizes[-1]]
                print(
                    f"    else:\n"
                    f'        return {{"blocks_per_peer": {last["blocks_per_peer"]}, '
                    f'"num_warps": {last["num_warps"]}, '
                    f'"chunk_size": {last["chunk_size"]}}}',
                    file=sys.stderr,
                )


def _format_size(b: int) -> str:
    if b >= 1024 * 1024:
        return f"{b // (1024 * 1024)}MB"
    if b >= 1024:
        return f"{b // 1024}KB"
    return f"{b}B"


def _is_ready() -> bool:
    return TRITON_AVAILABLE and RUN_DEVICE_API_TEST


class TestAutoTuneSweepIB(unittest.TestCase):
    """Auto-tune sweep test — runs as a distributed unittest."""

    wrapper: object = None
    torchcomm: object = None

    @classmethod
    def setUpClass(cls) -> None:
        if not _is_ready():
            raise unittest.SkipTest("Sweep test environment not ready")

        from torchcomms.tests.integration.py.TorchCommTestHelpers import (
            TorchCommTestWrapper,
        )

        cls.wrapper = TorchCommTestWrapper()
        cls.torchcomm = cls.wrapper.get_torchcomm()

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.torchcomm is not None:
            cls.torchcomm.barrier(False)  # pyre-ignore[16]
            cls.torchcomm.finalize()  # pyre-ignore[16]
        cls.torchcomm = None
        cls.wrapper = None

    def test_ib_autotune_sweep(self) -> None:
        self.assertIsNotNone(self.torchcomm)
        config = SweepConfig()
        sweep = IBAutoTuneSweep(self.torchcomm)  # pyre-ignore[6]
        try:
            sweep.run_sweep(config)
        finally:
            sweep.cleanup()
