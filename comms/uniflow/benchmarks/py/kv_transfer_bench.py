# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""
KV-cache transfer benchmark for UniFlow.

Measures get() bandwidth and latency for KV-cache-shaped block transfers
between two GPUs through the UniFlow Python API.  This exercises the same
code path that the vLLM UniflowConnector uses in production: segment
registration, connection establishment, and batched TransferRequest-based
data movement.

The --num-layers flag controls how many TransferRequests are issued per
batch (block_count × num_layers), simulating the connector's pattern of
transferring blocks across all transformer layers in one conn.get() call.

Usage (via buck):
    buck test //comms/uniflow/benchmarks/py:kv_transfer_bench

Usage (standalone, requires 2 GPUs):
    python kv_transfer_bench.py [options]
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import threading
import time
import unittest
from typing import Any

from pydantic import BaseModel, computed_field

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCH_CUDA_EXPANDABLE_SEGMENTS_IPC", "1")

import torch  # noqa: E402

_SKIP_NO_MULTI_GPU: bool = (
    not torch.cuda.is_available() or torch.cuda.device_count() < 2
)

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class KVCacheConfig(BaseModel, frozen=True):
    """Immutable KV cache geometry."""

    num_blocks: int = 256
    block_size: int = 16
    num_kv_heads: int = 8
    head_size: int = 128
    dtype: torch.dtype = torch.float16
    num_layers: int = 1
    coalesce: bool = False

    model_config = {"arbitrary_types_allowed": True}

    @computed_field  # type: ignore[prop-decorator]
    @property
    def block_elems(self) -> int:
        return self.block_size * self.num_kv_heads * self.head_size * 2

    @computed_field  # type: ignore[prop-decorator]
    @property
    def block_bytes(self) -> int:
        return self.block_elems * self.dtype.itemsize

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_elems(self) -> int:
        return self.num_blocks * self.block_elems

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> KVCacheConfig:
        return cls(
            num_blocks=args.num_blocks,
            block_size=args.block_size,
            num_kv_heads=args.num_kv_heads,
            head_size=args.head_size,
            dtype=_DTYPE_MAP[args.dtype],
            num_layers=args.num_layers,
            coalesce=args.coalesce,
        )


class Stats(BaseModel):
    """Latency statistics (matches comms/uniflow/benchmarks/Stats.h)."""

    min: float = 0.0
    max: float = 0.0
    avg: float = 0.0
    p50: float = 0.0
    p99: float = 0.0

    @staticmethod
    def compute(samples: list[float]) -> Stats:
        if not samples:
            return Stats()
        s = sorted(samples)
        n = len(s)
        p99_idx = min(math.ceil(n * 0.99) - 1, n - 1)
        return Stats(min=s[0], max=s[-1], avg=sum(s) / n, p50=s[n // 2], p99=s[p99_idx])


class BenchResult(BaseModel):
    block_count: int = 0
    num_layers: int = 0
    total_requests: int = 0
    total_bytes: int = 0
    iterations: int = 0
    latency: Stats = Stats()
    bw_gbps: float = 0.0


# ---------------------------------------------------------------------------
# TransferSession — manages agent lifecycle as a context manager
# ---------------------------------------------------------------------------


class TransferSession:
    """Point-to-point transfer session between a prefill GPU and a decode GPU.

    Models the disaggregated serving data flow::

        Prefill (GPU 0)                    Decode (GPU 1)
        ┌──────────────┐                   ┌──────────────┐
        │ KV cache     │◄── conn.get() ────│ empty buffer │
        │ (source data)│   (decode pulls)  │ (destination)│
        └──────────────┘                   └──────────────┘

    A TransferRequest is a single DMA operation with two spans:
      - dst: where to write (decode's registered segment)
      - src: where to read  (prefill's registered segment, imported by decode)

    conn.get(requests=[...]) issues all DMA operations as one batch.
    The decode side drives all transfers (pull model).

    Use as a context manager::

        with TransferSession(config) as session:
            future = session.decode_conn.get(requests=[
                TransferRequest(session.decode_reg.span(...),
                                session.prefill_remote_seg.span(...))
            ])
    """

    def __init__(self, config: KVCacheConfig) -> None:
        self.config = config

        # Tensors: prefill holds source data, decode receives it
        self.prefill_tensor: torch.Tensor | None = None
        self.decode_tensor: torch.Tensor | None = None

        self._prefill_agent: Any = None
        self._decode_agent: Any = None
        self._prefill_conn: Any = None
        self._decode_conn: Any = None

        # Registered segments (local memory mapped for DMA)
        self._prefill_reg: Any = None
        self._decode_reg: Any = None

        # Prefill's segment as seen by decode (imported remote handle)
        self._prefill_remote_seg: Any = None

    @property
    def decode_reg(self) -> Any:
        """Decode's local registered segment (DMA destination)."""
        return self._decode_reg

    @property
    def prefill_remote_seg(self) -> Any:
        """Prefill's segment imported into decode's address space (DMA source)."""
        return self._prefill_remote_seg

    @property
    def decode_conn(self) -> Any:
        """Decode-side connection used to pull data from prefill."""
        return self._decode_conn

    def __enter__(self) -> TransferSession:
        self._setup()
        return self

    def __exit__(self, *exc: object) -> None:
        self._teardown()

    def _setup(self) -> None:
        from uniflow._core import UniflowAgent, UniflowAgentConfig

        cfg = self.config
        self.prefill_tensor = torch.arange(
            cfg.total_elems, dtype=cfg.dtype, device="cuda:0"
        )
        self.decode_tensor = torch.zeros(
            cfg.total_elems, dtype=cfg.dtype, device="cuda:1"
        )

        self._prefill_agent = UniflowAgent(
            UniflowAgentConfig(device_id=0, name="prefill", listen_address="*:0")
        )
        self._decode_agent = UniflowAgent(
            UniflowAgentConfig(device_id=1, name="decode", listen_address="*:0")
        )

        uid_result = self._prefill_agent.get_unique_id()
        assert uid_result.has_value(), f"get_unique_id: {uid_result.error()}"
        prefill_uid: str = uid_result.value()

        self._prefill_reg = self._register(self._prefill_agent, self.prefill_tensor, 0)
        self._decode_reg = self._register(self._decode_agent, self.decode_tensor, 1)

        prefill_export = self._export(self._prefill_reg)
        decode_export = self._export(self._decode_reg)

        self._establish_connection(prefill_uid, prefill_export, decode_export)

    def _register(self, agent: Any, tensor: torch.Tensor, device_id: int) -> Any:
        from uniflow._core import MemoryType, Segment

        seg = Segment(
            ptr=tensor.data_ptr(),
            length=tensor.nbytes,
            mem_type=MemoryType.VRAM,
            device_id=device_id,
        )
        result = agent.register_segment(seg)
        assert result.has_value(), f"register device {device_id}: {result.error()}"
        return result.value()

    def _export(self, reg: Any) -> bytes:
        result = reg.export_id()
        assert result.has_value(), f"export_id: {result.error()}"
        return result.value()

    def _establish_connection(
        self, prefill_uid: str, prefill_export: bytes, decode_export: bytes
    ) -> None:
        """Connect prefill ↔ decode and exchange segment handles."""
        errors: list[str] = []
        holder: dict[str, Any] = {}

        def prefill_fn() -> None:
            try:
                r = self._prefill_agent.accept()
                assert r.has_value(), f"accept: {r.error()}"
                conn = r.value()
                r = conn.send_ctrl_msg(prefill_export)
                assert r.has_value(), f"send export: {r.error()}"
                r = conn.recv_ctrl_msg()
                assert r.has_value(), f"recv export: {r.error()}"
                r = self._prefill_agent.import_segment(r.value())
                assert r.has_value(), f"import: {r.error()}"
                holder["prefill_conn"] = conn
            except Exception as e:
                errors.append(f"Prefill: {e}")

        def decode_fn() -> None:
            try:
                r = self._decode_agent.connect(prefill_uid)
                assert r.has_value(), f"connect: {r.error()}"
                conn = r.value()
                r = conn.send_ctrl_msg(decode_export)
                assert r.has_value(), f"send export: {r.error()}"
                r = conn.recv_ctrl_msg()
                assert r.has_value(), f"recv export: {r.error()}"
                r = self._decode_agent.import_segment(r.value())
                assert r.has_value(), f"import: {r.error()}"
                holder["decode_conn"] = conn
                holder["prefill_remote_seg"] = r.value()
            except Exception as e:
                errors.append(f"Decode: {e}")

        prefill_t = threading.Thread(target=prefill_fn)
        decode_t = threading.Thread(target=decode_fn)
        prefill_t.start()
        decode_t.start()
        prefill_t.join(timeout=30)
        decode_t.join(timeout=30)
        assert not prefill_t.is_alive(), "Prefill thread hung during setup"
        assert not decode_t.is_alive(), "Decode thread hung during setup"
        assert not errors, f"Connection errors: {errors}"

        self._prefill_conn = holder["prefill_conn"]
        self._decode_conn = holder["decode_conn"]
        self._prefill_remote_seg = holder["prefill_remote_seg"]

    def _teardown(self) -> None:
        if self._decode_conn is not None and self._prefill_conn is not None:
            self._decode_conn.send_ctrl_msg(b"done")
            self._prefill_conn.recv_ctrl_msg()
        if self._decode_conn is not None:
            self._decode_conn.shutdown()
        if self._prefill_conn is not None:
            self._prefill_conn.shutdown()


# ---------------------------------------------------------------------------
# KVTransferBenchmark — orchestrates warmup, correctness, and timed runs
# ---------------------------------------------------------------------------


class KVTransferBenchmark:
    """Runs KV-cache transfer benchmarks over a TransferSession."""

    def __init__(
        self,
        session: TransferSession,
        warmup_iters: int = 5,
        bench_iters: int = 20,
    ) -> None:
        self.session = session
        self.config = session.config
        self.warmup_iters = warmup_iters
        self.bench_iters = bench_iters

    def run(self) -> list[BenchResult]:
        self._warmup_regions()
        self._verify_correctness()
        block_counts = [c for c in [1, 4, 16, 64, 256] if c <= self.config.num_blocks]
        return [self._bench_block_count(bc) for bc in block_counts]

    def _warmup_regions(self) -> None:
        """Transfer the full buffer to fault in pages and warm transport path."""
        from uniflow._core import TransferRequest

        total_bytes = self.config.num_blocks * self.config.block_bytes
        dst = self.session.decode_reg.span(0, total_bytes)
        src = self.session.prefill_remote_seg.span(0, total_bytes)
        future = self.session.decode_conn.get(requests=[TransferRequest(dst, src)])
        assert future.wait_for(timeout_ms=30000), "Region warmup timed out"
        result = future.get()
        assert result.has_value(), f"Region warmup: {result.error()}"
        torch.cuda.synchronize(1)

    def _verify_correctness(self) -> None:
        """Pull one block from prefill → decode and verify data matches."""
        from uniflow._core import TransferRequest

        bb = self.config.block_bytes
        dst = self.session.decode_reg.span(0, bb)
        src = self.session.prefill_remote_seg.span(0, bb)
        future = self.session.decode_conn.get(requests=[TransferRequest(dst, src)])
        assert future.wait_for(timeout_ms=10000), "Correctness transfer timed out"
        result = future.get()
        assert result.has_value(), f"Correctness get: {result.error()}"

        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        assert self.session.prefill_tensor is not None
        assert self.session.decode_tensor is not None
        be = self.config.block_elems
        expected = self.session.prefill_tensor[:be].to("cuda:1")
        torch.testing.assert_close(
            self.session.decode_tensor[:be],
            expected,
            msg="Data mismatch on correctness check",
        )

    def _build_requests(self, block_count: int) -> list[Any]:
        """Build TransferRequests: decode pulls block_count blocks from prefill.

        Each TransferRequest is one DMA operation:
          dst = decode_reg.span(offset, block_bytes)   -- where to write
          src = prefill_remote_seg.span(offset, ...)   -- where to read
        """
        from uniflow._core import TransferRequest

        bb = self.config.block_bytes
        decode_reg = self.session.decode_reg
        prefill_seg = self.session.prefill_remote_seg

        if self.config.coalesce:
            coalesced_bytes = block_count * bb
            return [
                TransferRequest(
                    decode_reg.span(0, coalesced_bytes),
                    prefill_seg.span(0, coalesced_bytes),
                )
                for _ in range(self.config.num_layers)
            ]
        return [
            TransferRequest(
                decode_reg.span(i * bb, bb),
                prefill_seg.span(i * bb, bb),
            )
            for _ in range(self.config.num_layers)
            for i in range(block_count)
        ]

    def _bench_block_count(self, block_count: int) -> BenchResult:
        requests = self._build_requests(block_count)
        transfer_bytes = block_count * self.config.num_layers * self.config.block_bytes
        conn = self.session.decode_conn
        assert self.session.decode_tensor is not None

        self.session.decode_tensor.zero_()
        torch.cuda.synchronize(1)

        for _ in range(self.warmup_iters):
            f = conn.get(requests=requests)
            assert f.wait_for(timeout_ms=60000), "Warmup transfer timed out"
            r = f.get()
            assert r.has_value(), f"Warmup get: {r.error()}"
        torch.cuda.synchronize(1)

        latencies_us: list[float] = []
        for _ in range(self.bench_iters):
            torch.cuda.synchronize(1)
            t0 = time.perf_counter()
            f = conn.get(requests=requests)
            f.wait_for(timeout_ms=60000)
            r = f.get()
            torch.cuda.synchronize(1)
            t1 = time.perf_counter()
            assert r.has_value(), f"get failed: {r.error()}"
            latencies_us.append((t1 - t0) * 1e6)

        lat = Stats.compute(latencies_us)
        bw = transfer_bytes / (lat.avg * 1e-6) / 1e9 if lat.avg > 0 else 0.0

        return BenchResult(
            block_count=block_count,
            num_layers=self.config.num_layers,
            total_requests=len(requests),
            total_bytes=transfer_bytes,
            iterations=self.bench_iters,
            latency=lat,
            bw_gbps=bw,
        )


# ---------------------------------------------------------------------------
# BenchmarkReporter — table and CSV output
# ---------------------------------------------------------------------------


class BenchmarkReporter:
    """Formats benchmark results as table or CSV."""

    def __init__(self, config: KVCacheConfig) -> None:
        self.config = config

    def print_table(self, results: list[BenchResult]) -> None:
        cfg = self.config
        out = sys.stderr

        out.write("\n")
        out.write("=" * 78 + "\n")
        out.write("              UniFlow KV-Cache Transfer Benchmark\n")
        out.write("  GPUs: cuda:0 (prefill) <-> cuda:1 (decode)\n")
        out.write(
            f"  KV layout: block_size={cfg.block_size}, "
            f"num_kv_heads={cfg.num_kv_heads}, "
            f"head_size={cfg.head_size}, dtype={cfg.dtype}\n"
        )
        coalesce_str = ", coalesced" if cfg.coalesce else ""
        out.write(
            f"  Per-layer block: {cfg.block_bytes / 1024:.2f} KB, "
            f"{cfg.num_blocks} blocks, "
            f"{cfg.num_layers} layer(s) per batch{coalesce_str}\n"
        )
        out.write("=" * 78 + "\n\n")
        out.write(
            "-- Block Transfer (get: decode reads from prefill) " + "-" * 28 + "\n"
        )
        out.write(
            f"  {'Blocks':>6}  {'Reqs':>6}    {'Total (MB)':>10}    {'Iters':>5}    "
            f"{'BW (GB/s)':>9}    {'Lat avg(us)':>11}    "
            f"{'Lat p50(us)':>11}    {'Lat p99(us)':>11}\n"
        )
        for r in results:
            mb = r.total_bytes / (1024 * 1024)
            out.write(
                f"  {r.block_count:>6}  {r.total_requests:>6}    {mb:>10.2f}    "
                f"{r.iterations:>5}    "
                f"{r.bw_gbps:>9.1f}    {r.latency.avg:>11.1f}    "
                f"{r.latency.p50:>11.1f}    {r.latency.p99:>11.1f}\n"
            )
        out.write("\n")
        out.flush()

    def write_csv(self, path: str, results: list[BenchResult]) -> None:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "blocks",
                    "num_layers",
                    "total_requests",
                    "total_bytes",
                    "iterations",
                    "bw_gbps",
                    "lat_avg_us",
                    "lat_p50_us",
                    "lat_p99_us",
                    "lat_min_us",
                    "lat_max_us",
                ]
            )
            for r in results:
                w.writerow(
                    [
                        r.block_count,
                        r.num_layers,
                        r.total_requests,
                        r.total_bytes,
                        r.iterations,
                        f"{r.bw_gbps:.3f}",
                        f"{r.latency.avg:.1f}",
                        f"{r.latency.p50:.1f}",
                        f"{r.latency.p99:.1f}",
                        f"{r.latency.min:.1f}",
                        f"{r.latency.max:.1f}",
                    ]
                )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="UniFlow KV-cache transfer benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--num-layers",
        type=int,
        default=1,
        help="Simulated transformer layers (multiplies TransferRequests per batch)",
    )
    p.add_argument(
        "--num-blocks", type=int, default=256, help="Total KV cache blocks per GPU"
    )
    p.add_argument(
        "--block-size", type=int, default=16, help="Tokens per KV cache block"
    )
    p.add_argument(
        "--num-kv-heads",
        type=int,
        default=8,
        help="Number of KV attention heads (GQA heads)",
    )
    p.add_argument(
        "--head-size", type=int, default=128, help="Dimension per attention head"
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=list(_DTYPE_MAP),
        help="Tensor data type",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations per block-count size point",
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Timed iterations per block-count size point",
    )
    p.add_argument(
        "--csv", type=str, default="", help="Path to write CSV results (empty = skip)"
    )
    p.add_argument(
        "--coalesce",
        action="store_true",
        help="Coalesce contiguous blocks into a single TransferRequest per layer",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Test entry point (for buck test)
# ---------------------------------------------------------------------------


class TestKVTransferBench(unittest.TestCase):
    @unittest.skipIf(_SKIP_NO_MULTI_GPU, "Need at least 2 GPUs")
    def test_benchmark(self) -> None:
        argv: list[str] = []
        if os.environ.get("UNIFLOW_BENCH_COALESCE"):
            argv.append("--coalesce")
        if os.environ.get("UNIFLOW_BENCH_BLOCK_SIZE"):
            argv.extend(["--block-size", os.environ["UNIFLOW_BENCH_BLOCK_SIZE"]])
        args = _parse_args(argv)
        config = KVCacheConfig.from_args(args)
        reporter = BenchmarkReporter(config)

        with TransferSession(config) as session:
            bench = KVTransferBenchmark(session, args.warmup, args.iterations)
            results = bench.run()

        self.assertGreater(len(results), 0, "No benchmark results")
        for r in results:
            self.assertGreater(
                r.bw_gbps, 0, f"Zero bandwidth for {r.block_count} blocks"
            )
        reporter.print_table(results)
        if args.csv:
            reporter.write_csv(args.csv, results)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if _SKIP_NO_MULTI_GPU:
        print("ERROR: Need at least 2 CUDA GPUs", file=sys.stderr)
        sys.exit(1)
    args = _parse_args()
    config = KVCacheConfig.from_args(args)
    reporter = BenchmarkReporter(config)

    with TransferSession(config) as session:
        bench = KVTransferBenchmark(session, args.warmup, args.iterations)
        results = bench.run()

    reporter.print_table(results)
    if args.csv:
        reporter.write_csv(args.csv, results)
        print(f"CSV written to {args.csv}", file=sys.stderr)
