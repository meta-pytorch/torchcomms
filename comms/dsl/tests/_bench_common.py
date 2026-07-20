# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe
# (suppression) Benchmark helpers over untyped torch.distributed / cute symbols pyre cannot
# model; strict typing adds no value here (not shipped code).

"""Backend-agnostic a2a benchmark helpers: CUDA-graph capture/replay timing, bus-bandwidth
math, the probed NCCL runtime-grid table + num_blocks policy, and the NCCL baseline."""

from __future__ import annotations

import json
import os
import socket
import statistics
from typing import Any, Callable

import torch
import torch.distributed as dist


_DTYPE: torch.dtype = torch.bfloat16


# NCCL all_to_all actual LAUNCHED a2a-kernel grid (ncclDevKernel_SendRecv gridDim.x; block is a
# fixed 640 threads, send+recv fused in one CTA) per (per-rank msg bytes, NCCL channel cap).
# Used to SM-match the framework grid (ws * num_blocks) to NCCL for an apple-to-apple compare.
#
# `cap` is the NCCL channel cap in force (NCCL_MAX_NCHANNELS=NCCL_MIN_NCHANNELS=cap); the value
# is the grid NCCL ACTUALLY launches under it. This is a real per-(size, cap) measurement, NOT
# min(natural, cap): NCCL's scheduleP2pTasksToPlan adaptively drops the active grid BELOW the
# cap/allocation for small messages, so the (size, cap) matrix does not collapse to a single
# per-size number. (NCCL_DEBUG=INFO "N p2p channels" is the ALLOCATION, which over-reports this
# launched grid.)
#
# Measured via tests/_probe_nccl_grid.py (torch.profiler launched-kernel grid; run once per cap
# with NCCL_MAX_NCHANNELS=NCCL_MIN_NCHANNELS=cap), cross-checked against nsys cuda_gpu_trace
# (64KB -> grid 8, block 640; both agree). Re-run on an NCCL version OR NVLink-topology change
# (the ceiling is topology-derived, not GPU-arch-derived -- NCCL's p2p channel math has no
# SM-count / sm_90-vs-sm_100 dependence). Validated on:
#   - NCCL/NCCLX version : 2.30.7
#   - topology           : 8-GPU single-node NVLink
_NCCL_A2A_GRID_TABLE: dict[tuple[int, int], int] = {
    (65536, 8): 8,  # 64KB
    (65536, 16): 8,
    (65536, 32): 8,
    (262144, 8): 8,  # 256KB
    (262144, 16): 8,
    (262144, 32): 8,
    (1048576, 8): 8,  # 1MB
    (1048576, 16): 16,
    (1048576, 32): 16,
    (8388608, 8): 8,  # 8MB
    (8388608, 16): 16,
    (8388608, 32): 32,
    (50331648, 8): 8,  # 48MB
    (50331648, 16): 16,
    (50331648, 32): 32,
    (67108864, 8): 8,  # 64MB
    (67108864, 16): 16,
    (67108864, 32): 32,
    (100663296, 8): 8,  # 96MB
    (100663296, 16): 16,
    (100663296, 32): 32,
    (268435456, 8): 8,  # 256MB
    (268435456, 16): 16,
    (268435456, 32): 32,
    (536870912, 8): 8,  # 512MB
    (536870912, 16): 16,
    (536870912, 32): 32,
    (1073741824, 8): 8,  # 1GB
    (1073741824, 16): 16,
    (1073741824, 32): 32,
    (2147483648, 8): 8,  # 2GB
    (2147483648, 16): 16,
    (2147483648, 32): 32,
}


_WARMUP_ITERS: int = 20


_CAPTURE_WARMUP_ITERS: int = 3


def _iters_for_size(msg_bytes: int) -> int:
    if msg_bytes <= 64 * 1024:
        return 500
    if msg_bytes <= 64 * 1024 * 1024:
        return 200
    return 50


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _capture_one_call(fn: Callable[[], None]) -> torch.cuda.CUDAGraph:
    """Warm up ``fn`` on a side stream, then capture a single call into a graph."""
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(_CAPTURE_WARMUP_ITERS):
            fn()
    torch.cuda.current_stream().wait_stream(side_stream)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=side_stream):
        fn()
    return g


def _time_replays(
    g: torch.cuda.CUDAGraph,
    iters: int,
    device: torch.device,
    group: dist.ProcessGroup,
    windows: int | None = None,
) -> float:
    """Steady-state us/iter: median over ``windows`` warmed, re-barriered windows.

    Each window event-times ``iters`` graph replays after a fresh ``dist.barrier``
    re-aligns both ranks, and the median across windows rejects the occasional bad window
    (a transient SM-clock dip or neighbour kernel) that a single long mean would bake in.
    """
    if windows is None:
        # Read lazily (not at module scope). `or "5"` covers an EMPTY value (env plumbing
        # often sets ""), try/except covers a non-numeric typo, and max(1, ...) floors
        # 0/negative -- so a bad A2A_PERF_WINDOWS falls back to 5 instead of crashing.
        try:
            windows = max(1, int(os.environ.get("A2A_PERF_WINDOWS") or "5"))
        except ValueError:
            windows = 5
    # The inner `iters` loop free-runs graph replays with NO per-iter cross-rank barrier
    # (only a per-window one), which looks like the reused-transport hazard the a2a docs warn
    # about. It is safe HERE: correctness is already gated before capture (in the caller), so
    # the timing loop only measures steady-state throughput -- a staging write-after-read
    # during timing corrupts data we never re-check. It cannot hang either: the transport's
    # monotonic step counters mean every device wait eventually sees its (ever-increasing)
    # signal even under rank drift. The per-window dist.barrier re-aligns both ranks so each
    # window's median is clean.
    for _ in range(_WARMUP_ITERS):
        g.replay()
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(windows):
        dist.barrier(group)
        torch.cuda.synchronize(device)
        start.record()
        for _ in range(iters):
            g.replay()
        end.record()
        torch.cuda.synchronize(device)
        samples.append((start.elapsed_time(end) * 1000.0) / iters)
    return statistics.median(samples)


def _max_rank_latency(
    lat_us: float, device: torch.device, group: dist.ProcessGroup
) -> float:
    latency = torch.tensor([lat_us], dtype=torch.float64, device=device)
    dist.all_reduce(latency, op=dist.ReduceOp.MAX, group=group)
    return float(latency.item())


def _bus_bytes(numel: int, ws: int) -> int:
    """Per-rank NVLink-traversing bytes (diagonal chunk excluded)."""
    chunk = numel // ws
    return chunk * (ws - 1) * torch.tensor([], dtype=_DTYPE).element_size()


def _bw_gbps(numel: int, ws: int, lat_us: float) -> float:
    return (_bus_bytes(numel, ws) / 1e9) / (lat_us / 1e6) if lat_us > 0 else 0.0


def _fmt_size(nbytes: int) -> str:
    """Human-readable per-rank byte size for the result table, e.g. 65536 -> ``64KB``,
    50331648 -> ``48MB``, 2147483648 -> ``2GB``."""
    x = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024.0:
            return f"{x:.0f}{unit}" if x == int(x) else f"{x:.1f}{unit}"
        x /= 1024.0
    return f"{x:.1f}PB"


def _make_input(rank: int, numel: int, device: torch.device) -> torch.Tensor:
    base = (rank + 1) * 4096
    return (base + torch.arange(numel, device=device, dtype=torch.float32)).to(_DTYPE)


def _max_num_blocks(ws: int, device: torch.device, mbp: int) -> int:
    # Grid is ws * num_blocks (one CTA per (peer, block)), so the SM budget bounds
    # num_blocks at sm // ws.
    sm = torch.cuda.get_device_properties(device).multi_processor_count
    return max(min(mbp, sm // ws), 1)


def _framework_num_blocks(
    msg_bytes: int, cap: int, ws: int, device: torch.device, mbp: int
) -> tuple[int, int | None]:
    """Pick framework num_blocks to track NCCL's runtime grid (SM-matched).

    Returns ``(num_blocks, nccl_grid_or_None)``. The framework device grid is
    ``ws * num_blocks``, matched to NCCL's active SMs via ``num_blocks ~= nccl_grid / ws``,
    where ``nccl_grid`` is NCCL's actual launched grid for this (size, cap) from
    ``_NCCL_A2A_GRID_TABLE``. If the (size, cap) has not been probed, fall back to the
    SM-budget ceiling and return ``None`` so the row is flagged as not-yet-matched.
    """
    nb_cap = _max_num_blocks(ws, device, mbp)
    nccl_grid = _NCCL_A2A_GRID_TABLE.get((msg_bytes, cap))
    if nccl_grid is None:
        return nb_cap, None
    # Framework grid = ws * num_blocks; match it to NCCL's active SMs.
    nb = max(round(nccl_grid / ws), 1)
    return min(nb, nb_cap), nccl_grid


def _bench_nccl(
    msg_bytes: int, rank: int, ws: int, group: dist.ProcessGroup
) -> tuple[float, float]:
    device = torch.device("cuda", torch.cuda.current_device())
    numel = max(msg_bytes // torch.tensor([], dtype=_DTYPE).element_size(), ws)
    numel -= numel % ws
    inp: torch.Tensor = _make_input(rank, numel, device)
    out: torch.Tensor = torch.empty_like(inp)

    def fn() -> None:
        dist.all_to_all_single(out, inp, group=group)

    fn()
    dist.barrier(group)
    g = _capture_one_call(fn)
    dist.barrier(group)

    iters = _iters_for_size(msg_bytes)
    lat = _max_rank_latency(_time_replays(g, iters, device, group), device, group)
    return lat, _bw_gbps(numel, ws, lat)


_RESULT_TAG: str = "A2A_RESULT_JSON"


def emit_result_rows(rows: list[dict[str, Any]]) -> None:
    """DSL-agnostic durable result sink: print one compact JSON object per row to stdout,
    each tagged with a stable prefix.

    This is the channel the AnyBench parser (``anybench_a2a_cute_parser.py``) reads back
    from each rank's ``$ANYBENCH_LOGS_DIR/rank_*/stdout.log`` and pushes to the
    ``anybench_parser_output`` Scuba dataset. stdout is durable on MAST via the AnyBench log
    capture, so no benchmark-side Scuba/Scribe dependency is needed. Best-effort: never
    raises into the benchmark."""
    for row in rows:
        try:
            print(f"{_RESULT_TAG} " + json.dumps(row, sort_keys=True), flush=True)
        except (TypeError, ValueError, OSError) as e:
            # Best-effort sink: an unserializable row (TypeError/ValueError) or a broken
            # stdout (OSError/BrokenPipeError, e.g. MAST log rotation) must not abort the
            # benchmark. The fallback print can itself hit the broken stream, so guard it too.
            try:
                print(f"(emit_result_rows skipped a row: {e})", flush=True)
            except OSError:
                pass
