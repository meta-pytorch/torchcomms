# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe
# (suppression) Benchmark helpers over untyped torch.distributed / cute symbols pyre cannot
# model; strict typing adds no value here (not shipped code).
# Shared benchmark helpers (timing / bandwidth / NCCL grid + baseline), extracted from
# benchmark_a2a so the Triton and CuTe a2a benchmarks import ONE lib instead of pulling the
# whole driver module as a src. Behaviour is unchanged (verbatim move).

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


_NCCL_A2A_GRID_TABLE: dict[tuple[int, int], int] = {
    (65536, 8): 8,  # 64KB cap=8
    (65536, 16): 8,  # 64KB cap=16
    (65536, 32): 8,  # 64KB cap=32
    (262144, 8): 8,  # 256KB cap=8
    (262144, 16): 8,  # 256KB cap=16
    (262144, 32): 8,  # 256KB cap=32
    (1048576, 8): 8,  # 1MB cap=8
    (1048576, 16): 16,  # 1MB cap=16
    (1048576, 32): 16,  # 1MB cap=32
    (8388608, 8): 8,  # 8MB cap=8
    (8388608, 16): 16,  # 8MB cap=16
    (8388608, 32): 32,  # 8MB cap=32
    (
        50331648,
        8,
    ): 8,  # 48MB cap=8 (estimated: cap-plateau, between probed 8MB and 64MB)
    (50331648, 16): 16,  # 48MB cap=16 (estimated)
    (50331648, 32): 32,  # 48MB cap=32 (estimated)
    (67108864, 8): 8,  # 64MB cap=8
    (67108864, 16): 16,  # 64MB cap=16
    (67108864, 32): 32,  # 64MB cap=32
    (
        100663296,
        8,
    ): 8,  # 96MB cap=8 (estimated: cap-plateau, between probed 64MB and 256MB)
    (100663296, 16): 16,  # 96MB cap=16 (estimated)
    (100663296, 32): 32,  # 96MB cap=32 (estimated)
    (268435456, 8): 8,  # 256MB cap=8
    (268435456, 16): 16,  # 256MB cap=16
    (268435456, 32): 32,  # 256MB cap=32
    # 512MB / 1GB: ESTIMATED at the cap-plateau (NCCL caps its P2P channels at the
    # allocation; the 256MB row and the sendrecv 1GB row both plateau cap=grid).
    # Re-probe with --nccl-only under nsys to lock exact values for these sizes.
    (536870912, 8): 8,  # 512MB cap=8 (estimated)
    (536870912, 16): 16,  # 512MB cap=16 (estimated)
    (536870912, 32): 32,  # 512MB cap=32 (estimated)
    (1073741824, 8): 8,  # 1GB cap=8 (estimated)
    (1073741824, 16): 16,  # 1GB cap=16 (estimated)
    (1073741824, 32): 32,  # 1GB cap=32 (estimated)
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
    """Steady-state μs/iter: median over ``windows`` warmed, re-barriered windows.

    Mirrors the perf2.md stability method -- each window event-times ``iters``
    graph replays after a fresh ``dist.barrier`` re-aligns both ranks, and the
    median across windows rejects the occasional bad window (a transient SM-clock
    dip or neighbour kernel) that a single long mean would bake in.
    """
    if windows is None:
        # Read lazily (not at module scope) per python.md. `or "5"` covers an EMPTY value (env
        # plumbing often sets ""), try/except covers a non-numeric typo, and max(1, ...) floors
        # 0/negative -- so a bad A2A_PERF_WINDOWS falls back to 5 instead of crashing a MAST sweep.
        try:
            windows = max(1, int(os.environ.get("A2A_PERF_WINDOWS") or "5"))
        except ValueError:
            windows = 5
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
    ``ws * num_blocks``, matched to NCCL's active SMs via ``num_blocks ~=
    nccl_grid / ws``. If the (msg, cap) grid has not been probed, fall back to the
    SM-budget ceiling and return ``None`` so the row is flagged as not-yet-matched.
    """
    nb_cap = _max_num_blocks(ws, device, mbp)
    nccl_grid = _NCCL_A2A_GRID_TABLE.get((msg_bytes, cap))
    if nccl_grid is None:
        return nb_cap, None
    # Framework grid = ws * num_blocks; match it to NCCL's active SMs.
    nb = max(round(nccl_grid / ws), 1)
    return min(nb, nb_cap), nccl_grid


def _full_sm_policy(
    msg_bytes: int, ws: int, device: torch.device, mbp: int
) -> tuple[int, dict[str, object]]:
    """Production (full-GPU) config the autotuner emits: ``(num_blocks, extra_cfg)``.

    NCCL P2P saturates at its channel ceiling (~32 SMs) and provably cannot use
    more (forced 48/64 channels ~= default), so during a comm-bound a2a the rest
    of the GPU is idle. The framework spends those idle SMs: small msgs stay
    latency-bound (1 CTA/peer), >=1 MiB scale to the SM budget, and the >=64 MiB
    band additionally takes the swept-best fine-slot (256 KiB) + deep run-ahead
    (pipeline_depth 8) + warp specialization (SEND/RECV on disjoint warp groups,
    which only wins at high CTA counts where each block's sub-chunk is small enough
    that the WS register-cap doesn't cost copy ILP -- the reverse of the SM-matched
    regime, where WS loses). Together these lift large-message bandwidth from the
    SM-matched ~1.8x to ~1.06-1.11x of NCCL on otherwise-idle SMs. Mirrors what a
    tuning run writes into TUNED_A2A_CONFIGS; the SM-matched table is the
    per-SM-efficiency lens.
    """
    nb_cap = _max_num_blocks(ws, device, mbp)
    if msg_bytes < 1024 * 1024:
        return 1, {}
    if msg_bytes < 64 * 1024 * 1024:
        return nb_cap, {}
    return nb_cap, {
        "block_stride_bytes": 256 * 1024,
        "pipeline_depth": 8,
        "primitive": "ws",
    }


def _report_full_cfg(
    msg_bytes: int, ws: int, device: torch.device, mbp: int
) -> tuple[int, dict[str, object]]:
    """Full-GPU launch base (num_blocks + stride/depth) WITHOUT a structure choice.

    Same SM-budget + large-band fine-slot/run-ahead as ``_full_sm_policy`` but with
    the ``primitive`` choice left out, so the perf report can layer each structure
    (primitive = "copy"/"ws"/"tma") on top of one identical framing and compare
    structures apple-to-apple at the full CTA count.
    """
    nb_cap = _max_num_blocks(ws, device, mbp)
    if msg_bytes < 1024 * 1024:
        return 1, {}
    if msg_bytes < 64 * 1024 * 1024:
        return nb_cap, {}
    return nb_cap, {"block_stride_bytes": 256 * 1024, "pipeline_depth": 8}


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

    This is the channel the UniBench / AnyBench parser
    (``comms/dsl/tests/anybench_a2a_cute_parser.py``) reads back from each rank's
    ``$ANYBENCH_LOGS_DIR/rank_*/stdout.log`` and pushes to the ``anybench_parser_output``
    Scuba dataset -- replacing the previous private Scuba sink. stdout is durable on MAST via
    the AnyBench log capture, so no benchmark-side Scuba/Scribe dependency is needed. Backend-
    neutral (both DSLs' benchmarks call it). Best-effort: never raises into the benchmark."""
    for row in rows:
        try:
            print(f"{_RESULT_TAG} " + json.dumps(row, sort_keys=True), flush=True)
        except (TypeError, ValueError, OSError) as e:
            # Best-effort sink: an unserializable row (TypeError/ValueError) or a broken stdout
            # (OSError/BrokenPipeError, e.g. MAST log rotation) must not abort the benchmark. The
            # fallback print can itself hit the broken stream, so it is guarded too.
            try:
                print(f"(emit_result_rows skipped a row: {e})", flush=True)
            except OSError:
                pass
