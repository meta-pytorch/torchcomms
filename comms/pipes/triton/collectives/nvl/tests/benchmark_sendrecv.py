# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""Benchmark for the NVLink copy-based Triton send/recv vs NCCL baseline.

**Runtime-SM-matched** apples-to-apples unidirectional and bidirectional
sendrecv between rank 0 and rank 1 on a single node, NVLink-only.

Key point: ``NCCL_{MIN,MAX}_NCHANNELS`` only controls *allocation*. NCCL's
in-kernel adaptive policy decides how many of those allocated channels
(= CTAs = SMs) to actually launch per call. Therefore we **first measure
NCCL's runtime grid for every (msg_size, allocation_cap)** via nsys (see
``_nccl_grid_probe.py`` companion target) and then pin Triton's
``num_blocks`` to that runtime grid — guaranteeing that the per-cell
comparison reflects identical SM usage on both kernels.

Measured NCCL P2P kernel grid (= active SM count) on H100, NCCLX 2.29::

      size   cap=4  cap=8  cap=16  cap=32
       32 B    1      1      1       1
       64 B    1      1      1       1
       1 KB    1      1      1       1
       8 KB    1      1      1       1
      64 KB    1      1      1       1
       1 MB    2      4      8      16
       8 MB    2      4      8      16
      64 MB    4      4      8      16
     256 MB    4      8     16      16
       1 GB    4      8     16      32

Both unidirectional and bidirectional ``batch_isend_irecv`` produce the
same NCCL grid (channels shared across the op set). Triton bidirectional
uses a 2*num_blocks grid by construction, so for bidirectional we set
``num_blocks = max(nccl_grid // 2, 1)`` (resulting Triton grid =
``max(nccl_grid, 2)``); the only mismatch is at small-msg bi where
NCCL=1 but Triton=2 — disclosed in commit summary.

Both calls are captured into CUDA graphs and timed via ``graph.replay()``
to remove the per-iteration Python and host launcher overhead.

Fairness caveats:
  * **Dtype**: bf16 only (representative ML-training payload dtype).
  * **NCCL P2P + CUDA graph**: requires PyTorch + a NCCL build with
    graph-aware P2P (NCCL 2.18+ or fbsource ``hpc_comms.use_ncclx=stable``).
  * **NCCL P2P uses Simple protocol only.** ``NCCL_PROTO=LL`` only
    affects collectives (verified via ``NCCL_DEBUG=INFO`` — AllReduce
    shows ``proto LL``, but ``batch_isend_irecv`` allocates Simple-style
    10 MB proxy buffers). NCCL has no LL/LL128 P2P path, so the small-
    msg comparison is Triton-Simple-vs-NCCL-Simple.
  * **Bidirectional minimum grid asymmetry**: Triton bidirectional grid
    is always ``2 * num_blocks >= 2``; NCCL bidirectional grid drops to
    1 at small msg. At cells where the table prescribes Triton
    num_blocks=1 (=>grid=2) vs NCCL grid=1, Triton uses 2× SMs, which
    we explicitly disclose.

Run::

    buck2 run @mode/opt -c hpc_comms.use_ncclx=stable \\
        //comms/pipes/triton/collectives/nvl/tests:benchmark_sendrecv
"""

from __future__ import annotations

import os
import socket
import sys
from typing import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from comms.pipes.triton.collectives.nvl.sendrecv_op import (
    triton_nvl_recv,
    triton_nvl_send,
    triton_nvl_sendrecv,
    triton_nvl_sendrecv_ws,
)


# Message sizes to sweep (in bytes). 8 B and 16 B are excluded:
# they are not target cases, and the current Triton codegen hits a
# tiny-message divisibility cliff there that is not representative of
# the copy-based path.
_MSG_SIZES_BYTES: list[int] = [
    32,
    64,
    1024,
    8 * 1024,
    64 * 1024,
    1 * 1024 * 1024,
    8 * 1024 * 1024,
    64 * 1024 * 1024,
    256 * 1024 * 1024,
    1024 * 1024 * 1024,
]


_WS_SENDER_WARPS: int = int(os.environ.get("TRITON_NVL_WS_SENDER_WARPS", "4"))
_WS_RECEIVER_WARPS: int = int(os.environ.get("TRITON_NVL_WS_RECEIVER_WARPS", "4"))
_WS_MAX_BYTES: int = int(
    os.environ.get("TRITON_NVL_WS_MAX_BYTES", str(64 * 1024 * 1024))
)


# NCCL ``NCCL_{MIN,MAX}_NCHANNELS`` allocation values to sweep.
# At each cap we compare against NCCL with that allocation, and Triton
# with ``num_blocks`` matched to NCCL's actual runtime grid (see table
# in module docstring).
_CAPS: list[int] = [4, 8, 16, 32]


def _env_int_list(name: str, default: list[int]) -> list[int]:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


# Hardcoded NCCL P2P kernel runtime grid table, keyed by
# (msg_bytes, cap) -> grid count. Generated via nsys profiling of
# ``ncclDevKernel_SendRecv`` on H100, NCCLX 2.29 (see
# ``_nccl_grid_probe.py``). ``ncclDevKernelArgs`` is the same for uni
# and bi; both directions share the channel pool, so the runtime grid
# is identical regardless of P2POp count.
_NCCL_GRID_TABLE: dict[tuple[int, int], int] = {
    (32, 4): 1,
    (32, 8): 1,
    (32, 16): 1,
    (32, 32): 1,
    (64, 4): 1,
    (64, 8): 1,
    (64, 16): 1,
    (64, 32): 1,
    (1024, 4): 1,
    (1024, 8): 1,
    (1024, 16): 1,
    (1024, 32): 1,
    (8 * 1024, 4): 1,
    (8 * 1024, 8): 1,
    (8 * 1024, 16): 1,
    (8 * 1024, 32): 1,
    (64 * 1024, 4): 1,
    (64 * 1024, 8): 1,
    (64 * 1024, 16): 1,
    (64 * 1024, 32): 1,
    (1 * 1024 * 1024, 4): 2,
    (1 * 1024 * 1024, 8): 4,
    (1 * 1024 * 1024, 16): 8,
    (1 * 1024 * 1024, 32): 16,
    (8 * 1024 * 1024, 4): 2,
    (8 * 1024 * 1024, 8): 4,
    (8 * 1024 * 1024, 16): 8,
    (8 * 1024 * 1024, 32): 16,
    (64 * 1024 * 1024, 4): 4,
    (64 * 1024 * 1024, 8): 4,
    (64 * 1024 * 1024, 16): 8,
    (64 * 1024 * 1024, 32): 16,
    (256 * 1024 * 1024, 4): 4,
    (256 * 1024 * 1024, 8): 8,
    (256 * 1024 * 1024, 16): 16,
    (256 * 1024 * 1024, 32): 16,
    (1024 * 1024 * 1024, 4): 4,
    (1024 * 1024 * 1024, 8): 8,
    (1024 * 1024 * 1024, 16): 16,
    (1024 * 1024 * 1024, 32): 32,
}


def _triton_num_blocks_uni(msg_bytes: int, cap: int) -> int:
    """Triton uni grid = num_blocks. Match NCCL runtime grid exactly."""
    return _NCCL_GRID_TABLE[(msg_bytes, cap)]


def _triton_num_blocks_bi(msg_bytes: int, cap: int) -> int:
    """Triton bi grid = 2 * num_blocks. Half the NCCL grid; minimum 1.

    At cells where NCCL bi grid = 1, Triton bi grid = 2 (slight
    Triton-side over-provision; disclosed in module docstring).
    """
    return max(_NCCL_GRID_TABLE[(msg_bytes, cap)] // 2, 1)


# Per-message iteration counts. The 64 KB / 64 MiB thresholds split the
# sweep into three latency regimes:
#   * <= 64 KB: copy time is microseconds; per-replay timing variance is
#     a meaningful fraction of the signal, so 500 replays drives stderr
#     down without inflating wall clock.
#   * 64 KB – 64 MiB: bandwidth-bound regime; 200 replays is enough for
#     stable means.
#   * > 64 MiB: each replay is milliseconds, 50 replays keeps the full
#     sweep under a couple of minutes per cap.
def _iters_for_size(msg_bytes: int) -> int:
    if msg_bytes <= 64 * 1024:
        return 500
    if msg_bytes <= 64 * 1024 * 1024:
        return 200
    return 50


# Empirically chosen on H100: 20 replays is enough for steady-state SM
# clock / cache state on the timed graph, and 3 capture-warmup replays
# is enough to populate any first-call lazy state inside ``fn`` (e.g.,
# Triton autotune cache, NCCL communicator hot path) before capture.
_WARMUP_ITERS: int = 20
_CAPTURE_WARMUP_ITERS: int = 3


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _capture_one_call(fn: Callable[[], None]) -> torch.cuda.CUDAGraph:
    """Warm up ``fn`` on a side stream, then capture a single call into a graph.

    Standard PyTorch CUDA-graph capture pattern (see
    https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs):

      1. Warm up on a non-default stream so the kernel's JIT compile,
         allocator, and any communicator setup happens before capture.
      2. Make the current stream wait for the warmup to finish.
      3. Capture a single ``fn`` call into the graph using the same
         non-default stream.

    The caller is responsible for ensuring ``fn`` does no Python-level
    allocation that hasn't already happened by the time this helper
    returns — graph replay only re-runs the recorded GPU ops.
    """
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
) -> float:
    """Time ``iters`` replays of ``g`` using CUDA events. Returns μs/iter."""
    for _ in range(_WARMUP_ITERS):
        g.replay()
    torch.cuda.synchronize(device)
    # Align ranks at the start of the timed window so warmup skew on either
    # side does not inflate the first few timed replays.
    dist.barrier(group)
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        g.replay()
    end.record()
    torch.cuda.synchronize(device)
    elapsed_ms = start.elapsed_time(end)
    return (elapsed_ms * 1000.0) / iters


def _max_rank_latency(
    lat_us: float,
    device: torch.device,
    group: dist.ProcessGroup,
) -> float:
    latency = torch.tensor([lat_us], dtype=torch.float64, device=device)
    dist.all_reduce(latency, op=dist.ReduceOp.MAX, group=group)
    return float(latency.item())


def _bench_triton(
    msg_bytes: int,
    local_rank: int,
    peer_rank: int,
    group: dist.ProcessGroup,
    *,
    bidirectional: bool,
    num_blocks: int,
    warp_specialized: bool = False,
) -> tuple[float, float]:
    """Bench Triton send/recv via CUDA-graph replay. Returns (μs, GB/s)."""
    device = torch.device(f"cuda:{local_rank}")
    numel = max(msg_bytes // torch.bfloat16.itemsize, 1)
    send: torch.Tensor = torch.ones(numel, dtype=torch.bfloat16, device=device)
    recv: torch.Tensor = torch.zeros(numel, dtype=torch.bfloat16, device=device)

    def fn() -> None:
        if bidirectional:
            if warp_specialized:
                triton_nvl_sendrecv_ws(
                    send,
                    recv,
                    peer_rank,
                    group=group,
                    num_blocks=num_blocks,
                    sender_warps=_WS_SENDER_WARPS,
                    receiver_warps=_WS_RECEIVER_WARPS,
                )
            else:
                triton_nvl_sendrecv(
                    send,
                    recv,
                    peer_rank,
                    group=group,
                    num_blocks=num_blocks,
                )
        elif local_rank == 0:
            triton_nvl_send(
                send,
                peer_rank,
                group=group,
                num_blocks=num_blocks,
            )
        else:
            triton_nvl_recv(
                recv,
                peer_rank,
                group=group,
                num_blocks=num_blocks,
            )

    fn()
    dist.barrier(group)

    g = _capture_one_call(fn)
    dist.barrier(group)

    iters = _iters_for_size(msg_bytes)
    lat_us = _max_rank_latency(_time_replays(g, iters, device, group), device, group)
    bw_gbps = (msg_bytes / 1e9) / (lat_us / 1e6) if lat_us > 0 else 0.0

    if bidirectional or local_rank == 1:
        torch.cuda.synchronize(device)
        expected = torch.ones(numel, dtype=torch.bfloat16, device=device)
        assert torch.equal(recv, expected), (
            f"Triton bench correctness check failed at msg_bytes={msg_bytes}, "
            f"bidirectional={bidirectional}, num_blocks={num_blocks}"
        )
    return lat_us, bw_gbps


def _bench_nccl(
    msg_bytes: int,
    local_rank: int,
    peer_rank: int,
    group: dist.ProcessGroup,
    *,
    bidirectional: bool,
) -> tuple[float, float]:
    """Bench NCCL ``batch_isend_irecv`` via CUDA-graph replay. Returns (μs, GB/s).

    NCCL channel allocation is set by ``NCCL_{MIN,MAX}_NCHANNELS`` in
    the parent ``main()`` before ``mp.spawn``; per-call runtime grid is
    NCCL's adaptive choice (captured in ``_NCCL_GRID_TABLE``).
    """
    device = torch.device(f"cuda:{local_rank}")
    numel = max(msg_bytes // torch.bfloat16.itemsize, 1)
    send: torch.Tensor = torch.ones(numel, dtype=torch.bfloat16, device=device)
    recv: torch.Tensor = torch.zeros(numel, dtype=torch.bfloat16, device=device)

    def fn() -> None:
        if bidirectional:
            ops = [
                dist.P2POp(dist.isend, send, peer_rank, group=group),
                dist.P2POp(dist.irecv, recv, peer_rank, group=group),
            ]
        elif local_rank == 0:
            ops = [dist.P2POp(dist.isend, send, peer_rank, group=group)]
        else:
            ops = [dist.P2POp(dist.irecv, recv, peer_rank, group=group)]
        reqs = dist.batch_isend_irecv(ops)
        for r in reqs:
            r.wait()

    fn()
    dist.barrier(group)

    g = _capture_one_call(fn)
    dist.barrier(group)

    iters = _iters_for_size(msg_bytes)
    lat_us = _max_rank_latency(_time_replays(g, iters, device, group), device, group)
    bw_gbps = (msg_bytes / 1e9) / (lat_us / 1e6) if lat_us > 0 else 0.0

    if bidirectional or local_rank == 1:
        torch.cuda.synchronize(device)
        expected = torch.ones(numel, dtype=torch.bfloat16, device=device)
        assert torch.equal(recv, expected), (
            f"NCCL bench correctness check failed at msg_bytes={msg_bytes}, "
            f"bidirectional={bidirectional}"
        )
    return lat_us, bw_gbps


def _fmt_size(b: int) -> str:
    units = [("B", 1), ("KB", 1024), ("MB", 1024**2), ("GB", 1024**3)]
    for name, scale in reversed(units):
        if b >= scale:
            return f"{b // scale}{name}" if b % scale == 0 else f"{b / scale:.1f}{name}"
    return f"{b}B"


def _print_header(rank: int, title: str) -> None:
    if rank != 0:
        return
    print()
    print("=" * 100)
    print(title)
    print(
        f"{'msg_size':>10} | {'iters':>6} | {'tri_nb':>6} {'nccl_grid':>9} | "
        f"{'triton_us':>10} {'triton_GB/s':>12} | "
        f"{'nccl_us':>10} {'nccl_GB/s':>12} | {'speedup':>8}"
    )
    print("-" * 100)


def _print_row(
    rank: int,
    msg_bytes: int,
    iters: int,
    triton_num_blocks: int,
    nccl_grid: int,
    triton_us: float,
    triton_bw: float,
    nccl_us: float,
    nccl_bw: float,
    ws_us: float | None = None,
    ws_bw: float | None = None,
) -> None:
    if rank != 0:
        return
    speedup = nccl_us / triton_us if triton_us > 0 else 0.0
    extra = ""
    if ws_us is not None and ws_bw is not None:
        ws_speedup = nccl_us / ws_us if ws_us > 0 else 0.0
        extra = f" | ws {ws_us:>10.2f} {ws_bw:>12.3f} {ws_speedup:>7.2f}x"
    print(
        f"{_fmt_size(msg_bytes):>10} | {iters:>6} | {triton_num_blocks:>6} {nccl_grid:>9} | "
        f"{triton_us:>10.2f} {triton_bw:>12.3f} | "
        f"{nccl_us:>10.2f} {nccl_bw:>12.3f} | {speedup:>7.2f}x{extra}"
    )


def _run_table(
    local_rank: int,
    peer_rank: int,
    group: dist.ProcessGroup,
    *,
    bidirectional: bool,
    cap: int,
) -> None:
    ws_only = os.environ.get("TRITON_NVL_BENCH_WS_ONLY") == "1"
    direction = (
        "Bidirectional WS sendrecv"
        if bidirectional and ws_only
        else "Bidirectional sendrecv"
        if bidirectional
        else "Unidirectional rank0->rank1 send/recv"
    )
    title = f"{direction} | NCCL_NCHANNELS=cap={cap} (Triton num_blocks tracks NCCL runtime grid)"
    _print_header(local_rank, title)
    for msg_bytes in _env_int_list("TRITON_NVL_BENCH_MSGS", _MSG_SIZES_BYTES):
        nccl_grid = _NCCL_GRID_TABLE[(msg_bytes, cap)]
        if bidirectional:
            triton_nb = _triton_num_blocks_bi(msg_bytes, cap)
        else:
            triton_nb = _triton_num_blocks_uni(msg_bytes, cap)

        ws_us: float | None = None
        ws_bw: float | None = None
        if bidirectional and ws_only and msg_bytes <= _WS_MAX_BYTES:
            triton_us, triton_bw = _bench_triton(
                msg_bytes,
                local_rank,
                peer_rank,
                group,
                bidirectional=True,
                num_blocks=nccl_grid,
                warp_specialized=True,
            )
        else:
            triton_us, triton_bw = _bench_triton(
                msg_bytes,
                local_rank,
                peer_rank,
                group,
                bidirectional=bidirectional,
                num_blocks=triton_nb,
            )
            dist.barrier(group)
            if bidirectional and msg_bytes <= _WS_MAX_BYTES:
                ws_us, ws_bw = _bench_triton(
                    msg_bytes,
                    local_rank,
                    peer_rank,
                    group,
                    bidirectional=True,
                    num_blocks=nccl_grid,
                    warp_specialized=True,
                )
        dist.barrier(group)
        nccl_us, nccl_bw = _bench_nccl(
            msg_bytes, local_rank, peer_rank, group, bidirectional=bidirectional
        )
        dist.barrier(group)
        display_nb = nccl_grid if bidirectional and ws_only else triton_nb
        _print_row(
            local_rank,
            msg_bytes,
            _iters_for_size(msg_bytes),
            display_nb,
            nccl_grid,
            triton_us,
            triton_bw,
            nccl_us,
            nccl_bw,
            ws_us,
            ws_bw,
        )


def _worker(local_rank: int, master_port: int, cap: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = "2"
    # Pin NCCL channel allocation to ``cap`` for the lifetime of this
    # process. Note: this only sets the allocation cap; NCCL's adaptive
    # in-kernel grid policy decides how many of those channels to
    # actually use per call (captured in ``_NCCL_GRID_TABLE``).
    os.environ["NCCL_MIN_NCHANNELS"] = str(cap)
    os.environ["NCCL_MAX_NCHANNELS"] = str(cap)

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    group = dist.group.WORLD
    assert group is not None
    peer_rank = 1 - local_rank

    if local_rank == 0:
        print()
        print("#" * 100)
        print(f"# NCCL_{{MIN,MAX}}_NCHANNELS = {cap}")
        print(
            "# Triton uni: num_blocks = NCCL runtime grid; "
            "Triton bi: num_blocks = max(NCCL runtime grid // 2, 1)."
        )
        print("# BW = unidirectional payload bytes / latency.")
        print(
            f"# WS config: sender_warps={_WS_SENDER_WARPS}, "
            f"receiver_warps={_WS_RECEIVER_WARPS}"
        )
        print("#" * 100)

    _run_table(local_rank, peer_rank, group, bidirectional=False, cap=cap)
    dist.barrier(group)
    _run_table(local_rank, peer_rank, group, bidirectional=True, cap=cap)

    dist.barrier(group)
    dist.destroy_process_group()


def main() -> None:
    print("Triton NVLink Copy-Based SendRecv vs NCCL batch_isend_irecv")
    print("Runtime-SM-matched 2-rank sweep via CUDA-graph replay.")
    print(
        "Triton num_blocks set per-cell to match NCCL adaptive runtime grid "
        f"(see _NCCL_GRID_TABLE); cap sweep: {_CAPS}"
    )

    for cap in _env_int_list("TRITON_NVL_BENCH_CAPS", _CAPS):
        os.environ["NCCL_MIN_NCHANNELS"] = str(cap)
        os.environ["NCCL_MAX_NCHANNELS"] = str(cap)
        port = _find_free_port()
        try:
            mp.spawn(_worker, args=(port, cap), nprocs=2, join=True)
        except Exception as e:
            print(f"benchmark failed at cap={cap}: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
