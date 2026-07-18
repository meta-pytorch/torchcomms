# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe
# Benchmark harness (not shipped code): exercises untyped cutlass/cute DSL kernels + dynamic
# torch.distributed symbols that pyre cannot model, so strict typing adds no value here.

"""Apple-to-apple all_to_all benchmark: CuTe-DSL copy kernel vs NCCL.

Runtime-SM matched (the framework grid ``world_size * num_blocks`` is set to NCCL's probed
active-SM count for each size, via ``_framework_num_blocks``), CUDA-graph timed, median of
windows -- so the CuTe copy kernel is compared to ``dist.all_to_all_single`` on the same SM
budget. Sweeps 32 B -> 2 GB per rank (2x step) plus 48 MB / 96 MB for mid-band resolution.

Both metrics are reported per size: **latency** (us/call) is the meaningful comparison at
small (latency-bound) sizes, **bus bandwidth** (GB/s/dir) at large (bandwidth-bound) sizes.
Every runnable size is correctness-gated against NCCL gold before timing. All tensors are
bf16 (the perf-relevant a2a dtype; see ``_bench_common._DTYPE``).

The SM match uses a probed NCCL active-SM grid table
(``_bench_common._NCCL_A2A_GRID_TABLE``, measured via ``tests/_probe_nccl_grid.py``). Only the
probed anchor sizes in that table are SM-matched; any other sweep size (below 64 KiB, and the
in-between 2x steps such as 128 KiB / 2 MB / 16 MB / 128 MB) falls back to the SM-budget
ceiling and is reported with ``nccl_grid=0`` -- indicative, not SM-matched.

Runs single-node (``mp.spawn``, default) for the local feedback loop, or one-process-per-rank
under ``torchrun`` / the conda launcher for GB300 2x4.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ._bench_common import (
    _bench_nccl,
    _bw_gbps,
    _capture_one_call,
    _DTYPE,
    _find_free_port,
    _fmt_size,
    _framework_num_blocks,
    _iters_for_size,
    _make_input,
    _max_rank_latency,
    _time_replays,
    emit_result_rows,
)

# 32 B -> 2 GB per rank, 2x stepping (27 sizes) plus 48 MB / 96 MB for mid-band resolution
# = 29 sizes.
_DEFAULT_SIZES: list[int] = [
    32,
    64,
    128,
    256,
    512,
    1024,
    2 * 1024,
    4 * 1024,
    8 * 1024,
    16 * 1024,
    32 * 1024,
    64 * 1024,
    128 * 1024,
    256 * 1024,
    512 * 1024,
    1 * 1024 * 1024,
    2 * 1024 * 1024,
    4 * 1024 * 1024,
    8 * 1024 * 1024,
    16 * 1024 * 1024,
    32 * 1024 * 1024,
    48 * 1024 * 1024,
    64 * 1024 * 1024,
    96 * 1024 * 1024,
    128 * 1024 * 1024,
    256 * 1024 * 1024,
    512 * 1024 * 1024,
    1024 * 1024 * 1024,
    2 * 1024 * 1024 * 1024,
]

# Below this per-rank size latency is the meaningful metric (latency-bound); at/above it
# bus bandwidth is (bandwidth-bound). Only affects which ratio the table highlights.
_LATENCY_BAND_BYTES: int = 1 * 1024 * 1024


def _get_sizes() -> list[int]:
    """Size ladder, overridable via A2A_SIZES=csv. Read lazily (not at import) per python.md."""
    env = os.environ.get("A2A_SIZES", "")
    if env:
        return [int(x) for x in env.split(",") if x]
    return _DEFAULT_SIZES


def _get_cap() -> int:
    """SM cap, overridable via A2A_CAPS. Read lazily (not at import) per python.md."""
    return int(os.environ.get("A2A_CAPS", "32"))


def _bench_cute(transport, msg_bytes, rank, ws, group, *, num_blocks):
    """CuTe copy a2a (latency_us, busbw) at this size. NCCL-gold correctness gate.

    The copy schedule uses the analytic adaptive config pinned to ``num_blocks`` (the
    SM-matched grid)."""
    from comms.dsl.cute import all_to_all
    from comms.dsl.cute.a2a.tuning import CuteA2AConfig

    device = torch.device("cuda", torch.cuda.current_device())
    elem = torch.tensor([], dtype=_DTYPE).element_size()
    numel = max(msg_bytes // elem, ws)
    numel -= numel % ws  # largest multiple of ws <= numel; always >= ws >= 1
    inp = _make_input(rank, numel, device)
    gold = torch.empty_like(inp)
    dist.all_to_all_single(gold, inp, group=group)
    out = torch.empty_like(inp)

    def fn():
        all_to_all(transport, out, inp, config=CuteA2AConfig(num_blocks=num_blocks))

    fn()
    torch.cuda.synchronize(device)
    # Decide correctness COLLECTIVELY before entering any follow-up collective. torch.equal is
    # a LOCAL check, so a kernel bug that corrupts only some ranks' output would have the
    # mismatching ranks raise here while the passing ranks march into _capture_one_call's
    # collectives -- a split that hangs the job. All-reduce the verdict so every rank raises
    # together and the caller's flag mechanism stays in sync.
    correct = bool(torch.equal(out, gold))
    verdict = torch.tensor([0 if correct else 1], device=device)
    dist.all_reduce(verdict, group=group)
    if verdict.item() > 0:
        raise AssertionError(
            f"cute a2a INCORRECT at msg_bytes={msg_bytes}, nb={num_blocks}"
        )
    g = _capture_one_call(fn)
    dist.barrier(group)
    iters = _iters_for_size(msg_bytes)
    lat = _max_rank_latency(_time_replays(g, iters, device, group), device, group)
    return lat, _bw_gbps(numel, ws, lat)


def _run(rank, ws, group, device) -> bool:  # noqa: C901
    from comms.dsl import nvl_rendezvous

    mbp = 32
    # Re-rendezvous one transport per size, sized to that size's chunk, so geometry stays
    # uniform across the sweep.
    rows: list[tuple[Any, ...]] = []
    ok = True
    for msg_bytes in _get_sizes():
        elem = torch.tensor([], dtype=_DTYPE).element_size()
        numel = max(msg_bytes // elem, ws)
        numel -= numel % ws  # largest multiple of ws <= numel; always >= ws >= 1
        chunk = numel // ws
        # _framework_num_blocks already returns min(nb, _max_num_blocks(ws, device, mbp)).
        nb, nccl_grid = _framework_num_blocks(msg_bytes, _get_cap(), ws, device, mbp)
        t = nvl_rendezvous(group, device, per_peer_bytes=chunk * elem)
        failed = False
        try:
            cute_lat, cute_bw = _bench_cute(
                t, msg_bytes, rank, ws, group, num_blocks=nb
            )
        except Exception as e:  # noqa: BLE001
            if rank == 0:
                msg = f"{type(e).__name__}: {e}"
                print(f"  # {_fmt_size(msg_bytes)} FAILED: {msg}", flush=True)
                emit_result_rows(
                    [
                        {
                            "backend": "cute_error",
                            "size_bytes": int(msg_bytes),
                            "error": msg[:900],
                        }
                    ]
                )
            ok = False
            cute_lat = cute_bw = None
            failed = True
        # Make the per-rank _bench_cute failure GLOBAL immediately after the raising call: a
        # mid-collective raise on one rank desyncs NCCL, so a surviving rank must NOT march
        # into the following barrier / _bench_nccl while a peer aborted. MAX-reduce the flag
        # and, on ANY rank's failure, stop the sweep on ALL ranks in lockstep.
        flag = torch.tensor([1.0 if failed else 0.0], device=device)
        dist.all_reduce(flag, group=group)
        if flag.item() > 0:
            ok = False
            break
        dist.barrier(group)
        nccl_lat, nccl_bw = _bench_nccl(msg_bytes, rank, ws, group)
        dist.barrier(group)
        if rank == 0:
            rows.append(
                (msg_bytes, nb, nccl_grid, cute_lat, nccl_lat, cute_bw, nccl_bw)
            )

    if rank == 0:
        _print_table(rows, ws)
        emit_result_rows(
            [
                {
                    "backend": "cute",
                    "variant": "copy",
                    "world_size": ws,
                    "size_bytes": int(msg_bytes),
                    "fw_ctas": int((nb or 0) * ws),
                    "nccl_grid": int(grid or 0),
                    "cute_us": float(cute_lat),
                    "nccl_us": float(nccl_lat or 0.0),
                    "cute_busbw_gbps": float(cute_bw),
                    "nccl_busbw_gbps": float(nccl_bw or 0.0),
                    "ratio": float(cute_bw / nccl_bw) if nccl_bw else 0.0,
                }
                for msg_bytes, nb, grid, cute_lat, nccl_lat, cute_bw, nccl_bw in rows
                if cute_bw is not None
            ]
        )
    return ok


def _print_table(rows, ws) -> None:
    """One row per size with both metrics; ``x`` is the size-appropriate ratio (latency for
    the latency-bound band, bus bandwidth above it), always >1 == CuTe faster."""
    lines = [
        "=== CuTe a2a (copy) vs NCCL -- SM-matched "
        "(x=latency<1MB / busbw>=1MB, >1=cute faster) ==="
    ]
    lines.append(
        f"{'size/rank':>10} {'fw_ctas':>7} {'nccl_grid':>9} {'cute_us':>9} "
        f"{'nccl_us':>9} {'cute_GB/s':>9} {'nccl_GB/s':>9} {'x':>6} {'metric':>6}"
    )
    for msg_bytes, nb, grid, cute_lat, nccl_lat, cute_bw, nccl_bw in rows:
        if cute_bw is None:
            lines.append(f"{_fmt_size(msg_bytes):>10} {'n/a':>7}")
            continue
        if msg_bytes < _LATENCY_BAND_BYTES:
            metric = "lat"
            ratio = (nccl_lat / cute_lat) if cute_lat else 0.0
        else:
            metric = "bw"
            ratio = (cute_bw / nccl_bw) if nccl_bw else 0.0
        ctas = (nb or 0) * ws
        flag = "" if ratio >= 1.0 else "  <NCCL"
        lines.append(
            f"{_fmt_size(msg_bytes):>10} {ctas:>7} {grid or 0:>9} {cute_lat:>9.1f} "
            f"{nccl_lat or 0.0:>9.1f} {cute_bw:>9.1f} {nccl_bw or 0.0:>9.1f} "
            f"{ratio:>5.2f}x {metric:>6}{flag}"
        )
    table = "\n".join(lines)
    print("\n" + table, flush=True)
    # Mirror to stderr: torchx keeps each worker's STDERR but truncates/drops STDOUT, so the
    # table is only reliably fetchable from the MAST job's stderr log.
    print("\n" + table, file=sys.stderr, flush=True)
    result_file = os.environ.get("A2A_RESULT_FILE")
    if result_file:
        try:
            with open(result_file, "w") as f:
                f.write(table + "\n")
        except OSError as e:
            print(f"(could not write A2A_RESULT_FILE: {e})", flush=True)
    _upload_result_to_manifold(table)


def _upload_result_to_manifold(table: str) -> None:
    """Best-effort upload of the result table to ``$A2A_RESULT_MANIFOLD`` (a
    ``manifold://bucket/tree/...`` path). No-op if the env is unset; never raises."""
    dest = os.environ.get("A2A_RESULT_MANIFOLD")
    if not dest:
        return
    # The MAST conda env has no `manifold` on PATH; the conda launcher installs the
    # `manifold.cli:prod` fbpkg at /packages/manifold.cli/manifold for exactly this.
    mf = "/packages/manifold.cli/manifold"
    if not os.path.exists(mf):
        mf = shutil.which("manifold") or "manifold"
    local = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(table + "\n")
            local = f.name
        out = subprocess.run(
            [mf, "put", "--overwrite", local, dest],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if out.returncode == 0:
            print(f"(uploaded result to {dest})", flush=True)
        else:
            print(
                f"(manifold put failed rc={out.returncode}: {out.stderr})", flush=True
            )
    except (OSError, subprocess.SubprocessError) as e:
        print(f"(manifold upload error: {e})", flush=True)
    finally:
        # delete=False above, so remove the temp file on every path (MAST would otherwise
        # leak a /tmp/tmpXXXX.txt per invocation).
        if local:
            try:
                os.remove(local)
            except OSError:
                pass


def _worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")
    ok = False
    try:
        ok = _run(rank, world_size, dist.group.WORLD, device)
        dist.barrier(dist.group.WORLD)
    finally:
        dist.destroy_process_group()  # always tear down, even if _run raises
    if not ok:
        sys.exit(1)


def _torchrun_main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}")
    ok = False
    try:
        ok = _run(rank, world_size, dist.group.WORLD, device)
        dist.barrier(dist.group.WORLD)
    finally:
        dist.destroy_process_group()  # always tear down, even if _run raises
    if not ok:
        sys.exit(1)


def main() -> None:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        _torchrun_main()
        return
    p = argparse.ArgumentParser()
    p.add_argument("--world-size", type=int, default=min(torch.cuda.device_count(), 8))
    args = p.parse_args()
    if args.world_size < 2:
        print("needs >=2 GPUs")
        return
    mp.spawn(
        _worker,
        args=(args.world_size, _find_free_port()),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
