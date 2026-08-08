# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe
# Benchmark harness (not shipped code): exercises untyped cutlass/cute DSL kernels + dynamic
# torch.distributed symbols that pyre cannot model, so strict typing adds no value here.

"""Apple-to-apple all_to_all benchmark: CuTe-DSL framework vs NCCL.

The CuTe counterpart of ``benchmark_a2a.py``. Same methodology -- runtime-SM
matched (the framework grid ``world_size * num_blocks`` is set to NCCL's probed
active-SM count), CUDA-graph timed, busbw/dir, median of windows -- so the CuTe
copy kernel is compared to ``dist.all_to_all_single`` on the same SM budget.
Reuses the backend-agnostic timing/bandwidth/NCCL helpers from ``benchmark_a2a``.

Runs single-node (``mp.spawn``, default) for the local feedback loop, or
one-process-per-rank under ``torchrun`` / the conda launcher for GB300 2x4.

The CuTe kernel currently requires ``chunk % 256 == 0`` (no tail handling), so
sub-tile sizes are reported as ``n/a (tail)`` until the masked-tail path lands;
every runnable size is correctness-gated against NCCL gold before timing.

The copy schedule's launch knobs come from the analytic adaptive default. Setting
``A2A_CUTE_USE_TUNED=1`` instead drives them from the tuned table: rank builds the
runtime key (``make_a2a_key``) and looks up its config (``get_a2a_config``,
falling back to the analytic default when no tuned entry exists), so a single job can
measure the tuned config against the analytic default at the same sizes -- the direct
tuned-vs-default A/B. The toggle scopes to the copy schedule only; the ``direct``/``ce``
zero-copy paths are unaffected.
"""

from __future__ import annotations

import os
import sys

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
    _max_num_blocks,
    _max_rank_latency,
    _time_replays,
    emit_result_rows,
)

# 32 B -> 2 GB per rank, 2x stepping (27 sizes) plus 48 MB / 96 MB for mid-band resolution
# where the Triton dip lives = 29 sizes, matched across all three benchmarks and the tuner.
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
    """CuTe a2a busbw at this size (None if unrunnable). NCCL-gold correctness gate.

    The copy schedule uses the analytic adaptive config pinned to ``num_blocks``.
    With ``A2A_CUTE_USE_TUNED=1`` it instead resolves the copy config from the tuned
    table (key built from the live input + transport; analytic default when no entry),
    measuring the tuned config head-to-head with the default. The toggle applies to the
    copy schedule only; the ``direct``/``ce`` zero-copy paths keep the analytic config."""
    from comms.dsl.cute import all_to_all, all_to_all_zc, CuteA2AConfig

    direct = os.environ.get("A2A_CUTE_DIRECT") == "1"
    ce = os.environ.get("A2A_CUTE_CE") == "1"
    device = torch.device("cuda", torch.cuda.current_device())
    elem = torch.tensor([], dtype=_DTYPE).element_size()
    numel = max(msg_bytes // elem, ws)
    numel -= numel % ws  # only need equal split; the kernel tiles any chunk now
    if numel == 0:
        return None, None
    inp = _make_input(rank, numel, device)
    gold = torch.empty_like(inp)
    dist.all_to_all_single(gold, inp, group=group)
    out = torch.empty_like(inp)

    if ce:
        # copy-engine zero-copy: cuMemcpyAsync moves data (zero SM), result read
        # from the transport's symm-mem buffer (returned).
        def fn():
            return all_to_all_zc(transport, inp, primitive="ce")

        out = fn()
    elif direct:
        # zero-copy: result is read from the transport's symm-mem buffer (returned).
        def fn():
            return all_to_all_zc(
                transport,
                inp,
                primitive="direct",
                config=CuteA2AConfig(num_blocks=num_blocks),
            )

        out = fn()
    elif os.environ.get("A2A_CUTE_USE_TUNED") == "1":
        from comms.dsl.cute.a2a.tuning import get_a2a_config, make_a2a_key

        key = make_a2a_key(inp, transport)
        cfg = get_a2a_config(key)

        def fn():
            all_to_all(transport, out, inp, config=cfg)

        fn()
    else:

        def fn():
            all_to_all(transport, out, inp, config=CuteA2AConfig(num_blocks=num_blocks))

        fn()
    torch.cuda.synchronize(device)
    # A2A_CUTE_SENDONLY is a diagnostic that skips the drain leg (output is
    # intentionally incomplete) to measure the SEND-leg NVLink ceiling.
    #
    # Decide correctness COLLECTIVELY before entering any follow-up collective. torch.equal is a
    # LOCAL check, so a kernel bug that corrupts only some ranks' output would have the mismatching
    # ranks raise here (their caller then jumps to its failure all_reduce) while the passing ranks
    # march on into _capture_one_call's collectives -- a split where the survivors block inside a
    # different collective than the aborted ranks, hanging the job instead of aborting cleanly. All-
    # reduce the verdict so every rank raises together and the caller's flag mechanism stays in sync.
    # (This does NOT cover a warmup fn() that itself raises on only some ranks -- a mid-collective,
    # already-desynced failure that belongs to NCCL async-error-handling / timeouts, not this flag.)
    correct = os.environ.get("A2A_CUTE_SENDONLY") == "1" or bool(torch.equal(out, gold))
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
    rows = []
    ok = True
    for msg_bytes in _get_sizes():
        elem = torch.tensor([], dtype=_DTYPE).element_size()
        numel = max(msg_bytes // elem, ws)
        numel -= numel % ws  # the kernel tiles any chunk (adaptive vec); no tail
        if numel == 0:
            if rank == 0:
                rows.append((msg_bytes, None, None, None, None))
            continue
        chunk = numel // ws
        nb, nccl_grid = _framework_num_blocks(msg_bytes, _get_cap(), ws, device, mbp)
        nb = min(nb, _max_num_blocks(ws, device, mbp))
        t = nvl_rendezvous(group, device, per_peer_bytes=chunk * elem)
        failed = False
        try:
            _, cute_bw = _bench_cute(t, msg_bytes, rank, ws, group, num_blocks=nb)
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
            cute_bw = None
            failed = True
        # Make the per-rank _bench_cute failure GLOBAL immediately after the raising call:
        # a mid-collective raise on one rank desyncs NCCL, so a surviving rank must NOT march
        # into the following dist.barrier / _bench_nccl while a peer aborted. MAX-reduce the
        # failure flag and, on ANY rank's failure, stop the sweep on ALL ranks in lockstep.
        flag = torch.tensor([1.0 if failed else 0.0], device=device)
        dist.all_reduce(flag, group=group)
        if flag.item() > 0:
            ok = False
            break
        dist.barrier(group)
        _, nccl_bw = _bench_nccl(msg_bytes, rank, ws, group)
        dist.barrier(group)
        if rank == 0:
            rows.append((msg_bytes, nb, nccl_grid, cute_bw, nccl_bw))

    if rank == 0:
        lines = ["=== CuTe a2a vs NCCL (busbw GB/s/dir, x = cute/nccl) ==="]
        lines.append(
            f"{'size/rank':>10} {'fw_ctas':>7} {'nccl_grid':>9} {'cute':>8} "
            f"{'nccl':>8} {'x':>6}"
        )
        for msg_bytes, nb, grid, cute_bw, nccl_bw in rows:
            if cute_bw is None:
                lines.append(f"{_fmt_size(msg_bytes):>10} {'n/a (tail)':>26}")
                continue
            ratio = cute_bw / nccl_bw if nccl_bw else 0.0
            ctas = (nb or 0) * ws
            flag = "" if ratio >= 1.0 else "  <NCCL"
            lines.append(
                f"{_fmt_size(msg_bytes):>10} {ctas:>7} {grid or 0:>9} "
                f"{cute_bw:>8.1f} {nccl_bw:>8.1f} {ratio:>5.2f}x{flag}"
            )
        table = "\n".join(lines)
        print("\n" + table, flush=True)
        # Also write to A2A_RESULT_FILE so the conda launcher can cat it back to the
        # task (agent) stdout that --logs fetches (worker stdout is redirected).
        result_file = os.environ.get("A2A_RESULT_FILE")
        if result_file:
            try:
                with open(result_file, "w") as f:
                    f.write(table + "\n")
            except OSError as e:
                print(f"(could not write A2A_RESULT_FILE: {e})", flush=True)
        # Durable retrieval on MAST via two readable sinks (best-effort, never fail the
        # bench): (1) Manifold text table (`manifold get`), and (2) tagged JSON rows on
        # stdout that the UniBench/AnyBench parser scrapes from the rank log into the
        # ``anybench_parser_output`` dataset (see anybench_a2a_cute_parser.py).
        _upload_result_to_manifold(table)
        emit_result_rows(
            [
                {
                    "backend": "cute",
                    "variant": "copy",
                    "world_size": ws,
                    "size_bytes": int(msg_bytes),
                    "fw_ctas": int((nb or 0) * ws),
                    "nccl_grid": int(grid or 0),
                    "cute_busbw_gbps": float(cute_bw),
                    "nccl_busbw_gbps": float(nccl_bw or 0.0),
                    "ratio": float(cute_bw / nccl_bw) if nccl_bw else 0.0,
                }
                for msg_bytes, nb, grid, cute_bw, nccl_bw in rows
                if cute_bw is not None
            ]
        )
    return ok


# Schedule variants the report sweeps, each selected by the one env knob the kernel
# reads at launch. ``copy`` is the default staging schedule (no knob); every other
# variant flips exactly one selector, so a single transport + size loop covers the
# whole matrix. ``tma`` is omitted: its multi-warp drain is still gated WIP.
_REPORT_VARIANTS: tuple[tuple[str, dict[str, str]], ...] = (
    ("copy", {}),
    ("direct", {"A2A_CUTE_DIRECT": "1"}),
    ("ce", {"A2A_CUTE_CE": "1"}),
)
# Env toggles cleared between variants so each report row measures exactly its labeled variant.
# A2A_CUTE_USE_TUNED is included: the report has no tuned variant (copy = analytic default), so
# an ambient A2A_CUTE_USE_TUNED=1 would otherwise make the "copy" row silently run the tuned
# config while still emitting variant="copy" -- corrupting the AnyBench copy-vs-nccl aggregation.
# Head-to-head tuned-vs-default measurement is a separate path (_bench_cute called directly).
_VARIANT_SELECTORS: tuple[str, ...] = (
    "A2A_CUTE_DIRECT",
    "A2A_CUTE_CE",
    "A2A_CUTE_TMA",
    "A2A_CUTE_USE_TUNED",
)


def _run_report(rank, ws, group, device) -> bool:  # noqa: C901
    """Every CuTe schedule variant vs NCCL across the size ladder (busbw + latency).

    The apple-to-apple twin of ``benchmark_a2a``'s perf report for the CuTe backend:
    one row per (size, variant) with cute/nccl busbw, ratio, and per-call latency, so
    a single GB300 job covers the whole variant x size matrix. Each variant is timed
    in isolation -- an unsupported or failing variant scores ``n/a`` and never aborts
    the sweep -- and the rows are emitted as tagged JSON on stdout for the AnyBench parser.
    """
    from comms.dsl import nvl_rendezvous

    mbp = 32
    rows = []
    ok = True
    for msg_bytes in _get_sizes():
        elem = torch.tensor([], dtype=_DTYPE).element_size()
        numel = max(msg_bytes // elem, ws)
        numel -= numel % ws
        if numel == 0:
            continue
        chunk = numel // ws
        nb, nccl_grid = _framework_num_blocks(msg_bytes, _get_cap(), ws, device, mbp)
        nb = min(nb, _max_num_blocks(ws, device, mbp))
        nccl_lat, nccl_bw = _bench_nccl(msg_bytes, rank, ws, group)
        dist.barrier(group)
        aborted = False
        for vname, env in _REPORT_VARIANTS:
            for sel in _VARIANT_SELECTORS:
                os.environ.pop(sel, None)
            os.environ.update(env)
            # Re-rendezvous a fresh transport per variant so a partially-advanced failed
            # variant cannot corrupt transport state for the next variant of this size.
            t = nvl_rendezvous(group, device, per_peer_bytes=chunk * elem)
            failed = False
            try:
                lat, bw = _bench_cute(t, msg_bytes, rank, ws, group, num_blocks=nb)
            except Exception as e:  # noqa: BLE001 -- recorded per-rank, then made global
                if rank == 0:
                    msg = f"{type(e).__name__}: {e}"
                    print(
                        f"  # {_fmt_size(msg_bytes)}/{vname} FAILED: {msg}",
                        flush=True,
                    )
                    # Emit a cute_error row like _run does, so the AnyBench parser's n_errors
                    # reflects report-path failures too (it counts backend == "cute_error").
                    emit_result_rows(
                        [
                            {
                                "backend": "cute_error",
                                "size_bytes": int(msg_bytes),
                                "variant": vname,
                                "error": msg[:900],
                            }
                        ]
                    )
                lat, bw = None, None
                ok = False
                failed = True
            for sel in _VARIANT_SELECTORS:
                os.environ.pop(sel, None)
            # Make the per-rank _bench_cute failure GLOBAL right after the raising call: a
            # mid-collective raise on one rank desyncs the communicator, so a surviving rank
            # must not continue the inner loop (its next variant's rendezvous/collectives
            # would mismatch the aborted peer). MAX-reduce the flag BEFORE the barrier; on
            # ANY rank's failure stop both the variant loop and the size loop in lockstep.
            flag = torch.tensor([1.0 if failed else 0.0], device=device)
            dist.all_reduce(flag, group=group)
            if flag.item() > 0:
                ok = False
                aborted = True
                break
            dist.barrier(group)
            if rank == 0:
                rows.append(
                    (msg_bytes, vname, nb, nccl_grid, bw, lat, nccl_bw, nccl_lat)
                )
        if aborted:
            break

    if rank == 0:
        _print_report(rows, ws)
        emit_result_rows(
            [
                {
                    "backend": "cute",
                    "variant": vname,
                    "world_size": ws,
                    "size_bytes": int(msg_bytes),
                    "fw_ctas": int((nb or 0) * ws),
                    "nccl_grid": int(grid or 0),
                    "cute_busbw_gbps": float(bw),
                    "nccl_busbw_gbps": float(nccl_bw or 0.0),
                    "ratio": float(bw / nccl_bw) if nccl_bw else 0.0,
                    "cute_us": float(lat or 0.0),
                    "nccl_us": float(nccl_lat or 0.0),
                }
                for msg_bytes, vname, nb, grid, bw, lat, nccl_bw, nccl_lat in rows
                if bw is not None
            ]
        )
    return ok


def _print_report(rows, ws) -> None:
    """Long-format matrix (one row per size+variant) with all three metrics."""
    lines = ["=== CuTe a2a variants vs NCCL (busbw GB/s/dir, lat us, x=cute/nccl) ==="]
    lines.append(
        f"{'size/rank':>10} {'variant':>7} {'fw_ctas':>7} {'cute_bw':>8} "
        f"{'nccl_bw':>8} {'x':>6} {'cute_us':>9} {'nccl_us':>9}"
    )
    for msg_bytes, vname, nb, _grid, bw, lat, nccl_bw, nccl_lat in rows:
        if bw is None:
            lines.append(f"{_fmt_size(msg_bytes):>10} {vname:>7} {'n/a':>7}")
            continue
        ratio = bw / nccl_bw if nccl_bw else 0.0
        ctas = (nb or 0) * ws
        lines.append(
            f"{_fmt_size(msg_bytes):>10} {vname:>7} {ctas:>7} {bw:>8.1f} "
            f"{nccl_bw:>8.1f} {ratio:>5.2f}x {lat or 0.0:>9.1f} {nccl_lat or 0.0:>9.1f}"
        )
    table = "\n".join(lines)
    print("\n" + table, flush=True)
    # Mirror to stderr: torchx keeps each worker's STDERR but truncates/drops STDOUT,
    # so the report is only reliably fetchable from the MAST job's stderr log.
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
    import shutil
    import subprocess
    import tempfile

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
    runner = _run_report if os.environ.get("A2A_CUTE_REPORT") == "1" else _run
    ok = runner(rank, world_size, dist.group.WORLD, device)
    dist.barrier(dist.group.WORLD)
    dist.destroy_process_group()
    if not ok:
        sys.exit(1)


def _torchrun_main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}")
    runner = _run_report if os.environ.get("A2A_CUTE_REPORT") == "1" else _run
    ok = runner(rank, world_size, dist.group.WORLD, device)
    dist.barrier(dist.group.WORLD)
    dist.destroy_process_group()
    if not ok:
        sys.exit(1)


def main() -> None:
    import argparse

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
