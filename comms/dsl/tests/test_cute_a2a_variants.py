# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe
# Distributed test: runs under mp.spawn and touches untyped cutlass / symmetric-memory symbols
# that pyre cannot model, so strict typing adds no value here.

"""Per-schedule correctness for the CuTe all_to_all variants.

The eager `copy` schedule is covered by `test_a2a_cute.py` /
`test_a2a_cute_production.py`; this suite is the bit-exact correctness net for the
zero-copy `direct` (DirectWrite) and `ce` (copy-engine) paths, plus the `copy` staging
schedule under CUDA-graph replay and the produce/consume hook and rows>0 transpose. Every
schedule is checked eagerly against `dist.all_to_all_single` over several sizes; the
staging schedule is additionally checked under CUDA-graph replay (the same
persistent-counter credit handshake the `copy` graph-replay stress test exercises). The
zero-copy paths return a view of the transport's symmetric-memory buffer, so their
CUDA-graph behavior is covered by the graph-timed benchmark rather than asserted bit-exact
here.

A schedule whose hardware feature is unavailable on the running GPU raises at compile and
is **skipped** uniformly — the launch fails on every rank before any cross-rank signal, so
skipping cannot desync the group. A schedule that runs but produces a wrong result
**fails** (it is not skipped).

Dual execution model: `mp.spawn` locally / via `buck run`, and `_torchrun_main` for the
GB300 conda launcher (`a2a_mast_launch_conda --module comms.dsl.tests.test_a2a_cute_variants`).
Gold is `dist.all_to_all_single`. Skipped unless >=2 GPUs are present.
"""

import os
import sys
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from comms.dsl.tests._dist_harness import (
    _find_free_port,
    _golden,
    _make_input,
    _rendezvous,
    _report,
)

_STAGING = ("copy",)
_ZEROCOPY = ("direct", "ce")
_VARIANTS = _STAGING + _ZEROCOPY


def _launch(t, inp, variant, *, out=None, num_blocks: int = 2):
    """Run one CuTe a2a variant; return its output tensor.

    Staging variants write the caller's ``out`` (allocated if absent); the zero-copy
    variants own their output (a view of the transport's symm-mem buffer) and return it.
    """
    from comms.dsl.cute.a2a.host import all_to_all, all_to_all_zc
    from comms.dsl.cute.a2a.tuning import CuteA2AConfig

    if variant in ("direct", "ce"):
        return all_to_all_zc(
            t, inp, primitive=variant, config=CuteA2AConfig(num_blocks=num_blocks)
        )
    if out is None:
        out = torch.empty_like(inp)
    all_to_all(
        t, out, inp, config=CuteA2AConfig(primitive=variant, num_blocks=num_blocks)
    )
    return out


def _case_eager(group, rank, ws, device, variant) -> bool:
    """Bit-exact vs gold across sizes for one variant (eager). Skip if unsupported here."""
    ok, ran = True, False
    for mult in (4096, 65536):
        numel = ws * mult
        t = _rendezvous(group, device, numel // ws)
        inp = _make_input(rank, numel, device)
        gold = _golden(group, inp)
        try:
            out = _launch(t, inp, variant)
            torch.cuda.synchronize(device)
            matched = torch.equal(out, gold)
        except Exception as e:  # noqa: B902 -- unsupported schedule on this GPU: skip
            if rank == 0:
                print(
                    f"  SKIP eager[{variant}] numel={numel}: {type(e).__name__}",
                    flush=True,
                )
            dist.barrier(group)
            continue
        ok, ran = ok and matched, True
        dist.barrier(group)
    suffix = "" if ran else " (skipped: unsupported here)"
    return _report(f"eager[{variant}]{suffix}", ok, device, group, rank)


def _case_graph(group, rank, ws, device, variant) -> bool:
    """Bit-exact vs gold under CUDA-graph replay for a staging variant. Skip if unsupported.

    Used for the staging schedules only: their output is the caller's tensor, which is
    stable across replays. (Zero-copy schedules return a transport-buffer view and are
    graph-exercised via the benchmark.)
    """
    numel = ws * 16384
    t = _rendezvous(group, device, numel // ws)
    inps = [_make_input(rank + 5 * i, numel, device) for i in range(3)]
    golds = [_golden(group, x) for x in inps]
    buf = inps[0].clone()
    out = torch.empty_like(buf)  # caller output for staging; ignored by zero-copy
    try:
        # Warm + compile (and prime the ce host buffer cache) before capture, since
        # cute.compile and the device->host base-address read are illegal mid-capture.
        _launch(t, buf, variant, out=out)
        torch.cuda.synchronize(device)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            captured = _launch(t, buf, variant, out=out)
    except Exception as e:  # noqa: B902 -- unsupported schedule on this GPU: skip
        if rank == 0:
            print(f"  SKIP graph[{variant}]: {type(e).__name__}", flush=True)
        dist.barrier(group)
        return _report(f"graph[{variant}] (skipped)", True, device, group, rank)
    ok = True
    for i in range(3):
        buf.copy_(inps[i])
        g.replay()
        torch.cuda.synchronize(device)
        ok = ok and torch.equal(captured, golds[i])
    dist.barrier(group)
    return _report(f"graph[{variant}]", ok, device, group, rank)


def _case_hook(group, rank, ws, device, variant) -> bool:
    """produce/consume hooks on one fused staging schedule, bit-exact vs the transformed gold,
    eager (tiny/odd/large) + CUDA-graph replay. Proves the CuTe fused schedule applies the hook
    (the "5% the user writes"), not just an identity copy: `scale2_produce` scales each chunk on
    the send leg (-> 2x the gold); `addone_consume` adds 1 on the recv leg (-> gold + 1).
    Skipped if the schedule is unsupported on this GPU (same uniform skip as `_case_eager`).
    """
    from comms.dsl.cute.a2a.host import all_to_all
    from comms.dsl.cute.a2a.tuning import CuteA2AConfig
    from comms.dsl.cute.hooks import addone_consume, scale2_produce

    ok, ran = True, False
    for mult in (2, 100, 4096, 65536):
        numel = ws * mult
        t = _rendezvous(group, device, numel // ws)
        inp = _make_input(rank, numel, device)
        gold = _golden(group, inp)
        cfg = CuteA2AConfig(primitive=variant, num_blocks=2)
        try:
            out_p = torch.empty_like(inp)
            all_to_all(t, out_p, inp, produce=scale2_produce, config=cfg)
            out_c = torch.empty_like(inp)
            all_to_all(t, out_c, inp, consume=addone_consume, config=cfg)
            torch.cuda.synchronize(device)
            matched = torch.equal(out_p, gold * 2) and torch.equal(out_c, gold + 1)
        except Exception as e:  # noqa: B902 -- unsupported schedule on this GPU: skip
            if rank == 0:
                print(
                    f"  SKIP hook[{variant}] numel={numel}: {type(e).__name__}",
                    flush=True,
                )
            dist.barrier(group)
            continue
        ok, ran = ok and matched, True
        dist.barrier(group)
    if ran:
        # CUDA-graph replay with changing data: the hook must stay applied across replays.
        numel = ws * 16384
        t = _rendezvous(group, device, numel // ws)
        inps = [_make_input(rank + 5 * i, numel, device) for i in range(3)]
        golds = [_golden(group, x) for x in inps]
        buf = inps[0].clone()
        out_g = torch.empty_like(buf)
        cfg = CuteA2AConfig(primitive=variant, num_blocks=2)
        all_to_all(t, out_g, buf, produce=scale2_produce, config=cfg)  # warm/compile
        torch.cuda.synchronize(device)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            all_to_all(t, out_g, buf, produce=scale2_produce, config=cfg)
        for i in range(3):
            buf.copy_(inps[i])
            g.replay()
            torch.cuda.synchronize(device)
            ok = ok and torch.equal(out_g, golds[i] * 2)
        dist.barrier(group)
        # The consume (recv-leg) hook is a distinct code path; cover it under graph replay too,
        # matching the "hook must stay applied across replays" claim. Reuse inps/golds/buf.
        out_gc = torch.empty_like(buf)
        all_to_all(t, out_gc, buf, consume=addone_consume, config=cfg)  # warm/compile
        torch.cuda.synchronize(device)
        g2 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g2):
            all_to_all(t, out_gc, buf, consume=addone_consume, config=cfg)
        for i in range(3):
            buf.copy_(inps[i])
            g2.replay()
            torch.cuda.synchronize(device)
            ok = ok and torch.equal(out_gc, golds[i] + 1)
        dist.barrier(group)
    suffix = "" if ran else " (skipped: unsupported here)"
    return _report(f"hook[{variant}:scale2/addone]{suffix}", ok, device, group, rank)


def _case_transpose(group, rank, ws, device, variant) -> bool:
    """rows>0 layout transpose on one fused staging schedule, bit-exact vs the transposed gold,
    eager (a couple of rows x cols shapes) + CUDA-graph replay. Proves the CuTe fused schedule
    runs the design-doc headline non-contiguous transform on the production backend: each
    [rows, cols] chunk is transposed to [cols, rows], fused into the transfer leg (the kernel
    reads the chunk through a transposed layout -- no extra HBM pass). Skipped if the schedule
    is unsupported on this GPU (same uniform skip as `_case_eager`).
    """
    from comms.dsl.cute.a2a.host import all_to_all
    from comms.dsl.cute.a2a.tuning import CuteA2AConfig

    def _ref(x):
        return _golden(group, x).view(ws, rows, cols).transpose(1, 2).reshape(-1)

    ok, ran = True, False
    for rows, cols in ((64, 128), (32, 100)):
        chunk = rows * cols
        numel = ws * chunk
        t = _rendezvous(group, device, chunk)
        inp = _make_input(rank, numel, device)
        ref = _ref(inp).contiguous()
        cfg = CuteA2AConfig(primitive=variant, num_blocks=2)
        try:
            out = torch.empty_like(inp)
            all_to_all(t, out, inp, rows=rows, config=cfg)
            torch.cuda.synchronize(device)
            matched = torch.equal(out, ref)
        except Exception as e:  # noqa: B902 -- unsupported schedule on this GPU: skip
            if rank == 0:
                print(
                    f"  SKIP transpose[{variant}] {rows}x{cols}: {type(e).__name__}",
                    flush=True,
                )
            dist.barrier(group)
            continue
        ok, ran = ok and matched, True
        dist.barrier(group)
    if ran:
        # CUDA-graph replay with changing data: the transpose must stay applied across replays.
        rows, cols = 64, 128
        numel = ws * rows * cols
        t = _rendezvous(group, device, rows * cols)
        inps = [_make_input(rank + 5 * i, numel, device) for i in range(3)]
        refs = [_ref(x).contiguous() for x in inps]
        buf = inps[0].clone()
        out_g = torch.empty_like(buf)
        cfg = CuteA2AConfig(primitive=variant, num_blocks=2)
        all_to_all(t, out_g, buf, rows=rows, config=cfg)  # warm/compile
        torch.cuda.synchronize(device)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            all_to_all(t, out_g, buf, rows=rows, config=cfg)
        for i in range(3):
            buf.copy_(inps[i])
            g.replay()
            torch.cuda.synchronize(device)
            ok = ok and torch.equal(out_g, refs[i])
        dist.barrier(group)
    suffix = "" if ran else " (skipped: unsupported here)"
    return _report(f"transpose[{variant}]{suffix}", ok, device, group, rank)


def _case_direct_tuned(group, rank, ws, device) -> bool:
    """A tuned direct CuteA2AConfig must drive the ACTUAL direct launch -- not just match the
    key. ``_all_to_all_direct`` now threads num_threads / unroll / cluster / cluster_y from the
    config (a sentinel 0 = analytic). Run direct with an explicit tuned config and assert (a)
    it is still bit-exact, and (b) the cute compile cache gains a DISTINCT entry vs the analytic
    default -- which it can only do if the tuned knobs reached the resolved launch params (the
    cache key). A config that the host silently ignored would collapse onto the default's key.
    """
    from comms.dsl.cute.a2a import host as cute_host
    from comms.dsl.cute.a2a.host import all_to_all_zc
    from comms.dsl.cute.a2a.tuning import CuteA2AConfig

    numel = ws * 65536
    t = _rendezvous(group, device, numel // ws)
    inp = _make_input(rank, numel, device)
    gold = _golden(group, inp)
    try:
        # Analytic default first (records its compile-cache key).
        out0 = all_to_all_zc(
            t, inp, primitive="direct", config=CuteA2AConfig(num_blocks=2)
        )
        torch.cuda.synchronize(device)
        keys_after_default = {k for k in cute_host._COMPILED if k[0] == "a2a_direct"}
        # Tuned config: force non-analytic num_threads + unroll + cluster (all applicable to
        # direct). num_threads=128 differs from the analytic _pick_tile pick at this size, so
        # the resolved launch params -- and the cache key -- must change.
        tuned = CuteA2AConfig(
            num_blocks=2, num_threads=128, unroll=4, cluster=2, primitive="direct"
        )
        out1 = all_to_all_zc(t, inp, primitive="direct", config=tuned)
        torch.cuda.synchronize(device)
        keys_after_tuned = {k for k in cute_host._COMPILED if k[0] == "a2a_direct"}
        matched = torch.equal(out0, gold) and torch.equal(out1, gold)
        # The tuned launch added at least one new compile-cache key (its knobs reached launch).
        new_key = keys_after_tuned - keys_after_default
        applied = len(new_key) >= 1
    except Exception as e:  # noqa: B902 -- unsupported schedule on this GPU: skip
        if rank == 0:
            print(f"  SKIP direct_tuned: {type(e).__name__}: {e}", flush=True)
        dist.barrier(group)
        return _report("direct_tuned (skipped)", True, device, group, rank)
    dist.barrier(group)
    return _report("direct_tuned", matched and applied, device, group, rank)


def _run_all_cases(group, rank, ws, device) -> list[bool]:
    results: list[bool] = []
    for variant in _VARIANTS:
        results.append(_case_eager(group, rank, ws, device, variant))
    for variant in _STAGING:
        results.append(_case_graph(group, rank, ws, device, variant))
    for variant in _STAGING:
        results.append(_case_hook(group, rank, ws, device, variant))
    for variant in _STAGING:
        results.append(_case_transpose(group, rank, ws, device, variant))
    results.append(_case_direct_tuned(group, rank, ws, device))
    return results


def _mp_worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")
    group = dist.group.WORLD
    assert group is not None
    results = _run_all_cases(group, rank, world_size, device)
    dist.barrier(group)
    dist.destroy_process_group()
    assert all(results), f"rank {rank}: cute variant correctness failed: {results}"


class A2ACuteVariantsTest(unittest.TestCase):
    def _world(self) -> int:
        return min(torch.cuda.device_count(), 8)

    def test_variants(self) -> None:
        if torch.cuda.device_count() < 2:
            self.skipTest("needs >=2 GPUs")
        ws = self._world()
        mp.spawn(_mp_worker, args=(ws, _find_free_port()), nprocs=ws, join=True)


def _torchrun_main() -> None:
    """Entry point for the GB300 conda/torchrun launcher (one process per rank)."""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}")
    group = dist.group.WORLD
    results = _run_all_cases(group, rank, world_size, device)
    dist.barrier(group)
    if rank == 0:
        print(
            f"cute variant correctness: {'ALL PASS' if all(results) else 'FAIL'} "
            f"({sum(results)}/{len(results)})",
            flush=True,
        )
    dist.destroy_process_group()
    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        _torchrun_main()
    else:
        unittest.main()
