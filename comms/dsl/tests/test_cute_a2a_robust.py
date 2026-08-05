# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe
# Distributed test: runs under mp.spawn and touches untyped cutlass / symmetric-memory symbols
# that pyre cannot model, so strict typing adds no value here.

"""Race / corruption robustness tests for the CuTe-DSL all_to_all.

The CuTe twin of the robustness half of ``test_a2a_robust.py`` (Triton). The CuTe kernel
drives the transport's signal pad with the same per-(peer, block) TAIL protocol, so the same
robustness bar applies:

* **Concurrent transports** -- two independent transports interleaved on the same ranks must
  not cross-talk; each keeps its own staging buffer + signal pad + step counters.
* **Sustained graph replay** -- CUDA-graph replay must stay bit-exact under sustained reuse
  with changing data (the persistent monotonic counters must never read a stale slot).
* **Geometry guard** -- switching a geometry field (``num_blocks``) on a reused transport
  must FIRE the runtime guard, exercised here through the zero-copy ``all_to_all_zc`` direct
  entry (the CuTe twin of ``test_a2a_robust.py``'s ``zc_geometry_switch_guard``).

(The hang watchdog is exercised in ``test_a2a_watchdog``, which covers both backends through
the DSL-agnostic ``watch`` API.)

Dual execution model so the same suite runs locally and on MAST GB300:

* Locally / ``buck test``: the unittest entry point ``mp.spawn`` the ranks.
* On GB300 2x4 via the conda launcher (``a2a_mast_launch_conda --module
  comms.dsl.tests.test_a2a_cute_production``): ``torchrun`` runs one process per rank across
  the two hosts and ``_torchrun_main`` drives the cases over the launcher-provided group.

Gold is ``dist.all_to_all_single``. Skipped unless >=2 GPUs are present.
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


# --------------------------------------------------------------------------- #
# Race / corruption + observability cases (non-destructive; run together)
# --------------------------------------------------------------------------- #


def _case_concurrent_transports(group, rank, ws, device) -> bool:
    """Two independent CuTe transports interleaved on the same ranks must not
    cross-talk -- each keeps its own staging buffer + signal pad + step counters,
    so interleaving their collectives stays bit-exact."""
    from comms.dsl.cute.a2a.host import all_to_all
    from comms.dsl.cute.a2a.tuning import CuteA2AConfig

    numel = ws * 8192
    chunk = numel // ws
    t1 = _rendezvous(group, device, chunk)
    t2 = _rendezvous(group, device, chunk)
    inp = _make_input(rank, numel, device)
    gold = _golden(group, inp)
    o1 = torch.empty_like(inp)
    o2 = torch.empty_like(inp)
    all_to_all(t1, o1, inp, config=CuteA2AConfig(num_blocks=2))
    all_to_all(t2, o2, inp, config=CuteA2AConfig(num_blocks=2))
    all_to_all(t1, o1, inp, config=CuteA2AConfig(num_blocks=2))
    all_to_all(t2, o2, inp, config=CuteA2AConfig(num_blocks=2))
    torch.cuda.synchronize(device)
    ok = torch.equal(o1, gold) and torch.equal(o2, gold)
    dist.barrier(group)
    return _report("concurrent_transports", ok, device, group, rank)


def _case_graph_replay_stress(group, rank, ws, device) -> bool:
    """Sustained CUDA-graph replay with changing data must stay bit-exact -- the
    persistent monotonic step counters must never read a stale signal slot across
    many replays (the production inference pattern)."""
    from comms.dsl.cute.a2a.host import all_to_all
    from comms.dsl.cute.a2a.tuning import CuteA2AConfig

    numel = ws * 8192
    chunk = numel // ws
    n_iters = 20
    t = _rendezvous(group, device, chunk)
    inps = [_make_input(rank + 7 * i, numel, device) for i in range(n_iters)]
    golds = [_golden(group, x) for x in inps]
    buf = inps[0].clone()
    out = torch.empty_like(buf)

    def fn() -> None:
        all_to_all(t, out, buf, config=CuteA2AConfig(num_blocks=2))

    fn()  # warm/compile + prime counters before capture
    torch.cuda.synchronize(device)
    dist.barrier(group)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    ok = True
    for i in range(n_iters):
        buf.copy_(inps[i])
        out.zero_()
        g.replay()
        torch.cuda.synchronize(device)
        ok = ok and torch.equal(out, golds[i])
    dist.barrier(group)
    return _report("graph_replay_stress", ok, device, group, rank)


def _case_zc_geometry_switch_guard(group, rank, ws, device) -> bool:
    """The geometry guard must FIRE on the CuTe zero-copy entry point (the CuTe twin of
    ``test_a2a_robust.py``'s ``zc_geometry_switch_guard``).

    The zero-copy direct path drives the transport's persistent step counters just like the
    staging path, so switching ``num_blocks`` (one of ``CUTE_A2A_GEOMETRY_FIELDS``) on a
    reused transport is the documented hazard the runtime guard catches -- this exercises the
    guard in the CuTe ``all_to_all_zc`` so the firing test cannot be silently deleted. The
    first call primes the geometry (num_blocks=2) and must SUCCEED; the second differs ONLY in
    ``num_blocks`` (2 -> 4, same numel) so the guard -- not the zero-copy sizing assert -- is
    what fires. Under ``COMMS_DSL_STRICT_GEOMETRY=1`` the guard raises ``ValueError`` before
    dispatch; the env is saved/restored so strict mode doesn't leak into other cases. The
    transport is sized ``per_peer_bytes == chunk * elem`` (the zero-copy contract), and both
    grids (ws*2, ws*4) stay within the SM budget for ws<=8.
    """
    from comms.dsl.cute.a2a.host import all_to_all_zc
    from comms.dsl.cute.a2a.tuning import CuteA2AConfig

    numel = ws * 4096
    chunk = numel // ws
    t = _rendezvous(group, device, chunk)
    inp = _make_input(rank, numel, device)

    # Prime the geometry: this first zero-copy call must succeed (num_blocks=2).
    out = all_to_all_zc(t, inp, primitive="direct", config=CuteA2AConfig(num_blocks=2))
    torch.cuda.synchronize(device)
    gold = _golden(group, inp)
    primed_ok = torch.equal(out, gold)

    # Switch geometry (num_blocks 2 -> 4) on the SAME transport under strict mode: the guard
    # must raise ValueError before dispatch.
    prev = os.environ.get("COMMS_DSL_STRICT_GEOMETRY")
    raised = False
    try:
        os.environ["COMMS_DSL_STRICT_GEOMETRY"] = "1"
        try:
            all_to_all_zc(
                t, inp, primitive="direct", config=CuteA2AConfig(num_blocks=4)
            )
        except ValueError:
            raised = True
    finally:
        if prev is None:
            os.environ.pop("COMMS_DSL_STRICT_GEOMETRY", None)
        else:
            os.environ["COMMS_DSL_STRICT_GEOMETRY"] = prev

    dist.barrier(group)
    return _report(
        "zc_geometry_switch_guard", primed_ok and raised, device, group, rank
    )


def _run_clean_cases(group, rank, ws, device) -> list[bool]:
    return [
        _case_concurrent_transports(group, rank, ws, device),
        _case_graph_replay_stress(group, rank, ws, device),
        _case_zc_geometry_switch_guard(group, rank, ws, device),
    ]


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #


def _mp_worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")
    group = dist.group.WORLD
    assert group is not None
    results = _run_clean_cases(group, rank, world_size, device)
    dist.barrier(group)
    dist.destroy_process_group()
    assert all(results), f"rank {rank}: cute production-hardening failed: {results}"


class A2ACuteProductionTest(unittest.TestCase):
    def _world(self) -> int:
        return min(torch.cuda.device_count(), 8)

    def test_production_robustness(self) -> None:
        if torch.cuda.device_count() < 2:
            self.skipTest("needs >=2 GPUs")
        ws = self._world()
        mp.spawn(_mp_worker, args=(ws, _find_free_port()), nprocs=ws, join=True)


def _torchrun_main() -> None:
    """Entry point for the GB300 conda/torchrun launcher (one process per rank).

    Runs the clean, non-destructive cases (concurrent transports, graph-replay stress, zc
    geometry-switch guard) over the launcher-provided process group.
    """
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{local_rank}")
    group = dist.group.WORLD
    results = _run_clean_cases(group, rank, world_size, device)
    dist.barrier(group)
    if rank == 0:
        print(
            f"cute production-hardening: {'ALL PASS' if all(results) else 'FAIL'} "
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
