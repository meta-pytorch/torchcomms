# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe
# Distributed test: runs under mp.spawn and touches untyped cutlass / symmetric-memory symbols
# that pyre cannot model, so strict typing adds no value here.

"""Robustness tests for the CuTe-DSL all_to_all copy schedule.

* **Concurrent transports** -- two independent transports interleaved on the same ranks
  must not cross-talk; each keeps its own staging buffer + signal pad + step counters.
* **Geometry guard** -- switching a geometry field (``num_blocks``) on a reused transport
  must FIRE the runtime guard (``check_geometry``) before dispatch.

HEAD slot-free credits protect fixed-geometry reused transports without host barriers.
Gold is ``dist.all_to_all_single``. Skipped unless >=2 GPUs are present.
"""

import os
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


def _case_concurrent_transports(group, rank, ws, device) -> bool:
    """Two independent CuTe transports interleaved on the same ranks must not cross-talk --
    each keeps its own staging buffer + signal pad + step counters, so interleaving their
    collectives stays bit-exact."""
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
    cfg = CuteA2AConfig(num_blocks=2)
    all_to_all(t1, o1, inp, config=cfg)
    all_to_all(t2, o2, inp, config=cfg)
    torch.cuda.synchronize(device)
    all_to_all(t1, o1, inp, config=cfg)
    all_to_all(t2, o2, inp, config=cfg)
    torch.cuda.synchronize(device)
    ok = torch.equal(o1, gold) and torch.equal(o2, gold)
    dist.barrier(group)
    return _report("concurrent_transports", ok, device, group, rank)


def _case_geometry_switch_guard(group, rank, ws, device) -> bool:
    """The geometry guard must FIRE when a geometry field (``num_blocks``) changes on a
    reused transport. The first call primes the geometry (num_blocks=2) and must SUCCEED;
    the second differs ONLY in ``num_blocks`` (2 -> 4, same numel) so ``check_geometry`` --
    a hard error by default (no runtime drain) -- raises ValueError before dispatch."""
    from comms.dsl.cute.a2a.host import all_to_all
    from comms.dsl.cute.a2a.tuning import CuteA2AConfig

    numel = ws * 4096
    chunk = numel // ws
    t = _rendezvous(group, device, chunk)
    inp = _make_input(rank, numel, device)
    out = torch.empty_like(inp)

    all_to_all(t, out, inp, config=CuteA2AConfig(num_blocks=2))
    torch.cuda.synchronize(device)
    primed_ok = torch.equal(out, _golden(group, inp))

    raised = False
    try:
        all_to_all(t, out, inp, config=CuteA2AConfig(num_blocks=4))
    except ValueError:
        raised = True

    dist.barrier(group)
    return _report("geometry_switch_guard", primed_ok and raised, device, group, rank)


def _case_cluster(group, rank, ws, device) -> bool:
    """The CGA cluster launch path (untested elsewhere): forcing ``cluster>1`` must stay
    bit-exact, and a cluster that does not divide ``num_blocks`` must collapse to 1 (the
    host guard) rather than trip CUDA_ERROR_INVALID_CLUSTER_SIZE."""
    from comms.dsl.cute.a2a.host import all_to_all
    from comms.dsl.cute.a2a.tuning import CuteA2AConfig

    numel = ws * 8192
    chunk = numel // ws
    inp = _make_input(rank, numel, device)
    gold = _golden(group, inp)

    # CGA on: num_blocks (4) divisible by cluster (2) -> cluster stays 2.
    t1 = _rendezvous(group, device, chunk)
    o1 = torch.empty_like(inp)
    all_to_all(t1, o1, inp, config=CuteA2AConfig(num_blocks=4, cluster=2))
    torch.cuda.synchronize(device)
    dist.barrier(group)
    # Collapse: num_blocks (3) NOT divisible by cluster (2) -> host collapses to 1.
    t2 = _rendezvous(group, device, chunk)
    o2 = torch.empty_like(inp)
    all_to_all(t2, o2, inp, config=CuteA2AConfig(num_blocks=3, cluster=2))
    torch.cuda.synchronize(device)
    ok = torch.equal(o1, gold) and torch.equal(o2, gold)
    dist.barrier(group)
    return _report("cluster_and_collapse", ok, device, group, rank)


def _case_allow_geometry_switch(group, rank, ws, device) -> bool:
    """``COMMS_DSL_ALLOW_GEOMETRY_SWITCH=1`` downgrades the geometry guard to a silent
    advance for sync-separated callers: a num_blocks 2 -> 4 switch on a reused transport
    must NOT raise and must stay bit-exact (a device sync + barrier separates the calls, so
    no bytes are in flight across the switch)."""
    from comms.dsl.cute.a2a.host import all_to_all
    from comms.dsl.cute.a2a.tuning import CuteA2AConfig

    numel = ws * 4096
    chunk = numel // ws
    inp = _make_input(rank, numel, device)
    gold = _golden(group, inp)
    t = _rendezvous(group, device, chunk)
    out = torch.empty_like(inp)

    prev = os.environ.get("COMMS_DSL_ALLOW_GEOMETRY_SWITCH")
    ok = False
    try:
        os.environ["COMMS_DSL_ALLOW_GEOMETRY_SWITCH"] = "1"
        all_to_all(t, out, inp, config=CuteA2AConfig(num_blocks=2))
        torch.cuda.synchronize(device)
        first_ok = torch.equal(out, gold)
        dist.barrier(group)
        # Geometry switch (num_blocks 2 -> 4) must be allowed (no raise) under the env.
        all_to_all(t, out, inp, config=CuteA2AConfig(num_blocks=4))
        torch.cuda.synchronize(device)
        ok = first_ok and torch.equal(out, gold)
    finally:
        if prev is None:
            os.environ.pop("COMMS_DSL_ALLOW_GEOMETRY_SWITCH", None)
        else:
            os.environ["COMMS_DSL_ALLOW_GEOMETRY_SWITCH"] = prev
    dist.barrier(group)
    return _report("allow_geometry_switch", ok, device, group, rank)


def _case_rejected_call_no_baseline_pollution(group, rank, ws, device) -> bool:
    """A call rejected by a pre-dispatch guard must NOT seed the geometry baseline: the
    geometry commit happens only just before dispatch. Here the first call is rejected
    (primitive='direct' is not shipped) and a subsequent legit copy call on the SAME
    transport must succeed rather than spuriously raise 'geometry changed'."""
    from comms.dsl.cute.a2a.host import all_to_all
    from comms.dsl.cute.a2a.tuning import CuteA2AConfig

    numel = ws * 4096
    chunk = numel // ws
    inp = _make_input(rank, numel, device)
    gold = _golden(group, inp)
    t = _rendezvous(group, device, chunk)
    out = torch.empty_like(inp)

    rejected = False
    try:
        all_to_all(t, out, inp, config=CuteA2AConfig(primitive="direct"))
    except ValueError:
        rejected = True
    # The rejected call dispatched nothing, so this legit copy call must succeed.
    all_to_all(t, out, inp, config=CuteA2AConfig(num_blocks=2))
    torch.cuda.synchronize(device)
    ok = rejected and torch.equal(out, gold)
    dist.barrier(group)
    return _report("rejected_call_no_baseline_pollution", ok, device, group, rank)


def _run_cases(group, rank, ws, device) -> list[bool]:
    return [
        _case_concurrent_transports(group, rank, ws, device),
        _case_geometry_switch_guard(group, rank, ws, device),
        _case_cluster(group, rank, ws, device),
        _case_allow_geometry_switch(group, rank, ws, device),
        _case_rejected_call_no_baseline_pollution(group, rank, ws, device),
    ]


def _mp_worker(rank: int, world_size: int, port: int, result_q) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        device = torch.device(f"cuda:{rank}")
        group = dist.group.WORLD
        assert group is not None
        result_q.put((rank, all(_run_cases(group, rank, world_size, device))))
    finally:
        dist.barrier()
        dist.destroy_process_group()


class A2ACuteRobustTest(unittest.TestCase):
    def test_robustness(self) -> None:
        if torch.cuda.device_count() < 2:
            self.skipTest("needs >=2 GPUs")
        ws = min(torch.cuda.device_count(), 8)
        ctx = mp.get_context("spawn")
        result_q = ctx.SimpleQueue()
        mp.spawn(
            _mp_worker, args=(ws, _find_free_port(), result_q), nprocs=ws, join=True
        )
        results = sorted(result_q.get() for _ in range(ws))
        self.assertEqual([r for r, _ in results], list(range(ws)))
        self.assertTrue(all(ok for _, ok in results), f"per-rank pass flags: {results}")
