# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe
# Distributed test: runs under mp.spawn and touches untyped cutlass / symmetric-memory symbols
# that pyre cannot model, so strict typing adds no value here.

"""Robustness tests for the CuTe-DSL all_to_all copy schedule.

* **Concurrent transports** -- two independent transports interleaved on the same ranks
  must not cross-talk; each keeps its own staging buffer + signal pad + step counters.
* **Geometry guard** -- switching a geometry field (``num_blocks``) on a reused transport
  or a resolved staging layout must FIRE the runtime guard (``check_geometry``) before
  dispatch.

Reused-transport calls rely on the device-side HEAD/TAIL protocol rather than a caller
barrier. Gold is ``dist.all_to_all_single``. Skipped unless >=2 GPUs are present.
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


def _configs_rejected(
    all_to_all, transport, out, inp, configs, expected_message: str | None = None
) -> bool:
    for config in configs:
        try:
            all_to_all(transport, out, inp, config=config)
        except ValueError as error:
            if expected_message is None or expected_message in str(error):
                continue
        return False
    return True


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
    all_to_all(t1, o1, inp, config=cfg)
    all_to_all(t2, o2, inp, config=cfg)
    torch.cuda.synchronize(device)
    ok = torch.equal(o1, gold) and torch.equal(o2, gold)
    dist.barrier(group)
    return _report("concurrent_transports", ok, device, group, rank)


def _case_channel_transport_state_isolation(group, rank, ws, device) -> bool:
    """Full-staging and bounded-ring transports keep independent signal/counter state
    while changed-input calls are interleaved without a host-side drain."""
    from comms.dsl import nvl_rendezvous
    from comms.dsl.cute.a2a.host import all_to_all
    from comms.dsl.cute.a2a.tuning import CuteA2AConfig

    numel = ws * 256 * 1024
    chunk = numel // ws
    full = _rendezvous(group, device, chunk)
    ring = nvl_rendezvous(group, device, per_peer_bytes=64 * 1024)
    full_out = torch.empty(numel, dtype=torch.float32, device=device)
    ring_out = torch.empty_like(full_out)
    full_inputs = [_make_input(rank + epoch * 17, numel, device) for epoch in (1, 2)]
    ring_inputs = [_make_input(rank + epoch * 29, numel, device) for epoch in (1, 2)]
    full_golds = [_golden(group, inp) for inp in full_inputs]
    ring_golds = [_golden(group, inp) for inp in ring_inputs]
    full_config = CuteA2AConfig(
        num_blocks=2,
        num_threads=256,
        primitive="copy_channel_full",
        send_threads=96,
        peer_fanout=1,
        unroll=8,
    )
    ring_config = CuteA2AConfig(
        num_blocks=2,
        num_threads=256,
        num_slots=3,
        primitive="copy_channel_ring",
        send_threads=160,
        unroll=4,
    )

    full_first = torch.empty_like(full_out)
    ring_first = torch.empty_like(ring_out)
    all_to_all(full, full_out, full_inputs[0], config=full_config)
    full_first.copy_(full_out)
    all_to_all(ring, ring_out, ring_inputs[0], config=ring_config)
    ring_first.copy_(ring_out)
    all_to_all(full, full_out, full_inputs[1], config=full_config)
    all_to_all(ring, ring_out, ring_inputs[1], config=ring_config)
    torch.cuda.synchronize(device)
    ok = all(
        (
            torch.equal(full_first, full_golds[0]),
            torch.equal(ring_first, ring_golds[0]),
            torch.equal(full_out, full_golds[1]),
            torch.equal(ring_out, ring_golds[1]),
        )
    )
    dist.barrier(group)
    return _report("channel_transport_state_isolation", ok, device, group, rank)


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


def _case_resolved_geometry_switch_guard(group, rank, ws, device) -> bool:
    """Effective classic tile ownership and bounded-ring slot layout are guarded even when
    the raw ``num_blocks``/``primitive`` fields stay unchanged."""
    from comms.dsl import nvl_rendezvous
    from comms.dsl.cute.a2a.host import all_to_all
    from comms.dsl.cute.a2a.tuning import CuteA2AConfig

    previous_nt = os.environ.pop("A2A_CUTE_NT", None)
    try:
        numel = ws * 8192
        chunk = numel // ws
        inp = _make_input(rank, numel, device)
        gold = _golden(group, inp)
        out = torch.empty_like(inp)
        classic = _rendezvous(group, device, chunk)
        all_to_all(
            classic,
            out,
            inp,
            config=CuteA2AConfig(num_blocks=2, num_threads=256),
        )
        torch.cuda.synchronize(device)
        classic_ok = torch.equal(out, gold)
        classic_raised = False
        try:
            all_to_all(
                classic,
                out,
                inp,
                config=CuteA2AConfig(num_blocks=2, num_threads=512),
            )
        except ValueError:
            classic_raised = True

        ring_bytes = 64 * 1024
        ring = nvl_rendezvous(group, device, per_peer_bytes=ring_bytes)
        ring_inp = _make_input(rank + 17, ws * 256 * 1024, device)
        ring_gold = _golden(group, ring_inp)
        ring_out = torch.empty_like(ring_inp)
        ring_cfg = CuteA2AConfig(
            num_blocks=2,
            num_threads=256,
            num_slots=3,
            unroll=4,
            primitive="copy_channel_ring",
            send_threads=160,
        )
        all_to_all(ring, ring_out, ring_inp, config=ring_cfg)
        torch.cuda.synchronize(device)
        ring_ok = torch.equal(ring_out, ring_gold)
        ring_raised = False
        try:
            all_to_all(
                ring,
                ring_out,
                ring_inp,
                config=CuteA2AConfig(
                    num_blocks=2,
                    num_threads=256,
                    num_slots=4,
                    unroll=4,
                    primitive="copy_channel_ring",
                    send_threads=160,
                ),
            )
        except ValueError:
            ring_raised = True
    finally:
        if previous_nt is not None:
            os.environ["A2A_CUTE_NT"] = previous_nt

    dist.barrier(group)
    return _report(
        "resolved_geometry_switch_guard",
        classic_ok and classic_raised and ring_ok and ring_raised,
        device,
        group,
        rank,
    )


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

    direct_rejected = False
    try:
        all_to_all(t, out, inp, config=CuteA2AConfig(primitive="direct"))
    except ValueError:
        direct_rejected = True
    legacy_primitives_rejected = _configs_rejected(
        all_to_all,
        t,
        out,
        inp,
        tuple(
            CuteA2AConfig(primitive=primitive)
            for primitive in (
                "copy_channel_ws",
                "copy_channel_ws_ring",
                "copy_channel_peer_groups",
                "copy_channel_peer_waves2",
                "copy_channel_peer_waves4",
                "copy_channel_cta_roles",
                "copy_channel_peer_groups_cta_roles",
                "copy_channel_peer_waves2_cta_roles",
                "copy_channel_peer_waves4_cta_roles",
            )
        ),
        "not shipped yet",
    )
    ring_grid_rejected = False
    try:
        all_to_all(
            t,
            out,
            inp,
            config=CuteA2AConfig(
                num_blocks=0,
                primitive="copy_channel_ring",
            ),
        )
    except ValueError:
        ring_grid_rejected = True
    # The rejected call dispatched nothing, so this legit copy call must succeed.
    all_to_all(t, out, inp, config=CuteA2AConfig(num_blocks=2))
    torch.cuda.synchronize(device)
    ok = (
        direct_rejected
        and legacy_primitives_rejected
        and ring_grid_rejected
        and torch.equal(out, gold)
    )
    dist.barrier(group)
    return _report("rejected_call_no_baseline_pollution", ok, device, group, rank)


def _case_input_and_config_guards(group, rank, ws, device) -> bool:
    from comms.dsl.cute.a2a.host import all_to_all
    from comms.dsl.cute.a2a.tuning import CuteA2AConfig

    numel = ws * 4096
    chunk = numel // ws
    inp = _make_input(rank, numel, device)
    out = torch.empty_like(inp)
    transport = _rendezvous(group, device, chunk)

    alias_rejected = False
    try:
        all_to_all(transport, inp, inp, config=CuteA2AConfig(num_blocks=2))
    except ValueError:
        alias_rejected = True
    misaligned_rejected = True
    misaligned_in = _make_input(rank, numel + 1, device)[1:]
    misaligned_out = torch.empty(numel + 1, dtype=inp.dtype, device=device)[1:]
    for candidate_out, candidate_in in (
        (out, misaligned_in),
        (misaligned_out, inp),
    ):
        try:
            all_to_all(
                transport,
                candidate_out,
                candidate_in,
                config=CuteA2AConfig(num_blocks=2),
            )
            misaligned_rejected = False
        except ValueError as error:
            misaligned_rejected = misaligned_rejected and "16-byte-aligned" in str(
                error
            )
    negative_threads_rejected = False
    try:
        all_to_all(
            transport,
            out,
            inp,
            config=CuteA2AConfig(num_blocks=2, num_threads=-32),
        )
    except ValueError:
        negative_threads_rejected = True
    fixed_fanout_world_check_ok = ws == 8
    if ws != 8:
        fixed_fanout_world_check_ok = _configs_rejected(
            all_to_all,
            transport,
            out,
            inp,
            tuple(
                CuteA2AConfig(
                    num_blocks=4,
                    num_threads=1024,
                    primitive="copy_channel_full",
                    send_threads=512,
                    peer_fanout=peer_fanout,
                )
                for peer_fanout in (2, 4)
            ),
        )

    fanout_shape_rejected = ws != 8
    fanout_asymmetric_splits_rejected = ws != 8
    if ws == 8:
        invalid_fanout_configs = tuple(
            config
            for peer_fanout in (2, 4)
            for config in (
                CuteA2AConfig(
                    num_blocks=2,
                    num_threads=1024,
                    primitive="copy_channel_full",
                    send_threads=512,
                    peer_fanout=peer_fanout,
                ),
                CuteA2AConfig(
                    num_blocks=4,
                    num_threads=512,
                    primitive="copy_channel_full",
                    send_threads=256,
                    peer_fanout=peer_fanout,
                ),
                CuteA2AConfig(
                    num_blocks=4,
                    num_threads=1024,
                    num_slots=1,
                    primitive="copy_channel_full",
                    send_threads=512,
                    peer_fanout=peer_fanout,
                ),
                CuteA2AConfig(
                    num_blocks=4,
                    num_threads=1024,
                    primitive="copy_channel_full",
                    cluster=2,
                    send_threads=512,
                    peer_fanout=peer_fanout,
                ),
            )
        )
        fanout_shape_rejected = _configs_rejected(
            all_to_all,
            transport,
            out,
            inp,
            invalid_fanout_configs,
        )
        asymmetric_fanout_configs = tuple(
            CuteA2AConfig(
                num_blocks=4,
                num_threads=1024,
                primitive="copy_channel_full",
                send_threads=send_threads,
                peer_fanout=peer_fanout,
            )
            for peer_fanout in (2, 4)
            for send_threads in (448, 480, 544, 576)
        )
        fanout_asymmetric_splits_rejected = _configs_rejected(
            all_to_all,
            transport,
            out,
            inp,
            asymmetric_fanout_configs,
            "requires exactly 512 send_threads",
        )
    invalid_fanout_rejected = _configs_rejected(
        all_to_all,
        transport,
        out,
        inp,
        tuple(
            CuteA2AConfig(
                num_blocks=4,
                num_threads=1024,
                primitive="copy_channel_full",
                send_threads=512,
                peer_fanout=peer_fanout,
            )
            for peer_fanout in (3, 7)
        ),
        "peer_fanout in",
    )
    nonfull_fanout_rejected = _configs_rejected(
        all_to_all,
        transport,
        out,
        inp,
        (
            CuteA2AConfig(num_blocks=2, primitive="copy", peer_fanout=1),
            CuteA2AConfig(
                num_blocks=4,
                num_threads=256,
                num_slots=4,
                primitive="copy_channel_ring",
                send_threads=128,
                peer_fanout=1,
            ),
        ),
        "does not use peer_fanout",
    )
    dist.barrier(group)
    return _report(
        "input_and_config_guards",
        alias_rejected
        and misaligned_rejected
        and negative_threads_rejected
        and fixed_fanout_world_check_ok
        and fanout_shape_rejected
        and fanout_asymmetric_splits_rejected
        and invalid_fanout_rejected
        and nonfull_fanout_rejected,
        device,
        group,
        rank,
    )


def _case_channel_cluster_guards(group, rank, ws, device) -> bool:
    from comms.dsl.cute.a2a.host import all_to_all
    from comms.dsl.cute.a2a.tuning import CuteA2AConfig

    numel = ws * 4096
    chunk = numel // ws
    inp = _make_input(rank, numel, device)
    out = torch.empty_like(inp)
    transport = _rendezvous(group, device, chunk)
    invalid_configs = [
        CuteA2AConfig(
            num_blocks=4,
            num_threads=256,
            primitive="copy_channel_full",
            send_threads=128,
            cluster=4,
        ),
        CuteA2AConfig(
            num_blocks=4,
            num_threads=256,
            primitive="copy_channel_full",
            send_threads=128,
            cluster_y=2,
        ),
        CuteA2AConfig(
            num_blocks=4,
            num_threads=256,
            num_slots=4,
            primitive="copy_channel_ring",
            send_threads=128,
            cluster=4,
        ),
    ]
    invalid_configs_rejected = _configs_rejected(
        all_to_all,
        transport,
        out,
        inp,
        tuple(invalid_configs),
        "requires cluster in",
    )

    dist.barrier(group)
    return _report(
        "channel_cluster_guards",
        invalid_configs_rejected,
        device,
        group,
        rank,
    )


def _run_cases(group, rank, ws, device) -> list[bool]:
    return [
        _case_concurrent_transports(group, rank, ws, device),
        _case_channel_transport_state_isolation(group, rank, ws, device),
        _case_geometry_switch_guard(group, rank, ws, device),
        _case_resolved_geometry_switch_guard(group, rank, ws, device),
        _case_cluster(group, rank, ws, device),
        _case_allow_geometry_switch(group, rank, ws, device),
        _case_rejected_call_no_baseline_pollution(group, rank, ws, device),
        _case_input_and_config_guards(group, rank, ws, device),
        _case_channel_cluster_guards(group, rank, ws, device),
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
