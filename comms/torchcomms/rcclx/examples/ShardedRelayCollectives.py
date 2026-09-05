#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Usage examples for the RCCLX sharded-relay collectives and its control plane.

The sharded relay is for 2D sparse parallelism: within a comm, a few "active"
ranks perform a logical collective while the remaining idle GPUs act as
passthrough "helpers" that relay sharded chunks (eliminating XGMI link
contention on MI300x/MI350x). These examples use a single active group on an
8-GPU node, for both A=2 (2 active + 6 helpers) and A=4 (4 active + 4 helpers):

    A=2  ->  active ranks {0, 1},        helpers {2, 3, 4, 5, 6, 7}
    A=4  ->  active ranks {0, 1, 2, 3},  helpers {4, 5, 6, 7}

Every rank in the comm (active AND helper) must call each collective, with
matching counts and in the same order. The hard part in a real deployment is not
the call -- it is that **a helper has no way to know what to call**. A
communicator is a data plane, not a scheduler: nothing in it can make a helper
process post a collective, and the helper is not running the model, so it never
sees the shapes.

That is what the control plane is for. Rank 0 publishes the plan for each forward
(op code, and one element count per relay call) into a shared-memory segment;
every rank that does not already know the plan consumes it and enqueues from what
it read. Both calls are host-only and bounded by a timeout, so a stalled
publisher raises rather than hanging the node.

    --control=shm   (default)  helpers learn each forward from the segment.
    --control=none             helpers are handed the counts by this script,
                               which is lockstep-by-construction and only works
                               because a demo knows both sides. It is the A/B
                               baseline, not a deployment option.

The role split in this file mirrors that: `run_active` drives a synthetic forward
and `run_helper` only consumes and enqueues. Helper code never sees the demo's
counts except through the plan.

Buffer contract per active rank (count = per-group element count):
    all_reduce      : tensor = count            (in-place)
    reduce_scatter  : input  = A x count -> output = count
    all_gather      : input  = count       -> output = A x count
    all_to_all      : input  = A x count -> output = A x count (distinct)

A default run does one forward per collective in that list, at each active width;
`--forwards` above that repeats the cycle and below it truncates the list.

    --low-precision            request the fp8e4m3 wire format on every relay
                               call. It is a per-call argument, so it composes
                               with every other flag rather than being a mode.
                               It also RAISES the per-call counts, because the
                               internal gate declines below a size crossover and
                               says nothing: at the default counts the flag would
                               be a silent no-op and the demo would still report
                               success. See LP_BASE_COUNT.

Helper ranks pass a single 1-element placeholder tensor -- the C++ kernel stages
helpers into its own internal scratch and never reads/writes the placeholder.

Each relay call in a forward gets its own buffers. Sharing one pair across the
calls of a forward is a cross-rank write-after-read: a rank that finished call i
races ahead and overwrites a send buffer a peer is still reading. A deployment
gives every relay call its own tensor, so this is a property of the demo, not a
constraint of the relay.

**A communicator's active set is fixed.** This script creates one comm per active
width rather than reusing a single comm for A=2 and A=4. The one-shot IPC region
is per-comm, and its handshake uses a per-block epoch counter that a rank bumps
only when it launches the kernel -- which helpers never do. Reusing one comm
across widths therefore promotes a rank from helper to active with its counter
behind its peers', and since the flags are deliberately never cleared, the
"a stale flag always compares as not-yet-arrived" invariant breaks and the
collective silently returns wrong data. A real deployment has a fixed membership
per sparse group, so it gets one comm per group and never sees this.

The widths also get one PROCESS GROUP each, rather than two comms built one after
the other in a single process. Finalizing the first comm is not enough of a
boundary: the caching allocator recycles that width's blocks, and the first
collective on the next comm can read them back -- A=2 followed by A=4 in one
process returns the A=2 reduce_scatter fill inside the A=4 all_reduce on roughly
three runs in four. Re-spawning also keeps each rank's role fixed for the life of
its process, which is what a deployment looks like.

Self-contained: this is a python_unittest that spawns 8 ranks with mp.spawn and
an explicit TCPStore (mirroring bench_sharded_relay_perf), so it builds in-place
(no standalone-PAR packaging). It uses only torch + torchcomms (no torchrec /
caffe2 deps), so the same script also runs standalone against an rcclx wheel:

    buck2 test @mode/opt-amd-gpu -m rocm70 -m rcclx_dev \\
        //comms/torchcomms/rcclx/examples:ShardedRelayCollectives

    # against a wheel venv (self-spawns 8 procs, needs 8 GPUs):
    /path/to/venv/bin/python ShardedRelayCollectives.py
    /path/to/venv/bin/python ShardedRelayCollectives.py --control=none
    /path/to/venv/bin/python ShardedRelayCollectives.py --forwards 5 \\
        --calls-per-forward 3
    /path/to/venv/bin/python ShardedRelayCollectives.py --inject=timeout

The control plane needs NCCL_SHARDED_RELAY_MODE_ENABLE=1 at comm creation; this
script sets it before creating the comm.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
import unittest
from collections.abc import Callable
from dataclasses import dataclass, replace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchcomms import new_comm, ReduceOp

WORLD = 8  # examples assume an 8-GPU node
ACTIVE_COUNTS = (2, 4)  # single-group sizes to demonstrate
BASE_COUNT = 1024  # per-group element count for the first call of a forward

# Per-group element count for the first call of a forward when --low-precision is
# on. It has to be THIS much larger than BASE_COUNT, and the reason is the whole
# trap this flag sets for a demo:
#
# Low precision is requested per call but GRANTED by an internal size-only gate,
# which declines silently below a measured crossover. At BASE_COUNT the calls are
# 4 KB / 2 KB / 1 KB, so `--low-precision` would run the entire demo in full
# precision and report success -- a demo that lies about what it demonstrated.
#
# counts_for_forward divides the base by 1, 2 or 4, so it is the SMALLEST variant
# that has to clear the crossover, not the base.
#
# THE SIZE GATE IS NOT THE ONLY GATE, and the binding constraint here is the other
# one. Low precision also declines when the selected ROUTE is not a relay -- below
# a route's own crossover the call is a direct exchange with the helpers idle, so
# there is no staged boundary-crossing traffic for the wire format to shrink. The
# highest of those crossovers is the A=4 reduce-scatter offload at ~48 MB, on a
# metric of A * count * elementSize, so the smallest variant needs
# 4 * count * 4 >= 48 MiB, i.e. count >= 3 Mi, i.e. LP_BASE_COUNT >= 12 Mi.
#
# 16 Mi clears it with margin: the smallest variant is 4 Mi elements = 64 MiB on
# that metric. It is also a multiple of 4 * 128, which the flat 4-active allreduce
# needs -- that schedule splits its direct region into A per-owner shards, and a
# shard is a whole number of 128-element wire blocks only when the count is a
# multiple of A * 128.
#
# _assert_low_precision_sizes() checks the size gate and the alignment at run time.
# It deliberately does NOT re-implement the route crossovers: that would duplicate
# eight thresholds from sharded_relay_route.h into a demo, and the C++ suites
# already assert route selection per schedule. The number above is the reason the
# route gate is satisfied; if a route threshold moves, the symptom is a decline
# logged at INFO under NCCL_DEBUG_SUBSYS=COLL, not a wrong answer.
LP_BASE_COUNT = 16 * 1024 * 1024

# Mirrors lpMinBytes() in meta/relay/sharded_relay_lp.h. Duplicated on purpose:
# there is no Python-visible accessor, and the alternative is a demo that cannot
# tell whether the flag it advertises did anything. The C++ side is the source of
# truth and sharded_relay_lp_test pins its value; if that value moves, the
# assertion below fires with a message saying to update this.
LP_MIN_BYTES = 4 << 20

_MS = 1_000_000
FORWARD_TIMEOUT_NS = 60_000 * _MS  # generous: a real forward's publish/consume
SHORT_TIMEOUT_NS = 300 * _MS  # for the fault cases, so they fail fast

# Mirrors ncclRelayOp_t in nccl.h. Shutdown is an opcode rather than a separate
# entry point, so a graceful stop needs no extra API.
OP_SHUTDOWN = 0
OP_ALL_REDUCE = 1
OP_REDUCE_SCATTER = 2
OP_ALL_GATHER = 3
OP_ALL_TO_ALL = 4

OP_NAMES = {
    OP_SHUTDOWN: "shutdown",
    OP_ALL_REDUCE: "all_reduce",
    OP_REDUCE_SCATTER: "reduce_scatter",
    OP_ALL_GATHER: "all_gather",
    OP_ALL_TO_ALL: "all_to_all",
}

# A phase cycles these one per forward, so a run shorter than the cycle exercises
# a prefix of it and silently leaves the tail untouched -- hence the default
# below is the full cycle rather than a round number.
RELAY_OPS = (OP_ALL_REDUCE, OP_REDUCE_SCATTER, OP_ALL_GATHER, OP_ALL_TO_ALL)

NCCL_FLOAT32 = 7  # ncclDataType_t
NCCL_SUM = 0  # ncclRedOp_t
NCCL_MAX = 2  # ncclRedOp_t


def expected_red_op(op: int) -> int:
    """The red_op a plan for `op` carries. One definition, two readers.

    The reducing collectives here run SUM, so that is what their plan must say.
    For all-gather and all-to-all the field is documented as ignored, which makes
    them the honest place to carry a NON-default value: the plan stays truthful
    while the value still proves the field survives the wire path.

    That matters because the plan crosses three records -- RcclxRelayPlan ->
    ncclRelayPlanInfo -> RelayPlanInfo -- as five adjacent same-typed uint32s
    copied by hand. Nothing here used to read dtype, red_op or flags, so
    transposing two of them changed no observable behaviour and every subtest
    still passed. Note NCCL_SUM is 0 and NCCL_FLOAT32 is 7: asserting both means
    even a dtype/red_op swap is caught.
    """
    return NCCL_MAX if op in (OP_ALL_GATHER, OP_ALL_TO_ALL) else NCCL_SUM


INJECT_MODES = ("none", "timeout", "crash", "mismatch", "abort", "overflow")


def _make_comm(store: dist.TCPStore, tag: str):
    """One RCCLX comm, namespaced so several can coexist over one store."""
    name = f"relay_demo_{tag}"
    return new_comm(
        "rcclx",
        torch.device("hip"),
        name=name,
        store=dist.PrefixStore(name, store),
    )


@dataclass
class Config:
    """Picklable so it can cross mp.spawn.

    active_counts lives here rather than being read from the module constant
    because mp.spawn re-imports this module in every child, so a constant rebound
    in the parent never reaches them -- only what is pickled into the worker args
    does.
    """

    control: str = "shm"
    forwards: int = len(RELAY_OPS)
    calls_per_forward: int = 2
    inject: str = "none"
    graph: bool = False
    active_counts: tuple[int, ...] = ACTIVE_COUNTS
    low_precision: bool = False


@dataclass
class _ShapeGraph:
    """A captured graph plus the buffers whose addresses it baked in."""

    graph: torch.cuda.CUDAGraph
    inp: torch.Tensor
    out: torch.Tensor


def _capture_shape(
    rcclx,
    rank: int,
    active: list[int],
    dev: torch.device,
    op: int,
    count: int,
    low_precision: bool = False,
) -> _ShapeGraph:
    """Capture one (op, count) shape, after warming it up outside the capture.

    The warm-up is required, not just polite: the first call on a comm builds the
    one-shot IPC region, which does a bootstrap all-gather and a synchronous
    memset. Neither is capturable, and the relay deliberately declines the
    one-shot path under capture unless the region already exists -- so capturing
    cold would silently record the slower route.

    Low precision has the SAME property and the same fix: its arena is built on
    the first call that asks for it, by a bootstrap all-gather, so a cold capture
    would record a full-precision graph. The warm-up below is what makes the
    captured graph a low-precision one.
    """
    inp, out = stage_call(op, rank, active, dev, count)
    enqueue_call(rcclx, op, active, inp, out, count, low_precision)
    torch.cuda.current_stream().synchronize()

    graph = torch.cuda.CUDAGraph()
    # Relaxed matches the C++ suite's hipStreamCaptureModeRelaxed: the relay's
    # own scratch bookkeeping would otherwise trip the stricter global mode.
    with torch.cuda.graph(graph, capture_error_mode="relaxed"):
        enqueue_call(rcclx, op, active, inp, out, count, low_precision)
    return _ShapeGraph(graph=graph, inp=inp, out=out)


def _capture_all(
    rcclx, rank: int, active: list[int], dev: torch.device, cfg: Config
) -> dict[tuple[int, int], _ShapeGraph]:
    """One graph per (op, count) the run will use.

    A graph is pinned to a single plan SHAPE, so a run that varies its counts
    needs one per shape and must pick between them per forward -- which is the
    whole reason the plan travels out of band rather than being baked in.

    Every rank captures the same (op, count) pairs in the same order, so no rank
    has to be told which shapes exist.
    """
    shapes = sorted(
        {
            count
            for forward in range(cfg.forwards)
            for count in counts_for_forward(
                forward, cfg.calls_per_forward, base_count(cfg)
            )
        }
    )
    graphs = {}
    for op in RELAY_OPS:
        for count in shapes:
            graphs[(op, count)] = _capture_shape(
                rcclx, rank, active, dev, op, count, cfg.low_precision
            )
    return graphs


def _placeholder(dev: torch.device) -> torch.Tensor:
    """1-element helper slot: the kernel uses internal scratch, ignores this."""
    return torch.empty(1, dtype=torch.float32, device=dev)


def _check(rank: int, name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    ok = torch.allclose(actual, expected)
    if not ok:
        diff = (actual != expected).nonzero().flatten()
        first = int(diff[0].item()) if diff.numel() else -1
        raise AssertionError(
            f"Rank {rank}: {name} mismatch: {diff.numel()}/{actual.numel()} elems "
            f"differ, first at {first}: got {actual[first].item()} "
            f"want {expected[first].item()}; "
            f"got[:4]={actual[:4].tolist()} want[:4]={expected[:4].tolist()}"
        )


def base_count(cfg: Config) -> int:
    """The first call's per-group count, which low precision has to raise.

    Not a module constant read directly, because mp.spawn re-imports this module
    in every child: only what is pickled into Config reaches them.
    """
    return LP_BASE_COUNT if cfg.low_precision else BASE_COUNT


def counts_for_forward(forward: int, calls: int, base: int = BASE_COUNT) -> list[int]:
    """Counts vary per forward and per call, mimicking a varying chunk count.

    A fixed shape would let a helper that ignored the plan still pass, which is
    the bug this example exists to make visible.
    """
    return [base // (1 << ((forward + i) % 3)) for i in range(calls)]


def _assert_low_precision_sizes(cfg: Config) -> None:
    """Fail loudly if --low-precision would be a silent no-op.

    The gate declines below a size threshold and says nothing, so a demo that only
    passed the flag would print success having exercised nothing. Checked here, on
    the smallest count any forward will use, because that is the one that decides.

    Engagement itself is asserted in the C++ suites, which can read the counters
    directly; this is the part reachable from Python, and it is the part that
    actually goes wrong.
    """
    if not cfg.low_precision:
        return
    counts = [
        count
        for forward in range(cfg.forwards)
        for count in counts_for_forward(forward, cfg.calls_per_forward, LP_BASE_COUNT)
    ]
    smallest = min(counts)
    smallest_bytes = smallest * 4  # fp32
    if smallest_bytes < LP_MIN_BYTES:
        raise AssertionError(
            f"--low-precision would be a silent no-op: smallest count {smallest} "
            f"is {smallest_bytes} B, below the {LP_MIN_BYTES} B gate. Raise "
            f"LP_BASE_COUNT, or lower LP_MIN_BYTES if lpMinBytes() moved."
        )
    align = 4 * 128  # A * kLpBlockElems at the widest active count here
    unaligned = [c for c in counts if c % align]
    if unaligned:
        raise AssertionError(
            f"--low-precision counts must be multiples of {align} for the flat "
            f"4-active allreduce; these are not: {sorted(set(unaligned))}"
        )


def stage_call(
    op: int, rank: int, active: list[int], dev: torch.device, count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate and fill the send/recv buffers for one relay call."""
    A = len(active)
    if rank not in active:
        slot = _placeholder(dev)  # helper: same placeholder in and out
        return slot, slot

    ai = active.index(rank)
    if op == OP_ALL_REDUCE:
        t = torch.full((count,), float(ai + 1), dtype=torch.float32, device=dev)
        return t, t  # in-place
    if op == OP_REDUCE_SCATTER:
        # block[i] value encodes both the sender (ai) and the destination (i).
        inp = torch.empty(A * count, dtype=torch.float32, device=dev)
        for i in range(A):
            inp[i * count : (i + 1) * count] = float((ai + 1) + 100 * (i + 1))
        return inp, torch.empty(count, dtype=torch.float32, device=dev)
    if op == OP_ALL_GATHER:
        inp = torch.full((count,), float(ai + 1), dtype=torch.float32, device=dev)
        return inp, torch.empty(A * count, dtype=torch.float32, device=dev)
    if op == OP_ALL_TO_ALL:
        # send segment j encodes sender (ai) and destination (j).
        inp = torch.empty(A * count, dtype=torch.float32, device=dev)
        for j in range(A):
            inp[j * count : (j + 1) * count] = float((ai + 1) * 10 + (j + 1))
        return inp, torch.empty(A * count, dtype=torch.float32, device=dev)
    raise ValueError(f"unsupported op code {op}")


def enqueue_call(
    rcclx,
    op: int,
    active: list[int],
    inp: torch.Tensor,
    out: torch.Tensor,
    count: int,
    low_precision: bool = False,
) -> None:
    """One relay call. Every rank in the comm calls this, active or not.

    low_precision is COLLECTIVE: it reaches here from Config, which is pickled
    into every worker, so active ranks and helpers cannot disagree about it. They
    must not -- ranks that disagree disagree on how many bytes cross each link, so
    the call hangs rather than degrading.
    """
    if op == OP_ALL_REDUCE:
        rcclx.sharded_relay_multi_group_all_reduce(
            [inp], ReduceOp.SUM, [active], [count], low_precision=low_precision
        )
    elif op == OP_REDUCE_SCATTER:
        rcclx.sharded_relay_multi_group_reduce_scatter(
            [inp],
            [out],
            ReduceOp.SUM,
            [active],
            [count],
            low_precision=low_precision,
        )
    elif op == OP_ALL_GATHER:
        rcclx.sharded_relay_multi_group_all_gather(
            [inp], [out], [active], [count], low_precision=low_precision
        )
    elif op == OP_ALL_TO_ALL:
        rcclx.sharded_relay_multi_group_all_to_all(
            [inp], [out], [active], [count], low_precision=low_precision
        )
    else:
        raise ValueError(f"unsupported op code {op}")


def verify_call(
    op: int,
    rank: int,
    active: list[int],
    dev: torch.device,
    count: int,
    out: torch.Tensor,
) -> None:
    """Check one relay call's result. Active ranks only; helpers hold scratch."""
    if rank not in active:
        return
    A = len(active)
    ai = active.index(rank)
    rank_sum = sum(a + 1 for a in range(A))
    name = f"{OP_NAMES[op]} A={A} count={count}"

    if op == OP_ALL_REDUCE:
        _check(rank, name, out, torch.full_like(out, float(rank_sum)))
    elif op == OP_REDUCE_SCATTER:
        # out = sum over senders a of block[ai] = sum_a[(a+1) + 100*(ai+1)]
        expected = torch.full_like(out, float(rank_sum + 100 * A * (ai + 1)))
        _check(rank, name, out, expected)
    elif op == OP_ALL_GATHER:
        expected = torch.empty(A * count, dtype=torch.float32, device=dev)
        for i in range(A):
            expected[i * count : (i + 1) * count] = float(i + 1)
        _check(rank, name, out, expected)
    elif op == OP_ALL_TO_ALL:
        expected = torch.empty(A * count, dtype=torch.float32, device=dev)
        for sp in range(A):
            expected[sp * count : (sp + 1) * count] = float((sp + 1) * 10 + (ai + 1))
        _check(rank, name, out, expected)


def publish_plan(
    rcclx, epoch: int, op: int, counts: list[int], timeout_ns: int = FORWARD_TIMEOUT_NS
) -> int:
    """Rank 0 only. Returns the host cost in ns."""
    t0 = time.perf_counter_ns()
    rcclx.relay_control_publish(
        epoch=epoch,
        counts=counts,
        op_code=op,
        dtype=NCCL_FLOAT32,
        timeout_ns=timeout_ns,
        red_op=expected_red_op(op),
    )
    return time.perf_counter_ns() - t0


def consume_plan(
    rcclx, epoch: int, timeout_ns: int = FORWARD_TIMEOUT_NS
) -> tuple[int, list[int], int]:
    """Returns (op_code, counts, host cost in ns).

    dtype, red_op and flags are checked here rather than returned: they are part
    of the plan's contract but nothing downstream in this example consumes them,
    and checking them at the one place they arrive is what stops them from being
    dead fields that a transposition could not disturb.
    """
    t0 = time.perf_counter_ns()
    op_code, dtype, red_op, flags, counts = rcclx.relay_control_consume(
        epoch=epoch, timeout_ns=timeout_ns
    )
    ns = time.perf_counter_ns() - t0
    if op_code != OP_SHUTDOWN:
        # A shutdown plan carries no shape, so only a live plan's fields are
        # meaningful.
        if dtype != NCCL_FLOAT32:
            raise AssertionError(
                f"plan dtype {dtype} != published {NCCL_FLOAT32} "
                f"(op_code={op_code}); the plan's fields are transposed"
            )
        want_red_op = expected_red_op(op_code)
        if red_op != want_red_op:
            raise AssertionError(
                f"plan red_op {red_op} != published {want_red_op} "
                f"(op_code={op_code}); the plan's fields are transposed"
            )
        if flags != 0:
            raise AssertionError(f"plan flags {flags} != 0 (op_code={op_code})")
    return op_code, list(counts), ns


def run_active(
    rcclx,
    rank: int,
    active: list[int],
    dev: torch.device,
    epoch: int,
    op: int,
    counts: list[int],
    cfg: Config,
    stats: dict[str, list[int]],
    graphs: dict[tuple[int, int], _ShapeGraph] | None = None,
) -> None:
    """Drives one synthetic forward: publishes the plan, then enqueues it.

    Only rank 0 publishes -- there is one ring per communicator, so a second
    publisher would race the same seqlock.
    """
    if cfg.control == "shm" and rank == 0:
        stats["publish_ns"].append(publish_plan(rcclx, epoch, op, counts))

    for count, out in _execute_calls(
        rcclx, op, rank, active, dev, counts, graphs, stats, cfg.low_precision
    ):
        verify_call(op, rank, active, dev, count, out)


def _execute_calls(
    rcclx,
    op: int,
    rank: int,
    active: list[int],
    dev: torch.device,
    counts: list[int],
    graphs: dict[tuple[int, int], _ShapeGraph] | None,
    stats: dict[str, list[int]],
    low_precision: bool = False,
) -> list[tuple[int, torch.Tensor]]:
    """Run one forward's calls, eager or by graph replay.

    Returns [(count, output)] for the caller to check. Note the publish/consume
    that chose these counts happened OUTSIDE any captured region: host code
    inside a capture runs at capture time and is not recorded, so a consume baked
    into a graph would happen exactly once no matter how many times it replays.
    """
    results: list[tuple[int, torch.Tensor]] = []
    # Every buffer stays referenced until the whole forward has completed. The
    # relay has helpers write results back into the active rank's buffer, so a
    # tensor dropped after its call can be handed straight back out by the
    # caching allocator and overwritten by an enqueue for the NEXT call while a
    # peer's write for the previous one is still in flight. Holding the
    # references removes the aliasing outright, which is cheaper and more honest
    # than synchronizing every call.
    keepalive: list[torch.Tensor] = []

    for count in counts:
        if graphs is not None:
            sg = graphs[(op, count)]
            # A graph bakes buffer POINTERS, so the input has to be refilled in
            # place; reallocating would leave the replay reading the old address.
            fresh_in, _fresh_out = stage_call(op, rank, active, dev, count)
            sg.inp.copy_(fresh_in)
            t0 = time.perf_counter_ns()
            sg.graph.replay()
            stats["enqueue_ns"].append(time.perf_counter_ns() - t0)
            # Snapshot rather than handing out sg.out itself. The graph for a
            # given (op, count) owns ONE output buffer, so two calls in the same
            # forward with the same count would otherwise append the same tensor
            # twice and the first call's check would silently be run against the
            # second call's output. That is reachable by configuration, not just
            # in theory: counts_for_forward cycles with period 3, so any
            # --calls-per-forward above 3 repeats a count within one forward. The
            # clone is enqueued on the same stream right after the replay, so it
            # is ordered against it and the loop's closing synchronize covers it.
            results.append((count, sg.out.clone()))
        else:
            inp, out = stage_call(op, rank, active, dev, count)
            t0 = time.perf_counter_ns()
            enqueue_call(rcclx, op, active, inp, out, count, low_precision)
            stats["enqueue_ns"].append(time.perf_counter_ns() - t0)
            keepalive.extend((inp, out))
            results.append((count, out))

    torch.cuda.current_stream().synchronize()
    return results


def run_helper(
    rcclx,
    rank: int,
    active: list[int],
    dev: torch.device,
    epoch: int,
    cfg: Config,
    stats: dict[str, list[int]],
    fallback_op: int,
    fallback_counts: list[int],
    graphs: dict[tuple[int, int], _ShapeGraph] | None = None,
) -> None:
    """Consumes the plan and enqueues from it. Knows nothing else.

    `fallback_*` are only used by --control=none, where the script hands the
    helper the shapes directly. They are what the control plane replaces.
    """
    if cfg.control == "shm":
        op, counts, ns = consume_plan(rcclx, epoch)
        stats["consume_ns"].append(ns)
    else:
        op, counts = fallback_op, fallback_counts

    _execute_calls(
        rcclx, op, rank, active, dev, counts, graphs, stats, cfg.low_precision
    )


def _inject_timeout(rcclx, rank: int, is_active: bool) -> None:
    """Nobody publishes. Every consumer must fail on its own deadline."""
    if is_active:
        return
    try:
        consume_plan(rcclx, 0, timeout_ns=SHORT_TIMEOUT_NS)
    except RuntimeError:
        return  # expected
    raise AssertionError(f"Rank {rank}: consume should have timed out")


def _inject_crash(rcclx, rank: int, is_active: bool) -> None:
    """A helper never consumes.

    The publisher must still return: publish does not wait on consumers until the
    ring is full.
    """
    if rank == 0:
        publish_plan(rcclx, 0, OP_ALL_REDUCE, [BASE_COUNT])


def _inject_mismatch(rcclx, rank: int, is_active: bool) -> None:
    """The published shape differs from what the demo would otherwise have used.

    The point is that the consumer sees the PUBLISHED value, so a caller can
    detect the divergence instead of enqueueing a mismatched collective.
    """
    published = [BASE_COUNT // 2]
    if rank == 0:
        publish_plan(rcclx, 0, OP_ALL_REDUCE, published)
    if not is_active:
        _op, counts, _ns = consume_plan(rcclx, 0)
        if counts != published:
            raise AssertionError(
                f"Rank {rank}: expected published {published}, got {counts}"
            )


def _inject_abort(rcclx, rank: int, is_active: bool) -> None:
    """A consumer that gives up marks the segment aborted.

    What this exercises is the consumer half: the timeout is bounded and sets the
    abort flag as a side effect. The publisher half -- that a publish onto a
    poisoned segment is refused rather than quietly filling slots no consumer will
    read -- is asserted in the C++ suite
    (RelayControlBlockTest.PublishStopsOnAPoisonedSegment) rather than here,
    because proving it needs rank 0 to publish strictly AFTER the helper's timeout
    has landed, and this pass deliberately has no barrier to order the two with.
    """
    if is_active:
        return
    try:
        consume_plan(rcclx, 0, timeout_ns=SHORT_TIMEOUT_NS)
    except RuntimeError:
        pass  # sets the abort as a side effect


def _inject_overflow(rcclx, rank: int, is_active: bool) -> None:
    """More calls than NCCL_RELAY_CONTROL_MAX_CALLS.

    Must be refused at publish rather than truncated silently.
    """
    if rank != 0:
        return
    cap = int(os.environ.get("NCCL_RELAY_CONTROL_MAX_CALLS", "128"))
    try:
        publish_plan(rcclx, 0, OP_ALL_REDUCE, [BASE_COUNT] * (cap + 1))
    except RuntimeError:
        return  # expected
    raise AssertionError(f"Rank {rank}: over-capacity plan was accepted")


def _injectors() -> dict[str, Callable[..., None]]:
    """Explicit mapping rather than a registry decorator, which would need an
    import side effect to populate."""
    return {
        "timeout": _inject_timeout,
        "crash": _inject_crash,
        "mismatch": _inject_mismatch,
        "abort": _inject_abort,
        "overflow": _inject_overflow,
    }


def run_injection(rcclx, rank: int, active: list[int], mode: str) -> None:
    """Exercise one fault, and require it to be bounded rather than a hang.

    Only the control-plane handshake runs here -- deliberately no collectives. A
    fault that made one rank skip a collective its peers entered would hang the
    node, so the fault cases must not be mixed with relay calls. That is the same
    reason the C++ suite votes before entering a collective.
    """
    injector = _injectors().get(mode)
    if injector is None:
        raise ValueError(f"unknown injection mode {mode}")
    injector(rcclx, rank, rank in active)


def _print_timings(cfg: Config, stats: dict[str, list[int]]) -> None:
    def summarize(label: str, key: str, per: str) -> None:
        vals = stats[key]
        if not vals:
            return
        vals_sorted = sorted(vals)
        median = vals_sorted[len(vals_sorted) // 2]
        print(
            f"  {label:<28} n={len(vals):<4} "
            f"median={median / 1000.0:8.2f} us  "
            f"max={max(vals) / 1000.0:8.2f} us   ({per})"
        )

    print(f"\n=== host cost, control={cfg.control} (rank 0) ===")
    summarize("relay_control_publish", "publish_ns", "per forward")
    summarize("relay_control_consume", "consume_ns", "per forward")
    summarize("relay call enqueue", "enqueue_ns", "per call")
    print(
        "  The publish/consume figures are what replaces a per-call TCP-store\n"
        "  round trip; the enqueue figure is the active-rank host cost they sit\n"
        "  beside, for scale."
    )


def _worker(rank: int, world_size: int, port: int, cfg: Config) -> None:
    """One rank: create the RCCLX comm, then run every relay collective.

    Uses an explicit TCPStore (same pattern as bench_sharded_relay_perf) so
    RCCLX comm creation does not depend on dist._get_default_store(), which can
    hang when called from spawned child processes.
    """
    store = dist.TCPStore(
        host_name="localhost",
        port=port,
        world_size=world_size,
        is_master=(rank == 0),
        wait_for_workers=True,
    )

    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")

    # The control plane is built at commInitRank under this switch, so it must be
    # set before the comm exists. --control=none does not need it, and leaving it
    # unset there makes that mode a true baseline for today's path.
    if cfg.control == "shm":
        os.environ["NCCL_SHARDED_RELAY_MODE_ENABLE"] = "1"
    else:
        os.environ.pop("NCCL_SHARDED_RELAY_MODE_ENABLE", None)
    # new_comm reads rank/size from these; the store handles the unique-id exchange.
    os.environ["TORCHCOMM_RANK"] = str(rank)
    os.environ["TORCHCOMM_SIZE"] = str(world_size)

    # Before any comm exists, so a misconfigured run fails immediately on every
    # rank rather than after a partial forward.
    _assert_low_precision_sizes(cfg)

    stats: dict[str, list[int]] = {
        "publish_ns": [],
        "consume_ns": [],
        "enqueue_ns": [],
    }

    if cfg.inject != "none":
        comm = _make_comm(store, "inject")
        try:
            # Fault cases get their own pass over a fixed shape, with no
            # collectives, so a rank that fails cannot strand its peers. The width
            # does not matter here, since none of them reach a collective.
            if rank == 0:
                print(f"\n=== injection: {cfg.inject} ===")
            run_injection(
                comm.get_backend_impl(),
                rank,
                list(range(cfg.active_counts[-1])),
                cfg.inject,
            )
            if rank == 0:
                print(f"injection {cfg.inject}: bounded as expected")
        finally:
            comm.finalize()
        return

    # Exactly one width per process, so this comm has ONE active set for its whole
    # lifetime and no rank ever changes role. _spawn re-spawns for each width
    # instead of looping here.
    (A,) = cfg.active_counts
    comm = _make_comm(store, f"a{A}")
    try:
        _run_phase(comm.get_backend_impl(), rank, dev, A, cfg, stats)
    finally:
        comm.finalize()

    if rank == 0:
        _print_timings(cfg, stats)
        print(f"\nShardedRelayCollectives: A={A} demos completed")


def _run_phase(
    rcclx,
    rank: int,
    dev: torch.device,
    A: int,
    cfg: Config,
    stats: dict[str, list[int]],
) -> None:
    """Every collective at one active width, one forward at a time."""
    active = list(range(A))  # single group: first A ranks are active
    if rank == 0:
        print(
            f"\n=== single-group sharded relay, A={A} active {active}, "
            f"control={cfg.control}"
            f"{', low precision' if cfg.low_precision else ''} ==="
        )

    # Epochs are per communicator, because the control-plane segment is.
    epoch = 0
    graphs = _capture_all(rcclx, rank, active, dev, cfg) if cfg.graph else None
    for forward in range(cfg.forwards):
        op = RELAY_OPS[forward % len(RELAY_OPS)]
        counts = counts_for_forward(forward, cfg.calls_per_forward, base_count(cfg))
        if rank == 0:
            print(
                f"  forward {forward}: {OP_NAMES[op]} "
                f"{len(counts)} call(s) counts={counts}"
                f"{' [replay]' if cfg.graph else ''}"
            )
        if rank in active:
            run_active(rcclx, rank, active, dev, epoch, op, counts, cfg, stats, graphs)
        else:
            run_helper(rcclx, rank, active, dev, epoch, cfg, stats, op, counts, graphs)
        epoch += 1

    # Shutdown is an opcode, not a third entry point.
    if cfg.control != "shm":
        return
    if rank == 0:
        publish_plan(rcclx, epoch, OP_SHUTDOWN, [])
    elif rank >= A:
        op_code, _counts, _ns = consume_plan(rcclx, epoch)
        if op_code != OP_SHUTDOWN:
            raise AssertionError(f"Rank {rank}: expected shutdown, got {op_code}")


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("localhost", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _spawn(cfg: Config) -> None:
    """One process group per active width.

    Not a loop inside the worker. A comm is finalized before the next width
    starts, but the caching allocator recycles the finished width's blocks and the
    next comm's first collective can read them back, so two widths in one process
    return stale data from the first. Re-spawning also keeps every rank's role
    fixed for the life of its process, which is what a deployment looks like.
    """
    for active in cfg.active_counts:
        mp.spawn(
            _worker,
            args=(WORLD, _free_port(), replace(cfg, active_counts=(active,))),
            nprocs=WORLD,
            join=True,
        )


class ShardedRelayCollectivesTest(unittest.TestCase):
    """Spawns 8 ranks and runs every relay collective for A=2 and A=4."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA/ROCm not available")
        if torch.cuda.device_count() < WORLD:
            self.skipTest(f"needs {WORLD} GPUs, found {torch.cuda.device_count()}")

    def test_relay_collectives(self) -> None:
        """Every configuration, in one process and strictly one at a time.

        Deliberately ONE test method rather than five. Each configuration spawns
        WORLD ranks across every GPU on the box, so two configurations running
        concurrently oversubscribe all of them -- which shows up as GPU Hang
        aborts and, worse, as numerical mismatches rather than a clean failure.
        tpx is free to run separate test methods in parallel, so the only way to
        guarantee serialization is to keep them in one. subTest preserves
        per-configuration reporting.
        """
        cases = [
            ("control_plane", Config(control="shm")),
            ("no_control_plane", Config(control="none")),
            ("dynamic_shape", Config(control="shm", forwards=5, calls_per_forward=3)),
            ("graph_capture", Config(control="shm", graph=True)),
            # Low precision is a per-call argument, so it composes with every
            # other configuration rather than needing its own mode. Eager and
            # captured are both covered because the arena's bootstrap is not
            # capturable: the captured case only carries the wire format because
            # _capture_shape warms each shape up first.
            ("low_precision", Config(control="shm", low_precision=True)),
            (
                "low_precision_graph",
                Config(control="shm", graph=True, low_precision=True),
            ),
        ]
        cases += [
            # Fault cases never reach a collective, so one width covers them.
            (
                f"inject_{mode}",
                Config(
                    control="shm",
                    inject=mode,
                    active_counts=(ACTIVE_COUNTS[-1],),
                ),
            )
            for mode in INJECT_MODES
            if mode != "none"
        ]
        for label, cfg in cases:
            with self.subTest(case=label):
                _spawn(cfg)


def _parse_args(argv: list[str]) -> Config:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control", choices=("none", "shm"), default="shm")
    p.add_argument(
        "--forwards",
        type=int,
        default=len(RELAY_OPS),
        help="synthetic forwards per active width. One collective per forward, "
        f"cycling {', '.join(OP_NAMES[op] for op in RELAY_OPS)}, so fewer than "
        f"{len(RELAY_OPS)} skips the tail of that list.",
    )
    p.add_argument("--calls-per-forward", type=int, default=2)
    p.add_argument("--inject", choices=INJECT_MODES, default="none")
    p.add_argument(
        "--graph",
        action="store_true",
        help="capture each shape once and replay, instead of enqueueing eagerly",
    )
    p.add_argument(
        "--low-precision",
        action="store_true",
        help="request the fp8e4m3 wire format. Raises the per-call counts to "
        f"LP_BASE_COUNT ({LP_BASE_COUNT}) because the internal gate declines "
        "below a size crossover SILENTLY, so the small default counts would run "
        "the whole demo in full precision and still report success.",
    )
    a = p.parse_args(argv)
    return Config(
        control=a.control,
        forwards=a.forwards,
        calls_per_forward=a.calls_per_forward,
        inject=a.inject,
        graph=a.graph,
        low_precision=a.low_precision,
    )


if __name__ == "__main__":
    # Standalone when any example flag is given; otherwise the unittest entry so
    # the buck target keeps working unchanged.
    flags = (
        "--control",
        "--forwards",
        "--calls-per-forward",
        "--inject",
        "--graph",
        "--low-precision",
    )
    if any(arg.split("=")[0] in flags for arg in sys.argv[1:]):
        _spawn(_parse_args(sys.argv[1:]))
    else:
        unittest.main()
