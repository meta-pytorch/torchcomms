#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Basic usage examples for the RCCLX sharded-relay collectives.

The sharded relay is for 2D sparse parallelism: within a comm, a few "active"
ranks perform a logical collective while the remaining idle GPUs act as
passthrough "helpers" that relay sharded chunks (eliminating XGMI link
contention on MI300x/MI350x). These examples use a single active group on an
8-GPU node, for both A=2 (2 active + 6 helpers) and A=4 (4 active + 4 helpers):

    A=2  ->  active ranks {0, 1},        helpers {2, 3, 4, 5, 6, 7}
    A=4  ->  active ranks {0, 1, 2, 3},  helpers {4, 5, 6, 7}

The relay methods live on the RCCLX backend, reached via
`comm.get_backend_impl()`. Every rank in the comm (active AND helper) must call
each collective. Active ranks pass their real tensors; helper ranks pass a
single 1-element placeholder tensor — the C++ kernel stages helpers into its
own internal scratch and never reads/writes the placeholder.

Buffer contract per active rank (count = per-group element count):
    all_reduce      : tensor = count            (in-place)
    reduce_scatter  : input  = A x count -> output = count
    all_gather      : input  = count       -> output = A x count
    all_to_all      : input  = A x count -> output = A x count (distinct)

Self-contained: this is a python_unittest that spawns 8 ranks with mp.spawn and
an explicit TCPStore (mirroring bench_sharded_relay_perf), so it builds in-place
(no standalone-PAR packaging). It uses only torch + torchcomms (no torchrec /
caffe2 deps), so the same script also runs standalone against an rcclx wheel:

    buck2 test @mode/opt-amd-gpu -m rocm70 -m rcclx_dev \\
        //comms/torchcomms/rcclx/examples:ShardedRelayCollectives

    # against a wheel venv (self-spawns 8 procs, needs 8 GPUs):
    /path/to/venv/bin/python ShardedRelayCollectives.py

"""

import os
import socket
import unittest
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchcomms import new_comm, ReduceOp

WORLD = 8  # examples assume an 8-GPU node
ACTIVE_COUNTS = (2, 4)  # single-group sizes to demonstrate
COUNT = 1024  # per-group element count


def _placeholder(dev: torch.device) -> torch.Tensor:
    """1-element helper slot: the kernel uses internal scratch, ignores this."""
    return torch.empty(1, dtype=torch.float32, device=dev)


def _check(rank: int, name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    ok = torch.allclose(actual, expected)
    print(f"Rank {rank}: {name}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise AssertionError(f"Rank {rank}: {name} mismatch")


def demo_all_reduce(
    rcclx: Any, rank: int, active: list[int], dev: torch.device
) -> None:
    """SUM allreduce over the active group; helpers relay the chunks."""
    A = len(active)
    if rank in active:
        ai = active.index(rank)
        tensor = torch.full((COUNT,), float(ai + 1), dtype=torch.float32, device=dev)
    else:
        tensor = _placeholder(dev)  # helper slot

    # In-place (output_tensors omitted); tensors is one entry per group.
    rcclx.sharded_relay_multi_group_all_reduce(
        [tensor], ReduceOp.SUM, [active], [COUNT]
    )
    torch.cuda.current_stream().synchronize()

    if rank in active:
        expected = torch.full_like(tensor, float(sum(a + 1 for a in range(A))))
        _check(rank, f"all_reduce A={A}", tensor, expected)


def demo_reduce_scatter(
    rcclx: Any, rank: int, active: list[int], dev: torch.device
) -> None:
    """SUM reduce-scatter: active input block[i] is destined for active index i."""
    A = len(active)
    if rank in active:
        ai = active.index(rank)
        # block[i] value encodes both the sender (ai) and the destination (i).
        inp = torch.empty(A * COUNT, dtype=torch.float32, device=dev)
        for i in range(A):
            inp[i * COUNT : (i + 1) * COUNT] = float((ai + 1) + 100 * (i + 1))
        out = torch.empty(COUNT, dtype=torch.float32, device=dev)
    else:
        inp = _placeholder(dev)  # helper slot (input)
        out = inp  # helper slot (output): same placeholder

    rcclx.sharded_relay_multi_group_reduce_scatter(
        [inp], [out], ReduceOp.SUM, [active], [COUNT]
    )
    torch.cuda.current_stream().synchronize()

    if rank in active:
        ai = active.index(rank)
        # out = sum over senders a of block[ai] = sum_a[(a+1) + 100*(ai+1)]
        rank_sum = sum(a + 1 for a in range(A))
        expected = torch.full_like(out, float(rank_sum + 100 * A * (ai + 1)))
        _check(rank, f"reduce_scatter A={A}", out, expected)


def demo_all_gather(
    rcclx: Any, rank: int, active: list[int], dev: torch.device
) -> None:
    """All-gather: active index i contributes `count` elements of value (i+1)."""
    A = len(active)
    if rank in active:
        ai = active.index(rank)
        inp = torch.full((COUNT,), float(ai + 1), dtype=torch.float32, device=dev)
        out = torch.empty(A * COUNT, dtype=torch.float32, device=dev)
    else:
        inp = _placeholder(dev)  # helper slot (input)
        out = inp  # helper slot (output): same placeholder

    rcclx.sharded_relay_multi_group_all_gather([inp], [out], [active], [COUNT])
    torch.cuda.current_stream().synchronize()

    if rank in active:
        # out[i*count:(i+1)*count] == (i+1), gathered from active index i.
        expected = torch.empty(A * COUNT, dtype=torch.float32, device=dev)
        for i in range(A):
            expected[i * COUNT : (i + 1) * COUNT] = float(i + 1)
        _check(rank, f"all_gather A={A}", out, expected)


def demo_all_to_all(
    rcclx: Any, rank: int, active: list[int], dev: torch.device
) -> None:
    """All-to-all: active index i sends segment j to active index j (out-of-place)."""
    A = len(active)
    if rank in active:
        ai = active.index(rank)
        # send segment j encodes sender (ai) and destination (j).
        inp = torch.empty(A * COUNT, dtype=torch.float32, device=dev)
        for j in range(A):
            inp[j * COUNT : (j + 1) * COUNT] = float((ai + 1) * 10 + (j + 1))
        out = torch.empty(A * COUNT, dtype=torch.float32, device=dev)  # distinct
    else:
        inp = _placeholder(dev)  # helper slot (input)
        out = inp  # helper slot (output): same placeholder

    rcclx.sharded_relay_multi_group_all_to_all([inp], [out], [active], [COUNT])
    torch.cuda.current_stream().synchronize()

    if rank in active:
        ai = active.index(rank)
        # recv segment from sender sp = value (sp+1)*10 + (my index + 1)
        expected = torch.empty(A * COUNT, dtype=torch.float32, device=dev)
        for sp in range(A):
            expected[sp * COUNT : (sp + 1) * COUNT] = float((sp + 1) * 10 + (ai + 1))
        _check(rank, f"all_to_all A={A}", out, expected)


def _worker(rank: int, world_size: int, port: int) -> None:
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

    # new_comm reads rank/size from these; the store handles the unique-id exchange.
    os.environ["TORCHCOMM_RANK"] = str(rank)
    os.environ["TORCHCOMM_SIZE"] = str(world_size)
    comm = new_comm(
        "rcclx",
        torch.device("hip"),
        name="relay_demo",
        store=dist.PrefixStore("relay_demo", store),
    )
    rcclx = comm.get_backend_impl()  # the RCCLX backend that exposes the relay methods

    try:
        for A in ACTIVE_COUNTS:
            active = list(range(A))  # single group: first A ranks are active
            if rank == 0:
                print(f"\n=== single-group sharded relay, A={A} active {active} ===")
            demo_all_reduce(rcclx, rank, active, dev)
            demo_reduce_scatter(rcclx, rank, active, dev)
            demo_all_gather(rcclx, rank, active, dev)
            demo_all_to_all(rcclx, rank, active, dev)
    finally:
        comm.finalize()

    if rank == 0:
        print("\nShardedRelayCollectives: all demos completed")


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("localhost", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class ShardedRelayCollectivesTest(unittest.TestCase):
    """Spawns 8 ranks and runs every relay collective for A=2 and A=4."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA/ROCm not available")
        if torch.cuda.device_count() < WORLD:
            self.skipTest(f"needs {WORLD} GPUs, found {torch.cuda.device_count()}")

    def test_relay_collectives(self) -> None:
        mp.spawn(_worker, args=(WORLD, _free_port()), nprocs=WORLD, join=True)


if __name__ == "__main__":
    unittest.main()
