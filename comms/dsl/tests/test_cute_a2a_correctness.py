# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe
# Distributed correctness test: the worker runs under mp.spawn and touches untyped cutlass /
# symmetric-memory symbols that pyre cannot model, so strict typing adds no value here.

"""Correctness test for the fused CuTe all_to_all (vs dist.all_to_all_single).

Bit-exact gold check across a couple of sizes + block counts, eager and under
CUDA-graph replay (the persistent-counter graph-safety contract). Skipped unless
>=2 GPUs are present.
"""

import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from comms.dsl.tests._dist_harness import _find_free_port, _golden, _make_input


def _worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")
    group = dist.group.WORLD
    assert group is not None

    from comms.dsl import nvl_rendezvous
    from comms.dsl.cute.a2a.host import all_to_all
    from comms.dsl.cute.a2a.tuning import CuteA2AConfig

    ok = True
    # Adaptive (num_threads, vec) covers any chunk: tiny (chunk=2 -> scalar-ish),
    # odd/non-power-of-2 (chunk=100 -> vec4), and large vectorized chunks.
    for numel, nb in [
        (world_size * 2, 1),
        (world_size * 100, 1),
        (world_size * 2048, 2),
        (world_size * 16384, 4),
        # Multi-slot pipeline coverage (chunk >= _MIN_PIPELINE_CHUNK_BYTES so
        # _pick_slots returns num_slots > 1): exercises the credit-ring send/drain
        # overlap -- the interleaved _recv_slot(s-1) and the deep (8-slot) path that
        # the small sizes above (single-shot, num_slots=1) never reach.
        (world_size * 2 * 1024 * 1024, 4),  # 8MB fp32 chunk -> ~4 slots
        (world_size * 16 * 1024 * 1024, 4),  # 64MB fp32 chunk -> ~8 slots (deep)
    ]:
        chunk = numel // world_size
        t = nvl_rendezvous(group, device, per_peer_bytes=chunk * 4)
        inp = _make_input(rank, numel, device)
        gold = _golden(group, inp)
        out = torch.empty_like(inp)

        all_to_all(t, out, inp, config=CuteA2AConfig(num_blocks=nb))
        torch.cuda.synchronize(device)
        eager_ok = torch.equal(out, gold)

        # Graph replay with distinct data (persistent-counter graph safety).
        buf = inp.clone()

        def fn(out=out, buf=buf, t=t, nb=nb):
            all_to_all(t, out, buf, config=CuteA2AConfig(num_blocks=nb))

        fn()
        torch.cuda.synchronize(device)
        dist.barrier(group)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        graph_ok = True
        for i in range(3):
            inp2 = _make_input(rank + 11 * (i + 1), numel, device)
            buf.copy_(inp2)
            out.zero_()
            g.replay()
            torch.cuda.synchronize(device)
            graph_ok = graph_ok and torch.equal(out, _golden(group, inp2))
        ok = ok and eager_ok and graph_ok
        if rank == 0:
            print(
                f"  numel={numel} nb={nb}: eager={'ok' if eager_ok else 'FAIL'} "
                f"graph={'ok' if graph_ok else 'FAIL'}",
                flush=True,
            )
        dist.barrier(group)

    status = torch.tensor([1 if ok else 0], dtype=torch.int32, device=device)
    dist.all_reduce(status, op=dist.ReduceOp.MIN, group=group)
    dist.destroy_process_group()
    if not bool(status.item()):
        raise RuntimeError(f"rank {rank}: cute a2a correctness failed")


class A2ACuteTest(unittest.TestCase):
    def test_cute_a2a(self) -> None:
        if torch.cuda.device_count() < 2:
            self.skipTest("needs >=2 GPUs")
        ws = min(torch.cuda.device_count(), 4)
        # mp.spawn(join=True) re-raises any worker's RuntimeError (the _worker correctness gate)
        # into this process. The success-flag assertion makes the pass condition explicit: it is
        # only reached (and only True) when every rank returned without raising.
        all_ranks_passed = False
        mp.spawn(_worker, args=(ws, _find_free_port()), nprocs=ws, join=True)
        all_ranks_passed = True
        self.assertTrue(all_ranks_passed, "a rank failed the cute a2a correctness gate")
