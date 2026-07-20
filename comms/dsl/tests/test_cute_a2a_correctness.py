# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe
# Distributed correctness test: the worker runs under mp.spawn and touches untyped cutlass /
# symmetric-memory symbols that pyre cannot model, so strict typing adds no value here.

"""Correctness test for the fused CuTe all_to_all (vs dist.all_to_all_single).

Bit-exact gold check across a few sizes / block counts / dtypes, eager and under
CUDA-graph replay (the persistent-counter graph-safety contract). Covers fp32 + bf16, a
single-shot and a pipelined size, small and ``num_blocks == max_blocks_per_peer``. HEAD
slot-free credits protect free-running transport reuse across graph replays. Skipped unless
>=2 GPUs are present.
"""

import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from comms.dsl.tests._dist_harness import _find_free_port, _golden, _make_input

# (numel factor per rank, num_blocks, dtype, elem_bytes): tiny (scalar-ish), small multi-
# block, a pipelined 8 MiB fp32 chunk (num_slots > 1), num_blocks == mbp (=32) boundary,
# and the bf16 path.
_CASES: tuple[tuple[int, int, torch.dtype, int], ...] = (
    (4, 1, torch.float32, 4),
    (2048, 2, torch.float32, 4),
    (2 * 1024 * 1024, 4, torch.float32, 4),
    (2048, 32, torch.float32, 4),
    (2048, 2, torch.bfloat16, 2),
)


def _dtype_input(rank: int, numel: int, device: torch.device, dtype: torch.dtype):
    if dtype == torch.float32:
        return _make_input(rank, numel, device)  # integer-valued fp32, exact
    # bf16-exact per-rank, position-varying pattern (values 0..250 < 256); a pure copy
    # preserves it bit-exactly, so torch.equal is valid.
    idx = torch.arange(numel, device=device, dtype=torch.int64)
    return ((idx + rank * 131) % 251).to(dtype)


def _worker(rank: int, world_size: int, port: int, result_q) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        device = torch.device(f"cuda:{rank}")
        group = dist.group.WORLD
        assert group is not None

        from comms.dsl import nvl_rendezvous
        from comms.dsl.cute.a2a.host import all_to_all
        from comms.dsl.cute.a2a.tuning import CuteA2AConfig

        ok = True
        for factor, nb, dtype, elem in _CASES:
            numel = world_size * factor
            chunk = numel // world_size
            t = nvl_rendezvous(group, device, per_peer_bytes=chunk * elem)
            inp = _dtype_input(rank, numel, device, dtype)
            gold = _golden(group, inp)
            out = torch.empty_like(inp)

            all_to_all(t, out, inp, config=CuteA2AConfig(num_blocks=nb))
            torch.cuda.synchronize(device)
            eager_ok = torch.equal(out, gold)

            # Graph replay with distinct data exercises persistent counter progression and
            # HEAD credit protection without a cross-rank barrier between replays.
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
                inp2 = _dtype_input(rank + 11 * (i + 1), numel, device, dtype)
                buf.copy_(inp2)
                out.zero_()
                g.replay()
                torch.cuda.synchronize(device)
                graph_ok = graph_ok and torch.equal(out, _golden(group, inp2))
            ok = ok and eager_ok and graph_ok
            if rank == 0:
                print(
                    f"  numel={numel} nb={nb} dtype={dtype}: "
                    f"eager={'ok' if eager_ok else 'FAIL'} "
                    f"graph={'ok' if graph_ok else 'FAIL'}",
                    flush=True,
                )
            dist.barrier(group)

        result_q.put((rank, ok))
    finally:
        dist.barrier()
        dist.destroy_process_group()


class A2ACuteTest(unittest.TestCase):
    def test_cute_a2a(self) -> None:
        if torch.cuda.device_count() < 2:
            self.skipTest("needs >=2 GPUs")
        ws = min(torch.cuda.device_count(), 4)
        ctx = mp.get_context("spawn")
        result_q = ctx.SimpleQueue()
        mp.spawn(_worker, args=(ws, _find_free_port(), result_q), nprocs=ws, join=True)
        results = sorted(result_q.get() for _ in range(ws))
        self.assertEqual([r for r, _ in results], list(range(ws)))
        self.assertTrue(all(ok for _, ok in results), f"per-rank pass flags: {results}")
