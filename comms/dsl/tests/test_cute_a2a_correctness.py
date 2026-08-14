# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe
# Distributed correctness test: the worker runs under mp.spawn and touches untyped cutlass /
# symmetric-memory symbols that pyre cannot model, so strict typing adds no value here.

"""Correctness test for the fused CuTe all_to_all (vs dist.all_to_all_single).

Bit-exact gold check across a few sizes / block counts / dtypes, eager and under
CUDA-graph replay (the persistent-counter graph-safety contract). Covers fp32 + bf16, a
single-shot and a pipelined size, small and ``num_blocks == max_blocks_per_peer``. Reused-
transport graph replays are deliberately queued back-to-back with changing inputs and no
cross-rank host barrier, exercising the device-side HEAD/TAIL staging-reuse protocol. Also
covers all public full-staging fanouts and the bounded-ring channel schedule. Fanouts two
and four are covered when all eight required GPUs are present.
Skipped unless >=2 GPUs are present.
"""

import os
import time
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from comms.dsl.tests._dist_harness import _find_free_port, _golden, _make_input

# factor, blocks, dtype, elem bytes, primitive, staging bytes, threads, send threads, slots,
# unroll, peer fanout. The bounded-ring case forces more logical steps than physical slots
# and a partial final slot at every tested world size.
_CASES: tuple[
    tuple[int, int, torch.dtype, int, str, int, int, int, int, int, int], ...
] = (
    (4, 1, torch.float32, 4, "copy", 0, 0, 0, 0, 0, 0),
    (2048, 2, torch.float32, 4, "copy", 0, 0, 0, 0, 0, 0),
    (2 * 1024 * 1024, 4, torch.float32, 4, "copy", 0, 0, 0, 0, 0, 0),
    (2048, 32, torch.float32, 4, "copy", 0, 0, 0, 0, 0, 0),
    (2048, 2, torch.bfloat16, 2, "copy", 0, 0, 0, 0, 0, 0),
    (
        256 * 1024,
        4,
        torch.float32,
        4,
        "copy_channel_full",
        0,
        256,
        96,
        0,
        8,
        1,
    ),
    (
        512 * 1024 + 12,
        4,
        torch.float32,
        4,
        "copy_channel_full",
        0,
        1024,
        480,
        0,
        4,
        1,
    ),
    (
        256 * 1024,
        4,
        torch.float32,
        4,
        "copy_channel_full",
        0,
        1024,
        512,
        0,
        8,
        2,
    ),
    (
        256 * 1024,
        4,
        torch.float32,
        4,
        "copy_channel_full",
        0,
        1024,
        512,
        0,
        8,
        4,
    ),
    (
        1024 * 1024,
        4,
        torch.float32,
        4,
        "copy_channel_ring",
        1024 * 1024,
        256,
        160,
        3,
        4,
        0,
    ),
)


def _dtype_input(rank: int, numel: int, device: torch.device, dtype: torch.dtype):
    if dtype == torch.float32:
        return _make_input(rank, numel, device)  # integer-valued fp32, exact
    # bf16-exact per-rank, position-varying pattern (values 0..250 < 256); a pure copy
    # preserves it bit-exactly, so torch.equal is valid.
    idx = torch.arange(numel, device=device, dtype=torch.int64)
    return ((idx + rank * 131) % 251).to(dtype)


def _worker(rank: int, world_size: int, port: int, result_q) -> None:  # noqa: C901
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    previous_nt = os.environ.pop("A2A_CUTE_NT", None)
    try:
        device = torch.device(f"cuda:{rank}")
        group = dist.group.WORLD
        assert group is not None

        from comms.dsl import nvl_rendezvous
        from comms.dsl.cute.a2a.host import all_to_all
        from comms.dsl.cute.a2a.tuning import CuteA2AConfig

        ok = True
        cases = tuple((*case, 0) for case in _CASES)
        for (
            factor,
            nb,
            dtype,
            elem,
            primitive,
            staging_bytes,
            num_threads,
            send_threads,
            num_slots,
            unroll,
            peer_fanout,
            cluster,
        ) in cases:
            fixed_peer_schedule = peer_fanout > 1
            if fixed_peer_schedule and world_size != 8:
                continue
            numel = world_size * factor
            chunk = numel // world_size
            if fixed_peer_schedule:
                effective_nb = nb
            else:
                effective_nb = max(
                    1,
                    min(
                        nb,
                        torch.cuda.get_device_properties(device).multi_processor_count
                        // world_size,
                    ),
                )
            per_peer_bytes = max(16, staging_bytes or chunk * elem)
            t = nvl_rendezvous(group, device, per_peer_bytes=per_peer_bytes)
            inp = _dtype_input(rank, numel, device, dtype)
            gold = _golden(group, inp)
            out = torch.empty_like(inp)
            cfg = CuteA2AConfig(
                num_blocks=effective_nb,
                num_threads=num_threads,
                num_slots=num_slots,
                unroll=unroll,
                primitive=primitive,
                cluster=cluster,
                send_threads=send_threads,
                peer_fanout=peer_fanout,
            )

            all_to_all(t, out, inp, config=cfg)
            torch.cuda.synchronize(device)
            eager_ok = torch.equal(out, gold)
            if primitive != "copy":
                metadata = t._get_a2a_launch_metadata()
                eager_ok = eager_ok and metadata["cluster"] == 1
                eager_ok = eager_ok and metadata["peer_fanout"] == peer_fanout
                if peer_fanout > 1:
                    eager_ok = eager_ok and all(
                        (
                            metadata["grid_ctas"] == 32,
                            metadata["num_threads"] == 1024,
                            metadata["send_threads"] == 512,
                        )
                    )
            # Graph replay with distinct data. Snapshot each output on the same stream, but
            # queue every replay before synchronizing so rank drift can exercise HEAD credit.
            buf = inp.clone()

            def fn(out=out, buf=buf, t=t, cfg=cfg):
                all_to_all(t, out, buf, config=cfg)

            fn()
            torch.cuda.synchronize(device)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            replay_count = max(3, num_slots + 2)
            payloads = [
                _dtype_input(rank + 11 * (i + 1), numel, device, dtype)
                for i in range(replay_count)
            ]
            golds = [_golden(group, payload) for payload in payloads]
            snapshots = [torch.empty_like(out) for _ in payloads]
            for epoch, (payload, snapshot) in enumerate(zip(payloads, snapshots)):
                # Best-effort host-side perturbation only; correctness does not depend on
                # the scheduler honoring the exact sleep duration.
                if rank == epoch % world_size:
                    time.sleep(0.001)
                buf.copy_(payload)
                g.replay()
                snapshot.copy_(out)
            torch.cuda.synchronize(device)
            graph_ok = all(
                torch.equal(snapshot, expected)
                for snapshot, expected in zip(snapshots, golds)
            )
            ok = ok and eager_ok and graph_ok
            if rank == 0:
                print(
                    f"  numel={numel} nb={effective_nb} dtype={dtype} primitive={primitive}: "
                    f"eager={'ok' if eager_ok else 'FAIL'} "
                    f"graph={'ok' if graph_ok else 'FAIL'}",
                    flush=True,
                )
            dist.barrier(group)

        result_q.put((rank, ok))
    finally:
        if previous_nt is not None:
            os.environ["A2A_CUTE_NT"] = previous_nt
        dist.barrier()
        dist.destroy_process_group()


class A2ACuteTest(unittest.TestCase):
    def test_cute_a2a(self) -> None:
        device_count = torch.cuda.device_count()
        if device_count < 2:
            self.skipTest("needs >=2 GPUs")
        for ws in (
            candidate for candidate in (2, 3, 4, 8) if candidate <= device_count
        ):
            with self.subTest(world_size=ws):
                ctx = mp.get_context("spawn")
                result_q = ctx.SimpleQueue()
                mp.spawn(
                    _worker,
                    args=(ws, _find_free_port(), result_q),
                    nprocs=ws,
                    join=True,
                )
                results = sorted(result_q.get() for _ in range(ws))
                self.assertEqual([rank for rank, _ in results], list(range(ws)))
                self.assertTrue(
                    all(ok for _, ok in results), f"per-rank pass flags: {results}"
                )
