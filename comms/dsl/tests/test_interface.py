# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe
# Contract/interface test: touches untyped cute symbols and ``cast(Any, ...)`` stubs that
# pyre cannot model, so strict typing adds no value here.

"""GPU-free interface tests for the composable send/recv framework.

Asserts the *contract* (transport abstraction, reserved stubs, CuTe ctx fields) without
compiling or launching any kernel, so this runs in CI with no GPU. The real end-to-end path
is exercised by the distributed ``test_cute_*`` suites.
"""

import unittest
from typing import Any, cast

from comms.dsl.cute import ib_ops as cute_ib_ops

# Pure-Python cute submodules (no cutlass); importable GPU-free thanks to the
# lazy cute/__init__.py, so they are safe to import at module scope.
from comms.dsl.cute.ctx import Ctx as CuteCtx


class TransportTest(unittest.TestCase):
    def test_nvl_link_kind(self) -> None:
        from comms.dsl import LinkKind, NvlTransport

        # handle is not touched by link_kind, so we can build this GPU-free.
        t = NvlTransport(
            handle=cast(Any, None), world_size=4, local_rank=0, per_peer_bytes=1024
        )
        self.assertIs(t.link_kind(1), LinkKind.NVLINK)

    def test_check_transfer_guards(self) -> None:
        import torch
        from comms.dsl import check_transfer, NvlTransport

        # per_peer_bytes=4096, fp32 -> 1024 elems per peer; 4 signal slots.
        t = NvlTransport(
            handle=cast(Any, None),
            world_size=2,
            local_rank=0,
            per_peer_bytes=4096,
            max_blocks_per_peer=4,
        )
        # Valid transfer: no raise.
        check_transfer(t, numel=512, dtype=torch.float32, num_blocks=2)
        # numel exceeds per-peer capacity -> raise (would corrupt the next peer).
        with self.assertRaises(ValueError):
            check_transfer(t, numel=2000, dtype=torch.float32, num_blocks=2)
        # num_blocks exceeds signal slots -> raise (would OOB-write the pad).
        with self.assertRaises(ValueError):
            check_transfer(t, numel=512, dtype=torch.float32, num_blocks=8)
        # num_blocks must be >= 1.
        with self.assertRaises(ValueError):
            check_transfer(t, numel=512, dtype=torch.float32, num_blocks=0)

    def test_step_state_cache_field(self) -> None:
        # Lock the contract: _step_state_cache is a DECLARED dataclass field defaulting to None
        # (eager allocation happens at rendezvous, not at construction), so a refactor that drops
        # the field or changes its default is caught here. step_state()'s runtime behavior (lazy
        # alloc + persistent per-(peer,block) counters) needs a device and is covered by the GPU
        # send/recv + graph-replay tests.
        import dataclasses

        from comms.dsl import NvlTransport

        defaults = {f.name: f.default for f in dataclasses.fields(NvlTransport)}
        self.assertIn("_step_state_cache", defaults)
        self.assertIsNone(defaults["_step_state_cache"])
        t = NvlTransport(
            handle=cast(Any, None),
            world_size=2,
            local_rank=0,
            per_peer_bytes=1024,
        )
        self.assertIsNone(t._step_state_cache)
        self.assertTrue(callable(t.step_state))


class CuteInterfaceTest(unittest.TestCase):
    def test_cute_ctx_contract(self) -> None:
        # Lock the STABLE hook contract: the read-only facts default to None (a hook may branch on
        # ctx.peer/coord being unset), and `cols` is DERIVED, not stored -- chunk // rows when
        # rows > 0, else the whole chunk. The derivation is the only behavior on Ctx; asserting it
        # (not just field storage) guards the shape math every layout hook relies on.
        ctx = CuteCtx(part=1, atom=2)
        self.assertEqual((ctx.part, ctx.atom), (1, 2))
        self.assertIsNone(ctx.coord)
        self.assertIsNone(ctx.peer)
        self.assertIsNone(ctx.rank)
        self.assertIsNone(ctx.world_size)
        # cols derivation: rows>0 partitions the chunk; rows==0 (plain a2a) leaves it whole.
        self.assertEqual(CuteCtx(part=0, atom=0, rows=4, chunk=32).cols, 8)
        self.assertEqual(CuteCtx(part=0, atom=0, rows=0, chunk=32).cols, 32)

    def test_cute_ib_ops_reserved(self) -> None:
        # The four reserved IB transport ops must raise NotImplementedError until
        # the IB stack wires them.
        with self.assertRaises(NotImplementedError):
            cute_ib_ops.put(None, None, None)
        with self.assertRaises(NotImplementedError):
            cute_ib_ops.get(None, None)
        with self.assertRaises(NotImplementedError):
            cute_ib_ops.signal(None, None)
        with self.assertRaises(NotImplementedError):
            cute_ib_ops.wait(None, None)
