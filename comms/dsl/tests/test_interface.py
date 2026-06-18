# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

"""GPU-free interface tests for the composable send/recv framework.

Asserts the *contract* (types, signatures, transport abstraction, reserved
stubs) without compiling or launching any kernel, so this runs in CI with no
GPU. The real end-to-end path is exercised by ``test_minimal_sendrecv``.
"""

import dataclasses
import inspect
import unittest
from typing import Any, cast

from comms.dsl.cute import ib_ops as cute_ib_ops

# Pure-Python cute submodules (no cutlass); importable GPU-free thanks to the
# lazy cute/__init__.py, so they are safe to import at module scope.
from comms.dsl.cute.ctx import Ctx as CuteCtx
from torch.utils._triton import has_triton


class ContractTest(unittest.TestCase):
    def test_ctx_fields(self) -> None:
        from comms.dsl import Ctx

        names = {f.name for f in dataclasses.fields(Ctx)}
        for expected in (
            "in_ptr",
            "out_ptr",
            "region_ptr",
            "tile_idx",
            "block_id",
            "flat_tid",
            "num_blocks",
            "peer",
            "recv_peer",
            "shard_idx",
            "in_strides",
            "out_strides",
            "extra",
        ):
            self.assertIn(expected, names)
        # Construct + extension point works.
        ctx = Ctx(in_ptr=1, extra={"scales": 7})
        self.assertEqual(ctx.in_ptr, 1)
        self.assertEqual(ctx.extra["scales"], 7)

    def test_hook_aliases_importable(self) -> None:
        from comms.dsl import ConsumeFn, ProduceFn

        self.assertIsNotNone(ProduceFn)
        self.assertIsNotNone(ConsumeFn)

    def test_exports(self) -> None:
        import comms.dsl as fw

        for name in (
            "Ctx",
            "NvlTransport",
            "nvl_rendezvous",
            "MeshTransport",
            "LinkKind",
        ):
            self.assertIn(name, fw.__all__)


class TransportTest(unittest.TestCase):
    def test_rendezvous_signature(self) -> None:
        from comms.dsl import nvl_rendezvous

        params = list(inspect.signature(nvl_rendezvous).parameters)
        for expected in ("group", "device", "per_peer_bytes"):
            self.assertIn(expected, params)

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
        # MeshTransport delegates max_blocks_per_peer to intra, so the
        # num_blocks guard still fires through a mesh.
        from comms.dsl import MeshTransport

        mesh = MeshTransport(intra=t)
        check_transfer(mesh, numel=512, dtype=torch.float32, num_blocks=2)
        with self.assertRaises(ValueError):
            check_transfer(mesh, numel=512, dtype=torch.float32, num_blocks=8)

    def test_ib_reserved(self) -> None:
        from comms.dsl import ib_rendezvous, IbTransport

        ib = IbTransport(world_size=8, per_peer_bytes=1024)
        with self.assertRaises(NotImplementedError):
            ib.endpoint(1, dtype=cast(Any, None))
        with self.assertRaises(NotImplementedError):
            ib_rendezvous(cast(Any, None), cast(Any, None), 1024)

    def test_mesh_routes_by_domain(self) -> None:
        from comms.dsl import IbTransport, LinkKind, MeshTransport, NvlTransport

        intra = NvlTransport(
            handle=cast(Any, None), world_size=8, local_rank=0, per_peer_bytes=1024
        )
        # No inter transport yet -> everything is NVLINK.
        mesh = MeshTransport(intra=intra)
        self.assertIs(mesh.link_kind(3), LinkKind.NVLINK)

        # With an inter transport + domain size 4: peers 0-3 NVLINK, 4-7 IB.
        mesh2 = MeshTransport(
            intra=intra,
            inter=IbTransport(world_size=8, per_peer_bytes=1024),
            local_domain_size=4,
        )
        self.assertIs(mesh2.link_kind(2), LinkKind.NVLINK)
        self.assertIs(mesh2.link_kind(5), LinkKind.IB)


@unittest.skipUnless(has_triton(), "Triton not available")
class TritonInterfaceTest(unittest.TestCase):
    def _arg_names(self, jit_fn: object) -> list[str]:
        names = getattr(jit_fn, "arg_names", None)
        if names is not None:
            return list(names)
        return list(inspect.signature(getattr(jit_fn, "fn", jit_fn)).parameters)

    def test_send_recv_are_jit(self) -> None:
        from comms.dsl.triton import recv_tiles as recv, send_tiles as send
        from triton.runtime.jit import JITFunction

        self.assertIsInstance(send, JITFunction)
        self.assertIsInstance(recv, JITFunction)
        for p in ("in_ptr", "produce", "put", "signal"):
            self.assertIn(p, self._arg_names(send))
        for p in ("out_ptr", "consume", "get", "wait"):
            self.assertIn(p, self._arg_names(recv))

    def test_nvl_and_ib_ops_present(self) -> None:
        from comms.dsl.triton import ib_ops, nvl_ops
        from triton.runtime.jit import JITFunction

        for ops in (nvl_ops, ib_ops):
            for name in ("put", "get", "signal", "wait"):
                self.assertIsInstance(getattr(ops, name), JITFunction)

    def test_hook_ctx_contract(self) -> None:
        # Hooks take a single Ctx aggregate; consume also gets the tile payload.
        from comms.dsl.triton import hooks
        from comms.dsl.triton.ctx import Ctx
        from triton.runtime.jit import JITFunction

        self.assertIsInstance(hooks.copy_produce, JITFunction)
        self.assertIsInstance(hooks.copy_consume, JITFunction)
        self.assertEqual(self._arg_names(hooks.copy_produce), ["ctx"])
        self.assertEqual(self._arg_names(hooks.copy_consume), ["ctx", "regs"])
        self.assertIsNotNone(Ctx)

    def test_hooks_present(self) -> None:
        from comms.dsl.triton import hooks
        from triton.runtime.jit import JITFunction

        for name in (
            "copy_produce",
            "copy_consume",
            "scale2_produce",
            "addone_consume",
        ):
            self.assertIsInstance(getattr(hooks, name), JITFunction)

    def test_launch_send_recv_hook_defaults(self) -> None:
        # The host launchers expose the hook seam as keyword-only params that
        # default to the identity copy hooks.
        from comms.dsl.triton import launch
        from comms.dsl.triton.hooks import copy_consume, copy_produce

        send_params = inspect.signature(launch.send).parameters
        recv_params = inspect.signature(launch.recv).parameters
        self.assertIs(send_params["produce"].default, copy_produce)
        self.assertIs(recv_params["consume"].default, copy_consume)

    def test_sendrecv_routes_hooks(self) -> None:
        # sendrecv routes produce -> send leg, consume -> recv leg, defaults
        # recv_peer to send_peer, and forwards the remaining launch kwargs to
        # both legs. Patch the leg launchers so this runs GPU-free.
        from unittest import mock

        from comms.dsl.triton import launch

        produce = object()
        consume = object()
        transport = cast(Any, object())
        send_buf = cast(Any, object())
        recv_buf = cast(Any, object())

        with (
            mock.patch.object(launch, "send") as send_mock,
            mock.patch.object(launch, "recv") as recv_mock,
        ):
            launch.sendrecv(
                transport,
                send_buf,
                recv_buf,
                send_peer=3,
                produce=produce,
                consume=consume,
                num_blocks=2,
            )

        send_mock.assert_called_once_with(
            transport, send_buf, 3, produce=produce, num_blocks=2
        )
        recv_mock.assert_called_once_with(
            transport, recv_buf, 3, consume=consume, num_blocks=2
        )


class CuteInterfaceTest(unittest.TestCase):
    def test_cute_ctx_fields(self) -> None:
        ctx = CuteCtx(part=1, atom=2)
        self.assertEqual(ctx.part, 1)
        self.assertEqual(ctx.atom, 2)

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

    def test_cute_backend_host_api(self) -> None:
        # The CuTe backend consumes the SAME host contract (transport + peer) as
        # Triton — only the device kernel differs. Importing it requires the
        # cutlass DSL, which may be unavailable in a GPU-less sandbox, so skip
        # rather than fail there.
        try:
            from comms.dsl.cute import recv, send, sendrecv
        except Exception as e:  # cutlass import / CUDA init may fail GPU-free
            self.skipTest(f"cutlass DSL not importable: {e}")

        for fn in (send, recv):
            params = list(inspect.signature(fn).parameters)
            self.assertIn("transport", params)
            self.assertIn("peer", params)
        for fn in (send, recv):
            params = list(inspect.signature(fn).parameters)
            self.assertIn("transport", params)
            self.assertIn("peer", params)
        self.assertIn("produce", inspect.signature(send).parameters)
        self.assertIn("consume", inspect.signature(recv).parameters)
        # sendrecv routes the disjoint hooks explicitly (produce->send,
        # consume->recv), so both are accepted keyword-only params.
        sr_params = list(inspect.signature(sendrecv).parameters)
        self.assertIn("produce", sr_params)
        self.assertIn("consume", sr_params)

        # Example CuTe hooks exist in the unified produce(ctx)/consume(ctx) form.
        from comms.dsl.cute import hooks as cute_hooks

        self.assertTrue(callable(cute_hooks.scale2_produce))
        self.assertTrue(callable(cute_hooks.addone_consume))
