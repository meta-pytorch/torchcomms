#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import unittest

import torch
import torch.distributed  # registers c10d ops
import torchcomms
from torchcomms.hooks import NanCheckHook


def _has_check_for_nan() -> bool:
    try:
        torch.ops.c10d.check_for_nan  # noqa: B018
        return True
    except AttributeError:
        return False


@unittest.skipUnless(_has_check_for_nan(), "requires c10d::check_for_nan op (nightly)")
class TestNanCheckHook(unittest.TestCase):
    def _create_comm(self, name: str) -> torchcomms.TorchComm:
        """Create a communicator using the dummy backend."""
        return torchcomms.new_comm("dummy", torch.device("cpu"), name=name)

    def test_clean_tensor_passes(self) -> None:
        """No error on clean tensors."""
        comm = self._create_comm("test_clean")
        hook = NanCheckHook()
        hook.register_with_comm(comm)

        tensor = torch.ones(10)
        comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, async_op=False)

        hook.unregister()
        comm.finalize()

    def test_nan_detected_in_input(self) -> None:
        """RuntimeError on NaN, error message includes op name."""
        comm = self._create_comm("test_nan_input")
        hook = NanCheckHook()
        hook.register_with_comm(comm)

        tensor = torch.tensor([1.0, float("nan"), 3.0])
        with self.assertRaises(RuntimeError) as ctx:
            comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, async_op=False)
        self.assertIn("all_reduce", str(ctx.exception))
        self.assertIn("test_nan_input", str(ctx.exception))

        hook.unregister()
        comm.finalize()

    def test_nan_in_broadcast(self) -> None:
        """NaN detected in broadcast input tensor."""
        comm = self._create_comm("test_nan_broadcast")
        hook = NanCheckHook()
        hook.register_with_comm(comm)

        tensor = torch.tensor([float("nan")])
        with self.assertRaises(RuntimeError) as ctx:
            comm.broadcast(tensor, root=0, async_op=False)
        self.assertIn("broadcast", str(ctx.exception))

        hook.unregister()
        comm.finalize()

    def test_inf_not_detected(self) -> None:
        """Inf does not trigger the check (check_for_nan only catches NaN)."""
        comm = self._create_comm("test_inf")
        hook = NanCheckHook()
        hook.register_with_comm(comm)

        tensor = torch.tensor([1.0, float("inf"), 3.0])
        # Should not raise — check_for_nan only detects NaN, not Inf
        comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, async_op=False)

        hook.unregister()
        comm.finalize()

    def test_non_float_tensor_passes(self) -> None:
        """Int tensors skip the check."""
        comm = self._create_comm("test_int")
        hook = NanCheckHook()
        hook.register_with_comm(comm)

        tensor = torch.tensor([1, 2, 3])
        # Should not raise even though we can't have NaN in int
        comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, async_op=False)

        hook.unregister()
        comm.finalize()

    def test_no_check_after_unregister(self) -> None:
        """NaN passes after unregister."""
        comm = self._create_comm("test_unreg")
        hook = NanCheckHook()
        hook.register_with_comm(comm)
        hook.unregister()

        tensor = torch.tensor([float("nan")])
        # Should not raise since hook was unregistered
        comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, async_op=False)

        comm.finalize()

    def test_check_outputs(self) -> None:
        """When check_outputs=True, output tensor NaN is detected."""
        comm = self._create_comm("test_check_out")
        hook = NanCheckHook(check_inputs=False, check_outputs=True)
        hook.register_with_comm(comm)

        # For all_gather_single, the output tensor is exposed in PreHookArgs
        output = torch.tensor([float("nan"), float("nan")])
        input_tensor = torch.tensor([1.0])
        with self.assertRaises(RuntimeError) as ctx:
            comm.all_gather_single(output, input_tensor, async_op=False)
        self.assertIn("output", str(ctx.exception))

        hook.unregister()
        comm.finalize()

    def test_barrier_no_tensors(self) -> None:
        """Barrier has no tensors, passes cleanly."""
        comm = self._create_comm("test_barrier")
        hook = NanCheckHook()
        hook.register_with_comm(comm)

        comm.barrier(async_op=False)

        hook.unregister()
        comm.finalize()

    def test_register_multiple_comms(self) -> None:
        """Works across multiple communicators."""
        comm1 = self._create_comm("test_multi_1")
        comm2 = self._create_comm("test_multi_2")
        hook = NanCheckHook()
        hook.register_with_comm(comm1)
        hook.register_with_comm(comm2)

        self.assertTrue(hook.is_enabled())

        # Clean tensors should pass on both
        tensor = torch.ones(10)
        comm1.all_reduce(tensor, torchcomms.ReduceOp.SUM, async_op=False)
        comm2.all_reduce(tensor, torchcomms.ReduceOp.SUM, async_op=False)

        # NaN should fail on either
        bad_tensor = torch.tensor([float("nan")])
        with self.assertRaises(RuntimeError) as ctx:
            comm1.all_reduce(bad_tensor, torchcomms.ReduceOp.SUM, async_op=False)
        self.assertIn("test_multi_1", str(ctx.exception))

        hook.unregister()
        self.assertFalse(hook.is_enabled())

        comm1.finalize()
        comm2.finalize()

    def test_is_enabled(self) -> None:
        """is_enabled returns correct state."""
        hook = NanCheckHook()
        self.assertFalse(hook.is_enabled())

        comm = self._create_comm("test_enabled")
        hook.register_with_comm(comm)
        self.assertTrue(hook.is_enabled())

        hook.unregister()
        self.assertFalse(hook.is_enabled())

        comm.finalize()


if __name__ == "__main__":
    unittest.main()
