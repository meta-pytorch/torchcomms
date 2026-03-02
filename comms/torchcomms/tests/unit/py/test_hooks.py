#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import unittest

import torch
import torchcomms
from torchcomms import OpName, PostHookArgs, PreHookArgs, RemovableHandle


class TestHooks(unittest.TestCase):
    def _create_comm(self, name: str) -> torchcomms.TorchComm:
        """Create a communicator using the dummy backend."""
        return torchcomms.new_comm("dummy", torch.device("cpu"), name=name)

    def test_register_pre_hook(self) -> None:
        """Test that pre-hooks are called before collective operations."""
        comm = self._create_comm("test_pre_hook")

        pre_hook_calls: list[OpName] = []

        def my_pre_hook(args: PreHookArgs) -> None:
            pre_hook_calls.append(args.name)

        handle = comm.register_pre_hook(my_pre_hook)
        self.assertIsInstance(handle, RemovableHandle)

        # Run a collective operation
        tensor = torch.ones(10)
        comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, async_op=False)

        # Verify hook was called
        self.assertEqual(len(pre_hook_calls), 1)
        self.assertEqual(pre_hook_calls[0], OpName.all_reduce)

        comm.finalize()

    def test_register_post_hook(self) -> None:
        """Test that post-hooks are called after collective operations."""
        comm = self._create_comm("test_post_hook")

        post_hook_calls: list[OpName] = []

        def my_post_hook(args: PostHookArgs) -> None:
            post_hook_calls.append(args.name)

        handle = comm.register_post_hook(my_post_hook)
        self.assertIsInstance(handle, RemovableHandle)

        # Run a collective operation
        tensor = torch.ones(10)
        comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, async_op=False)

        # Verify hook was called
        self.assertEqual(len(post_hook_calls), 1)
        self.assertEqual(post_hook_calls[0], OpName.all_reduce)

        comm.finalize()

    def test_hook_op_id_correlation(self) -> None:
        """Test that pre-hook and post-hook op_ids match for the same operation."""
        comm = self._create_comm("test_op_id")

        pre_op_ids: list[int] = []
        post_op_ids: list[int] = []

        def my_pre_hook(args: PreHookArgs) -> None:
            pre_op_ids.append(args.op_id)

        def my_post_hook(args: PostHookArgs) -> None:
            post_op_ids.append(args.op_id)

        comm.register_pre_hook(my_pre_hook)
        comm.register_post_hook(my_post_hook)

        # Run multiple operations
        tensor = torch.ones(10)
        comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, async_op=False)
        comm.barrier(async_op=False)

        # Verify op_ids match and increase
        self.assertEqual(len(pre_op_ids), 2)
        self.assertEqual(len(post_op_ids), 2)
        self.assertEqual(pre_op_ids[0], post_op_ids[0])
        self.assertEqual(pre_op_ids[1], post_op_ids[1])
        self.assertLess(pre_op_ids[0], pre_op_ids[1])

        comm.finalize()

    def test_hook_removal(self) -> None:
        """Test that hooks are not called after removal."""
        comm = self._create_comm("test_removal")

        call_count = 0

        def my_pre_hook(args: PreHookArgs) -> None:
            nonlocal call_count
            call_count += 1

        handle = comm.register_pre_hook(my_pre_hook)

        # First operation - hook should be called
        tensor = torch.ones(10)
        comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, async_op=False)
        self.assertEqual(call_count, 1)

        # Remove the hook
        handle.remove()

        # Second operation - hook should NOT be called
        comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, async_op=False)
        self.assertEqual(call_count, 1)

        comm.finalize()

    def test_multiple_hooks(self) -> None:
        """Test that multiple hooks are all called."""
        comm = self._create_comm("test_multi")

        hook1_calls = 0
        hook2_calls = 0

        def hook1(args: PreHookArgs) -> None:
            nonlocal hook1_calls
            hook1_calls += 1

        def hook2(args: PreHookArgs) -> None:
            nonlocal hook2_calls
            hook2_calls += 1

        comm.register_pre_hook(hook1)
        comm.register_pre_hook(hook2)

        # Run operation
        tensor = torch.ones(10)
        comm.all_reduce(tensor, torchcomms.ReduceOp.SUM, async_op=False)

        # Both hooks should be called
        self.assertEqual(hook1_calls, 1)
        self.assertEqual(hook2_calls, 1)

        comm.finalize()

    def test_pre_hook_args_properties(self) -> None:
        """Test that PreHookArgs has expected properties."""
        comm = self._create_comm("test_args")

        captured_args: list[PreHookArgs] = []

        def my_pre_hook(args: PreHookArgs) -> None:
            captured_args.append(args)

        comm.register_pre_hook(my_pre_hook)

        tensor = torch.ones(10)
        comm.broadcast(tensor, root=0, async_op=False)

        self.assertEqual(len(captured_args), 1)
        args = captured_args[0]
        self.assertEqual(args.name, OpName.broadcast)
        self.assertEqual(args.root, 0)
        self.assertIsInstance(args.op_id, int)
        self.assertFalse(args.async_op)

        comm.finalize()

    def test_register_abort_hook(self) -> None:
        """Test that abort hooks can be registered."""
        comm = self._create_comm("test_abort")

        abort_called = False

        def my_abort_hook() -> None:
            nonlocal abort_called
            abort_called = True

        handle = comm.register_abort_hook(my_abort_hook)
        self.assertIsInstance(handle, RemovableHandle)

        # We don't trigger abort in this test as it would terminate the process
        # Just verify registration works
        handle.remove()

        comm.finalize()


if __name__ == "__main__":
    unittest.main()
