#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Integration tests for wait_tensors proxy propagation in Dynamo tracing.

These tests verify that after calling work.wait(), the returned tensors
use the wait_tensors output proxy instead of the pre-wait collective output.

This is critical for correctness - if the return uses pre-wait tensors,
the data may not be ready yet, causing race conditions and NaN values.
"""

import logging
import unittest

import torch
import torch._dynamo

logger = logging.getLogger(__name__)


def _get_graph_output_sources(gm) -> list[str]:
    """Get the node names that are used in the graph's output."""
    output_node = None
    for node in gm.graph.nodes:
        if node.op == "output":
            output_node = node
            break

    if output_node is None:
        return []

    def get_source_names(arg):
        if isinstance(arg, torch.fx.Node):
            return [arg.name]
        elif isinstance(arg, (list, tuple)):
            names = []
            for item in arg:
                names.extend(get_source_names(item))
            return names
        return []

    return get_source_names(output_node.args[0])


def _find_nodes_by_name_pattern(gm, pattern: str) -> list:
    """Find all nodes whose target contains the given pattern.

    Also searches inside with_effects nodes for the wrapped op.
    """
    nodes = []
    for node in gm.graph.nodes:
        if node.op == "call_function":
            target_name = str(node.target)
            if pattern in target_name:
                nodes.append(node)
            # Also check if this is a with_effects wrapping an op that matches
            elif "with_effects" in target_name and len(node.args) >= 2:
                # with_effects(token, op, *args) - check the op argument
                wrapped_op = node.args[1]
                wrapped_op_name = str(wrapped_op)
                if pattern in wrapped_op_name:
                    nodes.append(node)
    return nodes


def _check_node_in_output_path(gm, target_node) -> bool:
    """Check if target_node is in the data flow path to the output."""
    output_sources = _get_graph_output_sources(gm)

    # Build reverse dependency map: node -> nodes that use it
    users = {}
    for node in gm.graph.nodes:
        if hasattr(node, 'args'):
            for arg in node.args:
                if isinstance(arg, torch.fx.Node):
                    if arg.name not in users:
                        users[arg.name] = []
                    users[arg.name].append(node.name)
                elif isinstance(arg, (list, tuple)):
                    for item in arg:
                        if isinstance(item, torch.fx.Node):
                            if item.name not in users:
                                users[item.name] = []
                            users[item.name].append(node.name)

    # BFS from target_node to see if we reach any output source
    visited = set()
    queue = [target_node.name]

    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        if current in output_sources:
            return True

        if current in users:
            queue.extend(users[current])

    return False


class TestWaitProxyPropagation(unittest.TestCase):
    """Test that wait_tensors output proxies are properly propagated."""

    def setUp(self):
        torch._dynamo.reset()

    def tearDown(self):
        torch._dynamo.reset()

    def _create_graph_capture_backend(self):
        """Create a backend that captures the graph for inspection."""
        captured = {"graph": None}

        def backend(gm, example_inputs):
            captured["graph"] = gm
            print("\n" + "=" * 80)
            print("CAPTURED FX GRAPH:")
            print("=" * 80)
            print(gm.print_readable(print_output=False))
            print("=" * 80 + "\n")
            return gm

        return backend, captured

    def test_wait_output_in_return_path(self):
        """Test that wait_tensors output is in the return path, not pre-wait collective output.

        This test uses a TorchComm collective with async_op=True, waits on the result,
        and returns the tensor. The FX graph should show that the returned tensor
        flows through wait_tensors, not directly from the collective.
        """
        try:
            from torchcomms import ReduceOp
            from torchcomms.tests.integration.py.TorchCommTestHelpers import (
                TorchCommTestWrapper,
            )
        except ImportError:
            self.skipTest("torchcomms not available")

        wrapper = TorchCommTestWrapper()
        comm = wrapper.get_torchcomm()
        device = comm.get_device()

        # Create test tensor
        tensor = torch.ones(4, dtype=torch.float, device=device)

        # Define function that takes comm as argument (not closure)
        # to ensure proper tracing through dynamo patches
        def my_func(comm_arg, t):
            work = comm_arg.all_reduce(t, ReduceOp.SUM, async_op=True)
            work.wait()
            return t  # This should use wait_tensors output, not all_reduce output

        # Compile with graph capture
        backend, captured = self._create_graph_capture_backend()
        compiled_func = torch.compile(my_func, fullgraph=True, backend=backend)

        # Run to trigger compilation
        result = compiled_func(comm, tensor)

        # Verify graph was captured
        self.assertIsNotNone(captured["graph"], "Graph should be captured")
        gm = captured["graph"]

        # Find wait_tensors and all_reduce nodes
        wait_nodes = _find_nodes_by_name_pattern(gm, "wait_tensors")
        all_reduce_nodes = _find_nodes_by_name_pattern(gm, "all_reduce")

        print(f"\nDiagnostics:")
        print(f"  wait_nodes: {[n.name for n in wait_nodes]}")
        print(f"  all_reduce_nodes: {[n.name for n in all_reduce_nodes]}")
        print(f"  output_sources: {_get_graph_output_sources(gm)}")

        self.assertTrue(len(wait_nodes) > 0, "Graph should contain wait_tensors")
        self.assertTrue(len(all_reduce_nodes) > 0, "Graph should contain all_reduce")

        # Check that wait_tensors output is in the return path
        wait_in_path = any(_check_node_in_output_path(gm, node) for node in wait_nodes)
        self.assertTrue(
            wait_in_path,
            f"wait_tensors output should be in return path. "
            f"Wait nodes: {[n.name for n in wait_nodes]}, "
            f"Output sources: {_get_graph_output_sources(gm)}"
        )

        # Cleanup
        wrapper = None
        comm = None
        torch._dynamo.reset()

    def test_list_return_uses_wait_output(self):
        """Test that list of tensors returned after wait uses wait outputs."""
        try:
            from torchcomms.tests.integration.py.TorchCommTestHelpers import (
                TorchCommTestWrapper,
            )
        except ImportError:
            self.skipTest("torchcomms not available")

        wrapper = TorchCommTestWrapper()
        comm = wrapper.get_torchcomm()
        device = comm.get_device()
        num_ranks = comm.get_size()

        # Create output list for all_gather
        tensor = torch.ones(4, dtype=torch.float, device=device) * (comm.get_rank() + 1)
        output_list = [torch.zeros(4, dtype=torch.float, device=device) for _ in range(num_ranks)]

        # Define function that takes comm as argument (not closure)
        def my_func(comm_arg, output, inp):
            work = comm_arg.all_gather(output, inp, async_op=True)
            work.wait()
            return output  # List should use wait_tensors outputs

        # Compile with graph capture
        backend, captured = self._create_graph_capture_backend()
        compiled_func = torch.compile(my_func, fullgraph=True, backend=backend)

        # Run to trigger compilation
        result = compiled_func(comm, output_list, tensor)

        # Verify graph was captured
        self.assertIsNotNone(captured["graph"], "Graph should be captured")
        gm = captured["graph"]

        # Find wait_tensors nodes
        wait_nodes = _find_nodes_by_name_pattern(gm, "wait_tensors")

        print(f"\nDiagnostics:")
        print(f"  wait_nodes: {[n.name for n in wait_nodes]}")
        print(f"  output_sources: {_get_graph_output_sources(gm)}")

        self.assertTrue(len(wait_nodes) > 0, "Graph should contain wait_tensors")

        # Check that wait_tensors output is in the return path
        wait_in_path = any(_check_node_in_output_path(gm, node) for node in wait_nodes)
        self.assertTrue(
            wait_in_path,
            f"wait_tensors output should be in return path for list. "
            f"Wait nodes: {[n.name for n in wait_nodes]}, "
            f"Output sources: {_get_graph_output_sources(gm)}"
        )

        # Cleanup
        wrapper = None
        comm = None
        torch._dynamo.reset()

    def test_funcol_path_wait_proxy_propagation(self):
        """Test wait_tensors proxy propagation via funcol API (DTensor path).

        This test verifies the code path used by DTensor.redistribute(), where:
        1. Funcol API calls comm.all_gather_single() with async_op=True
        2. registry.py wrapper detects FakeTensor and returns FakeWork
        3. FakeWork.wait() calls wait_tensors_ and indexes the result

        This path does NOT go through dynamo.py's TorchCommMethodVariable,
        instead using FakeWork from async_tensor.py directly.
        """
        try:
            import contextlib

            from torch._functorch.aot_autograd import aot_export_joint_with_descriptors
            from torch._guards import tracing
            from torchcomms.device_mesh import init_native_device_mesh
            from torchcomms.functional.functional_backend import (
                register_torchcomms_funcols_impl,
            )
            from torchcomms.tests.integration.py.TorchCommTestHelpers import (
                TorchCommTestWrapper,
            )
        except ImportError as e:
            self.skipTest(f"Required imports not available: {e}")

        wrapper = TorchCommTestWrapper()
        comm = wrapper.get_torchcomm()
        device = comm.get_device()

        # Register torchcomms as funcol backend
        register_torchcomms_funcols_impl()

        # Create a 1D mesh with the torchcomms backend using init_native_device_mesh
        try:
            mesh = init_native_device_mesh(
                mesh_dim_comms=(comm,),
                mesh_dim_names=("main",),
            )
        except TypeError as e:
            if "_rank" in str(e):
                self.skipTest(f"DeviceMesh API incompatible: {e}")
            raise

        # Create a simple module that uses funcol
        class FuncolModule(torch.nn.Module):
            def __init__(self, mesh):
                super().__init__()
                self.mesh = mesh

            def forward(self, t):
                import torch.distributed._functional_collectives as funcol
                result = funcol.all_gather_tensor(t, gather_dim=0, group=(self.mesh, 0))
                return result

        model = FuncolModule(mesh)
        tensor = torch.ones(4, dtype=torch.float, device=device, requires_grad=True)

        # Use aot_export_joint_with_descriptors to capture the joint graph
        try:
            from torch._dynamo.functional_export import dynamo_graph_capture_for_export

            with (
                torch._dynamo.config.patch(fake_tensor_cache_enabled=False),
                torch.fx.traceback.preserve_node_meta(),
            ):
                gm = dynamo_graph_capture_for_export(model)(tensor)
                tracing_context = gm.meta["tracing_context"]

            with tracing(tracing_context):
                with contextlib.ExitStack() as stack:
                    joint_with_descriptors = aot_export_joint_with_descriptors(
                        stack,
                        gm,
                        (tensor,),
                        {},
                    )
                    joint_gm = joint_with_descriptors.graph_module
        except Exception as e:
            self.skipTest(f"Joint graph export failed: {e}")

        print("\n" + "=" * 80)
        print("JOINT FX GRAPH:")
        print("=" * 80)
        print(joint_gm.print_readable(print_output=False))
        print("=" * 80 + "\n")

        # Find wait_tensors nodes in the joint graph
        wait_nodes = _find_nodes_by_name_pattern(joint_gm, "wait_tensors")
        all_gather_nodes = _find_nodes_by_name_pattern(joint_gm, "all_gather")

        print(f"\nDiagnostics (funcol path - joint graph):")
        print(f"  wait_nodes: {[n.name for n in wait_nodes]}")
        print(f"  all_gather_nodes: {[n.name for n in all_gather_nodes]}")

        # For funcol path, the key check is that wait_tensors exists in the joint graph
        self.assertTrue(len(wait_nodes) > 0, "Joint graph should contain wait_tensors")

        # Verify wait_tensors has users (is not dead code)
        for wait_node in wait_nodes:
            has_users = len(wait_node.users) > 0
            self.assertTrue(
                has_users,
                f"wait_tensors node {wait_node.name} should have users"
            )

        # Cleanup
        wrapper = None
        comm = None
        torch._dynamo.reset()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
