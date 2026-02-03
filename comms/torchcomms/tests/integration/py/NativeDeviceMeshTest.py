#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Test the native device mesh integration with functional collectives.

This test validates that funcol calls (all_reduce, all_gather_tensor,
reduce_scatter_tensor, broadcast, all_to_all_single) dispatch correctly
to torchcomms when using init_native_device_mesh.
"""

import os
import unittest

from comms.torchcomms.tests.helpers.py.test_helpers import (
    skip_if_torch_compile_not_supported_or_enabled,
)

os.environ["TORCHCOMMS_PATCH_FOR_COMPILE"] = "1"

import torch
import torch.distributed._functional_collectives as funcol
import torchcomms
from torchcomms.device_mesh import init_native_device_mesh
from torchcomms.functional.async_tensor import TorchCommsAsyncTensor


@skip_if_torch_compile_not_supported_or_enabled()
class NativeDeviceMeshTest(unittest.TestCase):
    """Test class for native DeviceMesh funcol integration."""

    def setUp(self):
        self.backend = os.environ["TEST_BACKEND"]
        self.device = torch.device(os.environ.get("TEST_DEVICE", "cuda"))

    def _sync_device(self):
        """Synchronize device if needed."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _create_graph_logging_backend(self, test_name: str = ""):
        """Create a custom backend that logs the FX graph before inductor compilation."""
        import logging

        from torch._inductor import compile_fx

        logger = logging.getLogger(__name__)

        def logging_backend(gm, example_inputs):
            logger.warning(f"\n{'=' * 60}\nFX Graph for {test_name}:\n{'=' * 60}")
            logger.warning(gm.graph)
            logger.warning(f"\n{'=' * 60}\nGraph Code:\n{'=' * 60}")
            logger.warning(gm.code)
            print(f"\n{'=' * 60}\nFX Graph for {test_name}:\n{'=' * 60}")
            print(gm.graph)
            print(f"\n{'=' * 60}\nGraph Code:\n{'=' * 60}")
            print(gm.code)
            # Pass to inductor for actual compilation
            return compile_fx.compile_fx(gm, example_inputs)

        return logging_backend

    def _create_joint_graph_logging_backend(self, test_name: str = ""):
        """Create a backend that logs the joint forward+backward graph."""
        from functorch.compile import make_boxed_compiler
        from torch._dynamo.backends.common import aot_autograd

        @make_boxed_compiler
        def fw_compiler(gm, example_inputs):
            print(f"\n{'=' * 60}")
            print(f"Forward Graph for {test_name}:")
            print(f"{'=' * 60}")
            print(gm.code)
            return gm

        @make_boxed_compiler
        def bw_compiler(gm, example_inputs):
            print(f"\n{'=' * 60}")
            print(f"Backward Graph for {test_name}:")
            print(f"{'=' * 60}")
            print(gm.code)
            return gm

        return aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)

    def test_init_native_device_mesh(self) -> None:
        """Test that init_native_device_mesh creates a valid DeviceMesh."""
        comm = torchcomms.new_comm(
            self.backend, self.device, name="native_mesh_init_test"
        )

        try:
            try:
                mesh = init_native_device_mesh(
                    mesh_dim_comms=(comm,),
                    mesh_dim_names=("main",),
                )
            except TypeError as e:
                if "_rank" in str(e):
                    return
                raise

            # Verify comm backend is registered
            self.assertTrue(hasattr(mesh, "_comm_backends"))
            self.assertIn("torchcomms", mesh._comm_backends)
            self.assertEqual(mesh.get_comm_object("torchcomms", "main"), comm)
            self.assertEqual(mesh.get_comm_object("torchcomms", 0), comm)
        finally:
            comm.finalize()

    def test_funcol_all_reduce(self) -> None:
        """Test that funcol.all_reduce dispatches to torchcomms."""
        comm = torchcomms.new_comm(
            self.backend, self.device, name="funcol_all_reduce_test"
        )

        try:
            try:
                mesh = init_native_device_mesh(
                    mesh_dim_comms=(comm,),
                    mesh_dim_names=("main",),
                )
            except TypeError as e:
                if "_rank" in str(e):
                    return
                raise

            rank = comm.get_rank()
            world_size = comm.get_size()

            # Test sum reduction - result should be async tensor (no requires_grad)
            t = torch.ones(10, device=self.device, dtype=torch.float32) * (rank + 1)
            result = funcol.all_reduce(t, "sum", (mesh, 0))

            # Verify we get an async tensor back (no requires_grad -> async path)
            self.assertIsInstance(result, TorchCommsAsyncTensor)

            # Access triggers wait automatically
            self._sync_device()
            expected_sum = sum(range(1, world_size + 1))
            self.assertTrue(
                torch.allclose(result, torch.full_like(result, expected_sum))
            )

            # Test avg reduction
            t = torch.ones(10, device=self.device, dtype=torch.float32) * (rank + 1)
            result = funcol.all_reduce(t, "avg", (mesh, 0))
            self._sync_device()

            expected_avg = sum(range(1, world_size + 1)) / world_size
            self.assertTrue(
                torch.allclose(result, torch.full_like(result, expected_avg), atol=1e-5)
            )

            # Test max reduction
            t = torch.ones(10, device=self.device, dtype=torch.float32) * (rank + 1)
            result = funcol.all_reduce(t, "max", (mesh, 0))
            self._sync_device()

            self.assertTrue(torch.allclose(result, torch.full_like(result, world_size)))
        finally:
            comm.finalize()

    def test_async_tensor_explicit_wait(self) -> None:
        """Test that TorchCommsAsyncTensor.wait() works correctly."""
        comm = torchcomms.new_comm(
            self.backend, self.device, name="funcol_async_wait_test"
        )

        try:
            try:
                mesh = init_native_device_mesh(
                    mesh_dim_comms=(comm,),
                    mesh_dim_names=("main",),
                )
            except TypeError as e:
                if "_rank" in str(e):
                    return
                raise

            rank = comm.get_rank()
            world_size = comm.get_size()

            t = torch.ones(10, device=self.device, dtype=torch.float32) * (rank + 1)
            result = funcol.all_reduce(t, "sum", (mesh, 0))

            # Verify it's an async tensor
            self.assertIsInstance(result, TorchCommsAsyncTensor)
            self.assertFalse(result.completed)

            # Explicitly wait
            unwrapped = result.wait()
            self.assertTrue(result.completed)

            # Verify result
            self._sync_device()
            expected_sum = sum(range(1, world_size + 1))
            self.assertTrue(
                torch.allclose(unwrapped, torch.full_like(unwrapped, expected_sum))
            )
        finally:
            comm.finalize()

    def test_torch_compile_all_reduce(self) -> None:
        """Test that funcol.all_reduce works with torch.compile."""
        comm = torchcomms.new_comm(
            self.backend, self.device, name="funcol_compile_test"
        )

        try:
            try:
                mesh = init_native_device_mesh(
                    mesh_dim_comms=(comm,),
                    mesh_dim_names=("main",),
                )
            except TypeError as e:
                if "_rank" in str(e):
                    return
                raise

            rank = comm.get_rank()
            world_size = comm.get_size()

            comm.barrier(async_op=False)

            def fn(t: torch.Tensor, mesh) -> torch.Tensor:
                result = funcol.all_reduce(t, "sum", (mesh, 0))
                # Trigger wait by using the tensor
                return result * 2

            compiled_fn = torch.compile(
                fn,
                fullgraph=True,
                backend=self._create_graph_logging_backend(
                    "test_torch_compile_all_reduce"
                ),
            )

            t = torch.ones(10, device=self.device, dtype=torch.float32) * (rank + 1)
            result = compiled_fn(t, mesh)
            self._sync_device()

            expected_sum = sum(range(1, world_size + 1))
            expected = torch.full(
                (10,), expected_sum * 2, device=self.device, dtype=torch.float32
            )
            self.assertTrue(torch.allclose(result, expected))
        finally:
            comm.finalize()

    def test_torch_compile_multiple_collectives(self) -> None:
        """Test that multiple collectives work with torch.compile."""
        comm = torchcomms.new_comm(
            self.backend, self.device, name="funcol_compile_multi_test"
        )

        try:
            try:
                mesh = init_native_device_mesh(
                    mesh_dim_comms=(comm,),
                    mesh_dim_names=("main",),
                )
            except TypeError as e:
                if "_rank" in str(e):
                    return
                raise

            rank = comm.get_rank()
            world_size = comm.get_size()

            comm.barrier(async_op=False)

            def fn(t: torch.Tensor) -> torch.Tensor:
                # Chain multiple collectives
                reduced = funcol.all_reduce(t, "sum", (mesh, 0))
                gathered = funcol.all_gather_tensor(reduced, 0, (mesh, 0))
                return gathered

            compiled_fn = torch.compile(
                fn,
                fullgraph=True,
                backend=self._create_graph_logging_backend(
                    "test_torch_compile_multiple_collectives"
                ),
            )

            t = torch.ones(10, device=self.device, dtype=torch.float32) * (rank + 1)
            result = compiled_fn(t)
            self._sync_device()

            expected_sum = sum(range(1, world_size + 1))
            self.assertEqual(result.shape[0], 10 * world_size)
            self.assertTrue(
                torch.allclose(
                    result,
                    torch.full_like(result, expected_sum),
                )
            )
        finally:
            comm.finalize()

    def test_funcol_broadcast(self) -> None:
        """Test that funcol.broadcast dispatches to torchcomms."""
        comm = torchcomms.new_comm(
            self.backend, self.device, name="funcol_broadcast_test"
        )

        try:
            try:
                mesh = init_native_device_mesh(
                    mesh_dim_comms=(comm,),
                    mesh_dim_names=("main",),
                )
            except TypeError as e:
                if "_rank" in str(e):
                    return
                raise

            rank = comm.get_rank()

            # Broadcast from rank 0
            t = torch.ones(10, device=self.device, dtype=torch.float32) * (rank + 1)
            result = funcol.broadcast(t, 0, (mesh, 0))
            self._sync_device()

            # All ranks should have rank 0's value (1.0)
            self.assertTrue(torch.allclose(result, torch.ones_like(result)))
        finally:
            comm.finalize()

    def test_funcol_scatter(self) -> None:
        """Test that funcol.scatter dispatches to torchcomms."""
        comm = torchcomms.new_comm(
            self.backend, self.device, name="funcol_scatter_test"
        )

        try:
            try:
                mesh = init_native_device_mesh(
                    mesh_dim_comms=(comm,),
                    mesh_dim_names=("main",),
                )
            except TypeError as e:
                if "_rank" in str(e):
                    return
                raise

            rank = comm.get_rank()
            world_size = comm.get_size()

            # Create scatter list on rank 0 (each tensor has value = dest_rank + 1)
            output = torch.zeros(10, device=self.device, dtype=torch.float32)
            if rank == 0:
                scatter_list = [
                    torch.ones(10, device=self.device, dtype=torch.float32) * (r + 1)
                    for r in range(world_size)
                ]
            else:
                scatter_list = []

            result = funcol.scatter(output, scatter_list, 0, (mesh, 0))
            self._sync_device()

            # Each rank should receive its corresponding tensor (value = rank + 1)
            expected = torch.ones(10, device=self.device, dtype=torch.float32) * (
                rank + 1
            )
            self.assertTrue(torch.allclose(result, expected))
        finally:
            comm.finalize()

    def test_funcol_scatter_non_zero_src(self) -> None:
        """Test that funcol.scatter works with non-zero source rank."""
        comm = torchcomms.new_comm(
            self.backend, self.device, name="funcol_scatter_src_test"
        )

        try:
            try:
                mesh = init_native_device_mesh(
                    mesh_dim_comms=(comm,),
                    mesh_dim_names=("main",),
                )
            except TypeError as e:
                if "_rank" in str(e):
                    return
                raise

            rank = comm.get_rank()
            world_size = comm.get_size()

            if world_size < 2:
                return  # Need at least 2 ranks

            src_rank = 1

            # Create scatter list on src_rank
            output = torch.zeros(10, device=self.device, dtype=torch.float32)
            if rank == src_rank:
                scatter_list = [
                    torch.ones(10, device=self.device, dtype=torch.float32) * (r + 10)
                    for r in range(world_size)
                ]
            else:
                scatter_list = []

            result = funcol.scatter(output, scatter_list, src_rank, (mesh, 0))
            self._sync_device()

            # Each rank should receive its corresponding tensor (value = rank + 10)
            expected = torch.ones(10, device=self.device, dtype=torch.float32) * (
                rank + 10
            )
            self.assertTrue(torch.allclose(result, expected))
        finally:
            comm.finalize()

    def test_funcol_all_gather_tensor(self) -> None:
        """Test that funcol.all_gather_tensor dispatches to torchcomms."""
        comm = torchcomms.new_comm(
            self.backend, self.device, name="funcol_all_gather_test"
        )

        try:
            try:
                mesh = init_native_device_mesh(
                    mesh_dim_comms=(comm,),
                    mesh_dim_names=("main",),
                )
            except TypeError as e:
                if "_rank" in str(e):
                    return
                raise

            rank = comm.get_rank()
            world_size = comm.get_size()

            # Each rank contributes its rank value
            t = torch.ones(10, device=self.device, dtype=torch.float32) * rank
            result = funcol.all_gather_tensor(t, 0, (mesh, 0))
            self._sync_device()

            # Result should be concatenated tensors from all ranks
            self.assertEqual(result.shape[0], 10 * world_size)
            for r in range(world_size):
                chunk = result[r * 10 : (r + 1) * 10]
                expected = torch.ones(10, device=self.device, dtype=torch.float32) * r
                self.assertTrue(torch.allclose(chunk, expected))
        finally:
            comm.finalize()

    def test_funcol_reduce_scatter_tensor(self) -> None:
        """Test that funcol.reduce_scatter_tensor dispatches to torchcomms."""
        comm = torchcomms.new_comm(
            self.backend, self.device, name="funcol_reduce_scatter_test"
        )

        try:
            try:
                mesh = init_native_device_mesh(
                    mesh_dim_comms=(comm,),
                    mesh_dim_names=("main",),
                )
            except TypeError as e:
                if "_rank" in str(e):
                    return
                raise

            rank = comm.get_rank()
            world_size = comm.get_size()

            # Input tensor of size world_size * 10
            t = torch.ones(world_size * 10, device=self.device, dtype=torch.float32) * (
                rank + 1
            )
            result = funcol.reduce_scatter_tensor(t, "sum", 0, (mesh, 0))
            self._sync_device()

            # Each rank gets a reduced scatter of size 10
            self.assertEqual(result.shape[0], 10)
            expected_sum = sum(range(1, world_size + 1))
            self.assertTrue(
                torch.allclose(result, torch.full_like(result, expected_sum))
            )
        finally:
            comm.finalize()

    def test_funcol_all_to_all_single(self) -> None:
        """Test that funcol.all_to_all_single dispatches to torchcomms."""
        comm = torchcomms.new_comm(
            self.backend, self.device, name="funcol_all_to_all_test"
        )

        try:
            try:
                mesh = init_native_device_mesh(
                    mesh_dim_comms=(comm,),
                    mesh_dim_names=("main",),
                )
            except TypeError as e:
                if "_rank" in str(e):
                    return
                raise

            rank = comm.get_rank()
            world_size = comm.get_size()

            # Each rank sends different data to each other rank
            # Rank r sends value (r * world_size + dest) to dest
            t = torch.arange(
                world_size * 10, device=self.device, dtype=torch.float32
            ).reshape(world_size, 10)
            t = t + rank * 100  # Make each rank's data unique
            t = t.flatten()

            result = funcol.all_to_all_single(t, None, None, (mesh, 0))
            self._sync_device()

            # Verify shape
            self.assertEqual(result.shape[0], world_size * 10)
        finally:
            comm.finalize()

    def test_funcol_all_reduce_coalesced(self) -> None:
        """Test that funcol.all_reduce_coalesced dispatches to torchcomms."""
        comm = torchcomms.new_comm(
            self.backend, self.device, name="funcol_all_reduce_coalesced_test"
        )

        try:
            try:
                mesh = init_native_device_mesh(
                    mesh_dim_comms=(comm,),
                    mesh_dim_names=("main",),
                )
            except TypeError as e:
                if "_rank" in str(e):
                    return
                raise

            rank = comm.get_rank()
            world_size = comm.get_size()

            # Create multiple tensors of different sizes
            t1 = torch.ones(10, device=self.device, dtype=torch.float32) * (rank + 1)
            t2 = (
                torch.ones(20, device=self.device, dtype=torch.float32) * (rank + 1) * 2
            )
            t3 = torch.ones(5, device=self.device, dtype=torch.float32) * (rank + 1) * 3

            results = funcol.all_reduce_coalesced([t1, t2, t3], "sum", (mesh, 0))
            self._sync_device()

            # Verify results
            expected_sum = sum(range(1, world_size + 1))
            self.assertEqual(len(results), 3)
            self.assertEqual(results[0].shape[0], 10)
            self.assertEqual(results[1].shape[0], 20)
            self.assertEqual(results[2].shape[0], 5)

            self.assertTrue(
                torch.allclose(results[0], torch.full_like(results[0], expected_sum))
            )
            self.assertTrue(
                torch.allclose(
                    results[1], torch.full_like(results[1], expected_sum * 2)
                )
            )
            self.assertTrue(
                torch.allclose(
                    results[2], torch.full_like(results[2], expected_sum * 3)
                )
            )
        finally:
            comm.finalize()

    def test_funcol_all_gather_into_tensor_coalesced(self) -> None:
        """Test that funcol.all_gather_into_tensor_coalesced dispatches to torchcomms."""
        comm = torchcomms.new_comm(
            self.backend, self.device, name="funcol_all_gather_coalesced_test"
        )

        try:
            try:
                mesh = init_native_device_mesh(
                    mesh_dim_comms=(comm,),
                    mesh_dim_names=("main",),
                )
            except TypeError as e:
                if "_rank" in str(e):
                    return
                raise

            rank = comm.get_rank()
            world_size = comm.get_size()

            # Create multiple tensors of different sizes
            t1 = torch.ones(10, device=self.device, dtype=torch.float32) * rank
            t2 = torch.ones(5, device=self.device, dtype=torch.float32) * (rank + 10)

            results = funcol.all_gather_into_tensor_coalesced([t1, t2], (mesh, 0))
            self._sync_device()

            # Verify results
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0].shape[0], 10 * world_size)
            self.assertEqual(results[1].shape[0], 5 * world_size)

            # Check first tensor - should have [0, 0, ..., 1, 1, ..., 2, 2, ...]
            for r in range(world_size):
                chunk = results[0][r * 10 : (r + 1) * 10]
                expected = torch.ones(10, device=self.device, dtype=torch.float32) * r
                self.assertTrue(torch.allclose(chunk, expected))

            # Check second tensor
            for r in range(world_size):
                chunk = results[1][r * 5 : (r + 1) * 5]
                expected = torch.ones(5, device=self.device, dtype=torch.float32) * (
                    r + 10
                )
                self.assertTrue(torch.allclose(chunk, expected))
        finally:
            comm.finalize()

    def test_funcol_reduce_scatter_tensor_coalesced(self) -> None:
        """Test that funcol.reduce_scatter_tensor_coalesced dispatches to torchcomms."""
        comm = torchcomms.new_comm(
            self.backend, self.device, name="funcol_reduce_scatter_coalesced_test"
        )

        try:
            try:
                mesh = init_native_device_mesh(
                    mesh_dim_comms=(comm,),
                    mesh_dim_names=("main",),
                )
            except TypeError as e:
                if "_rank" in str(e):
                    return
                raise

            rank = comm.get_rank()
            world_size = comm.get_size()

            # Create tensors that are divisible by world_size
            t1 = torch.ones(
                world_size * 10, device=self.device, dtype=torch.float32
            ) * (rank + 1)
            t2 = (
                torch.ones(world_size * 5, device=self.device, dtype=torch.float32)
                * (rank + 1)
                * 2
            )

            results = funcol.reduce_scatter_tensor_coalesced(
                [t1, t2], "sum", [0, 0], (mesh, 0)
            )
            self._sync_device()

            # Verify results
            expected_sum = sum(range(1, world_size + 1))
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0].shape[0], 10)
            self.assertEqual(results[1].shape[0], 5)

            self.assertTrue(
                torch.allclose(results[0], torch.full_like(results[0], expected_sum))
            )
            self.assertTrue(
                torch.allclose(
                    results[1], torch.full_like(results[1], expected_sum * 2)
                )
            )
        finally:
            comm.finalize()

    def test_torch_compile_coalesced_collectives(self) -> None:
        """Test that coalesced collectives work with torch.compile."""
        comm = torchcomms.new_comm(
            self.backend, self.device, name="funcol_compile_coalesced_test"
        )

        try:
            try:
                mesh = init_native_device_mesh(
                    mesh_dim_comms=(comm,),
                    mesh_dim_names=("main",),
                )
            except TypeError as e:
                if "_rank" in str(e):
                    return
                raise

            rank = comm.get_rank()
            world_size = comm.get_size()

            comm.barrier(async_op=False)

            def fn(t1: torch.Tensor, t2: torch.Tensor) -> list[torch.Tensor]:
                results = funcol.all_reduce_coalesced([t1, t2], "sum", (mesh, 0))
                # Use the results to trigger wait
                return [r * 2 for r in results]

            compiled_fn = torch.compile(
                fn,
                fullgraph=True,
                backend=self._create_graph_logging_backend(
                    "test_torch_compile_coalesced_collectives"
                ),
            )

            t1 = torch.ones(10, device=self.device, dtype=torch.float32) * (rank + 1)
            t2 = torch.ones(5, device=self.device, dtype=torch.float32) * (rank + 1)
            results = compiled_fn(t1, t2)
            self._sync_device()

            expected_sum = sum(range(1, world_size + 1))
            self.assertTrue(
                torch.allclose(
                    results[0], torch.full_like(results[0], expected_sum * 2)
                )
            )
            self.assertTrue(
                torch.allclose(
                    results[1], torch.full_like(results[1], expected_sum * 2)
                )
            )
        finally:
            comm.finalize()

    def test_torch_compile_reduce_scatter_coalesced(self) -> None:
        """Test that reduce_scatter_tensor_coalesced works with torch.compile."""
        comm = torchcomms.new_comm(
            self.backend, self.device, name="funcol_compile_rs_coalesced_test"
        )

        try:
            try:
                mesh = init_native_device_mesh(
                    mesh_dim_comms=(comm,),
                    mesh_dim_names=("main",),
                )
            except TypeError as e:
                if "_rank" in str(e):
                    return
                raise

            rank = comm.get_rank()
            world_size = comm.get_size()

            comm.barrier(async_op=False)

            def fn(t1: torch.Tensor, t2: torch.Tensor) -> list[torch.Tensor]:
                results = funcol.reduce_scatter_tensor_coalesced(
                    [t1, t2], "sum", [0, 0], (mesh, 0)
                )
                # Use the results to trigger wait
                return [r * 2 for r in results]

            compiled_fn = torch.compile(
                fn,
                fullgraph=True,
                backend=self._create_graph_logging_backend(
                    "test_torch_compile_reduce_scatter_coalesced"
                ),
            )

            # Create tensors that are divisible by world_size
            t1 = torch.ones(
                world_size * 10, device=self.device, dtype=torch.float32
            ) * (rank + 1)
            t2 = (
                torch.ones(world_size * 5, device=self.device, dtype=torch.float32)
                * (rank + 1)
                * 2
            )
            results = compiled_fn(t1, t2)
            self._sync_device()

            expected_sum = sum(range(1, world_size + 1))
            self.assertEqual(results[0].shape[0], 10)
            self.assertEqual(results[1].shape[0], 5)
            self.assertTrue(
                torch.allclose(
                    results[0], torch.full_like(results[0], expected_sum * 2)
                )
            )
            self.assertTrue(
                torch.allclose(
                    results[1], torch.full_like(results[1], expected_sum * 2 * 2)
                )
            )
        finally:
            comm.finalize()

    def test_torch_compile_all_gather_coalesced(self) -> None:
        """Test that all_gather_into_tensor_coalesced works with torch.compile."""
        comm = torchcomms.new_comm(
            self.backend, self.device, name="funcol_compile_ag_coalesced_test"
        )

        try:
            try:
                mesh = init_native_device_mesh(
                    mesh_dim_comms=(comm,),
                    mesh_dim_names=("main",),
                )
            except TypeError as e:
                if "_rank" in str(e):
                    return
                raise

            rank = comm.get_rank()
            world_size = comm.get_size()

            comm.barrier(async_op=False)

            def fn(t1: torch.Tensor, t2: torch.Tensor) -> list[torch.Tensor]:
                results = funcol.all_gather_into_tensor_coalesced([t1, t2], (mesh, 0))
                # Use the results to trigger wait
                return [r * 2 for r in results]

            compiled_fn = torch.compile(
                fn,
                fullgraph=True,
                backend=self._create_graph_logging_backend(
                    "test_torch_compile_all_gather_coalesced"
                ),
            )

            t1 = torch.ones(10, device=self.device, dtype=torch.float32) * (rank + 1)
            t2 = torch.ones(5, device=self.device, dtype=torch.float32) * (rank + 1)
            results = compiled_fn(t1, t2)
            self._sync_device()

            # Verify shapes
            self.assertEqual(results[0].shape[0], 10 * world_size)
            self.assertEqual(results[1].shape[0], 5 * world_size)

            # Verify values - each chunk should be from the corresponding rank * 2
            for r in range(world_size):
                expected_val = (r + 1) * 2.0
                self.assertTrue(
                    torch.allclose(
                        results[0][r * 10 : (r + 1) * 10],
                        torch.full(
                            (10,), expected_val, device=self.device, dtype=torch.float32
                        ),
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        results[1][r * 5 : (r + 1) * 5],
                        torch.full(
                            (5,), expected_val, device=self.device, dtype=torch.float32
                        ),
                    )
                )
        finally:
            comm.finalize()

    @unittest.skipIf(
        torch.cuda.device_count() < 4, "Need at least 4 GPUs for 2D mesh test"
    )
    def test_2d_mesh_funcol(self) -> None:
        """Test funcol with 2D device mesh."""
        world_size = torch.cuda.device_count()
        dp_degree = 2
        tp_degree = world_size // dp_degree
        mesh_tensor = torch.arange(world_size, dtype=torch.int, device="cpu").view(
            dp_degree, tp_degree
        )

        comm = torchcomms.new_comm(
            self.backend, self.device, name="funcol_2d_mesh_test"
        )

        cur_rank = comm.get_rank()

        # Find TP and DP ranks for current rank
        tp_ranks = None
        for row in mesh_tensor.tolist():
            if cur_rank in row:
                tp_ranks = row
                break

        dp_ranks = None
        for col in mesh_tensor.transpose(0, 1).tolist():
            if cur_rank in col:
                dp_ranks = col
                break

        tp_comm = comm.split(tp_ranks, "tp")
        dp_comm = comm.split(dp_ranks, "dp")

        try:
            try:
                mesh = init_native_device_mesh(
                    mesh_dim_comms=(dp_comm, tp_comm),
                    mesh_dim_names=("dp", "tp"),
                )
            except (TypeError, ValueError) as e:
                if "_rank" in str(e):
                    return
                raise

            # Test all_reduce on TP dimension
            t = torch.ones(10, device=self.device, dtype=torch.float32) * (cur_rank + 1)
            result = funcol.all_reduce(t.clone(), "sum", (mesh, 1))  # dim 1 = tp
            self._sync_device()

            # Sum should be sum of TP group ranks
            expected_sum = sum(r + 1 for r in tp_ranks)
            self.assertTrue(
                torch.allclose(result, torch.full_like(result, expected_sum))
            )

            # Test all_reduce on DP dimension
            t = torch.ones(10, device=self.device, dtype=torch.float32) * (cur_rank + 1)
            result = funcol.all_reduce(t.clone(), "sum", (mesh, 0))  # dim 0 = dp
            self._sync_device()

            # Sum should be sum of DP group ranks
            expected_sum = sum(r + 1 for r in dp_ranks)
            self.assertTrue(
                torch.allclose(result, torch.full_like(result, expected_sum))
            )
        finally:
            tp_comm.finalize()
            dp_comm.finalize()
            comm.finalize()

    # =========================================================================
    # Backward/Autograd Tests
    # =========================================================================

    def test_all_reduce_backward(self) -> None:
        """Test that all_reduce backward is identity (same all_reduce)."""
        comm = torchcomms.new_comm(
            self.backend, self.device, name="all_reduce_backward_test"
        )

        try:
            try:
                mesh = init_native_device_mesh(
                    mesh_dim_comms=(comm,),
                    mesh_dim_names=("main",),
                )
            except TypeError as e:
                if "_rank" in str(e):
                    return
                raise

            rank = comm.get_rank()
            world_size = comm.get_size()

            # Create input with requires_grad
            t = torch.full(
                (10,),
                rank + 1,
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
            )

            # Forward
            result = funcol.all_reduce(t, "sum", (mesh, 0))
            self._sync_device()

            # Backward - gradient should be all_reduce of upstream gradient
            upstream_grad = torch.ones_like(result) * (rank + 1)
            result.backward(upstream_grad)
            self._sync_device()

            # Gradient should be sum of all upstream gradients
            expected_grad_sum = sum(range(1, world_size + 1))
            self.assertIsNotNone(t.grad)
            self.assertTrue(
                torch.allclose(t.grad, torch.full_like(t.grad, expected_grad_sum))
            )
        finally:
            comm.finalize()

    def test_all_gather_backward(self) -> None:
        """Test that all_gather backward is reduce_scatter with sum."""
        comm = torchcomms.new_comm(
            self.backend, self.device, name="all_gather_backward_test"
        )

        try:
            try:
                mesh = init_native_device_mesh(
                    mesh_dim_comms=(comm,),
                    mesh_dim_names=("main",),
                )
            except TypeError as e:
                if "_rank" in str(e):
                    return
                raise

            rank = comm.get_rank()
            world_size = comm.get_size()

            # Create input with requires_grad
            t = torch.full(
                (10,),
                rank + 1,
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
            )

            # Forward: all_gather
            result = funcol.all_gather_tensor(t, 0, (mesh, 0))
            self._sync_device()

            # Verify forward result
            self.assertEqual(result.shape[0], 10 * world_size)

            # Backward with uniform gradient
            upstream_grad = torch.ones_like(result)
            result.backward(upstream_grad)
            self._sync_device()

            # Backward of all_gather is reduce_scatter with sum
            # Each rank's gradient is the sum of gradients that were scattered to it
            # With uniform gradient of 1s, each rank should get world_size
            self.assertIsNotNone(t.grad)
            self.assertTrue(
                torch.allclose(t.grad, torch.full_like(t.grad, float(world_size)))
            )
        finally:
            comm.finalize()

    def test_reduce_scatter_backward(self) -> None:
        """Test that reduce_scatter backward is all_gather."""
        comm = torchcomms.new_comm(
            self.backend, self.device, name="reduce_scatter_backward_test"
        )

        try:
            try:
                mesh = init_native_device_mesh(
                    mesh_dim_comms=(comm,),
                    mesh_dim_names=("main",),
                )
            except TypeError as e:
                if "_rank" in str(e):
                    return
                raise

            rank = comm.get_rank()
            world_size = comm.get_size()

            # Create input with requires_grad (size must be divisible by world_size)
            t = torch.full(
                (world_size * 10,),
                rank + 1,
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
            )

            # Forward: reduce_scatter
            result = funcol.reduce_scatter_tensor(t, "sum", 0, (mesh, 0))
            self._sync_device()

            # Verify forward result
            self.assertEqual(result.shape[0], 10)

            # Backward with gradient = rank + 1
            upstream_grad = torch.ones_like(result) * (rank + 1)
            result.backward(upstream_grad)
            self._sync_device()

            # Backward of reduce_scatter is all_gather
            # The gradient is gathered from all ranks
            self.assertIsNotNone(t.grad)
            # Gradient should be all_gather of upstream grads
            for r in range(world_size):
                chunk = t.grad[r * 10 : (r + 1) * 10]
                expected = torch.ones(10, device=self.device, dtype=torch.float32) * (
                    r + 1
                )
                self.assertTrue(torch.allclose(chunk, expected))
        finally:
            comm.finalize()

    def test_all_to_all_single_backward(self) -> None:
        """Test that all_to_all_single backward swaps input/output splits."""
        comm = torchcomms.new_comm(
            self.backend, self.device, name="all_to_all_backward_test"
        )

        try:
            try:
                mesh = init_native_device_mesh(
                    mesh_dim_comms=(comm,),
                    mesh_dim_names=("main",),
                )
            except TypeError as e:
                if "_rank" in str(e):
                    return
                raise

            world_size = comm.get_size()

            # Create input with requires_grad
            t = torch.arange(
                world_size * 10,
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
            )

            # Forward: all_to_all with equal splits
            result = funcol.all_to_all_single(t, None, None, (mesh, 0))
            self._sync_device()

            # Verify forward result shape
            self.assertEqual(result.shape[0], world_size * 10)

            # Backward
            upstream_grad = torch.ones_like(result)
            result.backward(upstream_grad)
            self._sync_device()

            # Backward of all_to_all is all_to_all with swapped splits
            # With equal splits and uniform gradient, result should be uniform
            self.assertIsNotNone(t.grad)
            self.assertTrue(torch.allclose(t.grad, torch.ones_like(t.grad)))
        finally:
            comm.finalize()

    def test_autograd_chain(self) -> None:
        """Test chaining multiple autograd collectives."""
        comm = torchcomms.new_comm(
            self.backend, self.device, name="autograd_chain_test"
        )

        try:
            try:
                mesh = init_native_device_mesh(
                    mesh_dim_comms=(comm,),
                    mesh_dim_names=("main",),
                )
            except TypeError as e:
                if "_rank" in str(e):
                    return
                raise

            rank = comm.get_rank()
            world_size = comm.get_size()

            # Create input with requires_grad
            t = torch.full(
                (10,),
                rank + 1,
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
            )

            # Chain: all_gather -> reduce_scatter (should be identity-ish)
            gathered = funcol.all_gather_tensor(t, 0, (mesh, 0))
            result = funcol.reduce_scatter_tensor(gathered, "sum", 0, (mesh, 0))
            self._sync_device()

            # Forward result: all_gather creates [t0, t1, ...] on each rank
            # reduce_scatter with sum: rank r gets sum of chunk r from all ranks
            # Since all ranks have the same gathered tensor, each chunk is summed world_size times
            # Rank r's chunk contains value (r+1), so result = world_size * (rank+1)
            expected_value = float(world_size * (rank + 1))
            self.assertTrue(
                torch.allclose(result, torch.full_like(result, expected_value))
            )

            # Backward
            upstream_grad = torch.ones_like(result)
            result.backward(upstream_grad)
            self._sync_device()

            # Gradient should flow through both operations
            self.assertIsNotNone(t.grad)
            # reduce_scatter backward = all_gather, all_gather backward = reduce_scatter
            # Net effect with uniform grads should give world_size
            self.assertTrue(
                torch.allclose(t.grad, torch.full_like(t.grad, float(world_size)))
            )
        finally:
            comm.finalize()

    def test_torch_compile_autograd(self) -> None:
        """Test that autograd works with torch.compile."""
        comm = torchcomms.new_comm(
            self.backend, self.device, name="compile_autograd_test"
        )

        try:
            try:
                mesh = init_native_device_mesh(
                    mesh_dim_comms=(comm,),
                    mesh_dim_names=("main",),
                )
            except TypeError as e:
                if "_rank" in str(e):
                    return
                raise

            rank = comm.get_rank()
            world_size = comm.get_size()

            comm.barrier(async_op=False)

            def fn(t: torch.Tensor, mesh) -> torch.Tensor:
                # all_reduce with autograd
                result = funcol.all_reduce(t, "sum", (mesh, 0))
                # Apply a differentiable operation
                return result * 2

            compiled_fn = torch.compile(
                fn,
                fullgraph=True,
                backend=self._create_joint_graph_logging_backend(
                    "test_torch_compile_autograd"
                ),
            )

            # Create input with requires_grad
            t = torch.full(
                (10,),
                rank + 1,
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
            )
            result = compiled_fn(t, mesh)
            self._sync_device()

            # Verify forward result
            expected_sum = sum(range(1, world_size + 1))
            expected = torch.full(
                (10,), expected_sum * 2, device=self.device, dtype=torch.float32
            )
            self.assertTrue(torch.allclose(result, expected))

            # Backward
            upstream_grad = torch.ones_like(result)
            result.backward(upstream_grad)
            self._sync_device()

            # Gradient should flow through: upstream (1) -> *2 (2) -> all_reduce (2*world_size)
            # all_reduce backward is identity (same all_reduce), so each rank gets sum of upstream grads
            # upstream_grad is 1, multiplied by 2 from the *2 operation
            # all_reduce of 2s across world_size ranks = 2 * world_size
            expected_grad = float(2 * world_size)
            self.assertIsNotNone(t.grad)
            self.assertTrue(
                torch.allclose(t.grad, torch.full_like(t.grad, expected_grad))
            )
        finally:
            comm.finalize()

    def test_torch_compile_autograd_all_gather(self) -> None:
        """Test that all_gather autograd works with torch.compile."""
        comm = torchcomms.new_comm(
            self.backend, self.device, name="compile_autograd_gather_test"
        )

        try:
            try:
                mesh = init_native_device_mesh(
                    mesh_dim_comms=(comm,),
                    mesh_dim_names=("main",),
                )
            except TypeError as e:
                if "_rank" in str(e):
                    return
                raise

            rank = comm.get_rank()
            world_size = comm.get_size()

            comm.barrier(async_op=False)

            def fn(t: torch.Tensor, mesh) -> torch.Tensor:
                # all_gather with autograd
                gathered = funcol.all_gather_tensor(t, 0, (mesh, 0))
                # Sum to get a scalar-ish result for backward
                return gathered.sum()

            compiled_fn = torch.compile(
                fn,
                fullgraph=True,
                backend=self._create_joint_graph_logging_backend(
                    "test_torch_compile_autograd_all_gather"
                ),
            )

            # Create input with requires_grad
            t = torch.full(
                (10,),
                rank + 1,
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
            )
            result = compiled_fn(t, mesh)
            self._sync_device()

            # Verify forward result: sum of all gathered values
            # Each rank contributes 10 * (rank+1), total = 10 * sum(1..world_size)
            expected_sum = 10 * sum(range(1, world_size + 1))
            self.assertTrue(
                torch.allclose(
                    result, torch.tensor(float(expected_sum), device=self.device)
                )
            )

            # Backward
            result.backward()
            self._sync_device()

            # all_gather backward is reduce_scatter with sum
            # Gradient from sum() is 1 for each element
            # reduce_scatter of uniform 1s gives world_size to each element
            self.assertIsNotNone(t.grad)
            self.assertTrue(
                torch.allclose(t.grad, torch.full_like(t.grad, float(world_size)))
            )
        finally:
            comm.finalize()

    def test_torch_compile_autograd_reduce_scatter(self) -> None:
        """Test that reduce_scatter autograd works with torch.compile."""
        comm = torchcomms.new_comm(
            self.backend, self.device, name="compile_autograd_reduce_scatter_test"
        )

        try:
            try:
                mesh = init_native_device_mesh(
                    mesh_dim_comms=(comm,),
                    mesh_dim_names=("main",),
                )
            except TypeError as e:
                if "_rank" in str(e):
                    return
                raise

            rank = comm.get_rank()
            world_size = comm.get_size()

            comm.barrier(async_op=False)

            def fn(t: torch.Tensor, mesh) -> torch.Tensor:
                # reduce_scatter with autograd
                scattered = funcol.reduce_scatter_tensor(t, "sum", 0, (mesh, 0))
                # Apply a differentiable operation
                return scattered * 2

            compiled_fn = torch.compile(
                fn,
                fullgraph=True,
                backend=self._create_joint_graph_logging_backend(
                    "test_torch_compile_autograd_reduce_scatter"
                ),
            )

            # Create input with requires_grad - size must be divisible by world_size
            t = torch.full(
                (world_size * 10,),
                rank + 1,
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
            )
            result = compiled_fn(t, mesh)
            self._sync_device()

            # Verify forward result: reduce_scatter sums across ranks then scatters
            # Each rank's chunk is sum of (rank+1) values from all ranks = sum(1..world_size)
            expected_sum = sum(range(1, world_size + 1))
            expected = torch.full(
                (10,), expected_sum * 2, device=self.device, dtype=torch.float32
            )
            self.assertEqual(result.shape[0], 10)
            self.assertTrue(torch.allclose(result, expected))

            # Backward
            upstream_grad = torch.ones_like(result)
            result.backward(upstream_grad)
            self._sync_device()

            # reduce_scatter backward is all_gather
            # Gradient 1 * 2 (from *2) = 2, all_gathered gives 2 to all elements
            self.assertIsNotNone(t.grad)
            self.assertTrue(torch.allclose(t.grad, torch.full_like(t.grad, 2.0)))
        finally:
            comm.finalize()

    def test_torch_compile_autograd_all_to_all(self) -> None:
        """Test that all_to_all autograd works with torch.compile."""
        comm = torchcomms.new_comm(
            self.backend, self.device, name="compile_autograd_all_to_all_test"
        )

        try:
            try:
                mesh = init_native_device_mesh(
                    mesh_dim_comms=(comm,),
                    mesh_dim_names=("main",),
                )
            except TypeError as e:
                if "_rank" in str(e):
                    return
                raise

            rank = comm.get_rank()
            world_size = comm.get_size()

            comm.barrier(async_op=False)

            def fn(t: torch.Tensor, mesh) -> torch.Tensor:
                # all_to_all with autograd
                exchanged = funcol.all_to_all_single(t, None, None, (mesh, 0))
                # Apply a differentiable operation
                return exchanged * 2

            compiled_fn = torch.compile(
                fn,
                fullgraph=True,
                backend=self._create_joint_graph_logging_backend(
                    "test_torch_compile_autograd_all_to_all"
                ),
            )

            # Create input with requires_grad - size must be divisible by world_size
            # Each chunk i contains value (rank * world_size + i + 1)
            t = torch.zeros(
                world_size * 10,
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
            )
            with torch.no_grad():
                for i in range(world_size):
                    t[i * 10 : (i + 1) * 10] = rank * world_size + i + 1

            result = compiled_fn(t, mesh)
            self._sync_device()

            # Verify forward result: all_to_all exchanges chunks
            # After exchange, chunk i comes from rank i
            # Chunk i from rank i had value (i * world_size + rank + 1)
            self.assertEqual(result.shape[0], world_size * 10)
            for i in range(world_size):
                chunk = result[i * 10 : (i + 1) * 10]
                # Chunk i came from rank i, which had value (i * world_size + rank + 1) * 2
                expected_val = (i * world_size + rank + 1) * 2
                expected = torch.full(
                    (10,), float(expected_val), device=self.device, dtype=torch.float32
                )
                self.assertTrue(torch.allclose(chunk, expected))

            # Backward
            upstream_grad = torch.ones_like(result)
            result.backward(upstream_grad)
            self._sync_device()

            # all_to_all backward is all_to_all (inverse permutation)
            # Gradient 1 * 2 (from *2) = 2 for all elements
            self.assertIsNotNone(t.grad)
            self.assertTrue(torch.allclose(t.grad, torch.full_like(t.grad, 2.0)))
        finally:
            comm.finalize()

    def test_funcol_gather(self) -> None:
        """Test that funcol.gather dispatches to torchcomms."""
        comm = torchcomms.new_comm(self.backend, self.device, name="funcol_gather_test")

        try:
            try:
                mesh = init_native_device_mesh(
                    mesh_dim_comms=(comm,),
                    mesh_dim_names=("dp",),
                )
            except TypeError as e:
                if "_rank" in str(e):
                    return
                raise

            rank = comm.get_rank()
            world_size = comm.get_size()

            # Input tensor
            t = torch.full(
                (10, 5),
                float(rank + 1),
                device=self.device,
                dtype=torch.float32,
            )

            # Gather list on rank 0
            if rank == 0:
                gather_list = [
                    torch.zeros(10, 5, device=self.device, dtype=torch.float32)
                    for _ in range(world_size)
                ]
            else:
                gather_list = []

            # Gather to rank 0
            result = funcol.gather(t, gather_list, 0, (mesh, 0))
            self._sync_device()

            # Verify forward result on rank 0
            if rank == 0:
                self.assertEqual(len(result), world_size)
                for i, gathered in enumerate(result):
                    expected = torch.full((10, 5), float(i + 1), device=self.device)
                    self.assertTrue(torch.allclose(gathered, expected))
            else:
                # Non-dst ranks should get empty list
                self.assertEqual(len(result), 0)
        finally:
            comm.finalize()

    def test_torch_compile_funcol_gather(self) -> None:
        """Test that funcol.gather works with torch.compile.

        This validates the tracing logic in _gather that creates a dummy tensor
        on non-dst ranks to anchor the gather->wait dependency in the graph.
        """
        comm = torchcomms.new_comm(
            self.backend, self.device, name="funcol_compile_gather_test"
        )

        try:
            try:
                mesh = init_native_device_mesh(
                    mesh_dim_comms=(comm,),
                    mesh_dim_names=("dp",),
                )
            except TypeError as e:
                if "_rank" in str(e):
                    return
                raise

            rank = comm.get_rank()
            world_size = comm.get_size()

            comm.barrier(async_op=False)

            def fn(
                t: torch.Tensor, gather_list: list[torch.Tensor], mesh
            ) -> list[torch.Tensor]:
                result = funcol.gather(t, gather_list, 0, (mesh, 0))
                return result

            compiled_fn = torch.compile(
                fn,
                fullgraph=True,
                backend=self._create_graph_logging_backend(
                    "test_torch_compile_funcol_gather"
                ),
            )

            # Input tensor
            t = torch.full(
                (10, 5),
                float(rank + 1),
                device=self.device,
                dtype=torch.float32,
            )

            # Gather list - dst rank has tensors, non-dst ranks have empty list
            if rank == 0:
                gather_list = [
                    torch.zeros(10, 5, device=self.device, dtype=torch.float32)
                    for _ in range(world_size)
                ]
            else:
                gather_list = []

            result = compiled_fn(t, gather_list, mesh)
            self._sync_device()

            # Verify results
            if rank == 0:
                self.assertEqual(len(result), world_size)
                for i, gathered in enumerate(result):
                    expected = torch.full((10, 5), float(i + 1), device=self.device)
                    self.assertTrue(torch.allclose(gathered, expected))
            else:
                # Non-dst ranks should get empty list
                # (the dummy tensor used during tracing is not returned)
                self.assertEqual(len(result), 0)
        finally:
            comm.finalize()


if __name__ == "__main__":
    unittest.main()
