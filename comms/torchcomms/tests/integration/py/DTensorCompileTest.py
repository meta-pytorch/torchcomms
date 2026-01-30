#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Tests for DTensor operations with torch.compile using torchcomms.

These tests verify that DTensor's tensor parallelism works correctly with
torch.compile when using torchcomms as the communication backend.

NOTE: For torch.compile to work correctly with torchcomms, the device_mesh
must be passed as an explicit argument to the compiled function so it becomes
a graph input rather than a captured constant.
"""

import copy
import logging
import unittest

import torch
import torch.nn as nn

# need for compile support
import torchcomms  # noqa: F401
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torchcomms.device_mesh import init_native_device_mesh
from torchcomms.tests.integration.py.TorchCommTestHelpers import TorchCommTestWrapper


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SimpleLinear(nn.Module):
    """Simple linear layer for testing."""

    def __init__(self, in_features: int, out_features: int, device=None):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLP(nn.Module):
    """Two-layer MLP for testing tensor parallelism."""

    def __init__(self, dim: int, hidden_dim: int, device=None):
        super().__init__()
        torch.manual_seed(42)
        self.fc1 = nn.Linear(dim, hidden_dim, device=device)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, dim, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class DTensorCompileTest(unittest.TestCase):
    """Test class for DTensor operations with torch.compile using torchcomms."""

    def setUp(self):
        """Set up test environment."""
        import torch._dynamo

        torch._dynamo.reset()

        self.wrapper = TorchCommTestWrapper()
        self.comm = self.wrapper.get_torchcomm()
        self.rank = self.comm.get_rank()
        self.world_size = self.comm.get_size()
        self.device = self.comm.get_device()

        # Skip if not enough GPUs
        if self.device.type == "cuda" and torch.cuda.device_count() < 2:
            self.skipTest("Need at least 2 GPUs for DTensor tests")

        # Initialize device mesh using native torchcomms path (no ProcessGroup)
        try:
            self.device_mesh = init_native_device_mesh(
                mesh_dim_comms=(self.comm,),
                mesh_dim_names=("tp",),
            )
        except TypeError as e:
            if "_rank" in str(e):
                self.skipTest("DTensor API incompatibility, skipping")
            raise

    def _to_local_tensor(self, tensor: torch.Tensor | DTensor) -> torch.Tensor:
        """Convert DTensor to local tensor if needed."""
        if isinstance(tensor, DTensor):
            # Log the original placement for debugging
            logger.info(f"Original DTensor placements: {tensor.placements}")
            logger.info(f"Original DTensor local shape: {tensor.to_local().shape}")

            redistributed = tensor.redistribute(
                device_mesh=tensor.device_mesh,
                placements=[Replicate()],
            )

            logger.info(f"After redistribute placements: {redistributed.placements}")
            logger.info(
                f"After redistribute local shape: {redistributed.to_local().shape}"
            )

            return redistributed.to_local()
        return tensor

    @unittest.skipIf(torch.cuda.device_count() < 2, "Need at least 2 GPUs")
    def test_dtensor_redistribute_compile(self):
        """Test torch.compile with explicit DTensor redistribute operations.

        This test passes the mesh as a separate argument to ensure it becomes
        a graph input rather than being captured as a constant.
        """
        logger.info("Testing DTensor redistribute with torch.compile")

        # Create a simple DTensor
        local_tensor = torch.randn(4, 8, device=self.device)

        # Create DTensor with Shard placement
        dtensor = DTensor.from_local(
            local_tensor,
            device_mesh=self.device_mesh,
            placements=[Shard(0)],
        )

        # Compile a function that takes BOTH the DTensor and mesh as inputs
        # This ensures the mesh becomes a graph input
        @torch.compile(fullgraph=True)
        def redistribute_and_compute(dt, mesh):
            # Redistribute to replicated using the explicit mesh argument
            replicated = dt.redistribute(
                device_mesh=mesh,
                placements=[Replicate()],
            )
            # Do some computation
            result = replicated * 2 + 1
            # Redistribute back to sharded
            return result.redistribute(
                device_mesh=mesh,
                placements=[Shard(0)],
            )

        # Run compiled function with mesh as explicit argument
        result = redistribute_and_compute(dtensor, self.device_mesh)

        # Verify result
        self.assertIsInstance(result, DTensor)
        self.assertEqual(result.placements, (Shard(0),))

        # Check computation is correct
        expected_local = local_tensor * 2 + 1
        self.assertTrue(
            torch.allclose(result.to_local(), expected_local, atol=1e-5),
        )

    @unittest.skipIf(torch.cuda.device_count() < 2, "Need at least 2 GPUs")
    def test_dtensor_matmul_compile(self):
        """Test torch.compile with DTensor matrix multiplication.

        Tests sharded matrix multiplication with redistribution.
        """
        logger.info("Testing DTensor matmul with torch.compile")

        dim = 8
        mesh = self.device_mesh

        # Create weight and input tensors
        torch.manual_seed(42)
        weight = torch.randn(dim, dim, device=self.device)
        inp = torch.randn(4, dim, device=self.device)

        # Create DTensor weight sharded on dim 1 (colwise)
        weight_dt = DTensor.from_local(
            weight,
            device_mesh=mesh,
            placements=[Shard(1)],
        )

        # Create replicated input DTensor
        inp_dt = DTensor.from_local(
            inp,
            device_mesh=mesh,
            placements=[Replicate()],
            run_check=False,
        )

        @torch.compile(fullgraph=True)
        def matmul_fn(x, w, mesh):
            # Matrix multiplication with DTensors
            result = torch.matmul(x, w)
            return result

        # Run compiled function with mesh as explicit argument
        result = matmul_fn(inp_dt, weight_dt, mesh)

        # Verify result (result should be Partial since we're doing colwise matmul)
        self.assertIsInstance(result, DTensor)

    @unittest.skipIf(torch.cuda.device_count() < 2, "Need at least 2 GPUs")
    def test_colwise_parallel_eager(self):
        """Test ColwiseParallel in eager mode (no compile) for baseline."""
        logger.info("Testing ColwiseParallel in eager mode")

        # Scale hidden_dim with world_size to ensure reasonable chunk sizes
        dim = 8
        hidden_dim = 16 * self.world_size  # Ensure divisible by world_size

        # Seed before model creation so all ranks have identical weights
        torch.manual_seed(42)
        model = SimpleLinear(dim, hidden_dim, device=self.device)
        ref_model = copy.deepcopy(model)

        # Parallelize model
        parallelize_module(
            model,
            self.device_mesh,
            {"linear": ColwiseParallel(use_local_output=False)},
        )

        # Create input
        torch.manual_seed(0)
        inp = torch.randn(4, dim, device=self.device)

        # Run eager forward (no compile)
        out = model(inp)
        ref_out = ref_model(inp)

        # Convert DTensor output to local and compare
        out_local = self._to_local_tensor(out)

        # ColwiseParallel shards output, so we need to gather
        self.assertEqual(out_local.shape, ref_out.shape)
        self.assertTrue(torch.allclose(out_local, ref_out, atol=1e-5))

    @unittest.skipIf(torch.cuda.device_count() < 2, "Need at least 2 GPUs")
    def test_mlp_tp_eager(self):
        """Test MLP tensor parallelism in eager mode (no compile) for baseline."""
        logger.info("Testing MLP tensor parallel in eager mode")

        # Scale hidden_dim with world_size to ensure reasonable chunk sizes
        dim = 8
        hidden_dim = 16 * self.world_size
        model = MLP(dim, hidden_dim, device=self.device)
        ref_model = copy.deepcopy(model)

        # Parallelize with ColwiseParallel -> RowwiseParallel pattern
        parallelize_module(
            model,
            self.device_mesh,
            {
                "fc1": ColwiseParallel(),
                "fc2": RowwiseParallel(use_local_output=False),
            },
        )

        # Create input
        torch.manual_seed(0)
        inp = torch.randn(4, dim, device=self.device)

        # Run eager forward (no compile)
        out = model(inp)
        ref_out = ref_model(inp)

        # RowwiseParallel produces replicated output
        out_local = self._to_local_tensor(out)

        self.assertEqual(out_local.shape, ref_out.shape)
        self.assertTrue(torch.allclose(out_local, ref_out, atol=1e-5))

    @unittest.skipIf(torch.cuda.device_count() < 2, "Need at least 2 GPUs")
    def test_mlp_tp_backward_eager(self):
        """Test MLP tensor parallelism backward in eager mode."""
        logger.info("Testing MLP tensor parallel backward in eager mode")

        # Scale hidden_dim with world_size to ensure reasonable chunk sizes
        dim = 8
        hidden_dim = 16 * self.world_size
        model = MLP(dim, hidden_dim, device=self.device)
        ref_model = copy.deepcopy(model)

        # Parallelize with ColwiseParallel -> RowwiseParallel pattern
        parallelize_module(
            model,
            self.device_mesh,
            {
                "fc1": ColwiseParallel(),
                "fc2": RowwiseParallel(use_local_output=False),
            },
        )

        # Create input
        torch.manual_seed(0)
        inp = torch.randn(4, dim, device=self.device, requires_grad=True)
        ref_inp = inp.clone().detach().requires_grad_(True)

        # Forward and backward (eager)
        out = model(inp)
        ref_out = ref_model(ref_inp)

        out_local = self._to_local_tensor(out)
        loss = out_local.sum()
        ref_loss = ref_out.sum()

        loss.backward()
        ref_loss.backward()

        # Compare gradients
        self.assertTrue(
            torch.allclose(inp.grad, ref_inp.grad, atol=1e-5),
            f"Input gradients don't match: {inp.grad} vs {ref_inp.grad}",
        )

    @unittest.skipIf(torch.cuda.device_count() < 2, "Need at least 2 GPUs")
    def test_dtensor_from_local_shard_compile(self):
        """Test creating sharded DTensor and operations inside compiled function."""
        logger.info("Testing DTensor from_local with Shard in compiled function")

        mesh = self.device_mesh

        @torch.compile(fullgraph=True)
        def fn(local_tensor, mesh):
            # Create DTensor inside compiled function
            dt = DTensor.from_local(
                local_tensor,
                device_mesh=mesh,
                placements=[Shard(0)],
            )
            # Do computation
            result = dt * 2
            return result.to_local()

        local_tensor = torch.randn(4, 8, device=self.device)
        result = fn(local_tensor, mesh)

        expected = local_tensor * 2
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    @unittest.skipIf(torch.cuda.device_count() < 2, "Need at least 2 GPUs")
    def test_dtensor_from_local_replicate_compile(self):
        """Test creating replicated DTensor inside compiled function."""
        logger.info("Testing DTensor from_local with Replicate in compiled function")

        mesh = self.device_mesh

        @torch.compile(fullgraph=True)
        def fn(local_tensor, mesh):
            # Create replicated DTensor inside compiled function
            dt = DTensor.from_local(
                local_tensor,
                device_mesh=mesh,
                placements=[Replicate()],
                run_check=False,
            )
            # Do computation
            result = dt + 1
            return result.to_local()

        local_tensor = torch.randn(4, 8, device=self.device)
        result = fn(local_tensor, mesh)

        expected = local_tensor + 1
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
