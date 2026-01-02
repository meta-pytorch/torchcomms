#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for coalesced collective operations in TorchComm."""

import logging
import unittest

import torch
import torchcomms
from torchcomms import ReduceOp
from torchcomms.tests.integration.py.TorchCommTestHelpers import TorchCommTestWrapper

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CoalescingTest(unittest.TestCase):
    """Test class for coalesced collective operations."""

    def get_wrapper(self):
        return TorchCommTestWrapper()

    def _create_graph_logging_backend(self, test_name: str):
        """
        Create a custom backend that captures and logs the compiled graph.

        Args:
            test_name: Name of the test for logging purposes

        Returns:
            A backend function that logs the graph and passes to inductor
        """

        def _read_and_log_inductor_artifacts():
            """Read and log inductor output_code from the code cache."""
            import torch._inductor.codecache as codecache

            logger.info(f"\n{'='*80}\nINDUCTOR OUTPUT CODE\n{'='*80}")

            # Get the cache directory and find the most recent output_code file
            try:
                import glob
                import os

                cache_dir = codecache.cache_dir()
                if cache_dir and os.path.exists(cache_dir):
                    # Find output_code*.py files (inductor's generated wrapper code)
                    pattern = os.path.join(cache_dir, "**", "*.py")
                    py_files = glob.glob(pattern, recursive=True)
                    # Sort by modification time, most recent first
                    py_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

                    for py_file in py_files[:3]:  # Log the 3 most recent
                        try:
                            with open(py_file, "r") as f:
                                code = f.read()
                            logger.info(f"\n--- {os.path.basename(py_file)} ---")
                            logger.info(code)
                        except Exception as e:
                            logger.warning(f"Could not read {py_file}: {e}")
                else:
                    logger.info(f"Cache dir not found: {cache_dir}")
            except Exception as e:
                logger.warning(f"Could not access inductor cache: {e}")

            logger.info(f"{'='*80}\n")

        def graph_capture_backend(gm, example_inputs):
            """Custom backend that captures the graph before passing to inductor."""
            logger.info(f"\n{'='*80}\nDYNAMO CAPTURED GRAPH for {test_name}\n{'='*80}")
            logger.info("\nGraph code:")
            logger.info(gm.code)
            logger.info(f"\nGraph has {len(list(gm.graph.nodes))} nodes")
            logger.info("\nGraph nodes:")

            for node in gm.graph.nodes:
                logger.info(f"  {node.op}: {node.target} -> {node.name}")

            has_torchcomms = any(
                "torchcomms" in str(node.target) for node in gm.graph.nodes
            )
            logger.info(f"\nContains torchcomms ops: {has_torchcomms}")

            # Check for custom ops that might bypass inductor
            custom_ops = [
                node
                for node in gm.graph.nodes
                if node.op == "call_function" and "torchcomms" in str(node.target)
            ]
            if custom_ops:
                logger.info(f"\nFound {len(custom_ops)} torchcomms custom ops:")
                for op in custom_ops:
                    logger.info(f"  - {op.target}")

            logger.info(f"{'='*80}\n")

            logger.info("Calling compile_fx (inductor default)...")
            logger.info(
                f"Example inputs: {[inp.shape if hasattr(inp, 'shape') else type(inp) for inp in example_inputs]}"
            )

            try:
                from torch._inductor.compile_fx import compile_fx, compile_fx_inner

                # Create a wrapper to log the post-AOT/functionalized graph
                def log_and_compile(gm, example_inputs, **kwargs):
                    """Log the graph after AOT/functionalization, then call default compiler."""
                    # Log fx_graph_readable
                    logger.info(
                        f"\n{'='*80}\nFX GRAPH READABLE for {test_name}\n{'='*80}"
                    )
                    try:
                        # print_readable() returns a string representation
                        readable = gm.print_readable(print_output=False)
                        logger.info(readable)
                    except Exception as e:
                        logger.info(f"Could not get readable graph: {e}")
                        # Fallback to code
                        logger.info(gm.code)
                    logger.info(f"{'='*80}\n")

                    # Call the default inductor compiler
                    result = compile_fx_inner(gm, example_inputs, **kwargs)

                    # Read and log the generated inductor output_code
                    _read_and_log_inductor_artifacts()

                    return result

                compiled = compile_fx(gm, example_inputs, inner_compile=log_and_compile)
                logger.info(f"Compilation complete, result type: {type(compiled)}")
                return compiled
            except Exception as e:
                logger.error(f"Compilation failed: {e}")
                import traceback

                logger.error(traceback.format_exc())
                raise

        return graph_capture_backend

    def setUp(self):
        """Set up test environment before each test."""
        # Reset dynamo state to avoid cross-test pollution
        torch._dynamo.reset()

        self.wrapper = self.get_wrapper()
        self.torchcomm = self.wrapper.get_torchcomm()
        self.rank = self.torchcomm.get_rank()
        self.num_ranks = self.torchcomm.get_size()
        self.device = self.torchcomm.get_device()

    def tearDown(self):
        """Clean up after each test."""
        # Reset dynamo state
        torch._dynamo.reset()

        self.torchcomm = None
        self.wrapper = None

    def test_coalesced_all_reduce_basic(self):
        """Test basic coalesced all_reduce with multiple tensors."""
        # Create input tensors with rank-specific values
        tensor1 = torch.full((10,), float(self.rank + 1), device=self.device)
        tensor2 = torch.full((20,), float(self.rank + 1), device=self.device)

        # Coalesce two all_reduce operations
        self.torchcomm.start_coalescing()
        self.torchcomm.all_reduce(tensor1, ReduceOp.SUM, async_op=True)
        self.torchcomm.all_reduce(tensor2, ReduceOp.SUM, async_op=True)
        work = self.torchcomm.end_coalescing()

        # Wait for completion
        if work is not None:
            work.wait()

        # Verify results - sum of ranks 0..n-1 is n*(n+1)/2 for values rank+1
        expected_sum = sum(range(1, self.num_ranks + 1))
        self.assertTrue(
            torch.allclose(tensor1, torch.full_like(tensor1, expected_sum)),
            f"tensor1 mismatch: expected {expected_sum}, got {tensor1[0].item()}",
        )
        self.assertTrue(
            torch.allclose(tensor2, torch.full_like(tensor2, expected_sum)),
            f"tensor2 mismatch: expected {expected_sum}, got {tensor2[0].item()}",
        )

    def test_coalesced_all_reduce_context_manager(self):
        """Test coalesced all_reduce using context manager."""
        tensor1 = torch.full((10,), float(self.rank + 1), device=self.device)
        tensor2 = torch.full((20,), float(self.rank + 1), device=self.device)
        tensor3 = torch.full((15,), float(self.rank + 1), device=self.device)

        with torchcomms.coalescing.coalesce(self.torchcomm) as cm:
            self.torchcomm.all_reduce(tensor1, ReduceOp.SUM, async_op=True)
            self.torchcomm.all_reduce(tensor2, ReduceOp.SUM, async_op=True)
            self.torchcomm.all_reduce(tensor3, ReduceOp.SUM, async_op=True)

        # Context manager should have populated work and tensors
        self.assertIsNotNone(cm.work)

        # Wait using the manager
        cm.wait()

        # Verify results
        expected_sum = sum(range(1, self.num_ranks + 1))
        for t in [tensor1, tensor2, tensor3]:
            self.assertTrue(
                torch.allclose(t, torch.full_like(t, expected_sum)),
                f"tensor mismatch: expected {expected_sum}, got {t[0].item()}",
            )

    def test_coalesced_mixed_collectives(self):
        """Test coalescing with different collective types."""
        # Input tensors
        reduce_tensor = torch.full((10,), float(self.rank + 1), device=self.device)
        gather_input = torch.full((5,), float(self.rank + 1), device=self.device)
        gather_output = torch.empty(5 * self.num_ranks, device=self.device)
        scatter_input = torch.full((5,), float(self.rank + 1), device=self.device)
        scatter_output = torch.empty(5 * self.num_ranks, device=self.device)

        with torchcomms.coalescing.coalesce(self.torchcomm) as cm:
            self.torchcomm.all_reduce(reduce_tensor, ReduceOp.SUM, async_op=True)
            self.torchcomm.all_gather_single(
                scatter_output, scatter_input, async_op=True
            )

        cm.wait()

        # Verify all_reduce result
        expected_sum = sum(range(1, self.num_ranks + 1))
        self.assertTrue(
            torch.allclose(reduce_tensor, torch.full_like(reduce_tensor, expected_sum)),
            f"reduce_tensor mismatch: expected {expected_sum}",
        )

        # Verify all_gather result
        for i in range(self.num_ranks):
            expected = torch.full((5,), float(i + 1), device=self.device)
            actual = scatter_output[i * 5 : (i + 1) * 5]
            self.assertTrue(
                torch.allclose(actual, expected),
                f"all_gather mismatch at rank {i}: expected {i + 1}, got {actual[0].item()}",
            )

    def test_coalesced_empty_block(self):
        """Test coalescing with no operations inside."""
        self.torchcomm.start_coalescing()
        work = self.torchcomm.end_coalescing()

        # Should return None when no operations were coalesced
        self.assertIsNone(work)

    def test_coalesced_single_operation(self):
        """Test coalescing with just one operation."""
        tensor = torch.full((10,), float(self.rank + 1), device=self.device)

        with torchcomms.coalescing.coalesce(self.torchcomm) as cm:
            self.torchcomm.all_reduce(tensor, ReduceOp.SUM, async_op=True)

        cm.wait()

        expected_sum = sum(range(1, self.num_ranks + 1))
        self.assertTrue(
            torch.allclose(tensor, torch.full_like(tensor, expected_sum)),
            f"tensor mismatch: expected {expected_sum}, got {tensor[0].item()}",
        )

    def test_coalesced_large_batch(self):
        """Test coalescing with many operations."""
        num_tensors = 10
        tensors = [
            torch.full((100,), float(self.rank + 1), device=self.device)
            for _ in range(num_tensors)
        ]

        with torchcomms.coalescing.coalesce(self.torchcomm) as cm:
            for t in tensors:
                self.torchcomm.all_reduce(t, ReduceOp.SUM, async_op=True)

        cm.wait()

        expected_sum = sum(range(1, self.num_ranks + 1))
        for i, t in enumerate(tensors):
            self.assertTrue(
                torch.allclose(t, torch.full_like(t, expected_sum)),
                f"tensor {i} mismatch: expected {expected_sum}, got {t[0].item()}",
            )

    def test_coalesced_sync_ops(self):
        """Test that sync ops inside coalescing block complete at block exit."""
        tensor1 = torch.full((10,), float(self.rank + 1), device=self.device)
        tensor2 = torch.full((20,), float(self.rank + 1), device=self.device)

        with torchcomms.coalescing.coalesce(self.torchcomm):
            # Using async_op=False - ops should complete at block exit
            self.torchcomm.all_reduce(tensor1, ReduceOp.SUM, async_op=False)
            self.torchcomm.all_reduce(tensor2, ReduceOp.SUM, async_op=False)

        # No cm.wait() needed - sync ops complete at block exit
        # Verify results are already available
        expected_sum = sum(range(1, self.num_ranks + 1))
        self.assertTrue(
            torch.allclose(tensor1, torch.full_like(tensor1, expected_sum)),
            f"tensor1 mismatch: expected {expected_sum}, got {tensor1[0].item()}",
        )
        self.assertTrue(
            torch.allclose(tensor2, torch.full_like(tensor2, expected_sum)),
            f"tensor2 mismatch: expected {expected_sum}, got {tensor2[0].item()}",
        )
        logging.info("[test] test_coalesced_sync_ops passed")

    def test_coalesced_reduce_scatter(self):
        """Test coalesced reduce_scatter operations."""
        # Each rank contributes data for all ranks
        input_tensor = torch.full(
            (10 * self.num_ranks,), float(self.rank + 1), device=self.device
        )
        output_tensor = torch.empty(10, device=self.device)

        with torchcomms.coalescing.coalesce(self.torchcomm) as cm:
            self.torchcomm.reduce_scatter_single(
                output_tensor, input_tensor, ReduceOp.SUM, async_op=True
            )

        cm.wait()

        # Each rank receives sum of all ranks' contributions
        expected_sum = sum(range(1, self.num_ranks + 1))
        self.assertTrue(
            torch.allclose(output_tensor, torch.full_like(output_tensor, expected_sum)),
            f"reduce_scatter mismatch: expected {expected_sum}, got {output_tensor[0].item()}",
        )

    def test_coalesced_mixed_ops(self):
        """Test coalescing with different operation types in the same block."""
        # Create input tensors
        all_reduce_tensor = torch.full((10,), float(self.rank + 1), device=self.device)
        broadcast_tensor = torch.full(
            (15,), float(self.rank + 1) if self.rank == 0 else 0.0, device=self.device
        )

        with torchcomms.coalescing.coalesce(self.torchcomm) as cm:
            # Mix all_reduce and broadcast in the same block
            self.torchcomm.all_reduce(all_reduce_tensor, ReduceOp.SUM, async_op=True)
            self.torchcomm.broadcast(broadcast_tensor, root=0, async_op=True)

        cm.wait()

        # Verify all_reduce result - sum of all ranks
        expected_sum = sum(range(1, self.num_ranks + 1))
        self.assertTrue(
            torch.allclose(
                all_reduce_tensor, torch.full_like(all_reduce_tensor, expected_sum)
            ),
            f"all_reduce mismatch: expected {expected_sum}, got {all_reduce_tensor[0].item()}",
        )

        # Verify broadcast result - should be rank 0's value (1.0) on all ranks
        self.assertTrue(
            torch.allclose(broadcast_tensor, torch.full_like(broadcast_tensor, 1.0)),
            f"broadcast mismatch: expected 1.0, got {broadcast_tensor[0].item()}",
        )

    def test_error_end_without_start(self):
        """Test that end_coalescing without start_coalescing raises error."""
        with self.assertRaises(RuntimeError):
            self.torchcomm.end_coalescing()

    def test_error_nested_coalescing(self):
        """Test that nested coalescing raises error."""
        self.torchcomm.start_coalescing()
        try:
            with self.assertRaises(RuntimeError):
                self.torchcomm.start_coalescing()
        finally:
            # Clean up
            self.torchcomm.end_coalescing()

    def test_coalescing_with_torch_compile(self):
        """Test coalescing with torch.compile using a custom backend that logs the FX graph."""
        import logging

        def coalesced_all_reduce(comm, tensor1, tensor2):
            """Function to compile that uses coalescing."""
            with torchcomms.coalescing.coalesce(comm) as cm:
                comm.all_reduce(tensor1, ReduceOp.SUM, async_op=True)
                comm.all_reduce(tensor2, ReduceOp.SUM, async_op=True)
            cm.wait()
            return tensor1, tensor2

        # Create input tensors
        tensor1 = torch.full((10,), float(self.rank + 1), device=self.device)
        tensor2 = torch.full((20,), float(self.rank + 1), device=self.device)

        # Compile with the graph logging backend
        compiled_fn = torch.compile(
            coalesced_all_reduce,
            backend=self._create_graph_logging_backend(
                "test_coalescing_with_torch_compile"
            ),
            fullgraph=True,
        )

        # Run the compiled function
        result1, result2 = compiled_fn(self.torchcomm, tensor1, tensor2)

        # Verify results
        expected_sum = sum(range(1, self.num_ranks + 1))
        self.assertTrue(
            torch.allclose(result1, torch.full_like(result1, expected_sum)),
            f"tensor1 mismatch: expected {expected_sum}, got {result1[0].item()}",
        )
        self.assertTrue(
            torch.allclose(result2, torch.full_like(result2, expected_sum)),
            f"tensor2 mismatch: expected {expected_sum}, got {result2[0].item()}",
        )

        logging.info("[test] test_coalescing_with_torch_compile passed")

    def test_coalescing_compile_graph_structure(self):
        """Test that compiled coalescing graph contains expected collective ops."""
        import logging

        def model_with_coalescing(comm, x, y):
            """A simple model that uses coalesced collectives."""
            # Some compute before collectives
            x = x * 2
            y = y + 1

            # Coalesced collectives
            with torchcomms.coalescing.coalesce(comm) as cm:
                comm.all_reduce(x, ReduceOp.SUM, async_op=True)
                comm.all_reduce(y, ReduceOp.SUM, async_op=True)
            cm.wait()

            # Some compute after collectives
            return x + y

        tensor1 = torch.full((10,), float(self.rank + 1), device=self.device)
        tensor2 = torch.full((10,), float(self.rank + 1), device=self.device)

        compiled_model = torch.compile(
            model_with_coalescing,
            backend=self._create_graph_logging_backend("test_coalescing_compile"),
            fullgraph=True,
        )

        result = compiled_model(self.torchcomm, tensor1, tensor2)

        # Verify the result is correct
        # x starts as (rank+1), then x*2, then all_reduce SUM
        # So x = sum over all ranks of (rank+1)*2 = 2 * sum(1..num_ranks)
        expected_x = 2 * sum(range(1, self.num_ranks + 1))
        # y starts as (rank+1), then y+1, then all_reduce SUM
        # So y = sum over all ranks of (rank+2) = sum(2..num_ranks+1)
        expected_y = sum(range(2, self.num_ranks + 2))
        expected_result = expected_x + expected_y
        self.assertTrue(
            torch.allclose(result, torch.full_like(result, expected_result)),
            f"result mismatch: expected {expected_result}, got {result[0].item()}",
        )

        logging.info("[test] test_coalescing_compile_graph_structure passed")

    def test_coalescing_compile_multiple_tensors(self):
        """Test that compiled coalescing works with multiple tensors."""
        import logging

        def coalesced_reduce(comm, tensors):
            with torchcomms.coalescing.coalesce(comm) as cm:
                for t in tensors:
                    comm.all_reduce(t, ReduceOp.SUM, async_op=True)
            cm.wait()
            return tensors

        # Use the graph logging backend
        compiled_fn = torch.compile(
            coalesced_reduce,
            backend=self._create_graph_logging_backend("test_multiple_tensors"),
            fullgraph=True,
        )

        # Run once with multiple tensors
        tensors = [
            torch.full((10,), float(self.rank + 1), device=self.device)
            for _ in range(3)
        ]
        result = compiled_fn(self.torchcomm, tensors)

        # Verify correctness
        expected_sum = sum(range(1, self.num_ranks + 1))
        for i, t in enumerate(result):
            self.assertTrue(
                torch.allclose(t, torch.full_like(t, expected_sum)),
                f"tensor {i} mismatch: expected {expected_sum}, got {t[0].item()}",
            )

        logging.info("[test] test_coalescing_compile_multiple_tensors passed")

    def test_coalescing_manager_double_wait_raises(self):
        """Test that calling wait() twice on a CoalescingManager raises an error."""
        tensor = torch.full((10,), float(self.rank + 1), device=self.device)

        with torchcomms.coalescing.coalesce(self.torchcomm) as cm:
            self.torchcomm.all_reduce(tensor, ReduceOp.SUM, async_op=True)

        # First wait should succeed
        cm.wait()

        # Second wait should raise RuntimeError
        with self.assertRaises(RuntimeError) as context:
            cm.wait()

        self.assertIn("already been called", str(context.exception))
        logging.info("[test] test_coalescing_manager_double_wait_raises passed")

    def test_unsupported_op_in_coalescing_block_raises(self):
        """Test that calling an unsupported operation inside a coalescing block raises an error."""
        # barrier does not support coalescing because it uses internal groupStart/groupEnd
        with self.assertRaises(RuntimeError) as context:
            with torchcomms.coalesce(self.torchcomm):
                self.torchcomm.barrier(async_op=True)

        self.assertIn("does not support coalescing", str(context.exception))
        logging.info("[test] test_unsupported_op_in_coalescing_block_raises passed")


if __name__ == "__main__":
    unittest.main()
