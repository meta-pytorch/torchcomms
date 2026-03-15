# Copyright (c) Meta Platforms, Inc. and affiliates.
# pyre-unsafe
"""
Unit tests for device_alltoallv_dynamic kernel implementation.

This module tests:
1. API availability and exports
2. compute_offsets_from_sizes helper function
3. Module structure and imports

Run with:
    buck2 test //comms/pipes/collectives/triton/tests:test_device_alltoallv_dynamic
"""

import sys
import unittest

import torch
from torch.utils._triton import has_triton


TRITON_AVAILABLE = has_triton()
CUDA_AVAILABLE = torch.cuda.is_available()


# =============================================================================
# API Availability Tests
# =============================================================================


class TestDeviceAlltoallvDynamicAPIAvailability(unittest.TestCase):
    """Tests for device_alltoallv_dynamic API availability and exports."""

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")
    def test_device_alltoallv_dynamic_importable(self) -> None:
        """Test that device_alltoallv_dynamic can be imported."""
        from comms.pipes.collectives.triton import device_alltoallv_dynamic

        self.assertTrue(callable(device_alltoallv_dynamic))

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")
    def test_compute_offsets_from_sizes_importable(self) -> None:
        """Test that compute_offsets_from_sizes can be imported."""
        from comms.pipes.collectives.triton import compute_offsets_from_sizes

        self.assertTrue(callable(compute_offsets_from_sizes))

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")
    def test_all_exports_present(self) -> None:
        """Test that all expected functions are in __all__."""
        from comms.pipes.collectives import triton as collectives

        expected_exports = [
            "device_alltoallv_dynamic",
            "compute_offsets_from_sizes",
            "exchange_offsets",
        ]

        for export in expected_exports:
            self.assertIn(
                export,
                collectives.__all__,
                f"Expected export '{export}' missing from __all__",
            )

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")
    def test_module_docstring_present(self) -> None:
        """Test that the module has a docstring."""
        from comms.pipes.collectives.triton import device_alltoallv_dynamic as mod

        # The module should have documentation
        self.assertIsNotNone(mod.__doc__)


# =============================================================================
# compute_offsets_from_sizes Tests
# =============================================================================


class TestComputeOffsetsFromSizes(unittest.TestCase):
    """Tests for the compute_offsets_from_sizes helper function."""

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")
    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA not available")
    def test_basic_offsets_computation(self) -> None:
        """Test basic exclusive prefix sum computation."""
        from comms.pipes.collectives.triton import compute_offsets_from_sizes

        # Input: sizes for 4 peers
        sizes = torch.tensor([100, 200, 150, 250], dtype=torch.int64, device="cuda")
        offsets = torch.zeros_like(sizes)

        # Compute offsets
        compute_offsets_from_sizes(sizes, offsets)

        # Expected: [0, 100, 300, 450]
        # offset[0] = 0 (start)
        # offset[1] = 0 + 100 = 100
        # offset[2] = 0 + 100 + 200 = 300
        # offset[3] = 0 + 100 + 200 + 150 = 450
        expected = torch.tensor([0, 100, 300, 450], dtype=torch.int64, device="cuda")

        torch.testing.assert_close(offsets, expected)

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")
    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA not available")
    def test_offsets_with_zeros(self) -> None:
        """Test offset computation when some sizes are zero."""
        from comms.pipes.collectives.triton import compute_offsets_from_sizes

        # Some peers have zero-length messages
        sizes = torch.tensor([100, 0, 150, 0, 250], dtype=torch.int64, device="cuda")
        offsets = torch.zeros_like(sizes)

        compute_offsets_from_sizes(sizes, offsets)

        # Expected: [0, 100, 100, 250, 250]
        expected = torch.tensor(
            [0, 100, 100, 250, 250], dtype=torch.int64, device="cuda"
        )

        torch.testing.assert_close(offsets, expected)

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")
    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA not available")
    def test_offsets_single_element(self) -> None:
        """Test offset computation with single element."""
        from comms.pipes.collectives.triton import compute_offsets_from_sizes

        sizes = torch.tensor([500], dtype=torch.int64, device="cuda")
        offsets = torch.zeros_like(sizes)

        compute_offsets_from_sizes(sizes, offsets)

        # Single element always starts at 0
        expected = torch.tensor([0], dtype=torch.int64, device="cuda")

        torch.testing.assert_close(offsets, expected)

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")
    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA not available")
    def test_offsets_all_same_size(self) -> None:
        """Test offset computation with uniform sizes."""
        from comms.pipes.collectives.triton import compute_offsets_from_sizes

        # All peers have same size (256 bytes each)
        sizes = torch.full((8,), 256, dtype=torch.int64, device="cuda")
        offsets = torch.zeros_like(sizes)

        compute_offsets_from_sizes(sizes, offsets)

        # Expected: [0, 256, 512, 768, 1024, 1280, 1536, 1792]
        expected = torch.arange(8, dtype=torch.int64, device="cuda") * 256

        torch.testing.assert_close(offsets, expected)

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")
    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA not available")
    def test_offsets_large_values(self) -> None:
        """Test offset computation with large byte counts (>4GB total)."""
        from comms.pipes.collectives.triton import compute_offsets_from_sizes

        # Large sizes that exceed 32-bit range when summed
        one_gb = 1024 * 1024 * 1024
        sizes = torch.tensor(
            [one_gb, one_gb, one_gb, one_gb, one_gb],
            dtype=torch.int64,
            device="cuda",
        )
        offsets = torch.zeros_like(sizes)

        compute_offsets_from_sizes(sizes, offsets)

        # Expected: [0, 1GB, 2GB, 3GB, 4GB]
        expected = torch.tensor(
            [0, one_gb, 2 * one_gb, 3 * one_gb, 4 * one_gb],
            dtype=torch.int64,
            device="cuda",
        )

        torch.testing.assert_close(offsets, expected)

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")
    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA not available")
    def test_offsets_preserves_input(self) -> None:
        """Test that compute_offsets_from_sizes doesn't modify the input sizes."""
        from comms.pipes.collectives.triton import compute_offsets_from_sizes

        sizes = torch.tensor([100, 200, 300], dtype=torch.int64, device="cuda")
        sizes_original = sizes.clone()
        offsets = torch.zeros_like(sizes)

        compute_offsets_from_sizes(sizes, offsets)

        # Sizes should be unchanged
        torch.testing.assert_close(sizes, sizes_original)


# =============================================================================
# Kernel Structure Tests
# =============================================================================


class TestKernelStructure(unittest.TestCase):
    """Tests for kernel function structure and decorators."""

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")
    def test_non_pipelined_kernel_exists(self) -> None:
        """Test that the non-pipelined kernel function exists."""
        from comms.pipes.collectives.triton.device_alltoallv_dynamic import (
            _device_alltoallv_dynamic_kernel,
        )

        # The kernel exists if we can import it without error
        # With @requires_torchcomms decorator, it may not be directly callable
        self.assertIsNotNone(_device_alltoallv_dynamic_kernel)

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")
    def test_compute_offsets_kernel_exists(self) -> None:
        """Test that the offset computation helper exists."""
        from comms.pipes.collectives.triton import compute_offsets_from_sizes

        # The public API is compute_offsets_from_sizes (uses PyTorch cumsum internally)
        self.assertTrue(callable(compute_offsets_from_sizes))


# =============================================================================
# Function Signature Tests
# =============================================================================


class TestFunctionSignatures(unittest.TestCase):
    """Tests for function signatures and required parameters."""

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")
    def test_device_alltoallv_dynamic_signature(self) -> None:
        """Test device_alltoallv_dynamic has expected parameters."""
        import inspect

        from comms.pipes.collectives.triton import device_alltoallv_dynamic

        sig = inspect.signature(device_alltoallv_dynamic)
        params = list(sig.parameters.keys())

        expected_params = [
            "send_buf",
            "recv_buf",
            "send_sizes",
            "send_offsets",
            "recv_sizes",
            "local_recv_slot_offsets",
            "remote_write_offsets",
            "dev_win_ptr",
            "src_info",
            "my_rank",
            "world_size",
            "num_warps",
        ]

        for param in expected_params:
            self.assertIn(
                param, params, f"Expected parameter '{param}' not in signature"
            )

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")
    def test_compute_offsets_from_sizes_signature(self) -> None:
        """Test compute_offsets_from_sizes has expected parameters."""
        import inspect

        from comms.pipes.collectives.triton import compute_offsets_from_sizes

        sig = inspect.signature(compute_offsets_from_sizes)
        params = list(sig.parameters.keys())

        expected_params = ["sizes", "offsets"]

        self.assertEqual(params, expected_params)


# =============================================================================
# Docstring Tests
# =============================================================================


class TestDocstrings(unittest.TestCase):
    """Tests for function documentation."""

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")
    def test_device_alltoallv_dynamic_has_docstring(self) -> None:
        """Test device_alltoallv_dynamic has documentation."""
        from comms.pipes.collectives.triton import device_alltoallv_dynamic

        self.assertIsNotNone(device_alltoallv_dynamic.__doc__)
        self.assertGreater(len(device_alltoallv_dynamic.__doc__), 100)

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")
    def test_compute_offsets_from_sizes_has_docstring(self) -> None:
        """Test compute_offsets_from_sizes has documentation."""
        from comms.pipes.collectives.triton import compute_offsets_from_sizes

        self.assertIsNotNone(compute_offsets_from_sizes.__doc__)


# =============================================================================
# Multi-dtype Support Tests
# =============================================================================


class TestMultiDtypeSupport(unittest.TestCase):
    """Tests that element_size() plumbing works for all supported dtypes."""

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")
    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA not available")
    def test_element_size_float32(self) -> None:
        """Verify element_size() returns 4 for float32 tensors."""
        buf = torch.zeros(16, dtype=torch.float32, device="cuda")
        self.assertEqual(buf.element_size(), 4)

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")
    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA not available")
    def test_element_size_float16(self) -> None:
        """Verify element_size() returns 2 for float16 tensors."""
        buf = torch.zeros(16, dtype=torch.float16, device="cuda")
        self.assertEqual(buf.element_size(), 2)

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")
    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA not available")
    def test_element_size_bfloat16(self) -> None:
        """Verify element_size() returns 2 for bfloat16 tensors."""
        buf = torch.zeros(16, dtype=torch.bfloat16, device="cuda")
        self.assertEqual(buf.element_size(), 2)

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")
    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA not available")
    def test_element_size_float64(self) -> None:
        """Verify element_size() returns 8 for float64 tensors."""
        buf = torch.zeros(16, dtype=torch.float64, device="cuda")
        self.assertEqual(buf.element_size(), 8)

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")
    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA not available")
    def test_element_size_int8(self) -> None:
        """Verify element_size() returns 1 for int8 tensors."""
        buf = torch.zeros(16, dtype=torch.int8, device="cuda")
        self.assertEqual(buf.element_size(), 1)

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")
    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA not available")
    def test_compute_offsets_works_with_int32_sizes(self) -> None:
        """Test compute_offsets_from_sizes works with int32 size tensors."""
        from comms.pipes.collectives.triton import compute_offsets_from_sizes

        sizes = torch.tensor([100, 200, 150], dtype=torch.int32, device="cuda")
        offsets = torch.zeros_like(sizes)
        compute_offsets_from_sizes(sizes, offsets)

        expected = torch.tensor([0, 100, 300], dtype=torch.int32, device="cuda")
        torch.testing.assert_close(offsets, expected)


# =============================================================================
# Dependency Tests (verify required APIs are available)
# =============================================================================


class TestRequiredAPIsAvailable(unittest.TestCase):
    """Tests that required TorchComms APIs are available."""

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")
    def test_wait_signal_from_available(self) -> None:
        """Test that wait_signal_from API is available (required for correctness)."""
        from torchcomms.triton.fb import wait_signal_from

        self.assertTrue(callable(wait_signal_from))

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")
    def test_put_block_available(self) -> None:
        """Test that put_block API is available."""
        from torchcomms.triton.fb import put_block

        self.assertTrue(callable(put_block))

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")
    def test_flush_block_available(self) -> None:
        """Test that flush_block API is available."""
        from torchcomms.triton.fb import flush_block

        self.assertTrue(callable(flush_block))

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")
    def test_requires_torchcomms_decorator_available(self) -> None:
        """Test that requires_torchcomms decorator is available."""
        from torchcomms.triton.fb import requires_torchcomms

        self.assertTrue(callable(requires_torchcomms))


# =============================================================================
# Monotonic Signal Counter Tests
# =============================================================================


class TestMonotonicSignalCounter(unittest.TestCase):
    """Tests for monotonic signal counter behavior."""

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")
    def test_completion_counters_initialized_to_zero(self) -> None:
        """Test that completion counters start at zero."""
        # Clear cache to ensure fresh counters
        from comms.pipes.collectives.triton.device_alltoallv_dynamic import (
            _COMPLETION_COUNTERS_CACHE,
            _get_completion_counters,
        )

        _COMPLETION_COUNTERS_CACHE.clear()

        if not CUDA_AVAILABLE:
            return

        world_size = 8
        device = torch.device("cuda")

        counters = _get_completion_counters(world_size, device)

        # Counters should be zeros
        expected = torch.zeros(world_size, dtype=torch.int32, device=device)
        torch.testing.assert_close(counters, expected)

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")
    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA not available")
    def test_completion_counters_cached(self) -> None:
        """Test that completion counters are cached and reused."""
        from comms.pipes.collectives.triton.device_alltoallv_dynamic import (
            _COMPLETION_COUNTERS_CACHE,
            _get_completion_counters,
        )

        _COMPLETION_COUNTERS_CACHE.clear()

        world_size = 4
        device = torch.device("cuda")

        counters1 = _get_completion_counters(world_size, device)
        counters2 = _get_completion_counters(world_size, device)

        # Should return the same tensor (same data_ptr)
        self.assertEqual(counters1.data_ptr(), counters2.data_ptr())


# =============================================================================
# Tests: sync_buffer API
# =============================================================================


class TestSyncBufferAPIAvailability(unittest.TestCase):
    """Tests for sync_buffer API availability and parameter handling."""

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")
    def test_sync_buffer_parameter_exists(self) -> None:
        """Test that device_alltoallv_dynamic accepts sync_buffer parameter."""
        import inspect

        from comms.pipes.collectives.triton import device_alltoallv_dynamic

        sig = inspect.signature(device_alltoallv_dynamic)
        self.assertIn(
            "sync_buffer",
            sig.parameters,
            "device_alltoallv_dynamic should accept sync_buffer parameter",
        )

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")
    def test_sync_buffer_default_is_true(self) -> None:
        """Test that sync_buffer defaults to True."""
        import inspect

        from comms.pipes.collectives.triton import device_alltoallv_dynamic

        sig = inspect.signature(device_alltoallv_dynamic)
        sync_buffer_param = sig.parameters["sync_buffer"]
        self.assertEqual(
            sync_buffer_param.default,
            True,
            "sync_buffer should default to True",
        )

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")
    def test_sync_buffer_documented(self) -> None:
        """Test that sync_buffer is documented in the function docstring."""
        from comms.pipes.collectives.triton import device_alltoallv_dynamic

        docstring = device_alltoallv_dynamic.__doc__
        self.assertIsNotNone(docstring, "Function should have a docstring")
        self.assertIn(
            "sync_buffer",
            docstring,
            "sync_buffer should be documented in the docstring",
        )


class TestSyncBufferAlltoallvOp(unittest.TestCase):
    """Tests for sync_buffer in AlltoallvOp class."""

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")
    def test_alltoallv_op_sync_buffer_parameter(self) -> None:
        """Test that AlltoallvOp accepts sync_buffer parameter."""
        import inspect

        from comms.pipes.collectives.triton import AlltoallvOp

        sig = inspect.signature(AlltoallvOp.__init__)
        self.assertIn(
            "sync_buffer",
            sig.parameters,
            "AlltoallvOp.__init__ should accept sync_buffer parameter",
        )

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")
    def test_alltoallv_op_sync_buffer_default_is_true(self) -> None:
        """Test that AlltoallvOp sync_buffer defaults to True."""
        import inspect

        from comms.pipes.collectives.triton import AlltoallvOp

        sig = inspect.signature(AlltoallvOp.__init__)
        sync_buffer_param = sig.parameters["sync_buffer"]
        self.assertEqual(
            sync_buffer_param.default,
            True,
            "sync_buffer should default to True",
        )

    @unittest.skipUnless(TRITON_AVAILABLE, "Triton not available")
    def test_alltoallv_op_get_or_create_accepts_sync_buffer(self) -> None:
        """Test that AlltoallvOp.get_or_create accepts sync_buffer parameter."""
        import inspect

        from comms.pipes.collectives.triton import AlltoallvOp

        sig = inspect.signature(AlltoallvOp.get_or_create)
        self.assertIn(
            "sync_buffer",
            sig.parameters,
            "AlltoallvOp.get_or_create should accept sync_buffer parameter",
        )


def main() -> int:
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(
        loader.loadTestsFromTestCase(TestDeviceAlltoallvDynamicAPIAvailability)
    )
    suite.addTests(loader.loadTestsFromTestCase(TestComputeOffsetsFromSizes))
    suite.addTests(loader.loadTestsFromTestCase(TestKernelStructure))
    suite.addTests(loader.loadTestsFromTestCase(TestFunctionSignatures))
    suite.addTests(loader.loadTestsFromTestCase(TestDocstrings))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiDtypeSupport))
    suite.addTests(loader.loadTestsFromTestCase(TestRequiredAPIsAvailable))
    suite.addTests(loader.loadTestsFromTestCase(TestMonotonicSignalCounter))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
