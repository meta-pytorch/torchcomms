#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import unittest

os.environ.setdefault("NCCL_COMM_STATE_DEBUG_TOPO", "nolocal")

import torch
from torchcomms.tests.integration.py.TorchCommTestHelpers import TorchCommTestWrapper


class RegisterAllTest(unittest.TestCase):
    """Test class for registerAll operations in TorchComm."""

    # Class variables for test parameters
    counts = [10 * 1024 * 1024]
    dtypes = [torch.float]
    num_iterations = 5
    iterations_before_empty = 3
    iterations_after_empty = 3

    # Class-level communicator (shared across all tests)
    _wrapper = None
    _torchcomm = None
    _ncclx_backend = None
    _rank = None
    _num_ranks = None
    _device = None

    @classmethod
    def setUpClass(cls) -> None:
        """Set up test environment once for all tests."""
        cls._wrapper = TorchCommTestWrapper()
        cls._torchcomm = cls._wrapper.get_torchcomm()
        cls._rank = cls._torchcomm.get_rank()
        cls._num_ranks = cls._torchcomm.get_size()
        cls._device = cls._torchcomm.get_device()

        # Get the NCCLX backend if available
        cls._ncclx_backend = None
        backend = cls._torchcomm.get_backend()
        if backend == "ncclx":
            cls._ncclx_backend = cls._torchcomm.unsafe_get_backend()

    @classmethod
    def tearDownClass(cls) -> None:
        """Clean up after all tests."""
        cls._ncclx_backend = None
        if cls._torchcomm is not None:
            cls._torchcomm.finalize()
        cls._torchcomm = None
        # Clear wrapper's torchcomm reference to prevent double-finalize in __del__
        if cls._wrapper is not None:
            cls._wrapper.torchcomm = None
        cls._wrapper = None

    def setUp(self) -> None:
        """Set up test environment before each test."""
        # Use class-level communicator
        self.wrapper = self._wrapper
        self.torchcomm = self._torchcomm
        self.rank = self._rank
        self.num_ranks = self._num_ranks
        self.device = self._device
        self.ncclx_backend = self._ncclx_backend

    def tearDown(self) -> None:
        """Clean up after each test."""
        # Synchronize GPU after each test
        torch.cuda.synchronize()

    def _create_input_tensor(self, count: int, dtype: torch.dtype) -> torch.Tensor:
        """Create input tensor with rank-specific values. Multiply in place to avoid extra allocations."""
        return torch.ones(count * self.num_ranks, dtype=dtype, device=self.device).mul_(
            self.rank + 1
        )

    def _create_output_tensor(self, count: int, dtype: torch.dtype) -> torch.Tensor:
        """Create output tensor to store results."""
        options = {"dtype": dtype, "device": self.device}
        return torch.zeros(count * self.num_ranks, **options)

    def _verify_results(self, output_tensor: torch.Tensor) -> None:
        """Verify the results of the all_to_all_single operation."""
        count = output_tensor.numel() // self.num_ranks

        for i in range(self.num_ranks):
            section = output_tensor[i * count : (i + 1) * count]
            expected = torch.ones(count, dtype=section.dtype) * float(i + 1)
            self.assertTrue(
                torch.allclose(section.cpu(), expected),
                f"Tensors not close enough for rank {i} section",
            )

    def _run_all_to_all_single(self, count: int, dtype: torch.dtype) -> None:
        """Run a basic all_to_all_single operation."""
        input_tensor = self._create_input_tensor(count, dtype)
        output_tensor = self._create_output_tensor(count, dtype)

        work = self.torchcomm.all_to_all_single(output_tensor, input_tensor, False)
        work.wait()

        self._verify_results(output_tensor)

    def _test_register_all_before_all_to_all(
        self, count: int, dtype: torch.dtype
    ) -> None:
        """Test registerAll before all_to_all_single."""
        input_tensor = self._create_input_tensor(count, dtype)
        output_tensor = self._create_output_tensor(count, dtype)

        if self.ncclx_backend is not None:
            self.ncclx_backend.register_all()

        work = self.torchcomm.all_to_all_single(output_tensor, input_tensor, False)
        work.wait()

        self._verify_results(output_tensor)

    def _test_multiple_all_to_all_operations(
        self, count: int, dtype: torch.dtype, num_iterations: int
    ) -> None:
        """Test multiple all_to_all_single operations."""
        input_tensor = self._create_input_tensor(count, dtype)
        output_tensor = self._create_output_tensor(count, dtype)

        if self.ncclx_backend is not None:
            self.ncclx_backend.register_all()

        for _iteration in range(num_iterations):
            output_tensor.zero_()

            work = self.torchcomm.all_to_all_single(output_tensor, input_tensor, False)
            work.wait()

            self._verify_results(output_tensor)

    def _test_empty_cache_after_all_to_all(
        self,
        count: int,
        dtype: torch.dtype,
        iterations_before_empty: int,
        iterations_after_empty: int,
    ) -> None:
        """Test empty_cache after all_to_all_single."""
        # Phase 1: Run iterations with allocation inside the loop
        for _iteration in range(iterations_before_empty):
            input_tensor = self._create_input_tensor(count, dtype)
            output_tensor = self._create_output_tensor(count, dtype)

            if self.ncclx_backend is not None:
                self.ncclx_backend.register_all()

            work = self.torchcomm.all_to_all_single(output_tensor, input_tensor, False)
            work.wait()

            self._verify_results(output_tensor)

        # Synchronize GPU before empty_cache
        torch.cuda.synchronize()

        # Call empty_cache - tensors from the loop are now out of scope
        # This should trigger SEGMENT_FREE events
        torch.cuda.empty_cache()

        # Phase 2: Run more iterations after empty_cache
        for _iteration in range(iterations_after_empty):
            input_tensor = self._create_input_tensor(count, dtype)
            output_tensor = self._create_output_tensor(count, dtype)

            if self.ncclx_backend is not None:
                self.ncclx_backend.register_all()

            work = self.torchcomm.all_to_all_single(output_tensor, input_tensor, False)
            work.wait()

            self._verify_results(output_tensor)

    def test_basic_register_all_before_all_to_all(self) -> None:
        """Test basic registerAll before all_to_all_single."""
        for count in self.counts:
            for dtype in self.dtypes:
                self._test_register_all_before_all_to_all(count, dtype)

    def test_multiple_all_to_all_operations(self) -> None:
        """Test multiple all_to_all_single operations with registerAll."""
        for count in self.counts:
            for dtype in self.dtypes:
                self._test_multiple_all_to_all_operations(
                    count, dtype, self.num_iterations
                )

    def test_empty_cache_after_all_to_all(self) -> None:
        """Test empty_cache after all_to_all_single with registerAll."""
        for count in self.counts:
            for dtype in self.dtypes:
                self._test_empty_cache_after_all_to_all(
                    count,
                    dtype,
                    self.iterations_before_empty,
                    self.iterations_after_empty,
                )

    def test_small_tensor_allocation_before_all_to_all(self) -> None:
        """Test registerAll with small tensors allocated before input/output tensors.

        PyTorch's caching allocator uses different memory pools for small vs large
        tensors (small pool for tensors < 1MB, large pool for >= 1MB). This test
        allocates small tensors first to ensure the segment cache is non-contiguous,
        which may expose bugs in registerAll when handling fragmented memory segments.
        """
        small_tensor_sizes = [256, 512, 1024]  # Small tensors (a few KB each)
        num_small_tensors = 3

        for count in self.counts:
            for dtype in self.dtypes:
                # Allocate small tensors first to fragment the memory pool
                # These go to PyTorch's small allocation pool (< 1MB)
                small_tensors = []
                for i, size in enumerate(small_tensor_sizes[:num_small_tensors]):
                    small_tensor = torch.ones(
                        size, dtype=dtype, device=self.device
                    ).mul_(i + 1)
                    small_tensors.append(small_tensor)

                # Now allocate larger input/output tensors (these go to large pool)
                input_tensor = self._create_input_tensor(count, dtype)
                output_tensor = self._create_output_tensor(count, dtype)

                # Call registerAll - this should handle non-contiguous segments
                if self.ncclx_backend is not None:
                    self.ncclx_backend.register_all()

                # Run all_to_all_single
                work = self.torchcomm.all_to_all_single(
                    output_tensor, input_tensor, False
                )
                work.wait()

                # Verify results
                self._verify_results(output_tensor)

                # Keep small tensors alive until after the operation completes
                # to ensure the memory fragmentation persists during registerAll
                del small_tensors

    def test_multi_stream_and_pool_allocation(self) -> None:
        """Test registerAll with tensors allocated on different streams and pool sizes.

        PyTorch's caching allocator organizes memory by:
        1. Stream: Different CUDA streams may use different memory blocks
        2. Pool size: Small pool (<=1MB) vs Large pool (>1MB)

        This test creates tensors on multiple streams and with different sizes to
        verify that registerAll correctly handles non-contiguous memory regions
        caused by different streams and pool types.
        """
        # Small tensors go to small pool (<=1MB = 1048576 bytes)
        # Large tensors go to large pool (>1MB)
        small_tensor_size = 1024  # ~1KB per element, well under 1MB
        large_tensor_size = 2 * 1024 * 1024  # 2MB worth of elements

        num_streams = 3
        dtype = torch.float32  # 4 bytes per element

        # Create multiple CUDA streams
        streams = [torch.cuda.Stream(device=self.device) for _ in range(num_streams)]

        # Keep all tensors alive
        all_tensors = []

        # Allocate tensors on different streams with different sizes
        for stream_idx, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                # Allocate a small tensor on this stream (goes to small pool)
                small_tensor = torch.ones(
                    small_tensor_size, dtype=dtype, device=self.device
                ).mul_(stream_idx + 1)
                all_tensors.append(small_tensor)

                # Allocate a large tensor on this stream (goes to large pool)
                large_tensor = torch.ones(
                    large_tensor_size, dtype=dtype, device=self.device
                ).mul_(stream_idx + 100)
                all_tensors.append(large_tensor)

        # Synchronize all streams before registerAll
        for stream in streams:
            stream.synchronize()

        # Allocate input/output tensors on default stream for the collective
        for count in self.counts:
            input_tensor = self._create_input_tensor(count, torch.float)
            output_tensor = self._create_output_tensor(count, torch.float)

            # Call registerAll - should handle multi-stream, multi-pool segments
            if self.ncclx_backend is not None:
                self.ncclx_backend.register_all()

            # Run all_to_all_single
            work = self.torchcomm.all_to_all_single(output_tensor, input_tensor, False)
            work.wait()

            # Verify results
            self._verify_results(output_tensor)

        # Keep all tensors alive until test completion
        del all_tensors

    def test_interleaved_stream_pool_allocation(self) -> None:
        """Test registerAll with interleaved allocations across streams and pools.

        This test interleaves allocations between small and large pools across
        different streams to create a maximally fragmented memory layout. This
        tests the robustness of registerAll in handling complex allocation patterns.
        """
        num_iterations = 4
        dtype = torch.float32  # 4 bytes per element

        # Sizes that cross the 1MB boundary
        sizes = [
            256,  # Small pool: ~1KB
            512 * 1024,  # Small pool: ~512KB (under 1MB)
            2 * 1024 * 1024,  # Large pool: 2MB
            4 * 1024 * 1024,  # Large pool: 4MB
        ]

        # Create two streams
        stream_a = torch.cuda.Stream(device=self.device)
        stream_b = torch.cuda.Stream(device=self.device)

        all_tensors = []

        for i in range(num_iterations):
            # Alternate between streams
            stream = stream_a if i % 2 == 0 else stream_b

            # Cycle through different sizes
            size = sizes[i % len(sizes)]

            with torch.cuda.stream(stream):
                tensor = torch.ones(size, dtype=dtype, device=self.device).mul_(i + 1)
                all_tensors.append(tensor)

        # Synchronize streams
        stream_a.synchronize()
        stream_b.synchronize()

        # Allocate and run collective
        for count in self.counts:
            input_tensor = self._create_input_tensor(count, torch.float)
            output_tensor = self._create_output_tensor(count, torch.float)

            if self.ncclx_backend is not None:
                self.ncclx_backend.register_all()

            work = self.torchcomm.all_to_all_single(output_tensor, input_tensor, False)
            work.wait()

            self._verify_results(output_tensor)

        del all_tensors

    def test_model_training_simulation_best_path(self) -> None:
        """Test best-case model training workflow with registerAll.

        Best path scenario:
        1. Allocate memory (tensors for model weights/activations)
        2. Run all_to_all for warmup iterations using default NCCL path
        3. Call registerAll to register all cached memory segments
        4. Continue running all_to_all reusing the SAME tensors (ctran path)

        This is the best case because all memory is already registered after
        registerAll, so post-registration iterations can fully utilize ctran
        without any dynamic registration overhead.
        """
        warmup_iterations = 3
        post_register_iterations = 5
        dtype = torch.float

        for count in self.counts:
            # Keep tensors alive to simulate model weights/activations
            model_tensors = []

            # Allocate some "model" tensors (small + large to test both pools)
            small_model_tensor = torch.ones(
                1024, dtype=dtype, device=self.device
            )  # ~1KB (small pool)
            large_model_tensor = torch.ones(
                2 * 1024 * 1024, dtype=dtype, device=self.device
            )  # 2MB (large pool)
            model_tensors.extend([small_model_tensor, large_model_tensor])

            # Phase 1: Warmup iterations (default NCCL path, before registerAll)
            for _iteration in range(warmup_iterations):
                input_tensor = self._create_input_tensor(count, dtype)
                output_tensor = self._create_output_tensor(count, dtype)

                work = self.torchcomm.all_to_all_single(
                    output_tensor, input_tensor, False
                )
                work.wait()

                self._verify_results(output_tensor)

            # Phase 2: Call registerAll to register all cached memory segments
            if self.ncclx_backend is not None:
                self.ncclx_backend.register_all()

            # Phase 3: Post-registration iterations (ctran path after registerAll)
            # Reuse the same tensors - best case, all memory is already registered
            for _iteration in range(post_register_iterations):
                input_tensor = self._create_input_tensor(count, dtype)
                output_tensor = self._create_output_tensor(count, dtype)

                work = self.torchcomm.all_to_all_single(
                    output_tensor, input_tensor, False
                )
                work.wait()

                self._verify_results(output_tensor)

            # Cleanup
            del model_tensors

    def test_model_training_simulation_good_path(self) -> None:
        """Test good-case model training workflow with registerAll.

        Good path scenario:
        1. Allocate memory (tensors for model weights/activations)
        2. Run all_to_all for warmup iterations using default NCCL path
        3. Call registerAll to register all cached memory segments
        4. Allocate MORE memory (simulating additional tensors needed later)
        5. Continue running all_to_all with NEW tensors

        This is a good case because while some memory is registered, new
        allocations after registerAll must fall back to existing registration
        methods (dynamic registration). This tests the graceful fallback path.
        """
        warmup_iterations = 3
        post_register_iterations = 5
        dtype = torch.float

        for count in self.counts:
            # Keep tensors alive to simulate model weights/activations
            model_tensors = []

            # Allocate some "model" tensors (small + large to test both pools)
            small_model_tensor = torch.ones(
                1024, dtype=dtype, device=self.device
            )  # ~1KB (small pool)
            large_model_tensor = torch.ones(
                2 * 1024 * 1024, dtype=dtype, device=self.device
            )  # 2MB (large pool)
            model_tensors.extend([small_model_tensor, large_model_tensor])

            # Phase 1: Warmup iterations (default NCCL path, before registerAll)
            for _iteration in range(warmup_iterations):
                input_tensor = self._create_input_tensor(count, dtype)
                output_tensor = self._create_output_tensor(count, dtype)

                work = self.torchcomm.all_to_all_single(
                    output_tensor, input_tensor, False
                )
                work.wait()

                self._verify_results(output_tensor)

            # Phase 2: Call registerAll to register all cached memory segments
            if self.ncclx_backend is not None:
                self.ncclx_backend.register_all()

            # Phase 3: Allocate MORE memory after registerAll
            # This simulates scenarios where additional tensors are needed later
            # These new allocations are NOT covered by the previous registerAll
            additional_tensors = []
            for _i in range(3):
                # Allocate new tensors that weren't part of the initial registration
                new_tensor = torch.ones(
                    1024 * 1024, dtype=dtype, device=self.device
                )  # 1MB each
                additional_tensors.append(new_tensor)

            # Phase 4: Post-registration iterations with NEW input/output tensors
            # These will use memory that was NOT registered by registerAll,
            # so they fall back to existing registration methods
            for _iteration in range(post_register_iterations):
                # Create new tensors - these may use memory from additional allocations
                # or new memory that wasn't registered
                input_tensor = self._create_input_tensor(count, dtype)
                output_tensor = self._create_output_tensor(count, dtype)

                work = self.torchcomm.all_to_all_single(
                    output_tensor, input_tensor, False
                )
                work.wait()

                self._verify_results(output_tensor)

            # Cleanup
            del additional_tensors
            del model_tensors

    def test_model_training_simulation_worst_path(self) -> None:
        """Test worst-case model training workflow with registerAll.

        Worst path scenario:
        1. Allocate memory (tensors for model weights/activations)
        2. Run all_to_all for warmup iterations using default NCCL path
        3. Call registerAll to register all cached memory segments
        4. Call empty_cache() - this invalidates ALL registrations
        5. Continue running all_to_all (must re-register everything)

        This is the worst case because empty_cache() releases all cached memory
        and deregisters all registered segments. Post-registration iterations
        must dynamically register all buffers again from scratch.
        """
        warmup_iterations = 3
        post_register_iterations = 5
        dtype = torch.float

        for count in self.counts:
            # Phase 1: Warmup iterations (default NCCL path, before registerAll)
            # Note: We allocate tensors inside the loop so they go out of scope
            # and can be freed by empty_cache
            for _iteration in range(warmup_iterations):
                input_tensor = self._create_input_tensor(count, dtype)
                output_tensor = self._create_output_tensor(count, dtype)

                work = self.torchcomm.all_to_all_single(
                    output_tensor, input_tensor, False
                )
                work.wait()

                self._verify_results(output_tensor)

            # Phase 2: Call registerAll to register all cached memory segments
            if self.ncclx_backend is not None:
                self.ncclx_backend.register_all()

            # Phase 3: Call empty_cache - this is the WORST case
            # empty_cache releases all cached GPU memory and triggers deregistration
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            # Phase 4: Post-registration iterations
            # Since empty_cache was called, all registrations are gone
            # Every buffer must be dynamically registered again
            for _iteration in range(post_register_iterations):
                input_tensor = self._create_input_tensor(count, dtype)
                output_tensor = self._create_output_tensor(count, dtype)

                work = self.torchcomm.all_to_all_single(
                    output_tensor, input_tensor, False
                )
                work.wait()

                self._verify_results(output_tensor)

    def test_deregister_all_then_register_all(self) -> None:
        """Test deregisterAll followed by registerAll pattern.

        This test verifies the workflow where:
        1. Memory is allocated and cached
        2. registerAll registers all cached memory segments
        3. deregisterAll removes all registrations (but keeps segments cached)
        4. registerAll re-registers all cached segments

        This pattern is useful for scenarios where registration state needs to be
        reset without losing the cached segment information, such as during
        checkpoint/restore or backend reconfiguration.
        """
        num_iterations_before_dereg = 3
        num_iterations_after_rereg = 3
        dtype = torch.float

        for count in self.counts:
            # Allocate tensors to be registered
            input_tensor = self._create_input_tensor(count, dtype)
            output_tensor = self._create_output_tensor(count, dtype)

            # Phase 1: Initial registerAll and run collectives
            if self.ncclx_backend is not None:
                self.ncclx_backend.register_all()

            for _iteration in range(num_iterations_before_dereg):
                output_tensor.zero_()
                work = self.torchcomm.all_to_all_single(
                    output_tensor, input_tensor, False
                )
                work.wait()
                self._verify_results(output_tensor)

            # Phase 2: Deregister all registrations
            if self.ncclx_backend is not None:
                self.ncclx_backend.deregister_all()

            # Phase 3: Re-register all cached segments
            if self.ncclx_backend is not None:
                self.ncclx_backend.register_all()

            # Phase 4: Run collectives after re-registration
            for _iteration in range(num_iterations_after_rereg):
                output_tensor.zero_()
                work = self.torchcomm.all_to_all_single(
                    output_tensor, input_tensor, False
                )
                work.wait()
                self._verify_results(output_tensor)

    def test_multiple_deregister_register_cycles(self) -> None:
        """Test multiple cycles of deregisterAll/registerAll.

        This test verifies that the deregisterAll/registerAll cycle can be
        repeated multiple times without issues. This simulates scenarios like
        periodic backend reconfiguration or repeated checkpoint/restore cycles.
        """
        num_cycles = 3
        iterations_per_cycle = 2
        dtype = torch.float

        for count in self.counts:
            # Allocate tensors
            input_tensor = self._create_input_tensor(count, dtype)
            output_tensor = self._create_output_tensor(count, dtype)

            for cycle in range(num_cycles):
                # Register
                if self.ncclx_backend is not None:
                    self.ncclx_backend.register_all()

                # Run collectives
                for _iteration in range(iterations_per_cycle):
                    output_tensor.zero_()
                    work = self.torchcomm.all_to_all_single(
                        output_tensor, input_tensor, False
                    )
                    work.wait()
                    self._verify_results(output_tensor)

                # Deregister (except on last cycle)
                if cycle < num_cycles - 1:
                    if self.ncclx_backend is not None:
                        self.ncclx_backend.deregister_all()


if __name__ == "__main__":
    unittest.main()
