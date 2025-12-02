#!/usr/bin/env python3
# pyre-unsafe
# Copyright (c) Meta Platforms, Inc. and affiliates.

import itertools
import os
import unittest

import torch
from torchcomms.tests.integration.py.TorchCommTestHelpers import (
    get_dtype_name,
    TorchCommTestWrapper,
)


class AllToAllvDynamicDispatchTest(unittest.TestCase):
    """Test class for alltoallv_dynamic_dispatch and alltoallv_dynamic_combine operations.

    This test mimics the DispatchCombineExample from AllToAllvDynamicTestMain.cpp:

    DISPATCH PHASE:
    - Each rank has num_ranks chunks of chunk_size elements
    - All chunks are filled with the rank's value (rank 0: all 0s, rank 1: all 1s, etc.)
    - Communication pattern: Each rank sends chunk i to rank i
    - Expected: Rank i receives chunk i from all ranks in separate output buffers

    COMBINE PHASE:
    - Flatten output_tensor_list into single tensor
    - Extract chunks using indices [rank, rank+num_ranks, rank+2*num_ranks, ...]
    - Communication: Send selected chunks back through combine
    - Expected: Reconstruct original dispatch input
    """

    # Class variables for test parameters
    # CTRAN requires minimum 1024 elements per chunk
    chunk_sizes = [1024]
    dtypes = [torch.int]
    # hidden_dim scales chunk sizes for C++ API (C++ sees chunk_sizes * hidden_dim)
    hidden_dim = 2

    def get_wrapper(self):
        return TorchCommTestWrapper()

    def setUp(self):
        """Set up test environment before each test."""
        # NCCLX alltoallvDynamic requires NCCL_CTRAN_ENABLE=1
        os.environ["NCCL_CTRAN_ENABLE"] = "1"
        self.wrapper = self.get_wrapper()
        self.torchcomm = self.wrapper.get_torchcomm()
        self.rank = self.torchcomm.get_rank()
        self.num_ranks = self.torchcomm.get_size()
        self.device = self.torchcomm.get_device()

        # Get the NCCLX backend for NCCLX-specific APIs
        self.ncclx_backend = self.torchcomm.unsafe_get_backend()

    def tearDown(self):
        """Clean up after each test."""
        # Explicitly reset the TorchComm object to ensure proper cleanup
        self.torchcomm = None
        self.wrapper = None

    def _sync_alltoallv_dynamic(self, chunk_size, dtype):
        """Test synchronous alltoallv_dynamic_dispatch with work object."""
        print(
            f"Testing sync alltoallv_dynamic_dispatch with chunk_size={chunk_size} and dtype={get_dtype_name(dtype)}"
        )

        # Create input and output tensors
        (
            input_tensor,
            input_chunk_sizes,
            input_chunk_indices,
            input_chunk_count_per_rank,
            output_tensor_list,
            output_chunk_sizes_per_rank,
        ) = self._create_tensors(chunk_size, dtype)

        # Save original input for later verification
        original_input_tensor = input_tensor.clone()

        # Actual chunk size (accounts for hidden dimension)
        actual_chunk_size = chunk_size * self.hidden_dim

        # Print input information
        print(
            f"[Rank {self.rank}] INPUT: tensor filled with value {self.rank}, shape={input_tensor.shape}"
        )
        print(
            f"[Rank {self.rank}] INPUT: chunk_sizes={input_chunk_sizes.cpu().tolist()}"
        )
        print(
            f"[Rank {self.rank}] INPUT: chunk_indices={input_chunk_indices.cpu().tolist()}"
        )
        print(
            f"[Rank {self.rank}] INPUT: chunk_count_per_rank={input_chunk_count_per_rank.cpu().tolist()}"
        )
        print(
            f"[Rank {self.rank}] INPUT: first 10 values={input_tensor[:10].cpu().tolist()}"
        )

        # Call alltoallv_dynamic_dispatch
        print(f"[Rank {self.rank}] Calling alltoallv_dynamic_dispatch...")
        work = self.ncclx_backend.alltoallv_dynamic_dispatch(
            output_tensor_list,
            output_chunk_sizes_per_rank,
            input_tensor,
            input_chunk_sizes,
            input_chunk_indices,
            input_chunk_count_per_rank,
            self.hidden_dim,
            False,
        )
        work.wait()
        print(f"[Rank {self.rank}] Dispatch completed successfully")

        # Print output information
        print(
            f"[Rank {self.rank}] OUTPUT: received chunk_sizes_per_rank={output_chunk_sizes_per_rank.cpu().tolist()}"
        )
        for sender_rank in range(self.num_ranks):
            offset = self.rank * actual_chunk_size
            section = output_tensor_list[sender_rank][
                offset : offset + min(10, actual_chunk_size)
            ]
            print(
                f"[Rank {self.rank}] OUTPUT: from sender {sender_rank}, first 10 values at offset {offset}={section.cpu().tolist()}"
            )

        # Verify the dispatch results
        self._verify_dispatch_results(
            output_tensor_list, output_chunk_sizes_per_rank, chunk_size
        )
        print(f"[Rank {self.rank}] Dispatch output verified successfully")

        # =====================================================================
        # COMBINE API PREPARATION
        # =====================================================================
        # Flatten output_tensor_list into a single tensor for combine API
        combine_input_tensor = torch.cat(output_tensor_list, dim=0)
        self.assertEqual(
            combine_input_tensor.size(0),
            actual_chunk_size * self.num_ranks * self.num_ranks,
            "Combine input size should be num_ranks * num_ranks * actual_chunk_size",
        )

        # Setup combine API parameters
        num_chunks = self.num_ranks * self.num_ranks  # Total chunks in flattened tensor

        # All chunk sizes are chunk_size (logical size, not actual)
        # The Python API will scale this up by hidden_dim before calling C++
        combine_input_chunk_sizes = torch.full(
            (num_chunks,), chunk_size, dtype=torch.long, device=self.device
        )

        # Create input_chunk_indices: [rank, rank+num_ranks, rank+2*num_ranks, ...]
        # This selects one chunk from each sender
        combine_input_chunk_indices = (
            torch.arange(0, self.num_ranks, dtype=torch.long, device=self.device)
            * self.num_ranks
            + self.rank
        )

        # One chunk per destination rank
        combine_input_chunk_count_per_rank = torch.ones(
            self.num_ranks, dtype=torch.long, device=self.device
        )

        # Create output tensor: should reconstruct original input (actual_chunk_size * num_ranks)
        combine_output_size = actual_chunk_size * self.num_ranks
        options = {"dtype": dtype, "device": self.device}
        combine_output_tensor = torch.zeros(combine_output_size, **options)

        # =====================================================================
        # COMBINE API CALL
        # =====================================================================
        print(f"[Rank {self.rank}] Calling alltoallv_dynamic_combine...")
        combine_work = self.ncclx_backend.alltoallv_dynamic_combine(
            combine_output_tensor,
            combine_input_tensor,
            combine_input_chunk_sizes,
            combine_input_chunk_indices,
            combine_input_chunk_count_per_rank,
            self.hidden_dim,
            False,
        )
        combine_work.wait()
        print(f"[Rank {self.rank}] Combine completed successfully")

        # =====================================================================
        # VERIFY COMBINE OUTPUT MATCHES ORIGINAL DISPATCH INPUT
        # =====================================================================
        self.assertEqual(
            combine_output_tensor.size(0),
            original_input_tensor.size(0),
            "Combine output size should match original dispatch input size",
        )

        # Verify tensors are equal (using torch.equal for integer types)
        self.assertTrue(
            torch.equal(combine_output_tensor, original_input_tensor),
            f"Rank {self.rank}: Combine output should exactly match original dispatch input",
        )

        print(
            f"[Rank {self.rank}] Combine output verified successfully - matches original dispatch input!"
        )
        print(f"[Rank {self.rank}] VERIFICATION PASSED for sync test")

    def _create_tensors(self, chunk_size, dtype):
        """Create input and output tensors matching SimpleDispatchExample pattern.

        Each rank has num_ranks chunks of chunk_size*hidden_dim elements.
        All chunks are filled with the rank's value.
        Communication: Each rank sends chunk i to rank i.
        Note: chunk_size is the logical size; actual size is chunk_size * hidden_dim.
        """
        # Actual chunk size in elements (accounts for hidden dimension)
        actual_chunk_size = chunk_size * self.hidden_dim

        # Total input size = actual_chunk_size * num_ranks
        maxSendcount = actual_chunk_size * self.num_ranks
        options = {"dtype": dtype, "device": self.device}

        # Create input tensor and fill each chunk with rank's value
        # Rank 0: all chunks filled with 0, Rank 1: all chunks filled with 1, etc.
        input_tensor = torch.zeros(maxSendcount, **options)

        for chunk_id in range(self.num_ranks):
            offset = chunk_id * actual_chunk_size
            section = input_tensor[offset : offset + actual_chunk_size]
            # Fill with rank's value (all chunks have same value)
            section.fill_(int(self.rank))

        # input_chunk_sizes: all values are chunk_size (logical size, not actual)
        # The Python API will scale this up by hidden_dim before calling C++
        input_chunk_sizes = torch.full(
            (self.num_ranks,), chunk_size, dtype=torch.long, device=self.device
        )

        # input_chunk_indices: [0, 1, 2, 3, ..., num_ranks-1]
        input_chunk_indices = torch.arange(
            self.num_ranks, dtype=torch.long, device=self.device
        )

        # input_chunk_count_per_rank: all values are 1 (one chunk per destination rank)
        # This means: chunk 0 to rank 0, chunk 1 to rank 1, ..., chunk (num_ranks-1) to rank (num_ranks-1)
        input_chunk_count_per_rank = torch.ones(
            self.num_ranks, dtype=torch.long, device=self.device
        )

        # Create output tensors (one per source rank)
        # Each output buffer can receive up to num_ranks chunks of actual_chunk_size
        output_tensor_list = [
            torch.zeros(actual_chunk_size * self.num_ranks, **options)
            for _ in range(self.num_ranks)
        ]

        # Create output_chunk_sizes_per_rank to receive chunk size info
        # Size = num_ranks * num_chunks_per_sender = num_ranks * num_ranks
        output_chunk_sizes_per_rank = torch.zeros(
            self.num_ranks * self.num_ranks, dtype=torch.long, device=self.device
        )

        return (
            input_tensor,
            input_chunk_sizes,
            input_chunk_indices,
            input_chunk_count_per_rank,
            output_tensor_list,
            output_chunk_sizes_per_rank,
        )

    def _verify_dispatch_results(
        self, output_tensor_list, output_chunk_sizes_per_rank, chunk_size
    ):
        """Verify dispatch results matching DispatchCombineExample pattern.

        For rank i:
        - Should receive chunk i from all ranks
        - output_tensor_list[sender_j] should contain sender_j's value at offset i*actual_chunk_size
        Note: chunk_size is the logical size; actual size is chunk_size * hidden_dim.
        """
        # Actual chunk size in elements (accounts for hidden dimension)
        actual_chunk_size = chunk_size * self.hidden_dim

        print(f"[Rank {self.rank}] VERIFY: Starting dispatch verification")

        # Verify output_chunk_sizes_per_rank structure
        # Expected: [num_ranks x num_chunks] where num_chunks = num_ranks
        expected_size = self.num_ranks * self.num_ranks
        self.assertEqual(
            output_chunk_sizes_per_rank.numel(),
            expected_size,
            "output_chunk_sizes_per_rank size mismatch",
        )

        # Verify that each sender sent chunk[rank] to this rank with size chunk_size (logical)
        # The Python API scales down the chunk sizes, so we should see chunk_size, not actual_chunk_size
        output_sizes_cpu = output_chunk_sizes_per_rank.cpu().tolist()
        print(f"[Rank {self.rank}] VERIFY: Checking chunk sizes from all senders")
        for sender_rank in range(self.num_ranks):
            # Index calculation: sender_rank * num_chunks_per_sender + chunk_id
            # chunk_id for this rank is self.rank
            idx = sender_rank * self.num_ranks + self.rank
            actual_size = output_sizes_cpu[idx]
            print(
                f"[Rank {self.rank}] VERIFY: From sender {sender_rank}, expected size={chunk_size}, actual size={actual_size}"
            )
            self.assertEqual(
                actual_size,
                chunk_size,
                f"Sender rank {sender_rank} should send chunk {self.rank} with size {chunk_size} to rank {self.rank}",
            )

        # Verify output_tensor_list contains correct data
        # For rank i, output_tensor_list[j] should contain data from sender j
        # at offset i*actual_chunk_size with value j
        print(f"[Rank {self.rank}] VERIFY: Checking received data values")
        for sender_rank in range(self.num_ranks):
            recvbuff = output_tensor_list[sender_rank]

            # Calculate offset where data should be placed (using actual size)
            expected_offset = self.rank * actual_chunk_size

            # Verify that the chunk at expected_offset contains sender_rank's value
            section = recvbuff[expected_offset : expected_offset + actual_chunk_size]

            expected_value = sender_rank
            actual_value = section[0].item()
            print(
                f"[Rank {self.rank}] VERIFY: From sender {sender_rank}, expected value={expected_value}, actual value={actual_value}"
            )
            expected = torch.ones(
                actual_chunk_size, dtype=section.dtype, device=section.device
            ) * int(expected_value)
            self.assertTrue(
                torch.equal(section, expected),
                f"Rank {self.rank} should receive value {sender_rank} from sender {sender_rank} at offset {expected_offset}",
            )

        print(f"[Rank {self.rank}] VERIFY: All dispatch checks passed!")

    def test_dispatch_combine(self):
        """Run dispatch-combine round-trip test with all parameter combinations."""
        # Nested loops for all parameter combinations
        for chunk_size, dtype in itertools.product(self.chunk_sizes, self.dtypes):
            # Create a descriptive test name for better test output
            test_name = f"ChunkSize_{chunk_size}_{get_dtype_name(dtype)}"
            print(f"Running dispatch-combine test with parameters: {test_name}")

            # Run sync alltoallv_dynamic_dispatch with combine
            self._sync_alltoallv_dynamic(chunk_size, dtype)


if __name__ == "__main__":
    unittest.main()
