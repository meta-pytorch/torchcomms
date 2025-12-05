// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>
#include "comms/torchcomms/ncclx/tests/integration/cpp/AllToAllvDynamicTest.hpp"
#include "folly/logging/xlog.h"

// Logging helper
#define LOG_TEST(msg) XLOG(INFO) << "[Rank " << rank_ << "] " << msg

// Test for alltoallv_dynamic_dispatch and alltoallv_dynamic_combine
//
// This test demonstrates a complete dispatch-combine round-trip:
//
// DISPATCH PHASE:
// - Setup (for 8 ranks):
//   * Each rank has 8 chunks of 1024 elements, all filled with the rank's value
//     e.g., rank 0: [0,0,0,...], rank 1: [1,1,1,...], etc.
//
// - Communication Pattern:
//   * Each rank sends chunk i to rank i (chunk 0→rank 0, chunk 1→rank 1, etc.)
//
// - Dispatch Output:
//   * Rank i receives chunk i from all ranks in separate output buffers
//   * output_tensor_list[j] contains sender j's data at offset i×1024
//
// COMBINE PHASE:
// - Preparation:
//   * Flatten 8 output tensors (8k each) into single 64k input tensor
//   * Use chunk indices [rank, rank+8, rank+16, ..., rank+56] to select
//     one chunk from each sender
//
// - Communication Pattern:
//   * Each rank extracts and sends relevant chunks back through combine
//
// - Combine Output:
//   * Should perfectly reconstruct the original 8k input used in dispatch
//   * All values should equal the rank's ID
//
// VERIFICATION:
// - Asserts that combine output matches original dispatch input
// element-by-element
TEST_F(AllToAllvDynamicTest, DispatchCombineExample) {
  // Configurable chunk size for easy testing
  // CTRAN requires minimum 1024 elements
  const int64_t chunk_size = 1024;

  // For 8 ranks, each rank has 8 chunks of chunk_size elements each
  std::vector<int64_t> input_chunk_sizes(num_ranks_, chunk_size);

  // Send all chunks: input_chunk_indices = [0, 1, 2, 3, 4, 5, 6, 7]
  std::vector<int64_t> input_chunk_indices(num_ranks_);
  for (int i = 0; i < num_ranks_; i++) {
    input_chunk_indices[i] = i;
  }

  // Send one chunk to each rank: input_chunk_count_per_rank = [1, 1, 1, 1, 1,
  // 1, 1, 1] This means: chunk 0 to rank 0, chunk 1 to rank 1, ..., chunk 7 to
  // rank 7
  std::vector<int64_t> input_chunk_count_per_rank(num_ranks_, 1);

  int64_t maxSendcount = chunk_size * num_ranks_; // Total send buffer size

  // Verify setup dimensions
  EXPECT_EQ(input_chunk_sizes.size(), num_ranks_);
  EXPECT_EQ(input_chunk_indices.size(), num_ranks_);
  EXPECT_EQ(input_chunk_count_per_rank.size(), num_ranks_);

  // Create input tensor with rank-specific pattern using bfloat16
  auto options = at::TensorOptions().dtype(at::kBFloat16).device(device_type_);
  at::Tensor input_tensor = at::zeros({maxSendcount}, options);

  // Fill each chunk with the rank's value
  // Rank 0: all chunks filled with 0
  // Rank 1: all chunks filled with 1, etc.
  int64_t offset = 0;
  for (size_t chunk_id = 0; chunk_id < input_chunk_sizes.size(); chunk_id++) {
    int64_t length = input_chunk_sizes[chunk_id];
    if (length > 0 && offset + length <= maxSendcount) {
      at::Tensor section = input_tensor.slice(0, offset, offset + length);
      float value = static_cast<float>(rank_); // All chunks have rank's value
      section.fill_(value);
    }
    offset += length;
  }

  // Verify input tensor is set up correctly
  EXPECT_EQ(input_tensor.size(0), maxSendcount);

  // Verify that all values in input_tensor equal the rank's value
  float expected_value = static_cast<float>(rank_);
  at::Tensor expected_tensor = at::full_like(input_tensor, expected_value);
  bool input_values_correct = at::equal(input_tensor, expected_tensor);
  EXPECT_TRUE(input_values_correct)
      << "Rank " << rank_
      << ": Input tensor should contain all values equal to " << expected_value;

  // Create GPU tensors for count arrays directly
  // Use int64 (torch.int64) for all metadata tensors
  auto int64_options =
      at::TensorOptions().dtype(at::kLong).device(device_type_);

  // input_chunk_sizes: all values are chunk_size
  at::Tensor input_chunk_sizes_gpu =
      at::full({num_ranks_}, chunk_size, int64_options);

  // input_chunk_indices: [0, 1, 2, 3, 4, 5, 6, 7]
  at::Tensor input_chunk_indices_gpu = at::arange(num_ranks_, int64_options);

  // input_chunk_count_per_rank: all values are 1
  at::Tensor input_chunk_count_per_rank_gpu =
      at::ones({num_ranks_}, int64_options);

  // Create output tensors (one per rank)
  std::vector<at::Tensor> output_tensor_list;
  output_tensor_list.reserve(num_ranks_);
  for (int i = 0; i < num_ranks_; i++) {
    output_tensor_list.push_back(at::zeros({chunk_size * num_ranks_}, options));
  }

  // Prepare output_chunk_sizes_per_rank output tensor
  at::Tensor output_chunk_sizes_per_rank = at::zeros(
      {num_ranks_ * static_cast<int64_t>(input_chunk_sizes.size())},
      int64_options);

  // Call alltoallv_dynamic_dispatch
  LOG_TEST("Calling alltoallv_dynamic_dispatch...");
  auto work = ncclx_comm_->alltoallv_dynamic_dispatch(
      output_tensor_list,
      output_chunk_sizes_per_rank,
      input_tensor,
      input_chunk_sizes_gpu,
      input_chunk_indices_gpu,
      input_chunk_count_per_rank_gpu,
      false);
  work->wait();
  LOG_TEST("Dispatch completed successfully");

  // Verify output_tensor_list contains correct data using GPU operations
  // For rank i, output_tensor_list[j] should contain data from sender j
  // at offset i*chunk_size with value j
  for (int sender_rank = 0; sender_rank < num_ranks_; sender_rank++) {
    // Calculate offset where data should be placed
    int64_t expected_offset = rank_ * chunk_size;

    // Extract the chunk at expected_offset using GPU slicing
    at::Tensor received_chunk = output_tensor_list[sender_rank].slice(
        0, expected_offset, expected_offset + chunk_size);

    // Create expected tensor with sender_rank's value
    at::Tensor expected_chunk =
        at::full_like(received_chunk, static_cast<float>(sender_rank));

    // Compare using GPU tensor equality
    bool chunk_correct = at::equal(received_chunk, expected_chunk);
    EXPECT_TRUE(chunk_correct)
        << "Rank " << rank_ << " should receive value " << sender_rank
        << " from sender " << sender_rank << " at offset " << expected_offset;
  }

  LOG_TEST("Dispatch output verified successfully");

  // ============================================================================
  // VERIFY output_chunk_sizes_per_rank IS AN ALL-GATHER OF input_chunk_sizes
  // ============================================================================
  // Directly call all_gather_single on input_chunk_sizes_gpu to verify that
  // output_chunk_sizes_per_rank matches the all-gathered result

  // Create single output tensor to hold all_gather_single result
  at::Tensor allgather_input_chunk_sizes_gpu =
      at::zeros({num_ranks_ * num_ranks_}, int64_options);

  LOG_TEST(
      "Calling all_gather_single to verify output_chunk_sizes_per_rank...");
  auto allgather_work = ncclx_comm_->all_gather_single(
      allgather_input_chunk_sizes_gpu, input_chunk_sizes_gpu, false);
  allgather_work->wait();

  // Compare with output_chunk_sizes_per_rank
  bool allgather_matches =
      at::equal(allgather_input_chunk_sizes_gpu, output_chunk_sizes_per_rank);
  EXPECT_TRUE(allgather_matches)
      << "Rank " << rank_
      << ": output_chunk_sizes_per_rank should match all_gather_single(input_chunk_sizes)";

  LOG_TEST(
      "Verified: output_chunk_sizes_per_rank matches all_gather_single result!");

  // ============================================================================
  // COMBINE API PREPARATION
  // ============================================================================
  // Now flatten output_tensor_list into a single 64k (8k * 8) tensor for
  // combine API

  // Flatten the 8 tensors (each 8k) into one 64k tensor
  at::Tensor combine_input_tensor = at::cat(output_tensor_list, 0);
  EXPECT_EQ(combine_input_tensor.size(0), chunk_size * num_ranks_ * num_ranks_);

  // Setup combine API parameters
  const int64_t num_chunks = num_ranks_ * num_ranks_; // 64 chunks total

  // ============================================================================
  // COMBINE API CALL
  // ============================================================================

  // Create GPU tensors for combine API parameters
  at::Tensor combine_input_chunk_sizes_gpu =
      at::full({num_chunks}, chunk_size, int64_options);

  // Create input_chunk_indices directly on GPU: [rank, rank+8, rank+16, ...,
  // rank+56] This is equivalent to rank + [0, 8, 16, 24, 32, 40, 48, 56]
  at::Tensor combine_input_chunk_indices_gpu =
      at::arange(0, num_ranks_, int64_options) * num_ranks_ + rank_;

  at::Tensor combine_input_chunk_count_per_rank_gpu =
      at::ones({num_ranks_}, int64_options);

  // Create output tensor: 8k elements (1024 * 8)
  const int64_t combine_output_size = chunk_size * num_ranks_; // 8192
  at::Tensor combine_output_tensor = at::zeros({combine_output_size}, options);

  // Call alltoallv_dynamic_combine
  LOG_TEST("Calling alltoallv_dynamic_combine...");
  auto combine_work = ncclx_comm_->alltoallv_dynamic_combine(
      combine_output_tensor,
      combine_input_tensor,
      combine_input_chunk_sizes_gpu,
      combine_input_chunk_indices_gpu,
      combine_input_chunk_count_per_rank_gpu,
      false);
  combine_work->wait();
  LOG_TEST("Combine completed successfully");

  // ============================================================================
  // VERIFY COMBINE OUTPUT MATCHES ORIGINAL DISPATCH INPUT
  // ============================================================================

  // The combine output should match the original input_tensor used for dispatch
  // We already verified input_tensor contains all rank's values at the
  // beginning
  EXPECT_EQ(combine_output_tensor.size(0), input_tensor.size(0))
      << "Combine output size should match original dispatch input size";

  // Use at::equal() to compare GPU tensors directly
  bool tensors_equal = at::equal(combine_output_tensor, input_tensor);
  EXPECT_TRUE(tensors_equal)
      << "Rank " << rank_
      << ": Combine output should exactly match original dispatch input";

  LOG_TEST(
      "Combine output verified successfully - matches original dispatch input!");
}

// Negative test: Ensure exception is thrown when metadata tensors have wrong
// type
TEST_F(AllToAllvDynamicTest, InvalidMetadataTypeThrowsException) {
  const int64_t chunk_size = 1024;

  // Setup basic parameters
  std::vector<int64_t> input_chunk_sizes(num_ranks_, chunk_size);
  std::vector<int64_t> input_chunk_indices(num_ranks_);
  for (int i = 0; i < num_ranks_; i++) {
    input_chunk_indices[i] = i;
  }
  std::vector<int64_t> input_chunk_count_per_rank(num_ranks_, 1);
  int64_t maxSendcount = chunk_size * num_ranks_;

  // Create input tensor using bfloat16
  auto options = at::TensorOptions().dtype(at::kBFloat16).device(device_type_);
  at::Tensor input_tensor = at::zeros({maxSendcount}, options);

  // Create correct int64 tensors
  auto int64_options =
      at::TensorOptions().dtype(at::kLong).device(device_type_);

  at::Tensor input_chunk_sizes_gpu =
      at::full({num_ranks_}, chunk_size, int64_options);
  at::Tensor input_chunk_indices_gpu = at::arange(num_ranks_, int64_options);
  at::Tensor input_chunk_count_per_rank_gpu =
      at::ones({num_ranks_}, int64_options);

  // Create output tensors
  std::vector<at::Tensor> output_tensor_list;
  output_tensor_list.reserve(num_ranks_);
  for (int i = 0; i < num_ranks_; i++) {
    output_tensor_list.push_back(at::zeros({chunk_size * num_ranks_}, options));
  }

  at::Tensor output_chunk_sizes_per_rank = at::zeros(
      {num_ranks_ * static_cast<int64_t>(input_chunk_sizes.size())},
      int64_options);

  // Test: Pass int32_t for input_chunk_sizes (should throw)
  LOG_TEST("Testing invalid type for input_chunk_sizes...");
  auto int32_options = at::TensorOptions().dtype(at::kInt).device(device_type_);
  at::Tensor invalid_input_chunk_sizes_gpu =
      at::full({num_ranks_}, static_cast<int>(chunk_size), int32_options);

  EXPECT_THROW(
      {
        auto work = ncclx_comm_->alltoallv_dynamic_dispatch(
            output_tensor_list,
            output_chunk_sizes_per_rank,
            input_tensor,
            invalid_input_chunk_sizes_gpu, // Invalid type (int32)
            input_chunk_indices_gpu,
            input_chunk_count_per_rank_gpu,
            false);
      },
      std::runtime_error);
  LOG_TEST("Correctly threw exception for invalid input_chunk_sizes type");

  LOG_TEST("Negative test passed successfully!");
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
