// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>
#include "comms/torchcomms/ncclx/tests/integration/cpp/AllToAllvDedupTest.hpp"
#include "folly/logging/xlog.h"

// Logging helper
#define LOG_TEST(msg) XLOG(INFO) << "[Rank " << rank_ << "] " << msg

// Test for alltoallv_dedup_init and alltoallv_dedup_exec
//
// This test demonstrates the AllToAllv dedup dispatch operation for MoE:
//
// DISPATCH PHASE:
// - Setup:
//   * Each rank generates random top-k expert IDs per token
//   * Tokens are dispatched to experts based on expert-to-rank mapping
//
// - Communication Pattern:
//   * Each rank sends tokens to the ranks owning the corresponding experts
//   * Deduplication ensures unique tokens are sent to each node
//
// - Output:
//   * Each rank receives tokens from all ranks that have experts assigned
//   * recv_token_ids contains the original token IDs for verification
//
// VERIFICATION:
// - Asserts that received tokens match expected values based on sender and
// token ID
// - Asserts that recv_token_ids matches the expected token IDs
TEST_F(AllToAllvDedupTest, DispatchExample) {
  // Test parameters matching Python test
  const int num_tokens = 4;
  const int token_numel = 16;
  const int topk = 2;
  const int num_experts = 16;
  const at::ScalarType dtype = at::kInt;

  LOG_TEST("Starting AllToAllv dedup dispatch test");
  LOG_TEST(
      "Parameters: num_tokens=" << num_tokens << ", token_numel=" << token_numel
                                << ", topk=" << topk
                                << ", num_experts=" << num_experts);

  // Reset test parameters
  resetRunTest(num_tokens, token_numel, num_experts, topk, dtype);

  // Run the test
  runTest();

  LOG_TEST("AllToAllv dedup dispatch test completed successfully");
}

// Test with different parameter combinations
TEST_F(AllToAllvDedupTest, DispatchMultipleConfigs) {
  // Define test parameter combinations
  std::vector<int> num_tokens_list = {4, 64};
  std::vector<int> token_numel_list = {16};
  std::vector<int> topk_list = {2, 4};
  std::vector<int> num_experts_list = {8, 16, 64};
  std::vector<at::ScalarType> dtype_list = {at::kInt};

  for (int num_tokens : num_tokens_list) {
    for (int token_numel : token_numel_list) {
      for (int num_experts : num_experts_list) {
        for (int topk : topk_list) {
          for (at::ScalarType dtype : dtype_list) {
            std::string test_name = "nTokens_" + std::to_string(num_tokens) +
                "_nExperts_" + std::to_string(num_experts) + "_topK_" +
                std::to_string(topk) + "_" + getDtypeName(dtype);
            LOG_TEST("Running dispatch test with parameters: " << test_name);

            resetRunTest(num_tokens, token_numel, num_experts, topk, dtype);
            runTest();

            LOG_TEST("Test passed: " << test_name);
          }
        }
      }
    }
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
