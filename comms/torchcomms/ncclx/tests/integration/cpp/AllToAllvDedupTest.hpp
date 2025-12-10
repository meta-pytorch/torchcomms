// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>
#include <vector>
#include "comms/torchcomms/ncclx/TorchCommNCCLX.hpp"
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

// Data class to hold test parameters for AllToAllv dedup dispatch tests.
struct DedupTestParams {
  int num_tokens;
  int token_numel;
  int topk;
  int num_experts;
  at::ScalarType dtype;
};

class AllToAllvDedupTest : public ::testing::Test {
 public:
  AllToAllvDedupTest() : AllToAllvDedupTest(c10::DeviceType::CUDA) {}
  explicit AllToAllvDedupTest(c10::DeviceType device_type)
      : rank_(0),
        num_ranks_(0),
        num_local_ranks_(0),
        num_nodes_(0),
        device_type_(device_type) {}

 protected:
  virtual std::unique_ptr<TorchCommTestWrapper> createWrapper();

  virtual void SetUp() override;

  virtual void TearDown() override;

  // Map a global expert ID to the owning rank.
  int expertToRank(int expert_id, int num_ranks) const;

  // Count total tokens this rank will receive from all peers based on expert
  // ownership.
  int countNumRecvTokens() const;

  // Return a compacted index map from mask where True positions are 0..N-1 and
  // False are -1.
  at::Tensor getIndiceMapFromMask(const at::Tensor& mask) const;

  // Return indices of True positions in the boolean mask.
  at::Tensor getIdListFromMask(const at::Tensor& mask) const;

  // Generate local random top-k expert IDs per token and all-gather them across
  // ranks.
  void setGlobalTopkIds(const at::Tensor& expert_range);

  // Generate send_indices for each node based on expert assignments in
  // all_ranks_topk_ids.
  at::Tensor prepareSendIndices() const;

  // Generate forward_indices for tokens received from each node.
  at::Tensor prepareForwardIndices() const;

  // Generate recv_indices for each local expert.
  at::Tensor prepareRecvIndices() const;

  // Allocate input tensor, output tensors, and recv_token_ids tensor.
  std::tuple<at::Tensor, at::Tensor, at::Tensor> prepareTensors() const;

  // Generate recv_token_ids for each local expert and each send rank.
  std::vector<std::vector<at::Tensor>> prepareExpectedRecvTokenIds() const;

  // Generate expected output tensor for each local expert, each send rank and
  // each token.
  std::vector<std::vector<std::vector<at::Tensor>>> prepareExpectedOutputTensor(
      const std::vector<std::vector<at::Tensor>>& expected_recv_token_ids)
      const;

  // Check output tensor and recv_token_ids against expected values.
  void checkExecOutput(
      const at::Tensor& output_tensor,
      const at::Tensor& recv_token_ids) const;

  // Reset test parameters and global top-k ids for each run_test call.
  void resetRunTest(
      int num_tokens,
      int token_numel,
      int num_experts,
      int topk,
      at::ScalarType dtype);

  // Run AllToAllv dedup dispatch test with given parameters.
  void runTest();

  std::unique_ptr<TorchCommTestWrapper> wrapper_;
  std::shared_ptr<torch::comms::TorchComm> torchcomm_;
  std::shared_ptr<torch::comms::TorchCommNCCLX> ncclx_comm_;
  int rank_;
  int num_ranks_;
  int num_local_ranks_;
  int num_nodes_;
  c10::DeviceType device_type_;

  // Store test parameters for each run_test call
  DedupTestParams test_params_{};
  // Store global top-k ids across ranks
  std::vector<at::Tensor> all_ranks_topk_ids_;
};
