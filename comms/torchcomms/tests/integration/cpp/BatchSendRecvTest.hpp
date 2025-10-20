// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>
#include <vector>
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

class BatchSendRecvTest
    : public ::testing::TestWithParam<std::tuple<int, at::ScalarType>> {
 public:
  BatchSendRecvTest() : BatchSendRecvTest(c10::DeviceType::CUDA) {}
  explicit BatchSendRecvTest(c10::DeviceType device_type)
      : rank_(0), num_ranks_(0), device_type_(device_type) {}

  // Test function declarations with parameters
  void testSyncBatchSendRecv(int count, at::ScalarType dtype);
  void testSyncBatchSendRecvNoWork(int count, at::ScalarType dtype);
  void testAsyncBatchSendRecv(int count, at::ScalarType dtype);
  void testAsyncBatchSendRecvEarlyReset(int count, at::ScalarType dtype);
  void testBatchSendRecvInputDeleted(int count, at::ScalarType dtype);
  void testGraphBatchSendRecv(int count, at::ScalarType dtype);
  void testGraphBatchSendRecvInputDeleted(int count, at::ScalarType dtype);

 protected:
  virtual std::unique_ptr<TorchCommTestWrapper> createWrapper();

  virtual void SetUp() override;

  virtual void TearDown() override;

  struct BatchSendRecvTestParams {
    std::vector<at::Tensor> send_tensors;
    std::vector<at::Tensor> recv_tensors;
    std::vector<int> send_ranks;
    std::vector<int> recv_ranks;
  };

  BatchSendRecvTestParams createBatchSendRecvParams(
      int count,
      at::ScalarType dtype) {
    BatchSendRecvTestParams params;

    // Create multiple send/recv pairs
    // Each rank sends to next rank and receives from previous rank
    int next_rank = (rank_ + 1) % num_ranks_;
    int prev_rank = (rank_ + num_ranks_ - 1) % num_ranks_;

    // Create 2 send operations and 2 recv operations for testing
    for (int i = 0; i < 2; ++i) {
      // Send tensors with rank-specific values
      at::Tensor send_tensor = createSendTensor(count, dtype, i);
      params.send_tensors.push_back(send_tensor);
      params.send_ranks.push_back(next_rank);

      // Recv tensors initialized to zero
      at::Tensor recv_tensor = createRecvTensor(count, dtype);
      params.recv_tensors.push_back(recv_tensor);
      params.recv_ranks.push_back(prev_rank);
    }

    return params;
  }

  std::unique_ptr<TorchCommTestWrapper> wrapper_;
  std::shared_ptr<torch::comms::TorchComm> torchcomm_;
  int rank_;
  int num_ranks_;
  c10::DeviceType device_type_;

  static constexpr int num_replays = 4;

  // Helper function declarations with parameters
  virtual at::Tensor
  createSendTensor(int count, at::ScalarType dtype, int tensor_id);
  virtual at::Tensor createRecvTensor(int count, at::ScalarType dtype);
  void verifyResults(
      const std::vector<at::Tensor>& recv_tensors,
      int recv_rank);
};
