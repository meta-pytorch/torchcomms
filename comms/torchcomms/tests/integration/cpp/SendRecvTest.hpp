// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

class SendRecvTest
    : public ::testing::TestWithParam<std::tuple<int, at::ScalarType>> {
 public:
  SendRecvTest() : SendRecvTest(c10::DeviceType::CUDA) {}
  explicit SendRecvTest(c10::DeviceType device_type)
      : rank_(0), num_ranks_(0), device_type_(device_type) {}

  // Test function declarations with parameters
  void testSyncSendRecv(int count, at::ScalarType dtype);
  void testSyncSendRecvNoWork(int count, at::ScalarType dtype);
  void testAsyncSendRecv(int count, at::ScalarType dtype);
  void testAsyncSendRecvEarlyReset(int count, at::ScalarType dtype);
  void testSendRecvInputDeleted(int count, at::ScalarType dtype);
  void testGraphSendRecv(int count, at::ScalarType dtype);
  void testGraphSendRecvInputDeleted(int count, at::ScalarType dtype);

 protected:
  virtual std::unique_ptr<TorchCommTestWrapper> createWrapper();

  virtual void SetUp() override;

  virtual void TearDown() override;

  struct SendRecvTestParams {
    int send_rank;
    int recv_rank;
    at::Tensor send_tensor;
    at::Tensor recv_tensor;
  };

  SendRecvTestParams createSendRecvParams(int count, at::ScalarType dtype) {
    // Each rank sends to the next rank and receives from the previous rank
    int send_rank = (rank_ + 1) % num_ranks_;
    int recv_rank = (rank_ + num_ranks_ - 1) % num_ranks_;

    // Create input tensor with rank-specific values
    at::Tensor send_tensor = createSendTensor(count, dtype);

    // Create output tensor to receive data
    at::Tensor recv_tensor = createRecvTensor(count, dtype);
    return SendRecvTestParams{
        send_rank, recv_rank, std::move(send_tensor), std::move(recv_tensor)};
  }

  std::unique_ptr<TorchCommTestWrapper> wrapper_;
  std::shared_ptr<torch::comms::TorchComm> torchcomm_;
  int rank_;
  int num_ranks_;
  c10::DeviceType device_type_;

  static constexpr int num_replays = 4;

  // Helper function declarations with parameters
  virtual at::Tensor createSendTensor(int count, at::ScalarType dtype);
  virtual at::Tensor createRecvTensor(int count, at::ScalarType dtype);
  void verifyResults(const at::Tensor& recv_tensor, int recv_rank);
};
