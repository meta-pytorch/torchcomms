// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "ReduceScatterVTest.hpp"

#include <gtest/gtest.h>
#include <vector>
#include "comms/torchcomms/TorchWork.hpp"
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

std::unique_ptr<TorchCommTestWrapper> ReduceScatterVTest::createWrapper() {
  return std::make_unique<TorchCommTestWrapper>();
}

void ReduceScatterVTest::SetUp() {
  wrapper_ = createWrapper();
  torchcomm_ = wrapper_->getTorchComm();
  rank_ = torchcomm_->getRank();
  num_ranks_ = torchcomm_->getSize();
  device_type_ = wrapper_->getDevice().type();
}

void ReduceScatterVTest::TearDown() {
  // Explicitly reset the TorchComm object to ensure proper cleanup
  torchcomm_.reset();
  wrapper_.reset();
}

// Test function for synchronous reduce_scatter_v with work object
void ReduceScatterVTest::testSyncReduceScatterV(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing sync reduce_scatter_v with count="
                           << count << " and dtype=" << getDtypeName(dtype)
                           << " and op=" << getOpName(op));

  // Create input and output tensors
  auto counts = std::vector<int>(num_ranks_, count);
  for (int i = 0; i < num_ranks_; i++) {
    counts[i] = count + i;
  }
  std::vector<at::Tensor> input_tensors = createInputTensors(counts, dtype);
  at::Tensor output = createOutputTensor(counts[rank_], dtype);

  // Call reduce_scatter_v
  auto work = torchcomm_->reduce_scatter_v(output, input_tensors, op, false);
  work->wait();

  // Verify the results
  verifyResults(output, op);
}

// Helper function to create input tensors
std::vector<at::Tensor> ReduceScatterVTest::createInputTensors(
    std::vector<int> counts,
    at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  std::vector<at::Tensor> input_tensors;
  input_tensors.reserve(num_ranks_);

  for (int r = 0; r < num_ranks_; r++) {
    // Each tensor has rank-specific values
    at::Tensor tensor;
    if (dtype == at::kFloat || dtype == at::kBFloat16 || dtype == at::kHalf) {
      tensor = at::ones({counts[r]}, options) * static_cast<float>(r + 1);
    } else if (dtype == at::kInt) {
      tensor = at::ones({counts[r]}, options) * static_cast<int>(r + 1);
    } else if (dtype == at::kChar) {
      tensor = at::ones({counts[r]}, options) * static_cast<signed char>(r + 1);
    }
    input_tensors.push_back(tensor);
  }

  return input_tensors;
}

// Helper function to create output tensor
at::Tensor ReduceScatterVTest::createOutputTensor(
    int count,
    at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  return at::zeros({count}, options);
}

// Helper function to calculate expected result
int ReduceScatterVTest::calculateExpectedResult(
    const torch::comms::ReduceOp& op) {
  if (op == torch::comms::ReduceOp::SUM) {
    return num_ranks_ * (rank_ + 1);
  } else if (op == torch::comms::ReduceOp::MAX) {
    return rank_ + 1;
  } else if (op == torch::comms::ReduceOp::AVG) {
    return rank_ + 1;
  } else {
    throw std::runtime_error("Unsupported reduce operation");
  }
}

// Helper function to verify results
void ReduceScatterVTest::verifyResults(
    const at::Tensor& output,
    const torch::comms::ReduceOp& op) {
  // Calculate expected result
  int expected = calculateExpectedResult(op);

  // Use verifyTensorEquality to compare output with expected tensor
  std::string description = "reduce_scatter_v with op " + getOpName(op);
  verifyTensorEquality(output.cpu(), expected, description);
}
