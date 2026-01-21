// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "AllGatherVTest.hpp"

#include <gtest/gtest.h>
#include <vector>
#include "TorchCommTestHelpers.h"

std::unique_ptr<TorchCommTestWrapper> AllGatherVTest::createWrapper() {
  return std::make_unique<TorchCommTestWrapper>();
}

void AllGatherVTest::SetUp() {
  wrapper_ = createWrapper();
  torchcomm_ = wrapper_->getTorchComm();
  rank_ = torchcomm_->getRank();
  num_ranks_ = torchcomm_->getSize();
}

void AllGatherVTest::TearDown() {
  // Explicitly reset the TorchComm object to ensure proper cleanup
  torchcomm_.reset();
  wrapper_.reset();
}

// Test function for synchronous all_gather_v with work object
void AllGatherVTest::testSyncAllGatherV(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing sync all_gather_v with count=" << count
                           << " and dtype=" << getDtypeName(dtype));

  // Create input and output tensors
  auto counts = std::vector<int>(num_ranks_, count);
  for (int i = 0; i < num_ranks_; i++) {
    counts[i] = count + i;
  }
  at::Tensor input = createInputTensor(counts[rank_], dtype);
  std::vector<at::Tensor> outputs =
      createOutputTensors(std::move(counts), dtype);

  // Call all_gather_v
  auto work = torchcomm_->all_gather_v(outputs, input, false);
  work->wait();

  // Verify the results
  verifyResults(outputs);
}

// Helper function to create input tensor
at::Tensor AllGatherVTest::createInputTensor(int count, at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  at::Tensor input;
  if (dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16) {
    input = at::ones({count}, options) * static_cast<float>(rank_ + 1);
  } else if (dtype == at::kInt) {
    input = at::ones({count}, options) * static_cast<int>(rank_ + 1);
  } else if (dtype == at::kChar) {
    input = at::ones({count}, options) * static_cast<signed char>(rank_ + 1);
  }
  return input;
}

// Helper function to create output tensors
std::vector<at::Tensor> AllGatherVTest::createOutputTensors(
    std::vector<int> counts,
    at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  std::vector<at::Tensor> outputs(num_ranks_);
  for (int i = 0; i < num_ranks_; i++) {
    outputs[i] = at::zeros({counts[i]}, options);
  }
  return outputs;
}

// Helper function to verify results
void AllGatherVTest::verifyResults(const std::vector<at::Tensor>& outputs) {
  for (int i = 0; i < num_ranks_; i++) {
    // Use verifyTensorEquality to compare output with expected tensor
    std::string description = "rank " + std::to_string(i) + " tensor";
    verifyTensorEquality(outputs[i].cpu(), i + 1, description);
  }
}
