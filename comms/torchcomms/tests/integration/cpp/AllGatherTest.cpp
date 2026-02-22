// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "AllGatherTest.hpp"

#include <gtest/gtest.h>
#include <memory>
#include "TorchCommTestHelpers.h"

// Test function for synchronous all_gather with work object
template <typename Fixture>
void AllGatherTest<Fixture>::testSync(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  at::Tensor input = createInputTensor(count, dtype);
  std::vector<at::Tensor> outputs = createOutputTensors(count, dtype);
  std::vector<at::Tensor> original_outputs;
  original_outputs.reserve(num_ranks_);
  for (const auto& output : outputs) {
    original_outputs.push_back(output.clone());
  }

  auto execute = [&]() {
    auto work = torchcomm_->all_gather(outputs, input, false);
    work->wait();
  };
  auto reset = [&]() {
    for (int i = 0; i < num_ranks_; i++) {
      outputs[i].copy_(original_outputs[i]);
    }
  };
  auto verify = [&]() { verifyResults(outputs); };
  run(execute, reset, verify);
}

// Test function for synchronous all_gather without work object
template <typename Fixture>
void AllGatherTest<Fixture>::testSyncNoWork(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  at::Tensor input = createInputTensor(count, dtype);
  std::vector<at::Tensor> outputs = createOutputTensors(count, dtype);
  std::vector<at::Tensor> original_outputs;
  original_outputs.reserve(num_ranks_);
  for (const auto& output : outputs) {
    original_outputs.push_back(output.clone());
  }

  auto execute = [&]() { torchcomm_->all_gather(outputs, input, false); };
  auto reset = [&]() {
    for (int i = 0; i < num_ranks_; i++) {
      outputs[i].copy_(original_outputs[i]);
    }
  };
  auto verify = [&]() { verifyResults(outputs); };
  run(execute, reset, verify);
}

// Test function for asynchronous all_gather with wait
template <typename Fixture>
void AllGatherTest<Fixture>::testAsync(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  at::Tensor input = createInputTensor(count, dtype);
  std::vector<at::Tensor> outputs = createOutputTensors(count, dtype);
  std::vector<at::Tensor> original_outputs;
  original_outputs.reserve(num_ranks_);
  for (const auto& output : outputs) {
    original_outputs.push_back(output.clone());
  }

  auto execute = [&]() {
    auto work = torchcomm_->all_gather(outputs, input, true);
    work->wait();
  };
  auto reset = [&]() {
    for (int i = 0; i < num_ranks_; i++) {
      outputs[i].copy_(original_outputs[i]);
    }
  };
  auto verify = [&]() { verifyResults(outputs); };
  run(execute, reset, verify);
}

// Test function for asynchronous all_gather with early reset
template <typename Fixture>
void AllGatherTest<Fixture>::testAsyncEarlyReset(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  at::Tensor input = createInputTensor(count, dtype);
  std::vector<at::Tensor> outputs = createOutputTensors(count, dtype);
  std::vector<at::Tensor> original_outputs;
  original_outputs.reserve(num_ranks_);
  for (const auto& output : outputs) {
    original_outputs.push_back(output.clone());
  }

  auto execute = [&]() {
    auto work = torchcomm_->all_gather(outputs, input, true);
    work->wait();
    work.reset();
  };
  auto reset = [&]() {
    for (int i = 0; i < num_ranks_; i++) {
      outputs[i].copy_(original_outputs[i]);
    }
  };
  auto verify = [&]() { verifyResults(outputs); };
  run(execute, reset, verify);
}

// Test function for asynchronous all_gather with input deleted after enqueue
template <typename Fixture>
void AllGatherTest<Fixture>::testInputDeleted(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  auto input = std::make_shared<at::Tensor>(createInputTensor(count, dtype));
  std::vector<at::Tensor> outputs = createOutputTensors(count, dtype);
  std::vector<at::Tensor> original_outputs;
  original_outputs.reserve(num_ranks_);
  for (const auto& output : outputs) {
    original_outputs.push_back(output.clone());
  }

  auto execute = [&]() { torchcomm_->all_gather(outputs, *input, false); };
  auto reset = [&]() {
    for (int i = 0; i < num_ranks_; i++) {
      outputs[i].copy_(original_outputs[i]);
    }
  };
  auto verify = [&]() { verifyResults(outputs); };
  auto cleanup = [&]() { input.reset(); };
  run(execute, reset, verify, cleanup);
}

// Helper function to create input tensor
template <typename Fixture>
at::Tensor AllGatherTest<Fixture>::createInputTensor(
    int count,
    at::ScalarType dtype) {
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
template <typename Fixture>
std::vector<at::Tensor> AllGatherTest<Fixture>::createOutputTensors(
    int count,
    at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  std::vector<at::Tensor> outputs(num_ranks_);
  for (int i = 0; i < num_ranks_; i++) {
    outputs[i] = at::zeros({count}, options);
  }
  return outputs;
}

// Helper function to verify results
template <typename Fixture>
void AllGatherTest<Fixture>::verifyResults(
    const std::vector<at::Tensor>& outputs) {
  for (int i = 0; i < num_ranks_; i++) {
    std::string description = "rank " + std::to_string(i) + " tensor";
    verifyTensorEquality(outputs[i].cpu(), i + 1, description);
  }
}

template class AllGatherTest<EagerTestFixture<AllGatherParams>>;
template class AllGatherTest<GraphTestFixture<AllGatherParams, 1>>;
template class AllGatherTest<GraphTestFixture<AllGatherParams, 2>>;
