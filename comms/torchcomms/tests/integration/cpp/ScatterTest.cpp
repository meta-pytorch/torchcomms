// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "ScatterTest.hpp"

#include <gtest/gtest.h>
#include <memory>
#include "TorchCommTestHelpers.h"

// Test function for synchronous scatter with work object
template <typename Fixture>
void ScatterTest<Fixture>::testSync(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  const int root_rank = 0;

  std::vector<at::Tensor> inputs;
  if (rank_ == root_rank) {
    inputs = createInputTensors(count, dtype);
  }

  at::Tensor output = createOutputTensor(count, dtype);
  auto original_output = output.clone();

  auto execute = [&]() {
    auto work = torchcomm_->scatter(output, inputs, root_rank, false);
    work->wait();
  };
  auto reset = [&]() { output.copy_(original_output); };
  auto verify = [&]() { verifyResults(output); };
  run(execute, reset, verify);
}

// Test function for synchronous scatter without work object
template <typename Fixture>
void ScatterTest<Fixture>::testSyncNoWork(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  const int root_rank = 0;

  std::vector<at::Tensor> inputs;
  if (rank_ == root_rank) {
    inputs = createInputTensors(count, dtype);
  }

  at::Tensor output = createOutputTensor(count, dtype);
  auto original_output = output.clone();

  auto execute = [&]() {
    torchcomm_->scatter(output, inputs, root_rank, false);
  };
  auto reset = [&]() { output.copy_(original_output); };
  auto verify = [&]() { verifyResults(output); };
  run(execute, reset, verify);
}

// Test function for asynchronous scatter with wait
template <typename Fixture>
void ScatterTest<Fixture>::testAsync(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  const int root_rank = 0;

  std::vector<at::Tensor> inputs;
  if (rank_ == root_rank) {
    inputs = createInputTensors(count, dtype);
  }

  at::Tensor output = createOutputTensor(count, dtype);
  auto original_output = output.clone();

  auto execute = [&]() {
    auto work = torchcomm_->scatter(output, inputs, root_rank, true);
    work->wait();
  };
  auto reset = [&]() { output.copy_(original_output); };
  auto verify = [&]() { verifyResults(output); };
  run(execute, reset, verify);
}

// Test function for asynchronous scatter with early reset
template <typename Fixture>
void ScatterTest<Fixture>::testAsyncEarlyReset(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  const int root_rank = 0;

  std::vector<at::Tensor> inputs;
  if (rank_ == root_rank) {
    inputs = createInputTensors(count, dtype);
  }

  at::Tensor output = createOutputTensor(count, dtype);
  auto original_output = output.clone();

  auto execute = [&]() {
    auto work = torchcomm_->scatter(output, inputs, root_rank, true);
    work->wait();
    work.reset();
  };
  auto reset = [&]() { output.copy_(original_output); };
  auto verify = [&]() { verifyResults(output); };
  run(execute, reset, verify);
}

// Test function for asynchronous scatter with input deleted after enqueue
template <typename Fixture>
void ScatterTest<Fixture>::testInputDeleted(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  const int root_rank = 0;

  auto input_tensors = std::make_shared<std::vector<at::Tensor>>();
  if (rank_ == root_rank) {
    *input_tensors = createInputTensors(count, dtype);
  }

  at::Tensor output = createOutputTensor(count, dtype);
  auto original_output = output.clone();

  auto execute = [&]() {
    torchcomm_->scatter(output, *input_tensors, root_rank, false);
  };
  auto reset = [&]() { output.copy_(original_output); };
  auto verify = [&]() { verifyResults(output); };
  auto cleanup = [&]() { input_tensors.reset(); };
  run(execute, reset, verify, cleanup);
}

// Helper function to create input tensors
template <typename Fixture>
std::vector<at::Tensor> ScatterTest<Fixture>::createInputTensors(
    int count,
    at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  std::vector<at::Tensor> inputs;
  inputs.reserve(num_ranks_);
  for (int i = 0; i < num_ranks_; i++) {
    at::Tensor tensor;
    if (dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16) {
      tensor = at::ones({count}, options) * static_cast<float>(i + 1);
    } else if (dtype == at::kInt) {
      tensor = at::ones({count}, options) * static_cast<int>(i + 1);
    } else if (dtype == at::kChar) {
      tensor = at::ones({count}, options) * static_cast<signed char>(i + 1);
    }
    inputs.push_back(tensor);
  }
  return inputs;
}

// Helper function to create output tensor
template <typename Fixture>
at::Tensor ScatterTest<Fixture>::createOutputTensor(
    int count,
    at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  return at::zeros({count}, options);
}

// Helper function to verify results
template <typename Fixture>
void ScatterTest<Fixture>::verifyResults(const at::Tensor& output) {
  std::string description = "rank " + std::to_string(rank_) + " tensor";
  verifyTensorEquality(output.cpu(), rank_ + 1, description);
}

template class ScatterTest<EagerTestFixture<ScatterParams>>;
template class ScatterTest<GraphTestFixture<ScatterParams, 1>>;
template class ScatterTest<GraphTestFixture<ScatterParams, 2>>;
