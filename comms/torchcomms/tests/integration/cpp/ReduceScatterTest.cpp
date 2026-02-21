// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "ReduceScatterTest.hpp"

#include <gtest/gtest.h>
#include <memory>
#include "TorchCommTestHelpers.h"

// Test function for synchronous reduce_scatter with work object
template <typename Fixture>
void ReduceScatterTest<Fixture>::testSync(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  std::vector<at::Tensor> input_tensors = createInputTensors(count, dtype);
  at::Tensor output = createOutputTensor(count, dtype);
  at::Tensor original_output = output.clone();

  auto execute = [&]() {
    auto work = torchcomm_->reduce_scatter(output, input_tensors, op, false);
    work->wait();
  };
  auto reset = [&]() { output.copy_(original_output); };
  auto verify = [&]() { this->verifyResults(output, op); };
  run(execute, reset, verify);
}

// Test function for synchronous reduce_scatter without work object
template <typename Fixture>
void ReduceScatterTest<Fixture>::testSyncNoWork(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  std::vector<at::Tensor> input_tensors = createInputTensors(count, dtype);
  at::Tensor output = createOutputTensor(count, dtype);
  at::Tensor original_output = output.clone();

  auto execute = [&]() {
    torchcomm_->reduce_scatter(output, input_tensors, op, false);
  };
  auto reset = [&]() { output.copy_(original_output); };
  auto verify = [&]() { this->verifyResults(output, op); };
  run(execute, reset, verify);
}

// Test function for asynchronous reduce_scatter with wait
template <typename Fixture>
void ReduceScatterTest<Fixture>::testAsync(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  std::vector<at::Tensor> input_tensors = createInputTensors(count, dtype);
  at::Tensor output = createOutputTensor(count, dtype);
  at::Tensor original_output = output.clone();

  auto execute = [&]() {
    auto work = torchcomm_->reduce_scatter(output, input_tensors, op, true);
    work->wait();
  };
  auto reset = [&]() { output.copy_(original_output); };
  auto verify = [&]() { this->verifyResults(output, op); };
  run(execute, reset, verify);
}

// Test function for asynchronous reduce_scatter with early reset
template <typename Fixture>
void ReduceScatterTest<Fixture>::testAsyncEarlyReset(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  std::vector<at::Tensor> input_tensors = createInputTensors(count, dtype);
  at::Tensor output = createOutputTensor(count, dtype);
  at::Tensor original_output = output.clone();

  auto execute = [&]() {
    auto work = torchcomm_->reduce_scatter(output, input_tensors, op, true);
    work->wait();
    work.reset();
  };
  auto reset = [&]() { output.copy_(original_output); };
  auto verify = [&]() { this->verifyResults(output, op); };
  run(execute, reset, verify);
}

// Test function for asynchronous reduce_scatter with input deleted after
// enqueue
template <typename Fixture>
void ReduceScatterTest<Fixture>::testInputDeleted(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  at::Tensor output = createOutputTensor(count, dtype);
  at::Tensor original_output = output.clone();
  auto input_tensors = std::make_shared<std::vector<at::Tensor>>(
      createInputTensors(count, dtype));

  auto execute = [&]() {
    torchcomm_->reduce_scatter(output, *input_tensors, op, false);
  };
  auto reset = [&]() { output.copy_(original_output); };
  auto verify = [&]() { this->verifyResults(output, op); };
  auto cleanup = [&]() { input_tensors.reset(); };
  run(execute, reset, verify, cleanup);
}

// Helper function to create input tensors
template <typename Fixture>
std::vector<at::Tensor> ReduceScatterTest<Fixture>::createInputTensors(
    int count,
    at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  std::vector<at::Tensor> input_tensors;
  input_tensors.reserve(num_ranks_);
  for (int r = 0; r < num_ranks_; ++r) {
    at::Tensor tensor;
    if (dtype == at::kFloat || dtype == at::kBFloat16 || dtype == at::kHalf ||
        dtype == at::kDouble) {
      tensor = at::ones({count}, options) * static_cast<float>(r + 1);
    } else if (dtype == at::kInt) {
      tensor = at::ones({count}, options) * static_cast<int>(r + 1);
    } else if (dtype == at::kChar) {
      tensor = at::ones({count}, options) * static_cast<signed char>(r + 1);
    }
    input_tensors.push_back(tensor);
  }
  return input_tensors;
}

// Helper function to create output tensor
template <typename Fixture>
at::Tensor ReduceScatterTest<Fixture>::createOutputTensor(
    int count,
    at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  return at::zeros({count}, options);
}

// Helper function to calculate expected result
template <typename Fixture>
int ReduceScatterTest<Fixture>::calculateExpectedResult(
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
template <typename Fixture>
void ReduceScatterTest<Fixture>::verifyResults(
    const at::Tensor& output,
    const torch::comms::ReduceOp& op) {
  int expected = calculateExpectedResult(op);
  std::string description = "reduce_scatter with op " + getOpName(op);
  verifyTensorEquality(output.cpu(), expected, description);
}

template class ReduceScatterTest<EagerTestFixture<ReduceScatterParams>>;
template class ReduceScatterTest<GraphTestFixture<ReduceScatterParams, 1>>;
template class ReduceScatterTest<GraphTestFixture<ReduceScatterParams, 2>>;
