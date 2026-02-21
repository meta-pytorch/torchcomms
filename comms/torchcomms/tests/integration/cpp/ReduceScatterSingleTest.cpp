// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "ReduceScatterSingleTest.hpp"

#include <gtest/gtest.h>
#include <memory>
#include "TorchCommTestHelpers.h"

// Test function for synchronous reduce_scatter_single with work object
template <typename Fixture>
void ReduceScatterSingleTest<Fixture>::testSync(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  at::Tensor input = createInputTensor(count, dtype);
  at::Tensor output = createOutputTensor(count, dtype);
  auto original_output = output.clone();

  auto execute = [&]() {
    auto work = torchcomm_->reduce_scatter_single(output, input, op, false);
    work->wait();
  };
  auto reset = [&]() { output.copy_(original_output); };
  auto verify = [&]() { this->verifyResults(output, op); };
  run(execute, reset, verify);
}

// Test function for synchronous reduce_scatter_single without work object
template <typename Fixture>
void ReduceScatterSingleTest<Fixture>::testSyncNoWork(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  at::Tensor input = createInputTensor(count, dtype);
  at::Tensor output = createOutputTensor(count, dtype);
  auto original_output = output.clone();

  auto execute = [&]() {
    torchcomm_->reduce_scatter_single(output, input, op, false);
  };
  auto reset = [&]() { output.copy_(original_output); };
  auto verify = [&]() { this->verifyResults(output, op); };
  run(execute, reset, verify);
}

// Test function for asynchronous reduce_scatter_single with wait
template <typename Fixture>
void ReduceScatterSingleTest<Fixture>::testAsync(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  at::Tensor input = createInputTensor(count, dtype);
  at::Tensor output = createOutputTensor(count, dtype);
  auto original_output = output.clone();

  auto execute = [&]() {
    auto work = torchcomm_->reduce_scatter_single(output, input, op, true);
    work->wait();
  };
  auto reset = [&]() { output.copy_(original_output); };
  auto verify = [&]() { this->verifyResults(output, op); };
  run(execute, reset, verify);
}

// Test function for asynchronous reduce_scatter_single with early reset
template <typename Fixture>
void ReduceScatterSingleTest<Fixture>::testAsyncEarlyReset(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  at::Tensor input = createInputTensor(count, dtype);
  at::Tensor output = createOutputTensor(count, dtype);
  auto original_output = output.clone();

  auto execute = [&]() {
    auto work = torchcomm_->reduce_scatter_single(output, input, op, true);
    work->wait();
    work.reset();
  };
  auto reset = [&]() { output.copy_(original_output); };
  auto verify = [&]() { this->verifyResults(output, op); };
  run(execute, reset, verify);
}

// Test function for asynchronous reduce_scatter_single with input deleted after
// enqueue
template <typename Fixture>
void ReduceScatterSingleTest<Fixture>::testInputDeleted(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  at::Tensor output = createOutputTensor(count, dtype);
  auto original_output = output.clone();
  auto input = std::make_shared<at::Tensor>(createInputTensor(count, dtype));

  auto execute = [&]() {
    torchcomm_->reduce_scatter_single(output, *input, op, false);
  };
  auto reset = [&]() { output.copy_(original_output); };
  auto verify = [&]() { this->verifyResults(output, op); };
  auto cleanup = [&]() { input.reset(); };
  run(execute, reset, verify, cleanup);
}

// Helper function to create input tensor
template <typename Fixture>
at::Tensor ReduceScatterSingleTest<Fixture>::createInputTensor(
    int count,
    at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  at::Tensor input = at::zeros({count * num_ranks_}, options);

  for (int r = 0; r < num_ranks_; ++r) {
    auto section = input.slice(0, r * count, (r + 1) * count);
    if (dtype == at::kFloat || dtype == at::kBFloat16 || dtype == at::kHalf ||
        dtype == at::kDouble) {
      section.fill_(static_cast<float>(r + 1));
    } else if (dtype == at::kInt) {
      section.fill_(static_cast<int>(r + 1));
    } else if (dtype == at::kChar) {
      section.fill_(static_cast<signed char>(r + 1));
    }
  }

  return input;
}

// Helper function to create output tensor
template <typename Fixture>
at::Tensor ReduceScatterSingleTest<Fixture>::createOutputTensor(
    int count,
    at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  return at::zeros({count}, options);
}

// Helper function to verify results
template <typename Fixture>
void ReduceScatterSingleTest<Fixture>::verifyResults(
    const at::Tensor& output,
    const torch::comms::ReduceOp& op) {
  // Calculate expected value based on operation type
  int expected_value = 0;
  if (op == torch::comms::ReduceOp::SUM) {
    // Sum: num_ranks * (rank+1)
    expected_value = num_ranks_ * (rank_ + 1);
  } else if (op == torch::comms::ReduceOp::MAX) {
    // Max: rank+1
    expected_value = rank_ + 1;
  } else if (op == torch::comms::ReduceOp::AVG) {
    // Avg: (num_ranks * (rank+1)) / num_ranks = rank+1
    expected_value = rank_ + 1;
  }

  // Use verifyTensorEquality with integer expected value
  std::string description = "reduce_scatter_single with op " + getOpName(op);
  verifyTensorEquality(output.cpu(), expected_value, description);
}

template class ReduceScatterSingleTest<
    EagerTestFixture<ReduceScatterSingleParams>>;
template class ReduceScatterSingleTest<
    GraphTestFixture<ReduceScatterSingleParams, 1>>;
template class ReduceScatterSingleTest<
    GraphTestFixture<ReduceScatterSingleParams, 2>>;
