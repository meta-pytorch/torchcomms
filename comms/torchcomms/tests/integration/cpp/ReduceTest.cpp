// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "ReduceTest.hpp"

#include <gtest/gtest.h>
#include <memory>
#include "TorchCommTestHelpers.h"

// Test function for synchronous reduce with work object
template <typename Fixture>
void ReduceTest<Fixture>::testSync(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  const int root_rank = 0;
  at::Tensor input = createInputTensor(count, dtype);
  auto original = input.clone();

  auto execute = [&]() {
    auto work = torchcomm_->reduce(input, root_rank, op, false);
    work->wait();
  };
  auto reset = [&]() { input.copy_(original); };
  auto verify = [&]() {
    if (rank_ == root_rank) {
      this->verifyResults(input, op);
    } else {
      this->synchronizeStream();
    }
  };
  run(execute, reset, verify);
}

// Test function for synchronous reduce without work object
template <typename Fixture>
void ReduceTest<Fixture>::testSyncNoWork(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  const int root_rank = 0;
  at::Tensor input = createInputTensor(count, dtype);
  auto original = input.clone();

  auto execute = [&]() { torchcomm_->reduce(input, root_rank, op, false); };
  auto reset = [&]() { input.copy_(original); };
  auto verify = [&]() {
    if (rank_ == root_rank) {
      this->verifyResults(input, op);
    } else {
      this->synchronizeStream();
    }
  };
  run(execute, reset, verify);
}

// Test function for asynchronous reduce with wait
template <typename Fixture>
void ReduceTest<Fixture>::testAsync(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  const int root_rank = 0;
  at::Tensor input = createInputTensor(count, dtype);
  auto original = input.clone();

  auto execute = [&]() {
    auto work = torchcomm_->reduce(input, root_rank, op, true);
    work->wait();
  };
  auto reset = [&]() { input.copy_(original); };
  auto verify = [&]() {
    if (rank_ == root_rank) {
      this->verifyResults(input, op);
    } else {
      this->synchronizeStream();
    }
  };
  run(execute, reset, verify);
}

// Test function for asynchronous reduce with early reset
template <typename Fixture>
void ReduceTest<Fixture>::testAsyncEarlyReset(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  const int root_rank = 0;
  at::Tensor input = createInputTensor(count, dtype);
  auto original = input.clone();

  auto execute = [&]() {
    auto work = torchcomm_->reduce(input, root_rank, op, true);
    work->wait();
    work.reset();
  };
  auto reset = [&]() { input.copy_(original); };
  auto verify = [&]() {
    if (rank_ == root_rank) {
      this->verifyResults(input, op);
    } else {
      this->synchronizeStream();
    }
  };
  run(execute, reset, verify);
}

// Test function for asynchronous reduce with input deleted after enqueue
template <typename Fixture>
void ReduceTest<Fixture>::testInputDeleted(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  const int root_rank = 0;
  auto input = std::make_shared<at::Tensor>(createInputTensor(count, dtype));

  auto execute = [&]() { torchcomm_->reduce(*input, root_rank, op, false); };
  auto cleanup = [&]() { input.reset(); };
  run(execute, {}, {}, cleanup);
}

// Helper function to create input tensor
template <typename Fixture>
at::Tensor ReduceTest<Fixture>::createInputTensor(
    int count,
    at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  at::Tensor input;
  if (dtype == at::kFloat || dtype == at::kBFloat16 || dtype == at::kHalf ||
      dtype == at::kDouble) {
    input = at::ones({count}, options) * static_cast<float>(rank_ + 1);
  } else if (dtype == at::kInt) {
    input = at::ones({count}, options) * static_cast<int>(rank_ + 1);
  } else if (dtype == at::kChar) {
    input = at::ones({count}, options) * static_cast<signed char>(rank_ + 1);
  }
  return input;
}

// Helper function to calculate expected result
template <typename Fixture>
double ReduceTest<Fixture>::calculateExpectedResult(
    const torch::comms::ReduceOp& op) {
  if (op == torch::comms::ReduceOp::SUM) {
    return num_ranks_ * (num_ranks_ + 1) / 2;
  } else if (op == torch::comms::ReduceOp::MAX) {
    return num_ranks_;
  } else if (op == torch::comms::ReduceOp::AVG) {
    return static_cast<double>(num_ranks_ + 1) / 2.0;
  } else {
    throw std::runtime_error("Unsupported reduce operation");
  }
}

// Helper function to verify results
template <typename Fixture>
void ReduceTest<Fixture>::verifyResults(
    const at::Tensor& output,
    const torch::comms::ReduceOp& op) {
  double expected = calculateExpectedResult(op);
  std::string description = "reduce with op " + getOpName(op);
  verifyTensorEquality(output.cpu(), expected, description);
}

template <typename Fixture>
void ReduceTest<Fixture>::synchronizeStream() {
  if (device_type_ == c10::DeviceType::CUDA) {
    at::cuda::getCurrentCUDAStream(0).synchronize();
  }
}

template class ReduceTest<EagerTestFixture<ReduceParams>>;
template class ReduceTest<GraphTestFixture<ReduceParams, 1>>;
template class ReduceTest<GraphTestFixture<ReduceParams, 2>>;
