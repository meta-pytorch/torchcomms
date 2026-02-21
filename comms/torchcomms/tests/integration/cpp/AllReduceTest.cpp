// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "AllReduceTest.hpp"

#include <gtest/gtest.h>
#include <memory>
#include "TorchCommTestHelpers.h"

// Test function for synchronous all_reduce with work object
template <typename Fixture>
void AllReduceTest<Fixture>::testSync(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  at::Tensor input = createInputTensor(count, dtype);
  auto original = input.clone();

  auto execute = [&]() {
    auto work = torchcomm_->all_reduce(input, op, false);
    work->wait();
  };
  auto reset = [&]() { input.copy_(original); };
  auto verify = [&]() { this->verifyResults(input, op); };
  run(execute, reset, verify);
}

// Test function for synchronous all_reduce without work object
template <typename Fixture>
void AllReduceTest<Fixture>::testSyncNoWork(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  at::Tensor input = createInputTensor(count, dtype);
  auto original = input.clone();

  auto execute = [&]() { torchcomm_->all_reduce(input, op, false); };
  auto reset = [&]() { input.copy_(original); };
  auto verify = [&]() { this->verifyResults(input, op); };
  run(execute, reset, verify);
}

// Test function for asynchronous all_reduce with wait
template <typename Fixture>
void AllReduceTest<Fixture>::testAsync(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  at::Tensor input = createInputTensor(count, dtype);
  auto original = input.clone();

  auto execute = [&]() {
    auto work = torchcomm_->all_reduce(input, op, true);
    work->wait();
  };
  auto reset = [&]() { input.copy_(original); };
  auto verify = [&]() { this->verifyResults(input, op); };
  run(execute, reset, verify);
}

// Test function for asynchronous all_reduce with early reset
template <typename Fixture>
void AllReduceTest<Fixture>::testAsyncEarlyReset(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  at::Tensor input = createInputTensor(count, dtype);
  auto original = input.clone();

  auto execute = [&]() {
    auto work = torchcomm_->all_reduce(input, op, true);
    work->wait();
    work.reset();
  };
  auto reset = [&]() { input.copy_(original); };
  auto verify = [&]() { this->verifyResults(input, op); };
  run(execute, reset, verify);
}

// Test function for asynchronous all_reduce with input deleted after enqueue
template <typename Fixture>
void AllReduceTest<Fixture>::testInputDeleted(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  auto input = std::make_shared<at::Tensor>(createInputTensor(count, dtype));

  auto execute = [&]() { torchcomm_->all_reduce(*input, op, false); };
  auto cleanup = [&]() { input.reset(); };
  run(execute, {}, {}, cleanup);
}

// Helper function to create input tensor
template <typename Fixture>
at::Tensor AllReduceTest<Fixture>::createInputTensor(
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

// Helper function to create PreMul tensor
template <typename Fixture>
at::Tensor AllReduceTest<Fixture>::createPreMulFactorTensor(
    at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  at::Tensor factor;
  if (dtype == at::kFloat || dtype == at::kBFloat16 || dtype == at::kHalf ||
      dtype == at::kDouble) {
    factor = at::ones({1}, options) * static_cast<float>(2.0);
  } else {
    throw std::runtime_error("Unsupported dtype for PreMul");
  }
  return factor;
}

// Helper function to calculate expected result
template <typename Fixture>
double AllReduceTest<Fixture>::calculateExpectedResult(
    const torch::comms::ReduceOp& op) {
  if (op == torch::comms::ReduceOp::SUM) {
    return num_ranks_ * (num_ranks_ + 1) / 2;
  } else if (op == torch::comms::ReduceOp::RedOpType::MAX) {
    return num_ranks_;
  } else if (op == torch::comms::ReduceOp::RedOpType::AVG) {
    return static_cast<double>(num_ranks_ * (num_ranks_ + 1) / 2) / num_ranks_;
  } else if (op == torch::comms::ReduceOp::RedOpType::PREMUL_SUM) {
    return num_ranks_ * (num_ranks_ + 1);
  } else {
    throw std::runtime_error("Unsupported reduce operation");
  }
}

// Helper function to verify results
template <typename Fixture>
void AllReduceTest<Fixture>::verifyResults(
    const at::Tensor& input,
    const torch::comms::ReduceOp& op) {
  double expected = calculateExpectedResult(op);
  std::string description = "all_reduce with op " + getOpName(op);
  verifyTensorEquality(input.cpu(), expected, description);
}

template class AllReduceTest<EagerTestFixture<AllReduceParams>>;
template class AllReduceTest<GraphTestFixture<AllReduceParams, 1>>;
template class AllReduceTest<GraphTestFixture<AllReduceParams, 2>>;

template class AllReduceTest<EagerTestFixture<PreMulSumParams>>;
template class AllReduceTest<GraphTestFixture<PreMulSumParams, 1>>;
template class AllReduceTest<GraphTestFixture<PreMulSumParams, 2>>;
