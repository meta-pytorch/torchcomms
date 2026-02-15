// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ReduceTest.hpp"

#include <memory>

#include <gtest/gtest.h>
#include "TorchCommTestHelpers.h"

at::Tensor ReduceHelper::createInputTensor(
    int count,
    at::ScalarType dtype,
    c10::DeviceType deviceType,
    int rank) {
  auto options = at::TensorOptions().dtype(dtype).device(deviceType);
  at::Tensor input;
  if (dtype == at::kFloat || dtype == at::kBFloat16 || dtype == at::kHalf ||
      dtype == at::kDouble) {
    input = at::ones({count}, options) * static_cast<float>(rank + 1);
  } else if (dtype == at::kInt) {
    input = at::ones({count}, options) * static_cast<int>(rank + 1);
  } else if (dtype == at::kChar) {
    input = at::ones({count}, options) * static_cast<signed char>(rank + 1);
  }
  return input;
}

void ReduceHelper::synchronizeStream(c10::DeviceType deviceType) {
  if (deviceType != c10::DeviceType::CPU) {
    at::cuda::getCurrentCUDAStream(0).synchronize();
  }
}

double ReduceHelper::calculateExpectedResult(
    const torch::comms::ReduceOp& op,
    int numRanks) {
  if (op == torch::comms::ReduceOp::SUM) {
    return numRanks * (numRanks + 1) / 2;
  } else if (op == torch::comms::ReduceOp::RedOpType::MAX) {
    return numRanks;
  } else if (op == torch::comms::ReduceOp::RedOpType::AVG) {
    return static_cast<double>(numRanks + 1) / 2.0;
  } else {
    throw std::runtime_error("Unsupported reduce operation");
  }
}

void ReduceHelper::verifyResults(
    const at::Tensor& output,
    const torch::comms::ReduceOp& op,
    int rank,
    int numRanks,
    c10::DeviceType deviceType) {
  if (rank != 0) {
    synchronizeStream(deviceType);
    return;
  }

  double expected = calculateExpectedResult(op, numRanks);
  std::string description = "reduce with op " + getOpName(op);
  verifyTensorEquality(output.cpu(), expected, description);
}

template <typename Fixture>
void ReduceTest<Fixture>::testSync(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  const int root_rank = 0;
  auto input =
      helper_.createInputTensor(count, dtype, this->deviceType_, this->rank_);
  auto original = input.clone();

  auto execute = [&]() {
    auto work = this->torchcomm_->reduce(input, root_rank, op, false);
    work->wait();
  };
  auto reset = [&]() { input.copy_(original); };
  auto verify = [&]() {
    helper_.verifyResults(
        input, op, this->rank_, this->numRanks_, this->deviceType_);
  };
  this->run(execute, reset, verify);
}

template <typename Fixture>
void ReduceTest<Fixture>::testSyncNoWork(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  const int root_rank = 0;
  auto input =
      helper_.createInputTensor(count, dtype, this->deviceType_, this->rank_);
  auto original = input.clone();

  auto execute = [&]() {
    this->torchcomm_->reduce(input, root_rank, op, false);
  };
  auto reset = [&]() { input.copy_(original); };
  auto verify = [&]() {
    helper_.verifyResults(
        input, op, this->rank_, this->numRanks_, this->deviceType_);
  };
  this->run(execute, reset, verify);
}

template <typename Fixture>
void ReduceTest<Fixture>::testAsync(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  const int root_rank = 0;
  auto input =
      helper_.createInputTensor(count, dtype, this->deviceType_, this->rank_);
  auto original = input.clone();

  auto execute = [&]() {
    auto work = this->torchcomm_->reduce(input, root_rank, op, true);
    work->wait();
  };
  auto reset = [&]() { input.copy_(original); };
  auto verify = [&]() {
    helper_.verifyResults(
        input, op, this->rank_, this->numRanks_, this->deviceType_);
  };
  this->run(execute, reset, verify);
}

template <typename Fixture>
void ReduceTest<Fixture>::testAsyncEarlyReset(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  const int root_rank = 0;
  auto input =
      helper_.createInputTensor(count, dtype, this->deviceType_, this->rank_);
  auto original = input.clone();

  auto execute = [&]() {
    auto work = this->torchcomm_->reduce(input, root_rank, op, true);
    work->wait();
    work.reset();
  };
  auto reset = [&]() { input.copy_(original); };
  auto verify = [&]() {
    helper_.verifyResults(
        input, op, this->rank_, this->numRanks_, this->deviceType_);
  };
  this->run(execute, reset, verify);
}

template <typename Fixture>
void ReduceTest<Fixture>::testInputDeleted(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  const int root_rank = 0;
  auto input = std::make_shared<at::Tensor>(
      helper_.createInputTensor(count, dtype, this->deviceType_, this->rank_));

  auto execute = [&]() {
    this->torchcomm_->reduce(*input, root_rank, op, false);
  };
  auto cleanup = [&]() { input.reset(); };
  this->run(execute, {}, {}, cleanup);
}

template class ReduceTest<EagerTestFixture<ReduceParams>>;
template class ReduceTest<GraphTestFixture<ReduceParams, 1>>;
template class ReduceTest<GraphTestFixture<ReduceParams, 2>>;
