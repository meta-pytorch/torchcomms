// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "AllReduceTest.hpp"

#include <memory>

#include <gtest/gtest.h>
#include "TorchCommTestHelpers.h"

at::Tensor AllReduceHelper::createInputTensor(
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

at::Tensor AllReduceHelper::createPreMulFactorTensor(
    at::ScalarType dtype,
    c10::DeviceType deviceType) {
  auto options = at::TensorOptions().dtype(dtype).device(deviceType);
  at::Tensor factor;
  if (dtype == at::kFloat || dtype == at::kBFloat16 || dtype == at::kHalf ||
      dtype == at::kDouble) {
    factor = at::ones({1}, options) * static_cast<float>(2.0);
  } else {
    throw std::runtime_error("Unsupported dtype for PreMul");
  }
  return factor;
}

double AllReduceHelper::calculateExpectedResult(
    const torch::comms::ReduceOp& op,
    int numRanks) {
  if (op == torch::comms::ReduceOp::SUM) {
    return numRanks * (numRanks + 1) / 2;
  } else if (op == torch::comms::ReduceOp::RedOpType::MAX) {
    return numRanks;
  } else if (op == torch::comms::ReduceOp::RedOpType::AVG) {
    return static_cast<double>(numRanks * (numRanks + 1) / 2) / numRanks;
  } else if (op == torch::comms::ReduceOp::RedOpType::PREMUL_SUM) {
    return numRanks * (numRanks + 1);
  } else {
    throw std::runtime_error("Unsupported reduce operation");
  }
}

void AllReduceHelper::verifyResults(
    const at::Tensor& input,
    const torch::comms::ReduceOp& op,
    int numRanks) {
  double expected = calculateExpectedResult(op, numRanks);
  std::string description = "all_reduce with op " + getOpName(op);
  verifyTensorEquality(input.cpu(), expected, description);
}

template <typename Fixture>
void AllReduceTest<Fixture>::testSync(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  auto input =
      helper_.createInputTensor(count, dtype, this->deviceType_, this->rank_);
  auto original = input.clone();

  auto execute = [&]() {
    auto work = this->torchcomm_->all_reduce(input, op, false);
    work->wait();
  };
  auto reset = [&]() { input.copy_(original); };
  auto verify = [&]() { helper_.verifyResults(input, op, this->numRanks_); };
  this->run(execute, reset, verify);
}

template <typename Fixture>
void AllReduceTest<Fixture>::testSyncNoWork(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  auto input =
      helper_.createInputTensor(count, dtype, this->deviceType_, this->rank_);
  auto original = input.clone();

  auto execute = [&]() { this->torchcomm_->all_reduce(input, op, false); };
  auto reset = [&]() { input.copy_(original); };
  auto verify = [&]() { helper_.verifyResults(input, op, this->numRanks_); };
  this->run(execute, reset, verify);
}

template <typename Fixture>
void AllReduceTest<Fixture>::testAsync(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  auto input =
      helper_.createInputTensor(count, dtype, this->deviceType_, this->rank_);
  auto original = input.clone();

  auto execute = [&]() {
    auto work = this->torchcomm_->all_reduce(input, op, true);
    work->wait();
  };
  auto reset = [&]() { input.copy_(original); };
  auto verify = [&]() { helper_.verifyResults(input, op, this->numRanks_); };
  this->run(execute, reset, verify);
}

template <typename Fixture>
void AllReduceTest<Fixture>::testAsyncEarlyReset(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  auto input =
      helper_.createInputTensor(count, dtype, this->deviceType_, this->rank_);
  auto original = input.clone();

  auto execute = [&]() {
    auto work = this->torchcomm_->all_reduce(input, op, true);
    work->wait();
    work.reset();
  };
  auto reset = [&]() { input.copy_(original); };
  auto verify = [&]() { helper_.verifyResults(input, op, this->numRanks_); };
  this->run(execute, reset, verify);
}

template <typename Fixture>
void AllReduceTest<Fixture>::testInputDeleted(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  auto input = std::make_shared<at::Tensor>(
      helper_.createInputTensor(count, dtype, this->deviceType_, this->rank_));

  auto execute = [&]() { this->torchcomm_->all_reduce(*input, op, false); };
  auto cleanup = [&]() { input.reset(); };
  this->run(execute, {}, {}, cleanup);
}

template class AllReduceTest<EagerTestFixture<AllReduceParams>>;
template class AllReduceTest<GraphTestFixture<AllReduceParams, 1>>;
template class AllReduceTest<GraphTestFixture<AllReduceParams, 2>>;

template class AllReduceTest<EagerTestFixture<PreMulSumParams>>;
template class AllReduceTest<GraphTestFixture<PreMulSumParams, 1>>;
template class AllReduceTest<GraphTestFixture<PreMulSumParams, 2>>;
