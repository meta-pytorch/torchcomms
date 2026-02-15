// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ReduceScatterSingleTest.hpp"

#include <memory>

#include <gtest/gtest.h>
#include "TorchCommTestHelpers.h"

at::Tensor ReduceScatterSingleHelper::createInputTensor(
    int count,
    at::ScalarType dtype,
    c10::DeviceType deviceType,
    int numRanks) {
  auto options = at::TensorOptions().dtype(dtype).device(deviceType);
  at::Tensor input = at::zeros({count * numRanks}, options);

  auto ranks = at::arange(1, numRanks + 1, options);

  for (int r = 0; r < numRanks; ++r) {
    auto section = input.slice(0, r * count, (r + 1) * count);
    section.fill_(ranks[r].item());
  }

  return input;
}

at::Tensor ReduceScatterSingleHelper::createOutputTensor(
    int count,
    at::ScalarType dtype,
    c10::DeviceType deviceType) {
  auto options = at::TensorOptions().dtype(dtype).device(deviceType);
  return at::zeros({count}, options);
}

int ReduceScatterSingleHelper::calculateExpectedResult(
    const torch::comms::ReduceOp& op,
    int rank,
    int numRanks) {
  if (op == torch::comms::ReduceOp::SUM) {
    return numRanks * (rank + 1);
  } else if (op == torch::comms::ReduceOp::RedOpType::MAX) {
    return rank + 1;
  } else if (op == torch::comms::ReduceOp::RedOpType::AVG) {
    return rank + 1;
  } else {
    throw std::runtime_error("Unsupported reduce operation");
  }
}

void ReduceScatterSingleHelper::verifyResults(
    const at::Tensor& output,
    const torch::comms::ReduceOp& op,
    int rank,
    int numRanks) {
  int expected = calculateExpectedResult(op, rank, numRanks);
  std::string description = "reduce_scatter_single with op " + getOpName(op);
  verifyTensorEquality(output.cpu(), expected, description);
}

template <typename Fixture>
void ReduceScatterSingleTest<Fixture>::testSync(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  auto input = helper_.createInputTensor(
      count, dtype, this->deviceType_, this->numRanks_);
  auto output = helper_.createOutputTensor(count, dtype, this->deviceType_);
  auto originalOutput = output.clone();

  auto execute = [&]() {
    auto work =
        this->torchcomm_->reduce_scatter_single(output, input, op, false);
    work->wait();
  };
  auto reset = [&]() { output.copy_(originalOutput); };
  auto verify = [&]() {
    helper_.verifyResults(output, op, this->rank_, this->numRanks_);
  };
  this->run(execute, reset, verify);
}

template <typename Fixture>
void ReduceScatterSingleTest<Fixture>::testSyncNoWork(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  auto input = helper_.createInputTensor(
      count, dtype, this->deviceType_, this->numRanks_);
  auto output = helper_.createOutputTensor(count, dtype, this->deviceType_);
  auto originalOutput = output.clone();

  auto execute = [&]() {
    this->torchcomm_->reduce_scatter_single(output, input, op, false);
  };
  auto reset = [&]() { output.copy_(originalOutput); };
  auto verify = [&]() {
    helper_.verifyResults(output, op, this->rank_, this->numRanks_);
  };
  this->run(execute, reset, verify);
}

template <typename Fixture>
void ReduceScatterSingleTest<Fixture>::testAsync(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  auto input = helper_.createInputTensor(
      count, dtype, this->deviceType_, this->numRanks_);
  auto output = helper_.createOutputTensor(count, dtype, this->deviceType_);
  auto originalOutput = output.clone();

  auto execute = [&]() {
    auto work =
        this->torchcomm_->reduce_scatter_single(output, input, op, true);
    work->wait();
  };
  auto reset = [&]() { output.copy_(originalOutput); };
  auto verify = [&]() {
    helper_.verifyResults(output, op, this->rank_, this->numRanks_);
  };
  this->run(execute, reset, verify);
}

template <typename Fixture>
void ReduceScatterSingleTest<Fixture>::testAsyncEarlyReset(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  auto input = helper_.createInputTensor(
      count, dtype, this->deviceType_, this->numRanks_);
  auto output = helper_.createOutputTensor(count, dtype, this->deviceType_);
  auto originalOutput = output.clone();

  auto execute = [&]() {
    auto work =
        this->torchcomm_->reduce_scatter_single(output, input, op, true);
    work->wait();
    work.reset();
  };
  auto reset = [&]() { output.copy_(originalOutput); };
  auto verify = [&]() {
    helper_.verifyResults(output, op, this->rank_, this->numRanks_);
  };
  this->run(execute, reset, verify);
}

template <typename Fixture>
void ReduceScatterSingleTest<Fixture>::testInputDeleted(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  auto output = helper_.createOutputTensor(count, dtype, this->deviceType_);
  auto originalOutput = output.clone();
  auto input = std::make_shared<at::Tensor>(helper_.createInputTensor(
      count, dtype, this->deviceType_, this->numRanks_));

  auto execute = [&]() {
    this->torchcomm_->reduce_scatter_single(output, *input, op, false);
  };
  auto reset = [&]() { output.copy_(originalOutput); };
  auto verify = [&]() {
    helper_.verifyResults(output, op, this->rank_, this->numRanks_);
  };
  auto cleanup = [&]() { input.reset(); };
  this->run(execute, reset, verify, cleanup);
}

template class ReduceScatterSingleTest<
    EagerTestFixture<ReduceScatterSingleParams>>;
template class ReduceScatterSingleTest<
    GraphTestFixture<ReduceScatterSingleParams, 1>>;
template class ReduceScatterSingleTest<
    GraphTestFixture<ReduceScatterSingleParams, 2>>;
