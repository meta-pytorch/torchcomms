// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ReduceScatterTest.hpp"

#include <memory>

#include <gtest/gtest.h>
#include "TorchCommTestHelpers.h"

std::vector<at::Tensor> ReduceScatterHelper::createInputTensors(
    int count,
    at::ScalarType dtype,
    c10::DeviceType deviceType,
    int numRanks) {
  auto options = at::TensorOptions().dtype(dtype).device(deviceType);
  std::vector<at::Tensor> inputTensors;
  inputTensors.reserve(numRanks);
  for (int r = 0; r < numRanks; ++r) {
    at::Tensor tensor;
    if (dtype == at::kFloat || dtype == at::kBFloat16 || dtype == at::kHalf ||
        dtype == at::kDouble) {
      tensor = at::ones({count}, options) * static_cast<float>(r + 1);
    } else if (dtype == at::kInt) {
      tensor = at::ones({count}, options) * static_cast<int>(r + 1);
    } else if (dtype == at::kChar) {
      tensor = at::ones({count}, options) * static_cast<signed char>(r + 1);
    }
    inputTensors.push_back(tensor);
  }
  return inputTensors;
}

at::Tensor ReduceScatterHelper::createOutputTensor(
    int count,
    at::ScalarType dtype,
    c10::DeviceType deviceType) {
  auto options = at::TensorOptions().dtype(dtype).device(deviceType);
  return at::zeros({count}, options);
}

int ReduceScatterHelper::calculateExpectedResult(
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

void ReduceScatterHelper::verifyResults(
    const at::Tensor& output,
    const torch::comms::ReduceOp& op,
    int rank,
    int numRanks) {
  int expected = calculateExpectedResult(op, rank, numRanks);
  std::string description = "reduce_scatter with op " + getOpName(op);
  verifyTensorEquality(output.cpu(), expected, description);
}

template <typename Fixture>
void ReduceScatterTest<Fixture>::testSync(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  auto inputTensors = helper_.createInputTensors(
      count, dtype, this->deviceType_, this->numRanks_);
  auto output = helper_.createOutputTensor(count, dtype, this->deviceType_);
  auto originalOutput = output.clone();

  auto execute = [&]() {
    auto work =
        this->torchcomm_->reduce_scatter(output, inputTensors, op, false);
    work->wait();
  };
  auto reset = [&]() { output.copy_(originalOutput); };
  auto verify = [&]() {
    helper_.verifyResults(output, op, this->rank_, this->numRanks_);
  };
  this->run(execute, reset, verify);
}

template <typename Fixture>
void ReduceScatterTest<Fixture>::testSyncNoWork(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  auto inputTensors = helper_.createInputTensors(
      count, dtype, this->deviceType_, this->numRanks_);
  auto output = helper_.createOutputTensor(count, dtype, this->deviceType_);
  auto originalOutput = output.clone();

  auto execute = [&]() {
    this->torchcomm_->reduce_scatter(output, inputTensors, op, false);
  };
  auto reset = [&]() { output.copy_(originalOutput); };
  auto verify = [&]() {
    helper_.verifyResults(output, op, this->rank_, this->numRanks_);
  };
  this->run(execute, reset, verify);
}

template <typename Fixture>
void ReduceScatterTest<Fixture>::testAsync(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  auto inputTensors = helper_.createInputTensors(
      count, dtype, this->deviceType_, this->numRanks_);
  auto output = helper_.createOutputTensor(count, dtype, this->deviceType_);
  auto originalOutput = output.clone();

  auto execute = [&]() {
    auto work =
        this->torchcomm_->reduce_scatter(output, inputTensors, op, true);
    work->wait();
  };
  auto reset = [&]() { output.copy_(originalOutput); };
  auto verify = [&]() {
    helper_.verifyResults(output, op, this->rank_, this->numRanks_);
  };
  this->run(execute, reset, verify);
}

template <typename Fixture>
void ReduceScatterTest<Fixture>::testAsyncEarlyReset(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  auto inputTensors = helper_.createInputTensors(
      count, dtype, this->deviceType_, this->numRanks_);
  auto output = helper_.createOutputTensor(count, dtype, this->deviceType_);
  auto originalOutput = output.clone();

  auto execute = [&]() {
    auto work =
        this->torchcomm_->reduce_scatter(output, inputTensors, op, true);
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
void ReduceScatterTest<Fixture>::testInputDeleted(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  auto output = helper_.createOutputTensor(count, dtype, this->deviceType_);
  auto originalOutput = output.clone();
  auto inputTensors =
      std::make_shared<std::vector<at::Tensor>>(helper_.createInputTensors(
          count, dtype, this->deviceType_, this->numRanks_));

  auto execute = [&]() {
    this->torchcomm_->reduce_scatter(output, *inputTensors, op, false);
  };
  auto reset = [&]() { output.copy_(originalOutput); };
  auto verify = [&]() {
    helper_.verifyResults(output, op, this->rank_, this->numRanks_);
  };
  auto cleanup = [&]() { inputTensors.reset(); };
  this->run(execute, reset, verify, cleanup);
}

template class ReduceScatterTest<EagerTestFixture<ReduceScatterParams>>;
template class ReduceScatterTest<GraphTestFixture<ReduceScatterParams, 1>>;
template class ReduceScatterTest<GraphTestFixture<ReduceScatterParams, 2>>;
