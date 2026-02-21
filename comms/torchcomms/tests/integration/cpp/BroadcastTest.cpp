// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "BroadcastTest.hpp"

#include <gtest/gtest.h>
#include <memory>
#include "TorchCommTestHelpers.h"

// Test function for synchronous broadcast with work object
template <typename Fixture>
void BroadcastTest<Fixture>::testSync(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  const int root_rank = 0;
  const int root_value = 99;

  at::Tensor tensor = createTensor(root_rank, root_value, count, dtype);
  auto originalTensor = tensor.clone();

  auto execute = [&]() {
    auto work = torchcomm_->broadcast(tensor, root_rank, false);
    work->wait();
  };
  auto reset = [&]() { tensor.copy_(originalTensor); };
  auto verify = [&]() { verifyResults(tensor, root_value); };
  run(execute, reset, verify);
}

// Test function for synchronous broadcast without work object
template <typename Fixture>
void BroadcastTest<Fixture>::testSyncNoWork(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  const int root_rank = 0;
  const int root_value = 99;

  at::Tensor tensor = createTensor(root_rank, root_value, count, dtype);
  auto originalTensor = tensor.clone();

  auto execute = [&]() { torchcomm_->broadcast(tensor, root_rank, false); };
  auto reset = [&]() { tensor.copy_(originalTensor); };
  auto verify = [&]() { verifyResults(tensor, root_value); };
  run(execute, reset, verify);
}

// Test function for asynchronous broadcast with wait
template <typename Fixture>
void BroadcastTest<Fixture>::testAsync(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  const int root_rank = 0;
  const int root_value = 99;

  at::Tensor tensor = createTensor(root_rank, root_value, count, dtype);
  auto originalTensor = tensor.clone();

  auto execute = [&]() {
    auto work = torchcomm_->broadcast(tensor, root_rank, true);
    work->wait();
  };
  auto reset = [&]() { tensor.copy_(originalTensor); };
  auto verify = [&]() { verifyResults(tensor, root_value); };
  run(execute, reset, verify);
}

// Test function for asynchronous broadcast with early reset
template <typename Fixture>
void BroadcastTest<Fixture>::testAsyncEarlyReset(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  const int root_rank = 0;
  const int root_value = 99;

  at::Tensor tensor = createTensor(root_rank, root_value, count, dtype);
  auto originalTensor = tensor.clone();

  auto execute = [&]() {
    auto work = torchcomm_->broadcast(tensor, root_rank, true);
    work->wait();
    work.reset();
  };
  auto reset = [&]() { tensor.copy_(originalTensor); };
  auto verify = [&]() { verifyResults(tensor, root_value); };
  run(execute, reset, verify);
}

// Test function for asynchronous broadcast with input deleted after enqueue
template <typename Fixture>
void BroadcastTest<Fixture>::testInputDeleted(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  const int root_rank = 0;
  const int root_value = 99;

  auto tensor = std::make_shared<at::Tensor>(
      createTensor(root_rank, root_value, count, dtype));

  auto execute = [&]() { torchcomm_->broadcast(*tensor, root_rank, false); };
  auto cleanup = [&]() { tensor.reset(); };
  run(execute, {}, {}, cleanup);
}

// Helper function to create tensor for broadcast
template <typename Fixture>
at::Tensor BroadcastTest<Fixture>::createTensor(
    int root_rank,
    int value,
    int count,
    at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  at::Tensor tensor;
  if (dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16) {
    tensor = rank_ == root_rank
        ? at::ones({count}, options) * static_cast<float>(value)
        : at::zeros({count}, options);
  } else if (dtype == at::kInt) {
    tensor = rank_ == root_rank
        ? at::ones({count}, options) * static_cast<int>(value)
        : at::zeros({count}, options);
  } else if (dtype == at::kChar) {
    tensor = rank_ == root_rank
        ? at::ones({count}, options) * static_cast<signed char>(value)
        : at::zeros({count}, options);
  }
  return tensor;
}

// Helper function to verify results
template <typename Fixture>
void BroadcastTest<Fixture>::verifyResults(
    const at::Tensor& tensor,
    int value) {
  std::string description = "broadcast with value " + std::to_string(value);
  verifyTensorEquality(tensor.cpu(), value, description);
}

template class BroadcastTest<EagerTestFixture<BroadcastParams>>;
template class BroadcastTest<GraphTestFixture<BroadcastParams, 1>>;
template class BroadcastTest<GraphTestFixture<BroadcastParams, 2>>;
