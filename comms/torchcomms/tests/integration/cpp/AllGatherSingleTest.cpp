// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "AllGatherSingleTest.hpp"

#include <gtest/gtest.h>
#include <memory>
#include "TorchCommTestHelpers.h"

// Test function for synchronous all_gather_single with work object
template <typename Fixture>
void AllGatherSingleTest<Fixture>::testSync(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  at::Tensor input = createInputTensor(count, dtype);
  at::Tensor output = createOutputTensor(count, dtype);
  auto original_output = output.clone();

  auto execute = [&]() {
    auto work = torchcomm_->all_gather_single(output, input, false);
    work->wait();
  };
  auto reset = [&]() { output.copy_(original_output); };
  auto verify = [&]() { verifyResults(output); };
  run(execute, reset, verify);
}

// Test function for synchronous all_gather_single without work object
template <typename Fixture>
void AllGatherSingleTest<Fixture>::testSyncNoWork(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  at::Tensor input = createInputTensor(count, dtype);
  at::Tensor output = createOutputTensor(count, dtype);
  auto original_output = output.clone();

  auto execute = [&]() { torchcomm_->all_gather_single(output, input, false); };
  auto reset = [&]() { output.copy_(original_output); };
  auto verify = [&]() { verifyResults(output); };
  run(execute, reset, verify);
}

// Test function for asynchronous all_gather_single with wait
template <typename Fixture>
void AllGatherSingleTest<Fixture>::testAsync(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  at::Tensor input = createInputTensor(count, dtype);
  at::Tensor output = createOutputTensor(count, dtype);
  auto original_output = output.clone();

  auto execute = [&]() {
    auto work = torchcomm_->all_gather_single(output, input, true);
    work->wait();
  };
  auto reset = [&]() { output.copy_(original_output); };
  auto verify = [&]() { verifyResults(output); };
  run(execute, reset, verify);
}

// Test function for asynchronous all_gather_single with early reset
template <typename Fixture>
void AllGatherSingleTest<Fixture>::testAsyncEarlyReset(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  at::Tensor input = createInputTensor(count, dtype);
  at::Tensor output = createOutputTensor(count, dtype);
  auto original_output = output.clone();

  auto execute = [&]() {
    auto work = torchcomm_->all_gather_single(output, input, true);
    work->wait();
    work.reset();
  };
  auto reset = [&]() { output.copy_(original_output); };
  auto verify = [&]() { verifyResults(output); };
  run(execute, reset, verify);
}

// Test function for asynchronous all_gather_single with input deleted after
// enqueue
template <typename Fixture>
void AllGatherSingleTest<Fixture>::testInputDeleted(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  auto input = std::make_shared<at::Tensor>(createInputTensor(count, dtype));
  at::Tensor output = createOutputTensor(count, dtype);
  auto original_output = output.clone();

  auto execute = [&]() {
    torchcomm_->all_gather_single(output, *input, false);
  };
  auto reset = [&]() { output.copy_(original_output); };
  auto verify = [&]() { verifyResults(output); };
  auto cleanup = [&]() { input.reset(); };
  run(execute, reset, verify, cleanup);
}

// Helper function to create input tensor
template <typename Fixture>
at::Tensor AllGatherSingleTest<Fixture>::createInputTensor(
    int count,
    at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  at::Tensor input;
  if (dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16) {
    input = at::ones({count}, options) * static_cast<float>(rank_ + 1);
  } else if (dtype == at::kInt) {
    input = at::ones({count}, options) * static_cast<int>(rank_ + 1);
  } else if (dtype == at::kChar) {
    input = at::ones({count}, options) * static_cast<signed char>(rank_ + 1);
  }
  return input;
}

// Helper function to create output tensor
template <typename Fixture>
at::Tensor AllGatherSingleTest<Fixture>::createOutputTensor(
    int count,
    at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  return at::zeros({count * num_ranks_}, options);
}

// Helper function to verify results
template <typename Fixture>
void AllGatherSingleTest<Fixture>::verifyResults(const at::Tensor& output) {
  int64_t count = output.numel() / num_ranks_;

  for (int i = 0; i < num_ranks_; i++) {
    at::Tensor section = output.slice(0, i * count, (i + 1) * count);
    std::string description = "rank " + std::to_string(i) + " section";
    verifyTensorEquality(section.cpu(), i + 1, description);
  }
}

template class AllGatherSingleTest<EagerTestFixture<AllGatherSingleParams>>;
template class AllGatherSingleTest<GraphTestFixture<AllGatherSingleParams, 1>>;
template class AllGatherSingleTest<GraphTestFixture<AllGatherSingleParams, 2>>;
