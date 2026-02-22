// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "AllGatherVTest.hpp"

#include <gtest/gtest.h>
#include <memory>
#include "TorchCommTestHelpers.h"

// Test function for synchronous all_gather_v with work object
template <typename Fixture>
void AllGatherVTest<Fixture>::testSync(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  auto counts = getCounts(count);
  at::Tensor input = createInputTensor(counts[rank_], dtype);
  std::vector<at::Tensor> outputs = createOutputTensors(counts, dtype);
  std::vector<at::Tensor> original_outputs;
  original_outputs.reserve(num_ranks_);
  for (const auto& output : outputs) {
    original_outputs.push_back(output.clone());
  }

  auto execute = [&]() {
    auto work = torchcomm_->all_gather_v(outputs, input, false);
    work->wait();
  };
  auto reset = [&]() {
    for (int i = 0; i < num_ranks_; i++) {
      outputs[i].copy_(original_outputs[i]);
    }
  };
  auto verify = [&]() { verifyResults(outputs); };
  run(execute, reset, verify);
}

template <typename Fixture>
void AllGatherVTest<Fixture>::testSyncNoWork(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  auto counts = getCounts(count);
  at::Tensor input = createInputTensor(counts[rank_], dtype);
  std::vector<at::Tensor> outputs = createOutputTensors(counts, dtype);
  std::vector<at::Tensor> original_outputs;
  original_outputs.reserve(num_ranks_);
  for (const auto& output : outputs) {
    original_outputs.push_back(output.clone());
  }

  auto execute = [&]() { torchcomm_->all_gather_v(outputs, input, false); };
  auto reset = [&]() {
    for (int i = 0; i < num_ranks_; i++) {
      outputs[i].copy_(original_outputs[i]);
    }
  };
  auto verify = [&]() { verifyResults(outputs); };
  run(execute, reset, verify);
}

template <typename Fixture>
void AllGatherVTest<Fixture>::testAsync(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  auto counts = getCounts(count);
  at::Tensor input = createInputTensor(counts[rank_], dtype);
  std::vector<at::Tensor> outputs = createOutputTensors(counts, dtype);
  std::vector<at::Tensor> original_outputs;
  original_outputs.reserve(num_ranks_);
  for (const auto& output : outputs) {
    original_outputs.push_back(output.clone());
  }

  auto execute = [&]() {
    auto work = torchcomm_->all_gather_v(outputs, input, true);
    work->wait();
  };
  auto reset = [&]() {
    for (int i = 0; i < num_ranks_; i++) {
      outputs[i].copy_(original_outputs[i]);
    }
  };
  auto verify = [&]() { verifyResults(outputs); };
  run(execute, reset, verify);
}

template <typename Fixture>
void AllGatherVTest<Fixture>::testAsyncEarlyReset(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  auto counts = getCounts(count);
  at::Tensor input = createInputTensor(counts[rank_], dtype);
  std::vector<at::Tensor> outputs = createOutputTensors(counts, dtype);
  std::vector<at::Tensor> original_outputs;
  original_outputs.reserve(num_ranks_);
  for (const auto& output : outputs) {
    original_outputs.push_back(output.clone());
  }

  auto execute = [&]() {
    auto work = torchcomm_->all_gather_v(outputs, input, true);
    work->wait();
    work.reset();
  };
  auto reset = [&]() {
    for (int i = 0; i < num_ranks_; i++) {
      outputs[i].copy_(original_outputs[i]);
    }
  };
  auto verify = [&]() { verifyResults(outputs); };
  run(execute, reset, verify);
}

template <typename Fixture>
void AllGatherVTest<Fixture>::testInputDeleted(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  auto counts = getCounts(count);
  auto input =
      std::make_shared<at::Tensor>(createInputTensor(counts[rank_], dtype));
  std::vector<at::Tensor> outputs = createOutputTensors(counts, dtype);
  std::vector<at::Tensor> original_outputs;
  original_outputs.reserve(num_ranks_);
  for (const auto& output : outputs) {
    original_outputs.push_back(output.clone());
  }

  auto execute = [&]() { torchcomm_->all_gather_v(outputs, *input, false); };
  auto reset = [&]() {
    for (int i = 0; i < num_ranks_; i++) {
      outputs[i].copy_(original_outputs[i]);
    }
  };
  auto verify = [&]() { verifyResults(outputs); };
  auto cleanup = [&]() { input.reset(); };
  run(execute, reset, verify, cleanup);
}

template <typename Fixture>
std::vector<int> AllGatherVTest<Fixture>::getCounts(int count) {
  std::vector<int> counts(num_ranks_);
  for (int i = 0; i < num_ranks_; i++) {
    counts[i] = count + i;
  }
  return counts;
}

// Helper function to create input tensor
template <typename Fixture>
at::Tensor AllGatherVTest<Fixture>::createInputTensor(
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

// Helper function to create output tensors
template <typename Fixture>
std::vector<at::Tensor> AllGatherVTest<Fixture>::createOutputTensors(
    const std::vector<int>& counts,
    at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  std::vector<at::Tensor> outputs(num_ranks_);
  for (int i = 0; i < num_ranks_; i++) {
    outputs[i] = at::zeros({counts[i]}, options);
  }
  return outputs;
}

// Helper function to verify results
template <typename Fixture>
void AllGatherVTest<Fixture>::verifyResults(
    const std::vector<at::Tensor>& outputs) {
  for (int i = 0; i < num_ranks_; i++) {
    std::string description = "rank " + std::to_string(i) + " tensor";
    verifyTensorEquality(outputs[i].cpu(), i + 1, description);
  }
}

template class AllGatherVTest<EagerTestFixture<AllGatherVParams>>;
template class AllGatherVTest<GraphTestFixture<AllGatherVParams, 1>>;
template class AllGatherVTest<GraphTestFixture<AllGatherVParams, 2>>;
