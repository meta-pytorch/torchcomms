// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "AllToAllTest.hpp"

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include "TorchCommTestHelpers.h"

// Test function for synchronous all_to_all with work object
template <typename Fixture>
void AllToAllTest<Fixture>::testSync(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  std::vector<at::Tensor> input_tensors = createInputTensors(count, dtype);
  std::vector<at::Tensor> output_tensors = createOutputTensors(count, dtype);

  std::vector<at::Tensor> original_output_tensors;
  original_output_tensors.reserve(output_tensors.size());
  for (const auto& t : output_tensors) {
    original_output_tensors.push_back(t.clone());
  }

  auto execute = [&]() {
    auto work = torchcomm_->all_to_all(output_tensors, input_tensors, false);
    work->wait();
  };
  auto reset = [&]() {
    for (size_t i = 0; i < output_tensors.size(); ++i) {
      output_tensors[i].copy_(original_output_tensors[i]);
    }
  };
  auto verify = [&]() { verifyResults(output_tensors); };
  run(execute, reset, verify);
}

// Test function for synchronous all_to_all without work object
template <typename Fixture>
void AllToAllTest<Fixture>::testSyncNoWork(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  std::vector<at::Tensor> input_tensors = createInputTensors(count, dtype);
  std::vector<at::Tensor> output_tensors = createOutputTensors(count, dtype);

  std::vector<at::Tensor> original_output_tensors;
  original_output_tensors.reserve(output_tensors.size());
  for (const auto& t : output_tensors) {
    original_output_tensors.push_back(t.clone());
  }

  auto execute = [&]() {
    torchcomm_->all_to_all(output_tensors, input_tensors, false);
  };
  auto reset = [&]() {
    for (size_t i = 0; i < output_tensors.size(); ++i) {
      output_tensors[i].copy_(original_output_tensors[i]);
    }
  };
  auto verify = [&]() { verifyResults(output_tensors); };
  run(execute, reset, verify);
}

// Test function for asynchronous all_to_all with wait
template <typename Fixture>
void AllToAllTest<Fixture>::testAsync(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  std::vector<at::Tensor> input_tensors = createInputTensors(count, dtype);
  std::vector<at::Tensor> output_tensors = createOutputTensors(count, dtype);

  std::vector<at::Tensor> original_output_tensors;
  original_output_tensors.reserve(output_tensors.size());
  for (const auto& t : output_tensors) {
    original_output_tensors.push_back(t.clone());
  }

  auto execute = [&]() {
    auto work = torchcomm_->all_to_all(output_tensors, input_tensors, true);
    work->wait();
  };
  auto reset = [&]() {
    for (size_t i = 0; i < output_tensors.size(); ++i) {
      output_tensors[i].copy_(original_output_tensors[i]);
    }
  };
  auto verify = [&]() { verifyResults(output_tensors); };
  run(execute, reset, verify);
}

// Test function for asynchronous all_to_all with early reset
template <typename Fixture>
void AllToAllTest<Fixture>::testAsyncEarlyReset(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  std::vector<at::Tensor> input_tensors = createInputTensors(count, dtype);
  std::vector<at::Tensor> output_tensors = createOutputTensors(count, dtype);

  std::vector<at::Tensor> original_output_tensors;
  original_output_tensors.reserve(output_tensors.size());
  for (const auto& t : output_tensors) {
    original_output_tensors.push_back(t.clone());
  }

  auto execute = [&]() {
    auto work = torchcomm_->all_to_all(output_tensors, input_tensors, true);
    work->wait();
    work.reset();
  };
  auto reset = [&]() {
    for (size_t i = 0; i < output_tensors.size(); ++i) {
      output_tensors[i].copy_(original_output_tensors[i]);
    }
  };
  auto verify = [&]() { verifyResults(output_tensors); };
  run(execute, reset, verify);
}

// Test function for asynchronous all_to_all with input deleted after enqueue
template <typename Fixture>
void AllToAllTest<Fixture>::testInputDeleted(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  auto input_tensors = std::make_shared<std::vector<at::Tensor>>(
      createInputTensors(count, dtype));
  std::vector<at::Tensor> output_tensors = createOutputTensors(count, dtype);

  std::vector<at::Tensor> original_output_tensors;
  original_output_tensors.reserve(output_tensors.size());
  for (const auto& t : output_tensors) {
    original_output_tensors.push_back(t.clone());
  }

  auto execute = [&]() {
    torchcomm_->all_to_all(output_tensors, *input_tensors, false);
  };
  auto reset = [&]() {
    for (size_t i = 0; i < output_tensors.size(); ++i) {
      output_tensors[i].copy_(original_output_tensors[i]);
    }
  };
  auto verify = [&]() { verifyResults(output_tensors); };
  auto cleanup = [&]() { input_tensors.reset(); };
  run(execute, reset, verify, cleanup);
}

// Helper function to create input tensors
template <typename Fixture>
std::vector<at::Tensor> AllToAllTest<Fixture>::createInputTensors(
    int count,
    at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  std::vector<at::Tensor> tensors;
  tensors.reserve(num_ranks_);
  for (int r = 0; r < num_ranks_; ++r) {
    if (dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16) {
      tensors.push_back(
          at::ones({count}, options) * static_cast<float>(rank_ + 1));
    } else if (dtype == at::kInt) {
      tensors.push_back(
          at::ones({count}, options) * static_cast<int>(rank_ + 1));
    } else if (dtype == at::kChar) {
      tensors.push_back(
          at::ones({count}, options) * static_cast<signed char>(rank_ + 1));
    }
  }
  return tensors;
}

// Helper function to create output tensors
template <typename Fixture>
std::vector<at::Tensor> AllToAllTest<Fixture>::createOutputTensors(
    int count,
    at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  std::vector<at::Tensor> tensors;
  tensors.reserve(num_ranks_);
  for (int r = 0; r < num_ranks_; ++r) {
    tensors.push_back(at::zeros({count}, options));
  }
  return tensors;
}

// Helper function to verify results
template <typename Fixture>
void AllToAllTest<Fixture>::verifyResults(
    const std::vector<at::Tensor>& output_tensors) {
  for (int r = 0; r < num_ranks_; ++r) {
    verifyTensorEquality(output_tensors[r].cpu(), r + 1);
  }
}

template class AllToAllTest<EagerTestFixture<AllToAllParams>>;
template class AllToAllTest<GraphTestFixture<AllToAllParams, 1>>;
template class AllToAllTest<GraphTestFixture<AllToAllParams, 2>>;
