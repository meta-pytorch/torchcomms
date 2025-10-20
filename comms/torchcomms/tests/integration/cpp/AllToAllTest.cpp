// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "AllToAllTest.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>
#include <vector>
#include "TorchCommTestHelpers.h"

std::unique_ptr<TorchCommTestWrapper> AllToAllTest::createWrapper() {
  return std::make_unique<TorchCommTestWrapper>();
}

void AllToAllTest::SetUp() {
  wrapper_ = createWrapper();
  torchcomm_ = wrapper_->getTorchComm();
  rank_ = torchcomm_->getRank();
  num_ranks_ = torchcomm_->getSize();
}

void AllToAllTest::TearDown() {
  // Explicitly reset the TorchComm object to ensure proper cleanup
  torchcomm_.reset();
  wrapper_.reset();
}

// Test function for synchronous all_to_all with work object
void AllToAllTest::testSyncAllToAll(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing sync all_to_all with count=" << count
                           << " and dtype=" << getDtypeName(dtype));

  auto input_tensors = createInputTensors(count, dtype);
  auto output_tensors = createOutputTensors(count, dtype);
  auto expected_output = createExpectedOutput();

  // Test synchronous all_to_all
  auto work = torchcomm_->all_to_all(output_tensors, input_tensors, false);
  work->wait();

  verifyResults(output_tensors, expected_output);
}

// Test function for synchronous all_to_all without work object
void AllToAllTest::testSyncAllToAllNoWork(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing sync all_to_all without work object with count=" << count
      << " and dtype=" << getDtypeName(dtype));

  auto input_tensors = createInputTensors(count, dtype);
  auto output_tensors = createOutputTensors(count, dtype);
  auto expected_output = createExpectedOutput();

  // Test synchronous all_to_all without keeping the work object
  torchcomm_->all_to_all(output_tensors, input_tensors, false);

  verifyResults(output_tensors, expected_output);
}

// Test function for asynchronous all_to_all with wait
void AllToAllTest::testAsyncAllToAll(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing async all_to_all with count=" << count
                           << " and dtype=" << getDtypeName(dtype));

  auto input_tensors = createInputTensors(count, dtype);
  auto output_tensors = createOutputTensors(count, dtype);
  auto expected_output = createExpectedOutput();

  // Test asynchronous all_to_all
  auto work = torchcomm_->all_to_all(output_tensors, input_tensors, true);

  // Wait for the all_to_all to complete
  work->wait();

  verifyResults(output_tensors, expected_output);
}

// Test function for asynchronous all_to_all with early reset
void AllToAllTest::testAsyncAllToAllEarlyReset(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing async all_to_all with early reset with count=" << count
      << " and dtype=" << getDtypeName(dtype));

  auto input_tensors = createInputTensors(count, dtype);
  auto output_tensors = createOutputTensors(count, dtype);
  auto expected_output = createExpectedOutput();

  // Test asynchronous all_to_all
  auto work = torchcomm_->all_to_all(output_tensors, input_tensors, true);

  // Wait for the work to complete before resetting
  work->wait();

  // Reset the work object
  work.reset();

  verifyResults(output_tensors, expected_output);
}

// Test function for asynchronous all_to_all with input deleted after enqueue
void AllToAllTest::testAllToAllInputDeleted(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing async all_to_all with input deleted after enqueue with count="
      << count << " and dtype=" << getDtypeName(dtype));

  // Create output tensors that persist throughout the test
  auto output_tensors = createOutputTensors(count, dtype);
  auto expected_output = createExpectedOutput();

  {
    // Create input tensors in a limited scope
    auto input_tensors = createInputTensors(count, dtype);

    // Call all_to_all
    torchcomm_->all_to_all(output_tensors, input_tensors, false);

    // Input tensors go out of scope here and get deleted
  }

  // Verify the results
  verifyResults(output_tensors, expected_output);
}

// CUDA Graph test function for all_to_all
void AllToAllTest::testGraphAllToAll(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing CUDA Graph all_to_all with count="
                           << count << " and dtype=" << getDtypeName(dtype));

  // Create input and output tensors BEFORE graph capture
  auto input_tensors = createInputTensors(count, dtype);
  auto output_tensors = createOutputTensors(count, dtype);
  std::vector<at::Tensor> original_output_tensors;
  original_output_tensors.reserve(output_tensors.size());
  for (const auto& output_tensor : output_tensors) {
    original_output_tensors.push_back(output_tensor.clone());
  }
  auto expected_output = createExpectedOutput();

  // Create a non-default CUDA stream (required for CUDA graph capture)
  at::cuda::CUDAStream stream = at::cuda::getStreamFromPool();

  // Set the stream as current for graph capture
  at::cuda::CUDAStreamGuard guard(stream);

  // Create PyTorch CUDA graph
  at::cuda::CUDAGraph graph;

  // Capture the all_to_all operation in the graph
  graph.capture_begin();

  // Call all_to_all without keeping the work object
  torchcomm_->all_to_all(output_tensors, input_tensors, false);

  graph.capture_end();

  // Replay the captured graph multiple times
  for (int i = 0; i < num_replays; ++i) {
    // Reset output tensors before each replay
    for (size_t j = 0; j < output_tensors.size(); ++j) {
      output_tensors[j].copy_(original_output_tensors[j]);
    }

    graph.replay();

    // Verify the results after each replay
    verifyResults(output_tensors, expected_output);
  }
}

// CUDA Graph test function for all_to_all with input deleted after graph
// creation
void AllToAllTest::testGraphAllToAllInputDeleted(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing CUDA Graph all_to_all with input deleted after graph creation with count="
      << count << " and dtype=" << getDtypeName(dtype));

  // Create output tensors that persist throughout the test
  auto output_tensors = createOutputTensors(count, dtype);
  std::vector<at::Tensor> original_output_tensors;
  original_output_tensors.reserve(output_tensors.size());
  for (const auto& output_tensor : output_tensors) {
    original_output_tensors.push_back(output_tensor.clone());
  }
  auto expected_output = createExpectedOutput();

  // Create a non-default CUDA stream (required for CUDA graph capture)
  at::cuda::CUDAStream stream = at::cuda::getStreamFromPool();

  // Set the stream as current for graph capture
  at::cuda::CUDAStreamGuard guard(stream);

  // Create PyTorch CUDA graph
  at::cuda::CUDAGraph graph;

  {
    // Create input tensors in a limited scope
    auto input_tensors = createInputTensors(count, dtype);

    // Capture the all_to_all operation in the graph
    graph.capture_begin();

    // Call all_to_all without keeping the work object
    torchcomm_->all_to_all(output_tensors, input_tensors, false);

    graph.capture_end();

    // Input tensors go out of scope here and get deleted
  }

  // Replay the captured graph multiple times even though input tensors are
  // deleted
  for (int i = 0; i < num_replays; ++i) {
    // Reset output tensors before each replay
    for (size_t j = 0; j < output_tensors.size(); ++j) {
      output_tensors[j].copy_(original_output_tensors[j]);
    }

    graph.replay();

    // Verify the results after each replay
    verifyResults(output_tensors, expected_output);
  }
}

// Helper function to create input tensors
std::vector<at::Tensor> AllToAllTest::createInputTensors(
    int count,
    at::ScalarType dtype) {
  std::vector<at::Tensor> input_tensors;
  input_tensors.reserve(num_ranks_);
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);

  for (int r = 0; r < num_ranks_; r++) {
    // Each tensor has rank-specific values
    at::Tensor tensor;
    if (dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16) {
      tensor = at::ones({count}, options) * static_cast<float>(rank_ + 1);
    } else if (dtype == at::kInt) {
      tensor = at::ones({count}, options) * static_cast<int>(rank_ + 1);
    } else if (dtype == at::kChar) {
      tensor = at::ones({count}, options) * static_cast<signed char>(rank_ + 1);
    }
    input_tensors.push_back(tensor);
  }
  return input_tensors;
}

// Helper function to create output tensors
std::vector<at::Tensor> AllToAllTest::createOutputTensors(
    int count,
    at::ScalarType dtype) {
  std::vector<at::Tensor> output_tensors;
  output_tensors.reserve(num_ranks_);
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);

  for (int r = 0; r < num_ranks_; r++) {
    auto tensor = at::zeros({count}, options);
    output_tensors.push_back(tensor);
  }
  return output_tensors;
}

// Helper function to create expected output tensors
std::vector<int> AllToAllTest::createExpectedOutput() {
  std::vector<int> expected_output;
  expected_output.reserve(num_ranks_);

  for (int r = 0; r < num_ranks_; r++) {
    expected_output.push_back(r + 1);
  }
  return expected_output;
}

// Helper function to verify results
void AllToAllTest::verifyResults(
    const std::vector<at::Tensor>& output_tensors,
    const std::vector<int>& expected_output) {
  for (int r = 0; r < num_ranks_; r++) {
    verifyTensorEquality(output_tensors[r].cpu(), expected_output[r]);
  }
}
