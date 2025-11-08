// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "AllToAllSingleTest.hpp"

#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>
#include "TorchCommTestHelpers.h"

std::unique_ptr<TorchCommTestWrapper> AllToAllSingleTest::createWrapper() {
  return std::make_unique<TorchCommTestWrapper>();
}

void AllToAllSingleTest::SetUp() {
  wrapper_ = createWrapper();
  torchcomm_ = wrapper_->getTorchComm();
  rank_ = torchcomm_->getRank();
  num_ranks_ = torchcomm_->getSize();
}

void AllToAllSingleTest::TearDown() {
  // Explicitly reset the TorchComm object to ensure proper cleanup
  torchcomm_.reset();
  wrapper_.reset();
}

// Test function for synchronous all_to_all_single with work object
void AllToAllSingleTest::testSyncAllToAllSingle(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing sync all_to_all_single with count="
                           << count << " and dtype=" << getDtypeName(dtype));

  // Create input and output tensors
  at::Tensor input = createInputTensor(count, dtype);
  at::Tensor output = createOutputTensor(count, dtype);

  // Call all_to_all_single
  auto work = torchcomm_->all_to_all_single(output, input, false);
  work->wait();

  // Verify the results
  verifyResults(output);
}

// Test function for synchronous all_to_all_single without work object
void AllToAllSingleTest::testSyncAllToAllSingleNoWork(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing sync all_to_all_single without work object with count="
      << count << " and dtype=" << getDtypeName(dtype));

  // Create input and output tensors
  at::Tensor input = createInputTensor(count, dtype);
  at::Tensor output = createOutputTensor(count, dtype);

  // Call all_to_all_single without keeping the work object
  torchcomm_->all_to_all_single(output, input, false);

  // Verify the results
  verifyResults(output);
}

// Test function for asynchronous all_to_all_single with wait
void AllToAllSingleTest::testAsyncAllToAllSingle(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing async all_to_all_single with count="
                           << count << " and dtype=" << getDtypeName(dtype));

  // Create input and output tensors
  at::Tensor input = createInputTensor(count, dtype);
  at::Tensor output = createOutputTensor(count, dtype);

  // Call all_to_all_single
  auto work = torchcomm_->all_to_all_single(output, input, true);

  // Wait for the all_to_all_single to complete
  work->wait();

  // Verify the results
  verifyResults(output);
}

// Test function for asynchronous all_to_all_single with early reset
void AllToAllSingleTest::testAsyncAllToAllSingleEarlyReset(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing async all_to_all_single with early reset with count=" << count
      << " and dtype=" << getDtypeName(dtype));

  // Create input and output tensors
  at::Tensor input = createInputTensor(count, dtype);
  at::Tensor output = createOutputTensor(count, dtype);

  // Call all_to_all_single
  auto work = torchcomm_->all_to_all_single(output, input, true);

  // Wait for the work to complete before resetting
  work->wait();

  // Reset the work object
  work.reset();

  // Verify the results
  verifyResults(output);
}

// Test function for asynchronous all_to_all_single with input deleted after
// enqueue
void AllToAllSingleTest::testAllToAllSingleInputDeleted(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing async all_to_all_single with input deleted after enqueue with count="
      << count << " and dtype=" << getDtypeName(dtype));

  // Create output tensor that persists throughout the test
  at::Tensor output = createOutputTensor(count, dtype);

  {
    // Create input tensor in a limited scope
    at::Tensor input = createInputTensor(count, dtype);

    // Call all_to_all_single
    torchcomm_->all_to_all_single(output, input, false);

    // Input tensor goes out of scope here and gets deleted
  }

  // Verify the results
  verifyResults(output);
}

// CUDA Graph test function for all_to_all_single
void AllToAllSingleTest::testGraphAllToAllSingle(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing CUDA Graph all_to_all_single with count="
                           << count << " and dtype=" << getDtypeName(dtype));

  // Create a non-default CUDA stream (required for CUDA graph capture)
  at::cuda::CUDAStream stream = at::cuda::getStreamFromPool();

  // Set the stream as current for graph capture
  at::cuda::CUDAStreamGuard guard(stream);

  // Create input and output tensors AFTER setting non-default stream but BEFORE
  // graph capture
  at::Tensor input = createInputTensor(count, dtype);
  at::Tensor output = createOutputTensor(count, dtype);
  at::Tensor original_output = output.clone();

  // Create PyTorch CUDA graph
  at::cuda::CUDAGraph graph;

  // Capture the all_to_all_single operation in the graph
  graph.capture_begin();

  // Call all_to_all_single without keeping the work object
  torchcomm_->all_to_all_single(output, input, false);

  graph.capture_end();

  // Replay the captured graph multiple times
  for (int i = 0; i < num_replays; ++i) {
    // Reset output buffer before each replay
    output.copy_(original_output);

    graph.replay();

    // Verify the results after each replay
    verifyResults(output);
  }
}

// CUDA Graph test function for all_to_all_single with input deleted after graph
// creation
void AllToAllSingleTest::testGraphAllToAllSingleInputDeleted(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing CUDA Graph all_to_all_single with input deleted after graph creation with count="
      << count << " and dtype=" << getDtypeName(dtype));

  // Create a non-default CUDA stream (required for CUDA graph capture)
  at::cuda::CUDAStream stream = at::cuda::getStreamFromPool();

  // Set the stream as current for graph capture
  at::cuda::CUDAStreamGuard guard(stream);

  // Create output tensor that persists throughout the test
  at::Tensor output = createOutputTensor(count, dtype);
  at::Tensor original_output = output.clone();

  // Create PyTorch CUDA graph
  at::cuda::CUDAGraph graph;

  {
    // Create input tensor in a limited scope
    at::Tensor input = createInputTensor(count, dtype);

    // Capture the all_to_all_single operation in the graph
    graph.capture_begin();

    // Call all_to_all_single without keeping the work object
    torchcomm_->all_to_all_single(output, input, false);

    graph.capture_end();

    // Input tensor goes out of scope here and gets deleted
  }

  // Replay the captured graph multiple times even though input is deleted
  for (int i = 0; i < num_replays; ++i) {
    // Reset output buffer before each replay
    output.copy_(original_output);

    graph.replay();

    // Verify the results after each replay
    verifyResults(output);
  }
}

// Helper function to create input tensor
at::Tensor AllToAllSingleTest::createInputTensor(
    int count,
    at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  at::Tensor input;
  if (dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16) {
    input =
        at::ones({count * num_ranks_}, options) * static_cast<float>(rank_ + 1);
  } else if (dtype == at::kInt) {
    input =
        at::ones({count * num_ranks_}, options) * static_cast<int>(rank_ + 1);
  } else if (dtype == at::kChar) {
    input = at::ones({count * num_ranks_}, options) *
        static_cast<signed char>(rank_ + 1);
  }
  return input;
}

// Helper function to create output tensor
at::Tensor AllToAllSingleTest::createOutputTensor(
    int count,
    at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  return at::zeros({count * num_ranks_}, options);
}

// Helper function to verify results
void AllToAllSingleTest::verifyResults(const at::Tensor& output) {
  // Extract count from the tensor
  int64_t count = output.numel() / num_ranks_;

  for (int i = 0; i < num_ranks_; i++) {
    // For each rank's section in the output tensor
    at::Tensor section = output.slice(0, i * count, (i + 1) * count);

    // Use verifyTensorEquality to compare section with expected tensor
    std::string description = "rank " + std::to_string(i) + " section";
    verifyTensorEquality(section.cpu(), i + 1, description);
  }
}
