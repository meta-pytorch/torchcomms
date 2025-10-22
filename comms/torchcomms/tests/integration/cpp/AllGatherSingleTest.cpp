// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "AllGatherSingleTest.hpp"

#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

std::unique_ptr<TorchCommTestWrapper> AllGatherSingleTest::createWrapper() {
  return std::make_unique<TorchCommTestWrapper>();
}

void AllGatherSingleTest::SetUp() {
  wrapper_ = createWrapper();
  torchcomm_ = wrapper_->getTorchComm();
  rank_ = torchcomm_->getRank();
  num_ranks_ = torchcomm_->getSize();
}

void AllGatherSingleTest::TearDown() {
  // Explicitly reset the TorchComm object to ensure proper cleanup
  torchcomm_.reset();
  wrapper_.reset();
}

// Test function for synchronous all_gather_single with work object
void AllGatherSingleTest::testSyncAllGatherSingle(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing sync all_gather_single with count="
                           << count << " and dtype=" << getDtypeName(dtype));

  // Create input and output tensors
  at::Tensor input = createInputTensor(count, dtype);
  at::Tensor output = createOutputTensor(count, dtype);

  // Call all_gather_single
  auto work = torchcomm_->all_gather_single(output, input, false);
  work->wait();

  // Verify the results
  verifyResults(output);
}

// Test function for synchronous all_gather_single without work object
void AllGatherSingleTest::testSyncAllGatherSingleNoWork(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing sync all_gather_single without work object with count="
      << count << " and dtype=" << getDtypeName(dtype));

  // Create input and output tensors
  at::Tensor input = createInputTensor(count, dtype);
  at::Tensor output = createOutputTensor(count, dtype);

  // Call all_gather_single without keeping the work object
  torchcomm_->all_gather_single(output, input, false);

  // Verify the results
  verifyResults(output);
}

// Test function for asynchronous all_gather_single with wait
void AllGatherSingleTest::testAsyncAllGatherSingle(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing async all_gather_single with count="
                           << count << " and dtype=" << getDtypeName(dtype));

  // Create input and output tensors
  at::Tensor input = createInputTensor(count, dtype);
  at::Tensor output = createOutputTensor(count, dtype);

  // Call all_gather_single
  auto work = torchcomm_->all_gather_single(output, input, true);

  // Wait for the all_gather_single to complete
  work->wait();

  // Verify the results
  verifyResults(output);
}

// Test function for asynchronous all_gather_single with early reset
void AllGatherSingleTest::testAsyncAllGatherSingleEarlyReset(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing async all_gather_single with early reset with count=" << count
      << " and dtype=" << getDtypeName(dtype));

  // Create input and output tensors
  at::Tensor input = createInputTensor(count, dtype);
  at::Tensor output = createOutputTensor(count, dtype);

  // Call all_gather_single
  auto work = torchcomm_->all_gather_single(output, input, true);

  // Wait for the work to complete before resetting
  work->wait();

  // Reset the work object
  work.reset();

  // Verify the results
  verifyResults(output);
}

// Test function for asynchronous all_gather_single with input deleted after
// enqueue
void AllGatherSingleTest::testAllGatherSingleInputDeleted(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing async all_gather_single with input deleted after enqueue with count="
      << count << " and dtype=" << getDtypeName(dtype));

  // Create output tensor
  at::Tensor output = createOutputTensor(count, dtype);

  {
    // Create input tensor in a limited scope
    at::Tensor input = createInputTensor(count, dtype);

    // Call all_gather_single
    torchcomm_->all_gather_single(output, input, false);

    // Input tensor goes out of scope here and gets deleted
  }

  // Verify the results
  verifyResults(output);
}

// CUDA Graph test function for all_gather_single
void AllGatherSingleTest::testGraphAllGatherSingle(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing CUDA Graph all_gather_single with count="
                           << count << " and dtype=" << getDtypeName(dtype));

  // Create input and output tensors BEFORE graph capture
  at::Tensor input = createInputTensor(count, dtype);
  at::Tensor output = createOutputTensor(count, dtype);
  at::Tensor original_output = output.clone();

  // Create a non-default CUDA stream (required for CUDA graph capture)
  at::cuda::CUDAStream stream = at::cuda::getStreamFromPool();

  // Set the stream as current for graph capture
  at::cuda::CUDAStreamGuard guard(stream);

  // Create PyTorch CUDA graph
  at::cuda::CUDAGraph graph;

  // Capture the all_gather_single operation in the graph
  graph.capture_begin();

  torchcomm_->all_gather_single(output, input, false);

  graph.capture_end();

  // Replay the captured graph multiple times
  for (int i = 0; i < num_replays; ++i) {
    // Reset output tensor before each replay
    output.copy_(original_output);

    graph.replay();

    // Verify the results after each replay
    verifyResults(output);
  }
}

// CUDA Graph test function for all_gather_single with input deleted after graph
// creation
void AllGatherSingleTest::testGraphAllGatherSingleInputDeleted(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing CUDA Graph all_gather_single with input deleted after graph creation with count="
      << count << " and dtype=" << getDtypeName(dtype));

  // Create output tensor that persists throughout the test
  at::Tensor output = createOutputTensor(count, dtype);
  at::Tensor original_output = output.clone();

  // Create a non-default CUDA stream (required for CUDA graph capture)
  at::cuda::CUDAStream stream = at::cuda::getStreamFromPool();

  // Set the stream as current for graph capture
  at::cuda::CUDAStreamGuard guard(stream);

  // Create PyTorch CUDA graph
  at::cuda::CUDAGraph graph;

  {
    // Create input tensor in a limited scope
    at::Tensor input = createInputTensor(count, dtype);

    // Capture the all_gather_single operation in the graph
    graph.capture_begin();

    torchcomm_->all_gather_single(output, input, false);

    graph.capture_end();

    // Input tensor goes out of scope here and gets deleted
  }

  // Replay the captured graph multiple times even though input is deleted
  for (int i = 0; i < num_replays; ++i) {
    // Reset output tensor before each replay
    output.copy_(original_output);

    graph.replay();

    // Verify the results after each replay
    verifyResults(output);
  }
}

// Helper function to create input tensor
at::Tensor AllGatherSingleTest::createInputTensor(
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
at::Tensor AllGatherSingleTest::createOutputTensor(
    int count,
    at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  return at::zeros({count * num_ranks_}, options);
}

// Helper function to verify results
void AllGatherSingleTest::verifyResults(const at::Tensor& output) {
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
