// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "ScatterTest.hpp"

#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>
#include <vector>
#include "TorchCommTestHelpers.h"
#include "comms/torchcomms/TorchWork.hpp"

std::unique_ptr<TorchCommTestWrapper> ScatterTest::createWrapper() {
  return std::make_unique<TorchCommTestWrapper>();
}

void ScatterTest::SetUp() {
  wrapper_ = createWrapper();
  torchcomm_ = wrapper_->getTorchComm();
  rank_ = torchcomm_->getRank();
  num_ranks_ = torchcomm_->getSize();
  device_type_ = wrapper_->getDevice().type();
}

void ScatterTest::TearDown() {
  // Explicitly reset the TorchComm object to ensure proper cleanup
  torchcomm_.reset();
  wrapper_.reset();
}

// Test function for synchronous scatter with work object
void ScatterTest::testSyncScatter(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing sync scatter with count=" << count
                           << " and dtype=" << getDtypeName(dtype));

  // Root rank will send data to all ranks
  const int root_rank = 0;

  // Only the root rank needs to create input tensors
  std::vector<at::Tensor> inputs;
  if (rank_ == root_rank) {
    inputs = createInputTensors(count, dtype);
  }

  // Create output tensor to receive data
  at::Tensor output = createOutputTensor(count, dtype);

  // Call scatter
  auto work = torchcomm_->scatter(output, inputs, root_rank, false);
  work->wait();

  // Verify the results
  verifyResults(output);
}

// Test function for synchronous scatter without work object
void ScatterTest::testSyncScatterNoWork(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing sync scatter without work object with count=" << count
      << " and dtype=" << getDtypeName(dtype));

  // Root rank will send data to all ranks
  const int root_rank = 0;

  // Only the root rank needs to create input tensors
  std::vector<at::Tensor> inputs;
  if (rank_ == root_rank) {
    inputs = createInputTensors(count, dtype);
  }

  // Create output tensor to receive data
  at::Tensor output = createOutputTensor(count, dtype);

  // Call scatter without keeping the work object
  torchcomm_->scatter(output, inputs, root_rank, false);

  // Verify the results
  verifyResults(output);
}

// Test function for asynchronous scatter with wait
void ScatterTest::testAsyncScatter(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing async scatter with count=" << count
                           << " and dtype=" << getDtypeName(dtype));

  // Root rank will send data to all ranks
  const int root_rank = 0;

  // Only the root rank needs to create input tensors
  std::vector<at::Tensor> inputs;
  if (rank_ == root_rank) {
    inputs = createInputTensors(count, dtype);
  }

  // Create output tensor to receive data
  at::Tensor output = createOutputTensor(count, dtype);

  // Call scatter
  auto work = torchcomm_->scatter(output, inputs, root_rank, true);

  // Wait for the scatter to complete
  work->wait();

  // Verify the results
  verifyResults(output);
}

// Test function for asynchronous scatter with early reset
void ScatterTest::testAsyncScatterEarlyReset(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing async scatter with early reset with count=" << count
      << " and dtype=" << getDtypeName(dtype));

  // Root rank will send data to all ranks
  const int root_rank = 0;

  // Only the root rank needs to create input tensors
  std::vector<at::Tensor> inputs;
  if (rank_ == root_rank) {
    inputs = createInputTensors(count, dtype);
  }

  // Create output tensor to receive data
  at::Tensor output = createOutputTensor(count, dtype);

  // Call scatter
  auto work = torchcomm_->scatter(output, inputs, root_rank, true);

  // Wait for the work to complete before resetting
  work->wait();

  // Reset the work object
  work.reset();

  // Verify the results
  verifyResults(output);
}

// Test function for asynchronous scatter with input deleted after enqueue
void ScatterTest::testScatterInputDeleted(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing async scatter with input deleted after enqueue with count="
      << count << " and dtype=" << getDtypeName(dtype));

  // Root rank will send data to all ranks
  const int root_rank = 0;

  // Create output tensor to receive data that persists throughout the test
  at::Tensor output = createOutputTensor(count, dtype);

  {
    // Only the root rank needs to create input tensors in a limited scope
    std::vector<at::Tensor> inputs;
    if (rank_ == root_rank) {
      inputs = createInputTensors(count, dtype);
    }

    // Call scatter
    torchcomm_->scatter(output, inputs, root_rank, false);

    // Input tensors go out of scope here and get deleted
  }

  // Verify the results
  verifyResults(output);
}

// CUDA Graph test function for scatter
void ScatterTest::testGraphScatter(int count, at::ScalarType dtype) {
  // Skip CUDA Graph tests when running on CPU
  if (isRunningOnCPU()) {
    GTEST_SKIP() << "CUDA Graph tests are not supported on CPU";
  }

  SCOPED_TRACE(
      ::testing::Message() << "Testing CUDA Graph scatter with count=" << count
                           << " and dtype=" << getDtypeName(dtype));

  // Create a non-default CUDA stream (required for CUDA graph capture)
  at::cuda::CUDAStream stream = at::cuda::getStreamFromPool();

  // Set the stream as current for graph capture
  at::cuda::CUDAStreamGuard guard(stream);

  // Root rank will send data to all ranks
  const int root_rank = 0;

  // Only the root rank needs to create input tensors AFTER setting non-default
  // stream but BEFORE graph capture
  std::vector<at::Tensor> inputs;
  if (rank_ == root_rank) {
    inputs = createInputTensors(count, dtype);
  }

  // Create output tensor to receive data AFTER setting non-default stream but
  // BEFORE graph capture
  at::Tensor output = createOutputTensor(count, dtype);
  at::Tensor original_output = output.clone();

  // Create PyTorch CUDA graph
  at::cuda::CUDAGraph graph;

  // Capture the scatter operation in the graph
  graph.capture_begin();

  // Call scatter without keeping the work object
  torchcomm_->scatter(output, inputs, root_rank, false);

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

// CUDA Graph test function for scatter with input deleted after graph creation
void ScatterTest::testGraphScatterInputDeleted(
    int count,
    at::ScalarType dtype) {
  // Skip CUDA Graph tests when running on CPU
  if (isRunningOnCPU()) {
    GTEST_SKIP() << "CUDA Graph tests are not supported on CPU";
  }

  SCOPED_TRACE(
      ::testing::Message()
      << "Testing CUDA Graph scatter with input deleted after graph creation with count="
      << count << " and dtype=" << getDtypeName(dtype));

  // Create a non-default CUDA stream (required for CUDA graph capture)
  at::cuda::CUDAStream stream = at::cuda::getStreamFromPool();

  // Set the stream as current for graph capture
  at::cuda::CUDAStreamGuard guard(stream);

  // Root rank will send data to all ranks
  const int root_rank = 0;

  // Create output tensor to receive data that persists throughout the test
  at::Tensor output = createOutputTensor(count, dtype);
  at::Tensor original_output = output.clone();

  // Create PyTorch CUDA graph
  at::cuda::CUDAGraph graph;

  {
    // Only the root rank needs to create input tensors in a limited scope
    std::vector<at::Tensor> inputs;
    if (rank_ == root_rank) {
      inputs = createInputTensors(count, dtype);
    }

    // Capture the scatter operation in the graph
    graph.capture_begin();

    // Call scatter without keeping the work object
    torchcomm_->scatter(output, inputs, root_rank, false);

    graph.capture_end();

    // Input tensors go out of scope here and get deleted
  }

  // Replay the captured graph multiple times even though input tensors are
  // deleted
  for (int i = 0; i < num_replays; ++i) {
    // Reset output buffer before each replay
    output.copy_(original_output);

    graph.replay();

    // Verify the results after each replay
    verifyResults(output);
  }
}

// Helper function to create input tensors
std::vector<at::Tensor> ScatterTest::createInputTensors(
    int count,
    at::ScalarType dtype) {
  std::vector<at::Tensor> inputs;
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);

  // Only root rank needs to allocate input tensors
  for (int i = 0; i < num_ranks_; i++) {
    // Each tensor has rank-specific values
    at::Tensor tensor;
    if (dtype == at::kFloat || dtype == at::kBFloat16) {
      tensor = at::ones({count}, options) * static_cast<float>(i + 1);
    } else if (dtype == at::kInt) {
      tensor = at::ones({count}, options) * static_cast<int>(i + 1);
    } else if (dtype == at::kChar) {
      tensor = at::ones({count}, options) * static_cast<signed char>(i + 1);
    }
    inputs.push_back(tensor);
  }

  return inputs;
}

// Helper function to create output tensor
at::Tensor ScatterTest::createOutputTensor(int count, at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  return at::zeros({count}, options);
}

// Helper function to verify results
void ScatterTest::verifyResults(const at::Tensor& output) {
  std::string description = "rank " + std::to_string(rank_) + " tensor";
  verifyTensorEquality(output.cpu(), rank_ + 1, description);
}
