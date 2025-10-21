// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "ReduceScatterSingleTest.hpp"

#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>
#include "TorchCommTestHelpers.h"

std::unique_ptr<TorchCommTestWrapper> ReduceScatterSingleTest::createWrapper() {
  return std::make_unique<TorchCommTestWrapper>();
}

void ReduceScatterSingleTest::SetUp() {
  wrapper_ = createWrapper();
  torchcomm_ = wrapper_->getTorchComm();
  rank_ = torchcomm_->getRank();
  num_ranks_ = torchcomm_->getSize();
}

void ReduceScatterSingleTest::TearDown() {
  // Explicitly reset the TorchComm object to ensure proper cleanup
  torchcomm_.reset();
  wrapper_.reset();
}

// Test function for synchronous reduce_scatter_single with work object
void ReduceScatterSingleTest::testSyncReduceScatterSingle(
    int count,
    at::ScalarType dtype,
    torch::comms::ReduceOp op) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing sync reduce_scatter_single with count="
                           << count << " and dtype=" << getDtypeName(dtype)
                           << " and op=" << getOpName(op));

  // Create input and output tensors
  at::Tensor input = createInputTensor(count, dtype);
  at::Tensor output = createOutputTensor(count, dtype);

  // Call reduce_scatter_single
  auto work = torchcomm_->reduce_scatter_single(output, input, op, false);
  work->wait();

  // Verify the results
  verifyResults(output, op);
}

// Test function for synchronous reduce_scatter_single without work object
void ReduceScatterSingleTest::testSyncReduceScatterSingleNoWork(
    int count,
    at::ScalarType dtype,
    torch::comms::ReduceOp op) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing sync reduce_scatter_single without work object with count="
      << count << " and dtype=" << getDtypeName(dtype)
      << " and op=" << getOpName(op));

  // Create input and output tensors
  at::Tensor input = createInputTensor(count, dtype);
  at::Tensor output = createOutputTensor(count, dtype);

  // Call reduce_scatter_single without keeping the work object
  torchcomm_->reduce_scatter_single(output, input, op, false);

  // Verify the results
  verifyResults(output, op);
}

// Test function for asynchronous reduce_scatter_single with wait
void ReduceScatterSingleTest::testAsyncReduceScatterSingle(
    int count,
    at::ScalarType dtype,
    torch::comms::ReduceOp op) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing async reduce_scatter_single with count="
                           << count << " and dtype=" << getDtypeName(dtype)
                           << " and op=" << getOpName(op));

  // Create input and output tensors
  at::Tensor input = createInputTensor(count, dtype);
  at::Tensor output = createOutputTensor(count, dtype);

  // Call reduce_scatter_single
  auto work = torchcomm_->reduce_scatter_single(output, input, op, true);

  // Wait for the reduce_scatter_single to complete
  work->wait();

  // Verify the results
  verifyResults(output, op);
}

// Test function for asynchronous reduce_scatter_single with early reset
void ReduceScatterSingleTest::testAsyncReduceScatterSingleEarlyReset(
    int count,
    at::ScalarType dtype,
    torch::comms::ReduceOp op) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing async reduce_scatter_single with early reset with count="
      << count << " and dtype=" << getDtypeName(dtype)
      << " and op=" << getOpName(op));

  // Create input and output tensors
  at::Tensor input = createInputTensor(count, dtype);
  at::Tensor output = createOutputTensor(count, dtype);

  // Call reduce_scatter_single
  auto work = torchcomm_->reduce_scatter_single(output, input, op, true);

  // Wait for the work to complete before resetting
  work->wait();

  // Reset the work object
  work.reset();

  // Verify the results
  verifyResults(output, op);
}

// Test function for asynchronous reduce_scatter_single with input deleted after
// enqueue
void ReduceScatterSingleTest::testReduceScatterSingleInputDeleted(
    int count,
    at::ScalarType dtype,
    torch::comms::ReduceOp op) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing async reduce_scatter_single with input deleted after enqueue with count="
      << count << " and dtype=" << getDtypeName(dtype)
      << " and op=" << getOpName(op));

  // Create output tensor that persists throughout the test
  at::Tensor output = createOutputTensor(count, dtype);

  {
    // Create input tensor in a limited scope
    at::Tensor input = createInputTensor(count, dtype);

    // Call reduce_scatter_single
    torchcomm_->reduce_scatter_single(output, input, op, false);

    // Input tensor goes out of scope here and gets deleted
  }

  // Verify the results
  verifyResults(output, op);
}

// CUDA Graph test function for reduce_scatter_single
void ReduceScatterSingleTest::testGraphReduceScatterSingle(
    int count,
    at::ScalarType dtype,
    torch::comms::ReduceOp op) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing CUDA Graph reduce_scatter_single with count=" << count
      << " and dtype=" << getDtypeName(dtype) << " and op=" << getOpName(op));

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

  // Capture the reset + reduce_scatter_single operations in the graph
  graph.capture_begin();

  // Call reduce_scatter_single without keeping the work object
  torchcomm_->reduce_scatter_single(output, input, op, false);

  graph.capture_end();

  // Replay the captured graph multiple times
  for (int i = 0; i < num_replays; ++i) {
    // Reset output buffer before graph replay
    output.copy_(original_output);

    graph.replay();

    // Verify the results after each replay
    verifyResults(output, op);
  }
}

// CUDA Graph test function for reduce_scatter_single with input deleted after
// graph creation
void ReduceScatterSingleTest::testGraphReduceScatterSingleInputDeleted(
    int count,
    at::ScalarType dtype,
    torch::comms::ReduceOp op) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing CUDA Graph reduce_scatter_single with input deleted after graph creation with count="
      << count << " and dtype=" << getDtypeName(dtype)
      << " and op=" << getOpName(op));

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

    // Capture the reset + reduce_scatter_single operations in the graph
    graph.capture_begin();

    // Call reduce_scatter_single without keeping the work object
    torchcomm_->reduce_scatter_single(output, input, op, false);

    graph.capture_end();

    // Input tensor goes out of scope here and gets deleted
  }

  // Replay the captured graph multiple times even though input is deleted
  for (int i = 0; i < num_replays; ++i) {
    // Reset output buffer before graph replay
    output.copy_(original_output);

    graph.replay();

    // Verify the results after each replay
    verifyResults(output, op);
  }
}

// Helper function to create input tensor
at::Tensor ReduceScatterSingleTest::createInputTensor(
    int count,
    at::ScalarType dtype) {
  // Create tensor directly on GPU with the specified dtype
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  at::Tensor input = at::zeros({count * num_ranks_}, options);

  // Create a tensor of rank values [1, 2, ..., num_ranks_]
  auto ranks = at::arange(1, num_ranks_ + 1, options);

  // For each rank, fill its section with its rank value
  for (int r = 0; r < num_ranks_; r++) {
    // Use slice operation to get the section for this rank
    auto section = input.slice(0, r * count, (r + 1) * count);
    // Fill the entire section with the rank value in one operation
    section.fill_(ranks[r].item());
  }

  return input;
}

// Helper function to create output tensor
at::Tensor ReduceScatterSingleTest::createOutputTensor(
    int count,
    at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  return at::zeros({count}, options);
}

// Helper function to verify results
void ReduceScatterSingleTest::verifyResults(
    const at::Tensor& output,
    torch::comms::ReduceOp op) {
  // Calculate expected value based on operation type
  int expected_value = 0;
  if (op == torch::comms::ReduceOp::SUM) {
    // Sum: num_ranks * (rank+1)
    expected_value = num_ranks_ * (rank_ + 1);
  } else if (op == torch::comms::ReduceOp::MAX) {
    // Max: rank+1
    expected_value = rank_ + 1;
  }

  // Use verifyTensorEquality with integer expected value
  std::string description = "reduce_scatter_single with op " + getOpName(op);
  verifyTensorEquality(output.cpu(), expected_value, description);
}
