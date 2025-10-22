// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "ReduceTest.hpp"

#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>
#include "TorchCommTestHelpers.h"
#include "comms/torchcomms/TorchWork.hpp"

std::unique_ptr<TorchCommTestWrapper> ReduceTest::createWrapper() {
  return std::make_unique<TorchCommTestWrapper>();
}

void ReduceTest::synchronizeStream() {
  at::cuda::getCurrentCUDAStream(device_index_).synchronize();
}

void ReduceTest::SetUp() {
  wrapper_ = createWrapper();
  torchcomm_ = wrapper_->getTorchComm();
  rank_ = torchcomm_->getRank();
  num_ranks_ = torchcomm_->getSize();
}

void ReduceTest::TearDown() {
  // Explicitly reset the TorchComm object to ensure proper cleanup
  torchcomm_.reset();
  wrapper_.reset();
}

// Test function for synchronous reduce with work object
void ReduceTest::testSyncReduce(
    int count,
    at::ScalarType dtype,
    torch::comms::ReduceOp op) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing sync reduce with count=" << count
                           << " and dtype=" << getDtypeName(dtype)
                           << " and op=" << getOpName(op));

  const int root_rank = 0;

  // Everyone creates the input tensor
  at::Tensor tensor = createInputTensor(count, dtype);

  // Call reduce
  auto work = torchcomm_->reduce(tensor, root_rank, op, false);
  work->wait();

  // Verify the results
  verifyResults(tensor, op, root_rank);
}

// Test function for synchronous reduce without work object
void ReduceTest::testSyncReduceNoWork(
    int count,
    at::ScalarType dtype,
    torch::comms::ReduceOp op) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing sync reduce without work object with count=" << count
      << " and dtype=" << getDtypeName(dtype) << " and op=" << getOpName(op));

  const int root_rank = 0;

  // Everyone creates the input tensor
  at::Tensor tensor = createInputTensor(count, dtype);

  // Call reduce without keeping the work object
  torchcomm_->reduce(tensor, root_rank, op, false);

  // Verify the results
  verifyResults(tensor, op, root_rank);
}

// Test function for asynchronous reduce with wait
void ReduceTest::testAsyncReduce(
    int count,
    at::ScalarType dtype,
    torch::comms::ReduceOp op) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing async reduce with count=" << count
                           << " and dtype=" << getDtypeName(dtype)
                           << " and op=" << getOpName(op));

  const int root_rank = 0;

  // Everyone creates the input tensor
  at::Tensor tensor = createInputTensor(count, dtype);

  // Call reduce
  auto work = torchcomm_->reduce(tensor, root_rank, op, true);

  // Wait for the reduce to complete
  work->wait();

  // Verify the results
  verifyResults(tensor, op, root_rank);
}

// Test function for asynchronous reduce with early reset
void ReduceTest::testAsyncReduceEarlyReset(
    int count,
    at::ScalarType dtype,
    torch::comms::ReduceOp op) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing async reduce with early reset with count=" << count
      << " and dtype=" << getDtypeName(dtype) << " and op=" << getOpName(op));

  const int root_rank = 0;

  // Everyone creates the input tensor
  at::Tensor tensor = createInputTensor(count, dtype);

  // Call reduce
  auto work = torchcomm_->reduce(tensor, root_rank, op, true);

  // Wait for the work to complete before resetting
  work->wait();

  // Reset the work object
  work.reset();

  // Verify the results
  verifyResults(tensor, op, root_rank);
}

// Test function for asynchronous reduce with input deleted after enqueue
void ReduceTest::testReduceInputDeleted(
    int count,
    at::ScalarType dtype,
    torch::comms::ReduceOp op) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing async reduce with input deleted after enqueue with count="
      << count << " and dtype=" << getDtypeName(dtype)
      << " and op=" << getOpName(op));

  const int root_rank = 0;

  // Create work object to hold the async operation
  std::shared_ptr<torch::comms::TorchWork> work;

  {
    // Create input tensor in a limited scope
    at::Tensor tensor = createInputTensor(count, dtype);

    // Call reduce
    work = torchcomm_->reduce(tensor, root_rank, op, true);

    // Tensor goes out of scope here and gets deleted
  }

  // Wait for the reduce to complete even though tensor is deleted
  work->wait();

  // Note: Cannot verify results since tensor is deleted
  // This test validates that the operation completes without crashing
}

// Helper function to create input tensor
at::Tensor ReduceTest::createInputTensor(int count, at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  at::Tensor input;
  if (dtype == at::kFloat || dtype == at::kBFloat16 || dtype == at::kHalf) {
    input = at::ones({count}, options) * static_cast<float>(rank_ + 1);
  } else if (dtype == at::kInt) {
    input = at::ones({count}, options) * static_cast<int>(rank_ + 1);
  } else if (dtype == at::kChar) {
    input = at::ones({count}, options) * static_cast<signed char>(rank_ + 1);
  }
  return input;
}

// Helper function to calculate expected result
int ReduceTest::calculateExpectedResult(torch::comms::ReduceOp op) {
  if (op == torch::comms::ReduceOp::SUM) {
    return num_ranks_ * (num_ranks_ + 1) / 2;
  } else if (op == torch::comms::ReduceOp::MAX) {
    return num_ranks_;
  } else {
    throw std::runtime_error("Unsupported reduce operation");
  }
}

// Helper function to verify results
void ReduceTest::verifyResults(
    const at::Tensor& output,
    torch::comms::ReduceOp op,
    int root_rank) {
  if (rank_ != root_rank) {
    synchronizeStream();
    return;
  }

  // Calculate expected result
  int expected = calculateExpectedResult(op);

  // Use verifyTensorEquality to compare output with expected tensor
  std::string description = "reduce with op " + getOpName(op);
  verifyTensorEquality(output.cpu(), expected, description);
}

// CUDA Graph test function for reduce
void ReduceTest::testGraphReduce(
    int count,
    at::ScalarType dtype,
    torch::comms::ReduceOp op) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing CUDA Graph reduce with count=" << count
                           << " and dtype=" << getDtypeName(dtype)
                           << " and op=" << getOpName(op));

  const int root_rank = 0;

  // Everyone creates the input tensor BEFORE graph capture
  at::Tensor tensor = createInputTensor(count, dtype);
  at::Tensor original_values = tensor.clone();

  // Create a non-default CUDA stream (required for CUDA graph capture)
  at::cuda::CUDAStream stream = at::cuda::getStreamFromPool();

  // Set the stream as current for graph capture
  at::cuda::CUDAStreamGuard guard(stream);

  // Create PyTorch CUDA graph
  at::cuda::CUDAGraph graph;

  // Capture the reset + reduce operations in the graph
  graph.capture_begin();

  // Call reduce without keeping the work object
  torchcomm_->reduce(tensor, root_rank, op, false);

  graph.capture_end();

  // Replay the captured graph multiple times
  for (int i = 0; i < num_replays; ++i) {
    // Reset tensor before graph replay
    tensor.copy_(original_values);

    graph.replay();

    // Verify the results after each replay
    verifyResults(tensor, op, root_rank);
  }
}

// CUDA Graph test function for reduce with input deleted after graph creation
void ReduceTest::testGraphReduceInputDeleted(
    int count,
    at::ScalarType dtype,
    torch::comms::ReduceOp op) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing CUDA Graph reduce with input deleted after graph creation with count="
      << count << " and dtype=" << getDtypeName(dtype)
      << " and op=" << getOpName(op));

  const int root_rank = 0;

  // Create a non-default CUDA stream (required for CUDA graph capture)
  at::cuda::CUDAStream stream = at::cuda::getStreamFromPool();

  // Set the stream as current for graph capture
  at::cuda::CUDAStreamGuard guard(stream);

  // Create PyTorch CUDA graph
  at::cuda::CUDAGraph graph;

  {
    // Create input tensor in a limited scope
    at::Tensor tensor = createInputTensor(count, dtype);

    // Store original values to reset with
    at::Tensor original_values = tensor.clone();

    // Capture the reset + reduce operations in the graph
    graph.capture_begin();

    // Call reduce without keeping the work object
    torchcomm_->reduce(tensor, root_rank, op, false);

    graph.capture_end();

    // Tensor goes out of scope here and gets deleted
  }

  // Replay the captured graph multiple times even though tensor is deleted
  for (int i = 0; i < num_replays; ++i) {
    graph.replay();

    // Note: Cannot verify results since tensor is deleted
    // This test validates that the graph replay completes without crashing
  }
}
