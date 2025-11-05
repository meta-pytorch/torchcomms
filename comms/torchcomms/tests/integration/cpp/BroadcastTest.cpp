// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "BroadcastTest.hpp"

#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>
#include "comms/torchcomms/TorchWork.hpp"
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

std::unique_ptr<TorchCommTestWrapper> BroadcastTest::createWrapper() {
  return std::make_unique<TorchCommTestWrapper>();
}

void BroadcastTest::SetUp() {
  wrapper_ = createWrapper();
  torchcomm_ = wrapper_->getTorchComm();
  rank_ = torchcomm_->getRank();
  num_ranks_ = torchcomm_->getSize();
}

void BroadcastTest::TearDown() {
  // Explicitly reset the TorchComm object to ensure proper cleanup
  torchcomm_.reset();
  wrapper_.reset();
}

// Test function for synchronous broadcast with work object
void BroadcastTest::testSyncBroadcast(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing sync broadcast with count=" << count
                           << " and dtype=" << getDtypeName(dtype));

  const int root_rank = 0;
  const int root_value = 99;

  // Create tensor with different values based on rank
  at::Tensor tensor =
      createBroadcastTensor(root_rank, root_value, count, dtype);

  // Call broadcast
  auto work = torchcomm_->broadcast(tensor, root_rank, false);
  work->wait();

  // Verify the results
  verifyBroadcastResults(tensor, root_value);
}

// Test function for synchronous broadcast without work object
void BroadcastTest::testSyncBroadcastNoWork(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing sync broadcast without work object with count=" << count
      << " and dtype=" << getDtypeName(dtype));

  const int root_rank = 0;
  const int root_value = 99;

  // Create tensor with different values based on rank
  at::Tensor tensor =
      createBroadcastTensor(root_rank, root_value, count, dtype);

  // Call broadcast without keeping the work object
  torchcomm_->broadcast(tensor, root_rank, false);

  // Verify the results
  verifyBroadcastResults(tensor, root_value);
}

// Test function for asynchronous broadcast with wait
void BroadcastTest::testAsyncBroadcast(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing async broadcast with count=" << count
                           << " and dtype=" << getDtypeName(dtype));

  const int root_rank = 0;
  const int root_value = 42;

  // Create tensor with different values based on rank
  at::Tensor tensor =
      createBroadcastTensor(root_rank, root_value, count, dtype);

  // Call broadcast
  auto work = torchcomm_->broadcast(tensor, root_rank, true);

  // Wait for the broadcast to complete
  work->wait();

  // Verify the results
  verifyBroadcastResults(tensor, root_value);
}

// Test function for asynchronous broadcast with early reset
void BroadcastTest::testAsyncBroadcastEarlyReset(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing async broadcast with early reset with count=" << count
      << " and dtype=" << getDtypeName(dtype));

  const int root_rank = 0;
  const int root_value = 42;

  // Create tensor with different values based on rank
  at::Tensor tensor =
      createBroadcastTensor(root_rank, root_value, count, dtype);

  // Call broadcast
  auto work = torchcomm_->broadcast(tensor, root_rank, true);

  // Wait for the work to complete before resetting
  work->wait();

  // Reset the work object
  work.reset();

  // Verify the results
  verifyBroadcastResults(tensor, root_value);
}

// Test function for asynchronous broadcast with input deleted after enqueue
void BroadcastTest::testBroadcastInputDeleted(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing async broadcast with input deleted after enqueue with count="
      << count << " and dtype=" << getDtypeName(dtype));

  const int root_rank = 0;
  const int root_value = 42;

  // Create work object to hold the async operation
  c10::intrusive_ptr<torch::comms::TorchWork> work;

  {
    // Create tensor in a limited scope
    at::Tensor tensor =
        createBroadcastTensor(root_rank, root_value, count, dtype);

    // Call broadcast
    work = torchcomm_->broadcast(tensor, root_rank, false);

    // Tensor goes out of scope here and gets deleted
  }

  // Wait for the broadcast to complete even though tensor is deleted
  work->wait();

  // Note: Cannot verify results since tensor is deleted
  // This test validates that the operation completes without crashing
}

// CUDA Graph test function for broadcast
void BroadcastTest::testGraphBroadcast(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing CUDA Graph broadcast with count="
                           << count << " and dtype=" << getDtypeName(dtype));

  const int root_rank = 0;
  const int root_value = 99;

  // Create tensor with different values based on rank BEFORE graph capture
  at::Tensor tensor =
      createBroadcastTensor(root_rank, root_value, count, dtype);
  at::Tensor original_values = tensor.clone();

  // Create a non-default CUDA stream (required for CUDA graph capture)
  at::cuda::CUDAStream stream = at::cuda::getStreamFromPool();

  // Set the stream as current for graph capture
  at::cuda::CUDAStreamGuard guard(stream);

  // Create PyTorch CUDA graph
  at::cuda::CUDAGraph graph;

  // Capture the broadcast operation in the graph
  graph.capture_begin();

  // Call broadcast without keeping the work object
  torchcomm_->broadcast(tensor, root_rank, false);

  graph.capture_end();

  // Replay the captured graph multiple times
  for (int i = 0; i < num_replays; ++i) {
    // Reset tensor to original values before each replay
    tensor.copy_(original_values);

    graph.replay();

    // Verify the results after each replay
    verifyBroadcastResults(tensor, root_value);
  }
}

// CUDA Graph test function for broadcast with input deleted after graph
// creation
void BroadcastTest::testGraphBroadcastInputDeleted(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing CUDA Graph broadcast with input deleted after graph creation with count="
      << count << " and dtype=" << getDtypeName(dtype));

  const int root_rank = 0;
  const int root_value = 99;

  // Create a non-default CUDA stream (required for CUDA graph capture)
  at::cuda::CUDAStream stream = at::cuda::getStreamFromPool();

  // Set the stream as current for graph capture
  at::cuda::CUDAStreamGuard guard(stream);

  // Create PyTorch CUDA graph
  at::cuda::CUDAGraph graph;

  {
    // Create tensor in a limited scope
    at::Tensor tensor =
        createBroadcastTensor(root_rank, root_value, count, dtype);

    // Capture the broadcast operation in the graph
    graph.capture_begin();

    // Call broadcast without keeping the work object
    torchcomm_->broadcast(tensor, root_rank, false);

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

// Helper function to create tensor for broadcast
at::Tensor BroadcastTest::createBroadcastTensor(
    int root_rank,
    int value,
    int count,
    at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  at::Tensor tensor;

  // Initialize tensor based on dtype
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
void BroadcastTest::verifyBroadcastResults(
    const at::Tensor& tensor,
    int value) {
  // Use verifyTensorEquality to compare tensor with expected tensor
  std::string description = "broadcast with value " + std::to_string(value);
  verifyTensorEquality(tensor.cpu(), value, description);
}
