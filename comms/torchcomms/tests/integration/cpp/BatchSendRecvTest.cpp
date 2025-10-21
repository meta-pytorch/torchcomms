// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "BatchSendRecvTest.hpp"

#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>
#include <vector>
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

std::unique_ptr<TorchCommTestWrapper> BatchSendRecvTest::createWrapper() {
  return std::make_unique<TorchCommTestWrapper>();
}

void BatchSendRecvTest::SetUp() {
  wrapper_ = createWrapper();
  torchcomm_ = wrapper_->getTorchComm();
  rank_ = torchcomm_->getRank();
  num_ranks_ = torchcomm_->getSize();
}

void BatchSendRecvTest::TearDown() {
  // Explicitly reset the TorchComm object to ensure proper cleanup
  torchcomm_.reset();
  wrapper_.reset();
}

// Test function for synchronous batch SendRecv operations
void BatchSendRecvTest::testSyncBatchSendRecv(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing sync batch SendRecv with count=" << count
                           << " and dtype=" << getDtypeName(dtype));

  auto params = createBatchSendRecvParams(count, dtype);

  // Create batch operation object
  auto batch_op = torchcomm_->batch_op_create();

  // Add send operations to batch
  for (size_t i = 0; i < params.send_tensors.size(); ++i) {
    batch_op.send(params.send_tensors[i], params.send_ranks[i]);
  }

  // Add recv operations to batch
  for (size_t i = 0; i < params.recv_tensors.size(); ++i) {
    batch_op.recv(params.recv_tensors[i], params.recv_ranks[i]);
  }

  // Issue batch operations synchronously
  auto work = batch_op.issue(false);

  work->wait();

  // Verify the results
  verifyResults(params.recv_tensors, params.recv_ranks[0]);
}

// Test function for synchronous batch SendRecv operations without work object
void BatchSendRecvTest::testSyncBatchSendRecvNoWork(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing sync batch SendRecv without work object with count=" << count
      << " and dtype=" << getDtypeName(dtype));

  auto params = createBatchSendRecvParams(count, dtype);

  // Create batch operation object
  auto batch_op = torchcomm_->batch_op_create();

  // Add send operations to batch
  for (size_t i = 0; i < params.send_tensors.size(); ++i) {
    batch_op.send(params.send_tensors[i], params.send_ranks[i]);
  }

  // Add recv operations to batch
  for (size_t i = 0; i < params.recv_tensors.size(); ++i) {
    batch_op.recv(params.recv_tensors[i], params.recv_ranks[i]);
  }

  // Issue batch operations synchronously without storing work object
  batch_op.issue(false);

  // Verify the results
  verifyResults(params.recv_tensors, params.recv_ranks[0]);
}

// Test function for asynchronous batch SendRecv operations
void BatchSendRecvTest::testAsyncBatchSendRecv(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing async batch SendRecv with count="
                           << count << " and dtype=" << getDtypeName(dtype));

  auto params = createBatchSendRecvParams(count, dtype);

  // Create batch operation object
  auto batch_op = torchcomm_->batch_op_create();

  // Add send operations to batch
  for (size_t i = 0; i < params.send_tensors.size(); ++i) {
    batch_op.send(params.send_tensors[i], params.send_ranks[i]);
  }

  // Add recv operations to batch
  for (size_t i = 0; i < params.recv_tensors.size(); ++i) {
    batch_op.recv(params.recv_tensors[i], params.recv_ranks[i]);
  }

  // Issue batch operations asynchronously
  auto work = batch_op.issue(true);

  // Wait for completion
  work->wait();

  // Verify the results
  verifyResults(params.recv_tensors, params.recv_ranks[0]);
}

// Test function for asynchronous batch SendRecv operations with early reset
void BatchSendRecvTest::testAsyncBatchSendRecvEarlyReset(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing async batch SendRecv with early reset with count=" << count
      << " and dtype=" << getDtypeName(dtype));

  auto params = createBatchSendRecvParams(count, dtype);

  // Create batch operation object
  auto batch_op = torchcomm_->batch_op_create();

  // Add send operations to batch
  for (size_t i = 0; i < params.send_tensors.size(); ++i) {
    batch_op.send(params.send_tensors[i], params.send_ranks[i]);
  }

  // Add recv operations to batch
  for (size_t i = 0; i < params.recv_tensors.size(); ++i) {
    batch_op.recv(params.recv_tensors[i], params.recv_ranks[i]);
  }

  // Issue batch operations asynchronously
  auto work = batch_op.issue(true);

  // Wait for completion before resetting
  work->wait();

  // Reset the work object
  work.reset();

  // Verify the results
  verifyResults(params.recv_tensors, params.recv_ranks[0]);
}

// Test function for asynchronous batch SendRecv operations with input deleted
// after enqueue
void BatchSendRecvTest::testBatchSendRecvInputDeleted(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing async batch SendRecv with input deleted after enqueue with count="
      << count << " and dtype=" << getDtypeName(dtype));

  // Create recv tensors that persist throughout the test
  std::vector<at::Tensor> recv_tensors;
  std::vector<int> recv_ranks;

  // Create recv parameters
  int prev_rank = (rank_ + num_ranks_ - 1) % num_ranks_;
  for (int i = 0; i < 2; ++i) {
    recv_tensors.push_back(createRecvTensor(count, dtype));
    recv_ranks.push_back(prev_rank);
  }

  {
    // Create send tensors and ranks in a limited scope
    std::vector<at::Tensor> send_tensors;
    std::vector<int> send_ranks;

    int next_rank = (rank_ + 1) % num_ranks_;
    for (int i = 0; i < 2; ++i) {
      send_tensors.push_back(createSendTensor(count, dtype, i));
      send_ranks.push_back(next_rank);
    }

    // Create batch operation object
    auto batch_op = torchcomm_->batch_op_create();

    // Add send operations to batch
    for (size_t i = 0; i < send_tensors.size(); ++i) {
      batch_op.send(send_tensors[i], send_ranks[i]);
    }

    // Add recv operations to batch
    for (size_t i = 0; i < recv_tensors.size(); ++i) {
      batch_op.recv(recv_tensors[i], recv_ranks[i]);
    }

    // Issue batch operations synchronously
    batch_op.issue(false);

    // Send tensors go out of scope here and get deleted
  }

  // Verify the results
  verifyResults(recv_tensors, recv_ranks[0]);
}

// CUDA Graph test function for batch SendRecv operations
void BatchSendRecvTest::testGraphBatchSendRecv(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing CUDA Graph batch SendRecv with count="
                           << count << " and dtype=" << getDtypeName(dtype));

  // Create batch send/recv parameters BEFORE graph capture
  auto params = createBatchSendRecvParams(count, dtype);
  std::vector<at::Tensor> original_recv_tensors;
  original_recv_tensors.reserve(params.recv_tensors.size());
  for (const auto& recv_tensor : params.recv_tensors) {
    original_recv_tensors.push_back(recv_tensor.clone());
  }

  // Create a non-default CUDA stream (required for CUDA graph capture)
  at::cuda::CUDAStream stream = at::cuda::getStreamFromPool();

  // Set the stream as current for graph capture
  at::cuda::CUDAStreamGuard guard(stream);

  // Create PyTorch CUDA graph
  at::cuda::CUDAGraph graph;

  // Capture the batch operations in the graph
  graph.capture_begin();

  // Create batch operation object
  auto batch_op = torchcomm_->batch_op_create();

  // Add send operations to batch
  for (size_t i = 0; i < params.send_tensors.size(); ++i) {
    batch_op.send(params.send_tensors[i], params.send_ranks[i]);
  }

  // Add recv operations to batch
  for (size_t i = 0; i < params.recv_tensors.size(); ++i) {
    batch_op.recv(params.recv_tensors[i], params.recv_ranks[i]);
  }

  // Issue batch operations synchronously without storing work object
  batch_op.issue(false);

  graph.capture_end();

  // Replay the captured graph multiple times
  for (int i = 0; i < num_replays; ++i) {
    // Reset output buffers before each replay
    for (size_t j = 0; j < params.recv_tensors.size(); ++j) {
      params.recv_tensors[j].copy_(original_recv_tensors[j]);
    }

    graph.replay();

    // Verify the results after each replay
    verifyResults(params.recv_tensors, params.recv_ranks[0]);
  }
}

// CUDA Graph test function for batch SendRecv operations with input deleted
// after graph creation
void BatchSendRecvTest::testGraphBatchSendRecvInputDeleted(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing CUDA Graph batch SendRecv with input deleted after graph creation with count="
      << count << " and dtype=" << getDtypeName(dtype));

  // Create recv tensors that persist throughout the test
  std::vector<at::Tensor> recv_tensors;
  std::vector<int> recv_ranks;

  // Create recv parameters
  int prev_rank = (rank_ + num_ranks_ - 1) % num_ranks_;
  for (int i = 0; i < 2; ++i) {
    recv_tensors.push_back(createRecvTensor(count, dtype));
    recv_ranks.push_back(prev_rank);
  }

  std::vector<at::Tensor> original_recv_tensors;
  original_recv_tensors.reserve(recv_tensors.size());
  for (const auto& recv_tensor : recv_tensors) {
    original_recv_tensors.push_back(recv_tensor.clone());
  }

  // Create a non-default CUDA stream (required for CUDA graph capture)
  at::cuda::CUDAStream stream = at::cuda::getStreamFromPool();

  // Set the stream as current for graph capture
  at::cuda::CUDAStreamGuard guard(stream);

  // Create PyTorch CUDA graph
  at::cuda::CUDAGraph graph;

  {
    // Create send tensors and ranks in a limited scope
    std::vector<at::Tensor> send_tensors;
    std::vector<int> send_ranks;

    int next_rank = (rank_ + 1) % num_ranks_;
    for (int i = 0; i < 2; ++i) {
      send_tensors.push_back(createSendTensor(count, dtype, i));
      send_ranks.push_back(next_rank);
    }

    // Capture the batch operations in the graph
    graph.capture_begin();

    // Create batch operation object
    auto batch_op = torchcomm_->batch_op_create();

    // Add send operations to batch
    for (size_t i = 0; i < send_tensors.size(); ++i) {
      batch_op.send(send_tensors[i], send_ranks[i]);
    }

    // Add recv operations to batch
    for (size_t i = 0; i < recv_tensors.size(); ++i) {
      batch_op.recv(recv_tensors[i], recv_ranks[i]);
    }

    // Issue batch operations synchronously without storing work object
    batch_op.issue(false);

    graph.capture_end();

    // Send tensors go out of scope here and get deleted
  }

  // Replay the captured graph multiple times even though send tensors are
  // deleted
  for (int i = 0; i < num_replays; ++i) {
    // Reset output buffers before each replay
    for (size_t j = 0; j < recv_tensors.size(); ++j) {
      recv_tensors[j].copy_(original_recv_tensors[j]);
    }

    graph.replay();

    // Verify the results after each replay
    verifyResults(recv_tensors, recv_ranks[0]);
  }
}

// Helper function to create send tensor with tensor-specific values
at::Tensor BatchSendRecvTest::createSendTensor(
    int count,
    at::ScalarType dtype,
    int tensor_id) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  at::Tensor send_tensor;
  int value = (rank_ + 1) * 10 + tensor_id; // Make each tensor unique

  if (dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16) {
    send_tensor = at::ones({count}, options) * static_cast<float>(value);
  } else if (dtype == at::kInt) {
    send_tensor = at::ones({count}, options) * static_cast<int>(value);
  } else if (dtype == at::kChar) {
    send_tensor =
        at::ones({count}, options) * static_cast<signed char>(value % 128);
  }
  return send_tensor;
}

// Helper function to create receive tensor
at::Tensor BatchSendRecvTest::createRecvTensor(
    int count,
    at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  return at::zeros({count}, options);
}

// Helper function to verify results
void BatchSendRecvTest::verifyResults(
    const std::vector<at::Tensor>& recv_tensors,
    int recv_rank) {
  for (size_t i = 0; i < recv_tensors.size(); ++i) {
    int expected_value = (recv_rank + 1) * 10 + static_cast<int>(i);
    std::string description = "recv rank " + std::to_string(recv_rank) +
        " tensor " + std::to_string(i);

    if (recv_tensors[i].scalar_type() == at::kChar) {
      expected_value = expected_value % 128;
    }

    verifyTensorEquality(recv_tensors[i].cpu(), expected_value, description);
  }
}
