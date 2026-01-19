// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "AllReduceTest.hpp"

#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>
#include "TorchCommTestHelpers.h"

std::unique_ptr<TorchCommTestWrapper> AllReduceTest::createWrapper() {
  return std::make_unique<TorchCommTestWrapper>();
}

void AllReduceTest::SetUp() {
  wrapper_ = createWrapper();
  torchcomm_ = wrapper_->getTorchComm();
  rank_ = torchcomm_->getRank();
  num_ranks_ = torchcomm_->getSize();
}

void AllReduceTest::TearDown() {
  // Explicitly reset the TorchComm object to ensure proper cleanup
  torchcomm_.reset();
  wrapper_.reset();
}

// Test function for synchronous all_reduce with work object
void AllReduceTest::testSyncAllReduce(
    int count,
    at::ScalarType dtype,
    torch::comms::ReduceOp op) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing sync all_reduce with count=" << count
                           << " and dtype=" << getDtypeName(dtype)
                           << " and op=" << getOpName(op));

  // Create input tensor with rank-specific values
  at::Tensor input = createInputTensor(count, dtype);

  // Call all_reduce
  auto work = torchcomm_->all_reduce(input, op, false);
  work->wait();

  // Verify the results
  verifyResults(input, op);
}

// Test function for synchronous all_reduce without work object
void AllReduceTest::testSyncAllReduceNoWork(
    int count,
    at::ScalarType dtype,
    torch::comms::ReduceOp op) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing sync all_reduce without work object with count=" << count
      << " and dtype=" << getDtypeName(dtype) << " and op=" << getOpName(op));

  // Create input tensor with rank-specific values
  at::Tensor input = createInputTensor(count, dtype);

  // Call all_reduce without keeping the work object
  torchcomm_->all_reduce(input, op, false);

  // Verify the results
  verifyResults(input, op);
}

// Test function for asynchronous all_reduce with wait
void AllReduceTest::testAsyncAllReduce(
    int count,
    at::ScalarType dtype,
    torch::comms::ReduceOp op) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing async all_reduce with count=" << count
                           << " and dtype=" << getDtypeName(dtype)
                           << " and op=" << getOpName(op));

  // Create input tensor with rank-specific values
  at::Tensor input = createInputTensor(count, dtype);

  // Call all_reduce
  auto work = torchcomm_->all_reduce(input, op, true);

  // Wait for the all_reduce to complete
  work->wait();

  // Verify the results
  verifyResults(input, op);
}

// Test function for asynchronous all_reduce with early reset
void AllReduceTest::testAsyncAllReduceEarlyReset(
    int count,
    at::ScalarType dtype,
    torch::comms::ReduceOp op) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing async all_reduce with early reset with count=" << count
      << " and dtype=" << getDtypeName(dtype) << " and op=" << getOpName(op));

  // Create input tensor with rank-specific values
  at::Tensor input = createInputTensor(count, dtype);

  // Call all_reduce
  auto work = torchcomm_->all_reduce(input, op, true);

  // Wait for the work to complete before resetting
  work->wait();

  // Reset the work object
  work.reset();

  // Verify the results
  verifyResults(input, op);
}

// Test function for asynchronous all_reduce with input deleted after enqueue
void AllReduceTest::testAllReduceInputDeleted(
    int count,
    at::ScalarType dtype,
    torch::comms::ReduceOp op) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing async all_reduce with input deleted after enqueue with count="
      << count << " and dtype=" << getDtypeName(dtype)
      << " and op=" << getOpName(op));

  {
    // Create input tensor in a limited scope
    at::Tensor input = createInputTensor(count, dtype);

    // Call all_reduce
    torchcomm_->all_reduce(input, op, false);

    // Input tensor goes out of scope here and gets deleted
  }

  // Note: For all_reduce, the result is stored in the input tensor which is now
  // deleted This test validates that the operation completes without crashing
}

// CUDA Graph test function for all_reduce
void AllReduceTest::testGraphAllReduce(
    int count,
    at::ScalarType dtype,
    torch::comms::ReduceOp op) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing CUDA Graph all_reduce with count="
                           << count << " and dtype=" << getDtypeName(dtype)
                           << " and op=" << getOpName(op));

  // Create a non-default CUDA stream (required for CUDA graph capture)
  at::cuda::CUDAStream stream = at::cuda::getStreamFromPool();

  // Set the stream as current for graph capture
  at::cuda::CUDAStreamGuard guard(stream);

  // Create input tensor AFTER setting non-default stream but BEFORE graph
  // capture
  at::Tensor input = createInputTensor(count, dtype);
  at::Tensor original_values = input.clone();

  // Create PyTorch CUDA graph
  at::cuda::CUDAGraph graph;

  // Capture the reset + all_reduce operations in the graph
  graph.capture_begin();

  // Call all_reduce without keeping the work object
  torchcomm_->all_reduce(input, op, false);

  graph.capture_end();

  // Replay the captured graph multiple times
  for (int i = 0; i < num_replays; ++i) {
    // Reset the output buffer before graph replay
    input.copy_(original_values);

    graph.replay();

    // Verify the results after each replay
    verifyResults(input, op);
  }

  graph.reset();
}

// CUDA Graph test function for all_reduce with input deleted after graph
// creation
void AllReduceTest::testGraphAllReduceInputDeleted(
    int count,
    at::ScalarType dtype,
    torch::comms::ReduceOp op) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing CUDA Graph all_reduce with input deleted after graph creation with count="
      << count << " and dtype=" << getDtypeName(dtype)
      << " and op=" << getOpName(op));

  // Create a non-default CUDA stream (required for CUDA graph capture)
  at::cuda::CUDAStream stream = at::cuda::getStreamFromPool();

  // Set the stream as current for graph capture
  at::cuda::CUDAStreamGuard guard(stream);

  // Create PyTorch CUDA graph
  at::cuda::CUDAGraph graph;

  {
    // Create input tensor in a limited scope
    at::Tensor input = createInputTensor(count, dtype);

    // Capture the reset + all_reduce operations in the graph
    graph.capture_begin();

    // Call all_reduce without keeping the work object
    torchcomm_->all_reduce(input, op, false);

    graph.capture_end();

    // Input tensor goes out of scope here and gets deleted
  }

  // Replay the captured graph multiple times even though input is deleted
  for (int i = 0; i < num_replays; ++i) {
    graph.replay();

    // Note: Cannot verify results since input tensor is deleted
    // This test validates that the graph replay completes without crashing
  }

  graph.reset();
}

// Helper function to create input tensor
at::Tensor AllReduceTest::createInputTensor(int count, at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  at::Tensor input;
  if (dtype == at::kFloat || dtype == at::kBFloat16 || dtype == at::kHalf ||
      dtype == at::kDouble) {
    input = at::ones({count}, options) * static_cast<float>(rank_ + 1);
  } else if (dtype == at::kInt) {
    input = at::ones({count}, options) * static_cast<int>(rank_ + 1);
  } else if (dtype == at::kChar) {
    input = at::ones({count}, options) * static_cast<signed char>(rank_ + 1);
  }
  return input;
}

// Helper function to create PreMul tensor
at::Tensor AllReduceTest::createPreMulFactorTensor(at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  at::Tensor factor;
  if (dtype == at::kFloat || dtype == at::kBFloat16 || dtype == at::kHalf ||
      dtype == at::kDouble) {
    factor = at::ones({1}, options) * static_cast<float>(2.0);
  } else {
    throw std::runtime_error("Unsupported dtype for PreMul");
  }
  return factor;
}

// Helper function to calculate expected result
double AllReduceTest::calculateExpectedResult(torch::comms::ReduceOp op) {
  if (op == torch::comms::ReduceOp::SUM) {
    return num_ranks_ * (num_ranks_ + 1) / 2;
  } else if (op == torch::comms::ReduceOp::RedOpType::MAX) {
    return num_ranks_;
  } else if (op == torch::comms::ReduceOp::RedOpType::AVG) {
    return static_cast<double>(num_ranks_ * (num_ranks_ + 1) / 2) / num_ranks_;
  } else if (op == torch::comms::ReduceOp::RedOpType::PREMUL_SUM) {
    return num_ranks_ * (num_ranks_ + 1);
  } else {
    throw std::runtime_error("Unsupported reduce operation");
  }
}

// Helper function to verify results
void AllReduceTest::verifyResults(
    const at::Tensor& input,
    torch::comms::ReduceOp op) {
  // Calculate expected result
  double expected = calculateExpectedResult(op);

  // Use verifyTensorEquality to compare input with expected tensor
  std::string description = "all_reduce with op " + getOpName(op);
  verifyTensorEquality(input.cpu(), expected, description);
}
