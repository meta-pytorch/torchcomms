// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "ReduceScatterVTest.hpp"

#include <gtest/gtest.h>
#include <vector>
#include "comms/torchcomms/TorchWork.hpp"
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

std::unique_ptr<TorchCommTestWrapper> ReduceScatterVTest::createWrapper() {
  return std::make_unique<TorchCommTestWrapper>();
}

void ReduceScatterVTest::SetUp() {
  wrapper_ = createWrapper();
  torchcomm_ = wrapper_->getTorchComm();
  rank_ = torchcomm_->getRank();
  num_ranks_ = torchcomm_->getSize();
  device_type_ = wrapper_->getDevice().type();
}

void ReduceScatterVTest::TearDown() {
  // Explicitly reset the TorchComm object to ensure proper cleanup
  torchcomm_.reset();
  wrapper_.reset();
}

// Helper function to get per-rank counts for reduce_scatter_v (variable sizes)
std::vector<int> ReduceScatterVTest::getCounts(int count) {
  std::vector<int> counts(num_ranks_, count);
  for (int i = 0; i < num_ranks_; i++) {
    counts[i] = count + i;
  }
  return counts;
}

// Test function for synchronous reduce_scatter_v with work object
void ReduceScatterVTest::testSyncReduceScatterV(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing sync reduce_scatter_v with count="
                           << count << " and dtype=" << getDtypeName(dtype)
                           << " and op=" << getOpName(op));

  // Create input and output tensors
  auto counts = getCounts(count);
  std::vector<at::Tensor> input_tensors = createInputTensors(counts, dtype);
  at::Tensor output = createOutputTensor(counts[rank_], dtype);

  // Call reduce_scatter_v
  auto work = torchcomm_->reduce_scatter_v(output, input_tensors, op, false);
  work->wait();

  // Verify the results
  verifyResults(output, op);
}

// Test function for synchronous reduce_scatter_v without work object
void ReduceScatterVTest::testSyncReduceScatterVNoWork(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing sync reduce_scatter_v without work object with count="
      << count << " and dtype=" << getDtypeName(dtype)
      << " and op=" << getOpName(op));

  // Create input and output tensors
  auto counts = getCounts(count);
  std::vector<at::Tensor> input_tensors = createInputTensors(counts, dtype);
  at::Tensor output = createOutputTensor(counts[rank_], dtype);

  // Call reduce_scatter_v without keeping the work object
  torchcomm_->reduce_scatter_v(output, input_tensors, op, false);

  // Verify the results
  verifyResults(output, op);
}

// Test function for asynchronous reduce_scatter_v with wait
void ReduceScatterVTest::testAsyncReduceScatterV(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing async reduce_scatter_v with count="
                           << count << " and dtype=" << getDtypeName(dtype)
                           << " and op=" << getOpName(op));

  // Create input and output tensors
  auto counts = getCounts(count);
  std::vector<at::Tensor> input_tensors = createInputTensors(counts, dtype);
  at::Tensor output = createOutputTensor(counts[rank_], dtype);

  // Call reduce_scatter_v
  auto work = torchcomm_->reduce_scatter_v(output, input_tensors, op, true);

  // Wait for the reduce_scatter_v to complete
  work->wait();

  // Verify the results
  verifyResults(output, op);
}

// Test function for asynchronous reduce_scatter_v with early reset
void ReduceScatterVTest::testAsyncReduceScatterVEarlyReset(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing async reduce_scatter_v with early reset with count=" << count
      << " and dtype=" << getDtypeName(dtype) << " and op=" << getOpName(op));

  // Create input and output tensors
  auto counts = getCounts(count);
  std::vector<at::Tensor> input_tensors = createInputTensors(counts, dtype);
  at::Tensor output = createOutputTensor(counts[rank_], dtype);

  // Call reduce_scatter_v
  auto work = torchcomm_->reduce_scatter_v(output, input_tensors, op, true);

  // Wait for the work to complete before resetting
  work->wait();

  // Reset the work object
  work.reset();

  // Verify the results
  verifyResults(output, op);
}

// Test function for asynchronous reduce_scatter_v with input deleted after
// enqueue
void ReduceScatterVTest::testReduceScatterVInputDeleted(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing async reduce_scatter_v with input deleted after enqueue with count="
      << count << " and dtype=" << getDtypeName(dtype)
      << " and op=" << getOpName(op));

  // Create output tensor that persists throughout the test
  auto counts = getCounts(count);
  at::Tensor output = createOutputTensor(counts[rank_], dtype);

  {
    // Create input tensors in a limited scope
    std::vector<at::Tensor> input_tensors = createInputTensors(counts, dtype);

    // Call reduce_scatter_v
    torchcomm_->reduce_scatter_v(output, input_tensors, op, false);

    // Input tensors go out of scope here and get deleted
  }

  // Verify the results
  verifyResults(output, op);
}

// CUDA Graph test function for reduce_scatter_v
void ReduceScatterVTest::testGraphReduceScatterV(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  // Skip CUDA Graph tests when running on CPU
  if (isRunningOnCPU()) {
    GTEST_SKIP() << "CUDA Graph tests are not supported on CPU";
  }

  SCOPED_TRACE(
      ::testing::Message() << "Testing CUDA Graph reduce_scatter_v with count="
                           << count << " and dtype=" << getDtypeName(dtype)
                           << " and op=" << getOpName(op));

  // Create a non-default CUDA stream (required for CUDA graph capture)
  at::cuda::CUDAStream stream = at::cuda::getStreamFromPool();

  // Set the stream as current for graph capture
  at::cuda::CUDAStreamGuard guard(stream);

  // Create input and output tensors AFTER setting non-default stream but BEFORE
  // graph capture
  auto counts = getCounts(count);
  std::vector<at::Tensor> input_tensors = createInputTensors(counts, dtype);
  at::Tensor output = createOutputTensor(counts[rank_], dtype);
  at::Tensor original_output = output.clone();

  // Create PyTorch CUDA graph
  at::cuda::CUDAGraph graph;

  // Capture the reset + reduce_scatter_v operations in the graph
  graph.capture_begin();

  // Call reduce_scatter_v without keeping the work object
  torchcomm_->reduce_scatter_v(output, input_tensors, op, false);

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

// CUDA Graph test function for reduce_scatter_v with input deleted after graph
// creation
void ReduceScatterVTest::testGraphReduceScatterVInputDeleted(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  // Skip CUDA Graph tests when running on CPU
  if (isRunningOnCPU()) {
    GTEST_SKIP() << "CUDA Graph tests are not supported on CPU";
  }

  SCOPED_TRACE(
      ::testing::Message()
      << "Testing CUDA Graph reduce_scatter_v with input deleted after graph creation with count="
      << count << " and dtype=" << getDtypeName(dtype)
      << " and op=" << getOpName(op));

  // Create a non-default CUDA stream (required for CUDA graph capture)
  at::cuda::CUDAStream stream = at::cuda::getStreamFromPool();

  // Set the stream as current for graph capture
  at::cuda::CUDAStreamGuard guard(stream);

  // Create output tensor that persists throughout the test
  auto counts = getCounts(count);
  at::Tensor output = createOutputTensor(counts[rank_], dtype);
  at::Tensor original_output = output.clone();

  // Create PyTorch CUDA graph
  at::cuda::CUDAGraph graph;

  {
    // Create input tensors in a limited scope
    std::vector<at::Tensor> input_tensors = createInputTensors(counts, dtype);

    // Capture the reset + reduce_scatter_v operations in the graph
    graph.capture_begin();

    // Call reduce_scatter_v without keeping the work object
    torchcomm_->reduce_scatter_v(output, input_tensors, op, false);

    graph.capture_end();

    // Input tensors go out of scope here and get deleted
  }

  // Replay the captured graph multiple times even though input tensors are
  // deleted
  for (int i = 0; i < num_replays; ++i) {
    // Reset output buffer before graph replay
    output.copy_(original_output);

    graph.replay();

    // Verify the results after each replay
    verifyResults(output, op);
  }
}

// Helper function to create input tensors
std::vector<at::Tensor> ReduceScatterVTest::createInputTensors(
    const std::vector<int>& counts,
    at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  std::vector<at::Tensor> input_tensors;
  input_tensors.reserve(num_ranks_);

  for (int r = 0; r < num_ranks_; r++) {
    // Each tensor has rank-specific values
    at::Tensor tensor;
    if (dtype == at::kFloat || dtype == at::kBFloat16 || dtype == at::kHalf) {
      tensor = at::ones({counts[r]}, options) * static_cast<float>(r + 1);
    } else if (dtype == at::kInt) {
      tensor = at::ones({counts[r]}, options) * static_cast<int>(r + 1);
    } else if (dtype == at::kChar) {
      tensor = at::ones({counts[r]}, options) * static_cast<signed char>(r + 1);
    }
    input_tensors.push_back(tensor);
  }

  return input_tensors;
}

// Helper function to create output tensor
at::Tensor ReduceScatterVTest::createOutputTensor(
    int count,
    at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  return at::zeros({count}, options);
}

// Helper function to calculate expected result
int ReduceScatterVTest::calculateExpectedResult(
    const torch::comms::ReduceOp& op) {
  if (op == torch::comms::ReduceOp::SUM) {
    return num_ranks_ * (rank_ + 1);
  } else if (op == torch::comms::ReduceOp::MAX) {
    return rank_ + 1;
  } else if (op == torch::comms::ReduceOp::AVG) {
    return rank_ + 1;
  } else {
    throw std::runtime_error("Unsupported reduce operation");
  }
}

// Helper function to verify results
void ReduceScatterVTest::verifyResults(
    const at::Tensor& output,
    const torch::comms::ReduceOp& op) {
  // Calculate expected result
  int expected = calculateExpectedResult(op);

  // Use verifyTensorEquality to compare output with expected tensor
  std::string description = "reduce_scatter_v with op " + getOpName(op);
  verifyTensorEquality(output.cpu(), expected, description);
}
