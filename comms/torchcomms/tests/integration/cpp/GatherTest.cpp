// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "GatherTest.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>
#include <vector>
#include "comms/torchcomms/tests/integration/cpp/TorchCommTestHelpers.h"

std::unique_ptr<TorchCommTestWrapper> GatherTest::createWrapper() {
  return std::make_unique<TorchCommTestWrapper>();
}

void GatherTest::synchronizeStream() {
  at::cuda::getCurrentCUDAStream(0).synchronize();
}

void GatherTest::SetUp() {
  wrapper_ = createWrapper();
  torchcomm_ = wrapper_->getTorchComm();
  rank_ = torchcomm_->getRank();
  num_ranks_ = torchcomm_->getSize();
}

void GatherTest::TearDown() {
  // Explicitly reset the TorchComm object to ensure proper cleanup
  torchcomm_.reset();
  wrapper_.reset();
}

// Test function for synchronous gather with work object
void GatherTest::testSyncGather(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing sync gather with count=" << count
                           << " and dtype=" << getDtypeName(dtype));

  // Create input tensor with rank-specific values
  at::Tensor input = createInputTensor(count, dtype);

  // Root rank will receive data from all ranks
  const int root_rank = 0;
  std::vector<at::Tensor> outputs =
      createOutputTensors(root_rank, count, dtype);

  // Call gather
  auto work = torchcomm_->gather(outputs, input, root_rank, false);
  work->wait();

  // Verify the results on root rank
  verifyGatherResults(outputs, root_rank);
}

// Test function for synchronous gather without work object
void GatherTest::testSyncGatherNoWork(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing sync gather without work object with count=" << count
      << " and dtype=" << getDtypeName(dtype));

  // Create input tensor with rank-specific values
  at::Tensor input = createInputTensor(count, dtype);

  // Root rank will receive data from all ranks
  const int root_rank = 0;
  std::vector<at::Tensor> outputs =
      createOutputTensors(root_rank, count, dtype);

  // Call gather without keeping the work object
  torchcomm_->gather(outputs, input, root_rank, false);

  // Verify the results on root rank
  verifyGatherResults(outputs, root_rank);
}

// Test function for asynchronous gather with wait
void GatherTest::testAsyncGather(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing async gather with count=" << count
                           << " and dtype=" << getDtypeName(dtype));

  // Create input tensor with rank-specific values
  at::Tensor input = createInputTensor(count, dtype);

  // Root rank will receive data from all ranks
  const int root_rank = 0;
  std::vector<at::Tensor> outputs =
      createOutputTensors(root_rank, count, dtype);

  // Call gather
  auto work = torchcomm_->gather(outputs, input, root_rank, true);

  // Wait for the gather to complete
  work->wait();

  // Verify the results on root rank
  verifyGatherResults(outputs, root_rank);
}

// Test function for asynchronous gather with early reset
void GatherTest::testAsyncGatherEarlyReset(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing async gather with early reset with count=" << count
      << " and dtype=" << getDtypeName(dtype));

  // Create input tensor with rank-specific values
  at::Tensor input = createInputTensor(count, dtype);

  // Root rank will receive data from all ranks
  const int root_rank = 0;
  std::vector<at::Tensor> outputs =
      createOutputTensors(root_rank, count, dtype);

  // Call gather
  auto work = torchcomm_->gather(outputs, input, root_rank, true);

  // Wait for the work to complete before resetting
  work->wait();

  // Reset the work object
  work.reset();

  // Verify the results on root rank
  verifyGatherResults(outputs, root_rank);
}

// Test function for asynchronous gather with input deleted after enqueue
void GatherTest::testGatherInputDeleted(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing async gather with input deleted after enqueue with count="
      << count << " and dtype=" << getDtypeName(dtype));

  // Root rank will receive data from all ranks
  const int root_rank = 0;
  std::vector<at::Tensor> outputs =
      createOutputTensors(root_rank, count, dtype);

  {
    // Create input tensor in a limited scope
    at::Tensor input = createInputTensor(count, dtype);

    // Call gather
    torchcomm_->gather(outputs, input, root_rank, false);

    // Input tensor goes out of scope here and gets deleted
  }

  // Verify the results on root rank
  verifyGatherResults(outputs, root_rank);
}

// CUDA Graph test function for gather
void GatherTest::testGraphGather(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing CUDA Graph gather with count=" << count
                           << " and dtype=" << getDtypeName(dtype));

  // Create input tensor with rank-specific values BEFORE graph capture
  at::Tensor input = createInputTensor(count, dtype);

  // Root rank will receive data from all ranks
  const int root_rank = 0;
  std::vector<at::Tensor> outputs =
      createOutputTensors(root_rank, count, dtype);
  std::vector<at::Tensor> original_outputs;
  original_outputs.reserve(outputs.size());
  for (const auto& output : outputs) {
    original_outputs.push_back(output.clone());
  }

  // Create a non-default CUDA stream (required for CUDA graph capture)
  at::cuda::CUDAStream stream = at::cuda::getStreamFromPool();

  // Set the stream as current for graph capture
  at::cuda::CUDAStreamGuard guard(stream);

  // Create PyTorch CUDA graph
  at::cuda::CUDAGraph graph;

  // Capture the gather operation in the graph
  graph.capture_begin();

  // Call gather without keeping the work object
  torchcomm_->gather(outputs, input, root_rank, false);

  graph.capture_end();

  // Replay the captured graph multiple times
  for (int i = 0; i < num_replays; ++i) {
    // Reset output tensors before each replay
    for (size_t j = 0; j < outputs.size(); ++j) {
      outputs[j].copy_(original_outputs[j]);
    }

    graph.replay();

    // Verify the results after each replay
    verifyGatherResults(outputs, root_rank);
  }
}

// CUDA Graph test function for gather with input deleted after graph creation
void GatherTest::testGraphGatherInputDeleted(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing CUDA Graph gather with input deleted after graph creation with count="
      << count << " and dtype=" << getDtypeName(dtype));

  // Root rank will receive data from all ranks
  const int root_rank = 0;
  std::vector<at::Tensor> outputs =
      createOutputTensors(root_rank, count, dtype);
  std::vector<at::Tensor> original_outputs;
  original_outputs.reserve(outputs.size());
  for (const auto& output : outputs) {
    original_outputs.push_back(output.clone());
  }

  // Create a non-default CUDA stream (required for CUDA graph capture)
  at::cuda::CUDAStream stream = at::cuda::getStreamFromPool();

  // Set the stream as current for graph capture
  at::cuda::CUDAStreamGuard guard(stream);

  // Create PyTorch CUDA graph
  at::cuda::CUDAGraph graph;

  {
    // Create input tensor in a limited scope
    at::Tensor input = createInputTensor(count, dtype);

    // Capture the gather operation in the graph
    graph.capture_begin();

    // Call gather without keeping the work object
    torchcomm_->gather(outputs, input, root_rank, false);

    graph.capture_end();

    // Input tensor goes out of scope here and gets deleted
  }

  // Replay the captured graph multiple times even though input is deleted
  for (int i = 0; i < num_replays; ++i) {
    // Reset output tensors before each replay
    for (size_t j = 0; j < outputs.size(); ++j) {
      outputs[j].copy_(original_outputs[j]);
    }

    graph.replay();

    // Verify the results after each replay
    verifyGatherResults(outputs, root_rank);
  }
}

// Helper function to create input tensor
at::Tensor GatherTest::createInputTensor(int count, at::ScalarType dtype) {
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

// Helper function to create output tensors
std::vector<at::Tensor> GatherTest::createOutputTensors(
    int root_rank,
    int count,
    at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  std::vector<at::Tensor> outputs;

  // Only root rank needs to allocate output tensors
  if (rank_ == root_rank) {
    outputs.reserve(num_ranks_);
    for (int i = 0; i < num_ranks_; i++) {
      outputs.push_back(at::zeros({count}, options));
    }
  }

  return outputs;
}

// Helper function to verify results
void GatherTest::verifyGatherResults(
    const std::vector<at::Tensor>& outputs,
    int root_rank) {
  // Only verify on root rank
  if (rank_ != root_rank) {
    synchronizeStream();
    return;
  }

  for (int i = 0; i < num_ranks_; i++) {
    // Use verifyTensorEquality to compare output with expected tensor
    std::string description = "gather rank " + std::to_string(i) + " tensor";
    verifyTensorEquality(outputs[i].cpu(), i + 1, description);
  }
}
