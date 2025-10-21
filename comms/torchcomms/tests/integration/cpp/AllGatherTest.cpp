// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "AllGatherTest.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>
#include <vector>
#include "TorchCommTestHelpers.h"

std::unique_ptr<TorchCommTestWrapper> AllGatherTest::createWrapper() {
  return std::make_unique<TorchCommTestWrapper>();
}

void AllGatherTest::SetUp() {
  wrapper_ = createWrapper();
  torchcomm_ = wrapper_->getTorchComm();
  rank_ = torchcomm_->getRank();
  num_ranks_ = torchcomm_->getSize();
}

void AllGatherTest::TearDown() {
  // Explicitly reset the TorchComm object to ensure proper cleanup
  torchcomm_.reset();
  wrapper_.reset();
}

// Test function for synchronous all_gather with work object
void AllGatherTest::testSyncAllGather(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing sync all_gather with count=" << count
                           << " and dtype=" << getDtypeName(dtype));

  // Create input and output tensors
  at::Tensor input = createInputTensor(count, dtype);
  std::vector<at::Tensor> outputs = createOutputTensors(count, dtype);

  // Call all_gather
  auto work = torchcomm_->all_gather(outputs, input, false);
  work->wait();

  // Verify the results
  verifyResults(outputs);
}

// Test function for synchronous all_gather without work object
void AllGatherTest::testSyncAllGatherNoWork(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing sync all_gather without work object with count=" << count
      << " and dtype=" << getDtypeName(dtype));

  // Create input and output tensors
  at::Tensor input = createInputTensor(count, dtype);
  std::vector<at::Tensor> outputs = createOutputTensors(count, dtype);

  // Call all_gather without keeping the work object
  torchcomm_->all_gather(outputs, input, false);

  // Verify the results
  verifyResults(outputs);
}

// Test function for asynchronous all_gather with wait
void AllGatherTest::testAsyncAllGather(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing async all_gather with count=" << count
                           << " and dtype=" << getDtypeName(dtype));

  // Create input and output tensors
  at::Tensor input = createInputTensor(count, dtype);
  std::vector<at::Tensor> outputs = createOutputTensors(count, dtype);

  // Call all_gather
  auto work = torchcomm_->all_gather(outputs, input, true);

  // Wait for the all_gather to complete
  work->wait();

  // Verify the results
  verifyResults(outputs);
}

// Test function for asynchronous all_gather with early reset
void AllGatherTest::testAsyncAllGatherEarlyReset(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing async all_gather with early reset with count=" << count
      << " and dtype=" << getDtypeName(dtype));

  // Create input and output tensors
  at::Tensor input = createInputTensor(count, dtype);
  std::vector<at::Tensor> outputs = createOutputTensors(count, dtype);

  // Call all_gather
  auto work = torchcomm_->all_gather(outputs, input, true);

  // Wait for the work to complete before resetting
  work->wait();

  // Reset the work object
  work.reset();

  // Verify the results
  verifyResults(outputs);
}

// Test function for asynchronous all_gather with input deleted after enqueue
void AllGatherTest::testAllGatherInputDeleted(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing async all_gather with input deleted after enqueue with count="
      << count << " and dtype=" << getDtypeName(dtype));

  // Create output tensors that persist throughout the test
  std::vector<at::Tensor> outputs = createOutputTensors(count, dtype);

  {
    // Create input tensor in a limited scope
    at::Tensor input = createInputTensor(count, dtype);

    // Call all_gather
    torchcomm_->all_gather(outputs, input, false);

    // Input tensor goes out of scope here and gets deleted
  }

  // Verify the results
  verifyResults(outputs);
}

// CUDA Graph test function for all_gather
void AllGatherTest::testGraphAllGather(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing CUDA Graph all_gather with count="
                           << count << " and dtype=" << getDtypeName(dtype));

  // Create input and output tensors BEFORE graph capture
  at::Tensor input = createInputTensor(count, dtype);
  std::vector<at::Tensor> outputs = createOutputTensors(count, dtype);
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

  // Capture the all_gather operation in the graph
  graph.capture_begin();

  torchcomm_->all_gather(outputs, input, false);

  graph.capture_end();

  // Replay the captured graph multiple times
  for (int i = 0; i < num_replays; ++i) {
    // Reset output tensors before each replay
    for (size_t j = 0; j < outputs.size(); ++j) {
      outputs[j].copy_(original_outputs[j]);
    }

    graph.replay();

    // Verify the results after each replay
    verifyResults(outputs);
  }
}

// CUDA Graph test function for all_gather with input deleted after graph
// creation
void AllGatherTest::testGraphAllGatherInputDeleted(
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing CUDA Graph all_gather with input deleted after graph creation with count="
      << count << " and dtype=" << getDtypeName(dtype));

  // Create output tensors that persist throughout the test
  std::vector<at::Tensor> outputs = createOutputTensors(count, dtype);
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

    // Capture the all_gather operation in the graph
    graph.capture_begin();

    torchcomm_->all_gather(outputs, input, false);

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
    verifyResults(outputs);
  }
}

// Helper function to create input tensor
at::Tensor AllGatherTest::createInputTensor(int count, at::ScalarType dtype) {
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
std::vector<at::Tensor> AllGatherTest::createOutputTensors(
    int count,
    at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  std::vector<at::Tensor> outputs(num_ranks_);
  for (int i = 0; i < num_ranks_; i++) {
    outputs[i] = at::zeros({count}, options);
  }
  return outputs;
}

// Helper function to verify results
void AllGatherTest::verifyResults(const std::vector<at::Tensor>& outputs) {
  for (int i = 0; i < num_ranks_; i++) {
    // Use verifyTensorEquality to compare output with expected tensor
    std::string description = "rank " + std::to_string(i) + " tensor";
    verifyTensorEquality(outputs[i].cpu(), i + 1, description);
  }
}
