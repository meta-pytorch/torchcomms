// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "AllToAllvSingleTest.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <gtest/gtest.h>
#include <vector>
#include "TorchCommTestHelpers.h"

std::unique_ptr<TorchCommTestWrapper> AllToAllvSingleTest::createWrapper() {
  return std::make_unique<TorchCommTestWrapper>();
}

void AllToAllvSingleTest::SetUp() {
  wrapper_ = createWrapper();
  torchcomm_ = wrapper_->getTorchComm();
  rank_ = torchcomm_->getRank();
  num_ranks_ = torchcomm_->getSize();
}

void AllToAllvSingleTest::TearDown() {
  // Explicitly reset the TorchComm object to ensure proper cleanup
  torchcomm_.reset();
  wrapper_.reset();
}

// Test function for synchronous all_to_all_v_single with work object
void AllToAllvSingleTest::testSyncAllToAllvSingle(
    const std::vector<uint64_t>& input_split_sizes,
    const std::vector<uint64_t>& output_split_sizes,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing sync all_to_all_v_single with dtype="
                           << getDtypeName(dtype));

  // Create input and output tensors
  at::Tensor input = createInputTensor(input_split_sizes, dtype);
  at::Tensor output = createOutputTensor(output_split_sizes, dtype);

  // Call all_to_all_v_single
  auto work = torchcomm_->all_to_all_v_single(
      output, input, output_split_sizes, input_split_sizes, false);
  work->wait();

  // Verify the results
  verifyResults(output, output_split_sizes);
}

// Test function for synchronous all_to_all_v_single without work object
void AllToAllvSingleTest::testSyncAllToAllvSingleNoWork(
    const std::vector<uint64_t>& input_split_sizes,
    const std::vector<uint64_t>& output_split_sizes,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing sync all_to_all_v_single without work object with dtype="
      << getDtypeName(dtype));

  // Create input and output tensors
  at::Tensor input = createInputTensor(input_split_sizes, dtype);
  at::Tensor output = createOutputTensor(output_split_sizes, dtype);

  // Call all_to_all_v_single without keeping the work object
  torchcomm_->all_to_all_v_single(
      output, input, output_split_sizes, input_split_sizes, false);

  // Verify the results
  verifyResults(output, output_split_sizes);
}

// Test function for asynchronous all_to_all_v_single with wait
void AllToAllvSingleTest::testAsyncAllToAllvSingle(
    const std::vector<uint64_t>& input_split_sizes,
    const std::vector<uint64_t>& output_split_sizes,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing async all_to_all_v_single with dtype="
                           << getDtypeName(dtype));

  // Create input and output tensors
  at::Tensor input = createInputTensor(input_split_sizes, dtype);
  at::Tensor output = createOutputTensor(output_split_sizes, dtype);

  // Call all_to_all_v_single
  auto work = torchcomm_->all_to_all_v_single(
      output, input, output_split_sizes, input_split_sizes, true);

  // Wait for the all_to_all_v_single to complete
  work->wait();

  // Verify the results
  verifyResults(output, output_split_sizes);
}

// Test function for asynchronous all_to_all_v_single with early reset
void AllToAllvSingleTest::testAsyncAllToAllvSingleEarlyReset(
    const std::vector<uint64_t>& input_split_sizes,
    const std::vector<uint64_t>& output_split_sizes,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing async all_to_all_v_single with early reset with dtype="
      << getDtypeName(dtype));

  // Create input and output tensors
  at::Tensor input = createInputTensor(input_split_sizes, dtype);
  at::Tensor output = createOutputTensor(output_split_sizes, dtype);

  // Call all_to_all_v_single
  auto work = torchcomm_->all_to_all_v_single(
      output, input, output_split_sizes, input_split_sizes, true);

  // Wait for the work to complete before resetting
  work->wait();

  // Reset the work object
  work.reset();

  // Verify the results
  verifyResults(output, output_split_sizes);
}

// Test function for asynchronous all_to_all_v_single with input deleted after
// enqueue
void AllToAllvSingleTest::testAllToAllvSingleInputDeleted(
    const std::vector<uint64_t>& input_split_sizes,
    const std::vector<uint64_t>& output_split_sizes,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing async all_to_all_v_single with input deleted after enqueue with dtype="
      << getDtypeName(dtype));

  // Create output tensor that persists throughout the test
  at::Tensor output = createOutputTensor(output_split_sizes, dtype);

  {
    // Create input tensor in a limited scope
    at::Tensor input = createInputTensor(input_split_sizes, dtype);

    // Call all_to_all_v_single
    torchcomm_->all_to_all_v_single(
        output, input, output_split_sizes, input_split_sizes, false);

    // Input tensor goes out of scope here and gets deleted
  }

  // Verify the results
  verifyResults(output, output_split_sizes);
}

// CUDA Graph test function for all_to_all_v_single
void AllToAllvSingleTest::testGraphAllToAllvSingle(
    const std::vector<uint64_t>& input_split_sizes,
    const std::vector<uint64_t>& output_split_sizes,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing CUDA Graph all_to_all_v_single with dtype="
      << getDtypeName(dtype));

  // Create input and output tensors BEFORE graph capture
  at::Tensor input = createInputTensor(input_split_sizes, dtype);
  at::Tensor output = createOutputTensor(output_split_sizes, dtype);
  at::Tensor original_output = output.clone();

  // Create a non-default CUDA stream (required for CUDA graph capture)
  at::cuda::CUDAStream stream = at::cuda::getStreamFromPool();

  // Set the stream as current for graph capture
  at::cuda::CUDAStreamGuard guard(stream);

  // Create PyTorch CUDA graph
  at::cuda::CUDAGraph graph;

  // Capture the all_to_all_v_single operation in the graph
  graph.capture_begin();

  // Call all_to_all_v_single without keeping the work object
  torchcomm_->all_to_all_v_single(
      output, input, output_split_sizes, input_split_sizes, false);

  graph.capture_end();

  // Replay the captured graph multiple times
  for (int i = 0; i < num_replays; ++i) {
    // Reset output buffer before each replay
    output.copy_(original_output);

    graph.replay();

    // Verify the results after each replay
    verifyResults(output, output_split_sizes);
  }
}

// CUDA Graph test function for all_to_all_v_single with input deleted after
// graph creation
void AllToAllvSingleTest::testGraphAllToAllvSingleInputDeleted(
    const std::vector<uint64_t>& input_split_sizes,
    const std::vector<uint64_t>& output_split_sizes,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing CUDA Graph all_to_all_v_single with input deleted after graph creation with dtype="
      << getDtypeName(dtype));

  // Create output tensor that persists throughout the test
  at::Tensor output = createOutputTensor(output_split_sizes, dtype);
  at::Tensor original_output = output.clone();

  // Create a non-default CUDA stream (required for CUDA graph capture)
  at::cuda::CUDAStream stream = at::cuda::getStreamFromPool();

  // Set the stream as current for graph capture
  at::cuda::CUDAStreamGuard guard(stream);

  // Create PyTorch CUDA graph
  at::cuda::CUDAGraph graph;

  {
    // Create input tensor in a limited scope
    at::Tensor input = createInputTensor(input_split_sizes, dtype);

    // Capture the all_to_all_v_single operation in the graph
    graph.capture_begin();

    // Call all_to_all_v_single without keeping the work object
    torchcomm_->all_to_all_v_single(
        output, input, output_split_sizes, input_split_sizes, false);

    graph.capture_end();

    // Input tensor goes out of scope here and gets deleted
  }

  // Replay the captured graph multiple times even though input is deleted
  for (int i = 0; i < num_replays; ++i) {
    // Reset output buffer before each replay
    output.copy_(original_output);

    graph.replay();

    // Verify the results after each replay
    verifyResults(output, output_split_sizes);
  }
}

// Test function for synchronous all_to_all_v_single with work object
void AllToAllvSingleTest::testSyncAllToAllvSingleMultiDimTensor(
    const std::vector<uint64_t>& input_split_sizes_,
    const std::vector<uint64_t>& output_split_sizes_,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing sync all_to_all_v_single with dtype="
                           << getDtypeName(dtype));

  std::vector<uint64_t> input_split_sizes = input_split_sizes_;
  std::vector<uint64_t> output_split_sizes = output_split_sizes_;

  // Create input and output tensors
  at::Tensor input = createInputTensor(input_split_sizes, dtype);
  input = input.reshape({input.numel() / 2, 2});
  at::Tensor output = createOutputTensor(output_split_sizes, dtype);
  output = output.reshape({output.numel() / 2, 2});

  // Reduce each value in input_split_sizes and output_split_sizes by half
  for (size_t i = 0; i < input_split_sizes.size(); ++i) {
    if (input_split_sizes[i] % 2 != 0) {
      std::cout << "Input size must be divisible by 2 for multi-dim tensor test"
                << std::endl;
      return;
    } else {
      input_split_sizes[i] /= 2;
    }
  }
  for (size_t i = 0; i < output_split_sizes.size(); ++i) {
    if (output_split_sizes[i] % 2 != 0) {
      std::cout
          << "Output size must be divisible by 2 for multi-dim tensor test"
          << std::endl;
      return;
    } else {
      output_split_sizes[i] /= 2;
    }
  }

  // Call all_to_all_v_single
  auto work = torchcomm_->all_to_all_v_single(
      output, input, output_split_sizes, input_split_sizes, false);
  work->wait();

  output = output.reshape({output.numel()});

  // Verify the results with the original output_split_sizes
  verifyResults(output, output_split_sizes_);
}

// Helper function to create input tensor
at::Tensor AllToAllvSingleTest::createInputTensor(
    const std::vector<uint64_t>& input_split_sizes,
    at::ScalarType dtype) {
  uint64_t total_size = 0;
  for (uint64_t size : input_split_sizes) {
    total_size += size;
  }

  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  at::Tensor input = at::zeros({static_cast<int64_t>(total_size)}, options);

  // Fill each section with rank-specific values
  uint64_t offset = 0;
  for (int i = 0; i < num_ranks_; i++) {
    if (input_split_sizes[i] > 0) {
      at::Tensor section =
          input.slice(0, offset, offset + input_split_sizes[i]);
      if (dtype == at::kFloat || dtype == at::kHalf || dtype == at::kBFloat16) {
        section.fill_(static_cast<float>(rank_ * 100 + i + 1));
      } else if (dtype == at::kInt) {
        section.fill_(static_cast<int>(rank_ * 100 + i + 1));
      } else if (dtype == at::kChar) {
        section.fill_(static_cast<signed char>((rank_ * 10 + i + 1) % 128));
      }
    }
    offset += input_split_sizes[i];
  }
  return input;
}

// Helper function to create output tensor
at::Tensor AllToAllvSingleTest::createOutputTensor(
    const std::vector<uint64_t>& output_split_sizes,
    at::ScalarType dtype) {
  uint64_t total_size = 0;
  for (uint64_t size : output_split_sizes) {
    total_size += size;
  }

  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  return at::zeros({static_cast<int64_t>(total_size)}, options);
}

// Helper function to verify results
void AllToAllvSingleTest::verifyResults(
    const at::Tensor& output,
    const std::vector<uint64_t>& output_split_sizes) {
  uint64_t offset = 0;
  for (int i = 0; i < num_ranks_; i++) {
    if (output_split_sizes[i] > 0) {
      // For each rank's section in the output tensor
      at::Tensor section =
          output.slice(0, offset, offset + output_split_sizes[i]);

      // Expected value: what rank i would have sent to this rank
      int expected_value = i * 100 + rank_ + 1;
      if (output.dtype() == at::kChar) {
        expected_value = (i * 10 + rank_ + 1) % 128;
      }

      // Use verifyTensorEquality to compare section with expected value
      std::string description = "rank " + std::to_string(i) + " section";
      verifyTensorEquality(section.cpu(), expected_value, description);
    }
    offset += output_split_sizes[i];
  }
}
