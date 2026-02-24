// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "AllToAllvSingleTest.hpp"

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include "TorchCommTestHelpers.h"

// Test function for synchronous all_to_all_v_single with work object
template <typename Fixture>
void AllToAllvSingleTest<Fixture>::testSync(
    AllToAllvSizePattern pattern,
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  auto split_sizes = computeSplitSizes(pattern, count);
  std::vector<uint64_t> input_split_sizes = split_sizes.first;
  std::vector<uint64_t> output_split_sizes = split_sizes.second;
  at::Tensor input = createInputTensor(input_split_sizes, dtype);
  at::Tensor output = createOutputTensor(output_split_sizes, dtype);

  auto original_output = output.clone();

  auto execute = [&]() {
    auto work = torchcomm_->all_to_all_v_single(
        output, input, output_split_sizes, input_split_sizes, false);
    work->wait();
  };
  auto reset = [&]() { output.copy_(original_output); };
  auto verify = [&]() { verifyResults(output, output_split_sizes); };
  run(execute, reset, verify);
}

// Test function for synchronous all_to_all_v_single without work object
template <typename Fixture>
void AllToAllvSingleTest<Fixture>::testSyncNoWork(
    AllToAllvSizePattern pattern,
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  auto split_sizes = computeSplitSizes(pattern, count);
  std::vector<uint64_t> input_split_sizes = split_sizes.first;
  std::vector<uint64_t> output_split_sizes = split_sizes.second;
  at::Tensor input = createInputTensor(input_split_sizes, dtype);
  at::Tensor output = createOutputTensor(output_split_sizes, dtype);

  auto original_output = output.clone();

  auto execute = [&]() {
    torchcomm_->all_to_all_v_single(
        output, input, output_split_sizes, input_split_sizes, false);
  };
  auto reset = [&]() { output.copy_(original_output); };
  auto verify = [&]() { verifyResults(output, output_split_sizes); };
  run(execute, reset, verify);
}

// Test function for asynchronous all_to_all_v_single with wait
template <typename Fixture>
void AllToAllvSingleTest<Fixture>::testAsync(
    AllToAllvSizePattern pattern,
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  auto split_sizes = computeSplitSizes(pattern, count);
  std::vector<uint64_t> input_split_sizes = split_sizes.first;
  std::vector<uint64_t> output_split_sizes = split_sizes.second;
  at::Tensor input = createInputTensor(input_split_sizes, dtype);
  at::Tensor output = createOutputTensor(output_split_sizes, dtype);

  auto original_output = output.clone();

  auto execute = [&]() {
    auto work = torchcomm_->all_to_all_v_single(
        output, input, output_split_sizes, input_split_sizes, true);
    work->wait();
  };
  auto reset = [&]() { output.copy_(original_output); };
  auto verify = [&]() { verifyResults(output, output_split_sizes); };
  run(execute, reset, verify);
}

// Test function for asynchronous all_to_all_v_single with early reset
template <typename Fixture>
void AllToAllvSingleTest<Fixture>::testAsyncEarlyReset(
    AllToAllvSizePattern pattern,
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  auto split_sizes = computeSplitSizes(pattern, count);
  std::vector<uint64_t> input_split_sizes = split_sizes.first;
  std::vector<uint64_t> output_split_sizes = split_sizes.second;
  at::Tensor input = createInputTensor(input_split_sizes, dtype);
  at::Tensor output = createOutputTensor(output_split_sizes, dtype);

  auto original_output = output.clone();

  auto execute = [&]() {
    auto work = torchcomm_->all_to_all_v_single(
        output, input, output_split_sizes, input_split_sizes, true);
    work->wait();
    work.reset();
  };
  auto reset = [&]() { output.copy_(original_output); };
  auto verify = [&]() { verifyResults(output, output_split_sizes); };
  run(execute, reset, verify);
}

// Test function for asynchronous all_to_all_v_single with input deleted after
// enqueue
template <typename Fixture>
void AllToAllvSingleTest<Fixture>::testInputDeleted(
    AllToAllvSizePattern pattern,
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  auto split_sizes = computeSplitSizes(pattern, count);
  std::vector<uint64_t> input_split_sizes = split_sizes.first;
  std::vector<uint64_t> output_split_sizes = split_sizes.second;
  auto input =
      std::make_shared<at::Tensor>(createInputTensor(input_split_sizes, dtype));
  at::Tensor output = createOutputTensor(output_split_sizes, dtype);

  auto original_output = output.clone();

  auto execute = [&]() {
    torchcomm_->all_to_all_v_single(
        output, *input, output_split_sizes, input_split_sizes, false);
  };
  auto reset = [&]() { output.copy_(original_output); };
  auto verify = [&]() { verifyResults(output, output_split_sizes); };
  auto cleanup = [&]() { input.reset(); };
  run(execute, reset, verify, cleanup);
}

// Test function for synchronous all_to_all_v_single with work object
template <typename Fixture>
void AllToAllvSingleTest<Fixture>::testMultiDimTensor(
    AllToAllvSizePattern pattern,
    int count,
    at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  auto split_sizes = computeSplitSizes(pattern, count);
  const auto& input_split_sizes = split_sizes.first;
  std::vector<uint64_t> output_split_sizes = split_sizes.second;

  at::Tensor input = createInputTensor(input_split_sizes, dtype);
  at::Tensor output = createOutputTensor(output_split_sizes, dtype);

  // Reshape tensors to 2D
  input = input.reshape({input.numel() / 2, 2});
  output = output.reshape({output.numel() / 2, 2});

  // Halve split sizes for the 2D tensor
  auto halved_input_sizes = input_split_sizes;
  auto halved_output_sizes = output_split_sizes;
  for (auto& s : halved_input_sizes) {
    if (s % 2 != 0) {
      return;
    }
    s /= 2;
  }
  for (auto& s : halved_output_sizes) {
    if (s % 2 != 0) {
      return;
    }
    s /= 2;
  }

  auto original_output = output.clone();

  auto execute = [&]() {
    auto work = torchcomm_->all_to_all_v_single(
        output, input, halved_output_sizes, halved_input_sizes, false);
    work->wait();
  };
  auto reset = [&]() { output.copy_(original_output); };
  auto verify = [&]() {
    auto flat_output = output.reshape({output.numel()});
    verifyResults(flat_output, output_split_sizes);
  };
  run(execute, reset, verify);
}

template <typename Fixture>
std::string AllToAllvSingleTest<Fixture>::getPatternName(
    AllToAllvSizePattern pattern) {
  switch (pattern) {
    case AllToAllvSizePattern::Uniform:
      return "Uniform";
    case AllToAllvSizePattern::Variable:
      return "Variable";
    case AllToAllvSizePattern::ZeroSizes:
      return "ZeroSizes";
    case AllToAllvSizePattern::AllZero:
      return "AllZero";
    case AllToAllvSizePattern::Asymmetric:
      return "Asymmetric";
  }
  return "Unknown";
}

template <typename Fixture>
std::pair<std::vector<uint64_t>, std::vector<uint64_t>> AllToAllvSingleTest<
    Fixture>::computeSplitSizes(AllToAllvSizePattern pattern, int count) {
  std::vector<uint64_t> inputSizes, outputSizes;
  switch (pattern) {
    case AllToAllvSizePattern::Uniform:
      inputSizes.assign(num_ranks_, count);
      outputSizes.assign(num_ranks_, count);
      break;
    case AllToAllvSizePattern::Variable:
      for (int i = 0; i < num_ranks_; i++) {
        inputSizes.push_back(static_cast<uint64_t>((rank_ + 1) * count));
        outputSizes.push_back(static_cast<uint64_t>((i + 1) * count));
      }
      break;
    case AllToAllvSizePattern::ZeroSizes:
      for (int i = 0; i < num_ranks_; i++) {
        uint64_t sendSize = (rank_ + i) % 3 == 0 ? 0 : count;
        uint64_t recvSize = (i + rank_) % 3 == 0 ? 0 : count;
        inputSizes.push_back(sendSize);
        outputSizes.push_back(recvSize);
      }
      break;
    case AllToAllvSizePattern::AllZero:
      inputSizes.assign(num_ranks_, 0);
      outputSizes.assign(num_ranks_, 0);
      break;
    case AllToAllvSizePattern::Asymmetric:
      for (int i = 0; i < num_ranks_; i++) {
        inputSizes.push_back(rank_ % 2 == 0 ? count : 0);
        outputSizes.push_back(i % 2 == 0 ? count : 0);
      }
      break;
  }
  return {inputSizes, outputSizes};
}

// Helper function to create input tensor
template <typename Fixture>
at::Tensor AllToAllvSingleTest<Fixture>::createInputTensor(
    const std::vector<uint64_t>& input_split_sizes,
    at::ScalarType dtype) {
  uint64_t total_size = 0;
  for (uint64_t s : input_split_sizes) {
    total_size += s;
  }
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  at::Tensor input = at::zeros({static_cast<int64_t>(total_size)}, options);

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
template <typename Fixture>
at::Tensor AllToAllvSingleTest<Fixture>::createOutputTensor(
    const std::vector<uint64_t>& output_split_sizes,
    at::ScalarType dtype) {
  uint64_t total_size = 0;
  for (uint64_t s : output_split_sizes) {
    total_size += s;
  }
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  return at::zeros({static_cast<int64_t>(total_size)}, options);
}

// Helper function to verify results
template <typename Fixture>
void AllToAllvSingleTest<Fixture>::verifyResults(
    const at::Tensor& output,
    const std::vector<uint64_t>& output_split_sizes) {
  uint64_t offset = 0;
  for (int i = 0; i < num_ranks_; i++) {
    if (output_split_sizes[i] > 0) {
      at::Tensor section =
          output.slice(0, offset, offset + output_split_sizes[i]);
      int expected_value = i * 100 + rank_ + 1;
      if (output.dtype() == at::kChar) {
        expected_value = (i * 10 + rank_ + 1) % 128;
      }
      verifyTensorEquality(section.cpu(), expected_value);
    }
    offset += output_split_sizes[i];
  }
}

template class AllToAllvSingleTest<EagerTestFixture<AllToAllvSingleParams>>;
template class AllToAllvSingleTest<GraphTestFixture<AllToAllvSingleParams, 1>>;
template class AllToAllvSingleTest<GraphTestFixture<AllToAllvSingleParams, 2>>;
