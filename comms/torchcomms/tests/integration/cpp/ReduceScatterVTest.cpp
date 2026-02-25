// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "ReduceScatterVTest.hpp"

#include <gtest/gtest.h>
#include <memory>
#include "TorchCommTestHelpers.h"

// Test function for synchronous reduce_scatter_v with work object
template <typename Fixture>
void ReduceScatterVTest<Fixture>::testSync(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  auto counts = getCounts(count);
  std::vector<at::Tensor> input_tensors = createInputTensors(counts, dtype);
  at::Tensor output = createOutputTensor(counts[rank_], dtype);
  auto original_output = output.clone();

  auto execute = [&]() {
    auto work = torchcomm_->reduce_scatter_v(output, input_tensors, op, false);
    work->wait();
  };
  auto reset = [&]() { output.copy_(original_output); };
  auto verify = [&]() { verifyResults(output, op); };
  run(execute, reset, verify);
}

// Test function for synchronous reduce_scatter_v without work object
template <typename Fixture>
void ReduceScatterVTest<Fixture>::testSyncNoWork(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  auto counts = getCounts(count);
  std::vector<at::Tensor> input_tensors = createInputTensors(counts, dtype);
  at::Tensor output = createOutputTensor(counts[rank_], dtype);
  auto original_output = output.clone();

  auto execute = [&]() {
    torchcomm_->reduce_scatter_v(output, input_tensors, op, false);
  };
  auto reset = [&]() { output.copy_(original_output); };
  auto verify = [&]() { verifyResults(output, op); };
  run(execute, reset, verify);
}

// Test function for asynchronous reduce_scatter_v with wait
template <typename Fixture>
void ReduceScatterVTest<Fixture>::testAsync(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  auto counts = getCounts(count);
  std::vector<at::Tensor> input_tensors = createInputTensors(counts, dtype);
  at::Tensor output = createOutputTensor(counts[rank_], dtype);
  auto original_output = output.clone();

  auto execute = [&]() {
    auto work = torchcomm_->reduce_scatter_v(output, input_tensors, op, true);
    work->wait();
  };
  auto reset = [&]() { output.copy_(original_output); };
  auto verify = [&]() { verifyResults(output, op); };
  run(execute, reset, verify);
}

// Test function for asynchronous reduce_scatter_v with early reset
template <typename Fixture>
void ReduceScatterVTest<Fixture>::testAsyncEarlyReset(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  auto counts = getCounts(count);
  std::vector<at::Tensor> input_tensors = createInputTensors(counts, dtype);
  at::Tensor output = createOutputTensor(counts[rank_], dtype);
  auto original_output = output.clone();

  auto execute = [&]() {
    auto work = torchcomm_->reduce_scatter_v(output, input_tensors, op, true);
    work->wait();
    work.reset();
  };
  auto reset = [&]() { output.copy_(original_output); };
  auto verify = [&]() { verifyResults(output, op); };
  run(execute, reset, verify);
}

// Test function for asynchronous reduce_scatter_v with input deleted after
// enqueue
template <typename Fixture>
void ReduceScatterVTest<Fixture>::testInputDeleted(
    int count,
    at::ScalarType dtype,
    const torch::comms::ReduceOp& op) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count << " dtype="
                           << getDtypeName(dtype) << " op=" << getOpName(op));

  auto counts = getCounts(count);
  at::Tensor output = createOutputTensor(counts[rank_], dtype);
  auto original_output = output.clone();
  auto input_tensors = std::make_shared<std::vector<at::Tensor>>(
      createInputTensors(counts, dtype));

  auto execute = [&]() {
    torchcomm_->reduce_scatter_v(output, *input_tensors, op, false);
  };
  auto reset = [&]() { output.copy_(original_output); };
  auto verify = [&]() { verifyResults(output, op); };
  auto cleanup = [&]() { input_tensors.reset(); };
  run(execute, reset, verify, cleanup);
}

// Helper function to get per-rank counts for reduce_scatter_v (variable sizes)
template <typename Fixture>
std::vector<int> ReduceScatterVTest<Fixture>::getCounts(int count) {
  std::vector<int> counts(num_ranks_);
  for (int i = 0; i < num_ranks_; i++) {
    counts[i] = count + i;
  }
  return counts;
}

// Helper function to create input tensors
template <typename Fixture>
std::vector<at::Tensor> ReduceScatterVTest<Fixture>::createInputTensors(
    const std::vector<int>& counts,
    at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  size_t numRanks = counts.size();
  std::vector<at::Tensor> input_tensors;
  input_tensors.reserve(numRanks);

  for (size_t r = 0; r < numRanks; r++) {
    // Each tensor has rank-specific values
    at::Tensor tensor;
    if (dtype == at::kFloat || dtype == at::kBFloat16 || dtype == at::kHalf ||
        dtype == at::kDouble) {
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
template <typename Fixture>
at::Tensor ReduceScatterVTest<Fixture>::createOutputTensor(
    int count,
    at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  return at::zeros({count}, options);
}

// Helper function to calculate expected result
template <typename Fixture>
int ReduceScatterVTest<Fixture>::calculateExpectedResult(
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
template <typename Fixture>
void ReduceScatterVTest<Fixture>::verifyResults(
    const at::Tensor& output,
    const torch::comms::ReduceOp& op) {
  // Calculate expected result
  int expected = calculateExpectedResult(op);

  // Use verifyTensorEquality to compare output with expected tensor
  std::string description = "reduce_scatter_v with op " + getOpName(op);
  verifyTensorEquality(output.cpu(), expected, description);
}

template class ReduceScatterVTest<EagerTestFixture<ReduceScatterVParams>>;
template class ReduceScatterVTest<GraphTestFixture<ReduceScatterVParams, 1>>;
template class ReduceScatterVTest<GraphTestFixture<ReduceScatterVParams, 2>>;
