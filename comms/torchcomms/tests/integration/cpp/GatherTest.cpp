// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "GatherTest.hpp"

#include <gtest/gtest.h>
#include <memory>
#include "TorchCommTestHelpers.h"

// Test function for synchronous gather with work object
template <typename Fixture>
void GatherTest<Fixture>::testSync(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  const int root_rank = 0;
  at::Tensor input = createInputTensor(count, dtype);
  std::vector<at::Tensor> outputs =
      createOutputTensors(root_rank, count, dtype);

  std::vector<at::Tensor> original_outputs;
  original_outputs.reserve(outputs.size());
  for (const auto& output : outputs) {
    original_outputs.push_back(output.clone());
  }

  auto execute = [&]() {
    auto work = torchcomm_->gather(outputs, input, root_rank, false);
    work->wait();
  };
  auto reset = [&]() {
    for (size_t i = 0; i < outputs.size(); i++) {
      outputs[i].copy_(original_outputs[i]);
    }
  };
  auto verify = [&]() {
    if (rank_ == root_rank) {
      verifyResults(outputs);
    } else {
      synchronizeStream();
    }
  };
  run(execute, reset, verify);
}

// Test function for synchronous gather without work object
template <typename Fixture>
void GatherTest<Fixture>::testSyncNoWork(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  const int root_rank = 0;
  at::Tensor input = createInputTensor(count, dtype);
  std::vector<at::Tensor> outputs =
      createOutputTensors(root_rank, count, dtype);

  std::vector<at::Tensor> original_outputs;
  original_outputs.reserve(outputs.size());
  for (const auto& output : outputs) {
    original_outputs.push_back(output.clone());
  }

  auto execute = [&]() {
    torchcomm_->gather(outputs, input, root_rank, false);
  };
  auto reset = [&]() {
    for (size_t i = 0; i < outputs.size(); i++) {
      outputs[i].copy_(original_outputs[i]);
    }
  };
  auto verify = [&]() {
    if (rank_ == root_rank) {
      verifyResults(outputs);
    } else {
      synchronizeStream();
    }
  };
  run(execute, reset, verify);
}

// Test function for asynchronous gather with wait
template <typename Fixture>
void GatherTest<Fixture>::testAsync(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  const int root_rank = 0;
  at::Tensor input = createInputTensor(count, dtype);
  std::vector<at::Tensor> outputs =
      createOutputTensors(root_rank, count, dtype);

  std::vector<at::Tensor> original_outputs;
  original_outputs.reserve(outputs.size());
  for (const auto& output : outputs) {
    original_outputs.push_back(output.clone());
  }

  auto execute = [&]() {
    auto work = torchcomm_->gather(outputs, input, root_rank, true);
    work->wait();
  };
  auto reset = [&]() {
    for (size_t i = 0; i < outputs.size(); i++) {
      outputs[i].copy_(original_outputs[i]);
    }
  };
  auto verify = [&]() {
    if (rank_ == root_rank) {
      verifyResults(outputs);
    } else {
      synchronizeStream();
    }
  };
  run(execute, reset, verify);
}

// Test function for asynchronous gather with early reset
template <typename Fixture>
void GatherTest<Fixture>::testAsyncEarlyReset(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  const int root_rank = 0;
  at::Tensor input = createInputTensor(count, dtype);
  std::vector<at::Tensor> outputs =
      createOutputTensors(root_rank, count, dtype);

  std::vector<at::Tensor> original_outputs;
  original_outputs.reserve(outputs.size());
  for (const auto& output : outputs) {
    original_outputs.push_back(output.clone());
  }

  auto execute = [&]() {
    auto work = torchcomm_->gather(outputs, input, root_rank, true);
    work->wait();
    work.reset();
  };
  auto reset = [&]() {
    for (size_t i = 0; i < outputs.size(); i++) {
      outputs[i].copy_(original_outputs[i]);
    }
  };
  auto verify = [&]() {
    if (rank_ == root_rank) {
      verifyResults(outputs);
    } else {
      synchronizeStream();
    }
  };
  run(execute, reset, verify);
}

// Test function for asynchronous gather with input deleted after enqueue
template <typename Fixture>
void GatherTest<Fixture>::testInputDeleted(int count, at::ScalarType dtype) {
  SCOPED_TRACE(
      ::testing::Message() << "count=" << count
                           << " dtype=" << getDtypeName(dtype));

  const int root_rank = 0;
  auto input = std::make_shared<at::Tensor>(createInputTensor(count, dtype));
  std::vector<at::Tensor> outputs =
      createOutputTensors(root_rank, count, dtype);

  std::vector<at::Tensor> original_outputs;
  original_outputs.reserve(outputs.size());
  for (const auto& output : outputs) {
    original_outputs.push_back(output.clone());
  }

  auto execute = [&]() {
    torchcomm_->gather(outputs, *input, root_rank, false);
  };
  auto reset = [&]() {
    for (size_t i = 0; i < outputs.size(); i++) {
      outputs[i].copy_(original_outputs[i]);
    }
  };
  auto verify = [&]() {
    if (rank_ == root_rank) {
      verifyResults(outputs);
    } else {
      synchronizeStream();
    }
  };
  auto cleanup = [&]() { input.reset(); };
  run(execute, reset, verify, cleanup);
}

// Helper function to create input tensor
template <typename Fixture>
at::Tensor GatherTest<Fixture>::createInputTensor(
    int count,
    at::ScalarType dtype) {
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
template <typename Fixture>
std::vector<at::Tensor> GatherTest<Fixture>::createOutputTensors(
    int root_rank,
    int count,
    at::ScalarType dtype) {
  auto options = at::TensorOptions().dtype(dtype).device(device_type_);
  std::vector<at::Tensor> outputs;
  if (rank_ == root_rank) {
    outputs.reserve(num_ranks_);
    for (int i = 0; i < num_ranks_; i++) {
      outputs.push_back(at::zeros({count}, options));
    }
  }
  return outputs;
}

// Helper function to verify results
template <typename Fixture>
void GatherTest<Fixture>::verifyResults(
    const std::vector<at::Tensor>& outputs) {
  for (int i = 0; i < num_ranks_; i++) {
    std::string description = "gather rank " + std::to_string(i) + " tensor";
    verifyTensorEquality(outputs[i].cpu(), i + 1, description);
  }
}

template <typename Fixture>
void GatherTest<Fixture>::synchronizeStream() {
  if (device_type_ == c10::DeviceType::CUDA) {
    at::cuda::getCurrentCUDAStream(0).synchronize();
  }
}

template class GatherTest<EagerTestFixture<GatherParams>>;
template class GatherTest<GraphTestFixture<GatherParams, 1>>;
template class GatherTest<GraphTestFixture<GatherParams, 2>>;
