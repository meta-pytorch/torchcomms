// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "ProfilerTest.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <gtest/gtest.h>
#include <json/reader.h>
#include <json/value.h>
#include <torch/csrc/autograd/profiler_kineto.h> // @manual=//caffe2:torch-cpp-cpu
#include <filesystem>
#include <fstream>
#include <vector>

void ProfilerTest::SetUp() {
  // Create CUDA device
  c10::Device device = c10::Device(device_type_);

  // Create TorchComm with options (but no store)
  const char* test_backend_env = std::getenv("TEST_BACKEND");
  if (!test_backend_env) {
    throw std::runtime_error("TEST_BACKEND environment variable is not set");
  }
  std::string backend = test_backend_env;

  torch::comms::CommOptions options;
  torchcomm_ =
      torch::comms::new_comm(backend, device, "comms_test_name", options);
}

void ProfilerTest::TearDown() {
  // Reset the TorchComm object to ensure proper cleanup
  if (torchcomm_) {
    torchcomm_.reset();
  }
}

Json::Value ProfilerTest::readTraceFile(std::filesystem::path& trace_file) {
  Json::Value result;
  std::ifstream people_file(trace_file, std::ifstream::binary);
  people_file >> result;
  return result;
}

void ProfilerTest::sanityCheckProfilerMeta(
    Json::Value& json_value,
    std::map<std::string, std::vector<Json::Value>>& events) {
  ASSERT_GT(json_value["traceEvents"].size(), 1u);

  for (auto& event : json_value["traceEvents"]) {
    if (event["name"] != "record_param_comms") {
      continue;
    }

    auto& args = event["args"];
    auto coll_name = args["Collective name"].asString();
    ASSERT_NE(coll_name, "");
    ASSERT_NE(args["dtype"], "");

    if (events.find(coll_name) == events.end()) {
      events[coll_name] = std::vector<Json::Value>();
    }
    events[coll_name].push_back(args);

    ASSERT_EQ(args["Process Group Name"], "comms_test_name");
    ASSERT_NE(args["Process Group Ranks"], "");

    ASSERT_GE(args["In msg nelems"], 0);
    ASSERT_GE(args["Out msg nelems"], 0);
    ASSERT_GE(args["Group size"], 0);
  }
}

std::shared_ptr<torch::comms::TorchWork>
ProfilerTest::runAllCollectiveOperations() {
  auto options = at::TensorOptions().dtype(kTensorDtype).device(device_type_);

  // Prepare multiple tensors for all the operations to be tested.
  auto send_tensor = at::ones(kTensorCount, options) * float(rank_ + 1);
  auto recv_tensor = at::zeros(kTensorCount, options);
  std::vector<at::Tensor> tensors_all_ranks;
  tensors_all_ranks.reserve(num_ranks_);
  for (int i = 0; i < num_ranks_; i++) {
    tensors_all_ranks.push_back(at::zeros(kTensorCount, options));
  }
  std::vector<at::Tensor> input_tensors;
  input_tensors.reserve(num_ranks_);
  for (int i = 0; i < num_ranks_; i++) {
    input_tensors.push_back(at::zeros(kTensorCount, options));
  }
  auto recv_tensor_single = at::zeros(kTensorCount * num_ranks_, options);
  auto send_tensor_single = at::zeros(kTensorCount * num_ranks_, options);

  // Create split sizes for all_to_all_v_single
  std::vector<uint64_t> input_split_sizes(num_ranks_, kTensorCount);
  std::vector<uint64_t> output_split_sizes(num_ranks_, kTensorCount);

  int send_rank = (rank_ + 1) % num_ranks_;
  int recv_rank = (rank_ + num_ranks_ - 1) % num_ranks_;

  if (rank_ % 2 == 0) {
    // Even ranks: send first, then receive
    torchcomm_->send(send_tensor, send_rank, false);
    torchcomm_->recv(recv_tensor, recv_rank, false);
  } else {
    // Odd ranks: receive first, then send
    torchcomm_->recv(recv_tensor, recv_rank, false);
    torchcomm_->send(send_tensor, send_rank, false);
  }

  torchcomm_->all_reduce(send_tensor, torch::comms::ReduceOp::SUM, false);
  torchcomm_->reduce(send_tensor, 0, torch::comms::ReduceOp::SUM, false);

  torchcomm_->all_gather_single(recv_tensor_single, send_tensor, false);
  torchcomm_->all_gather(tensors_all_ranks, send_tensor, false);
  torchcomm_->gather(tensors_all_ranks, send_tensor, 0, false);

  torchcomm_->reduce_scatter_single(
      recv_tensor, send_tensor_single, torch::comms::ReduceOp::SUM, false);
  torchcomm_->reduce_scatter(
      recv_tensor, tensors_all_ranks, torch::comms::ReduceOp::SUM, false);
  torchcomm_->scatter(recv_tensor, tensors_all_ranks, 0, false);

  torchcomm_->all_to_all(tensors_all_ranks, input_tensors, false);
  torchcomm_->all_to_all_single(recv_tensor_single, send_tensor_single, false);
  torchcomm_->all_to_all_v_single(
      recv_tensor_single,
      send_tensor_single,
      output_split_sizes,
      input_split_sizes,
      false);

  torchcomm_->broadcast(send_tensor, 0, false);

  return torchcomm_->barrier(false);
}
