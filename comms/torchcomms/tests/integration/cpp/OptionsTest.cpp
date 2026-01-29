// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>
#include <torch/csrc/autograd/profiler_kineto.h>
#include <vector>
#include "TorchCommTestHelpers.h"
#include "comms/torchcomms/TorchCommOptions.hpp"
#include "comms/torchcomms/TorchWork.hpp"

constexpr int kTensorCount = 4;
constexpr at::ScalarType kTensorDtype = at::kFloat;
const std::unordered_map<std::string, std::string> kDefaultHintsMap = {
    {"hint_key", "hint_value"},
};

/**
 * Test class for options classes in all operations in TorchComm.
 *
 * This class verifies that the options classes are being accepted correctly
 * as a C++ APIargument.
 *
 * TODO: this test needs to be augmented to verify that the options are being
 * passed correctly to the backend class, which will require a getter in the
 * work class or similar method.
 */
class OptionsTest : public ::testing::Test {
 public:
  OptionsTest() : rank_(0), num_ranks_(0) {}

  // Helper function declarations with parameters
  at::Tensor createInputTensor(int count, at::ScalarType dtype);

 protected:
  void SetUp() override {
    wrapper_ = std::make_unique<TorchCommTestWrapper>();
    torchcomm_ = wrapper_->getTorchComm();
    rank_ = torchcomm_->getRank();
    num_ranks_ = torchcomm_->getSize();

    auto options =
        at::TensorOptions().dtype(kTensorDtype).device(wrapper_->getDevice());

    // Prepare multiple tensors for all the operations to be tested.
    send_tensor_ = at::ones(kTensorCount, options) * float(rank_ + 1);
    recv_tensor_ = at::zeros(kTensorCount, options);
    tensors_all_ranks_.reserve(num_ranks_);
    for (int i = 0; i < num_ranks_; i++) {
      tensors_all_ranks_.push_back(at::zeros(kTensorCount, options));
    }
    input_tensors_.reserve(num_ranks_);
    for (int i = 0; i < num_ranks_; i++) {
      input_tensors_.push_back(at::zeros(kTensorCount, options));
    }
    recv_tensor_single_ = at::zeros(kTensorCount * num_ranks_, options);
    send_tensor_single_ = at::zeros(kTensorCount * num_ranks_, options);

    // Create split sizes for all_to_all_v_single
    input_split_sizes_.resize(num_ranks_, kTensorCount);
    output_split_sizes_.resize(num_ranks_, kTensorCount);

    send_rank_ = (rank_ + 1) % num_ranks_;
    recv_rank_ = (rank_ + num_ranks_ - 1) % num_ranks_;
  }

  void TearDown() override {
    // Explicitly reset the TorchComm object to ensure proper cleanup
    if (torchcomm_) {
      torchcomm_.reset();
    }
    if (wrapper_) {
      wrapper_.reset();
    }
  }

  std::unique_ptr<TorchCommTestWrapper> wrapper_;
  std::shared_ptr<torch::comms::TorchComm> torchcomm_;
  int rank_;
  int num_ranks_;

  // Test tensors and variables
  at::Tensor send_tensor_;
  at::Tensor recv_tensor_;
  std::vector<at::Tensor> tensors_all_ranks_;
  std::vector<at::Tensor> input_tensors_;
  at::Tensor recv_tensor_single_;
  at::Tensor send_tensor_single_;
  std::vector<uint64_t> input_split_sizes_;
  std::vector<uint64_t> output_split_sizes_;
  int send_rank_{};
  int recv_rank_{};
};

at::Tensor OptionsTest::createInputTensor(int count, at::ScalarType dtype) {
  auto device = wrapper_->getDevice();
  auto options = at::TensorOptions().dtype(dtype).device(device);
  at::Tensor input;
  if (dtype == at::kFloat) {
    input = at::ones({count}, options) * static_cast<float>(rank_ + 1);
  } else if (dtype == at::kInt) {
    input = at::ones({count}, options) * static_cast<int>(rank_ + 1);
  } else if (dtype == at::kChar) {
    input = at::ones({count}, options) * static_cast<signed char>(rank_ + 1);
  }
  return input;
}

TEST_F(OptionsTest, SendRecv) {
  auto send_options = torch::comms::SendOptions();
  send_options.hints = kDefaultHintsMap;
  send_options.timeout = torch::comms::kDefaultTimeout;

  auto recv_options = torch::comms::RecvOptions();
  recv_options.hints = kDefaultHintsMap;
  recv_options.timeout = torch::comms::kDefaultTimeout;

  if (rank_ % 2 == 0) {
    // Even ranks: send first, then receive
    torchcomm_->send(send_tensor_, send_rank_, false, send_options);
    torchcomm_->recv(recv_tensor_, recv_rank_, false, recv_options);
  } else {
    // Odd ranks: receive first, then send
    torchcomm_->recv(recv_tensor_, recv_rank_, false, recv_options);
    torchcomm_->send(send_tensor_, send_rank_, false, send_options);
  }
}

TEST_F(OptionsTest, AllReduce) {
  auto options = torch::comms::AllReduceOptions();
  options.hints = kDefaultHintsMap;
  options.timeout = torch::comms::kDefaultTimeout;
  torchcomm_->all_reduce(
      send_tensor_, torch::comms::ReduceOp::SUM, false, options);
}

TEST_F(OptionsTest, Reduce) {
  auto options = torch::comms::ReduceOptions();
  options.hints = kDefaultHintsMap;
  options.timeout = torch::comms::kDefaultTimeout;
  torchcomm_->reduce(
      send_tensor_, 0, torch::comms::ReduceOp::SUM, false, options);
}

TEST_F(OptionsTest, AllGatherSingle) {
  auto options = torch::comms::AllGatherSingleOptions();
  options.hints = kDefaultHintsMap;
  options.timeout = torch::comms::kDefaultTimeout;
  torchcomm_->all_gather_single(
      recv_tensor_single_, send_tensor_, false, options);
}

TEST_F(OptionsTest, AllGather) {
  auto options = torch::comms::AllGatherOptions();
  options.hints = kDefaultHintsMap;
  options.timeout = torch::comms::kDefaultTimeout;
  torchcomm_->all_gather(tensors_all_ranks_, send_tensor_, false, options);
}

TEST_F(OptionsTest, Gather) {
  auto options = torch::comms::GatherOptions();
  options.hints = kDefaultHintsMap;
  options.timeout = torch::comms::kDefaultTimeout;
  torchcomm_->gather(tensors_all_ranks_, send_tensor_, 0, false, options);
}

TEST_F(OptionsTest, ReduceScatterSingle) {
  auto options = torch::comms::ReduceScatterSingleOptions();
  options.hints = kDefaultHintsMap;
  options.timeout = torch::comms::kDefaultTimeout;
  torchcomm_->reduce_scatter_single(
      recv_tensor_,
      send_tensor_single_,
      torch::comms::ReduceOp::SUM,
      false,
      options);
}

TEST_F(OptionsTest, ReduceScatter) {
  auto options = torch::comms::ReduceScatterOptions();
  options.hints = kDefaultHintsMap;
  options.timeout = torch::comms::kDefaultTimeout;
  torchcomm_->reduce_scatter(
      recv_tensor_,
      tensors_all_ranks_,
      torch::comms::ReduceOp::SUM,
      false,
      options);
}

TEST_F(OptionsTest, Scatter) {
  auto options = torch::comms::ScatterOptions();
  options.hints = kDefaultHintsMap;
  options.timeout = torch::comms::kDefaultTimeout;
  torchcomm_->scatter(recv_tensor_, tensors_all_ranks_, 0, false, options);
}

TEST_F(OptionsTest, AllToAll) {
  auto options = torch::comms::AllToAllOptions();
  options.hints = kDefaultHintsMap;
  options.timeout = torch::comms::kDefaultTimeout;
  torchcomm_->all_to_all(tensors_all_ranks_, input_tensors_, false, options);
}

TEST_F(OptionsTest, AllToAllSingle) {
  auto options = torch::comms::AllToAllSingleOptions();
  options.hints = kDefaultHintsMap;
  options.timeout = torch::comms::kDefaultTimeout;
  torchcomm_->all_to_all_single(
      recv_tensor_single_, send_tensor_single_, false, options);
}

TEST_F(OptionsTest, AllToAllVSingle) {
  auto options = torch::comms::AllToAllvSingleOptions();
  options.hints = kDefaultHintsMap;
  options.timeout = torch::comms::kDefaultTimeout;
  torchcomm_->all_to_all_v_single(
      recv_tensor_single_,
      send_tensor_single_,
      output_split_sizes_,
      input_split_sizes_,
      false,
      options);
}

TEST_F(OptionsTest, Broadcast) {
  auto options = torch::comms::BroadcastOptions();
  options.hints = kDefaultHintsMap;
  options.timeout = torch::comms::kDefaultTimeout;
  torchcomm_->broadcast(send_tensor_, 0, false, options);
}

TEST_F(OptionsTest, Barrier) {
  auto options = torch::comms::BarrierOptions();
  options.hints = kDefaultHintsMap;
  options.timeout = torch::comms::kDefaultTimeout;
  auto work = torchcomm_->barrier(false, options);
  work->wait();
}

// This main function is provided by gtest
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
