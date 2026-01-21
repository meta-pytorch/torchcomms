// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>
#include <string>
#include <tuple>

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <c10/util/intrusive_ptr.h>
#include <gtest/gtest.h>
#include <torch/csrc/distributed/c10d/Store.hpp> // @manual=//caffe2:torch-cpp-cpu
#include "comms/torchcomms/TorchComm.hpp"

std::string getDtypeName(at::ScalarType dtype);
std::string getOpName(const torch::comms::ReduceOp& op);
std::tuple<int, int> getRankAndSize();
c10::intrusive_ptr<c10d::Store> createStore();
void destroyStore(
    c10::intrusive_ptr<c10d::Store>&& store,
    const std::shared_ptr<torch::comms::TorchComm>& torchcomm);

// Convert a tensor to a string representation with nested brackets for each
// dimension. Supports any N-dimensional tensor.
// All aten datatypes are supported.
// Example output:
//   Tensor([[1, 2, 3], [4, 5, 6]]) -> [[1, 2, 3], [4, 5, 6]]
// See more examples in TensorToStringTest.cpp
std::string tensorToString(const at::Tensor& tensor);

void verifyTensorEquality(
    const at::Tensor& output,
    const at::Tensor& expected,
    const std::string& description = "");

void verifyTensorEquality(
    const at::Tensor& output,
    const double expected_value,
    const std::string& description = "");

class TorchCommTestWrapper {
 public:
  TorchCommTestWrapper(c10::intrusive_ptr<c10d::Store> store = nullptr);

  virtual ~TorchCommTestWrapper() {
    if (torchcomm_) {
      torchcomm_->finalize();
      torchcomm_.reset();
    }
  }

  // Delete copy and move operations to follow rule of five
  TorchCommTestWrapper(const TorchCommTestWrapper&) = delete;
  TorchCommTestWrapper& operator=(const TorchCommTestWrapper&) = delete;
  TorchCommTestWrapper(TorchCommTestWrapper&&) = delete;
  TorchCommTestWrapper& operator=(TorchCommTestWrapper&&) = delete;

  std::shared_ptr<torch::comms::TorchComm> getTorchComm() const {
    return torchcomm_;
  }

  virtual c10::Device getDevice() {
    // We don't need to pass the exact device index here.  TorchComm will figure
    // out based on our local rank
    return c10::Device(c10::DeviceType::CUDA);
  }

 protected:
  std::shared_ptr<torch::comms::TorchComm> torchcomm_;
};
