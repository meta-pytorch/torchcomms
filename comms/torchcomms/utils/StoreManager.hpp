// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <mutex>
#include <unordered_set>

#include <torch/csrc/distributed/c10d/Store.hpp> // @manual=//caffe2:torch-cpp-cpu

#include "comms/torchcomms/TorchCommBackend.hpp"

namespace torch::comms {

class StoreManager {
 private:
  StoreManager() = default;

 public:
  static StoreManager& get();

  c10::intrusive_ptr<c10d::Store> createPrefixedStore(
      std::string_view backendName,
      std::string_view commName,
      std::chrono::milliseconds timeout);

 private:
  std::mutex mutex_{};
  c10::intrusive_ptr<c10d::Store> root_{};
  std::unordered_set<std::string> storeNames_{};
};

} // namespace torch::comms
