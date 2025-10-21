// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <comms/torchcomms/TorchCommBackend.hpp>
#include <comms/torchcomms/TorchCommOptions.hpp>

namespace torch {
namespace comms {

class TorchCommFactory {
 public:
  static TorchCommFactory& get();

  std::shared_ptr<TorchCommBackend> create_backend(
      const std::string& backend,
      at::Device device,
      const std::string& name,
      const CommOptions& options = CommOptions());

  void register_backend(
      const std::string& backend,
      const std::function<std::shared_ptr<TorchCommBackend>()>& factory);

 private:
  std::shared_ptr<TorchCommBackend> create_generic_backend(
      const std::string& backend);

 private:
  std::mutex mutex_;
  std::unordered_map<
      std::string,
      std::function<std::shared_ptr<TorchCommBackend>()>>
      backends_;
};
} // namespace comms
} // namespace torch
