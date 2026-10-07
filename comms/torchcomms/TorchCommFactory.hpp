// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <comms/torchcomms/TorchCommBackend.hpp>
#include <comms/torchcomms/TorchCommOptions.hpp>

// IWYU pragma: no_include <ATen/ATen.h>

namespace torch::comms {

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

  void register_backend(
      const std::string& backend,
      const std::function<std::shared_ptr<TorchCommBackend>()>& factory,
      std::string_view backendAbiVersion);

  // Allocator factory methods
  std::shared_ptr<c10::Allocator> get_allocator(const std::string& backend);

  void register_allocator_factory(
      const std::string& backend,
      const std::function<std::shared_ptr<c10::Allocator>()>& factory);

  bool is_backend_registered(const std::string& backend) const;

 private:
  using BackendFactory = std::function<std::shared_ptr<TorchCommBackend>()>;

  struct BackendRegistration {
    BackendFactory factory;
    std::string abiVersion;
  };

  std::shared_ptr<TorchCommBackend> create_generic_backend(
      const std::string& backend);

  mutable std::mutex mutex_;
  std::unordered_map<std::string, BackendRegistration> backends_;
  std::unordered_map<
      std::string,
      std::function<std::shared_ptr<c10::Allocator>()>>
      allocator_factories_;
};
} // namespace torch::comms
