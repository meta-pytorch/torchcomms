// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/ATen.h>
#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy

namespace torch::comms {

// Forward declaration
class TorchCommNCCLX;

class TorchCommNCCLXPersistentRequest : public torch::CustomClassHolder {
 public:
  explicit TorchCommNCCLXPersistentRequest(
      std::shared_ptr<TorchCommNCCLX> comm,
      void* hdl,
      std::optional<cudaStream_t> stream);
  ~TorchCommNCCLXPersistentRequest() noexcept;

  // Delete copy and move operations
  TorchCommNCCLXPersistentRequest(const TorchCommNCCLXPersistentRequest&) =
      delete;
  TorchCommNCCLXPersistentRequest(TorchCommNCCLXPersistentRequest&&) = delete;
  TorchCommNCCLXPersistentRequest& operator=(
      const TorchCommNCCLXPersistentRequest&) = delete;
  TorchCommNCCLXPersistentRequest& operator=(
      TorchCommNCCLXPersistentRequest&&) = delete;

  void* getRequestPtr() const;
  std::optional<cudaStream_t> getStream() const;

 private:
  std::shared_ptr<TorchCommNCCLX> comm_;
  void* hdl_;
  std::optional<cudaStream_t> stream_{std::nullopt};
};

} // namespace torch::comms
