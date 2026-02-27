// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <memory>
#include <mutex>
#include "comms/torchcomms/ncclx/NcclxApi.hpp"
#include "comms/torchcomms/ncclx/TorchCommNCCLX.hpp"

namespace torch::comms {

class CachingAllocatorHookImpl {
 public:
  virtual ~CachingAllocatorHookImpl() = default;
  virtual void regDeregMem(
      const c10::cuda::CUDACachingAllocator::TraceEntry& te);
  virtual void registerMemPreHook();

  // For testing purposes.
  virtual bool isMemPreHookRegistered();

  // Set the NcclxApi to use for registration. For testing.
  void setNcclApi(std::shared_ptr<NcclxApi> api) {
    nccl_api_ = std::move(api);
  }

  NcclxApi* getNcclApi() const {
    return nccl_api_.get();
  }

 protected:
  // Flag to indicate if the memory pre hook is registered. (For testing
  // purposes)
  bool mem_pre_hook_registered_ = false;

  // NcclxApi used for global registration operations.
  // Initialized to DefaultNcclxApi by default.
  std::shared_ptr<NcclxApi> nccl_api_ = std::make_shared<DefaultNcclxApi>();
};

class DefaultCachingAllocatorHookImpl : public CachingAllocatorHookImpl {
 public:
  DefaultCachingAllocatorHookImpl();
  virtual ~DefaultCachingAllocatorHookImpl() = default;

  // Delete copy constructor and assignment operator
  DefaultCachingAllocatorHookImpl(const DefaultCachingAllocatorHookImpl&) =
      delete;
  DefaultCachingAllocatorHookImpl& operator=(
      const DefaultCachingAllocatorHookImpl&) = delete;
  // Delete move constructor and assignment operator
  DefaultCachingAllocatorHookImpl(DefaultCachingAllocatorHookImpl&&) = delete;
  DefaultCachingAllocatorHookImpl& operator=(
      DefaultCachingAllocatorHookImpl&&) = delete;
};

class CachingAllocatorHook {
 public:
  // Get the singleton instance
  static CachingAllocatorHookImpl& getInstance();

  // only for use by tests
  static void setInstance(std::unique_ptr<CachingAllocatorHookImpl> instance) {
    instance_ = std::move(instance);
  }

 protected:
  static void createInstance() {
    if (!instance_) {
      instance_ = std::make_unique<DefaultCachingAllocatorHookImpl>();
    }
  }

  inline static std::unique_ptr<CachingAllocatorHookImpl> instance_ = nullptr;
  // NOLINTNEXTLINE(facebook-hte-std::once_flag)
  inline static std::once_flag init_flag_;
};

// Global function to be registered as a hook
void cachingAllocatorHookFn(
    const c10::cuda::CUDACachingAllocator::TraceEntry& te);

} // namespace torch::comms
