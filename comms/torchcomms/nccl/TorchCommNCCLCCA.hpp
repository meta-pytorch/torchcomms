// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <memory>
#include <mutex>
#include "comms/torchcomms/nccl/TorchCommNCCL.hpp"

namespace torch {
namespace comms {

class CachingAllocatorHookImpl {
 public:
  virtual ~CachingAllocatorHookImpl() = default;
  virtual void regDeregMem(
      const c10::cuda::CUDACachingAllocator::TraceEntry& te);
  virtual void registerComm(TorchCommNCCL* comm);
  virtual void deregisterComm(TorchCommNCCL* comm);
  virtual void registerMemPreHook();
  virtual void clear();

  virtual bool isCommRegistered(TorchCommNCCL* comm);

 private:
  std::mutex mutex_;

  struct MemInfo {
    size_t len;
    int32_t device;

    MemInfo(size_t l, int32_t d) : len(l), device(d) {}
  };

  // Map of registered memory addresses to their sizes and device
  std::unordered_map<void*, MemInfo> registeredMemMap_;
  // Set of registered communicators. TorchComms, manages it's membership inside
  // this set.
  std::set<TorchCommNCCL*> registeredComms_;
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
  inline static std::once_flag init_flag_;
};

// Global function to be registered as a hook
void cachingAllocatorHookFn(
    const c10::cuda::CUDACachingAllocator::TraceEntry& te);

} // namespace comms
} // namespace torch
