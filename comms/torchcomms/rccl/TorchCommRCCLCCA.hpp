// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/hip/HIPContext.h> // @manual
#include <ATen/hip/impl/HIPCachingAllocatorMasqueradingAsCUDA.h> // @manual
#include <memory>
#include <mutex>
#include "comms/torchcomms/rccl/TorchCommRCCL.hpp"

namespace torch {
namespace comms {

class CachingAllocatorHookImpl {
 public:
  virtual ~CachingAllocatorHookImpl() = default;
  virtual void regDeregMem(const c10::hip::HIPCachingAllocator::TraceEntry& te);
  virtual void registerComm(TorchCommRCCL* comm);
  virtual void deregisterComm(TorchCommRCCL* comm);
  virtual void clear();

  virtual bool isCommRegistered(TorchCommRCCL* comm);

 private:
  std::mutex mutex_;

  // Map of registered memory addresses to their sizes
  std::unordered_map<void*, size_t> registeredMemMap_;
  // Set of registered communicators. TorchComms, manages it's membership inside
  // this set.
  std::set<TorchCommRCCL*> registeredComms_;
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
};

// Global function to be registered as a hook
void cachingAllocatorHookFn(
    const c10::hip::HIPCachingAllocator::TraceEntry& te);

} // namespace comms
} // namespace torch
