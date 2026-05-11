// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/hip/HIPContext.h> // @manual
#ifdef HIPIFY_V2
#include <c10/hip/HIPCachingAllocator.h> // @manual
#else
#include <ATen/hip/impl/HIPCachingAllocatorMasqueradingAsCUDA.h> // @manual
#endif
#include <memory>
#include <mutex>
#include "comms/torchcomms/rcclx/TorchCommRCCLX.hpp"

namespace torch::comms {

using c10::cuda::CUDACachingAllocator::attachAllocatorTraceTracker;
using c10::cuda::CUDACachingAllocator::snapshot;
using c10::cuda::CUDACachingAllocator::TraceEntry;

class RcclxCachingAllocatorHookImpl {
 public:
  virtual ~RcclxCachingAllocatorHookImpl() = default;
  virtual void regDeregMem(const TraceEntry& te);
  virtual void registerComm(TorchCommRCCLX* comm);
  virtual void deregisterComm(TorchCommRCCLX* comm);
  virtual void registerMemPreHook();
  virtual void clear();

  virtual bool isCommRegistered(TorchCommRCCLX* comm);

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
  std::set<TorchCommRCCLX*> registeredComms_;
};

class DefaultRcclxCachingAllocatorHookImpl
    : public RcclxCachingAllocatorHookImpl {
 public:
  DefaultRcclxCachingAllocatorHookImpl();
  virtual ~DefaultRcclxCachingAllocatorHookImpl() = default;

  // Delete copy constructor and assignment operator
  DefaultRcclxCachingAllocatorHookImpl(
      const DefaultRcclxCachingAllocatorHookImpl&) = delete;
  DefaultRcclxCachingAllocatorHookImpl& operator=(
      const DefaultRcclxCachingAllocatorHookImpl&) = delete;
  // Delete move constructor and assignment operator
  DefaultRcclxCachingAllocatorHookImpl(DefaultRcclxCachingAllocatorHookImpl&&) =
      delete;
  DefaultRcclxCachingAllocatorHookImpl& operator=(
      DefaultRcclxCachingAllocatorHookImpl&&) = delete;
};

class RcclxCachingAllocatorHook {
 public:
  // Get the singleton instance
  static RcclxCachingAllocatorHookImpl& getInstance();

  // only for use by tests
  static void setInstance(
      std::unique_ptr<RcclxCachingAllocatorHookImpl> instance) {
    instance_ = std::move(instance);
  }

 protected:
  static void createInstance() {
    if (!instance_) {
      instance_ = std::make_unique<DefaultRcclxCachingAllocatorHookImpl>();
    }
  }

  inline static std::unique_ptr<RcclxCachingAllocatorHookImpl> instance_ =
      nullptr;
  // NOLINTNEXTLINE(facebook-hte-std::once_flag)
  inline static std::once_flag init_flag_;
};

// Global function to be registered as a hook
void rcclxCachingAllocatorHookFn(const TraceEntry& te);

} // namespace torch::comms
