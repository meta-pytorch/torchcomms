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
#include "comms/torchcomms/rccl/TorchCommRCCL.hpp"

namespace torch::comms {

using c10::cuda::CUDACachingAllocator::attachAllocatorTraceTracker;
using c10::cuda::CUDACachingAllocator::snapshot;
using c10::cuda::CUDACachingAllocator::TraceEntry;

class RcclCachingAllocatorHookImpl {
 public:
  virtual ~RcclCachingAllocatorHookImpl() = default;
  virtual void regDeregMem(const TraceEntry& te);
  virtual void registerComm(TorchCommRCCL* comm);
  virtual void deregisterComm(TorchCommRCCL* comm);
  virtual void registerMemPreHook();
  virtual void clear();

  virtual bool isCommRegistered(TorchCommRCCL* comm);

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
  std::set<TorchCommRCCL*> registeredComms_;
};

class DefaultRcclCachingAllocatorHookImpl
    : public RcclCachingAllocatorHookImpl {
 public:
  DefaultRcclCachingAllocatorHookImpl();
  virtual ~DefaultRcclCachingAllocatorHookImpl() = default;

  // Delete copy constructor and assignment operator
  DefaultRcclCachingAllocatorHookImpl(
      const DefaultRcclCachingAllocatorHookImpl&) = delete;
  DefaultRcclCachingAllocatorHookImpl& operator=(
      const DefaultRcclCachingAllocatorHookImpl&) = delete;
  // Delete move constructor and assignment operator
  DefaultRcclCachingAllocatorHookImpl(DefaultRcclCachingAllocatorHookImpl&&) =
      delete;
  DefaultRcclCachingAllocatorHookImpl& operator=(
      DefaultRcclCachingAllocatorHookImpl&&) = delete;
};

class RcclCachingAllocatorHook {
 public:
  // Get the singleton instance
  static RcclCachingAllocatorHookImpl& getInstance();

  // only for use by tests
  static void setInstance(
      std::unique_ptr<RcclCachingAllocatorHookImpl> instance) {
    instance_ = std::move(instance);
  }

 protected:
  static void createInstance() {
    if (!instance_) {
      instance_ = std::make_unique<DefaultRcclCachingAllocatorHookImpl>();
    }
  }

  inline static std::unique_ptr<RcclCachingAllocatorHookImpl> instance_ =
      nullptr;
  // NOLINTNEXTLINE(facebook-hte-std::once_flag)
  inline static std::once_flag init_flag_;
};

// Global function to be registered as a hook
void rcclCachingAllocatorHookFn(const TraceEntry& te);

} // namespace torch::comms
