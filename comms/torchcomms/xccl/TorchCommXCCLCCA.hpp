#pragma once

#include <ATen/xpu/XPUContext.h>
#include <c10/xpu/XPUCachingAllocator.h>
#include <memory>
#include <mutex>
#include "comms/torchcomms/xccl/TorchCommXCCL.hpp"

namespace torch::comms {

class XcclCachingAllocatorHookImpl {
 public:
  virtual ~XcclCachingAllocatorHookImpl() = default;
  virtual void regDeregMem(const c10::CachingDeviceAllocator::TraceEntry& te);
  virtual void registerComm(TorchCommXCCL* comm);
  virtual void deregisterComm(TorchCommXCCL* comm);
  virtual void registerMemPreHook();
  virtual void clear();

  virtual bool isCommRegistered(TorchCommXCCL* comm);

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
  std::set<TorchCommXCCL*> registeredComms_;
};

class DefaultXcclCachingAllocatorHookImpl
    : public XcclCachingAllocatorHookImpl {
 public:
  DefaultXcclCachingAllocatorHookImpl();
  virtual ~DefaultXcclCachingAllocatorHookImpl() = default;

  // Delete copy constructor and assignment operator
  DefaultXcclCachingAllocatorHookImpl(
      const DefaultXcclCachingAllocatorHookImpl&) = delete;
  DefaultXcclCachingAllocatorHookImpl& operator=(
      const DefaultXcclCachingAllocatorHookImpl&) = delete;
  // Delete move constructor and assignment operator
  DefaultXcclCachingAllocatorHookImpl(DefaultXcclCachingAllocatorHookImpl&&) =
      delete;
  DefaultXcclCachingAllocatorHookImpl& operator=(
      DefaultXcclCachingAllocatorHookImpl&&) = delete;
};

class XcclCachingAllocatorHook {
 public:
  // Get the singleton instance
  static XcclCachingAllocatorHookImpl& getInstance();

  // only for use by tests
  static void setInstance(
      std::unique_ptr<XcclCachingAllocatorHookImpl> instance) {
    instance_ = std::move(instance);
  }

 protected:
  static void createInstance() {
    if (!instance_) {
      instance_ = std::make_unique<DefaultXcclCachingAllocatorHookImpl>();
    }
  }

  inline static std::unique_ptr<XcclCachingAllocatorHookImpl> instance_ =
      nullptr;
  // NOLINTNEXTLINE(facebook-hte-std::once_flag)
  inline static std::once_flag init_flag_;
};

// Global function to be registered as a hook
void xcclCachingAllocatorHookFn(
    const c10::CachingDeviceAllocator::TraceEntry& te);

} // namespace torch::comms