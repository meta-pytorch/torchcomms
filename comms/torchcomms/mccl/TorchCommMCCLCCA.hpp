// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <memory>
#include <mutex>
#include <vector>
#include "comms/torchcomms/mccl/TorchCommMCCL.hpp"

namespace torch::comms {

// McclCachingAllocatorHookImpl is a class that does self registration with
// pytorch tensor memory.
class McclCachingAllocatorHookImpl {
 public:
  virtual ~McclCachingAllocatorHookImpl() = default;
  void regDeregMem(const c10::cuda::CUDACachingAllocator::TraceEntry& te);
  void registerComm(TorchCommMCCL* comm);
  void deregisterComm(TorchCommMCCL* comm);
  void registerMemPreHook();
  void clear();

  bool isCommRegistered(TorchCommMCCL* comm);

  // Drain any queued registration operations. Called before collectives to
  // avoid contending with regRange's segmentsAvl_ wlock during the allocator
  // callback (which would deadlock the allocator thread).
  static void drainPendingRegistrations();

 private:
  std::mutex mutex_;

  struct MemInfo {
    size_t len;
    int32_t device;

    MemInfo(size_t l, int32_t d) : len(l), device(d) {}
  };

  // Map of registered memory addresses to their sizes and device
  std::unordered_map<void*, MemInfo> registeredMemMap_;
  std::set<TorchCommMCCL*> registeredComms_;

  enum class PendingOperationType {
    Register,
    Deregister,
  };

  struct PendingOperation {
    PendingOperationType type;
    void* addr;
    size_t len;
    int32_t device;
  };

  void queuePendingOperation(PendingOperation operation);
  bool removePendingRegistrations(void* addr, size_t len);
  bool isRegisteredRangeCovered(void* addr, size_t len, int32_t device) const;
  bool removeRegisteredRange(void* addr, size_t len, int32_t device);
  void drainPendingRegistrationOps();
  void drainPendingRegistrationOpsLocked();

  std::mutex pendingMutex_;
  std::vector<PendingOperation> pendingOps_;
};

class DefaultMcclCachingAllocatorHookImpl
    : public McclCachingAllocatorHookImpl {
 public:
  DefaultMcclCachingAllocatorHookImpl();
  ~DefaultMcclCachingAllocatorHookImpl() override;

  DefaultMcclCachingAllocatorHookImpl(
      const DefaultMcclCachingAllocatorHookImpl&) = delete;
  DefaultMcclCachingAllocatorHookImpl& operator=(
      const DefaultMcclCachingAllocatorHookImpl&) = delete;
  DefaultMcclCachingAllocatorHookImpl(DefaultMcclCachingAllocatorHookImpl&&) =
      delete;
  DefaultMcclCachingAllocatorHookImpl& operator=(
      DefaultMcclCachingAllocatorHookImpl&&) = delete;
};

class McclCachingAllocatorHook {
 public:
  // Get the singleton instance
  static McclCachingAllocatorHookImpl& getInstance();
  static bool isDestroyed();
  static void setDestroyed();

  static bool isEnabled();

  // only for use by tests
  static void setInstance(
      std::unique_ptr<McclCachingAllocatorHookImpl> instance) {
    instance_ = std::move(instance);
  }

 protected:
  inline static std::atomic<bool> destroyed_ = false;
  static void createInstance() {
    if (!instance_) {
      instance_ = std::make_unique<DefaultMcclCachingAllocatorHookImpl>();
    }
  }

  inline static std::unique_ptr<McclCachingAllocatorHookImpl> instance_ =
      nullptr;
  // NOLINTNEXTLINE(facebook-hte-std::once_flag)
  inline static std::once_flag init_flag_;
};

// Global function to be registered as a hook
void mcclCachingAllocatorHookFn(
    const c10::cuda::CUDACachingAllocator::TraceEntry& te);

} // namespace torch::comms
