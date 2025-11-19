// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/ncclx/TorchCommNCCLXCCA.hpp"
#include <mutex>

namespace torch {
namespace comms {

// Global function to be registered as a hook
void cachingAllocatorHookFn(
    const c10::cuda::CUDACachingAllocator::TraceEntry& te) {
  // Forward to the singleton instance
  CachingAllocatorHook::getInstance().regDeregMem(te);
}

CachingAllocatorHookImpl& CachingAllocatorHook::getInstance() {
  // Create a static instance of the class based on the first call.
  // This allows threads to override the device type if needed.
  if (!instance_) {
    static std::mutex init_mutex;
    std::lock_guard<std::mutex> lock(init_mutex);
    if (!instance_) {
      createInstance();
    }
  }
  return *instance_;
}

DefaultCachingAllocatorHookImpl::DefaultCachingAllocatorHookImpl() {
  // Setup memory registration hooks
  at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
  registerMemPreHook();
  c10::cuda::CUDACachingAllocator::attachAllocatorTraceTracker(
      &cachingAllocatorHookFn);
}

void CachingAllocatorHookImpl::registerMemPreHook() {
  int device = c10::cuda::current_device();
  // We assume no mem pool and no comm has been created yet, we just loop up the
  // snapshot of the default pool for the current device.
#ifdef TORCHCOMMS_CONDA_BUILD
  auto snapshot = c10::cuda::CUDACachingAllocator::snapshot();
#else
  auto snapshot = c10::cuda::CUDACachingAllocator::snapshot({device, 0});
#endif
  for (const auto& segmentInfo : snapshot.segments) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    void* addr = reinterpret_cast<void*>(segmentInfo.address);
    size_t len = segmentInfo.total_size;

    if (registeredMemMap_.contains(addr)) {
      throw std::runtime_error("Memory already registered with NCCLX");
    } else {
      registeredMemMap_.emplace(addr, MemInfo{len, segmentInfo.device});
    }
  }
  mem_pre_hook_registered_ = true;
}

void CachingAllocatorHookImpl::regDeregMem(
    const c10::cuda::CUDACachingAllocator::TraceEntry& te) {
  std::lock_guard<std::mutex> lock(mutex_);
  bool register_mem = te.action_ ==
      c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_ALLOC;
  bool unregister_mem = te.action_ ==
      c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_FREE;

  if (register_mem) {
    // Memory got allocated, register it with NCCL
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    void* addr = reinterpret_cast<void*>(static_cast<uintptr_t>(te.addr_));
    size_t len = te.size_;

    if (registeredMemMap_.contains(addr)) {
      throw std::runtime_error("Memory already registered with NCCL");
    } else {
      registeredMemMap_.emplace(addr, MemInfo{len, te.device_});
    }

    // Register the memory through ncclCommRegister and add to commRegHandles_
    for (auto& comm : registeredComms_) {
      if (te.device_ == comm->getDevice().index()) {
        comm->register_address(TorchCommNCCLX::AddressWithLen(addr, len));
      }
    }
  } else if (unregister_mem) {
    // Memory got freed, deregister it with NCCL
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    void* addr = reinterpret_cast<void*>(static_cast<uintptr_t>(te.addr_));

    if (!registeredMemMap_.contains(addr)) {
      throw std::runtime_error("Memory not registered with NCCL");
    } else {
      registeredMemMap_.erase(addr);
    }

    for (auto& comm : registeredComms_) {
      if (te.device_ == comm->getDevice().index()) {
        comm->deregister_address(TorchCommNCCLX::Address(addr));
      }
    }
  }
}

void CachingAllocatorHookImpl::registerComm(TorchCommNCCLX* comm) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Check if the communicator is already registered
  if (registeredComms_.contains(comm)) {
    throw std::runtime_error("Communicator already registered");
  }

  // Register all memory that has already been allocated
  for (const auto& [addr, mem_info] : registeredMemMap_) {
    if (mem_info.device == comm->getDevice().index()) {
      comm->register_address(
          TorchCommNCCLX::AddressWithLen(addr, mem_info.len));
    }
  }

  registeredComms_.insert(comm);
}

void CachingAllocatorHookImpl::deregisterComm(TorchCommNCCLX* comm) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!registeredComms_.contains(comm)) {
    // Should this be fatal?
    return;
  }

  // De-register all memory that has already been allocated
  for (const auto& [addr, mem_info] : registeredMemMap_) {
    if (mem_info.device == comm->getDevice().index()) {
      comm->deregister_address(TorchCommNCCLX::Address(addr));
    }
  }

  registeredComms_.erase(comm);
}

void CachingAllocatorHookImpl::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& comm : registeredComms_) {
    for (const auto& [addr, mem_info] : registeredMemMap_) {
      if (mem_info.device == comm->getDevice().index()) {
        comm->deregister_address(TorchCommNCCLX::Address(addr));
      }
    }
  }
  registeredMemMap_.clear();
  registeredComms_.clear();
}

bool CachingAllocatorHookImpl::isCommRegistered(TorchCommNCCLX* comm) {
  std::lock_guard<std::mutex> lock(mutex_);
  return registeredComms_.contains(comm);
}

bool CachingAllocatorHookImpl::isMemRegisteredCalled() {
  return mem_pre_hook_registered_;
}

} // namespace comms
} // namespace torch
