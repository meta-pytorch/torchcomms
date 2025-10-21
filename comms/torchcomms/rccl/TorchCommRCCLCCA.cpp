// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/rccl/TorchCommRCCLCCA.hpp"
#include <mutex>

namespace torch {
namespace comms {

// Global function to be registered as a hook
void cachingAllocatorHookFn(
    const c10::hip::HIPCachingAllocator::TraceEntry& te) {
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
  c10::hip::HIPCachingAllocator::attachAllocatorTraceTracker(
      &cachingAllocatorHookFn);
}

void CachingAllocatorHookImpl::regDeregMem(
    const c10::hip::HIPCachingAllocator::TraceEntry& te) {
  std::lock_guard<std::mutex> lock(mutex_);
  bool register_mem = te.action_ ==
      c10::hip::HIPCachingAllocator::TraceEntry::Action::SEGMENT_ALLOC;
  bool unregister_mem = te.action_ ==
      c10::hip::HIPCachingAllocator::TraceEntry::Action::SEGMENT_FREE;

  if (register_mem) {
    // Memory got allocated, register it with NCCL
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    void* addr = reinterpret_cast<void*>(static_cast<uintptr_t>(te.addr_));
    size_t len = te.size_;

    if (registeredMemMap_.find(addr) != registeredMemMap_.end()) {
      throw std::runtime_error("Memory already registered with NCCL");
    } else {
      registeredMemMap_[addr] = len;
    }

    // Register the memory through ncclCommRegister and add to commRegHandles_
    for (auto& comm : registeredComms_) {
      comm->register_address(TorchCommRCCL::AddressWithLen{addr, len});
    }
  } else if (unregister_mem) {
    // Memory got freed, deregister it with NCCL
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    void* addr = reinterpret_cast<void*>(static_cast<uintptr_t>(te.addr_));

    if (registeredMemMap_.find(addr) == registeredMemMap_.end()) {
      throw std::runtime_error("Memory not registered with NCCL");
    } else {
      registeredMemMap_.erase(addr);
    }

    for (auto& comm : registeredComms_) {
      comm->deregister_address(TorchCommRCCL::Address{addr});
    }
  }
}

void CachingAllocatorHookImpl::registerComm(TorchCommRCCL* comm) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Check if the communicator is already registered
  if (registeredComms_.find(comm) != registeredComms_.end()) {
    throw std::runtime_error("Communicator already registered");
  }

  // Register all memory that has already been allocated
  for (const auto& [addr, len] : registeredMemMap_) {
    comm->register_address(TorchCommRCCL::AddressWithLen{addr, len});
  }

  registeredComms_.insert(comm);
}

void CachingAllocatorHookImpl::deregisterComm(TorchCommRCCL* comm) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (registeredComms_.find(comm) == registeredComms_.end()) {
    // Should this be fatal?
    return;
  }

  // De-register all memory that has already been allocated
  for (const auto& [addr, len] : registeredMemMap_) {
    comm->deregister_address(TorchCommRCCL::Address{addr});
  }

  registeredComms_.erase(comm);
}

void CachingAllocatorHookImpl::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& comm : registeredComms_) {
    for (const auto& [addr, len] : registeredMemMap_) {
      comm->deregister_address(TorchCommRCCL::Address{addr});
    }
  }
  registeredMemMap_.clear();
  registeredComms_.clear();
}

bool CachingAllocatorHookImpl::isCommRegistered(TorchCommRCCL* comm) {
  std::lock_guard<std::mutex> lock(mutex_);
  return registeredComms_.find(comm) != registeredComms_.end();
}

} // namespace comms
} // namespace torch
