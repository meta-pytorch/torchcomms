// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/rccl/TorchCommRCCLCCA.hpp"
#include <mutex>

namespace torch::comms {

// Global function to be registered as a hook
void cachingAllocatorHookFn(const TraceEntry& te) {
  // Forward to the singleton instance
  CachingAllocatorHook::getInstance().regDeregMem(te);
}

CachingAllocatorHookImpl& CachingAllocatorHook::getInstance() {
  // Use std::call_once for thread-safe singleton initialization
  std::call_once(init_flag_, createInstance);
  return *instance_;
}

DefaultCachingAllocatorHookImpl::DefaultCachingAllocatorHookImpl() {
  // Setup memory registration hooks
  at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
  registerMemPreHook();
  attachAllocatorTraceTracker(&cachingAllocatorHookFn);
}

void CachingAllocatorHookImpl::registerMemPreHook() {
  // We assume no mem pool and no comm has been created yet, we just loop up the
  // snapshot of the default pool for all devices.
  auto the_snapshot = snapshot();
  for (const auto& segmentInfo : the_snapshot.segments) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    void* addr = reinterpret_cast<void*>(segmentInfo.address);
    size_t len = segmentInfo.total_size;

    if (registeredMemMap_.find(addr) != registeredMemMap_.end()) {
      throw std::runtime_error("Memory already registered with RCCL");
    } else {
      registeredMemMap_.emplace(addr, MemInfo{len, segmentInfo.device});
    }
  }
}

void CachingAllocatorHookImpl::regDeregMem(const TraceEntry& te) {
  std::lock_guard<std::mutex> lock(mutex_);
  bool register_mem = te.action_ == TraceEntry::Action::SEGMENT_ALLOC;
  bool unregister_mem = te.action_ == TraceEntry::Action::SEGMENT_FREE;

  if (register_mem) {
    // Memory got allocated, register it with NCCL
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    void* addr = reinterpret_cast<void*>(static_cast<uintptr_t>(te.addr_));
    size_t len = te.size_;

    if (registeredMemMap_.find(addr) != registeredMemMap_.end()) {
      throw std::runtime_error("Memory already registered with NCCL");
    } else {
      registeredMemMap_.emplace(addr, MemInfo{len, te.device_});
    }

    // Register the memory through ncclCommRegister and add to commRegHandles_
    for (auto& comm : registeredComms_) {
      if (te.device_ == comm->getDevice().index()) {
        comm->register_address(TorchCommRCCL::AddressWithLen{addr, len});
      }
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
      if (te.device_ == comm->getDevice().index()) {
        comm->deregister_address(TorchCommRCCL::Address{addr});
      }
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
  for (const auto& [addr, mem_info] : registeredMemMap_) {
    if (mem_info.device == comm->getDevice().index()) {
      comm->register_address(TorchCommRCCL::AddressWithLen{addr, mem_info.len});
    }
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
  for (const auto& [addr, mem_info] : registeredMemMap_) {
    if (mem_info.device == comm->getDevice().index()) {
      comm->deregister_address(TorchCommRCCL::Address{addr});
    }
  }

  registeredComms_.erase(comm);
}

void CachingAllocatorHookImpl::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& comm : registeredComms_) {
    for (const auto& [addr, mem_info] : registeredMemMap_) {
      if (mem_info.device == comm->getDevice().index()) {
        comm->deregister_address(TorchCommRCCL::Address{addr});
      }
    }
  }
  registeredMemMap_.clear();
  registeredComms_.clear();
}

bool CachingAllocatorHookImpl::isCommRegistered(TorchCommRCCL* comm) {
  std::lock_guard<std::mutex> lock(mutex_);
  return registeredComms_.find(comm) != registeredComms_.end();
}

} // namespace torch::comms
