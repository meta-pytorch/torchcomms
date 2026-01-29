// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/ncclx/TorchCommNCCLXCCA.hpp"

namespace torch::comms {

// Global function to be registered as a hook
void cachingAllocatorHookFn(
    const c10::cuda::CUDACachingAllocator::TraceEntry& te) {
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
  c10::cuda::CUDACachingAllocator::attachAllocatorTraceTracker(
      &cachingAllocatorHookFn);
}

void CachingAllocatorHookImpl::registerMemPreHook() {
  // We assume no mem pool and no comm has been created yet, we just loop up the
  // snapshot of the default pool for all devices.
  auto snapshot = c10::cuda::CUDACachingAllocator::snapshot();
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

  if (te.action_ ==
          c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_ALLOC ||
      te.action_ ==
          c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_MAP) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    void* addr = reinterpret_cast<void*>(static_cast<uintptr_t>(te.addr_));
    size_t len = te.size_;

    if (registeredMemMap_.contains(addr)) {
      LOG(ERROR) << "[CCA] Memory already registered at 0x" << std::hex << addr
                 << std::dec << " size=" << len
                 << " existing_size=" << registeredMemMap_.at(addr).len;
      throw std::runtime_error("Memory already registered with NCCLX");
    }

    registeredMemMap_.emplace(addr, MemInfo{len, te.device_});

    ncclResult_t result = ncclGlobalRegisterWithPtr(addr, len, te.device_);
    if (result != ncclSuccess) {
      LOG(ERROR) << "[CCA] Global registration failed for addr=0x" << std::hex
                 << addr << std::dec << " size=" << len
                 << " device=" << te.device_
                 << " error=" << ncclGetErrorString(result);
    }
  } else if (
      te.action_ ==
          c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_FREE ||
      te.action_ ==
          c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_UNMAP) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    void* addr = reinterpret_cast<void*>(static_cast<uintptr_t>(te.addr_));

    auto it = registeredMemMap_.find(addr);
    if (it == registeredMemMap_.end() || it->second.device != te.device_) {
      LOG(ERROR) << "[CCA] Memory not registered at 0x" << std::hex << addr
                 << std::dec << " size=" << te.size_;
      throw std::runtime_error("Memory not registered with NCCLX");
    }

    size_t len = it->second.len;

    // Global deregistration
    ncclResult_t result = ncclGlobalDeregisterWithPtr(addr, len, te.device_);
    if (result != ncclSuccess) {
      LOG(ERROR) << "[CCA] Global deregistration failed for addr=0x" << std::hex
                 << addr << std::dec << " size=" << len
                 << " device=" << te.device_
                 << " error=" << ncclGetErrorString(result);
    }

    registeredMemMap_.erase(it);
  }
}

void CachingAllocatorHookImpl::registerComm(TorchCommNCCLX* comm) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Check if the communicator is already registered
  if (registeredComms_.contains(comm)) {
    throw std::runtime_error("Communicator already registered");
  }

  registeredComms_.insert(comm);
}

void CachingAllocatorHookImpl::deregisterComm(TorchCommNCCLX* comm) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!registeredComms_.contains(comm)) {
    // Should this be fatal?
    return;
  }

  // Deregister all memory for this device when the comm is deregistered.
  // This ensures that memory is properly deregistered from ctran before the
  // RegCache singleton is destroyed at process exit.
  //
  // Note: For global registration, we only deregister when the LAST comm for
  // a device is deregistered. If multiple comms share the same device, we
  // should keep the registration until all are gone.
  bool lastCommForDevice = true;
  for (const auto& otherComm : registeredComms_) {
    if (otherComm != comm &&
        otherComm->getDevice().index() == comm->getDevice().index()) {
      lastCommForDevice = false;
      break;
    }
  }

  if (lastCommForDevice) {
    std::vector<void*> deregisteredAddrs;
    for (const auto& [addr, mem_info] : registeredMemMap_) {
      if (mem_info.device == comm->getDevice().index()) {
        ncclResult_t result =
            ncclGlobalDeregisterWithPtr(addr, mem_info.len, mem_info.device);
        if (result != ncclSuccess) {
          LOG(WARNING) << "[CCA] Deregistration failed during deregisterComm "
                       << "for addr=0x" << std::hex << addr << std::dec
                       << " size=" << mem_info.len
                       << " device=" << mem_info.device
                       << " error=" << ncclGetErrorString(result);
        }
        deregisteredAddrs.push_back(addr);
      }
    }
    for (auto addr : deregisteredAddrs) {
      registeredMemMap_.erase(addr);
    }
  }

  registeredComms_.erase(comm);
}

void CachingAllocatorHookImpl::clear() {
  std::lock_guard<std::mutex> lock(mutex_);

  // Deregister all tracked memory globally
  for (const auto& [addr, mem_info] : registeredMemMap_) {
    ncclResult_t result =
        ncclGlobalDeregisterWithPtr(addr, mem_info.len, mem_info.device);
    if (result != ncclSuccess) {
      LOG(WARNING) << "[CCA] Clear deregistration failed for addr=0x"
                   << std::hex << addr << std::dec << " size=" << mem_info.len
                   << " device=" << mem_info.device
                   << " error=" << ncclGetErrorString(result);
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

} // namespace torch::comms
