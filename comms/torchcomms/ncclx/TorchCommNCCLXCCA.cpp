// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/ncclx/TorchCommNCCLXCCA.hpp"
#include <c10/cuda/driver_api.h>

// Helper function to get allocation granularity for a device
namespace {
size_t getAllocationGranularity(int device) {
  auto driver_api = c10::cuda::DriverAPI::get();
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;

  size_t granularity;
  C10_CUDA_DRIVER_CHECK(driver_api->cuMemGetAllocationGranularity_(
      &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

  return granularity;
}
} // namespace

namespace torch {
namespace comms {

// Global function to be registered as a hook
void cachingAllocatorHookFn(
    const c10::cuda::CUDACachingAllocator::TraceEntry& te) {
  // Forward to the singleton instance
  CachingAllocatorHook::getInstance().regDeregMem(te);
}

CachingAllocatorHookImpl& CachingAllocatorHook::getInstance() {
  // Use folly::call_once for thread-safe singleton initialization
  folly::call_once(init_flag_, createInstance);
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
      c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_ALLOC) {
    // Memory got allocated, register it with NCCL
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    void* addr = reinterpret_cast<void*>(static_cast<uintptr_t>(te.addr_));
    size_t len = te.size_;

    if (registeredMemMap_.contains(addr)) {
      LOG(ERROR) << "[CCA] SEGMENT_ALLOC: Memory already registered at 0x"
                 << std::hex << addr << std::dec << " size=" << len
                 << " existing_size=" << registeredMemMap_.at(addr).len;
      throw std::runtime_error("Memory already registered with NCCLX");
    } else {
      registeredMemMap_.emplace(addr, MemInfo{len, te.device_});
    }

    // Register the memory through ncclCommRegister and add to commRegHandles_
    for (auto& comm : registeredComms_) {
      if (te.device_ == comm->getDevice().index()) {
        comm->register_address(TorchCommNCCLX::AddressWithLen(addr, len));
      }
    }
  } else if (
      te.action_ ==
      c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_MAP) {
    // Memory got mapped, register it with NCCL

    // PyTorch expandable segments can send MAP events covering one or more
    // chunks. We need to register each chunk separately.
    // Chunk size is determined by the allocation granularity for the device.
    size_t total_size = te.size_;
    size_t chunk_size = getAllocationGranularity(te.device_);

    for (size_t offset = 0; offset < total_size;) {
      // Determine the chunk size at this offset
      size_t remaining = total_size - offset;

      if (remaining < chunk_size) {
        LOG(ERROR) << "[CCA] SEGMENT_MAP: Invalid remaining size=" << remaining
                   << " at offset=" << offset << " total_size=" << total_size
                   << " chunk_size=" << chunk_size;
        throw std::runtime_error(
            "SEGMENT_MAP: Remaining size must be a multiple of allocation granularity");
      }

      void* chunk_addr =
          reinterpret_cast<void*>(  // NOLINT(performance-no-int-to-ptr)
              static_cast<uintptr_t>(te.addr_) + offset);

      if (registeredMemMap_.contains(chunk_addr)) {
        LOG(ERROR) << "[CCA] SEGMENT_MAP: Memory already registered at 0x"
                   << std::hex << chunk_addr << std::dec
                   << " size=" << chunk_size
                   << " existing_size=" << registeredMemMap_.at(chunk_addr).len;
        throw std::runtime_error("Memory already registered with NCCLX");
      }

      registeredMemMap_.emplace(chunk_addr, MemInfo{chunk_size, te.device_});

      // Register the memory through ncclCommRegister
      for (auto& comm : registeredComms_) {
        if (te.device_ == comm->getDevice().index()) {
          comm->register_address(
              TorchCommNCCLX::AddressWithLen(chunk_addr, chunk_size));
        }
      }

      // Move to the next chunk
      offset += chunk_size;
    }
  } else if (
      te.action_ ==
      c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_FREE) {
    // Memory got freed, deregister it with NCCL
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    void* addr = reinterpret_cast<void*>(static_cast<uintptr_t>(te.addr_));

    if (!registeredMemMap_.contains(addr)) {
      LOG(ERROR) << "[CCA] SEGMENT_FREE: Memory not registered at 0x"
                 << std::hex << addr << std::dec << " size=" << te.size_;
      throw std::runtime_error("Memory not registered with NCCLX");
    } else {
      registeredMemMap_.erase(addr);
    }

    for (auto& comm : registeredComms_) {
      if (te.device_ == comm->getDevice().index()) {
        comm->deregister_address(TorchCommNCCLX::Address(addr));
      }
    }
  } else if (
      te.action_ ==
      c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_UNMAP) {
    // Memory got unmapped, deregister it with NCCL

    // PyTorch expandable segments have chunks sized according to the
    // allocation granularity. UNMAP events can cover multiple chunks.
    // We iterate through registered chunks within the unmapped range and
    // deregister each one.
    size_t total_size = te.size_;

    for (size_t offset = 0; offset < total_size;) {
      void* chunk_addr =
          reinterpret_cast<void*>(  // NOLINT(performance-no-int-to-ptr)
              static_cast<uintptr_t>(te.addr_) + offset);

      // Check if this chunk address is in our registered map
      auto it = registeredMemMap_.find(chunk_addr);
      if (it != registeredMemMap_.end() && it->second.device == te.device_) {
        // Found a registered chunk, deregister it
        size_t registered_size = it->second.len;

        for (auto& comm : registeredComms_) {
          if (te.device_ == comm->getDevice().index()) {
            comm->deregister_address(TorchCommNCCLX::Address(chunk_addr));
          }
        }

        registeredMemMap_.erase(it);

        // Move to the next potential chunk by the size we just deregistered
        offset += registered_size;
      } else {
        // No registered chunk found at this address - this indicates a
        // serious inconsistency between MAP and UNMAP events
        LOG(ERROR) << "[CCA] SEGMENT_UNMAP: No registered chunk at 0x"
                   << std::hex << chunk_addr << std::dec << " offset=" << offset
                   << " total_size=" << total_size;
        throw std::runtime_error(
            "SEGMENT_UNMAP: Expected registered chunk not found");
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
