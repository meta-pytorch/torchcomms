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
  // Register all memory that has already been allocated by querying the
  // CUDACachingAllocator snapshot directly. This captures any allocations
  // that occurred before the trace hook was attached.
  //
  // We iterate through all segments in the snapshot. The
  // global_register_address API auto-detects the correct cudaDev from the
  // buffer pointer, so we don't need to filter by device here. Each segment
  // will be registered with its correct device automatically.
  auto snapshot = c10::cuda::CUDACachingAllocator::snapshot();
  for (const auto& segmentInfo : snapshot.segments) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    void* addr = reinterpret_cast<void*>(segmentInfo.address);
    size_t len = segmentInfo.total_size;
    TorchCommNCCLX::global_register_address(
        TorchCommNCCLX::AddressWithLen(addr, len), nccl_api_.get());
  }

  mem_pre_hook_registered_ = true;
}

void CachingAllocatorHookImpl::regDeregMem(
    const c10::cuda::CUDACachingAllocator::TraceEntry& te) {
  if (te.action_ ==
          c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_ALLOC ||
      te.action_ ==
          c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_MAP) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    void* addr = reinterpret_cast<void*>(static_cast<uintptr_t>(te.addr_));
    size_t len = te.size_;

    TorchCommNCCLX::global_register_address(
        TorchCommNCCLX::AddressWithLen{addr, len}, nccl_api_.get());
  } else if (
      te.action_ ==
          c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_FREE ||
      te.action_ ==
          c10::cuda::CUDACachingAllocator::TraceEntry::Action::SEGMENT_UNMAP) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    void* addr = reinterpret_cast<void*>(static_cast<uintptr_t>(te.addr_));
    size_t len = te.size_;

    TorchCommNCCLX::global_deregister_address(
        TorchCommNCCLX::AddressWithLen{addr, len}, nccl_api_.get());
  }
}

bool CachingAllocatorHookImpl::isMemPreHookRegistered() {
  return mem_pre_hook_registered_;
}

} // namespace torch::comms
