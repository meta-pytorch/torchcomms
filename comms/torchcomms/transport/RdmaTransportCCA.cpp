// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/transport/RdmaTransportCCA.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <folly/logging/xlog.h>
#include <folly/synchronization/CallOnce.h>

#include "comms/ctran/regcache/RegCache.h"
#include "comms/utils/commSpecs.h"

namespace torch::comms {

namespace {

// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
folly::once_flag attachHookFlag;

// Trace callback invoked by the CUDA caching allocator on every segment
// alloc/free. Registers on alloc/map and deregisters on free/unmap.
void regDeregMem(const c10::cuda::CUDACachingAllocator::TraceEntry& te) {
  using Action = c10::cuda::CUDACachingAllocator::TraceEntry::Action;
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  void* addr = reinterpret_cast<void*>(static_cast<uintptr_t>(te.addr_));
  const size_t len = te.size_;
  const auto regCache = ctran::RegCache::getInstance();
  if (regCache == nullptr) {
    return;
  }
  if (te.action_ == Action::SEGMENT_ALLOC ||
      te.action_ == Action::SEGMENT_MAP) {
    const auto result = regCache->globalRegister(
        addr, len, /*forceReg=*/false, /*ncclManaged=*/false);
    if (result != commSuccess) {
      XLOGF(
          WARN,
          "[RDMA] Failed to register memory with RegCache (addr={}, len={})",
          addr,
          len);
    }
  } else if (
      te.action_ == Action::SEGMENT_FREE ||
      te.action_ == Action::SEGMENT_UNMAP) {
    const auto result = regCache->globalDeregister(addr, len);
    if (result != commSuccess) {
      XLOGF(
          WARN,
          "[RDMA] Failed to deregister memory with RegCache (addr={}, len={})",
          addr,
          len);
    }
  }
}

// Register all segments already allocated by the caching allocator before the
// trace hook was attached. Device is auto-detected from the pointer.
void registerMemPreHook() {
  const auto snapshot = c10::cuda::CUDACachingAllocator::snapshot();
  const auto regCache = ctran::RegCache::getInstance();
  if (regCache == nullptr) {
    return;
  }
  for (const auto& segmentInfo : snapshot.segments) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    void* addr = reinterpret_cast<void*>(segmentInfo.address);
    const size_t len = segmentInfo.total_size;
    const auto result = regCache->globalRegister(
        addr, len, /*forceReg=*/false, /*ncclManaged=*/false);
    if (result != commSuccess) {
      XLOGF(
          WARN,
          "[RDMA] Failed to register pre-existing memory with RegCache (addr={}, len={})",
          addr,
          len);
    }
  }
}

} // namespace

void attachRdmaMemoryHook() {
  folly::call_once(attachHookFlag, [] {
    at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
    registerMemPreHook();
    c10::cuda::CUDACachingAllocator::attachAllocatorTraceTracker(&regDeregMem);
  });
}

} // namespace torch::comms
