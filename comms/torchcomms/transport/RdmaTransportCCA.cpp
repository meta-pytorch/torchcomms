// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/transport/RdmaTransportCCA.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <folly/logging/xlog.h>
#include <folly/synchronization/CallOnce.h>

namespace torch::comms {

namespace {

// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
folly::once_flag attachHookFlag;

// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
RdmaRegFn g_reg = nullptr;
// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
RdmaRegFn g_dereg = nullptr;

// Trace callback invoked by the CUDA caching allocator on every segment
// alloc/free. Forwards alloc/map to g_reg and free/unmap to g_dereg.
void regDeregMem(const c10::cuda::CUDACachingAllocator::TraceEntry& te) {
  using Action = c10::cuda::CUDACachingAllocator::TraceEntry::Action;
  // NOLINTNEXTLINE(performance-no-int-to-ptr)
  void* addr = reinterpret_cast<void*>(static_cast<uintptr_t>(te.addr_));
  const size_t len = te.size_;
  if (te.action_ == Action::SEGMENT_ALLOC ||
      te.action_ == Action::SEGMENT_MAP) {
    if (g_reg == nullptr) {
      return;
    }
    const int result = g_reg(addr, len);
    if (result != 0) {
      XLOGF(
          WARN,
          "[RDMA] Failed to register memory (addr={}, len={})",
          addr,
          len);
    }
  } else if (
      te.action_ == Action::SEGMENT_FREE ||
      te.action_ == Action::SEGMENT_UNMAP) {
    if (g_dereg == nullptr) {
      return;
    }
    const int result = g_dereg(addr, len);
    if (result != 0) {
      XLOGF(
          WARN,
          "[RDMA] Failed to deregister memory (addr={}, len={})",
          addr,
          len);
    }
  }
}

// Register all segments already allocated by the caching allocator before the
// trace hook was attached. The reg callback receives only (addr, len); the
// owning device is auto-detected downstream via cudaPointerGetAttributes on the
// segment pointer, which resolves device memory to its true owning device
// regardless of the caller's current CUDA device, so no explicit device needs
// to be threaded through.
void registerMemPreHook(RdmaRegFn reg) {
  const auto snapshot = c10::cuda::CUDACachingAllocator::snapshot();
  for (const auto& segmentInfo : snapshot.segments) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    void* addr = reinterpret_cast<void*>(segmentInfo.address);
    const size_t len = segmentInfo.total_size;
    const int result = reg(addr, len);
    if (result != 0) {
      XLOGF(
          WARN,
          "[RDMA] Failed to register pre-existing memory (addr={}, len={})",
          addr,
          len);
    }
  }
}

} // namespace

void attachRdmaMemoryHook(RdmaRegFn reg, RdmaRegFn dereg) {
  folly::call_once(attachHookFlag, [reg, dereg] {
    g_reg = reg;
    g_dereg = dereg;
    at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
    // Pre-warm the reg callback's one-time environment init (ctran cvars,
    // logging, CUDA lib) on this thread, before the trace tracker is armed.
    // Otherwise the first reg() could run that init inside the allocator trace
    // callback while the allocator lock is held (re-entrancy/lock-order risk).
    // reg() with a null buffer is a no-op registration but still triggers init.
    if (reg != nullptr) {
      reg(nullptr, 0);
    }
    registerMemPreHook(reg);
    c10::cuda::CUDACachingAllocator::attachAllocatorTraceTracker(&regDeregMem);
    XLOGF(INFO, "[RDMA] Attached RDMA memory hook");
  });
}

void detachRdmaMemoryHook() {
  g_reg = nullptr;
  g_dereg = nullptr;
}

} // namespace torch::comms
