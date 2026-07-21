// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/util/Exception.h>
#include <nccl.h>
#include <pybind11/pybind11.h>

#include <atomic>
#include <cstdint>
#include <mutex>

namespace {

std::once_flag g_init_flag;
// Read from the is_initialized binding on arbitrary threads while do_init()
// writes it, so it must be atomic (std::call_once only synchronizes threads
// that pass through it).
std::atomic<bool> g_initialized{false};

void trace_hook(const c10::cuda::CUDACachingAllocator::TraceEntry& te) {
  auto* addr = reinterpret_cast<void*>(static_cast<uintptr_t>(te.addr_));
  auto len = te.size_;

  using Action = c10::cuda::CUDACachingAllocator::TraceEntry::Action;
  if (te.action_ == Action::SEGMENT_ALLOC ||
      te.action_ == Action::SEGMENT_MAP) {
    ncclResult_t result = ncclGlobalRegisterWithPtr(addr, len);
    if (result != ncclSuccess) {
      TORCH_WARN(
          "cca4py: ncclGlobalRegisterWithPtr failed: ",
          ncclGetErrorString(result));
    }
  } else if (
      te.action_ == Action::SEGMENT_FREE ||
      te.action_ == Action::SEGMENT_UNMAP) {
    ncclResult_t result = ncclGlobalDeregisterWithPtr(addr, len);
    if (result != ncclSuccess) {
      TORCH_WARN(
          "cca4py: ncclGlobalDeregisterWithPtr failed: ",
          ncclGetErrorString(result));
    }
  }
}

void register_existing_segments() {
  auto snapshot = c10::cuda::CUDACachingAllocator::snapshot();
  for (const auto& seg : snapshot.segments) {
    auto* addr = reinterpret_cast<void*>(seg.address);
    ncclResult_t result = ncclGlobalRegisterWithPtr(addr, seg.total_size);
    if (result != ncclSuccess) {
      TORCH_WARN(
          "cca4py: failed to register existing segment: ",
          ncclGetErrorString(result));
    }
  }
}

void do_init() {
  at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
  // Attach the trace hook before snapshotting existing segments so a segment
  // allocated concurrently during init isn't lost in the gap between the two.
  // A segment observed by both paths is registered twice, which the NCCL
  // registration layer dedupes.
  c10::cuda::CUDACachingAllocator::attachAllocatorTraceTracker(&trace_hook);
  register_existing_segments();
  g_initialized.store(true, std::memory_order_release);
}

} // namespace

PYBIND11_MODULE(cca4py, m) {
  m.doc() = "CCA4Py: PyTorch CUDACachingAllocator ↔ NCCL memory registration";

  m.def(
      "init_hook",
      []() { std::call_once(g_init_flag, do_init); },
      "Attach CCA trace hook for automatic NCCL memory registration. "
      "Safe to call multiple times (no-op after first).");

  m.def(
      "is_initialized",
      []() { return g_initialized.load(std::memory_order_acquire); },
      "Return True if the CCA hook has been initialized.");
}
