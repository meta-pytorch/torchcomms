// Copyright (c) Meta Platforms, Inc. and affiliates.
// TorchComms Triton Device API - nvcc-compiled implementations
//
// This file provides extern "C" wrappers around TorchCommDeviceWindow methods.
// Compiled with nvcc to support NCCLX GIN templates.
// Linked at compile time into Triton kernels via extern_libs bitcode.
//
// Design:
// - All functions take void* handles (TorchCommsWindowHandle,
// TorchCommsBufferHandle)
// - Internally cast to TorchCommDeviceWindow<NCCLGinBackend>* and
// RegisteredBuffer*
// - 1:1 mapping with TorchCommDeviceWindow methods
//
// IMPORTANT: Side-effecting functions (put, signal, wait, flush, reset) use a
// threadIdx.x == 0 guard because Triton's extern_elementwise emits a call per
// thread in the block.  With the default num_warps=4 that is 128 threads, so
// without the guard these operations would execute 128 times per kernel launch.
// Pure query functions (rank, num_ranks, base, size) are idempotent and don't
// need the guard.

#include <cuda_runtime.h>

#include "comms/torchcomms/device/ncclx/TorchCommDeviceNCCLX.cuh"

using namespace torchcomms::device;

using DeviceWindow = TorchCommDeviceWindow<NCCLGinBackend>;

extern "C" {

// =============================================================================
// RMA Operations
// =============================================================================

// torchcomms_put takes expanded RegisteredBuffer fields as arguments instead
// of a pointer to a GPU-allocated struct. This avoids GPU memory allocation
// conflicts with NCCLX's cuMemMap-based memory management.
//
// The RegisteredBuffer is constructed on the stack from the passed arguments.
__device__ int torchcomms_put(
    void* win_ptr,
    unsigned long long dst_offset,
    void* src_base_ptr,
    unsigned long long src_size,
    void* src_nccl_win,
    unsigned long long src_offset,
    int dst_rank,
    unsigned long long bytes,
    int signal_id,
    int counter_id) {
  // HACK: Triton's extern_elementwise vectorizes calls when arguments have
  // block shape. This guard ensures only thread 0 executes the actual put.
  // All other threads return early with success.
  if (threadIdx.x != 0) {
    return 0;
  }

  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);

  // Construct RegisteredBuffer on the stack from passed arguments
  RegisteredBuffer src_buf;
  src_buf.base_ptr = src_base_ptr;
  src_buf.size = static_cast<size_t>(src_size);
  src_buf.backend_window = src_nccl_win;

  return win->put(
      static_cast<size_t>(dst_offset),
      src_buf,
      static_cast<size_t>(src_offset),
      dst_rank,
      static_cast<size_t>(bytes),
      signal_id,
      counter_id);
}

// =============================================================================
// Signal Operations (Remote Notification)
// =============================================================================

__device__ int torchcomms_signal(
    void* win_ptr,
    int peer,
    int signal_id,
    unsigned long long value) {
  // HACK: Guard against Triton vectorization - only thread 0 executes
  if (threadIdx.x != 0) {
    return 0;
  }
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->signal(peer, signal_id, SignalOp::ADD, value);
}

__device__ int torchcomms_wait_signal(
    void* win_ptr,
    int signal_id,
    unsigned long long expected_value) {
  // HACK: Guard against Triton vectorization - only thread 0 executes
  if (threadIdx.x != 0) {
    return 0;
  }
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->wait_signal(signal_id, CmpOp::GE, expected_value);
}

__device__ unsigned long long torchcomms_read_signal(
    void* win_ptr,
    int signal_id) {
  // HACK: Guard against Triton vectorization - only thread 0 executes
  if (threadIdx.x != 0) {
    return 0;
  }
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->read_signal(signal_id);
}

__device__ void torchcomms_reset_signal(void* win_ptr, int signal_id) {
  // HACK: Guard against Triton vectorization - only thread 0 executes
  if (threadIdx.x != 0) {
    return;
  }
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  win->reset_signal(signal_id);
}

// =============================================================================
// Counter Operations (Local Completion)
// =============================================================================

__device__ int torchcomms_wait_local(
    void* win_ptr,
    int counter_id,
    unsigned long long expected_value) {
  // HACK: Guard against Triton vectorization - only thread 0 executes
  if (threadIdx.x != 0) {
    return 0;
  }
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->wait_local(counter_id, CmpOp::GE, expected_value);
}

__device__ unsigned long long torchcomms_read_counter(
    void* win_ptr,
    int counter_id) {
  // HACK: Guard against Triton vectorization - only thread 0 executes
  if (threadIdx.x != 0) {
    return 0;
  }
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->read_counter(counter_id);
}

__device__ void torchcomms_reset_counter(void* win_ptr, int counter_id) {
  // HACK: Guard against Triton vectorization - only thread 0 executes
  if (threadIdx.x != 0) {
    return;
  }
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  win->reset_counter(counter_id);
}

// =============================================================================
// Synchronization & Completion
// =============================================================================

__device__ int torchcomms_fence(void* win_ptr) {
  // HACK: Guard against Triton vectorization - only thread 0 executes
  if (threadIdx.x != 0) {
    return 0;
  }
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->fence();
}

__device__ int torchcomms_flush(void* win_ptr) {
  // HACK: Guard against Triton vectorization - only thread 0 executes
  if (threadIdx.x != 0) {
    return 0;
  }
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->flush();
}

// =============================================================================
// Window Properties
// =============================================================================

__device__ int torchcomms_rank(void* win_ptr) {
  // HACK: Guard against Triton vectorization - only thread 0 executes
  if (threadIdx.x != 0) {
    return 0;
  }
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->rank();
}

__device__ int torchcomms_num_ranks(void* win_ptr) {
  // HACK: Guard against Triton vectorization - only thread 0 executes
  if (threadIdx.x != 0) {
    return 0;
  }
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->num_ranks();
}

__device__ void* torchcomms_base(void* win_ptr) {
  // HACK: Guard against Triton vectorization - only thread 0 executes
  if (threadIdx.x != 0) {
    return nullptr;
  }
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return win->base();
}

__device__ unsigned long long torchcomms_size(void* win_ptr) {
  // HACK: Guard against Triton vectorization - only thread 0 executes
  if (threadIdx.x != 0) {
    return 0;
  }
  auto* win = reinterpret_cast<DeviceWindow*>(win_ptr);
  return static_cast<unsigned long long>(win->size());
}

} // extern "C"
