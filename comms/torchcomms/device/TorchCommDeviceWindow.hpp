// Copyright (c) Meta Platforms, Inc. and affiliates.
// TorchComm Device API - C++ Header
//
// Device-side API for TorchComm that can be used from CUDA kernels or Triton.
//
// Design:
//   - Window is the first-class citizen; device comm is internal
//   - Each device window has isolated signal/counter/barrier namespace
//   - 1:1 mapping between host window and device window
//
// Backend Types (from DeviceBackendTraits.hpp):
//   - NCCLDeviceBackend: Unified NCCL backend (GIN + LSA)
//   - Future: NVSHMEMBackend, etc.
//
// Type Aliases (defined in backend-specific headers):
//   - DeviceWindowNCCL = TorchCommDeviceWindow<NCCLDeviceBackend>
//   - RegisteredBufferNCCL = RegisteredBuffer

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace torchcomms::device {

// =============================================================================
// Forward Declarations
// =============================================================================

template <typename Backend>
class TorchCommDeviceWindow;
struct RegisteredBuffer;

// =============================================================================
// Enums
// =============================================================================

enum class SignalOp : int {
  SET = 0, // signal = value
  ADD = 1, // signal += value
};

enum class CmpOp : int {
  EQ = 0, // ==
  NE = 1, // !=
  LT = 2, // <
  LE = 3, // <=
  GT = 4, // >
  GE = 5, // >= (most common for wait operations)
};

// Cooperative scope for device-side operations.
// Determines how many threads participate in each API call.
//
// Usage: Pass as parameter to scope-aware API overloads.
//   - THREAD: Single thread (default, backward-compatible)
//   - WARP:   All 32 threads in a warp must call together
//   - BLOCK:  All threads in a block must call together
//
// For WARP/BLOCK scope, the kernel launch config must provide enough
// threads (>= 32 for WARP, >= blockDim.x for BLOCK).
enum class CoopScope : int {
  THREAD = 0,
  WARP = 1,
  BLOCK = 2,
};

// =============================================================================
// RegisteredBuffer - Handle for Local Registered Source Buffers
// =============================================================================
//
// Represents a registered local memory region for RMA put operations.
// Created on host via hostWindow.register_local_buffer().
//
// IMPORTANT: Must be used with the SAME DeviceWindow that created it.

struct RegisteredBuffer {
  void* base_ptr;
  size_t size;
  void* backend_window; // Backend-specific window handle (e.g., ncclWindow_t)

  __device__ void* ptr() const {
    return base_ptr;
  }
  __device__ size_t buffer_size() const {
    return size;
  }
};

// =============================================================================
// TorchCommDeviceWindow - Device-side Window Handle
// =============================================================================
//
// Primary device-side handle for communication and synchronization.
// Created from host window via hostWindow.get_device_window().
//
// Passed by value to CUDA kernels (CUDA copies automatically):
//   DeviceWindowNCCL win = hostWindow.get_device_window();
//   myKernel<<<grid, block>>>(win, ...);
//
// Template parameter Backend provides backend traits (Comm, Window types).
//
// Contains:
//   - Window metadata (rank, num_ranks, base, size)
//   - RMA operations (put)
//   - Synchronization primitives (signal, counter, barrier, fence, flush)
//
// Signal vs Counter:
//   - Signal: Written to REMOTE peer's memory (data arrival notification)
//   - Counter: Written to LOCAL memory (source buffer consumed notification)

template <typename Backend>
class TorchCommDeviceWindow {
 public:
  // =========================================================================
  // Constructor
  // =========================================================================

  // Default constructor for backwards compatibility
  TorchCommDeviceWindow() = default;

  // Constructor with initializer list - preferred for explicit initialization
  TorchCommDeviceWindow(
      typename Backend::Comm comm,
      typename Backend::Window window,
      void* base,
      size_t size,
      int rank,
      int num_ranks,
      uint32_t signal_buffer_handle = 0)
      : comm_(comm),
        window_(window),
        base_(base),
        size_(size),
        rank_(rank),
        num_ranks_(num_ranks),
        signal_buffer_handle_(signal_buffer_handle) {}

  // =========================================================================
  // Metadata
  // =========================================================================

  __device__ int rank() const {
    return rank_;
  }
  __device__ int num_ranks() const {
    return num_ranks_;
  }

  // =========================================================================
  // Window Properties
  // =========================================================================

  __device__ void* base() const {
    return base_;
  }
  __device__ size_t size() const {
    return size_;
  }

  // =========================================================================
  // Backend Access
  // =========================================================================

  __device__ const typename Backend::Comm& comm() const {
    return comm_;
  }
  __device__ typename Backend::Window window() const {
    return window_;
  }

  // =========================================================================
  // Signal Buffer Handle
  // =========================================================================

  __device__ uint32_t signal_buffer_handle() const {
    return signal_buffer_handle_;
  }

  // =========================================================================
  // RMA Operations - Put
  // =========================================================================

  // Put data from local src_buf to dst_rank's window.
  // All threads in the cooperative group must call together when scope !=
  // THREAD.
  __device__ int put(
      size_t dst_offset,
      const RegisteredBuffer& src_buf,
      size_t src_offset,
      int dst_rank,
      size_t bytes,
      int signal_id = -1,
      int counter_id = -1,
      CoopScope scope = CoopScope::THREAD);

  // =========================================================================
  // Signaling Operations (Remote Notification)
  // =========================================================================

  __device__ int signal(
      int peer,
      int signal_id,
      SignalOp op = SignalOp::ADD,
      uint64_t value = 1,
      CoopScope scope = CoopScope::THREAD);

  __device__ int wait_signal(int signal_id, CmpOp cmp, uint64_t value);
  __device__ int
  wait_signal_from(int peer, int signal_id, CmpOp cmp, uint64_t value);
  __device__ uint64_t read_signal(int signal_id) const;
  __device__ void reset_signal(int signal_id);

  // =========================================================================
  // Counter Operations (Local Completion)
  // =========================================================================

  __device__ int wait_local(int op_id, CmpOp cmp, uint64_t value);
  __device__ uint64_t read_counter(int counter_id) const;
  __device__ void reset_counter(int counter_id);

  // =========================================================================
  // Synchronization & Completion
  // =========================================================================

  __device__ int fence();
  __device__ int flush(CoopScope scope = CoopScope::THREAD);
  __device__ int barrier(int barrier_id, CoopScope scope = CoopScope::THREAD);

  // =========================================================================
  // Data Members
  // =========================================================================

  typename Backend::Comm comm_; // e.g., ncclDevComm
  typename Backend::Window window_; // e.g., ncclWindow_t
  void* base_; // Local window base pointer
  size_t size_; // Window size in bytes
  int rank_;
  int num_ranks_;
  uint32_t
      signal_buffer_handle_; // Resource buffer handle for per-peer signal slots
};

// Type alias (also defined in backend-specific headers for convenience)
// Note: Requires DeviceBackendTraits.hpp to be included first

} // namespace torchcomms::device
