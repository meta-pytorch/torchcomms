// Copyright (c) Meta Platforms, Inc. and affiliates.
// TorchComm Device API - C++ Header
//
// This defines the device-side API for TorchComm that can be used from
// CUDA kernels (C++) or Triton (via extern functions).
//
// Design Philosophy:
//   Window is the first-class citizen. The device comm is internal and hidden.
//   Each device window has isolated signal/counter/barrier namespace.
//   1:1 mapping between host window and device window.
//
// Source Buffer Registration (NCCL GIN Backend):
//   To enable NON-COLLECTIVE local buffer registration, we use a split-comm
//   approach:
//
//   1. INITIALIZATION (collective, done ONCE per TorchComm):
//      Each rank calls ncclCommSplit(main_comm, myRank, 0, &local_comm)
//      - This is COLLECTIVE (all ranks must participate)
//      - Each rank gets its own 1-rank communicator
//      - The split comm SHARES ginState with parent (same ginComms[], ginCtx[])
//
//   2. LOCAL BUFFER REGISTRATION (non-collective, can be called anytime):
//      ncclCommWindowRegister(local_comm, buffer, size, ...)
//      - This is NON-COLLECTIVE because local_comm has nranks=1
//      - All bootstrap barriers become no-ops: if (nranks == 1) return
//      ncclSuccess
//      - The resulting ginWins[] are compatible with parent's windows
//
//   Benefits:
//     - Source buffer can be ANY GPU memory (not restricted to window region)
//     - No coordination with other ranks required for registration
//     - ginWins[] are valid for PUT operations through parent's DevComm

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace torch::comms::device {

// =============================================================================
// Forward Declarations
// =============================================================================

class TorchCommDeviceWindow;
struct RegisteredBuffer;

// Internal - not exposed to user
class TorchCommDeviceComm_;

// =============================================================================
// Enums
// =============================================================================

enum class BackendType : int {
  NCCL_GIN = 0,
  NVSHMEM = 1,
};

// Signal operation type - how to update the signal value
enum class SignalOp : int {
  SET = 0, // signal = value
  ADD = 1, // signal += value
};

// Comparison operation type - how to compare signal/counter values in wait
enum class CmpOp : int {
  EQ = 0, // ==
  NE = 1, // !=
  LT = 2, // <
  LE = 3, // <=
  GT = 4, // >
  GE = 5, // >= (most common, default for wait operations)
};

// =============================================================================
// RegisteredBuffer - Handle for Local Registered Source Buffers
// =============================================================================
//
// Represents a registered local memory region that can be used as a source
// for RMA put operations. Created on host via
// hostWindow.register_local_buffer().
//
// IMPORTANT: RegisteredBuffer must be used with the SAME DeviceWindow that
// created it. This is required because:
//   1. The backend_window was registered via the host window's local_comm
//   2. local_comm shares ginState with parent (same ginComms[], ginCtx[])
//   3. The ginWins[] array is indexed by contextId from the DevComm doing the
//   PUT
//   4. Both src and dst windows must have compatible ginWins[] (same ginState)
//
// Implementation (NCCL GIN):
//   - During TorchComm/TorchCommWindow creation (ONCE, collective):
//     ncclCommSplit(parent_comm, myRank, 0, &local_comm)
//   - During register_local_buffer() (NON-COLLECTIVE, can call anytime):
//     ncclCommWindowRegister(local_comm, buf, size, 0, &src_win)
//   - The local_comm is reused for all subsequent registrations
//
// Usage:
//   // Host side (register_local_buffer is NON-COLLECTIVE)
//   RegisteredBuffer buf1 = hostWindow.register_local_buffer(tensor1);
//   RegisteredBuffer buf2 = hostWindow.register_local_buffer(tensor2);
//
//   // Kernel launch
//   my_kernel<<<grid, block>>>(devWindow, buf1, buf2, ...);
//
//   // Device side (MUST use same devWindow that created the buffers)
//   devWindow->put(dst_offset, buf1, src_offset, dst_rank, bytes, ...);

struct RegisteredBuffer {
  void* base_ptr; // Base pointer of registered buffer
  size_t size; // Size of registered buffer in bytes
  void*
      backend_window; // Backend-specific window handle (e.g., ncclWindow_t)
                      // Created via local_comm for non-collective registration

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
// The primary device-side handle for all communication and synchronization.
// Created from host window via hostWindow.get_device_window().
//
// Contains:
//   - Window metadata (rank, size, base_ptr, window_size)
//   - RMA operations (put)
//   - Synchronization primitives (signal, counter, barrier, fence, flush)
//
// Signal vs Counter:
//   - Signal: Written to REMOTE peer's memory. Used to notify remote that
//             data has arrived (remote completion notification).
//   - Counter: Written to LOCAL memory. Updated when local operation completes
//              (source buffer consumed, safe to reuse).
//
// Host Window Initialization (NCCL GIN):
//   When a TorchCommWindow is created, it:
//     1. Creates the destination window via ncclCommWindowRegister (COLLECTIVE)
//     2. Creates a 1-rank local_comm via ncclCommSplit (COLLECTIVE, but done
//     once)
//   The local_comm is then reused for all subsequent register_local_buffer()
//   calls.
//
// Usage pattern:
//   // Host side - window creation (COLLECTIVE)
//   TorchCommWindow hostWin = comm.new_window(dst_tensor);
//
//   // Host side - local buffer registration (NON-COLLECTIVE, can call anytime)
//   RegisteredBuffer buf1 = hostWin.register_local_buffer(tensor1);
//   RegisteredBuffer buf2 = hostWin.register_local_buffer(tensor2);
//
//   // Get device handle
//   TorchCommDeviceWindow* devWin = hostWin.get_device_window(
//       signal_count, counter_count, barrier_count);
//
//   // Kernel launch
//   my_kernel<<<grid, block>>>(devWin, buf1, buf2, ...);
//
//   // Device side - Sender
//   __global__ void sender_kernel(TorchCommDeviceWindow* win,
//                                 RegisteredBuffer buf, ...) {
//       win->put(dst_offset, buf, src_offset, peer, bytes,
//                signal_id, counter_id);
//       // Wait for source buffer safe (counter >= 1)
//       win->wait_counter(counter_id, CmpOp::GE, 1);
//   }
//
//   // Device side - Receiver
//   __global__ void receiver_kernel(TorchCommDeviceWindow* win, ...) {
//       // Wait for data arrival (signal >= 1)
//       win->wait_signal(signal_id, CmpOp::GE, 1);
//   }

class TorchCommDeviceWindow {
 public:
  // =========================================================================
  // Metadata (read-only on device)
  // =========================================================================

  // Returns the rank of this process in the communicator
  __device__ int rank() const;

  // Returns the number of ranks in the communicator (consistent with
  // TorchComm::getSize())
  __device__ int size() const;

  // Returns the backend type (NCCL_GIN or NVSHMEM)
  __device__ BackendType backend_type() const;

  // =========================================================================
  // Window Properties
  // =========================================================================

  // Base pointer of this window's local buffer
  __device__ void* base_ptr() const;

  // Size of this window in bytes
  __device__ size_t window_size() const;

  // Get direct pointer for a specific peer's window (for NVLink load/store)
  // Returns nullptr if peer is not directly accessible (cross-node)
  __device__ void* peer_ptr(int peer) const;

  // =========================================================================
  // RMA Operations - Put
  // =========================================================================
  //
  // Put data from local registered buffer to this window on remote peer.
  //
  // IMPORTANT: The RegisteredBuffer MUST have been created by the same host
  // window that created this DeviceWindow. This ensures:
  //   1. buf.backend_window was registered via a split comm sharing ginState
  //   2. ginWins[] arrays are compatible (same NIC contexts)
  //   3. contextId indexing works correctly
  //
  // Implementation (NCCL GIN):
  //   ncclGinPut(this->backend_handle_, buf.backend_window, dst_offset, ...)
  //   Both windows' ginWins[contextId] are valid because they share ginState.
  //
  // signal_id: -1 = no signal, >=0 = signal peer after data settles remotely
  // counter_id: -1 = no counter, >=0 = increment local counter when source
  //             consumed
  //
  // Returns: 0 on success, non-zero on error

  __device__ int put(
      size_t dst_offset, // Offset into THIS window (remote destination)
      const RegisteredBuffer& buf, // Local registered source buffer
      size_t src_offset, // Offset into source buffer
      int dst_rank, // Destination peer rank
      size_t bytes,
      int signal_id = -1, // -1 = no signal, >=0 = signal peer
      int counter_id = -1 // -1 = no counter, >=0 = increment local counter
  );

  // =========================================================================
  // Signaling Operations (Remote Notification)
  // =========================================================================
  //
  // Signals are stored in REMOTE peer's memory.
  // Used to notify a peer that data has arrived/settled.
  //
  // Model: Each rank has a signal "inbox" indexed by signal_id.
  // Remote ranks can write to your inbox to notify you.
  //
  // Signal namespace is isolated per DeviceWindow.

  // Send signal to peer (write to peer's signal[signal_id])
  // op: SET writes value directly, ADD atomically adds value
  __device__ int signal(
      int peer,
      int signal_id,
      SignalOp op = SignalOp::ADD,
      uint64_t value = 1);

  // Wait for local signal to satisfy comparison
  // (Waits for remote peer to signal us)
  // cmp: comparison operator (EQ, NE, LT, LE, GT, GE)
  // GE is most common for "wait until signal >= value"
  __device__ int wait_signal(int signal_id, CmpOp cmp, uint64_t value);

  // Read current signal value (non-blocking)
  __device__ uint64_t read_signal(int signal_id) const;

  // Reset signal to 0 (use between phases)
  __device__ void reset_signal(int signal_id);

  // =========================================================================
  // Counter Operations (Local Completion)
  // =========================================================================
  //
  // Counters are stored in LOCAL memory.
  // Updated when a local operation completes (source buffer consumed).
  // Used to track when it's safe to reuse source buffers.
  //
  // Flow:
  //   1. put(..., counter_id=X) - posts RMA with counter
  //   2. When source data is consumed, counter X is incremented
  //   3. wait_counter(X, N) - waits until counter >= N
  //
  // Counter namespace is isolated per DeviceWindow.

  // Wait for local counter to satisfy comparison
  // (Waits for N puts with this counter_id to complete locally)
  // cmp: comparison operator (EQ, NE, LT, LE, GT, GE)
  // GE is most common for "wait until counter >= value"
  __device__ int wait_counter(int counter_id, CmpOp cmp, uint64_t value);

  // Read current counter value (non-blocking)
  __device__ uint64_t read_counter(int counter_id) const;

  // Reset counter to 0 (use between phases)
  __device__ void reset_counter(int counter_id);

  // =========================================================================
  // Synchronization & Completion
  // =========================================================================

  // Fence: Ordering guarantee - ensures prior remote writes to same peer
  //        are visible before subsequent writes.
  //        NCCL: implicit in signal, NVSHMEM: nvshmem_fence()
  __device__ int fence();

  // Flush: Local completion - ensures all prior async operations have
  //        completed. After this, ALL send buffers are safe to reuse.
  //        This is a heavier operation than waiting on individual counters.
  //        NCCL: gin.flush(), NVSHMEM: nvshmem_quiet()
  __device__ int flush();

  // Barrier: Synchronize all ranks using this window
  // Uses signal-based distributed counting internally
  // Barrier namespace is isolated per DeviceWindow.
  __device__ int barrier(int barrier_id);

 private:
  // Internal device comm (hidden from user)
  // Contains backend handle, transport state, signal/counter/barrier arrays
  TorchCommDeviceComm_* comm_;

  // Window properties
  void* local_base_;
  size_t size_;
  void** peer_ptrs_; // Array of peer base pointers (for NVLink)

  // Backend-specific window handle (e.g., ncclWindow_t)
  // This is the destination window - registered via main N-rank communicator
  void* backend_handle_;
};

// =============================================================================
// TorchCommDeviceComm_ - Internal Device Communicator (Not User-Facing)
// =============================================================================
//
// This is an internal structure that holds the device-side communication state.
// Users do not interact with this directly - it is managed by
// TorchCommDeviceWindow.
//
// Contains:
//   - Rank/size metadata
//   - Backend handle (ncclDevComm* + ncclGin_C* or nvshmem state)
//   - Signal/counter/barrier state arrays
//
// Created when hostWindow.get_device_window() is called.
// One DeviceComm_ instance per DeviceWindow (1:1 mapping with isolated state).

class TorchCommDeviceComm_ {
 public:
  __device__ int rank() const {
    return rank_;
  }
  __device__ int size() const {
    return size_;
  }
  __device__ BackendType backend_type() const {
    return backend_type_;
  }

 private:
  friend class TorchCommDeviceWindow;

  BackendType backend_type_;
  int rank_;
  int size_;
  void* backend_handle_; // ncclDevComm* + ncclGin_C* or nvshmem state

  // Signal state (for remote notification)
  uint64_t* signals_; // Local signal inbox (written by remote peers)
  int signal_count_;

  // Counter state (for local completion)
  uint64_t* counters_; // Local counters (written by local HCA/NIC)
  int counter_count_;

  // Barrier state
  uint32_t* barrier_epochs_;
  int barrier_count_;
};

} // namespace torch::comms::device
