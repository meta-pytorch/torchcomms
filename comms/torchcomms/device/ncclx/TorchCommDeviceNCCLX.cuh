// Copyright (c) Meta Platforms, Inc. and affiliates.
// TorchComms Device API - NCCL GIN Backend Implementation Header
//
// This header provides device-side implementations for the TorchComms device
// API using NCCL's GPU-Initiated Networking (GIN) APIs.
//
// IMPORTANT: This is a HEADER-ONLY library. All implementations are inline
// because they use templated NCCL GIN APIs that must be instantiated in the
// context of the kernel that uses them.
//
// Usage:
//   #include "comms/torchcomms/device/ncclx/TorchCommDeviceNCCLX.cuh"
//
//   __global__ void myKernel(TorchCommDeviceWindow* win, ...) {
//     // Use win->put(), win->signal(), etc.
//   }

#pragma once

#include <cuda_runtime.h>

// NCCL device headers for GIN operations
#include <nccl_device/core.h> // @manual=//comms/ncclx:nccl
#include <nccl_device/gin.h> // @manual=//comms/ncclx:nccl

// TorchComms device API header
#include "comms/torchcomms/device/TorchCommDeviceComm.hpp"

namespace torch::comms::device {

// =============================================================================
// Constants
// =============================================================================

// Default GIN context index (single context mode)
constexpr int kDefaultGinContextIndex = 0;

// Default bits for signal/counter operations (full 64-bit)
constexpr int kDefaultSignalBits = 64;
constexpr int kDefaultCounterBits = 56; // NCCL limits counters to 56 bits

// =============================================================================
// TorchCommDeviceWindow Metadata Methods (Inline Implementations)
// =============================================================================

__device__ inline int TorchCommDeviceWindow::rank() const {
  return comm_->rank();
}

__device__ inline int TorchCommDeviceWindow::size() const {
  return comm_->size();
}

__device__ inline BackendType TorchCommDeviceWindow::backend_type() const {
  return comm_->backend_type();
}

// =============================================================================
// TorchCommDeviceWindow Property Methods (Inline Implementations)
// =============================================================================

__device__ inline void* TorchCommDeviceWindow::base_ptr() const {
  return local_base_;
}

__device__ inline size_t TorchCommDeviceWindow::window_size() const {
  return size_;
}

__device__ inline void* TorchCommDeviceWindow::peer_ptr(int peer) const {
  if (peer_ptrs_ == nullptr) {
    return nullptr;
  }
  if (peer < 0 || peer >= comm_->size()) {
    return nullptr;
  }
  return peer_ptrs_[peer];
}

// =============================================================================
// TorchCommDeviceWindow RMA Operations (Inline Implementations)
// =============================================================================

__device__ inline int TorchCommDeviceWindow::put(
    size_t dst_offset,
    const RegisteredBuffer& buf,
    size_t src_offset,
    int dst_rank,
    size_t bytes,
    int signal_id,
    int counter_id) {
  // Get backend state
  const ncclDevComm* dev_comm =
      static_cast<const ncclDevComm*>(comm_->backend_state_);

  // Create GIN context
  ncclGin gin(*dev_comm, kDefaultGinContextIndex);

  // Get window handles
  ncclWindow_t dst_win = static_cast<ncclWindow_t>(backend_handle_);
  ncclWindow_t src_win = static_cast<ncclWindow_t>(buf.backend_window);

  // Determine signal and counter actions
  if (signal_id >= 0 && counter_id >= 0) {
    // Both signal and counter
    gin.put(
        ncclTeamWorld(*dev_comm),
        dst_rank,
        dst_win,
        dst_offset,
        src_win,
        src_offset,
        bytes,
        ncclGin_SignalInc{static_cast<ncclGinSignal_t>(signal_id)},
        ncclGin_CounterInc{static_cast<ncclGinCounter_t>(counter_id)},
        ncclCoopThread{});
  } else if (signal_id >= 0) {
    // Signal only
    gin.put(
        ncclTeamWorld(*dev_comm),
        dst_rank,
        dst_win,
        dst_offset,
        src_win,
        src_offset,
        bytes,
        ncclGin_SignalInc{static_cast<ncclGinSignal_t>(signal_id)},
        ncclGin_None{},
        ncclCoopThread{});
  } else if (counter_id >= 0) {
    // Counter only
    gin.put(
        ncclTeamWorld(*dev_comm),
        dst_rank,
        dst_win,
        dst_offset,
        src_win,
        src_offset,
        bytes,
        ncclGin_None{},
        ncclGin_CounterInc{static_cast<ncclGinCounter_t>(counter_id)},
        ncclCoopThread{});
  } else {
    // Neither signal nor counter
    gin.put(
        ncclTeamWorld(*dev_comm),
        dst_rank,
        dst_win,
        dst_offset,
        src_win,
        src_offset,
        bytes,
        ncclGin_None{},
        ncclGin_None{},
        ncclCoopThread{});
  }

  return 0;
}

// =============================================================================
// TorchCommDeviceWindow Signal Operations (Inline Implementations)
// =============================================================================

__device__ inline int TorchCommDeviceWindow::signal(
    int peer,
    int signal_id,
    SignalOp op,
    uint64_t value) {
  // Only ADD operation is supported by NCCL GIN
  // SET can be added later if NCCL adds support
  if (op != SignalOp::ADD) {
    return -1; // Unsupported signal operation
  }

  const ncclDevComm* dev_comm =
      static_cast<const ncclDevComm*>(comm_->backend_state_);
  ncclGin gin(*dev_comm, kDefaultGinContextIndex);

  gin.signal(
      ncclTeamWorld(*dev_comm),
      peer,
      ncclGin_SignalAdd{static_cast<ncclGinSignal_t>(signal_id), value},
      ncclCoopThread{});

  return 0;
}

__device__ inline int
TorchCommDeviceWindow::wait_signal(int signal_id, CmpOp cmp, uint64_t value) {
  // Only GE comparison is supported by NCCL GIN
  // Other comparison operators can be added later if needed
  if (cmp != CmpOp::GE) {
    return -1; // Unsupported comparison operator
  }

  const ncclDevComm* dev_comm =
      static_cast<const ncclDevComm*>(comm_->backend_state_);
  ncclGin gin(*dev_comm, kDefaultGinContextIndex);

  gin.waitSignal(
      ncclCoopThread{},
      static_cast<ncclGinSignal_t>(signal_id),
      value,
      kDefaultSignalBits);

  return 0;
}

__device__ inline uint64_t TorchCommDeviceWindow::read_signal(
    int signal_id) const {
  const ncclDevComm* dev_comm =
      static_cast<const ncclDevComm*>(comm_->backend_state_);
  ncclGin gin(*dev_comm, kDefaultGinContextIndex);

  return gin.readSignal(
      static_cast<ncclGinSignal_t>(signal_id), kDefaultSignalBits);
}

__device__ inline void TorchCommDeviceWindow::reset_signal(int signal_id) {
  const ncclDevComm* dev_comm =
      static_cast<const ncclDevComm*>(comm_->backend_state_);
  ncclGin gin(*dev_comm, kDefaultGinContextIndex);

  gin.resetSignal(static_cast<ncclGinSignal_t>(signal_id));
}

// =============================================================================
// TorchCommDeviceWindow Counter Operations (Inline Implementations)
// =============================================================================

__device__ inline int
TorchCommDeviceWindow::wait_counter(int counter_id, CmpOp cmp, uint64_t value) {
  // Only GE comparison is supported by NCCL GIN
  // Other comparison operators can be added later if needed
  if (cmp != CmpOp::GE) {
    return -1; // Unsupported comparison operator
  }

  const ncclDevComm* dev_comm =
      static_cast<const ncclDevComm*>(comm_->backend_state_);
  ncclGin gin(*dev_comm, kDefaultGinContextIndex);

  gin.waitCounter(
      ncclCoopThread{},
      static_cast<ncclGinCounter_t>(counter_id),
      value,
      kDefaultCounterBits);

  return 0;
}

__device__ inline uint64_t TorchCommDeviceWindow::read_counter(
    int counter_id) const {
  const ncclDevComm* dev_comm =
      static_cast<const ncclDevComm*>(comm_->backend_state_);
  ncclGin gin(*dev_comm, kDefaultGinContextIndex);

  return gin.readCounter(
      static_cast<ncclGinCounter_t>(counter_id), kDefaultCounterBits);
}

__device__ inline void TorchCommDeviceWindow::reset_counter(int counter_id) {
  const ncclDevComm* dev_comm =
      static_cast<const ncclDevComm*>(comm_->backend_state_);
  ncclGin gin(*dev_comm, kDefaultGinContextIndex);

  gin.resetCounter(static_cast<ncclGinCounter_t>(counter_id));
}

// =============================================================================
// TorchCommDeviceWindow Synchronization Operations (Inline Implementations)
// =============================================================================

__device__ inline int TorchCommDeviceWindow::fence() {
  // No-op for NCCL GIN backend.
  // NCCL GIN guarantees ordering: put and signal operations to the same peer
  // are delivered in order. No explicit fence is needed.
  // TODO: Implement proper fence when adding LSA (NVLink/PCIe direct) support.
  return 0;
}

__device__ inline int TorchCommDeviceWindow::flush() {
  const ncclDevComm* dev_comm =
      static_cast<const ncclDevComm*>(comm_->backend_state_);
  ncclGin gin(*dev_comm, kDefaultGinContextIndex);

  gin.flush(ncclCoopThread{});
  return 0;
}

__device__ inline int TorchCommDeviceWindow::barrier(int barrier_id) {
  // No-op for now. Full world-scope barrier requires host-side setup.
  //
  // Why ncclGinBarrierSession can't be used directly:
  //   - ncclTeamTagRail constructor only syncs ranks within the same NIC rail
  //   - Full constructor needs ncclGinBarrierHandle allocated at host via
  //     ncclGinBarrierCreateRequirement(comm, ncclTeamWorld, ...) BEFORE
  //     ncclDevCommCreate()
  //
  // Future: Allocate world-scope barrier handle at host, store in
  // TorchCommDeviceComm_, use full ncclGinBarrierSession constructor.
  (void)barrier_id;
  return 0;
}

} // namespace torch::comms::device
