// Copyright (c) Meta Platforms, Inc. and affiliates.
// TorchComms Device API - NCCL Backend Implementation Header
//
// Device-side implementations for TorchComms using NCCL's GIN APIs.
// Header-only library - implementations are inline for template instantiation.
//
// IMPORTANT: This header contains CUDA device code and must ONLY be included
// from .cu files compiled with nvcc. For type aliases that can be used from
// non-CUDA code, include TorchCommDeviceNCCLXTypes.hpp instead.
//
// Usage:
//   #include "comms/torchcomms/device/ncclx/TorchCommDeviceNCCLX.cuh"
//
//   __global__ void myKernel(DeviceWindowNCCL win, ...) {
//     win.put(...);
//   }

#pragma once

// Guard to ensure this header is only compiled with nvcc
#ifndef __CUDACC__
#error \
    "TorchCommDeviceNCCLX.cuh must be compiled with nvcc. For type aliases, include TorchCommDeviceNCCLXTypes.hpp instead."
#endif

#include <cuda_runtime.h>

#include <nccl_device.h> // @manual=//comms/ncclx:nccl
#include <nccl_device/impl/comm__types.h> // @manual=//comms/ncclx:nccl_device_api

#include "comms/torchcomms/device/ncclx/TorchCommDeviceNCCLXTypes.hpp"

namespace torchcomms::device {

// =============================================================================
// Constants
// =============================================================================

constexpr int kDefaultGinContextIndex = 0;
constexpr int kDefaultSignalBits = 64;
constexpr int kDefaultCounterBits = 56;

// =============================================================================
// TorchCommDeviceWindow<NCCLDeviceBackend> RMA Operations
// =============================================================================

template <>
__device__ inline int TorchCommDeviceWindow<NCCLDeviceBackend>::put(
    size_t dst_offset,
    const RegisteredBuffer& src_buf,
    size_t src_offset,
    int dst_rank,
    size_t bytes,
    int signal_id,
    int counter_id) {
  // Get backend comm directly - no nested struct!
  const ncclDevComm& dev_comm = comm_;

  // Create GIN context
  ncclGin gin(dev_comm, kDefaultGinContextIndex);

  // Get window handles
  ncclWindow_t dst_win = window_;
  ncclWindow_t src_win = static_cast<ncclWindow_t>(src_buf.backend_window);

  // Determine signal and counter actions
  if (signal_id >= 0 && counter_id >= 0) {
    // Both signal and counter
    gin.put(
        ncclTeamWorld(dev_comm),
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
        ncclTeamWorld(dev_comm),
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
        ncclTeamWorld(dev_comm),
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
        ncclTeamWorld(dev_comm),
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
// TorchCommDeviceWindow<NCCLDeviceBackend> Signal Operations
// =============================================================================

template <>
__device__ inline int TorchCommDeviceWindow<NCCLDeviceBackend>::signal(
    int peer,
    int signal_id,
    SignalOp op,
    uint64_t value) {
  // Only ADD operation is supported by NCCL GIN
  // SET can be added later if NCCL adds support
  if (op != SignalOp::ADD) {
    return -1; // Unsupported signal operation
  }

  const ncclDevComm& dev_comm = comm_;
  ncclGin gin(dev_comm, kDefaultGinContextIndex);

  gin.signal(
      ncclTeamWorld(dev_comm),
      peer,
      ncclGin_SignalAdd{static_cast<ncclGinSignal_t>(signal_id), value},
      ncclCoopThread{});

  return 0;
}

template <>
__device__ inline int TorchCommDeviceWindow<NCCLDeviceBackend>::wait_signal(
    int signal_id,
    CmpOp cmp,
    uint64_t value) {
  // Only GE comparison is supported by NCCL GIN
  // Other comparison operators can be added later if needed
  if (cmp != CmpOp::GE) {
    return -1; // Unsupported comparison operator
  }

  const ncclDevComm& dev_comm = comm_;
  ncclGin gin(dev_comm, kDefaultGinContextIndex);

  gin.waitSignal(
      ncclCoopThread{},
      static_cast<ncclGinSignal_t>(signal_id),
      value,
      kDefaultSignalBits);

  return 0;
}

template <>
__device__ inline uint64_t
TorchCommDeviceWindow<NCCLDeviceBackend>::read_signal(int signal_id) const {
  const ncclDevComm& dev_comm = comm_;
  ncclGin gin(dev_comm, kDefaultGinContextIndex);

  return gin.readSignal(
      static_cast<ncclGinSignal_t>(signal_id), kDefaultSignalBits);
}

template <>
__device__ inline void TorchCommDeviceWindow<NCCLDeviceBackend>::reset_signal(
    int signal_id) {
  const ncclDevComm& dev_comm = comm_;
  ncclGin gin(dev_comm, kDefaultGinContextIndex);

  gin.resetSignal(static_cast<ncclGinSignal_t>(signal_id));
}

// =============================================================================
// TorchCommDeviceWindow<NCCLDeviceBackend> Counter Operations
// =============================================================================

template <>
__device__ inline int TorchCommDeviceWindow<NCCLDeviceBackend>::wait_local(
    int op_id,
    CmpOp cmp,
    uint64_t value) {
  // Only GE comparison is supported by NCCL GIN
  // Other comparison operators can be added later if needed
  if (cmp != CmpOp::GE) {
    return -1; // Unsupported comparison operator
  }

  const ncclDevComm& dev_comm = comm_;
  ncclGin gin(dev_comm, kDefaultGinContextIndex);

  gin.waitCounter(
      ncclCoopThread{},
      static_cast<ncclGinCounter_t>(op_id),
      value,
      kDefaultCounterBits);

  return 0;
}

template <>
__device__ inline uint64_t
TorchCommDeviceWindow<NCCLDeviceBackend>::read_counter(int counter_id) const {
  const ncclDevComm& dev_comm = comm_;
  ncclGin gin(dev_comm, kDefaultGinContextIndex);

  return gin.readCounter(
      static_cast<ncclGinCounter_t>(counter_id), kDefaultCounterBits);
}

template <>
__device__ inline void TorchCommDeviceWindow<NCCLDeviceBackend>::reset_counter(
    int counter_id) {
  const ncclDevComm& dev_comm = comm_;
  ncclGin gin(dev_comm, kDefaultGinContextIndex);

  gin.resetCounter(static_cast<ncclGinCounter_t>(counter_id));
}

// =============================================================================
// TorchCommDeviceWindow<NCCLDeviceBackend> Synchronization Operations
// =============================================================================

template <>
__device__ inline int TorchCommDeviceWindow<NCCLDeviceBackend>::fence() {
  // No-op for NCCL GIN backend.
  // NCCL GIN guarantees ordering: put and signal operations to the same peer
  // are delivered in order. No explicit fence is needed.
  // TODO: Implement proper fence when adding LSA (NVLink/PCIe direct) support.
  return 0;
}

template <>
__device__ inline int TorchCommDeviceWindow<NCCLDeviceBackend>::flush() {
  const ncclDevComm& dev_comm = comm_;
  ncclGin gin(dev_comm, kDefaultGinContextIndex);

  gin.flush(ncclCoopThread{});
  return 0;
}

template <>
__device__ inline int TorchCommDeviceWindow<NCCLDeviceBackend>::barrier(
    int barrier_id) {
  // NOT IMPLEMENTED - trap to prevent accidental usage.
  //
  // Full world-scope barrier requires host-side setup:
  //   - ncclTeamTagRail constructor only syncs ranks within the same NIC rail
  //   - Full constructor needs ncclGinBarrierHandle allocated at host via
  //     ncclGinBarrierCreateRequirement(comm, ncclTeamWorld, ...) BEFORE
  //     ncclDevCommCreate()
  //
  // Future: Allocate world-scope barrier handle at host, store in
  // TorchCommDeviceCommState, use full ncclGinBarrierSession constructor.
  (void)barrier_id;
  __trap();
  return -1; // Unreachable
}

} // namespace torchcomms::device
