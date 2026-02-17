// Copyright (c) Meta Platforms, Inc. and affiliates.
// TorchComms Device API - Unified NCCL Backend Implementation (GIN + LSA)
//
// Device-side implementations for TorchComms using NCCL's GIN (RDMA) and
// LSA (NVLink) APIs. Each operation dispatches to the optimal transport
// based on peer reachability:
//   - LSA-reachable peers (same node): NVLink direct load/store
//   - Remote peers: GIN RDMA
//
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

#include "comms/common/AtomicUtils.cuh"
#include "comms/torchcomms/device/ncclx/TorchCommDeviceNCCLXTypes.hpp"

namespace torchcomms::device {

// =============================================================================
// Constants
// =============================================================================

constexpr int kDefaultGinContextIndex = 0;
constexpr int kDefaultSignalBits = 64;
constexpr int kDefaultCounterBits = 56;

// =============================================================================
// Internal Helpers
// =============================================================================

namespace detail {

// Compare two uint64_t values using the given comparison operator.
__device__ inline bool cmp_op(CmpOp cmp, uint64_t lhs, uint64_t rhs) {
  switch (cmp) {
    case CmpOp::EQ:
      return lhs == rhs;
    case CmpOp::NE:
      return lhs != rhs;
    case CmpOp::LT:
      return lhs < rhs;
    case CmpOp::LE:
      return lhs <= rhs;
    case CmpOp::GT:
      return lhs > rhs;
    case CmpOp::GE:
      return lhs >= rhs;
  }
  return false;
}

// Vectorized memcpy for NVLink stores (single-thread scope for now).
//
// Uses 128-bit (uint4) loads/stores with loop unrolling for better memory
// pipeline utilization, inspired by comms::pipes::memcpy_vectorized.
//
// No volatile or ordering semantics — put() provides no ordering guarantees.
// Callers must use signal(), fence(), or flush() for store visibility.
__device__ inline void memcpy_nvl(void* dst, const void* src, size_t bytes) {
  constexpr int kUnroll = 4;
  constexpr size_t kVecSize = sizeof(uint4);

  if (reinterpret_cast<uintptr_t>(dst) % kVecSize == 0 &&
      reinterpret_cast<uintptr_t>(src) % kVecSize == 0) {
    auto* __restrict__ d = static_cast<uint4*>(dst);
    auto* __restrict__ s = static_cast<const uint4*>(src);
    size_t nvecs = bytes / kVecSize;
    size_t nvecs_aligned = (nvecs / kUnroll) * kUnroll;

    for (size_t i = 0; i < nvecs_aligned; i += kUnroll) {
      uint4 v[kUnroll];
#pragma unroll
      for (int j = 0; j < kUnroll; j++) {
        v[j] = s[i + j];
      }
#pragma unroll
      for (int j = 0; j < kUnroll; j++) {
        d[i + j] = v[j];
      }
    }
    for (size_t i = nvecs_aligned; i < nvecs; i++) {
      d[i] = s[i];
    }

    size_t tail = bytes % kVecSize;
    if (tail > 0) {
      auto* db = reinterpret_cast<char*>(d + nvecs);
      auto* sb = reinterpret_cast<const char*>(s + nvecs);
      for (size_t i = 0; i < tail; i++) {
        db[i] = sb[i];
      }
    }
    return;
  }

  // Fallback: unaligned path using 8-byte copies
  auto* __restrict__ d = static_cast<uint64_t*>(dst);
  auto* __restrict__ s = static_cast<const uint64_t*>(src);
  size_t chunks = bytes / sizeof(uint64_t);
  for (size_t i = 0; i < chunks; i++) {
    d[i] = s[i];
  }
  size_t tail = bytes % sizeof(uint64_t);
  if (tail > 0) {
    auto* db = reinterpret_cast<char*>(d + chunks);
    auto* sb = reinterpret_cast<const char*>(s + chunks);
    for (size_t i = 0; i < tail; i++) {
      db[i] = sb[i];
    }
  }
}

// Flat index into the signal buffer: slots[signal_id * num_ranks + rank].
__device__ __forceinline__ size_t
signal_slot_index(int signal_id, int num_ranks, int rank) {
  return static_cast<size_t>(signal_id) * num_ranks + rank;
}

// Returns pointer to the first per-peer signal slot for |signal_id|.
__device__ inline uint64_t* signal_slot_base(
    const ncclDevComm& dev_comm,
    uint32_t signal_buffer_handle,
    int signal_id,
    int num_ranks) {
  void* local_buf =
      ncclGetResourceBufferLocalPointer(dev_comm, signal_buffer_handle);
  return reinterpret_cast<uint64_t*>(local_buf) +
      signal_slot_index(signal_id, num_ranks, 0);
}

} // namespace detail

// =============================================================================
// TorchCommDeviceWindow<NCCLDeviceBackend> Signal Operations
// =============================================================================
//
// Signals use per-peer resource buffer slots instead of GIN hardware signals.
// Layout: slots[signal_id * num_ranks + sender_world_rank] = uint64_t
// Each sender writes only to its own slot, avoiding cross-transport atomicity
// hazards between NVLink volatile stores and RDMA atomics.
//
// NOTE: signal() is defined before put() because put() calls signal() inline,
// and C++ requires explicit specializations to precede their first use.

template <>
__device__ inline int TorchCommDeviceWindow<NCCLDeviceBackend>::signal(
    int peer,
    int signal_id,
    SignalOp op,
    uint64_t value) {
  const ncclDevComm& dev_comm = comm_;

  if (ncclTeamRankIsMember(
          ncclTeamLsa(dev_comm), ncclTeamWorld(dev_comm), peer)) {
    // ---- LSA (NVLink) path ----
    // Write to peer's resource buffer via NVLink-mapped pointer.
    int lsa_peer = ncclTeamRankToTeam(
        ncclTeamLsa(dev_comm), ncclTeamWorld(dev_comm), peer);
    void* peer_buf = ncclGetResourceBufferLsaPointer(
        dev_comm, signal_buffer_handle_, lsa_peer);
    uint64_t* slot = reinterpret_cast<uint64_t*>(peer_buf) +
        detail::signal_slot_index(signal_id, num_ranks_, rank_);

    if (op == SignalOp::ADD) {
      // atom.release.sys.add.u64 — single NVLink atomic with release
      // semantics, ensuring all prior stores (data writes from put())
      // are visible before the counter increment.
      comms::device::atomic_fetch_add_release_sys_global(slot, value);
    } else {
      // st.release.sys — release store ensures all prior writes are
      // visible before the signal value lands on the peer.
      comms::device::st_release_sys_global(slot, value);
    }
  } else {
    // ---- GIN (RDMA) path ----
    // SET is not supported on RDMA (no atomic store opcode).
    if (op != SignalOp::ADD) {
      return -1;
    }

    ncclGin gin(dev_comm, kDefaultGinContextIndex);

    size_t offset = ncclGetResourceBufferOffset(signal_buffer_handle_) +
        detail::signal_slot_index(signal_id, num_ranks_, rank_) *
            sizeof(uint64_t);
    gin.atomicAdd(
        ncclTeamWorld(dev_comm),
        peer,
        dev_comm.resourceWindow,
        offset,
        value,
        ncclCoopThread{});
    gin.flush(ncclCoopThread{});
  }

  return 0;
}

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
  const ncclDevComm& dev_comm = comm_;

  ncclWindow_t dst_win = window_;
  ncclWindow_t src_win = static_cast<ncclWindow_t>(src_buf.backend_window);

  if (ncclTeamRankIsMember(
          ncclTeamLsa(dev_comm), ncclTeamWorld(dev_comm), dst_rank)) {
    // ---- LSA (NVLink) path ----
    // Direct memcpy through NVLink-mapped pointers.
    void* src = ncclGetLocalPointer(src_win, src_offset);
    void* dst = ncclGetPeerPointer(dst_win, dst_offset, dst_rank);

    detail::memcpy_nvl(dst, src, bytes);
    // No explicit fence needed here — the signal() call below uses
    // st.release.sys / atom.release.sys which orders all prior stores
    // (including the memcpy data writes) before the signal write.

    if (signal_id >= 0) {
      signal(dst_rank, signal_id, SignalOp::ADD, 1);
    }
    // counter_id silently ignored for LSA — counters are GIN hardware only
  } else {
    // ---- GIN (RDMA) path ----
    ncclGin gin(dev_comm, kDefaultGinContextIndex);

    if (counter_id >= 0) {
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

    if (signal_id >= 0) {
      signal(dst_rank, signal_id, SignalOp::ADD, 1);
    }
  }

  return 0;
}

template <>
__device__ inline int TorchCommDeviceWindow<NCCLDeviceBackend>::wait_signal(
    int signal_id,
    CmpOp cmp,
    uint64_t value) {
  const ncclDevComm& dev_comm = comm_;
  uint64_t* base = detail::signal_slot_base(
      dev_comm, signal_buffer_handle_, signal_id, num_ranks_);

  // Spin-poll with acquire loads
  // that once we see a signal value, all prior stores from the signaler
  // (i.e. the data written by put()) are visible to us.
  for (;;) {
    uint64_t sum = 0;
    for (int i = 0; i < num_ranks_; i++) {
      sum += comms::device::ld_acquire_sys_global(base + i);
    }
    if (detail::cmp_op(cmp, sum, value)) {
      return 0;
    }
  }
}

template <>
__device__ inline int
TorchCommDeviceWindow<NCCLDeviceBackend>::wait_signal_from(
    int peer,
    int signal_id,
    CmpOp cmp,
    uint64_t value) {
  const ncclDevComm& dev_comm = comm_;
  uint64_t* slot = detail::signal_slot_base(
                       dev_comm, signal_buffer_handle_, signal_id, num_ranks_) +
      peer;

  for (;;) {
    uint64_t val = comms::device::ld_acquire_sys_global(slot);
    if (detail::cmp_op(cmp, val, value)) {
      return 0;
    }
  }
}

template <>
__device__ inline uint64_t
TorchCommDeviceWindow<NCCLDeviceBackend>::read_signal(int signal_id) const {
  const ncclDevComm& dev_comm = comm_;
  uint64_t* base = detail::signal_slot_base(
      dev_comm, signal_buffer_handle_, signal_id, num_ranks_);

  uint64_t sum = 0;
  for (int i = 0; i < num_ranks_; i++) {
    sum += comms::device::ld_acquire_sys_global(base + i);
  }
  return sum;
}

template <>
__device__ inline void TorchCommDeviceWindow<NCCLDeviceBackend>::reset_signal(
    int signal_id) {
  const ncclDevComm& dev_comm = comm_;
  uint64_t* base = detail::signal_slot_base(
      dev_comm, signal_buffer_handle_, signal_id, num_ranks_);

  for (int i = 0; i < num_ranks_; i++) {
    comms::device::st_release_sys_global(base + i, 0ULL);
  }
}

// =============================================================================
// TorchCommDeviceWindow<NCCLDeviceBackend> Counter Operations
// =============================================================================
// Counters remain on GIN hardware — they track local DMA completion
// (NIC increments after source buffer read). Only meaningful for RDMA path.

template <>
__device__ inline int TorchCommDeviceWindow<NCCLDeviceBackend>::wait_local(
    int op_id,
    CmpOp cmp,
    uint64_t value) {
  if (cmp != CmpOp::GE) {
    // GIN hardware counters only support GE comparison.
    __trap();
    return -1; // Unreachable
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
  // fence.acq_rel.sys ensures all prior stores (NVLink data writes, signal
  // updates) are globally visible before subsequent operations. Strictly
  // stronger than release — use when you need a full barrier rather than
  // a paired release/acquire on a specific location.
  comms::device::fence_acq_rel_sys();
  return 0;
}

template <>
__device__ inline int TorchCommDeviceWindow<NCCLDeviceBackend>::flush() {
  // Ensure NVLink stores are globally visible, then flush any pending
  // GIN WQEs (RDMA work queue entries).
  comms::device::fence_acq_rel_sys();

  const ncclDevComm& dev_comm = comm_;
  ncclGin gin(dev_comm, kDefaultGinContextIndex);
  gin.flush(ncclCoopThread{});

  return 0;
}

template <>
__device__ inline int TorchCommDeviceWindow<NCCLDeviceBackend>::barrier(
    int barrier_id) {
  // NOT IMPLEMENTED — barrier requires LSA barrier handle allocation
  // at host side (ncclLsaBarrierCreateRequirement). Will be added in a
  // future phase.
  (void)barrier_id;
  __trap();
  return -1; // Unreachable
}

} // namespace torchcomms::device
