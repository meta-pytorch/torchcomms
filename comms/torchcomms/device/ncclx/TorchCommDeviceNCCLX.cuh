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
#include <nccl_device/impl/comm__types.h> // @manual=//comms/ncclx:nccl

#include "comms/common/AtomicUtils.cuh"
#include "comms/pipes/CopyUtils.cuh"
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

// Build a pipes::ThreadGroup for the given CoopScope.
// Delegates to the pipes factory functions for each scope.
__device__ inline comms::pipes::ThreadGroup make_thread_group(CoopScope scope) {
  switch (scope) {
    case CoopScope::WARP:
      return comms::pipes::make_warp_group();
    case CoopScope::BLOCK:
      return comms::pipes::make_block_group();
    case CoopScope::THREAD:
      return comms::pipes::make_thread_solo();
  }
  // Unreachable — all CoopScope values are handled above.
  __builtin_unreachable();
}

// NVLink memcpy using pipes::memcpy_vectorized with the given cooperative
// scope. No volatile or ordering semantics — put() provides no ordering
// guarantees. Callers must use signal(), fence(), or flush() for store
// visibility.
__device__ inline void
memcpy_nvl(void* dst, const void* src, size_t bytes, CoopScope scope) {
  auto group = make_thread_group(scope);
  comms::pipes::memcpy_vectorized(
      static_cast<char*>(dst), static_cast<const char*>(src), bytes, group);
}

// Dispatch a callable with the appropriate NCCL coop type based on CoopScope.
// GIN methods are templated on coop type, so we need a dispatch function.
template <typename Func>
__device__ inline auto nccl_coop_dispatch(CoopScope scope, Func&& func) {
  switch (scope) {
    case CoopScope::WARP:
      return func(ncclCoopWarp{});
    case CoopScope::BLOCK:
      return func(ncclCoopCta{});
    case CoopScope::THREAD:
      return func(ncclCoopThread{});
  }
  // Unreachable — all CoopScope values are handled above.
  __builtin_unreachable();
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
    uint64_t value,
    CoopScope scope) {
  const ncclDevComm& dev_comm = comm_;

  if (ncclTeamRankIsMember(
          ncclTeamLsa(dev_comm), ncclTeamWorld(dev_comm), peer)) {
    // ---- LSA (NVLink) path ----
    // Signal is a single atomic/store — only thread 0 needs to execute it.
    // For warp/block scope, all threads reach this point but only thread 0
    // performs the actual write (same pattern as GIN internally).
    auto group = detail::make_thread_group(scope);
    if (group.is_leader()) {
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
    // atomicAdd posts a WQE and rings the doorbell inline — no flush needed.
    // QP ordering guarantees prior puts on this QP complete before this atomic.
    // User calls flush() explicitly if they need local completion.
    detail::nccl_coop_dispatch(scope, [&](auto coop) {
      gin.atomicAdd(
          ncclTeamWorld(dev_comm),
          peer,
          dev_comm.resourceWindow,
          offset,
          value,
          coop);
    });
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
    int counter_id,
    CoopScope scope) {
  const ncclDevComm& dev_comm = comm_;

  ncclWindow_t dst_win = window_;
  ncclWindow_t src_win = static_cast<ncclWindow_t>(src_buf.backend_window);

  if (ncclTeamRankIsMember(
          ncclTeamLsa(dev_comm), ncclTeamWorld(dev_comm), dst_rank)) {
    // ---- LSA (NVLink) path ----
    // Cooperative memcpy through NVLink-mapped pointers.
    void* src = ncclGetLocalPointer(src_win, src_offset);
    void* dst = ncclGetPeerPointer(dst_win, dst_offset, dst_rank);

    detail::memcpy_nvl(dst, src, bytes, scope);
    // No explicit fence needed here — the signal() call below uses
    // st.release.sys / atom.release.sys which orders all prior stores
    // (including the memcpy data writes) before the signal write.

    if (signal_id >= 0) {
      signal(dst_rank, signal_id, SignalOp::ADD, 1, scope);
    }
    // counter_id silently ignored for LSA — counters are GIN hardware only
  } else {
    // ---- GIN (RDMA) path ----
    ncclGin gin(dev_comm, kDefaultGinContextIndex);

    detail::nccl_coop_dispatch(scope, [&](auto coop) {
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
            coop);
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
            coop);
      }
    });

    if (signal_id >= 0) {
      signal(dst_rank, signal_id, SignalOp::ADD, 1, scope);
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
  // Compiler barrier: prevents the compiler from reordering put() calls
  // across this point. No hardware fence needed — GPU hardware does not
  // reorder stores within a single thread's instruction stream. Cross-GPU
  // visibility is handled by signal()'s release semantics (atom.release.sys).
  comms::device::compiler_barrier();
  return 0;
}

template <>
__device__ inline int TorchCommDeviceWindow<NCCLDeviceBackend>::flush(
    CoopScope scope) {
  // flush() = local completion: all threads in the cooperative group have
  // finished issuing their operations and the source buffers are safe to reuse.
  //
  // NVLink path: stores are inline (no async DMA). group.sync() is sufficient
  // — once all threads have passed the sync, every store they issued has
  // already executed. Cross-GPU visibility is NOT flush's responsibility;
  // signal()'s atom.release.sys / st.release.sys handles that.
  //
  // RDMA (GIN) path: puts are async WQEs posted to the NIC. gin.flush() spins
  // until the NIC signals local completion (source buffer safe to reuse).
  // gin.flush() internally handles any necessary group synchronization.
  auto group = detail::make_thread_group(scope);
  group.sync();

  const ncclDevComm& dev_comm = comm_;
  ncclGin gin(dev_comm, kDefaultGinContextIndex);
  detail::nccl_coop_dispatch(scope, [&](auto coop) { gin.flush(coop); });

  return 0;
}

template <>
__device__ inline int TorchCommDeviceWindow<NCCLDeviceBackend>::barrier(
    int barrier_id,
    CoopScope scope) {
  const ncclDevComm& dev_comm = comm_;
  ncclGin gin(dev_comm, kDefaultGinContextIndex);

  // World barrier: syncs LSA team (NVLink) first, then Rail team (RDMA)
  detail::nccl_coop_dispatch(scope, [&](auto coop) {
    ncclBarrierSession barrier(
        coop,
        ncclTeamTagWorld{},
        gin,
        static_cast<uint32_t>(barrier_id),
        false /* multimem */);
    barrier.sync(coop, cuda::memory_order_acq_rel, ncclGinFenceLevel::Relaxed);
  });

  return 0;
}

} // namespace torchcomms::device
