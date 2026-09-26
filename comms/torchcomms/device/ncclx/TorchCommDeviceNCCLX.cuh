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

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <nccl_device.h> // @manual=//comms/ncclx:nccl
#include <nccl_device/impl/comm__types.h> // @manual=//comms/ncclx:nccl

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

template <typename Coop>
__device__ __forceinline__ void coop_sync(Coop& coop) {
  comms::device::compiler_barrier();
  coop.sync();
}

template <typename T, int kUnroll = 8>
__device__ __forceinline__ void memcpy_nvl_aligned(
    T* __restrict__ dst,
    const T* __restrict__ src,
    size_t count,
    int thread_rank,
    int thread_count) {
  const size_t stride = thread_count * kUnroll;
  const size_t aligned_count = (count / stride) * stride;

  for (size_t i = thread_rank; i < aligned_count; i += stride) {
    T values[kUnroll];
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      values[j] = src[i + j * thread_count];
    }
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      dst[i + j * thread_count] = values[j];
    }
  }

  for (size_t i = aligned_count + thread_rank; i < count; i += thread_count) {
    dst[i] = src[i];
  }
}

__device__ __forceinline__ void memcpy_nvl_cooperative(
    char* dst,
    const char* src,
    size_t bytes,
    int thread_rank,
    int thread_count) {
  if (bytes == 0 || dst == src) {
    return;
  }

  const uintptr_t dst_begin = reinterpret_cast<uintptr_t>(dst);
  const uintptr_t src_begin = reinterpret_cast<uintptr_t>(src);
  const uintptr_t dst_end = dst_begin + bytes;
  const uintptr_t src_end = src_begin + bytes;
  if (dst_begin < src_end && src_begin < dst_end) {
    if (thread_rank == 0) {
      printf("TorchComms NVLink copy does not support overlapping ranges\n");
    }
    __trap();
  }

  constexpr size_t kAlignment = sizeof(uint4);
  if (dst_begin % kAlignment == 0 && src_begin % kAlignment == 0) {
    const size_t vector_count = bytes / kAlignment;
    auto* dst_vector = reinterpret_cast<uint4*>(dst);
    const auto* src_vector = reinterpret_cast<const uint4*>(src);
    memcpy_nvl_aligned(
        dst_vector, src_vector, vector_count, thread_rank, thread_count);
    const size_t copied = vector_count * kAlignment;
    dst += copied;
    src += copied;
    bytes -= copied;
  }

  memcpy_nvl_aligned(dst, src, bytes, thread_rank, thread_count);
}

// The copy itself has no ordering semantics. Callers provide any required
// synchronization before publishing or reusing the buffer.
__device__ inline void
memcpy_nvl(void* dst, const void* src, size_t bytes, CoopScope scope) {
  int thread_rank = 0;
  int thread_count = 0;
  // TorchComms device kernels use one-dimensional thread blocks.
  switch (scope) {
    case CoopScope::WARP:
      thread_rank = threadIdx.x % 32;
      thread_count = 32;
      break;
    case CoopScope::BLOCK:
      thread_rank = threadIdx.x;
      thread_count = blockDim.x;
      break;
    case CoopScope::THREAD:
      thread_rank = 0;
      thread_count = 1;
      break;
  }
  if (thread_count == 0) {
    __builtin_unreachable();
  }
  memcpy_nvl_cooperative(
      static_cast<char*>(dst),
      static_cast<const char*>(src),
      bytes,
      thread_rank,
      thread_count);
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
    detail::nccl_coop_dispatch(scope, [&](auto coop) {
      if (coop.thread_rank() == 0) {
        int lsa_peer = ncclTeamRankToTeam(
            ncclTeamLsa(dev_comm), ncclTeamWorld(dev_comm), peer);
        void* peer_buf = ncclGetResourceBufferLsaPointer(
            dev_comm, signal_buffer_handle_, lsa_peer);
        uint64_t* slot = reinterpret_cast<uint64_t*>(peer_buf) +
            detail::signal_slot_index(signal_id, num_ranks_, rank_);

        if (op == SignalOp::ADD) {
          // atom.release.sys.add.u64 — release for the issuing thread.
          comms::device::atomic_fetch_add_release_sys_global(slot, value);
        } else {
          // st.release.sys — release for the issuing thread.
          comms::device::st_release_sys_global(slot, value);
        }
      }
    });
  } else {
    // ---- GIN (RDMA) path ----
    if (!gin_enabled_) {
      return -1;
    }

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
    const torch::comms::RegisteredBuffer& src_buf,
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
    if (src_buf.base_ptr == nullptr) {
      return -1;
    }
    void* src =
        static_cast<void*>(static_cast<char*>(src_buf.base_ptr) + src_offset);
    void* dst = ncclGetPeerPointer(dst_win, dst_offset, dst_rank);

    detail::memcpy_nvl(dst, src, bytes, scope);
    if (signal_id >= 0) {
      signal(dst_rank, signal_id, SignalOp::ADD, 1, scope);
    }
    // counter_id silently ignored for LSA — counters are GIN hardware only
  } else {
    // ---- GIN (RDMA) path ----
    if (!gin_enabled_ || src_win == nullptr) {
      return -1;
    }

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
    uint64_t value,
    CoopScope scope) {
  detail::nccl_coop_dispatch(scope, [&](auto coop) {
    if (coop.thread_rank() == 0) {
      const ncclDevComm& dev_comm = comm_;
      uint64_t* base = detail::signal_slot_base(
          dev_comm, signal_buffer_handle_, signal_id, num_ranks_);

      // Spin-poll the signal slots with acquire loads.
      for (;;) {
        uint64_t sum = 0;
        for (int i = 0; i < num_ranks_; i++) {
          sum += comms::device::ld_acquire_sys_global(base + i);
        }
        if (detail::cmp_op(cmp, sum, value)) {
          break;
        }
      }
    }
    detail::coop_sync(coop);
  });
  return 0;
}

template <>
__device__ inline int
TorchCommDeviceWindow<NCCLDeviceBackend>::wait_signal_from(
    int peer,
    int signal_id,
    CmpOp cmp,
    uint64_t value,
    CoopScope scope) {
  detail::nccl_coop_dispatch(scope, [&](auto coop) {
    if (coop.thread_rank() == 0) {
      const ncclDevComm& dev_comm = comm_;
      uint64_t* slot =
          detail::signal_slot_base(
              dev_comm, signal_buffer_handle_, signal_id, num_ranks_) +
          peer;

      for (;;) {
        uint64_t val = comms::device::ld_acquire_sys_global(slot);
        if (detail::cmp_op(cmp, val, value)) {
          break;
        }
      }
    }
    detail::coop_sync(coop);
  });
  return 0;
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
    int signal_id,
    CoopScope scope) {
  detail::nccl_coop_dispatch(scope, [&](auto coop) {
    detail::coop_sync(coop);
    if (coop.thread_rank() == 0) {
      const ncclDevComm& dev_comm = comm_;
      uint64_t* base = detail::signal_slot_base(
          dev_comm, signal_buffer_handle_, signal_id, num_ranks_);

      for (int i = 0; i < num_ranks_; i++) {
        comms::device::st_release_sys_global(base + i, 0ULL);
      }
    }
  });
}

// =============================================================================
// TorchCommDeviceWindow<NCCLDeviceBackend> Counter Operations
// =============================================================================
// Counters remain on GIN hardware — they track local DMA completion
// (NIC increments after source buffer read). Only meaningful for RDMA path.

template <>
__device__ inline int TorchCommDeviceWindow<NCCLDeviceBackend>::wait_counter(
    int counter_id,
    CmpOp cmp,
    uint64_t value,
    CoopScope scope) {
  if (cmp != CmpOp::GE) {
    // GIN hardware counters only support GE comparison.
    __trap();
    return -1; // Unreachable
  }
  if (!gin_enabled_) {
    return -1;
  }

  const ncclDevComm& dev_comm = comm_;
  ncclGin gin(dev_comm, kDefaultGinContextIndex);

  detail::nccl_coop_dispatch(scope, [&](auto coop) {
    gin.waitCounter(
        coop,
        static_cast<ncclGinCounter_t>(counter_id),
        value,
        kDefaultCounterBits);
  });

  return 0;
}

template <>
__device__ inline uint64_t
TorchCommDeviceWindow<NCCLDeviceBackend>::read_counter(int counter_id) const {
  if (!gin_enabled_) {
    return 0;
  }

  const ncclDevComm& dev_comm = comm_;
  ncclGin gin(dev_comm, kDefaultGinContextIndex);

  return gin.readCounter(
      static_cast<ncclGinCounter_t>(counter_id), kDefaultCounterBits);
}

template <>
__device__ inline void TorchCommDeviceWindow<NCCLDeviceBackend>::reset_counter(
    int counter_id,
    CoopScope scope) {
  detail::nccl_coop_dispatch(scope, [&](auto coop) {
    detail::coop_sync(coop);
    if (gin_enabled_ && coop.thread_rank() == 0) {
      const ncclDevComm& dev_comm = comm_;
      ncclGin gin(dev_comm, kDefaultGinContextIndex);
      gin.resetCounter(static_cast<ncclGinCounter_t>(counter_id));
    }
  });
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
  detail::nccl_coop_dispatch(
      scope, [&](auto coop) { detail::coop_sync(coop); });

  if (!gin_enabled_) {
    return 0;
  }

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

  if (!gin_enabled_) {
    detail::nccl_coop_dispatch(scope, [&](auto coop) {
      ncclLsaBarrierSession barrier(
          coop,
          dev_comm,
          ncclTeamTagLsa{},
          static_cast<uint32_t>(barrier_id),
          false /* multimem */);
      barrier.sync(coop, cuda::memory_order_acq_rel);
    });
    return 0;
  }

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

// =============================================================================
// TorchCommDeviceWindow<NCCLDeviceBackend> NVLink Address Query
// =============================================================================

template <>
__device__ inline void*
TorchCommDeviceWindow<NCCLDeviceBackend>::get_nvlink_address(int peer) {
  return ncclGetPeerPointer(window_, 0, peer);
}

// =============================================================================
// TorchCommDeviceWindow<NCCLDeviceBackend> Multimem Address Query
// =============================================================================

template <>
__device__ inline void*
TorchCommDeviceWindow<NCCLDeviceBackend>::get_multimem_address(size_t offset) {
  if (comm_.lsaMultimem.mcBasePtr == nullptr) {
    return nullptr;
  }
  return ncclGetLsaMultimemPointer(window_, offset, comm_);
}

} // namespace torchcomms::device
