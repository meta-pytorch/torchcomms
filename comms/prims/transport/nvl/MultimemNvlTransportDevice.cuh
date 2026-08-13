// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "comms/common/AtomicUtils.cuh"
#include "comms/prims/core/SignalState.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/Timeout.cuh"
#include "comms/prims/memory/DeviceSpan.cuh"

namespace comms::prims {

// Per-lane internal-signal count for the staging pipeline: the single source of
// truth shared by the host-side signal-region sizing (MCCL) and the
// device-side `make_stage_layout` (MultimemNvlStageLayout.cuh) so the region is
// sized identically on both sides. Layout per lane: `nvlRanks` per-peer ready[]
// + `nvlRanks` per-peer ack[] + 4 staging arrival-barrier slots (ready/ack
// counter+epoch, laid out past the SET-mode slots so ADD residue never
// contaminates a later SET-mode CMP_GE wait) => 2 * nvlRanks + 4.
__host__ __device__ constexpr uint32_t multimem_staging_signals_per_lane(
    uint32_t nvlRanks) {
  return 2 * nvlRanks + 4;
}

namespace detail {

__device__ __forceinline__ void multimem_store_release_sys_u64(
    uint64_t* dst,
    uint64_t v) {
  (void)dst;
  (void)v;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("multimem.st.release.sys.global.u64 [%0], %1;"
               :
               : "l"(dst), "l"(v)
               : "memory");
#elif defined(__CUDA_ARCH__)
  // A plain store on a multimem VA would only update this rank's backing and
  // silently diverge from multicast semantics. Host-side isEligible() /
  // isMultimemSupported() gates prevent this path from ever being taken with
  // a multimem pointer, but trap so any accidental pre-SM90 use is loud.
  __trap();
#endif
}

__device__ __forceinline__ void multimem_red_release_sys_add_u64(
    uint64_t* dst,
    uint64_t v) {
  (void)dst;
  (void)v;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("multimem.red.release.sys.global.add.u64 [%0], %1;"
               :
               : "l"(dst), "l"(v)
               : "memory");
#elif defined(__CUDA_ARCH__)
  __trap();
#endif
}

} // namespace detail

/**
 * Device-side handle for one multicast staging window.
 *
 * `multimemData` and multimem signal spans are multicast VAs. Writes into the
 * multicast pointer preserve multicast semantics; `localData` and local signal
 * spans are this rank's backing memory after multicast replication. Callers
 * that want to reduce from the multicast VA should obtain
 * `multimem_data_ptr()` and pass it to `multimem::load_reduce_at()`, defined
 * in `MultimemNvlReduce.cuh`. Broadcast-store helpers are introduced by the
 * later store-primitives diff.
 */
struct MultimemNvlTransportDevice {
  char* localData{nullptr};
  char* multimemData{nullptr};
  DeviceSpan<SignalState> userLocalSignals{};
  DeviceSpan<SignalState> userMultimemSignals{};
  DeviceSpan<SignalState> internalLocalSignals{};
  DeviceSpan<SignalState> internalMultimemSignals{};
  std::size_t dataBufferSize{0};
  int nvlRank{0};
  int nvlRanks{1};

  __device__ __forceinline__ char* local_data_ptr(std::size_t offset) const {
    return localData + offset;
  }

  __device__ __forceinline__ char* multimem_data_ptr(std::size_t offset) const {
    return multimemData + offset;
  }

  __device__ __forceinline__ void signal(
      ThreadGroup& group,
      uint64_t signal_id,
      SignalOp op,
      uint64_t value) const {
    signal_at(group, user_multimem_signal_ptr(signal_id), op, value);
  }

  __device__ __forceinline__ uint64_t read_signal(uint64_t signal_id) const {
    return user_local_signal_ptr(signal_id)->load();
  }

  __device__ __forceinline__ bool wait_signal_until(
      ThreadGroup& group,
      uint64_t signal_id,
      CmpOp op,
      uint64_t expected,
      const AbortDevice& timeout = AbortDevice()) const {
    return user_local_signal_ptr(signal_id)->wait_until(
        group, op, expected, timeout);
  }

  __device__ __forceinline__ void signal_internal(
      ThreadGroup& group,
      uint64_t signal_id,
      SignalOp op,
      uint64_t value) const {
    signal_at(group, internal_multimem_signal_ptr(signal_id), op, value);
  }

  __device__ __forceinline__ uint64_t
  read_internal_signal(uint64_t signal_id) const {
    return internal_local_signal_ptr(signal_id)->load();
  }

  __device__ __forceinline__ bool wait_internal_signal_until(
      ThreadGroup& group,
      uint64_t signal_id,
      CmpOp op,
      uint64_t expected,
      const AbortDevice& timeout = AbortDevice()) const {
    return internal_local_signal_ptr(signal_id)->wait_until(
        group, op, expected, timeout);
  }

 private:
  __device__ __forceinline__ SignalState* signal_ptr(
      DeviceSpan<SignalState> signals,
      uint64_t signal_id,
      [[maybe_unused]] const char* kind) const {
#if defined(__CUDA_ARCH__)
    if (signal_id >= signals.size()) {
      printf(
          "MultimemNvlTransportDevice: %s signal_id=%llu out of range "
          "(count=%u)\n",
          kind,
          static_cast<unsigned long long>(signal_id),
          static_cast<unsigned>(signals.size()));
      __trap();
    }
#endif
    return signals.data() + signal_id;
  }

  __device__ __forceinline__ SignalState* user_local_signal_ptr(
      uint64_t signal_id) const {
    return signal_ptr(userLocalSignals, signal_id, "user local");
  }

  __device__ __forceinline__ SignalState* user_multimem_signal_ptr(
      uint64_t signal_id) const {
    return signal_ptr(userMultimemSignals, signal_id, "user multimem");
  }

  __device__ __forceinline__ SignalState* internal_local_signal_ptr(
      uint64_t signal_id) const {
    return signal_ptr(internalLocalSignals, signal_id, "internal local");
  }

  __device__ __forceinline__ SignalState* internal_multimem_signal_ptr(
      uint64_t signal_id) const {
    return signal_ptr(internalMultimemSignals, signal_id, "internal multimem");
  }

  __device__ __forceinline__ void signal_at(
      ThreadGroup& group,
      SignalState* signal,
      SignalOp op,
      uint64_t value) const {
    comms::device::fence_acq_rel_sys();
    group.sync();
    if (group.is_leader()) {
      switch (op) {
        case SignalOp::SIGNAL_SET:
          detail::multimem_store_release_sys_u64(&signal->signal_, value);
          break;
        case SignalOp::SIGNAL_ADD:
          detail::multimem_red_release_sys_add_u64(&signal->signal_, value);
          break;
      }
    }
  }
};

} // namespace comms::prims
