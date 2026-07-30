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
// truth shared by the host-side signal-region sizing (CtranPipes) and the
// device-side `make_stage_layout` (MultimemNvlStageLayout.cuh) so the region is
// sized identically on both sides. Layout per lane: `nvlRanks` per-peer ready[]
// + `nvlRanks` per-peer ack[] + 4 staging arrival-barrier slots (ready/ack
// counter+epoch, laid out past the SET-mode slots so ADD residue never
// contaminates a later SET-mode CMP_GE wait) => 2 * nvlRanks + 4.
__host__ __device__ constexpr uint32_t multimem_staging_signals_per_lane(
    int nvlRanks) {
  return static_cast<uint32_t>(2 * nvlRanks + 4);
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

/** Reduction operator for the multimem data reduce verbs. Only Add today. */
enum class MultimemRedOp { Add };

/**
 * Device-side handle for one multicast staging window.
 *
 * `multimemData` and multimem signal spans are multicast VAs. Writes into the
 * multicast pointer preserve multicast semantics; `localData` and local signal
 * spans are this rank's backing memory after multicast replication. Callers
 * that want to broadcast into or reduce from the multicast VA should obtain
 * `multimem_data_ptr()` and pass it to the `multimem::store()` (multimem.st) /
 * `multimem::load_reduce_at()` (multimem.ld_reduce) free functions, defined in
 * `MultimemNvlStore.cuh` / `MultimemNvlReduce.cuh`.
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
  // When true, the STAGING full barriers (input-ready in
  // stage_and_wait_all_inputs, ack in reduce_round_to_all_ranks) use the O(1)
  // arrival-counter barrier instead of the per-peer SIGNAL_SET loop. Plumbed
  // from NCCL_CTRAN_ALLREDUCE_CNVLMM_STAGING_ARRIVAL_BARRIER. The arrival
  // counter/epoch slots are repurposed from the per-(group,lane) staging signal
  // region (disjoint from the rsag ack[] slots), so this needs no host sizing
  // change. The device cannot read cvars, hence this bool field.
  bool stagingArrivalBarrier{false};

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

  __device__ __forceinline__ void wait_signal_until(
      ThreadGroup& group,
      uint64_t signal_id,
      CmpOp op,
      uint64_t expected,
      const Timeout& timeout = Timeout()) const {
    user_local_signal_ptr(signal_id)->wait_until(group, op, expected, timeout);
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

  __device__ __forceinline__ void wait_internal_signal_until(
      ThreadGroup& group,
      uint64_t signal_id,
      CmpOp op,
      uint64_t expected,
      const Timeout& timeout = Timeout()) const {
    internal_local_signal_ptr(signal_id)->wait_until(
        group, op, expected, timeout);
  }

  /**
   * O(1) arrival-counter barrier over the NVL team (NVLS lsa_barrier-style),
   * built from the existing SIGNAL_ADD / wait primitives.
   *
   * `counter_id` is a multicast-backed internal slot used ONLY by this barrier
   * point (one per CTA group), and `epoch_id` is a separate internal slot used
   * as this rank's PRIVATE (local-view-only) baseline. Each barrier: the group
   * leader multimem.red.add's 1 into the counter (fanning +1 into every rank's
   * backing), so after all `nvlRanks` arrive every rank's local copy of the
   * counter has advanced by exactly `nvlRanks`. The group then waits its single
   * local counter slot to reach `epoch + nvlRanks` and advances the epoch. The
   * u64 counter is monotonic and never reset, so no wraparound handling is
   * needed; the wait `Timeout` turns a mismatch into a clean abort, not a hang.
   *
   * Replaces the legacy O(nvlRanks) per-peer SIGNAL_SET barrier (every rank
   * polls all N peer slots). Caller must use disjoint (counter_id, epoch_id)
   * slots per barrier point and per CTA group; both must be zero-initialized
   * (the internal signal region is) and persist across launches (it does).
   *
   * PRECONDITION: for a given communicator+signal-region, the (group_id, lane,
   * barrier-point) -> physical slot mapping must be stable across every op that
   * uses this barrier. The per-group base depends only on group_id, so varying
   * numBlocks/total_groups across ops is safe; `nvlRanks` is fixed per comm.
   * The one variable that would break the counter/epoch pairing is
   * `pipelineDepth` (it scales the per-lane stride): all ops on a comm that use
   * this barrier MUST use the same pipelineDepth, else a slot inherits residue
   * from an op with a different geometry and `target` becomes unreachable
   * (Timeout) or already-satisfied (premature release). Unlike the SET path
   * (absolute monotonic roundId), the ADD counter has no self-correcting
   * absolute value.
   */
  __device__ __forceinline__ void arrival_barrier(
      ThreadGroup& group,
      uint64_t counter_id,
      uint64_t epoch_id,
      const Timeout& timeout = Timeout()) const {
    // The arrival target is this transport's multicast team size (`nvlRanks`);
    // taken from the member so a caller cannot pass a mismatched count (which
    // would skew the target: too large -> hang until Timeout, too small ->
    // premature release).
    const uint64_t target = internal_local_signal_ptr(epoch_id)->load() +
        static_cast<uint64_t>(nvlRanks);
    // Arrive: leader adds 1 to the shared multicast counter (fences +
    // group.sync internally, then the multimem.red.add fans the increment to
    // all backings).
    signal_internal(group, counter_id, SignalOp::SIGNAL_ADD, 1);
    // Wait: O(1) - spin this rank's single local counter slot until all
    // arrived.
    wait_internal_signal_until(
        group, counter_id, CmpOp::CMP_GE, target, timeout);
    // Advance this rank's private baseline for the next use of this slot.
    if (group.is_leader()) {
      internal_local_signal_ptr(epoch_id)->store(target);
    }
    group.sync();
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
