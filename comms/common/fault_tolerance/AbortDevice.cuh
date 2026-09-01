// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h>
#ifndef __HIP_PLATFORM_AMD__
#include <cuda/atomic>
#endif

#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "comms/common/fault_tolerance/Abort.h"

namespace comms::fault_tolerance {

/**
 * Shared-abort-state polls per millisecond of device time.
 *
 * Each poll is a read of mapped pinned host memory, which is an uncached PCIe
 * round trip measured at ~1.1us on H100 (`CudaAtomicDeviceLoadLoop` in
 * `benchmarks/Perf.md`). So this constant is not a free knob: it sets a
 * *fraction of kernel runtime* spent polling, and that fraction is
 * `kAbortPollsPerMs * 1.1us / 1000us`.
 *
 * At the previous value of 100 that is **11% of every collective**, and it was
 * measured as exactly that. A 4-rank IB_ONLY AllReduce on GB300 runs ~170us, so
 * it paid ~17 polls; neutralizing the poll while leaving every barrier and
 * branch in place took the fault-tolerance overhead from +11.4us to +1.1us
 * (tree) and +10.8us to +3.0us (ring). Polling was the overhead.
 *
 * 1 poll/ms bounds abort-observation latency at ~1ms instead of ~10us, for
 * ~0.1% of runtime. That is still four orders of magnitude finer than the
 * deadlines this exists to serve -- `MCCL_ABORT_TIMEOUT_MS` defaults to 30
 * seconds -- and the thing being delayed is only how fast an *already failed*
 * collective unwinds. Nobody can measure a millisecond added to that.
 *
 * Deadline expiry is deliberately not affected: `checkExpired()` tests
 * `deadlineDue` ahead of the throttle, so a timeout still fires on time no
 * matter what this is set to. This governs only how quickly one rank notices
 * *another* rank's abort.
 *
 * If you raise this, re-run `abort_bench` and the GB300 sweep and update both
 * this comment and `FAULT_TOLERANCE.md`. The cost is linear in the value.
 */
inline constexpr uint64_t kAbortPollsPerMs = 1;

namespace detail {

inline int hostCurrentDevice() {
  int device = 0;
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    throw std::runtime_error(
        "cudaGetDevice failed for Abort::getDeviceHandle(): " +
        std::string(cudaGetErrorString(status)));
  }
  return device;
}

__device__ __forceinline__ uint64_t deviceClock() {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__)
  return wall_clock64();
#elif defined(__CUDA_ARCH__)
  return clock64();
#else
  return 0;
#endif
}

inline uint64_t hostDeviceCyclesPerMs(int device) {
#ifdef __HIP_PLATFORM_AMD__
  (void)device;
  // TODO: Revisit this when AMD device-side abort timeout support is added.
  return 100000;
#else
  constexpr int kMaxCachedDevices = 64;
  thread_local std::array<uint64_t, kMaxCachedDevices> cachedCyclesPerMs{};
  const bool cacheableDevice = device >= 0 && device < kMaxCachedDevices;
  if (cacheableDevice && cachedCyclesPerMs[device] != 0) {
    return cachedCyclesPerMs[device];
  }

  int clockRateKHz = 0;
  auto status =
      cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrClockRate, device);
  if (status != cudaSuccess) {
    throw std::runtime_error(
        "cudaDeviceGetAttribute(cudaDevAttrClockRate) failed for Abort::getDeviceHandle(): " +
        std::string(cudaGetErrorString(status)));
  }
  const auto cyclesPerMs = static_cast<uint64_t>(clockRateKHz);
  if (cacheableDevice) {
    cachedCyclesPerMs[device] = cyclesPerMs;
  }
  return cyclesPerMs;
#endif
}

__device__ __forceinline__ bool deviceIsValidTerminalReason(
    AbortReason reason) {
  switch (reason) {
    case AbortReason::ABORTED:
    case AbortReason::TIMED_OUT:
    case AbortReason::BOOTSTRAP_POLL:
    case AbortReason::NETWORK_ERROR:
    case AbortReason::INTERNAL_ERROR:
    case AbortReason::IBRC_PROXY_TIMEOUT:
      return true;
    case AbortReason::NONE:
      return false;
  }
  return false;
}

template <typename T>
__device__ __forceinline__ T deviceLoadAcquireSystem(T* value) {
#ifdef __HIP_PLATFORM_AMD__
  return __atomic_load_n(value, __ATOMIC_ACQUIRE);
#else
  return cuda::atomic_ref<T, cuda::thread_scope_system>{*value}.load(
      cuda::memory_order_acquire);
#endif
}

__device__ __forceinline__ bool
deviceCompareExchangeSystem(int* value, int* expected, int desired) {
#ifdef __HIP_PLATFORM_AMD__
  return __atomic_compare_exchange_n(
      value, expected, desired, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
#else
  return cuda::atomic_ref<int, cuda::thread_scope_system>{*value}
      .compare_exchange_strong(
          *expected,
          desired,
          cuda::memory_order_acq_rel,
          cuda::memory_order_acquire);
#endif
}

/**
 * Records a terminal reason from device code, first-writer-wins, and emits the
 * common first-writer marker if this call is the winner.
 *
 * Shared by both device writers of `AbortState::abort` -- `AbortDevice` and
 * `AbortFlag`. They differ in what else they carry (a poll throttle, a
 * per-operation identity), but the transition itself and the line that reports
 * it are the same event, and a writer that transitions without logging leaves
 * an abort with no greppable origin.
 *
 * Takes the raw `AbortState*` rather than either handle type so it can be
 * defined once, before both, without a forward declaration dance.
 */
__device__ __forceinline__ bool deviceTrySetAbort(
    AbortState* state,
    AbortReason newReason,
    const char* context) {
  if (state == nullptr) {
    return false;
  }
  const bool validReason = deviceIsValidTerminalReason(newReason);
  assert(validReason);
  if (!validReason) {
    return false;
  }

  int expected = static_cast<int>(AbortReason::NONE);
  const bool won = deviceCompareExchangeSystem(
      &state->abort, &expected, static_cast<int>(newReason));
  if (won) {
    // NOLINTNEXTLINE(facebook-security-vulnerable-printf)
    printf(
        FT_ABORT_FIRST_WRITER_DEVICE_ "reason=%d context=%s\n",
        static_cast<int>(newReason),
        context == nullptr ? "" : context);
  }
  return won;
}

} // namespace detail

/**
 * Non-owning CUDA device view of an `Abort` object's shared state.
 *
 * The owning `Abort` object allocates and frees the mapped pinned `AbortState`.
 * Kernels may copy this handle by value, but the handle must not be used after
 * the owning `Abort` is destroyed. All reads and writes of shared state use
 * system-scope atomics so CPU and device updates are mutually visible. Create
 * handles through `Abort::getDeviceHandle()` so the mapped device pointer and
 * timeout clock-rate conversion are initialized together.
 */
struct AbortDevice final {
  /**
   * Creates a disabled no-op device handle.
   *
   * This makes `AbortDevice` usable in aggregate kernel arguments and as a
   * temporary compatibility replacement for disabled Prims timeouts.
   */
  __host__ __device__ AbortDevice() = default;

  /**
   * Returns whether this handle references enabled shared abort state.
   *
   * Disabled handles are valid kernel arguments. They have a null state pointer
   * and all mutating/query APIs behave like the host disabled `Abort` object.
   */
  __host__ __device__ bool isEnabled() const {
    return state_ != nullptr;
  }

  /**
   * Returns the abort behavior captured when this device handle was created.
   */
  __host__ __device__ AbortBehavior behavior() const {
    return behavior_;
  }

  /**
   * Raw shared-state pointer, for constructing an `AbortFlag` view of this
   * handle. Not part of the general API -- everything else goes through the
   * typed accessors.
   */
  __host__ __device__ AbortState* stateForFlag() const {
    return state_;
  }

  /**
   * Overrides the deadline for one operation, in milliseconds.
   *
   * Set this on a per-operation COPY of the handle. Handles obtained from
   * `MultiPeerTransport::get_device_handle()` are communicator-scoped and
   * shared, so mutating one in place would leak the override into unrelated
   * operations:
   *
   *     auto abort = mpt->get_device_handle(peers).abort;  // copy
   *     abort.setOpTimeoutMs(opTimeoutMs);
   *     params.abort = abort;                              // into kern args
   *
   * A negative value (the default) means "no override": the deadline then
   * comes from the communicator-level timeout in shared state, which stays
   * late-bound so `Abort::setDefaultTimeout()` is always observed.
   *
   * Only set this from a timeout the caller explicitly supplied on the
   * collective API. Seeding it from the communicator default would snapshot
   * that value at launch and defeat late-binding.
   */
  __host__ __device__ void setOpTimeoutMs(int64_t timeoutMs) {
    opTimeoutMs_ = timeoutMs;
  }

  /**
   * Returns the per-operation override, or a negative value when unset.
   */
  __host__ __device__ int64_t opTimeoutMs() const {
    return opTimeoutMs_;
  }

  /**
   * Starts this handle's device-side timeout from the shared default duration.
   *
   * Non-positive or unset default timeouts leave the device deadline inactive.
   * If converting the timeout to device cycles would overflow, the deadline is
   * set to the current device clock so the next timeout check fails fast rather
   * than silently creating an effectively infinite deadline.
   */
  __device__ void startTimeout() {
    if (!isEnabled()) {
      deadlineCycles_ = 0;
      return;
    }
    const auto now = detail::deviceClock();
    // Seeded here rather than left at zero: `nextPollCycles_{0}` makes the
    // first `checkExpired()` on every armed handle unthrottled, which is one
    // uncached mapped-pinned read on the one call guaranteed to happen.
    nextPollCycles_ = now + pollIntervalCycles_;
    const auto timeoutMs = resolveTimeoutMs();
    if (timeoutMs <= 0 || cyclesPerMs_ == 0) {
      deadlineCycles_ = 0;
      return;
    }
    const auto timeoutMsU = static_cast<uint64_t>(timeoutMs);
    if (timeoutMsU > (UINT64_MAX - now) / cyclesPerMs_) {
      deadlineCycles_ = now;
      return;
    }
    deadlineCycles_ = now + timeoutMsU * cyclesPerMs_;
  }

  /**
   * Legacy spelling of `startTimeout()`, kept for the Prims waits that were
   * written against the retired standalone timeout type.
   *
   * New device code should prefer `startTimeout()`.
   */
  __device__ void start() {
    startTimeout();
  }

  /**
   * Cancels this handle's active device-side timeout.
   *
   * The deadline is local to this copied handle. Canceling it does not clear a
   * shared abort reason that has already been recorded.
   */
  __device__ void cancelTimeout() {
    deadlineCycles_ = 0;
  }

  /**
   * Returns true when an explicit abort or this handle's expired timeout has
   * recorded a terminal abort reason.
   *
   * If the local deadline has expired, this attempts to record
   * `AbortReason::TIMED_OUT` as the shared reason. The shared reason remains
   * first-writer-wins.
   */
  __device__ bool isAborted() const {
    return checkExpired();
  }

  /**
   * Checks abort state and returns the action the caller should take.
   *
   * `CONTINUE` means no abort is visible. `SKIP` means the caller should
   * unwind or return without consuming incomplete transport data. `TRAP` means
   * the caller should preserve legacy Prims trap behavior; the trap itself is
   * performed by Prims helpers so common fault-tolerance code stays transport
   * agnostic.
   */
  __device__ AbortCheckResult check(bool* flippedHere = nullptr) const {
    if (!checkExpired(flippedHere)) {
      return AbortCheckResult::CONTINUE;
    }
    return behavior_ == AbortBehavior::TRAP ? AbortCheckResult::TRAP
                                            : AbortCheckResult::SKIP;
  }

  /**
   * Legacy spelling of the raw expiry predicate, kept for migrated Prims waits.
   *
   * Returns true for either an explicit abort or an expired local device
   * timeout. If this handle's local deadline has expired, this records
   * `AbortReason::TIMED_OUT` in the shared state.
   */
  __device__ bool checkExpired(bool* flippedHere = nullptr) const {
    if (flippedHere != nullptr) {
      *flippedHere = false;
    }
    if (!isEnabled()) {
      return false;
    }
    // Terminal reasons are first-writer-wins and never cleared, so once this
    // handle has observed one it can answer from a register forever.
    if (sawTerminalReason_) {
      return true;
    }

    // `state_` lives in mapped pinned host memory, so every read here is an
    // uncached PCIe round trip. The retired standalone Prims timeout compared
    // an on-chip clock and touched no memory at all, so a naive port turns each
    // spin-loop iteration into a host access - worst case 32 lanes x 2 loads
    // per warp per iteration in the LL small-message path. Gate the shared
    // read on the free device clock: steady-state polling costs a register
    // compare, and an abort is still observed within one poll interval.
    const uint64_t now = detail::deviceClock();
    const bool deadlineDue = deadlineCycles_ != 0 && now >= deadlineCycles_;
    if (!deadlineDue && now < nextPollCycles_) {
      return false;
    }
    nextPollCycles_ = now + pollIntervalCycles_;

    if (reason() != AbortReason::NONE) {
      sawTerminalReason_ = true;
      return true;
    }
    if (deadlineDue && markTimedOutIfExpired(flippedHere)) {
      sawTerminalReason_ = true;
      return true;
    }
    return false;
  }

  /**
   * Group-scoped form of the legacy raw expiry predicate.
   *
   * Only the group leader polls shared abort state, matching the current
   * timeout check shape used by Prims wait loops.
   */
  template <typename ThreadGroup>
  __device__ bool checkExpired(const ThreadGroup& group) const {
    return group.is_leader() && checkExpired();
  }

  /**
   * Returns the shared abort reason currently visible to device code.
   *
   * Disabled handles always report `AbortReason::NONE`.
   */
  __device__ AbortReason reason() const {
    if (!isEnabled()) {
      return AbortReason::NONE;
    }
    return static_cast<AbortReason>(
        detail::deviceLoadAcquireSystem(&state_->abort));
  }

  /**
   * Returns the shared default timeout duration in milliseconds.
   *
   * A negative value means no default timeout has been configured.
   */
  __device__ int64_t getTimeoutMs() const {
    if (!isEnabled()) {
      return -1;
    }
    return detail::deviceLoadAcquireSystem(&state_->timeoutMs);
  }

  /**
   * Records the first shared abort reason from device code.
   *
   * Every non-`NONE` AbortReason is terminal. `AbortReason::NONE` and unknown
   * enum values are invalid; debug/device assert builds catch them, and
   * release-compatible builds return before touching shared state. The CAS
   * only transitions the shared state from `NONE`, so later writers cannot
   * overwrite the first terminal reason. `context` matches the host API but is
   * never persisted in shared state; device-side diagnostics may consume it at
   * the winning callsite without adding mapped-memory traffic.
   *
   * Returns whether this call performed the `NONE` to terminal transition.
   */
  __device__ bool setAbort(
      AbortReason newReason = AbortReason::ABORTED,
      const char* context = nullptr) const {
    return detail::deviceTrySetAbort(state_, newReason, context);
  }

 private:
  friend class Abort;

  /**
   * Deadline duration for this operation, in milliseconds.
   *
   * A per-operation override wins because it is the more specific request and
   * it travels by value in the kernel arguments, so it costs no shared-state
   * read. Otherwise the communicator-level timeout is read from mapped shared
   * state on every start, which keeps it late-bound: a handle created before
   * `setDefaultTimeout()` still observes the new value.
   *
   * Deliberately NOT cached in the handle. Transports keep one handle for the
   * communicator's lifetime, so caching here would silently ignore every later
   * `Abort::setDefaultTimeout()`.
   *
   * That also makes it load-bearing for **CUDA graphs**: a captured graph
   * replays with no host code in between, so anything the host stamped into the
   * handle at capture time would be frozen for every replay and the deadline
   * would never lapse. `AbortState` lives in mapped memory precisely so
   * graph-mode device code reads the live value;
   * `DroppedRankTimeoutLapsesDuringGraphReplay` is the test that says so.
   */
  __device__ int64_t resolveTimeoutMs() const {
    if (opTimeoutMs_ >= 0) {
      return opTimeoutMs_;
    }
    return getTimeoutMs();
  }

  /**
   * Creates a device handle for mapped pinned state owned by `Abort`.
   *
   * Passing this handle by value to a kernel passes only this small non-owning
   * view. The shared abort state remains in mapped pinned memory; device code
   * does not receive a private copy of the state. Disabled handles use
   * `state == nullptr`, which makes all public device APIs no-op/non-aborted.
   *
   * `cyclesPerMs` is sampled on the host while creating the handle. CUDA device
   * clocks can change dynamically, so long-running timeouts may be approximate.
   * AMD uses `wall_clock64()`, whose frequency is fixed at 100 MHz; NVIDIA uses
   * `clock64()` and `cudaDevAttrClockRate`.
   *
   * The mapped device pointer and clock conversion are for the CUDA device that
   * was current when `Abort::getDeviceHandle()` created this handle. Launch
   * kernels that consume the handle on that same device, or create a new handle
   * after switching devices.
   *
   * TODO: Evaluate CUPTI APIs for a more accurate timeout conversion source.
   */
  explicit AbortDevice(
      AbortState* state,
      uint64_t cyclesPerMs,
      AbortBehavior behavior = AbortBehavior::SKIP)
      : state_{state},
        cyclesPerMs_{cyclesPerMs},
        pollIntervalCycles_{cyclesPerMs / kAbortPollsPerMs},
        behavior_{behavior} {}

  __device__ bool deadlineExpired() const {
    return deadlineCycles_ != 0 && detail::deviceClock() >= deadlineCycles_;
  }

  __device__ bool markTimedOutIfExpired(bool* flippedHere = nullptr) const {
    if (!isEnabled() || !deadlineExpired()) {
      return false;
    }

    int expected = static_cast<int>(AbortReason::NONE);
    if (detail::deviceCompareExchangeSystem(
            &state_->abort,
            &expected,
            static_cast<int>(AbortReason::TIMED_OUT))) {
      if (flippedHere != nullptr) {
        *flippedHere = true;
      }
      return true;
    }
    return expected == static_cast<int>(AbortReason::TIMED_OUT);
  }

  /**
   * Device pointer to the same mapped pinned state owned by the host `Abort`.
   *
   * This pointer is non-owning and is expected to be non-null for handles
   * returned by `Abort::getDeviceHandle()`.
   */
  AbortState* state_{nullptr};

  /**
   * Device clock cycles per millisecond captured when the handle is created.
   *
   * Device timeout setup uses this to convert the shared default timeout from
   * milliseconds into device clock cycles.
   */
  uint64_t cyclesPerMs_{0};

  /**
   * Per-operation deadline override in milliseconds; negative means unset.
   *
   * Copied into kernel arguments with the rest of the handle, so device code
   * reads it from registers rather than mapped host memory.
   */
  int64_t opTimeoutMs_{-1};

  /**
   * Minimum device-clock cycles between reads of the mapped shared state.
   *
   * Bounds abort-observation latency to one interval while keeping spin loops
   * off the PCIe bus. Zero (disabled handles) polls every call, which is free
   * because disabled handles short-circuit before the read.
   */
  uint64_t pollIntervalCycles_{0};

  /**
   * Device clock value after which the shared state may be read again.
   */
  mutable uint64_t nextPollCycles_{0};

  /**
   * Sticky: a terminal reason was already observed through this handle.
   */
  mutable bool sawTerminalReason_{false};

  /**
   * Device abort behavior selected by the owning host `Abort`.
   */
  AbortBehavior behavior_{AbortBehavior::SKIP};

  /**
   * Per-handle device deadline in `detail::deviceClock()` cycles.
   *
   * A value of zero means no device-side timeout is active. This is local to
   * the copied handle and is not part of the host-owned shared state.
   */
  uint64_t deadlineCycles_{0};
};

/**
 * A poll-state-free view of the shared abort state, safe to store in device
 * memory that many blocks read.
 *
 * `AbortDevice` is designed to be copied per thread: it carries mutable poll
 * throttle state (`nextPollCycles_`, `sawTerminalReason_`) whose whole purpose
 * is to be private to one poller. Placing one in device global memory breaks
 * that assumption in two ways at once -- several blocks write the throttle
 * non-atomically, and the absolute `clock64()` it stores is per-SM, so a value
 * stamped by a leading SM can suppress a lagging SM's polls for far longer than
 * the interval and hide the abort it is waiting for.
 *
 * This type makes that unrepresentable rather than merely discouraged: it holds
 * no mutable state and exposes no way to poll. A transport that needs a
 * communicator-scoped handle in device memory stores one of these, and bounds
 * its waits on the device clock instead of on shared reads.
 */
struct AbortFlag final {
  __host__ __device__ AbortFlag() = default;

  __host__ __device__ explicit AbortFlag(const AbortDevice& handle)
      : state_(handle.stateForFlag()), behavior_(handle.behavior()) {}

  __host__ __device__ bool isEnabled() const {
    return state_ != nullptr;
  }

  __host__ __device__ AbortBehavior behavior() const {
    return behavior_;
  }

  /**
   * Records a terminal reason, first-writer-wins. Returns whether this call
   * performed the transition.
   *
   * Writing shared state is safe to do from a shared handle -- it is a
   * system-scope CAS, which is exactly what the shared state is for. Only
   * *polling* needs per-thread throttle state, and this type has none.
   *
   * The winner emits the same first-writer marker `AbortDevice` does, through
   * the same helper. That costs this type nothing it was built to avoid: the
   * `printf` is gated on the CAS win, so it happens at most once per
   * communicator, and it adds no mutable state and no poll. Without it the IBRC
   * proxy watchdogs -- which reach the shared reason only through this type --
   * can leave a communicator aborted with no greppable origin at all, and the
   * `context` they pass describing which watchdog fired is discarded.
   */
  __device__ bool setAbort(
      AbortReason newReason = AbortReason::ABORTED,
      const char* context = nullptr) const {
    return detail::deviceTrySetAbort(state_, newReason, context);
  }

  /**
   * Whether a terminal reason has been recorded. One system-scope load from
   * mapped host state, every call -- there is deliberately no throttle here,
   * because throttle state is exactly what must not live in a shared handle.
   *
   * Use this for one-shot decisions ("should this operation start at all?"),
   * never inside a spin loop. A loop should bound itself on the device clock
   * against a fixed budget instead.
   */
  __device__ bool isAborted() const {
    if (!isEnabled()) {
      return false;
    }
    return static_cast<AbortReason>(detail::deviceLoadAcquireSystem(
               &state_->abort)) != AbortReason::NONE;
  }

 private:
  AbortState* state_{nullptr};
  AbortBehavior behavior_{AbortBehavior::SKIP};
};

/**
 * Debug guard for collective onboarding.
 *
 * Logs once when `comm` has fault tolerance enabled but `handle` is disabled —
 * the signature of a collective that never wired the communicator abort into
 * its launch parameters, and so silently has no fault tolerance.
 *
 * Host-side by necessity: on the device a disabled handle is just a null state
 * pointer, which is indistinguishable between "FT is off for this
 * communicator" and "the collective forgot to pass the handle". Only the host
 * can see both sides. Compiled out in optimized builds.
 */
inline void debugCheckAbortWired(
    const Abort* comm,
    const AbortDevice& handle,
    const char* opName) {
#ifndef NDEBUG
  if (comm == nullptr || !comm->isEnabled() || handle.isEnabled()) {
    return;
  }
  // Shared, not `thread_local`: this reports a static wiring mistake in a
  // collective, so the second thread to launch it has nothing new to say. A
  // per-thread flag would repeat the same diagnostic once per thread in a pool.
  //
  // Marked library-local because this is an inline function: under
  // `-fvisibility-inlines-hidden` each shared object gets its own copy, so the
  // dedup is per-DSO rather than per-process. That is fine here -- the worst
  // case is one extra line in a debug build, and it still collapses the
  // per-thread repetition this exists to prevent.
  /* library-local */ static std::atomic<bool> warned{false};
  if (!warned.exchange(true, std::memory_order_relaxed)) {
    fprintf(
        stderr,
        "comms fault tolerance: %s launched with a disabled AbortDevice while "
        "the communicator has fault tolerance enabled. This collective has no "
        "device deadline and cannot be aborted. See the Collective Enablement "
        "notes in comms/common/fault_tolerance/FAULT_TOLERANCE.md\n",
        opName);
  }
#else
  (void)comm;
  (void)handle;
  (void)opName;
#endif
}

inline AbortDevice Abort::getDeviceHandle() const {
  if (state_ == nullptr) {
    return AbortDevice{/*state=*/nullptr, /*cyclesPerMs=*/0, behavior_};
  }
  if (!stateMapped_) {
    throw std::runtime_error(
        "Abort::getDeviceHandle() requires CUDA host-mapped Abort state");
  }
  const auto device = detail::hostCurrentDevice();
  void* deviceState = nullptr;
  auto status = cudaHostGetDevicePointer(&deviceState, state_, 0);
  if (status != cudaSuccess) {
    throw std::runtime_error(
        "cudaHostGetDevicePointer failed for Abort::getDeviceHandle(): " +
        std::string(cudaGetErrorString(status)));
  }
  return AbortDevice{
      static_cast<AbortState*>(deviceState),
      detail::hostDeviceCyclesPerMs(device),
      behavior_};
}

} // namespace comms::fault_tolerance
