// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

// Host-only includes. The guard uses __HIPCC__ || !__CUDACC__ so that:
//  - Regular .cc files: visible (neither macro defined)
//  - NVCC .cu files: hidden (folly headers don't compile under nvcc)
//  - HIP .cc/-x hip files: visible (clang handles folly fine)
// After hipify (__CUDACC__ → __HIPCC__), this becomes __HIPCC__ || !__HIPCC__
// which is always true — correct since HIP/clang has no folly issues.
#if defined(__HIPCC__) || !defined(__CUDACC__)
#include <atomic>
#include <chrono>
#include <memory>

#include <folly/Synchronized.h>
#endif

// Globaltimer (ns) is reduced to ~1024ns ticks (`>> shift`) to fit in 32
// bits when packed into HRDWEntry. 32 bits of 1024ns ticks wraps every
// ~73 minutes; the host reader reconstructs the full value via
// GlobaltimerCalibration::toWallClock(uint32_t).
#define US_TICK_TIMESTAMP_SHIFT 10

namespace hrdw_ring_buffer {

// Calibration anchor mapping GPU %globaltimer nanoseconds to wall-clock time.
// The first anchor is taken lazily on first use of get(); subsequent anchors
// are taken by callers via refresh() to bound drift between GPU oscillator
// and PTP-aligned wall time.
//
// Thread-safety: toWallClock() is safe under any concurrency, including
// concurrent with refresh(). refresh() is also safe to call from multiple
// threads — concurrent callers race on the refresh_in_progress_ flag and
// the loser short-circuits (returns false without touching the CUDA stream
// or pinned buffer), so the dedicated stream and mapped buffer have
// effectively single-caller semantics at the CUDA layer.
//
// WARNING: This class owns a CUDA stream and a pinned host buffer that are
// intentionally NOT destroyed at process exit. Calling cudaStreamDestroy /
// cudaFreeHost during static destruction races with CUDA runtime teardown
// (cudaErrorCudartUnloading) and segfaults. The singleton in get() is
// heap-allocated and leaked; matches CudaReferencePoint and PrecisionClockImpl.
#if defined(__HIPCC__) || !defined(__CUDACC__)
class GlobaltimerCalibration {
 public:
  // Convert a device globaltimer nanosecond reading to a wall-clock
  // time_point using the most recent anchor. Thread-safe vs. concurrent
  // refresh().
  std::chrono::system_clock::time_point toWallClock(uint64_t gpu_ns) const;

  // Convert a packed 32-bit timestamp (globaltimer_ns >> 10, i.e. ~1024ns
  // ticks, as written by HRDWRingBuffer) into a system_clock time_point.
  // Reconstructs the high 32 bits against the current wall-clock-derived
  // "now" device time. Events are assumed to be in the past; a small
  // future window (~10s) is allowed for host/device clock skew, and
  // anything beyond is treated as a past wrap. Correct as long as the
  // event is within ~73 minutes in the past of "now".
  std::chrono::system_clock::time_point toWallClock(
      uint32_t gpu_ticks_low32) const;

  // Re-read %globaltimer + precisionNow() and atomically replace the cached
  // anchor. Cost: 1 single-thread kernel launch + 1 stream sync + 1 fbclock
  // read on a dedicated side stream (no device-wide sync). Caller is
  // responsible for rate-limiting; the colltrace poll thread invokes this
  // at ~1 Hz, which keeps residual oscillator drift between anchors below
  // 100 ns at 100 ppm. Failures are logged and the prior anchor is kept;
  // returns true when the anchor was successfully replaced. Safe under
  // concurrent invocation: at most one caller proceeds to touch the CUDA
  // stream/buffer; others observe `refresh_in_progress_` already set and
  // return false (the in-flight refresh will publish a fresh anchor before
  // the next caller's wakeup, so no information is lost).
  bool refresh();

  // Process-global singleton.
  static GlobaltimerCalibration& get();

  // TEST-ONLY: construct a calibration with a synthetic anchor and no CUDA
  // initialization. Used by GpuClockCalibrationTest to exercise the
  // toWallClock math against known anchor values without a real GPU.
  // Production code must use get() — refresh() is a no-op on a test
  // instance because there is no CUDA stream to drive.
  static std::unique_ptr<GlobaltimerCalibration> createForTest(
      uint64_t device_ns,
      std::chrono::system_clock::time_point host_time);

  GlobaltimerCalibration(const GlobaltimerCalibration&) = delete;
  GlobaltimerCalibration& operator=(const GlobaltimerCalibration&) = delete;
  GlobaltimerCalibration(GlobaltimerCalibration&&) = delete;
  GlobaltimerCalibration& operator=(GlobaltimerCalibration&&) = delete;
  // Trivial: present only to satisfy clang-tidy
  // cppcoreguidelines-special-member-functions. Never runs in practice — the
  // singleton in get() is heap-allocated and intentionally leaked, so the
  // cudaErrorCudartUnloading risk documented above is unchanged.
  ~GlobaltimerCalibration() = default;

 private:
  GlobaltimerCalibration();

  // Tag-dispatched ctor used by createForTest. Skips CUDA init and just
  // populates the anchor with the provided values.
  struct TestOnlyTag {};
  GlobaltimerCalibration(
      TestOnlyTag,
      uint64_t device_ns,
      std::chrono::system_clock::time_point host_time);

  struct Anchor {
    uint64_t device_ns{};
    std::chrono::system_clock::time_point host_time;
  };

  folly::Synchronized<Anchor> anchor_;
  uint64_t* mapped_ptr_ = nullptr;
  cudaStream_t stream_ = nullptr;
  // Serializes refresh() callers without blocking: contenders that find the
  // flag already set short-circuit (return false) so the CUDA stream and
  // pinned buffer are never touched by more than one thread at a time.
  std::atomic_flag refresh_in_progress_ = ATOMIC_FLAG_INIT;
};

// Launch a single-thread kernel that writes globaltimer() to *out.
cudaError_t launchReadGlobaltimer(cudaStream_t stream, uint64_t* out);
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
// Device-side globaltimer read. Returns nanoseconds since device boot.
__device__ __forceinline__ uint64_t readGlobaltimer() {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  return wall_clock64();
#else
  uint64_t timer;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timer));
  return timer;
#endif
}
#endif

} // namespace hrdw_ring_buffer
