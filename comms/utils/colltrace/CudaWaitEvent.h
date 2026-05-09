// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>

#include <cuda_runtime.h> // @manual

#include <folly/Synchronized.h>
#include <folly/synchronization/Baton.h>

#include "comms/utils/CudaRAII.h"
#include "comms/utils/colltrace/CollWaitEvent.h"
#include "comms/utils/colltrace/CudaEventPool.h"

namespace meta::comms::colltrace {

// WARNING: The cudaStream and the latest cudaEvent used by this class are
// intentionally NOT destroyed at process exit; this class is only intended
// to be used as a global static. Creating too many of these or destroying
// them during shutdown can segfault under cudaErrorCudartUnloading.
//
// During normal operation, refresh() rotates the cached cudaEvent to bound
// drift between the GPU's elapsed-time clock and PTP-aligned wall time, and
// destroys the previous event under the writer lock once it is unreachable
// to readers — so we don't accumulate one event per refresh.
class CudaReferencePoint {
  using system_clock_time_point = ICollWaitEvent::system_clock_time_point;

 public:
  CudaReferencePoint();

  CommsMaybe<system_clock_time_point> getTimeViaEvent(const CudaEvent& event);
  CommsMaybe<system_clock_time_point> getTimeViaEvent(cudaEvent_t event);

  // Re-record the reference cudaEvent on the dedicated stream and pair it
  // with a fresh precisionNow() reading. Safe under concurrent invocation
  // from multiple threads (e.g., one CollTrace poll thread per NCCL
  // communicator): contenders that find refresh_in_progress_ already set
  // return immediately without touching the dedicated stream, so the CUDA
  // operations are effectively serialized at the API layer. Safe vs.
  // concurrent getTimeViaEvent() readers via folly::Synchronized<Anchor>.
  // Cost when the caller wins the flag: 1 cudaEventCreate + 1
  // cudaEventRecord + 1 cudaStreamSynchronize on a private idle stream + 1
  // fbclock read. CUDA failures are logged (rate-limited) and the prior
  // anchor is kept. Caller is responsible for rate-limiting; the colltrace
  // poll thread invokes this at ~1 Hz, bounding residual drift between
  // anchors below 100 ns at 100 ppm oscillator tolerance.
  void refresh();

  // Returns the singleton if it has been constructed, otherwise nullptr.
  // Safe to call from any thread; never triggers construction. Used by the
  // colltrace poll thread to avoid forcing CUDA initialization in CPU-only
  // colltrace deployments where no CudaWaitEvent is ever created.
  static CudaReferencePoint* tryGet();

 private:
  struct Anchor {
    cudaEvent_t event{nullptr};
    system_clock_time_point time;
  };

  // We intentionally do NOT use the RAII version for stream_ to avoid
  // segfault when calling cudaStreamDestroy during the program exit. The
  // anchor's cudaEvent is rotated by refresh(); the latest one leaks at
  // exit by design (see class comment).
  cudaStream_t stream_{nullptr};
  folly::Synchronized<Anchor> anchor_;
  // Serializes refresh() callers without blocking: contenders that find
  // the flag already set short-circuit so the dedicated CUDA stream is
  // never touched by more than one thread at a time. Multiple CollTrace
  // poll threads (one per NCCL communicator) all call refresh on this
  // singleton — without this, concurrent cudaEventRecord on the same
  // stream would race.
  std::atomic_flag refresh_in_progress_ = ATOMIC_FLAG_INIT;
};

class CudaWaitPoint {
 public:
  using system_clock_time_point = ICollWaitEvent::system_clock_time_point;

  enum class WaitPointType {
    start,
    end,
  };

  CudaWaitPoint(cudaStream_t stream, WaitPointType type);

  CommsMaybeVoid recordEvent() noexcept;

  CommsMaybe<bool> waitEventFinish(
      std::chrono::milliseconds sleepTimeMs) noexcept;

  CommsMaybe<system_clock_time_point> getEventFinishTime() noexcept;

  static CudaReferencePoint& getReferencePoint();

 private:
  // We do not own the stream, so we use raw pointer here.
  cudaStream_t recordStream_;
  WaitPointType type_;
  CachedCudaEvent event_;

  enum class CudaEventStatus {
    unrecorded = 0,
    recorded = 1,
    finished = 2,
  } eventStatus_{CudaEventStatus::unrecorded};

  std::string_view getWaitPointTypeString() const noexcept;

  static CommsMaybe<system_clock_time_point> getTimeViaReference(
      const CudaEvent& event);
};

class CudaWaitEvent : public ICollWaitEvent {
 public:
  CudaWaitEvent(cudaStream_t stream);

  ~CudaWaitEvent() = default;

  CommsMaybeVoid beforeCollKernelScheduled() noexcept override;

  CommsMaybeVoid afterCollKernelScheduled() noexcept override;

  CommsMaybe<bool> waitCollStart(
      std::chrono::milliseconds sleepTimeMs) noexcept override;

  CommsMaybe<bool> waitCollEnd(
      std::chrono::milliseconds sleepTimeMs) noexcept override;

  CommsMaybeVoid signalCollStart() noexcept override;

  CommsMaybeVoid signalCollEnd() noexcept override;

  CommsMaybe<system_clock_time_point> getCollEnqueueTime() noexcept override;

  CommsMaybe<system_clock_time_point> getCollStartTime() noexcept override;

  CommsMaybe<system_clock_time_point> getCollEndTime() noexcept override;

 private:
  system_clock_time_point enqueueTime_;

  CudaWaitPoint startWaitPoint_;
  CudaWaitPoint endWaitPoint_;
};

} // namespace meta::comms::colltrace
