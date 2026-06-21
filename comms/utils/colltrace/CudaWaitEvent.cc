// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/colltrace/CudaWaitEvent.h"

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy

#include <folly/ScopeGuard.h>
#include <folly/Unit.h>
#include <folly/stop_watch.h>

#include <folly/logging/xlog.h>

#include "comms/utils/CudaRAII.h"
#include "comms/utils/PrecisionClock.h"
#include "comms/utils/checks.h"
#include "comms/utils/colltrace/CudaEventPool.h"
#include "comms/utils/cvars/nccl_cvars.h" // @manual=fbcode//comms/utils/cvars:ncclx-cvars

namespace meta::comms::colltrace {

namespace {

// Singleton-existence flag set by the CudaReferencePoint ctor (release) and
// read by tryGet() (acquire). Lets the colltrace poll thread skip refresh
// when no CudaWaitEvent has ever been created (CPU-only colltrace usage).
// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
std::atomic<CudaReferencePoint*> g_referencePointInstance{nullptr};

CommsMaybe<bool> waitCudaEventFinish(
    const CudaEvent& event,
    std::chrono::milliseconds sleepTimeMs) {
  StreamCaptureModeGuard guard{cudaStreamCaptureModeRelaxed};
  // async polling case, query cuda whether event is ready every
  // NCCL_COLLTRACE_CHECK_INTERVAL_MS milliseconds
  folly::stop_watch<std::chrono::milliseconds> timer;
  auto res = cudaEventQuery(event.get());
  while (res != cudaSuccess && timer.elapsed() < sleepTimeMs) {
    if (res != cudaErrorNotReady) {
      CUDA_CHECK_EXPECTED(res);
    }
    std::this_thread::sleep_for(
        std::min(
            // In case timeout is smaller than the check interval specified
            std::chrono::milliseconds(NCCL_COLLTRACE_CHECK_INTERVAL_MS),
            sleepTimeMs));
    res = cudaEventQuery(event.get());
  }
  // Check whether we get out of the while loop due to event ready or timeout
  // reached
  return res == cudaSuccess;
}

} // namespace

CudaReferencePoint::CudaReferencePoint() {
  CUDA_CHECK(cudaStreamCreate(&stream_));
  cudaEvent_t event = nullptr;
  CUDA_CHECK(cudaEventCreate(&event));
  CUDA_CHECK(cudaEventRecord(event, stream_));
  // Wait for the reference event to fire so the host_time anchor below is
  // not earlier than the GPU reading. Stream is empty so this is fast.
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  *anchor_.wlock() = Anchor{.event = event, .time = precisionNow()};

  // Publish singleton handle for tryGet() — release-store so other threads'
  // acquire-loads see a fully-constructed object.
  g_referencePointInstance.store(this, std::memory_order_release);
}

/* static */ CudaReferencePoint* CudaReferencePoint::tryGet() {
  return g_referencePointInstance.load(std::memory_order_acquire);
}

void CudaReferencePoint::refresh() {
  // Try-and-skip: if another thread is already refreshing, short-circuit
  // without touching the dedicated CUDA stream. Multiple CollTrace
  // instances exist per process (one per NCCL communicator) and each calls
  // refresh from its own poll thread; the in-flight refresh will publish a
  // fresh anchor before our next wakeup, so we lose nothing by skipping.
  if (refresh_in_progress_.test_and_set(std::memory_order_acquire)) {
    return;
  }
  auto cleanup = folly::makeGuard(
      [this] { refresh_in_progress_.clear(std::memory_order_release); });

  cudaEvent_t newEvent = nullptr;
  if (auto err = cudaEventCreate(&newEvent); err != cudaSuccess) {
    XLOG_EVERY_MS(WARN, 5000)
        << "CudaReferencePoint::refresh: cudaEventCreate failed: "
        << cudaGetErrorString(err) << " — keeping previous anchor";
    return;
  }
  CHECK(newEvent != nullptr);
  if (auto err = cudaEventRecord(newEvent, stream_); err != cudaSuccess) {
    XLOG_EVERY_MS(WARN, 5000)
        << "CudaReferencePoint::refresh: cudaEventRecord failed: "
        << cudaGetErrorString(err) << " — keeping previous anchor";
    // Best-effort cleanup of an event we just created; any destroy error
    // here is unrecoverable and would only mask the record failure above.
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    (void)cudaEventDestroy(newEvent);
    return;
  }
  if (auto err = cudaStreamSynchronize(stream_); err != cudaSuccess) {
    XLOG_EVERY_MS(WARN, 5000)
        << "CudaReferencePoint::refresh: cudaStreamSynchronize failed: "
        << cudaGetErrorString(err) << " — keeping previous anchor";
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    (void)cudaEventDestroy(newEvent);
    return;
  }
  auto newTime = precisionNow();

  // Swap the anchor under wlock. Once wlock is granted, all in-flight
  // readers have released their rlock, so the old event is no longer
  // reachable from any subsequent reader. We destroy it *after* releasing
  // wlock so we never hold a writer lock across a CUDA call.
  cudaEvent_t oldEvent = nullptr;
  {
    auto a = anchor_.wlock();
    oldEvent = a->event;
    a->event = newEvent;
    a->time = newTime;
  }
  if (oldEvent != nullptr) {
    // Best-effort: a destroy error here would only be logged after the new
    // anchor is already published, so propagating it serves no purpose.
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    (void)cudaEventDestroy(oldEvent);
  }
}

CommsMaybe<ICollWaitEvent::system_clock_time_point>
CudaReferencePoint::getTimeViaEvent(const CudaEvent& event) {
  return getTimeViaEvent(event.get());
}

CommsMaybe<ICollWaitEvent::system_clock_time_point>
CudaReferencePoint::getTimeViaEvent(cudaEvent_t event) {
  // Hold rlock for the entire CUDA call so refresh() cannot destroy the
  // reference event mid-read. Multiple concurrent readers are unaffected
  // (shared_mutex); refresh briefly blocks waiting for in-flight rlocks.
  auto a = anchor_.rlock();
  float offsetMs;
  CUDA_CHECK_EXPECTED(cudaEventElapsedTime(&offsetMs, a->event, event));
  auto eventTime = a->time +
      std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::duration<float, std::milli>{offsetMs});
  return eventTime;
}

CudaWaitPoint::CudaWaitPoint(
    cudaStream_t stream,
    CudaWaitPoint::WaitPointType type)
    : recordStream_(stream), type_(type), event_(CudaEventPool::getEvent()) {
  // Record the reference point for the first time when the first wait point is
  // created.
  CudaWaitPoint::getReferencePoint();
}

std::string_view CudaWaitPoint::getWaitPointTypeString() const noexcept {
  switch (type_) {
    case WaitPointType::start: {
      return "start";
    }
    case WaitPointType::end: {
      return "end";
    }
    default: {
      return "unknown";
    }
  }
}

CommsMaybeVoid CudaWaitPoint::recordEvent() noexcept {
  if (eventStatus_ != CudaEventStatus::unrecorded) {
    return folly::makeUnexpected(CommsError(
        fmt::format(
            "CudaWaitPoint: recordEvent called for a already recorded event!",
            getWaitPointTypeString()),
        commInternalError));
  }
  CUDA_CHECK_EXPECTED(cudaEventRecord(event_.get(), recordStream_));
  eventStatus_ = CudaEventStatus::recorded;
  return folly::unit;
}

CommsMaybe<bool> CudaWaitPoint::waitEventFinish(
    std::chrono::milliseconds sleepTimeMs) noexcept {
  switch (eventStatus_) {
    case CudaEventStatus::unrecorded: {
      return folly::makeUnexpected(CommsError(
          fmt::format(
              "CudaWaitPoint: waitEventFinish called before {} event being recorded!",
              getWaitPointTypeString()),
          commInternalError));
    }
    case CudaEventStatus::finished: {
      return true;
    }
    case CudaEventStatus::recorded: {
      auto res = waitCudaEventFinish(event_.getRef(), sleepTimeMs);
      if (res.hasValue() && res.value()) {
        eventStatus_ = CudaEventStatus::finished;
      }
      return res;
    }
    default: {
      return folly::makeUnexpected(CommsError(
          fmt::format(
              "CudaWaitPoint: {} event is in unexpected status: {}",
              getWaitPointTypeString(),
              static_cast<int>(eventStatus_)),
          commInternalError));
    }
  }
}

/* static */ CudaReferencePoint& CudaWaitPoint::getReferencePoint() {
  static CudaReferencePoint referencePoint{};
  return referencePoint;
}

CommsMaybe<CudaWaitPoint::system_clock_time_point>
CudaWaitPoint::getEventFinishTime() noexcept {
  if (eventStatus_ != CudaEventStatus::finished) {
    return folly::makeUnexpected(CommsError(
        fmt::format(
            "CudaWaitPoint: getEventFinishTime called before {} event ready",
            getWaitPointTypeString()),
        commInternalError));
  }
  return CudaWaitPoint::getReferencePoint().getTimeViaEvent(event_.getRef());
}

CudaWaitEvent::CudaWaitEvent(cudaStream_t stream)
    : enqueueTime_(precisionNow()),
      startWaitPoint_(stream, CudaWaitPoint::WaitPointType::start),
      endWaitPoint_(stream, CudaWaitPoint::WaitPointType::end) {}

CommsMaybeVoid CudaWaitEvent::beforeCollKernelScheduled() noexcept {
  return startWaitPoint_.recordEvent();
}

CommsMaybeVoid CudaWaitEvent::afterCollKernelScheduled() noexcept {
  return endWaitPoint_.recordEvent();
}

CommsMaybe<bool> CudaWaitEvent::waitCollStart(
    std::chrono::milliseconds sleepTimeMs) noexcept {
  return startWaitPoint_.waitEventFinish(sleepTimeMs);
}

CommsMaybe<bool> CudaWaitEvent::waitCollEnd(
    std::chrono::milliseconds sleepTimeMs) noexcept {
  return endWaitPoint_.waitEventFinish(sleepTimeMs);
}

CommsMaybeVoid CudaWaitEvent::signalCollStart() noexcept {
  // For CudaWaitEvent, we ignore signal coll start/end from the CPU
  return folly::unit;
}

CommsMaybeVoid CudaWaitEvent::signalCollEnd() noexcept {
  // For CudaWaitEvent, we ignore signal coll start/end from the CPU
  return folly::unit;
}

CommsMaybe<CudaWaitEvent::system_clock_time_point>
CudaWaitEvent::getCollEnqueueTime() noexcept {
  return enqueueTime_;
}

CommsMaybe<CudaWaitEvent::system_clock_time_point>
CudaWaitEvent::getCollStartTime() noexcept {
  return startWaitPoint_.getEventFinishTime();
}

CommsMaybe<CudaWaitEvent::system_clock_time_point>
CudaWaitEvent::getCollEndTime() noexcept {
  return endWaitPoint_.getEventFinishTime();
}

} // namespace meta::comms::colltrace
