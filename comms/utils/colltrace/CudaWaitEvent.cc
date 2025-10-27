// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/colltrace/CudaWaitEvent.h"

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy

#include <folly/Unit.h>
#include <folly/stop_watch.h>

#include "comms/utils/checks.h"
#include "comms/utils/colltrace/CudaEventPool.h"
#include "comms/utils/cvars/nccl_cvars.h" // @manual=fbcode//comms/utils/cvars:ncclx-cvars

namespace meta::comms::colltrace {

namespace {
CommsMaybe<bool> waitCudaEventFinish(
    const CudaEvent& event,
    std::chrono::milliseconds sleepTimeMs) {
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
  CUDA_CHECK(cudaEventRecord(event_.get(), stream_.get()));
  time_ = std::chrono::system_clock::now();
}

CommsMaybe<ICollWaitEvent::system_clock_time_point>
CudaReferencePoint::getTimeViaEvent(const CudaEvent& event) {
  float offsetMs;
  CUDA_CHECK_EXPECTED(
      cudaEventElapsedTime(&offsetMs, this->event_.get(), event.get()));
  auto eventTime = this->time_ +
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
  return folly::makeUnexpected(
      CommsError("CudaWaitPoint: Reached unexpected code", commInternalError));
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
    : enqueueTime_(std::chrono::system_clock::now()),
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
