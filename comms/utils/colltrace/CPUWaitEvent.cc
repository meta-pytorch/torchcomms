// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/colltrace/CPUWaitEvent.h"
#include <folly/Unit.h>

#include "comms/utils/colltrace/PrecisionClock.h"

namespace meta::comms::colltrace {

CPUWaitEvent::CPUWaitEvent() : enqueueTime_(precisionNow()) {}

CommsMaybeVoid CPUWaitEvent::beforeCollKernelScheduled() noexcept {
  // No op for CPU Wait event before collective kernel scheduled
  return folly::unit;
}

CommsMaybeVoid CPUWaitEvent::afterCollKernelScheduled() noexcept {
  // No op for CPU Wait event after collective kernel scheduled
  return folly::unit;
}

CommsMaybe<bool> CPUWaitEvent::waitCollStart(
    std::chrono::milliseconds sleepTimeMs) noexcept {
  return startEvent_.waitSemaphore.try_wait_for(sleepTimeMs);
}

CommsMaybe<bool> CPUWaitEvent::waitCollEnd(
    std::chrono::milliseconds sleepTimeMs) noexcept {
  // Dummy implementation
  return endEvent_.waitSemaphore.try_wait_for(sleepTimeMs);
}

CommsMaybeVoid CPUWaitEvent::signalCollStart() noexcept {
  startEvent_.waitSemaphore.post();
  startEvent_.timePoint = precisionNow();
  return folly::unit;
}

CommsMaybeVoid CPUWaitEvent::signalCollEnd() noexcept {
  endEvent_.waitSemaphore.post();
  endEvent_.timePoint = precisionNow();
  return folly::unit;
}

CommsMaybe<CPUWaitEvent::system_clock_time_point>
CPUWaitEvent::getCollEnqueueTime() noexcept {
  return enqueueTime_;
}

CommsMaybe<CPUWaitEvent::system_clock_time_point>
CPUWaitEvent::getCollStartTime() noexcept {
  if (!startEvent_.waitSemaphore.ready()) {
    return folly::makeUnexpected(CommsError(
        "CPUWaitEvent: getCollStartTime called before start time ready",
        commInternalError));
  }
  return startEvent_.timePoint;
}

CommsMaybe<CPUWaitEvent::system_clock_time_point>
CPUWaitEvent::getCollEndTime() noexcept {
  if (!endEvent_.waitSemaphore.ready()) {
    return folly::makeUnexpected(CommsError(
        "CPUWaitEvent: getCollEndTime called before end time ready",
        commInternalError));
  }
  return endEvent_.timePoint;
}

} // namespace meta::comms::colltrace
