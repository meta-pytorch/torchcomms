// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/utils/colltrace/GraphCudaWaitEvent.h"

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy

#include <folly/Unit.h>
#include <folly/logging/xlog.h>

#include "comms/utils/PrecisionClock.h"
#include "comms/utils/checks.h"
#include "comms/utils/hrdw_ring_buffer/HRDWRingBuffer.h"

namespace meta::comms::colltrace {

GraphCudaWaitEvent::GraphCudaWaitEvent(cudaStream_t stream, uint32_t collId)
    : stream_(stream), collId_(collId), enqueueTime_(precisionNow()) {}

GraphCudaWaitEvent::~GraphCudaWaitEvent() = default;

void GraphCudaWaitEvent::attachRingBuffer(
    ::hrdw_ring_buffer::HRDWRingBuffer<GraphCollTraceEvent>*
        ringBuffer) noexcept {
  ringBuffer_ = ringBuffer;
}

CommsMaybeVoid GraphCudaWaitEvent::beforeCollKernelScheduled() noexcept {
  // The collective kernel publishes its own start timestamp into the ring from
  // inside the kernel, so there is nothing to schedule on the host. If no ring
  // was attached, the kernel emits nothing and there is no host-launched
  // fallback (removed at the in-kernel cutover), so this graph-captured
  // collective goes untimed. Warn a few times so that silent timing loss is
  // detectable rather than invisible, without spamming the log per collective.
  if (ringBuffer_ == nullptr) {
    XLOG_FIRST_N(WARNING, 8)
        << "GraphCudaWaitEvent: no in-kernel colltrace ring attached (collId="
        << collId_
        << "); graph-captured collective timing is unavailable on this device.";
  }
  return folly::unit;
}

CommsMaybeVoid GraphCudaWaitEvent::afterCollKernelScheduled() noexcept {
  // The collective kernel publishes its own end timestamp into the ring from
  // inside the kernel, so there is nothing to schedule on the host.
  return folly::unit;
}

CommsMaybe<bool> GraphCudaWaitEvent::waitCollStart(
    std::chrono::milliseconds /* sleepTimeMs */) noexcept {
  return false;
}

CommsMaybe<bool> GraphCudaWaitEvent::waitCollEnd(
    std::chrono::milliseconds /*sleepTimeMs*/) noexcept {
  // Graph completion is detected via the ring buffer poll thread, not here.
  return false;
}

CommsMaybeVoid GraphCudaWaitEvent::signalCollStart() noexcept {
  return folly::unit;
}

CommsMaybeVoid GraphCudaWaitEvent::signalCollEnd() noexcept {
  return folly::unit;
}

CommsMaybe<GraphCudaWaitEvent::system_clock_time_point>
GraphCudaWaitEvent::getCollEnqueueTime() noexcept {
  return enqueueTime_;
}

CommsMaybe<GraphCudaWaitEvent::system_clock_time_point>
GraphCudaWaitEvent::getCollStartTime() noexcept {
  // Graph timing is read from the ring buffer by the poll thread and set
  // directly on CollRecord — this method should not be called.
  return folly::makeUnexpected(CommsError(
      "GraphCudaWaitEvent: timing is provided by the poll thread, not via getCollStartTime",
      commInternalError));
}

CommsMaybe<GraphCudaWaitEvent::system_clock_time_point>
GraphCudaWaitEvent::getCollEndTime() noexcept {
  return folly::makeUnexpected(CommsError(
      "GraphCudaWaitEvent: timing is provided by the poll thread, not via getCollEndTime",
      commInternalError));
}

} // namespace meta::comms::colltrace
