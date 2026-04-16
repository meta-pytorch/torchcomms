// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/utils/colltrace/GraphCudaWaitEvent.h"

#include <cstring>

#include <cuda_runtime.h> // @manual=third-party//cuda:cuda-lazy
#include "comms/utils/colltrace/GpuClockCalibration.h"

#include <folly/Unit.h>
#include <folly/logging/xlog.h>

#include "comms/utils/CudaRAII.h"
#include "comms/utils/HRDWRingBuffer.h"
#include "comms/utils/checks.h"

namespace meta::comms::colltrace {

GraphCudaWaitEvent::GraphCudaWaitEvent(cudaStream_t stream, uint32_t collId)
    : stream_(stream),
      collId_(collId),
      enqueueTime_(std::chrono::system_clock::now()) {
  // Eagerly initialize the globaltimer calibration singleton so that the
  // calibration kernel doesn't fire during graph capture.
  GlobaltimerCalibration::get();

  // Create per-collective CUDA resources using relaxed capture mode so
  // they don't interfere with the graph capture in progress.
  StreamCaptureModeGuard guard{cudaStreamCaptureModeRelaxed};
  CUDA_CHECK(cudaStreamCreate(&timestampStream_));
  CUDA_CHECK(cudaEventCreateWithFlags(&depEvent_, cudaEventDisableTiming));
}

GraphCudaWaitEvent::~GraphCudaWaitEvent() {
  if (timestampStream_) {
    CUDA_CHECK_WITH_IGNORE(
        cudaStreamDestroy(timestampStream_),
        cudaErrorCudartUnloading,
        cudaErrorContextIsDestroyed);
  }
  if (depEvent_) {
    CUDA_CHECK_WITH_IGNORE(
        cudaEventDestroy(depEvent_),
        cudaErrorCudartUnloading,
        cudaErrorContextIsDestroyed);
  }
}

void GraphCudaWaitEvent::attachRingBuffer(
    HRDWRingBuffer<GraphCollTraceEvent>* ringBuffer) noexcept {
  ringBuffer_ = ringBuffer;
}

CommsMaybeVoid GraphCudaWaitEvent::beforeCollKernelScheduled() noexcept {
  // Fork: collective stream -> timestamp stream so the start kernel is
  // captured into the graph. We use cudaStreamWaitEvent to bring the
  // timestamp stream into the capture with a dependency on the main
  // stream's current position (right before the collective launches).
  CUDA_CHECK_EXPECTED(cudaEventRecord(depEvent_, stream_));
  CUDA_CHECK_EXPECTED(cudaStreamWaitEvent(timestampStream_, depEvent_));

  // Launch the start timestamp kernel on this collective's timestamp stream.
  CUDA_CHECK_EXPECTED(ringBuffer_->write(
      timestampStream_,
      GraphCollTraceEvent{collId_, GraphCollTracePhase::kStart}));

  return folly::unit;
}

CommsMaybeVoid GraphCudaWaitEvent::afterCollKernelScheduled() noexcept {
  // Fork: collective stream -> timestamp stream. This DAG edge ensures the
  // end timestamp kernel fires only after the collective completes.
  CUDA_CHECK_EXPECTED(cudaEventRecord(depEvent_, stream_));
  CUDA_CHECK_EXPECTED(cudaStreamWaitEvent(timestampStream_, depEvent_));

  // Launch the end timestamp kernel on this collective's timestamp stream.
  CUDA_CHECK_EXPECTED(ringBuffer_->write(
      timestampStream_,
      GraphCollTraceEvent{collId_, GraphCollTracePhase::kEnd}));

  // Save the collective stream's capture deps before the rejoin.
  cudaStreamCaptureStatus status;
  const cudaGraphNode_t* preRejoinDeps = nullptr;
  const cudaGraphEdgeData* preRejoinEdgeData = nullptr;
  size_t numPreRejoinDeps = 0;
#if CUDART_VERSION >= 13000
  CUDA_CHECK_EXPECTED(cudaStreamGetCaptureInfo(
      stream_,
      &status,
      nullptr,
      nullptr,
      &preRejoinDeps,
      &preRejoinEdgeData,
      &numPreRejoinDeps));
#else
  CUDA_CHECK_EXPECTED(cudaStreamGetCaptureInfo_v2(
      stream_, &status, nullptr, nullptr, &preRejoinDeps, &numPreRejoinDeps));
#endif
  std::vector<cudaGraphNode_t> savedDeps(
      preRejoinDeps, preRejoinDeps + numPreRejoinDeps);
  std::vector<cudaGraphEdgeData> savedEdgeData(
      preRejoinEdgeData,
      preRejoinEdgeData ? preRejoinEdgeData + numPreRejoinDeps
                        : preRejoinEdgeData);

  // Rejoin: timestamp stream -> collective stream. Required by
  // cudaStreamEndCapture (all forked streams must rejoin). We immediately
  // restore the pre-rejoin deps afterward so the rejoin node exists in
  // the graph DAG but is NOT a dependency of subsequent ops on the
  // collective stream. This means non-traced ops between collectives
  // won't be serialized behind the end kernel.
  CUDA_CHECK_EXPECTED(cudaEventRecord(depEvent_, timestampStream_));
  CUDA_CHECK_EXPECTED(cudaStreamWaitEvent(stream_, depEvent_));

  // Immediately undo: restore pre-rejoin deps so the main stream
  // doesn't carry the endK dependency forward.
  CUDA_CHECK_EXPECTED(cudaStreamUpdateCaptureDependencies(
      stream_,
      savedDeps.data(),
#if CUDART_VERSION >= 13000
      savedEdgeData.empty() ? nullptr : savedEdgeData.data(),
#endif
      savedDeps.size(),
      cudaStreamSetCaptureDependencies));

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
