// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "CollTraceEvent.h"

#include <chrono>
#include <ratio>

#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/logger/DebugExt.h"

#include "strongstream.h"

namespace {
ncclResult_t waitGraphCaptureComplete(cudaStream_t stream) {
  struct ncclCudaGraph graph;
  do {
    auto res = ncclCudaGetCapturingGraph(&graph, stream);
    if (res != ncclSuccess) {
      WARN_FIRST_N(
          1, "Internal error: ncclCudaGetCapturingGraph failed by %d", res);
      return ncclInternalError;
    }
    if (graph.graph != nullptr) {
      std::this_thread::sleep_for(
          std::chrono::milliseconds(NCCL_COLLTRACE_CHECK_INTERVAL_MS));
    }
  } while (graph.graph != nullptr);
  return ncclSuccess;
}
} // namespace

ncclResult_t CpuWaitEvent::waitEventFinish() {
  bool finished;
  finishedSync_.withLock(
      [&finished](const bool& curStat) { finished = curStat; });
  if (finished) {
    return ncclSuccess;
  }
  auto lockedFinishStat = finishedSync_.lock();
  cv_.wait(lockedFinishStat.as_lock(), [&lockedFinishStat]() {
    return *lockedFinishStat;
  });
  return ncclSuccess;
}

ncclResult_t CpuWaitEvent::waitEventFinishAndExecute(
    const std::function<void()>& func) {
  bool finished;
  finishedSync_.withLock(
      [&finished](const bool& curStat) { finished = curStat; });
  if (finished) {
    return ncclSuccess;
  }
  auto lockedFinishStat = finishedSync_.lock();
  while (!finished) {
    cv_.wait_for(
        lockedFinishStat.as_lock(),
        std::chrono::milliseconds(NCCL_COLLTRACE_CHECK_INTERVAL_MS),
        [&lockedFinishStat]() { return *lockedFinishStat; });
    func();
    finished = *lockedFinishStat;
  }
  return ncclSuccess;
}

std::optional<float> CpuWaitEvent::getElapsedTimeSinceEvent(
    CollWaitEvent* start) {
  if (typeid(*start) != typeid(CpuWaitEvent)) {
    WARN_FIRST_N(
        kDebugRepeatLogCount,
        "CollTrace: Error trying to compare %s with CpuWaitEvent. getElapsedTimeSinceEvent only supports comparing events of the same type",
        typeid(*start).name());
    return std::nullopt;
  }
  auto* startCpuEvent = static_cast<CpuWaitEvent*>(start);
  auto timeDuration = this->finishTime_ - startCpuEvent->finishTime_;
  return std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(
             timeDuration)
      .count();
}

void CpuWaitEvent::setNotFinished() {
  *finishedSync_.lock() = false;
}

void CpuWaitEvent::setFinished() {
  finishTime_ = std::chrono::high_resolution_clock::now();
  finishedSync_.withLock([](bool& curStat) { curStat = true; });
  cv_.notify_all();
}

std::chrono::time_point<std::chrono::high_resolution_clock>
CpuWaitEvent::getFinishTime() {
  bool finished;
  finishedSync_.withLock(
      [&finished](const bool& curStat) { finished = curStat; });
  if (!finished) {
    ERR("CpuWaitEvent::getFinishTime() called before event finished!");
    // Return current time as a placeholder
    return std::chrono::high_resolution_clock::now();
  }
  return finishTime_;
}

std::optional<float> CudaWaitEvent::getElapsedTime(cudaEvent_t start) {
  float elapsedTime;
  auto res = cudaEventElapsedTime(&elapsedTime, start, event_.get());
  if (res != cudaSuccess) {
    return std::nullopt;
  }
  return elapsedTime;
}

void CudaWaitEvent::setStream(cudaStream_t stream) {
  stream_ = stream;
}

ncclResult_t CudaWaitEvent::waitEventFinish() {
  NCCLCHECK(waitGraphCaptureComplete(stream_));
  if (NCCL_COLLTRACE_EVENT_BLOCKING_SYNC) {
    CUDACHECK(cudaEventSynchronize(event_.get()));
    return ncclSuccess;
  }
  // async polling case, query cuda whether event is ready every
  // NCCL_COLLTRACE_CHECK_INTERVAL_MS milliseconds
  auto res = cudaEventQuery(event_.get());
  while (res != cudaSuccess) {
    if (res != cudaErrorNotReady) {
      CUDACHECK(res);
    }
    std::this_thread::sleep_for(
        std::chrono::milliseconds(NCCL_COLLTRACE_CHECK_INTERVAL_MS));
    res = cudaEventQuery(event_.get());
  }
  return ncclSuccess;
}

ncclResult_t CudaWaitEvent::waitEventFinishAndExecute(
    const std::function<void()>& func) {
  NCCLCHECK(waitGraphCaptureComplete(stream_));
  if (NCCL_COLLTRACE_EVENT_BLOCKING_SYNC) {
    CUDACHECK(cudaEventSynchronize(event_.get()));
    return ncclSuccess;
  }
  // async polling case, query cuda whether event is ready every
  // NCCL_COLLTRACE_CHECK_INTERVAL_MS milliseconds
  auto res = cudaEventQuery(event_.get());
  while (res != cudaSuccess) {
    if (res != cudaErrorNotReady) {
      CUDACHECK(res);
    }
    func();
    std::this_thread::sleep_for(
        std::chrono::milliseconds(NCCL_COLLTRACE_CHECK_INTERVAL_MS));
    res = cudaEventQuery(event_.get());
  }
  return ncclSuccess;
}

std::optional<float> CudaWaitEvent::getElapsedTimeSinceEvent(
    CollWaitEvent* start) {
  if (typeid(*start) != typeid(CudaWaitEvent)) {
    WARN_FIRST_N(
        kDebugRepeatLogCount,
        "CollTrace: Error trying to compare %s with CudaWaitEvent. getElapsedTimeSinceEvent only supports comparing events of the same type",
        typeid(*start).name());
    return std::nullopt;
  }
  auto* startCudaEvent = static_cast<CudaWaitEvent*>(start);
  return getElapsedTime(startCudaEvent->event_.get());
}
