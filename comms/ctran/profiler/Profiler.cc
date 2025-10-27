// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/ctran/profiler/Profiler.h"
#include "comms/utils/logger/EventMgr.h"
#include "comms/utils/logger/ScubaLogger.h"

namespace {

template <typename Duration>
uint64_t getDurationUs(Duration d) {
  return std::chrono::duration_cast<std::chrono::microseconds>(d).count();
}

template <typename TimePoint>
uint64_t getDurationUs(TimePoint start, TimePoint end) {
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
      .count();
}

template <typename TimePoint>
uint64_t getTimeStamp(TimePoint timePoint) {
  return std::chrono::duration_cast<std::chrono::seconds>(
             timePoint.time_since_epoch())
      .count();
}

} // namespace

namespace ctran {

void Profiler::initForEachColl(int opCount, int samplingWeight) {
  shouldTrace_ = samplingWeight > 0 && (opCount % samplingWeight) == 0;
  if (shouldTrace_) {
    opCount_ = opCount;
    durations_.fill(0);
    readyTs_ = 0;
    controlTs_ = 0;
    startEvent(ProfilerEvent::ALGO_TOTAL);
  }
}

void Profiler::startEvent(
    ProfilerEvent event,
    const std::function<void(Profiler&)>& callback) {
  if (!shouldTrace_) {
    return;
  }
  const size_t idx = static_cast<size_t>(event);
  timers_[idx].reset();
  if (event == ProfilerEvent::ALGO_CTRL) {
    readyTs_ = getTimeStamp(timers_[idx].getCheckpoint());
  }
  if (callback) {
    callback(*this);
  }
}

void Profiler::endEvent(
    ProfilerEvent event,
    const std::function<void(Profiler&)>& callback) {
  if (!shouldTrace_) {
    return;
  }
  const size_t idx = static_cast<size_t>(event);
  durations_[idx] += getDurationUs(timers_[idx].lap());
  if (event == ProfilerEvent::ALGO_CTRL) {
    controlTs_ = getTimeStamp(timers_[idx].getCheckpoint());
  }
  if (callback) {
    callback(*this);
  }
}

void Profiler::reportToScuba() {
  if (!shouldTrace_) {
    return;
  }
  endEvent(ctran::ProfilerEvent::ALGO_TOTAL);
  logNcclProfilingAlgo();
  shouldTrace_ = false;
}

void Profiler::logNcclProfilingAlgo() const {
  const uint64_t bufferRegistrationTimeUs =
      durations_[static_cast<size_t>(ProfilerEvent::BUF_REG)];

  const uint64_t controlSyncTimeUs =
      durations_[static_cast<size_t>(ProfilerEvent::ALGO_CTRL)];

  const uint64_t dataTransferTimeUs =
      durations_[static_cast<size_t>(ProfilerEvent::ALGO_DATA)];

  const uint64_t timeFromDataToCollEndUs = getDurationUs(
      timers_[static_cast<size_t>(ProfilerEvent::ALGO_DATA)].getCheckpoint(),
      timers_[static_cast<size_t>(ProfilerEvent::ALGO_TOTAL)].getCheckpoint());

  const uint64_t collDurationUs =
      durations_[static_cast<size_t>(ProfilerEvent::ALGO_TOTAL)];

  NcclScubaEvent scubaEvent(
      std::make_unique<CtranProfilerAlgoEvent>(
          &comm_->logMetaData_,
          "algoProfilingV2",
          "",
          0,
          algoContext.peerRank,
          algoContext.deviceName,
          "",
          algoContext.algorithmName,
          algoContext.sendContext.messageSizes,
          algoContext.recvContext.messageSizes,
          "",
          algoContext.sendContext.totalBytes,
          algoContext.recvContext.totalBytes,
          bufferRegistrationTimeUs,
          controlSyncTimeUs,
          dataTransferTimeUs,
          opCount_,
          readyTs_,
          controlTs_,
          timeFromDataToCollEndUs,
          collDurationUs));
  scubaEvent.record();
}

} // namespace ctran
