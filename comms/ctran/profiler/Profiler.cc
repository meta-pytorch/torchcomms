// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/ctran/profiler/Profiler.h"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/profiler/DefaultAlgoProfilerReporter.h"

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

Profiler::Profiler(CtranComm* comm, std::unique_ptr<IProfilerReporter> reporter)
    : comm_(comm),
      reporter_(
          reporter ? std::move(reporter)
                   : std::make_unique<DefaultAlgoProfilerReporter>()) {}

Profiler::~Profiler() = default;

void Profiler::initForEachColl(int opCount, int samplingWeight) {
  shouldTrace_ =
      forceTrace_ || (samplingWeight > 0 && (opCount % samplingWeight) == 0);
  if (shouldTrace_) {
    opCount_ = opCount;
    durations_.fill(0);
    readyTs_ = 0;
    controlTs_ = 0;
    startEvent(ProfilerEvent::ALGO_TOTAL);
  }
}

void Profiler::startEvent(ProfilerEvent event) {
  if (!shouldTrace_) {
    return;
  }
  const size_t idx = static_cast<size_t>(event);
  timers_[idx].reset();
  if (event == ProfilerEvent::ALGO_CTRL) {
    readyTs_ = getTimeStamp(timers_[idx].getCheckpoint());
  }
  for (const auto& hook : startHooks_[idx]) {
    hook();
  }
}

void Profiler::endEvent(ProfilerEvent event) {
  if (!shouldTrace_) {
    return;
  }
  const size_t idx = static_cast<size_t>(event);
  durations_[idx] += getDurationUs(timers_[idx].lap());
  if (event == ProfilerEvent::ALGO_CTRL) {
    controlTs_ = getTimeStamp(timers_[idx].getCheckpoint());
  }
  for (const auto& hook : endHooks_[idx]) {
    hook();
  }
}

AlgoProfilerReport Profiler::buildReport() const {
  return {
      .algoContext = &algoContext,
      .logMetaData = &comm_->logMetaData_,
      .opCount = opCount_,
      .bufferRegistrationTimeUs =
          durations_[static_cast<size_t>(ProfilerEvent::BUF_REG)],
      .controlSyncTimeUs =
          durations_[static_cast<size_t>(ProfilerEvent::ALGO_CTRL)],
      .dataTransferTimeUs =
          durations_[static_cast<size_t>(ProfilerEvent::ALGO_DATA)],
      .collectiveDurationUs =
          durations_[static_cast<size_t>(ProfilerEvent::ALGO_TOTAL)],
      .readyTs = readyTs_,
      .controlTs = controlTs_,
      .timeFromDataToCollEndUs = getDurationUs(
          timers_[static_cast<size_t>(ProfilerEvent::ALGO_DATA)]
              .getCheckpoint(),
          timers_[static_cast<size_t>(ProfilerEvent::ALGO_TOTAL)]
              .getCheckpoint()),
  };
}

void Profiler::reportToScuba() {
  if (!shouldTrace_) {
    return;
  }
  endEvent(ctran::ProfilerEvent::ALGO_TOTAL);

  if (reporter_) {
    reporter_->report(buildReport());
  }

  shouldTrace_ = false;
}

void Profiler::registerStartHook(ProfilerEvent event, EventHook hook) {
  startHooks_[static_cast<size_t>(event)].push_back(std::move(hook));
}

void Profiler::registerEndHook(ProfilerEvent event, EventHook hook) {
  endHooks_[static_cast<size_t>(event)].push_back(std::move(hook));
}

} // namespace ctran
