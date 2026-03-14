// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/ctran/profiler/Profiler.h"
#include "comms/ctran/profiler/AlgoProfilerReport.h"
#include "comms/ctran/profiler/IAlgoProfilerReporter.h"
#include "comms/ctran/profiler/NcclxAlgoProfilerReporter.h"

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

Profiler::Profiler(CtranComm* comm)
    : comm_(comm), reporter_(std::make_unique<NcclxAlgoProfilerReporter>()) {}

Profiler::~Profiler() = default;

void Profiler::setReporter(std::unique_ptr<IAlgoProfilerReporter> reporter) {
  reporter_ = std::move(reporter);
}

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

  if (reporter_) {
    AlgoProfilerReport report;
    report.algoContext = &algoContext;
    report.logMetaData = &comm_->logMetaData_;
    report.opCount = opCount_;
    report.bufferRegistrationTimeUs =
        durations_[static_cast<size_t>(ProfilerEvent::BUF_REG)];
    report.controlSyncTimeUs =
        durations_[static_cast<size_t>(ProfilerEvent::ALGO_CTRL)];
    report.dataTransferTimeUs =
        durations_[static_cast<size_t>(ProfilerEvent::ALGO_DATA)];
    report.collectiveDurationUs =
        durations_[static_cast<size_t>(ProfilerEvent::ALGO_TOTAL)];
    report.readyTs = readyTs_;
    report.controlTs = controlTs_;
    report.timeFromDataToCollEndUs = getDurationUs(
        timers_[static_cast<size_t>(ProfilerEvent::ALGO_DATA)].getCheckpoint(),
        timers_[static_cast<size_t>(ProfilerEvent::ALGO_TOTAL)]
            .getCheckpoint());
    reporter_->report(report);
  }

  shouldTrace_ = false;
}

} // namespace ctran
