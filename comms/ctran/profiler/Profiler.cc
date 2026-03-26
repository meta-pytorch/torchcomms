// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/ctran/profiler/Profiler.h"
#include "comms/ctran/profiler/NcclxAlgoProfilerReporter.h"

#include <unordered_map>

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

namespace {

std::unordered_map<ReporterType, AlgoProfilerReporterFactory>&
getFactoryRegistry() {
  static std::unordered_map<ReporterType, AlgoProfilerReporterFactory> registry;
  return registry;
}

std::unique_ptr<IAlgoProfilerReporter> createReporter(
    ReporterType type,
    CtranComm* comm) {
  auto& registry = getFactoryRegistry();
  auto it = registry.find(type);
  if (it != registry.end()) {
    return it->second(comm);
  }
  return std::make_unique<NcclxAlgoProfilerReporter>();
}

} // namespace

void registerAlgoProfilerReporterFactory(
    ReporterType type,
    AlgoProfilerReporterFactory factory) {
  getFactoryRegistry()[type] = std::move(factory);
}

Profiler::Profiler(CtranComm* comm, ReporterType reporterType)
    : comm_(comm), reporter_(createReporter(reporterType, comm)) {}

Profiler::~Profiler() = default;

void Profiler::initForEachColl(uint64_t opCount, int samplingWeight) {
  invocationCount_++;
  shouldTrace_ = samplingWeight > 0 && (invocationCount_ % samplingWeight) == 0;
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

} // namespace ctran
