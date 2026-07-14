// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <atomic>
#include <memory>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/profiler/AlgoProfilerReport.h"
#include "comms/ctran/profiler/IProfilerReporter.h"
#include "comms/ctran/utils/StopWatch.h"

namespace ctran {

// execute the profiling command only if the condition is true
#define CTRAN_PROFILER_IF(profiler_, cmd_)     \
  if (profiler_ && profiler_->shouldTrace()) { \
    cmd_;                                      \
  }

#define CTRAN_PROFILER_CONDITION_IF(profiler_, condition, cmd_) \
  if (profiler_ && profiler_->shouldTrace() && condition) {     \
    cmd_;                                                       \
  }

enum ProfilerEvent {
  ALGO_TOTAL, // Total time for the algorithm, user should not use this
  ALGO_CTRL,
  ALGO_DATA,
  BUF_REG,
  NUM_PROFILER_EVENT_TYPES,
};

class Profiler {
 public:
  using Clock = std::chrono::system_clock;
  using EventDurationArray = std::array<uint64_t, NUM_PROFILER_EVENT_TYPES>;
  using EventTimerArray =
      std::array<utils::StopWatch<Clock>, NUM_PROFILER_EVENT_TYPES>;
  using EventHook = std::function<void()>;
  using EventHookArray =
      std::array<std::vector<EventHook>, NUM_PROFILER_EVENT_TYPES>;

 public:
  // Construct with a reporter. If nullptr, defaults to
  // DefaultAlgoProfilerReporter.
  // The reporter is immutable after construction.
  Profiler(
      CtranComm* comm,
      std::unique_ptr<IProfilerReporter> reporter = nullptr);
  ~Profiler();

  // This should be called at the beginning of the collective
  void initForEachColl(int opCount, int samplingWeight);

  void setForceTrace(bool force) {
    forceTrace_ = force;
  }

  bool shouldTrace() const {
    return shouldTrace_;
  }

  uint64_t getOpCount() const {
    return opCount_;
  }

  uint64_t getEventDurationUs(ProfilerEvent event) const {
    return durations_[static_cast<size_t>(event)];
  }

  uint64_t getReadyTs() const {
    return readyTs_;
  }

  uint64_t getControlTs() const {
    return controlTs_;
  }

  void startEvent(ProfilerEvent event);

  void endEvent(ProfilerEvent event);

  void reportToScuba();

  // Register a hook that fires at the start / end of the given
  // ProfilerEvent. Multiple hooks per (event, phase) are supported and
  // fire in registration order. Hooks should be cheap and exception-free
  // -- they run on the hot path of every traced collective.
  void registerStartHook(ProfilerEvent event, EventHook hook);
  void registerEndHook(ProfilerEvent event, EventHook hook);

 public:
  AlgoContext algoContext{};

 private:
  AlgoProfilerReport buildReport() const;
  CtranComm* comm_{nullptr};
  bool shouldTrace_{false};
  std::atomic<bool> forceTrace_{false};
  uint64_t opCount_{std::numeric_limits<uint64_t>::max()};
  EventDurationArray durations_{};
  EventTimerArray timers_{};
  uint64_t readyTs_{0};
  uint64_t controlTs_{0};
  std::unique_ptr<IProfilerReporter> reporter_;
  EventHookArray startHooks_{};
  EventHookArray endHooks_{};
};

} // namespace ctran
