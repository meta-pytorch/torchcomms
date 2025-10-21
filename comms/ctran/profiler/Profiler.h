// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/ctran/CtranComm.h"
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

struct DataContext {
  uint64_t totalBytes{0};
  std::string messageSizes{};
};

struct AlgoContext {
  std::string deviceName{};
  std::string algorithmName{};
  DataContext sendContext{};
  DataContext recvContext{};
  uint64_t peerRank{0};
};

class Profiler {
 public:
  using Clock = std::chrono::system_clock;
  using EventDurationArray = std::array<uint64_t, NUM_PROFILER_EVENT_TYPES>;
  using EventTimerArray =
      std::array<utils::StopWatch<Clock>, NUM_PROFILER_EVENT_TYPES>;

 public:
  Profiler(CtranComm* comm) : comm_(comm){};
  ~Profiler() = default;

  // This should be called at the beginning of the collective
  void initForEachColl(int opCount, int samplingWeight);

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

  void startEvent(
      ProfilerEvent event,
      const std::function<void(Profiler&)>& callback = {});

  void endEvent(
      ProfilerEvent event,
      const std::function<void(Profiler&)>& callback = {});

  void reportToScuba();

 public:
  AlgoContext algoContext{};

 private:
  CtranComm* comm_{nullptr};
  bool shouldTrace_{false};
  uint64_t opCount_{std::numeric_limits<uint64_t>::max()};
  EventDurationArray durations_{};
  EventTimerArray timers_{};
  uint64_t readyTs_{0};
  uint64_t controlTs_{0};

  void logNcclProfilingAlgo() const;

  friend class ProfilerTest;
};

} // namespace ctran
