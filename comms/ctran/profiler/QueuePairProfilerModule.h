// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <sys/types.h>

#include <chrono>
#include <deque>

#include "comms/ctran/profiler/CtranProfiler.h"

/*
 * The QP profiling module is responsible for profiling the performance of queue
 * pairs during training jobs.
 */

class QueuePairProfilerModule : public CtranProfilerModule {
 public:
  struct RdmaDataTransfer {
    int rank;
    int remoteRank;
    std::string deviceName;
    std::string hostName;
    std::string remoteHostName;
    uint64_t bytes;
    uint64_t bytesInFlightOnPost;
    uint64_t bytesInFlightOnComplete;
    int scope;
    size_t putSize;
    size_t opCode;
    size_t messageSize;
    std::string algorithmName;
    std::chrono::microseconds durationUs;
    std::chrono::microseconds idleTimeBeforeUs;
    std::chrono::microseconds timeFromPreviousWRPostUs;
    std::chrono::time_point<std::chrono::high_resolution_clock> postTs;
    std::chrono::time_point<std::chrono::high_resolution_clock> completionTs;
  };
  QueuePairProfilerModule(CtranProfiler* profiler);
  ~QueuePairProfilerModule() override;

  void onWqeComplete(const Wqe& wqe) override;
  std::vector<uint32_t> getQueuePairsProfiled() const;
  std::unordered_map<uint32_t, std::deque<RdmaDataTransfer>> const&
  getQueuePairs() const;

 private:
  std::unordered_map<uint32_t, std::deque<RdmaDataTransfer>> queuePairs_;
  CtranProfiler* profiler_;
};
