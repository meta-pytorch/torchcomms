// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <math.h>
#include <cstdint>
#include <unordered_map>

#include "comms/ctran/profiler/CtranProfiler.h"
#include "comms/utils/colltrace/NetworkPerfMonitor.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/LogUtils.h"

/*
 * The slow rank module reads rdma events like wqe start, end times
 * collected by the nccl profiler and calculates the average bus bw for each
 * rank. The average bus bw per rank is measured as Avg bw = sum of the bytes
 * sent / sum of WQE completion times This metric is calculated every time the
 * windowSize_ gets full and gets written to scuba when logging is enabled. If a
 * rank's bus bw is consistently below the threshold for a few sampling
 * intervals then it is considered a slow rank and an error is logged.(In future
 * we will have some alerting in place)
 */

#define MIN_RDMA_MESSAGE_SIZE 4096
#define MAX_RDMA_PERFORMANCE_EFFICIENCY_PERC 100

class CtranProfilerSlowRankModule : public CtranProfilerModule {
 public:
  explicit CtranProfilerSlowRankModule(CtranProfiler* profiler)
      : profiler_(profiler) {
    auto networkPerfMonitorPtr =
        ncclx::colltrace::NetworkPerfMonitor::getInstance();
    if (networkPerfMonitorPtr != nullptr) {
      const auto comm = profiler->getComm();
      if (comm != nullptr) {
        networkPerfMonitorPtr->storeCommInfo(
            comm->logMetaData_,
            comm->statex_->cudaDev(),
            comm->statex_->busId());
      }
    }
    CLOGF(INFO, "CtranProfilerSlowRankModule initialized");
  }
  ~CtranProfilerSlowRankModule() {
    CLOGF(INFO, "CtranProfilerSlowRankModule destroyed ");
    lastWindowAvgStats();
  }

  struct WqeCompletionStats {
    int localRank;
    int globalRank;
    int remoteRank;
    uint64_t totalBytes;
    uint64_t totalWqeTimeUs;
    uint64_t nReqsPerWindow;
    std::string hostName;
    std::string deviceName;
    std::string remoteHostName;
    std::string algorithmName;
    uint64_t messageSize;
    std::deque<double> rdmaPerfSlidingWindow;
    std::chrono::time_point<std::chrono::high_resolution_clock> firstWqePostTs_{
        std::chrono::high_resolution_clock::duration::zero()};
    std::chrono::time_point<std::chrono::high_resolution_clock>
        lastWqeCompleteTs_{
            std::chrono::high_resolution_clock::duration::zero()};
    bool isSrcRank{false};
    double rooflineBWGBps;
    double rdmaPerfEfficiencyPerc{MAX_RDMA_PERFORMANCE_EFFICIENCY_PERC};
    double avgBWGBps;
    std::chrono::time_point<std::chrono::high_resolution_clock> lastLogTs_{
        std::chrono::high_resolution_clock::duration::zero()};
    int scope;
    int nLoggedSamples{0};
    bool currSlowState{false};
    int wqeWindowSize{NCCL_SLOW_RANK_WQE_WINDOW_SIZE};
    unsigned long startDeviceByteOffset{0};
  };

  WqeCompletionStats* getWqeCompletionStats(int rank) {
    if (wqeCompletionStatsPerRank_.find(rank) !=
        wqeCompletionStatsPerRank_.end()) {
      return &wqeCompletionStatsPerRank_[rank];
    }
    return nullptr;
  }

  void setWqeWindowSize(WqeCompletionStats& stats, const Wqe& wqe) {
    // Set the wqe window size based on the collective and put size of the
    // message
    stats.wqeWindowSize =
        ceil(((wqe.messageSize + wqe.totalBytes - 1) / wqe.totalBytes)) *
        NCCL_SLOW_RANK_WQE_WINDOW_SIZE;
  }

 private:
  void onWqeComplete(const Wqe& wqe) override;
  void pushRdmaPerf(WqeCompletionStats& stats);
  void updateRdmaEfficiency(WqeCompletionStats& stats);
  double checkIfRankSlow(WqeCompletionStats& stats);
  bool checkRdmaEfficiency(WqeCompletionStats& stats);
  void setRooflineBw(WqeCompletionStats& stats, const Wqe& wqe);
  virtual void reportProfilingResultsToScuba(WqeCompletionStats& stats);
  void updateWqeCompletionStats(const Wqe& wqe, WqeCompletionStats& stats);
  void lastWindowAvgStats();
  void createOrUpdateWqeCompletionStats(
      std::unordered_map<int, CtranProfilerSlowRankModule::WqeCompletionStats>::
          iterator it,
      const Wqe& wqe,
      bool isSrcRank);
  void resetCounters(WqeCompletionStats& stats);
  void resetConfig(WqeCompletionStats& stats, const Wqe& wqe);
  int getConcurrentTransfers(WqeCompletionStats& stats);
  bool shouldLogRankEvent(WqeCompletionStats& stats);
  // The first item in map is the src rank stats
  std::unordered_map<int, WqeCompletionStats> wqeCompletionStatsPerRank_;
  CtranProfiler* profiler_;
};
