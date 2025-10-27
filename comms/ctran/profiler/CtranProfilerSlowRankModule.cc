// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/profiler/CtranProfilerSlowRankModule.h"

#include <algorithm>
#include <chrono>
#include <cmath>

#include "comms/utils/colltrace/NetworkPerfMonitor.h"
#include "comms/utils/logger/LogUtils.h"
#include "comms/utils/logger/ScubaLogger.h"

void CtranProfilerSlowRankModule::resetCounters(WqeCompletionStats& stats) {
  stats.totalBytes = 0;
  stats.totalWqeTimeUs = 0;
  stats.nReqsPerWindow = 0;
}

void CtranProfilerSlowRankModule::resetConfig(
    WqeCompletionStats& stats,
    const Wqe& wqe) {
  resetCounters(stats);
  stats.algorithmName = wqe.algorithmName;
  stats.messageSize = wqe.messageSize;
  stats.rdmaPerfEfficiencyPerc = MAX_RDMA_PERFORMANCE_EFFICIENCY_PERC;
}

void CtranProfilerSlowRankModule::pushRdmaPerf(WqeCompletionStats& stats) {
  stats.avgBWGBps = stats.totalBytes / (stats.totalWqeTimeUs * 1E3);
  double rdmaPerfEfficiencyPerc =
      std::min(100.0, (stats.avgBWGBps / stats.rooflineBWGBps) * 100);
  if (stats.rdmaPerfSlidingWindow.size() >= NCCL_SLOW_RANK_PERF_WINDOW_SIZE) {
    stats.rdmaPerfSlidingWindow.pop_front();
  }
  stats.rdmaPerfSlidingWindow.push_back(rdmaPerfEfficiencyPerc);
}

double CtranProfilerSlowRankModule::checkIfRankSlow(WqeCompletionStats& stats) {
  // If all rdma performance efficiency values in the window is less than 70%
  // then we consider it as a slow rank. If the rank is slow we return
  // lowest value in the window else we return the first value above the
  // threshold
  double currPerf = MAX_RDMA_PERFORMANCE_EFFICIENCY_PERC;
  stats.currSlowState = true;
  if (stats.rdmaPerfSlidingWindow.size() < NCCL_SLOW_RANK_PERF_WINDOW_SIZE) {
    return false;
  }
  for (auto& i : stats.rdmaPerfSlidingWindow) {
    if (i > NCCL_SLOW_RANK_RDMA_PERF_EFFICIENCY_PERC) {
      stats.currSlowState = false;
      currPerf = i;
      break;
    }
    if (currPerf > i) {
      currPerf = i;
    }
  }
  return currPerf;
}

bool CtranProfilerSlowRankModule::shouldLogRankEvent(
    WqeCompletionStats& stats) {
  // We log data in scuba if one of the following conditions are met:
  // 1. The rank is slow and it was not previously slow.
  // 2. The rank is not slow and it was previously slow.
  // 3. The rank is slow and the rdma performance efficiency changed by more
  // than 20% lower than the previous value.
  bool prevState = stats.currSlowState;
  double currPerf = checkIfRankSlow(stats);
  auto percDiff = (abs(currPerf - stats.rdmaPerfEfficiencyPerc) /
                   stats.rdmaPerfEfficiencyPerc) *
      100;
  if ((prevState && !stats.currSlowState) ||
      (!prevState && stats.currSlowState) ||
      (prevState && stats.currSlowState &&
       percDiff > NCCL_SLOW_RANK_VARIANCE_PERC)) {
    stats.rdmaPerfEfficiencyPerc = currPerf;
    return true;
  }
  return false;
}

int CtranProfilerSlowRankModule::getConcurrentTransfers(
    WqeCompletionStats& stats) {
  // Calculate the number of concurrent communications that occur
  // simulataneously. We check if the current transmit window for a given
  // destination rank overlaps with other communicating ranks and calculate the
  // overlap percentage.
  int nConcurrentTransfers = 1;
  auto len1 = std::chrono::duration_cast<std::chrono::microseconds>(
                  stats.lastWqeCompleteTs_ - stats.firstWqePostTs_)
                  .count();
  for (const auto& it : wqeCompletionStatsPerRank_) {
    if (it.first == stats.remoteRank || it.second.isSrcRank) {
      continue;
    }
    // // Calculate the length of the interval
    auto len2 = std::chrono::duration_cast<std::chrono::microseconds>(
                    it.second.lastWqeCompleteTs_ - it.second.firstWqePostTs_)
                    .count();
    // Calculate the overlap length
    auto overlap_len = std::fmax(
        0,
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::min(stats.lastWqeCompleteTs_, it.second.lastWqeCompleteTs_)
                .time_since_epoch())
                .count() -
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::max(stats.firstWqePostTs_, it.second.firstWqePostTs_)
                    .time_since_epoch())
                .count() +
            1);
    // Calculate the overlap percentage
    double overlap_percentage =
        ((double)overlap_len / (len1 + len2 - overlap_len)) * 100.0;
    if (overlap_percentage > 0) {
      nConcurrentTransfers++;
    }
  }
  return nConcurrentTransfers;
}

void CtranProfilerSlowRankModule::setRooflineBw(
    WqeCompletionStats& stats,
    const Wqe& wqe) {
  // If its a 1:1 communication pattern, then the max bw is equal to NIC line
  // rate. If its 1:N comm like in the case of A2A then NIC is shared between
  // ranks. So the max bw per rank is equal to NIC line rate / number of peer
  // ranks. The thresholds computed based on inrack flows.  Depending on the
  // number of hops to the destination rank the max achievable bw can vary.
  // https://fb.workplace.com/groups/270400687389521/permalink/1087712588991656/
  const int thresholdDevice = 50; // 50GBps(400Gbps) is assuming its GT. Need to
                                  // updated based on model type
  // If the total bytes transferred from the ib device during the window is
  // greater than the total bytes transferred by the communicator then we
  // consider an overlapping comm is active
  auto totalBytesOnDevice =
      wqe.deviceByteOffsetAfterPost - stats.startDeviceByteOffset + 1;
  auto commTrafficRatio =
      std::min(1.0, ((stats.totalBytes * 1.0) / totalBytesOnDevice));
  int thresholdComm = thresholdDevice * commTrafficRatio;
  int thresholdTransfer = thresholdComm;
  if (!stats.isSrcRank && wqeCompletionStatsPerRank_.size() > 1) {
    thresholdTransfer = thresholdComm / getConcurrentTransfers(stats);
  }

  // Expected thresholds for various message sizes -  P1395220141
  auto msgSize = std::log2(stats.messageSize) - 12;
  auto msgSize_threshold = thresholdTransfer /
      std::pow((1 + std::exp(std::sqrt(thresholdTransfer) - msgSize)), 0.5);
  // TODO: More tuning maybe required for smaller message sizes.
  stats.rooflineBWGBps = msgSize_threshold;
}

void CtranProfilerSlowRankModule::updateRdmaEfficiency(
    WqeCompletionStats& stats) {
  if (stats.totalWqeTimeUs == 0 || stats.rooflineBWGBps == 0 ||
      stats.isSrcRank) {
    return;
  }
  pushRdmaPerf(stats);
  auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::system_clock::now().time_since_epoch() -
                         stats.lastLogTs_.time_since_epoch())
                         .count();

  // If the state transitions or an anomaly is detected(rdma perf efficiency is
  // consistently low) and the time since last logging
  // is greater than NCCL_SLOW_RANK_MODULE_SCUBA_LOGGING_INTERVAL then we log
  // the results to scuba and reset the last logging time.
  if (elapsedTime > NCCL_SLOW_RANK_SCUBA_LOGGING_INTERVAL_IN_USECS) {
    if (stats.nLoggedSamples < NCCL_SLOW_RANK_LOG_NSAMPLES) {
      stats.nLoggedSamples++;
      reportProfilingResultsToScuba(stats);
      CLOGF_SUBSYS(
          INFO,
          COLL,
          "comm {} nranks: {} localrank: {} remote rank: {}  deviceName: {} "
          "messageSize: {} algorithm {} scope {} totalBytes {} totalWqeTimeUs {} "
          "avgBS {} (GB/s) roofline: {} (GB/s) nReqsPerWindow: {} srcNIC: {}, "
          "firstWqeCompleteTs {} lastWqestats {} peernodes {}",
          profiler_->getComm()->logMetaData_.commId,
          profiler_->getComm()->logMetaData_.nRanks,
          stats.globalRank,
          stats.remoteRank,
          stats.deviceName.c_str(),
          stats.messageSize,
          stats.algorithmName.c_str(),
          stats.scope,
          stats.totalBytes,
          stats.totalWqeTimeUs,
          stats.totalBytes / (stats.totalWqeTimeUs * 1E3),
          stats.rooflineBWGBps,
          stats.nReqsPerWindow,
          stats.isSrcRank,
          std::chrono::duration_cast<std::chrono::microseconds>(
              stats.firstWqePostTs_.time_since_epoch())
              .count(),
          std::chrono::duration_cast<std::chrono::microseconds>(
              stats.lastWqeCompleteTs_.time_since_epoch())
              .count(),
          wqeCompletionStatsPerRank_.size());
    } else if (shouldLogRankEvent(stats)) {
      reportProfilingResultsToScuba(stats);
      CLOGF_SUBSYS(
          INFO,
          COLL,
          "Sender rank: {} remote rank: {} deviceName: {} Rdma performance efficiency: {} Observed BW: {} GBps Roofline: {} GBps",
          stats.globalRank,
          stats.remoteRank,
          stats.deviceName.c_str(),
          stats.rdmaPerfSlidingWindow.back(),
          stats.avgBWGBps,
          stats.rooflineBWGBps);
    }
  }
}

void CtranProfilerSlowRankModule::updateWqeCompletionStats(
    const Wqe& wqe,
    WqeCompletionStats& stats) {
  // Every time a WQE is completed we update the total bytes and time taken by
  // all WQE's till that point. If the number of WQE's reach
  // NCCL_SLOW_RANK_MODULE_WQE_WINDOW_SIZE, we compute the average BW, push it
  // in queue and log it in scuba The Average BW in MBps given by = totalBytes /
  // (totalTime * 1E3)

  auto wqePostTs = wqe.postTs;
  if (stats.nReqsPerWindow == 0) {
    stats.firstWqePostTs_ = wqePostTs;
    stats.startDeviceByteOffset = wqe.deviceByteOffsetAfterPost;
  } else {
    if ((wqePostTs.time_since_epoch().count() <
         stats.firstWqePostTs_.time_since_epoch().count())) {
      // If the new WQE starts before the previous started then we just add
      // (start_time of WQE 1 - start_time of WQE 2) to the total completion
      // time. For example: wqe 1: s:2 -> e:4, wqe 2: s:1 -> e:5
      stats.totalWqeTimeUs +=
          std::chrono::duration_cast<std::chrono::microseconds>(
              stats.firstWqePostTs_ - wqePostTs)
              .count();
      stats.firstWqePostTs_ = wqePostTs;
      stats.startDeviceByteOffset = wqe.deviceByteOffsetAfterPost;
    }
    if (wqePostTs.time_since_epoch().count() <
        stats.lastWqeCompleteTs_.time_since_epoch().count()) {
      // If the new WQE starts before the previous one finished then we add
      // (end_time of WQE 2 - end_time of WQE 1) to the total completion time
      // For example: wqe 1: s:0 -> e:2, wqe 2: s:1 -> e:4
      wqePostTs = stats.lastWqeCompleteTs_;
    }
  }
  assert(wqe.completionTs >= stats.lastWqeCompleteTs_);
  stats.lastWqeCompleteTs_ = wqe.completionTs;
  stats.totalBytes += wqe.totalBytes;
  stats.totalWqeTimeUs += std::chrono::duration_cast<std::chrono::microseconds>(
                              wqe.completionTs - wqePostTs)
                              .count();
  stats.nReqsPerWindow++;

  if ((stats.nReqsPerWindow % stats.wqeWindowSize) == 0) {
    setRooflineBw(stats, wqe);
    updateRdmaEfficiency(stats);
    resetCounters(stats);
  }
}

void CtranProfilerSlowRankModule::createOrUpdateWqeCompletionStats(
    std::unordered_map<int, CtranProfilerSlowRankModule::WqeCompletionStats>::
        iterator it,
    const Wqe& wqe,
    bool isSrcRank) {
  auto rank = isSrcRank ? wqe.globalRank : wqe.remoteRank;
  if (it == wqeCompletionStatsPerRank_.end()) {
    auto stats = WqeCompletionStats{
        .localRank = wqe.localRank,
        .globalRank = wqe.globalRank,
        .remoteRank = wqe.remoteRank,
        .hostName = wqe.hostName,
        .deviceName = wqe.deviceName,
        .remoteHostName = wqe.remoteHostName,
        .algorithmName = wqe.algorithmName,
        .messageSize = wqe.messageSize,
        .firstWqePostTs_ = wqe.postTs,
        .lastWqeCompleteTs_ = wqe.completionTs,
        .isSrcRank = isSrcRank,
        .lastLogTs_ = std::chrono::system_clock::now(),
        .scope = wqe.scope,
    };
    CLOGF_SUBSYS(
        INFO,
        COLL,
        "New stats creation comm {} nranks: {} localrank: {} remote rank: {}  deviceName: {} messageSize: {} algorithm {} scope {}",
        profiler_->getComm()->logMetaData_.commId,
        profiler_->getComm()->logMetaData_.nRanks,
        stats.globalRank,
        stats.remoteRank,
        stats.deviceName.c_str(),
        stats.messageSize,
        stats.algorithmName.c_str(),
        stats.scope);

    setWqeWindowSize(stats, wqe);
    wqeCompletionStatsPerRank_.insert(std::make_pair(rank, stats));
  }
  it = wqeCompletionStatsPerRank_.find(rank);
  // If the message size or algorithm name changes, we reset the stats.
  if (it->second.messageSize != wqe.messageSize ||
      it->second.algorithmName != wqe.algorithmName) {
    CLOGF_SUBSYS(
        INFO,
        COLL,
        "New message size comm {} nranks: {} localrank: {} remote rank: {}  deviceName: {} messageSize: {} algorithm {} scope {}",
        profiler_->getComm()->logMetaData_.commId,
        profiler_->getComm()->logMetaData_.nRanks,
        it->second.globalRank,
        it->second.remoteRank,
        it->second.deviceName.c_str(),
        it->second.messageSize,
        it->second.algorithmName.c_str(),
        it->second.scope);
    // Reset the window size if the message size changes
    setWqeWindowSize(it->second, wqe);
    resetConfig(it->second, wqe);
  }
  updateWqeCompletionStats(wqe, it->second);
}

void CtranProfilerSlowRankModule::onWqeComplete(const Wqe& wqe) {
  // Ignore small control messages like barriers that get exchanged between
  // ranks.
  if (wqe.messageSize < MIN_RDMA_MESSAGE_SIZE) {
    return;
  }
  auto networkPerfMonitorPtr =
      ncclx::colltrace::NetworkPerfMonitor::getInstance();
  if (networkPerfMonitorPtr != nullptr &&
      networkPerfMonitorPtr->checkIfRecordRDMAEvent(wqe.messageSize)) {
    ncclx::colltrace::RDMACompletionEvent event{
        .postTs = wqe.postTs,
        .completionTs = wqe.completionTs,
        .remoteRank = wqe.remoteRank,
        .totalBytes = wqe.totalBytes,
        .messageSize = wqe.messageSize,
        .commHash = profiler_->getComm()->statex_->commHash(),
    };
    networkPerfMonitorPtr->recordRDMAEvent(std::move(event));
  }
  // global src rank view
  auto it = wqeCompletionStatsPerRank_.find(wqe.globalRank);
  createOrUpdateWqeCompletionStats(it, wqe, true);

  // per destination rank view
  it = wqeCompletionStatsPerRank_.find(wqe.remoteRank);
  createOrUpdateWqeCompletionStats(it, wqe, false);
}

void CtranProfilerSlowRankModule::reportProfilingResultsToScuba(
    WqeCompletionStats& stats) {
  // Logs data only if NCCL_CTRAN_PROFILER_SLOW_RANK_LOGGING env flag is set.
  // nccl_profiler_slow_rank is the default scuba table for logging slow rank
  // profiling data.
  stats.lastLogTs_ = std::chrono::system_clock::now();
  NcclScubaEvent scubaEvent(
      std::make_unique<CtranProfilerSlowRankEvent>(
          &profiler_->getComm()->logMetaData_,
          "slowRankProfiling",
          "",
          0,
          stats.remoteRank,
          stats.deviceName,
          stats.remoteHostName,
          stats.algorithmName,
          std::to_string(stats.messageSize),
          "0",
          stats.avgBWGBps,
          stats.nReqsPerWindow,
          stats.rooflineBWGBps,
          stats.rdmaPerfSlidingWindow.back()));
  scubaEvent.record();
}

void CtranProfilerSlowRankModule::lastWindowAvgStats() {
  for (auto& stats : wqeCompletionStatsPerRank_) {
    if ((stats.second.nReqsPerWindow % stats.second.wqeWindowSize) != 0) {
      updateRdmaEfficiency(stats.second);
    }
  }
}
