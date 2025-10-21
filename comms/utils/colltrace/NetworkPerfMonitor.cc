// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/colltrace/NetworkPerfMonitor.h"

#include <folly/Singleton.h>
#include <folly/json/dynamic.h>
#include <folly/json/json.h>
#include <folly/system/ThreadName.h>
#include <chrono>
#include <thread>

#include "comms/utils/StrUtils.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/cvars/nccl_cvars.h" // @manual=fbcode//comms/utils/cvars:ncclx-cvars
#include "comms/utils/logger/LoggingFormat.h"
#include "comms/utils/logger/ScubaLogger.h"

namespace ncclx::colltrace {

static constexpr int kMinRDMAMessageSize{4096};

static folly::Singleton<NetworkPerfMonitor> networkPerfMonitorSingleton{};

std::shared_ptr<NetworkPerfMonitor> NetworkPerfMonitor::getInstance() {
  if (!NCCL_NETWORK_PERF_MONITOR_ENABLE) {
    return nullptr;
  }
  return networkPerfMonitorSingleton.try_get();
}

NetworkPerfMonitor::NetworkPerfMonitor(
    std::chrono::seconds bandwidthComputeIntervalTimeInSecs)
    : bandwidthComputeIntervalTimeInSecs_(bandwidthComputeIntervalTimeInSecs),
      lastComputeTs_(std::chrono::system_clock::now()),
      running_(true) {
  worker_ = std::thread([this] {
    const std::string commDesc = "Singleton";
    COMMS_NAMED_THREAD_START_EXT("NetworkPerfMonitor", -1, 0UL, commDesc);
    this->processEventsThreadFn();
  });

  commHashToCommInfo_[0] = CommInfo{
      .logMetaData = CommLogData{
          .commDesc = "AvgBw",
      }};
}

NetworkPerfMonitor::~NetworkPerfMonitor() {
  running_ = false;
  worker_.join();
}

bool NetworkPerfMonitor::checkIfRecordRDMAEvent(uint64_t messageSize) {
  // Ignore small control messages like barriers that get exchanged between
  // ranks.
  if (messageSize <= kMinRDMAMessageSize) {
    return false;
  }
  return true;
}

void NetworkPerfMonitor::recordRDMAEvent(RDMACompletionEvent event) {
  if (!running_) {
    return;
  }
  queue_.enqueue(std::move(event));
}

void NetworkPerfMonitor::processEventsThreadFn() {
  folly::Optional<RDMACompletionEvent> event;
  while (running_) {
    event = queue_.try_dequeue();
    if (!event.has_value()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }
    uint64_t postTs = std::chrono::duration_cast<std::chrono::microseconds>(
                          event->postTs.time_since_epoch())
                          .count();
    uint64_t completionTs =
        std::chrono::duration_cast<std::chrono::microseconds>(
            event->completionTs.time_since_epoch())
            .count();

    eventsAggregator_.eventsTsIntervals +=
        Interval::right_open(postTs, completionTs);
    eventsAggregator_.totalBytes += event->totalBytes;

    auto& commHashAggregator = commHashToEventsAggregator_[event->commHash];
    commHashAggregator.eventsTsIntervals +=
        Interval::right_open(postTs, completionTs);
    commHashAggregator.totalBytes += event->totalBytes;

    if ((std::chrono::system_clock::now() - lastComputeTs_) >=
        bandwidthComputeIntervalTimeInSecs_) {
      computeBandwidth();
      lastComputeTs_ = std::chrono::system_clock::now();
    }
  }
}

void NetworkPerfMonitor::computeBandwidth() {
  NetworkPerfStats stats;
  stats.avgBw = eventsAggregator_.totalBytes /
      (boost::icl::length(eventsAggregator_.eventsTsIntervals) * 1E3);
  eventsAggregator_.totalBytes = 0;
  eventsAggregator_.eventsTsIntervals.clear();

  for (auto& [commHash, commHashAggregator] : commHashToEventsAggregator_) {
    stats.commHashToAvgBw[commHash] = commHashAggregator.totalBytes /
        (boost::icl::length(commHashAggregator.eventsTsIntervals) * 1E3);
  }
  commHashToEventsAggregator_.clear();
  networkPerfStats_ = stats;
  reportPerfStatsToScuba(stats);
}

NetworkPerfStats NetworkPerfMonitor::reportPerfStats() {
  NetworkPerfStats stats = networkPerfStats_.copy();
  return stats;
}

void NetworkPerfMonitor::reportPerfStatsAsMap(
    std::unordered_map<std::string, std::string>& map) {
  const auto& stats = reportPerfStats();
  folly::dynamic obj = folly::dynamic::object();
  obj["avgBw"] = stats.avgBw;
  obj["commAvgBw"] = folly::dynamic::array();
  for (const auto& [commHash, avgBw] : stats.commHashToAvgBw) {
    folly::dynamic commInfo = folly::dynamic::object();
    commInfo["commHash"] = hashToHexStr(commHash);
    commInfo["avgBw"] = avgBw;
    if (commHashToCommInfo_.find(commHash) != commHashToCommInfo_.end()) {
      commInfo["commDesc"] =
          commHashToCommInfo_.at(commHash).logMetaData.commDesc;
    }
    obj["commAvgBw"].push_back(commInfo);
  }
  map["NetworkPerfMonitor"] = folly::toJson(obj);
}

void NetworkPerfMonitor::reportPerfStatsToScuba(const NetworkPerfStats& stats) {
  if (!NCCL_NETWORK_PERF_MONITOR_SCUBA_LOGGING_ENABLE) {
    return;
  }
  for (const auto& [commHash, avgBw] : stats.commHashToAvgBw) {
    if (commHashToCommInfo_.find(commHash) != commHashToCommInfo_.end()) {
      const auto& commInfo = commHashToCommInfo_.at(commHash);
      NcclScubaEvent scubaEvent(std::make_unique<NetworkPerfMonitorEvent>(
          commInfo.logMetaData, commInfo.cudaDev, commInfo.busId, avgBw));
      scubaEvent.record();
    }
  }
  if (stats.commHashToAvgBw.size() != 0) {
    const auto& commInfo = commHashToCommInfo_[0];
    NcclScubaEvent scubaEvent(std::make_unique<NetworkPerfMonitorEvent>(
        commInfo.logMetaData, commInfo.cudaDev, commInfo.busId, stats.avgBw));
    scubaEvent.record();
  }
}

} // namespace ncclx::colltrace
