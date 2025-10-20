// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <boost/icl/discrete_interval.hpp>
#include <boost/icl/interval_map.hpp>
#include <folly/concurrency/UnboundedQueue.h>
#include <chrono>
#include <memory>

#include "comms/utils/commSpecs.h"

namespace ncclx::colltrace {

using namespace std::chrono_literals;

struct RDMACompletionEvent {
  std::chrono::time_point<std::chrono::high_resolution_clock> postTs;
  std::chrono::time_point<std::chrono::high_resolution_clock> completionTs;
  int remoteRank;
  uint64_t totalBytes;
  uint64_t messageSize;
  uint64_t commHash;
};

using Interval = boost::icl::discrete_interval<uint64_t>;
using IntervalSet = boost::icl::interval_set<uint64_t>;

struct EventsAggregator {
  IntervalSet eventsTsIntervals;
  uint64_t totalBytes;
};

struct NetworkPerfStats {
  // Unit: GBps
  std::unordered_map<uint64_t, double> commHashToAvgBw; // commHash to avgBw
  double avgBw; // Overall average bandwidth of current rank
};

struct CommInfo {
  CommLogData logMetaData;
  int cudaDev;
  int64_t busId;
};

class NetworkPerfMonitor {
 public:
  explicit NetworkPerfMonitor(
      std::chrono::seconds bandwidthComputeIntervalTimeInSecs = 1s);
  ~NetworkPerfMonitor();

  static std::shared_ptr<NetworkPerfMonitor> getInstance();

  void storeCommInfo(CommLogData logMetadata, int cudaDev, int busId) {
    CommInfo commInfo = {
        .logMetaData = logMetadata, .cudaDev = cudaDev, .busId = busId};
    commHashToCommInfo_[logMetadata.commHash] = commInfo;
  }

  // Check if we need to record RDMA event
  bool checkIfRecordRDMAEvent(uint64_t messageSize);
  // Input API
  void recordRDMAEvent(RDMACompletionEvent event);
  // Output API, called by commDump
  NetworkPerfStats reportPerfStats();

  void reportPerfStatsToScuba(const NetworkPerfStats& stats);

  void reportPerfStatsAsMap(std::unordered_map<std::string, std::string>& map);

 private:
  void processEventsThreadFn();
  void computeBandwidth();

  folly::UMPSCQueue<RDMACompletionEvent, false /*mayBlock*/> queue_;
  folly::Synchronized<NetworkPerfStats> networkPerfStats_;
  std::unordered_map<uint64_t, CommInfo>
      commHashToCommInfo_; // commHash to commInfo
  // to calculate bandwidth for the rank
  EventsAggregator eventsAggregator_;
  // per commHash bandwidth
  std::unordered_map<uint64_t, EventsAggregator> commHashToEventsAggregator_;

  std::chrono::seconds bandwidthComputeIntervalTimeInSecs_;
  std::chrono::time_point<std::chrono::high_resolution_clock> lastComputeTs_;
  std::atomic<bool> running_;
  std::thread worker_;
};

} // namespace ncclx::colltrace
