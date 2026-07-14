// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <memory>
#include <unordered_set>

#include "comm.h"
#include "device.h"

#include "comms/ctran/colltrace/MapperTrace.h"
#include "comms/utils/colltrace/CollTraceInterface.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/memtrace/MemoryTrace.h"
#include "meta/colltrace/ProxyTrace.h"

namespace ncclx::comms_monitor {
struct CommStateInfo {
  int localRank{0};
  int node{0};
  int nLocalRanks{1};
  int nNodes{1};
  int cliqueSize{0};
};

struct NcclCommMonitorInfo {
  CommLogData logMetaData;
  CommStateInfo stateInfo;
  std::shared_ptr<colltrace::MapperTrace> mapperTrace;
  std::shared_ptr<ProxyTrace> proxyTrace;
  // ptr for the new colltrace interface.
  std::shared_ptr<meta::comms::colltrace::ICollTrace> newCollTrace;

  enum class CommStatus {
    ALIVE,
    DEAD,
  } status = CommStatus::ALIVE;

  // Adopted from MemoryTrace::getOrCreate() at registerComm time.
  // See MemoryTrace::getOrCreate() for dual ownership rationale.
  std::shared_ptr<meta::comms::memtrace::MemoryTrace> memTracer;

  // Shared ownership of AlgoStats so it survives comm destruction.
  // Baseline and ctran share the same instance via AlgoStats::getOrCreate.
  std::shared_ptr<meta::comms::colltrace::AlgoStats> algoStats;

  static NcclCommMonitorInfo fromNcclComm(ncclComm_t comm);
};

// {CommHash: {key: value}}
using CommDumpAllMap = std::
    unordered_map<std::string, std::unordered_map<std::string, std::string>>;

class CommsMonitor {
  // Should only be used in the CommsMonitor UT, need a friend class to
  // specifically test the case of holding lock too long.
  friend class CommsMonitorTest;

 public:
  static bool registerComm(ncclComm_t comm);
  static bool deregisterComm(ncclComm_t comm);
  static std::optional<CommDumpAllMap> commDumpAll(
      const std::unordered_map<std::string, std::string>& hints = {});

  static std::optional<NcclCommMonitorInfo> getCommInfoByCommPtr(
      ncclComm_t comm);

  // Get the total number of communicators CommsMonitor is currently monitoring
  // If any failure happened during calling this function, it will return -1.
  static int64_t getNumOfCommMonitoring();

  // For testing only. Clears all registered communicators from the singleton.
  static void testOnlyClearComms();

 private:
  bool registerCommImpl(ncclComm_t comm);
  bool deregisterCommImpl(ncclComm_t comm);
  CommDumpAllMap commDumpAllImpl(
      const std::unordered_map<std::string, std::string>& hints);

  static std::shared_ptr<CommsMonitor> getInstance();

  folly::Synchronized<std::unordered_map<ncclComm_t, NcclCommMonitorInfo>>
      commsMap_;
};

} // namespace ncclx::comms_monitor

std::unordered_map<std::string, std::string> commDumpByMonitorInfo(
    const ncclx::comms_monitor::NcclCommMonitorInfo& info,
    const std::unordered_set<std::string>& requestFields =
        {}); // resides in commDump.cc
