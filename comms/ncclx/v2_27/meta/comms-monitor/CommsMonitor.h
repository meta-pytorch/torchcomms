// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <memory>

#include "comm.h"

#include "comms/ctran/tracing/MapperTrace.h"
#include "comms/utils/colltrace/CollTraceInterface.h"
#include "comms/utils/commSpecs.h"
#include "meta/colltrace/CollTrace.h"
#include "meta/colltrace/ProxyTrace.h"

namespace ncclx::comms_monitor {

struct NcclCommMonitorInfo {
  CommLogData logMetaData;
  ncclx::CommStateX commState;
  // This one will be deprecated soon.
  std::shared_ptr<CollTrace> collTrace;
  std::shared_ptr<colltrace::MapperTrace> mapperTrace;
  std::shared_ptr<ProxyTrace> proxyTrace;
  // ptr for the new colltrace interface.
  std::shared_ptr<meta::comms::colltrace::ICollTrace> newCollTrace;

  enum class CommStatus {
    ALIVE,
    DEAD,
  } status = CommStatus::ALIVE;

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
  static std::optional<CommDumpAllMap> commDumpAll();

  static std::optional<NcclCommMonitorInfo> getCommInfoByCommPtr(
      ncclComm_t comm);

  // Get the total number of communicators CommsMonitor is currently monitoring
  // If any failure happened during calling this function, it will return -1.
  static int64_t getNumOfCommMonitoring();

 private:
  bool registerCommImpl(ncclComm_t comm);
  bool deregisterCommImpl(ncclComm_t comm);
  CommDumpAllMap commDumpAllImpl();

  static std::shared_ptr<CommsMonitor> getInstance();

  folly::Synchronized<std::unordered_map<ncclComm_t, NcclCommMonitorInfo>>
      commsMap_;
};

} // namespace ncclx::comms_monitor

std::unordered_map<std::string, std::string> commDumpByMonitorInfo(
    const ncclx::comms_monitor::NcclCommMonitorInfo&
        info); // resides in commDump.cc
