// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "meta/comms-monitor/CommsMonitor.h"

#include <folly/Singleton.h>

#include "comms/utils/StrUtils.h"
#include "comms/utils/colltrace/NetworkPerfMonitor.h"
#include "comms/utils/colltrace/plugins/CommDumpPlugin.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/commDump.h"

constexpr static auto kGlobalInfoDumpMapKey = "GlobalInfo";

namespace ncclx::comms_monitor {

namespace {
struct CommsMonitorSingletonTag {};
} // namespace

folly::Singleton<CommsMonitor, CommsMonitorSingletonTag>
    commsMonitorSingleton{};

/*static*/ NcclCommMonitorInfo NcclCommMonitorInfo::fromNcclComm(
    ncclComm_t comm) {
  std::shared_ptr<ProxyTrace> proxyTrace;
  if (comm->proxyState != nullptr) {
    proxyTrace = comm->proxyState->trace;
  }
  return NcclCommMonitorInfo{
      .logMetaData = comm->logMetaData,
      .stateInfo =
          CommStateInfo{
              .localRank = comm->localRank,
              .node = comm->rankToNode ? comm->rankToNode[comm->rank] : 0,
              .nLocalRanks = comm->localRanks,
              .nNodes = comm->nNodes,
              .cliqueSize = comm->clique.size},
      .proxyTrace = proxyTrace,
      .newCollTrace = comm->newCollTrace,
      .memTracer = meta::comms::memtrace::MemoryTrace::getOrCreate(
          comm->logMetaData.commHash),
      .algoStats = comm->algoStats};
}

bool CommsMonitor::deregisterCommImpl(ncclComm_t comm) {
  auto lockedMap = commsMap_.wlock();
  if (!lockedMap->contains(comm)) {
    ERR("Deregistering comm %p that is not registered", comm);
    return false;
  }
  // Just mark the comm as dead. Since we have all the information, we can
  // still dump information for dead communicators.
  lockedMap->at(comm).status = NcclCommMonitorInfo::CommStatus::DEAD;
  return true;
}

bool CommsMonitor::registerCommImpl(ncclComm_t comm) {
  auto lockedMap = commsMap_.wlock();
  lockedMap->emplace(comm, NcclCommMonitorInfo::fromNcclComm(comm));
  return true;
}

CommDumpAllMap CommsMonitor::commDumpAllImpl(
    const std::unordered_map<std::string, std::string>& hints) {
  using meta::comms::colltrace::CommDumpPlugin;
  using meta::comms::ncclx::isKeyRequested;

  auto rfIt = hints.find("comm_dump::requestFields");
  auto requestFields = meta::comms::ncclx::parseRequestFields(
      rfIt != hints.end() ? std::optional<std::string>(rfIt->second)
                          : std::nullopt);

  bool flush = false;
  if (auto it = hints.find("comm_dump::flush"); it != hints.end()) {
    flush = it->second == "1";
  }

  std::vector<NcclCommMonitorInfo> commInfos;
  {
    auto lockedMap = commsMap_.rlock();

    for (const auto& [_, commMonitorInfo] : *lockedMap) {
      commInfos.push_back(commMonitorInfo);
    }
  }
  INFO(
      NCCL_ALL,
      "CommsMonitor: Dumping info for %lu communicators",
      commInfos.size());

  if (flush) {
    std::vector<std::pair<meta::comms::colltrace::ICollTrace*, uint64_t>>
        flushTokens;
    flushTokens.reserve(commInfos.size());
    for (const auto& info : commInfos) {
      if (info.newCollTrace != nullptr) {
        auto gen = info.newCollTrace->requestFlush();
        flushTokens.emplace_back(info.newCollTrace.get(), gen);
      }
    }
    for (auto& [ct, gen] : flushTokens) {
      ct->waitFlush(gen);
    }
  }

  CommDumpAllMap commDumpAllMap;

  bool onlyGlobalInfo = !requestFields.empty() &&
      std::all_of(
          requestFields.begin(),
          requestFields.end(),
          [](const std::string& key) {
            return key.starts_with("GlobalInfo::");
          });

  if (!onlyGlobalInfo) {
    for (const auto& commMonitorInfo : commInfos) {
      commDumpAllMap[hashToHexStr(commMonitorInfo.logMetaData.commHash)] =
          commDumpByMonitorInfo(commMonitorInfo, requestFields);
    }
  }

  std::unordered_map<std::string, std::string> globalInfoDumpMap;

  if (isKeyRequested(requestFields, "GlobalInfo::NetworkPerfInfo")) {
    auto networkPerfMonitorPtr =
        ncclx::colltrace::NetworkPerfMonitor::getInstance();
    if (networkPerfMonitorPtr != nullptr) {
      networkPerfMonitorPtr->reportPerfStatsAsMap(globalInfoDumpMap);
    }
  }

  if (isKeyRequested(requestFields, "GlobalInfo::totalCommDurPerIterationUs")) {
    int64_t totalCommTimeUs = 0;
    int64_t curCommIter = -1;
    for (const auto& commInfo : commInfos) {
      if (commInfo.newCollTrace == nullptr) {
        continue;
      }
      auto* plugin =
          dynamic_cast<CommDumpPlugin*>(commInfo.newCollTrace->getPluginByName(
              std::string{CommDumpPlugin::kCommDumpPluginName}));
      if (plugin == nullptr) {
        continue;
      }
      auto iterTime = plugin->getCurrentIterationCommTime();
      if (iterTime.iteration > curCommIter) {
        curCommIter = iterTime.iteration;
        totalCommTimeUs = iterTime.commTimeUs;
      } else if (iterTime.iteration == curCommIter) {
        totalCommTimeUs += iterTime.commTimeUs;
      }
    }
    globalInfoDumpMap["totalCommDurPerIterationUs"] =
        std::to_string(totalCommTimeUs);
  }

  if (!globalInfoDumpMap.empty()) {
    commDumpAllMap[kGlobalInfoDumpMapKey] = globalInfoDumpMap;
  }
  return commDumpAllMap;
}

/*static*/ bool CommsMonitor::deregisterComm(ncclComm_t comm) {
  if (!NCCL_COMMSMONITOR_ENABLE) {
    return false;
  }
  auto commMonitorPtr = getInstance();
  if (commMonitorPtr == nullptr) {
    ERR("Failed to get comms monitor instance to register comm %p", comm);
    return false;
  }
  return commMonitorPtr->deregisterCommImpl(comm);
}

/*static*/ bool CommsMonitor::registerComm(ncclComm_t comm) {
  if (!NCCL_COMMSMONITOR_ENABLE) {
    return false;
  }
  auto commMonitorPtr = getInstance();
  if (commMonitorPtr == nullptr) {
    ERR("Failed to get comms monitor instance to register comm %p", comm);
    return false;
  }
  return commMonitorPtr->registerCommImpl(comm);
}

/*static*/ int64_t CommsMonitor::getNumOfCommMonitoring() {
  auto commMonitorPtr = getInstance();
  if (commMonitorPtr == nullptr) {
    return -1;
  }
  auto lockedMap = commMonitorPtr->commsMap_.rlock();
  if (!lockedMap) {
    ERR("Getting commsMap_ lock to get number of monitored comms timed out");
    return -1;
  }
  return lockedMap->size();
}

/*static*/ std::shared_ptr<CommsMonitor> CommsMonitor::getInstance() {
  return commsMonitorSingleton.try_get();
}

/*static*/ void CommsMonitor::testOnlyClearComms() {
  auto commMonitorPtr = getInstance();
  if (commMonitorPtr != nullptr) {
    commMonitorPtr->commsMap_.wlock()->clear();
  }
}

/*static*/ std::optional<CommDumpAllMap> CommsMonitor::commDumpAll(
    const std::unordered_map<std::string, std::string>& hints) {
  if (!NCCL_COMMSMONITOR_ENABLE) {
    static bool logDumpAllUnavailable = false;
    if (!logDumpAllUnavailable) {
      WARN(
          "NCCL_COMMSMONITOR_ENABLE is not enabled. commDumpAll is unavailable");
      logDumpAllUnavailable = true;
    }
    return std::nullopt;
  }
  auto commMonitorPtr = getInstance();
  if (commMonitorPtr == nullptr) {
    return std::nullopt;
  }
  return commMonitorPtr->commDumpAllImpl(hints);
}

/*static*/ std::optional<NcclCommMonitorInfo>
CommsMonitor::getCommInfoByCommPtr(ncclComm_t comm) {
  if (!NCCL_COMMSMONITOR_ENABLE) {
    return std::nullopt;
  }
  auto commMonitorPtr = getInstance();
  if (commMonitorPtr == nullptr) {
    return std::nullopt;
  }
  auto commsMap = commMonitorPtr->commsMap_.rlock();
  if (auto commInfoPtr = folly::get_ptr(*commsMap, comm);
      commInfoPtr == nullptr) {
    return std::nullopt;
  } else {
    return *commInfoPtr;
  }
}

} // namespace ncclx::comms_monitor
