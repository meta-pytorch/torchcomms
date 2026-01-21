// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "meta/comms-monitor/CommsMonitor.h"

#include <folly/Singleton.h>

#include "comms/ctran/Ctran.h" // access to incomplete type

#include "comms/utils/colltrace/NetworkPerfMonitor.h"
#include "comms/utils/cvars/nccl_cvars.h"

constexpr static auto kGlobalInfoDumpMapKey = "GlobalInfo";

namespace ncclx::comms_monitor {

namespace {
struct CommsMonitorSingletonTag {};

std::vector<int> getVectorRingFromArray(int nRanks, int* ring) {
  std::vector<int> ringVec;
  ringVec.reserve(nRanks);
  for (int i = 0; i < nRanks; i++) {
    ringVec.push_back(ring[i]);
  }
  return ringVec;
}

std::array<int, NCCL_MAX_TREE_ARITY> getTreeChildArray(int* treeDown) {
  std::array<int, NCCL_MAX_TREE_ARITY> childNodes{};
  for (int i = 0; i < NCCL_MAX_TREE_ARITY; i++) {
    childNodes[i] = treeDown[i];
  }
  return childNodes;
}
} // namespace

folly::Singleton<CommsMonitor, CommsMonitorSingletonTag>
    commsMonitorSingleton{};

/*static*/ NcclCommMonitorInfo NcclCommMonitorInfo::fromNcclComm(
    ncclComm_t comm) {
  std::shared_ptr<colltrace::MapperTrace> mapperTrace;
  if (comm->ctranComm_ && comm->ctranComm_->ctran_ &&
      comm->ctranComm_->ctran_->isInitialized()) {
    mapperTrace = comm->ctranComm_->ctran_->mapper->mapperTrace;
  }

  std::shared_ptr<ProxyTrace> proxyTrace;
  if (comm->proxyState != nullptr) {
    proxyTrace = comm->proxyState->trace;
  }
  return NcclCommMonitorInfo{
      .logMetaData = comm->logMetaData,
      .commState = ncclx::CommStateX{*comm->ctranComm_->statex_},
      .topoInfo = NcclTopoInfo::fromNcclComm(comm),
      .collTrace = comm->collTrace,
      .mapperTrace = mapperTrace,
      .proxyTrace = proxyTrace,
      .newCollTrace = comm->newCollTrace};
}

NcclTopoInfo NcclTopoInfo::fromNcclComm(ncclComm_t comm) {
  NcclTopoInfo topoInfo;
  // TODO: currently we only support
  topoInfo.nChannels = comm->nChannels;
  topoInfo.rings.reserve(topoInfo.nChannels);
  topoInfo.trees.reserve(topoInfo.nChannels);
  for (int i = 0; i < topoInfo.nChannels; i++) {
    auto& channel = comm->channels[i];
    if (comm->lazySetupChannels && comm->nChannels != comm->nChannelsReady) {
      topoInfo.rings.emplace_back(getVectorRingFromArray(
          comm->nRanks, comm->rings.value().data() + i * comm->nRanks));
    } else {
      topoInfo.rings.emplace_back(
          getVectorRingFromArray(comm->nRanks, channel.ring.userRanks));
    }
    topoInfo.trees.emplace_back(
        NcclTreeNodeInfo{
            .parentNode = channel.tree.up,
            .childrenNodes = getTreeChildArray(channel.tree.down)});
  }
  return topoInfo;
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

CommDumpAllMap CommsMonitor::commDumpAllImpl() {
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

  CommDumpAllMap commDumpAllMap;
  for (const auto& commMonitorInfo : commInfos) {
    commDumpAllMap[hashToHexStr(commMonitorInfo.logMetaData.commHash)] =
        commDumpByMonitorInfo(commMonitorInfo);
  }

  std::unordered_map<std::string, std::string> globalInfoDumpMap;
  auto networkPerfMonitorPtr =
      ncclx::colltrace::NetworkPerfMonitor::getInstance();
  if (networkPerfMonitorPtr != nullptr) {
    networkPerfMonitorPtr->reportPerfStatsAsMap(globalInfoDumpMap);
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

/*static*/ std::optional<CommDumpAllMap> CommsMonitor::commDumpAll() {
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
  return commMonitorPtr->commDumpAllImpl();
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
