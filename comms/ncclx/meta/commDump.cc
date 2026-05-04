// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <fmt/core.h>
#include <string>
#include <unordered_map>

#include <folly/json/dynamic.h>
#include <folly/json/json.h>

#include "comm.h"
#include "meta/NcclxConfig.h" // @manual
#include "nccl.h"

#include "comms/ctran/colltrace/MapperTrace.h"
#include "comms/utils/StrUtils.h"
#include "comms/utils/colltrace/CollTraceInterface.h"
#include "comms/utils/colltrace/plugins/CommDumpPlugin.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/ProcessGlobalErrorsUtil.h"
#include "meta/colltrace/ProxyTrace.h"
#include "meta/commDump.h"
#include "meta/comms-monitor/CommsMonitor.h"

using meta::comms::colltrace::CommDumpPlugin;

namespace meta::comms::ncclx {
std::unordered_map<std::string, std::string> dumpNewCollTrace(
    meta::comms::colltrace::ICollTrace& colltrace) {
  auto commDumpPluginMaybe = colltrace.getPluginByName(
      std::string{CommDumpPlugin::kCommDumpPluginName});
  auto commDumpPluginPtr = dynamic_cast<CommDumpPlugin*>(commDumpPluginMaybe);
  if (commDumpPluginPtr == nullptr) {
    return {};
  }
  auto dump = commDumpPluginPtr->dump();
  if (dump.hasError()) {
    return {};
  }

  return meta::comms::colltrace::commDumpToMap(dump.value());
}
} // namespace meta::comms::ncclx

using meta::comms::ncclx::dumpNewCollTrace;

namespace {
// NOTE: Keep in sync with scripts/lobanova/nccl/nccl_dump.thrift
void dumpProcessGlobalErrors(
    std::unordered_map<std::string, std::string>& map) {
  if (!NCCL_COMM_DUMP_ENABLE_PROCESS_GLOBAL_ERRORS) {
    return;
  }

  auto state = ProcessGlobalErrorsUtil::getAllState();

  folly::dynamic obj = folly::dynamic::object();
  obj["badNics"] = folly::dynamic::object();
  for (const auto& [device, portMap] : state.badNics) {
    obj["badNics"][device] = folly::dynamic::object();
    for (const auto& [port, nicError] : portMap) {
      auto portStr = fmt::format("{}", port);
      obj["badNics"][device][portStr] = folly::dynamic::object();
      obj["badNics"][device][portStr]["timestampMs"] =
          nicError.timestampMs.count();
      obj["badNics"][device][portStr]["errorMessage"] = nicError.errorMessage;
    }
  }

  obj["errorAndStackTraces"] = folly::dynamic::array();
  for (const auto& errorAndStackTrace : state.errorAndStackTraces) {
    folly::dynamic errorAndStackTraceObj = folly::dynamic::object();
    errorAndStackTraceObj["timestampMs"] =
        errorAndStackTrace.timestampMs.count();
    errorAndStackTraceObj["errorMessage"] = errorAndStackTrace.errorMessage;
    errorAndStackTraceObj["stackTrace"] = folly::dynamic::array(
        errorAndStackTrace.stackTrace.begin(),
        errorAndStackTrace.stackTrace.end());

    obj["errorAndStackTraces"].push_back(std::move(errorAndStackTraceObj));
  }

  map["processGlobalErrors"] = folly::toJson(obj);
}
} // namespace

static void dumpCommInfo(
    const ncclComm_t comm,
    std::unordered_map<std::string, std::string>& map) {
  map["commHash"] = toQuotedString(hashToHexStr(comm->commHash));
  map["rank"] = std::to_string(comm->rank);
  map["localRank"] = std::to_string(comm->localRank);
  map["node"] = std::to_string(comm->node);

  map["nRanks"] = std::to_string(comm->nRanks);
  map["localRanks"] = std::to_string(comm->localRanks);
  map["nNodes"] = std::to_string(comm->nNodes);
  map["commDesc"] = NCCLX_CONFIG_FIELD(comm->config, commDesc);
}

static void dumpCommInfo(
    const CommLogData* logMetaData,
    const ncclx::comms_monitor::CommStateInfo& stateInfo,
    std::unordered_map<std::string, std::string>& map) {
  if (logMetaData != nullptr) {
    map["commHash"] = toQuotedString(hashToHexStr(logMetaData->commHash));
    map["rank"] = std::to_string(logMetaData->rank);
    map["commDesc"] = toQuotedString(logMetaData->commDesc);
    map["nRanks"] = std::to_string(logMetaData->nRanks);
  } else {
    XLOGF(DBG2, "CommDump: logMetaData is disabled. No trace to dump");
    return;
  }

  map["localRank"] = std::to_string(stateInfo.localRank);
  map["node"] = std::to_string(stateInfo.node);
  map["localRanks"] = std::to_string(stateInfo.nLocalRanks);
  map["nNodes"] = std::to_string(stateInfo.nNodes);
}

static void dumpMapperTrace(
    ncclx::colltrace::MapperTrace& mapperTrace,
    std::unordered_map<std::string, std::string>& map) {
  auto dump = mapperTrace.dump();

  XLOGF(
      DBG2,
      "CommDump: MAPPERTRACE dump: {} unfinished req, {} current collective records",
      dump.unfinishedRequests.size(),
      dump.currentColl != nullptr ? 1 : 0);

  if (dump.currentColl != nullptr) {
    map["MT_currentColl"] = folly::toJson(dump.currentColl->toDynamic());
  } else {
    map["MT_currentColl"] = "null";
  }

  map["MT_unfinishedRequests"] = serializeObjects(dump.unfinishedRequests);
  map["MT_recvNotifiedByPeer"] = mapToJson(dump.recvNotifiedByPeer);
  map["MT_putFinishedByPeer"] = mapToJson(dump.putFinishedByPeer);
}

static void dumpProxyTrace(
    const ProxyTrace* ProxyTrace,
    uint64_t commHash,
    std::unordered_map<std::string, std::string>& map) {
  if (ProxyTrace) {
    auto dump = ProxyTrace->dump(commHash);

    XLOGF(
        DBG2,
        "CommDump: PROXYTRACE dump: {} past collectives, {} active network operations",
        dump.pastColls.size(),
        dump.activeOps.size());

    map["PT_pastColls"] = serializeObjects(dump.pastColls);
    map["PT_activeOps"] = serializeObjects(dump.activeOps);
    map["PT_activeColls"] = serializeObjects(dump.activeColls);
  } else {
    XLOGF(DBG2, "CommDump: PROXYTRACE is disabled. No trace to dump");
  }
}

std::unordered_map<std::string, std::string> commDumpByMonitorInfo(
    const ncclx::comms_monitor::NcclCommMonitorInfo& info) {
  std::unordered_map<std::string, std::string> map;
  dumpCommInfo(&info.logMetaData, info.stateInfo, map);
  if (info.newCollTrace != nullptr) {
    map.merge(dumpNewCollTrace(*info.newCollTrace));
    XLOGF(DBG2, "commDumpByMonitorInfo: Dumped from colltrace");
  }
  dumpProxyTrace(info.proxyTrace.get(), info.logMetaData.commHash, map);

  if (info.mapperTrace != nullptr) {
    dumpMapperTrace(*info.mapperTrace, map);
  } else {
    XLOGF(DBG2, "CommDump: MAPPERTRACE is disabled. No trace to dump");
  }
  dumpProcessGlobalErrors(map);
  return map;
}

__attribute__((visibility("default"))) ncclResult_t ncclCommDump(
    const ncclComm_t comm,
    std::unordered_map<std::string, std::string>& map) {
  initEnv();
  if (NCCL_COMMSMONITOR_ENABLE) {
    auto commInfoMaybe =
        ncclx::comms_monitor::CommsMonitor::getCommInfoByCommPtr(comm);
    if (!commInfoMaybe.has_value()) {
      return ncclSuccess;
    }
    map = commDumpByMonitorInfo(commInfoMaybe.value());
    return ncclSuccess;
  }

  if (comm != nullptr) {
    XLOGF(
        DBG2,
        "ncclCommDump by comm: rank {} comm {} commHash {} commDesc {}",
        comm->rank,
        fmt::ptr(comm),
        comm->commHash,
        NCCLX_CONFIG_FIELD(comm->config, commDesc));

    dumpCommInfo(comm, map);
    if (comm->newCollTrace != nullptr) {
      map.merge(dumpNewCollTrace(*comm->newCollTrace));
      XLOGF(DBG2, "CommDump: Dumped from colltrace");
    }
    if (comm->proxyState != nullptr) {
      dumpProxyTrace(comm->proxyState->trace.get(), comm->commHash, map);
    }

    auto mapperTrace = ncclx::colltrace::getMapperTrace(comm->ctranComm_.get());
    if (mapperTrace != nullptr) {
      dumpMapperTrace(*mapperTrace, map);
    } else {
      XLOGF(DBG2, "CommDump: MAPPERTRACE is disabled. No trace to dump");
    }
  }
  dumpProcessGlobalErrors(map);

  return ncclSuccess;
}

__attribute__((visibility("default"))) ncclResult_t ncclCommDumpAll(
    std::unordered_map<
        std::string,
        std::unordered_map<std::string, std::string>>& map) {
  initEnv();
  auto commDumpsMaybe = ncclx::comms_monitor::CommsMonitor::commDumpAll();
  if (!commDumpsMaybe.has_value()) {
    return ncclInternalError;
  }

  map.swap(commDumpsMaybe.value());
  return ncclSuccess;
}
