// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <fmt/core.h>
#include <sstream>
#include <string>
#include <unordered_map>

#include <folly/json/dynamic.h>
#include <folly/json/json.h>

#include "comm.h"
#include "meta/NcclxConfig.h" // @manual
#include "nccl.h"

#include "comms/utils/StrUtils.h"
#include "comms/utils/colltrace/CollTraceInterface.h"
#include "comms/utils/colltrace/plugins/CommDumpPlugin.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/memtrace/MemoryTrace.h"
#include "meta/commDump.h"
#include "meta/comms-monitor/CommsMonitor.h"

using meta::comms::colltrace::CommDumpPlugin;
using meta::comms::ncclx::DumpFieldSet;

namespace {

bool anyKeyRequested(
    const DumpFieldSet& fields,
    std::initializer_list<std::string_view> keys) {
  if (fields.empty()) {
    return true;
  }
  for (auto key : keys) {
    if (fields.contains(std::string{key})) {
      return true;
    }
  }
  return false;
}

} // namespace

namespace meta::comms::ncclx {

DumpFieldSet parseRequestFields(
    const std::optional<std::string>& requestFieldsStr) {
  if (!requestFieldsStr.has_value() || requestFieldsStr->empty()) {
    return {};
  }
  DumpFieldSet fields;
  std::istringstream ss(*requestFieldsStr);
  std::string token;
  while (std::getline(ss, token, ';')) {
    if (!token.empty()) {
      fields.insert(std::move(token));
    }
  }
  return fields;
}

bool isKeyRequested(const DumpFieldSet& fields, std::string_view key) {
  return fields.empty() || fields.contains(std::string{key});
}

std::unordered_map<std::string, std::string> dumpNewCollTrace(
    meta::comms::colltrace::ICollTrace& colltrace,
    const DumpFieldSet& requestFields) {
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
  return meta::comms::colltrace::commDumpToMap(dump.value(), requestFields);
}

bool waitForCollTraceDrain(
    meta::comms::colltrace::ICollTrace& colltrace,
    int timeoutMs) {
  constexpr int kPollIntervalMs = 50;
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
  while (std::chrono::steady_clock::now() < deadline) {
    std::unordered_map<std::string, std::string> dumpMap =
        meta::comms::ncclx::dumpNewCollTrace(colltrace);
    auto currentIt = dumpMap.find("CT_currentColls");
    auto pendingIt = dumpMap.find("CT_pendingColls");
    bool currentEmpty = currentIt != dumpMap.end() && currentIt->second == "[]";
    bool pendingEmpty = pendingIt != dumpMap.end() && pendingIt->second == "[]";
    if (currentEmpty && pendingEmpty) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(kPollIntervalMs));
  }
  return false;
}
} // namespace meta::comms::ncclx

using meta::comms::ncclx::dumpNewCollTrace;
using meta::comms::ncclx::isKeyRequested;

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
  map["cliqueSize"] = std::to_string(comm->clique.size);
  map["commDesc"] = toQuotedString(NCCLX_CONFIG_FIELD(comm->config, commDesc));
}

static void dumpCommInfo(
    const CommLogData* logMetaData,
    const ncclx::comms_monitor::CommStateInfo& stateInfo,
    std::unordered_map<std::string, std::string>& map,
    const DumpFieldSet& requestFields = {}) {
  if (logMetaData != nullptr) {
    if (isKeyRequested(requestFields, "commHash")) {
      map["commHash"] = toQuotedString(hashToHexStr(logMetaData->commHash));
    }
    if (isKeyRequested(requestFields, "rank")) {
      map["rank"] = std::to_string(logMetaData->rank);
    }
    if (isKeyRequested(requestFields, "commDesc")) {
      map["commDesc"] = toQuotedString(logMetaData->commDesc);
    }
    if (isKeyRequested(requestFields, "nRanks")) {
      map["nRanks"] = std::to_string(logMetaData->nRanks);
    }
  } else {
    XLOGF(DBG2, "CommDump: logMetaData is disabled. No trace to dump");
    return;
  }

  if (isKeyRequested(requestFields, "localRank")) {
    map["localRank"] = std::to_string(stateInfo.localRank);
  }
  if (isKeyRequested(requestFields, "node")) {
    map["node"] = std::to_string(stateInfo.node);
  }
  if (isKeyRequested(requestFields, "localRanks")) {
    map["localRanks"] = std::to_string(stateInfo.nLocalRanks);
  }
  if (isKeyRequested(requestFields, "nNodes")) {
    map["nNodes"] = std::to_string(stateInfo.nNodes);
  }
  if (isKeyRequested(requestFields, "cliqueSize")) {
    map["cliqueSize"] = std::to_string(stateInfo.cliqueSize);
  }
}

static void dumpMemoryTrace(
    const std::shared_ptr<meta::comms::memtrace::MemoryTrace>& memTracer,
    std::unordered_map<std::string, std::string>& map,
    const DumpFieldSet& requestFields = {}) {
  if (memTracer && isKeyRequested(requestFields, "memory")) {
    map["memory"] = memTracer->dump();
  }
}

static void dumpAlgoStatToMap(
    const std::shared_ptr<meta::comms::colltrace::AlgoStats>& algoStats,
    std::unordered_map<std::string, std::string>& map,
    const DumpFieldSet& requestFields = {}) {
  if (!isKeyRequested(requestFields, "algoStat") || !algoStats) {
    return;
  }
  auto dump = algoStats->dump();
  if (dump.entries.empty()) {
    return;
  }
  folly::dynamic obj = folly::dynamic::object();
  for (const auto& [opName, algoMap] : dump.entries) {
    folly::dynamic algoObj = folly::dynamic::object();
    for (const auto& [algoName, sizeMap] : algoMap) {
      folly::dynamic sizeObj = folly::dynamic::object();
      for (const auto& [sz, count] : sizeMap) {
        sizeObj[std::to_string(sz)] = count;
      }
      algoObj[algoName] = std::move(sizeObj);
    }
    obj[opName] = std::move(algoObj);
  }
  map["algoStat"] = folly::toJson(obj);
}

std::unordered_map<std::string, std::string> commDumpByMonitorInfo(
    const ncclx::comms_monitor::NcclCommMonitorInfo& info,
    const std::unordered_set<std::string>& requestFields) {
  std::unordered_map<std::string, std::string> map;

  if (anyKeyRequested(
          requestFields,
          {"commHash",
           "rank",
           "localRank",
           "node",
           "nRanks",
           "localRanks",
           "nNodes",
           "cliqueSize",
           "commDesc"})) {
    dumpCommInfo(&info.logMetaData, info.stateInfo, map, requestFields);
  }

  if (info.newCollTrace != nullptr &&
      anyKeyRequested(
          requestFields,
          {"CT_pastColls",
           "CT_currentColls",
           "CT_pendingColls",
           "CT_currentIteration",
           "CT_currentIterationCommTimeUs"})) {
    map.merge(dumpNewCollTrace(*info.newCollTrace, requestFields));
    XLOGF(DBG2, "commDumpByMonitorInfo: Dumped from colltrace");
  }

  dumpAlgoStatToMap(info.algoStats, map, requestFields);
  dumpMemoryTrace(info.memTracer, map, requestFields);

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
  }

  return ncclSuccess;
}

__attribute__((visibility("default"))) ncclResult_t ncclCommDumpAll(
    std::unordered_map<
        std::string,
        std::unordered_map<std::string, std::string>>& map,
    const std::unordered_map<std::string, std::string>& hints) {
  initEnv();
  auto commDumpsMaybe = ncclx::comms_monitor::CommsMonitor::commDumpAll(hints);
  if (!commDumpsMaybe.has_value()) {
    return ncclInternalError;
  }

  map.swap(commDumpsMaybe.value());
  return ncclSuccess;
}
