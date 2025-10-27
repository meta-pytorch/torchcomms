// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/tracing/MapperTrace.h"

#include <optional>
#include <string>
#include <variant>

#include <folly/json/json.h>

#include "comms/ctran/mapper/CtranMapper.h"
#include "comms/utils/StrUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/EventMgrHelperTypes.h"
#include "comms/utils/logger/LogUtils.h"

namespace ncclx::colltrace {

thread_local bool shouldMapperTraceCurrentThread = false;

constexpr int kDebugRepeatLogCount = 5;

void MapperTrace::CurCollInfo::clear() {
  currentColl.reset();
  recvNotifiedByPeer.clear();
  putFinishedByPeer.clear();
  unfinishedRequests.clear();
  readIndex = 0;
}

MapperTrace::MapperTrace()
    : eventHistory_(), maxEventCount_(NCCL_MAPPERTRACE_EVENT_RECORD_MAX) {
  eventHistory_.reserve(NCCL_MAPPERTRACE_EVENT_RECORD_MAX);
}

// TODO: Move it to TraceUtils.h. Currently we couldn't put it there due to
// circular dependency with EventMgr.h
std::unordered_map<std::string, std::string> retrieveMap(
    ScubaEntry& scubaEntry,
    bool quoted) {
  std::unordered_map<std::string, std::string> infoMap;

  const auto& normalMap = scubaEntry.getNormalMap();
  const auto& intMap = scubaEntry.getIntMap();

  for (const auto& [key, value] : normalMap) {
    infoMap[key] = quoted ? toQuotedString(value) : value;
  }
  for (const auto& [key, value] : intMap) {
    infoMap[key] = std::to_string(value);
  }
  return infoMap;
}

std::string CollStart::serialize(bool quoted) const {
  ScubaEntry entry;
  entry.addNormalValue("type", "CollStart");
  entry.addNormalValue(
      "coll", coll == nullptr ? "{}" : folly::toJson(coll->toDynamic()));
  return serializeMap(
      std::vector<std::string>{"type", "coll"},
      retrieveMap(entry, quoted),
      quoted);
}

std::string CollEnd::serialize(bool quoted) const {
  ScubaEntry entry;
  entry.addNormalValue("type", "CollEnd");
  return serializeMap(
      std::vector<std::string>{"type"}, retrieveMap(entry, quoted), quoted);
}

std::string CopyStart::serialize(bool quoted) const {
  ScubaEntry entry;
  entry.addNormalValue("type", "CopyStart");

  entry.addIntValue(
      "sourceBuffer", reinterpret_cast<ScubaEntry::int_type>(sourceBuffer));
  entry.addIntValue(
      "destBuffer", reinterpret_cast<ScubaEntry::int_type>(destBuffer));
  entry.addIntValue("length", length);
  entry.addIntValue("stream", reinterpret_cast<ScubaEntry::int_type>(stream));
  entry.addIntValue("reqAddr", reinterpret_cast<ScubaEntry::int_type>(req));
  return serializeMap(
      std::vector<std::string>{
          "type", "sourceBuffer", "destBuffer", "length", "stream", "reqAddr"},
      retrieveMap(entry, quoted),
      quoted);
}

std::string SendCtrlStart::serialize(bool quoted) const {
  ScubaEntry entry;
  entry.addNormalValue("type", "SendCtrlStart");
  entry.addIntValue("buffer", reinterpret_cast<ScubaEntry::int_type>(buffer));
  entry.addIntValue(
      "mapperHandle", reinterpret_cast<ScubaEntry::int_type>(mapperHandle));
  entry.addIntValue("peerRank", peerRank);
  entry.addIntValue("reqAddr", reinterpret_cast<ScubaEntry::int_type>(req));

  return serializeMap(
      std::vector<std::string>{
          "type", "buffer", "mapperHandle", "peerRank", "reqAddr"},
      retrieveMap(entry, quoted),
      quoted);
}

std::string RecvCtrlStart::serialize(bool quoted) const {
  ScubaEntry entry;
  entry.addNormalValue("type", "RecvCtrlStart");
  entry.addIntValue(
      "recvBufferPtr", reinterpret_cast<ScubaEntry::int_type>(*recvBufferPtr));
  entry.addIntValue(
      "accessKeyPtr", reinterpret_cast<ScubaEntry::int_type>(accessKeyPtr));
  entry.addIntValue("peerRank", peerRank);
  entry.addIntValue("reqAddr", reinterpret_cast<ScubaEntry::int_type>(req));

  return serializeMap(
      std::vector<std::string>{
          "type", "recvBufferPtr", "accessKeyPtr", "peerRank", "reqAddr"},
      retrieveMap(entry, quoted),
      quoted);
}

std::string SendSyncCtrlStart::serialize(bool quoted) const {
  ScubaEntry entry;
  entry.addNormalValue("type", "SendSyncCtrlStart");
  entry.addIntValue("peerRank", peerRank);
  entry.addIntValue("reqAddr", reinterpret_cast<ScubaEntry::int_type>(req));

  return serializeMap(
      std::vector<std::string>{"type", "peerRank", "reqAddr"},
      retrieveMap(entry, quoted),
      quoted);
}

std::string RecvSyncCtrlStart::serialize(bool quoted) const {
  ScubaEntry entry;
  entry.addNormalValue("type", "RecvSyncCtrlStart");
  entry.addIntValue("peerRank", peerRank);
  entry.addIntValue("reqAddr", reinterpret_cast<ScubaEntry::int_type>(req));

  return serializeMap(
      std::vector<std::string>{"type", "peerRank", "reqAddr"},
      retrieveMap(entry, quoted),
      quoted);
}

std::string PutStart::serialize(bool quoted) const {
  ScubaEntry entry;
  entry.addNormalValue("type", "RecvCtrlStart");
  entry.addIntValue(
      "remoteBuffer", reinterpret_cast<ScubaEntry::int_type>(remoteBuffer));
  entry.addIntValue("length", length);
  entry.addIntValue("peerRank", peerRank);
  entry.addIntValue(
      "sourceHandle", reinterpret_cast<ScubaEntry::int_type>(sourceHandle));
  entry.addIntValue(
      "accessKeyPtr", reinterpret_cast<ScubaEntry::int_type>(&remoteAccessKey));
  entry.addIntValue("reqAddr", reinterpret_cast<ScubaEntry::int_type>(req));

  return serializeMap(
      std::vector<std::string>{
          "type",
          "remoteBuffer",
          "length",
          "peerRank",
          "sourceHandle",
          "accessKeyPtr",
          "reqAddr"},
      retrieveMap(entry, quoted),
      quoted);
}

std::string MapperRequestEnd::serialize(bool quoted) const {
  ScubaEntry entry;
  entry.addNormalValue("type", "MapperRequestEnd");
  entry.addIntValue("reqAddr", reinterpret_cast<ScubaEntry::int_type>(req));
  return serializeMap(
      std::vector<std::string>{"type", "reqAddr"},
      retrieveMap(entry, quoted),
      quoted);
}

std::string RecvNotified::serialize(bool quoted) const {
  ScubaEntry entry;
  entry.addNormalValue("type", "RecvNotified");
  entry.addIntValue("peerRank", peerRank);
  return serializeMap(
      std::vector<std::string>{"type", "peerRank"},
      retrieveMap(entry, quoted),
      quoted);
}

std::string MapperRequestEventInfo::serialize(bool quoted) const {
  ScubaEntry entry;
  entry.addIntValue("seqNum", seqNum);
  auto infoMap = retrieveMap(entry, quoted);
  // event itself is a json, it doesn't need to be quoted
  infoMap["event"] = std::visit(
      [quoted](auto&& event) { return event.serialize(quoted); }, event);
  return serializeMap(
      std::vector<std::string>{"event", "seqNum", "timestampMs"},
      infoMap,
      quoted);
}

MapperTrace::Dump MapperTrace::dump() {
  return curCollInfoLocked_.withWLock([this](CurCollInfo& curCollInfo) {
    auto eventHistorySize =
        eventHistorySizeAtomic_.load(std::memory_order_relaxed);

    while (curCollInfo.readIndex < eventHistorySize) {
      std::visit(
          [this, &curCollInfo](auto&& event) {
            this->recordMapperEventImpl(event, curCollInfo);
          },
          eventHistory_.at(curCollInfo.readIndex));
      curCollInfo.readIndex += 1;
    }
    MapperTrace::Dump dump = {
        .currentColl = curCollInfo.currentColl,
        .recvNotifiedByPeer = curCollInfo.recvNotifiedByPeer,
        .putFinishedByPeer = curCollInfo.putFinishedByPeer,
        .unfinishedRequests = {},
    };
    for (auto& [_, eventIndex] : curCollInfo.unfinishedRequests) {
      dump.unfinishedRequests.emplace_back(
          MapperRequestEventInfo{
              .event = eventHistory_.at(eventIndex),
              .seqNum = eventIndex,
          });
    }
    return dump;
  });
}

void MapperTrace::recordMapperEventImpl(
    CollStart collStart,
    CurCollInfo& curCollInfo) {
  if (collStart.coll == nullptr) {
    return;
  }
  CLOGF_SUBSYS(
      INFO,
      COLL,
      "MapperTrace: Received CollStart for collId {}",
      collStart.coll->getCollId());
  if (curCollInfo.currentColl != nullptr) {
    CLOGF_FIRST_N(
        WARN,
        kDebugRepeatLogCount,
        "MapperTrace: Received the start event of collective w/ opCount {} when collective w/ opCount {} is still active",
        collStart.coll->getCollId(),
        curCollInfo.currentColl->getCollId());
    return;
  }
  curCollInfo.currentColl = collStart.coll;
}

void MapperTrace::recordMapperEventImpl(
    CollEnd collEnd,
    CurCollInfo& curCollInfo) {
  // Reset curCollInfo
  curCollInfo.clear();
  eventHistorySizeAtomic_.store(0, std::memory_order_relaxed);
}

void MapperTrace::recordMapperEventImpl(
    RecvNotified recvNotified,
    CurCollInfo& curCollInfo) {
  curCollInfo.recvNotifiedByPeer[recvNotified.peerRank] += 1;
}

void MapperTrace::recordMapperEventImpl(
    MapperRequestEnd mapperRequestEnd,
    CurCollInfo& curCollInfo) {
  if (curCollInfo.unfinishedRequests.count(mapperRequestEnd.req) == 0) {
    CLOGF_FIRST_N(
        WARN,
        kDebugRepeatLogCount,
        "MapperTrace: Received MapperRequestEnd %p but it was not found in unfinishedRequests",
        reinterpret_cast<uintptr_t>(mapperRequestEnd.req));
    return;
  }
  const auto& eventIndex =
      curCollInfo.unfinishedRequests.at(mapperRequestEnd.req);
  if (eventHistorySizeAtomic_.load(std::memory_order_relaxed) <= eventIndex) {
    CLOGF_FIRST_N(
        WARN,
        kDebugRepeatLogCount,
        "MapperTrace Internal Error: Received MapperRequestEnd %p but eventIndex %lu is out of range of eventHistory",
        reinterpret_cast<uintptr_t>(mapperRequestEnd.req),
        eventIndex);
    return;
  }
  const auto& reqEvent = eventHistory_.at(eventIndex);
  if (std::holds_alternative<PutStart>(reqEvent)) {
    auto peerRank = std::get<PutStart>(reqEvent).peerRank;
    curCollInfo.putFinishedByPeer[peerRank] += 1;
  }
  curCollInfo.unfinishedRequests.erase(mapperRequestEnd.req);
}

void MapperTrace::registerBeforeCollEndCallback(
    std::function<void()> callback) {
  callback_ = std::move(callback);
}

MapperTrace* getMapperTrace(CtranComm* comm) {
  if (comm == nullptr || comm->ctran_ == nullptr ||
      comm->ctran_->mapper == nullptr) {
    return nullptr;
  }
  return comm->ctran_->mapper->mapperTrace.get();
}
} // namespace ncclx::colltrace
