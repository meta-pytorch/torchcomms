// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <deque>
#include <memory>
#include <string>
#include <vector>

#include <folly/DynamicConverter.h>
#include <folly/dynamic.h>

namespace meta::colltrace {

struct CollStats {
  const int collId;
  const int percent;
  const float minLatencyUs;
  const float maxLatencyUs;
  const std::string opName;
  const std::string dataType;
  const int64_t count;
};

struct CollTraceInfo {
  int64_t collId;
  std::string opName;
  uint64_t opCount; // opCount received from ncclComm
  std::string dataType;
  int64_t count;
  float latencyMs{-1};

  std::string algoName;
  std::optional<const void*> sendbuff;
  std::optional<const void*> recvbuff;
  std::optional<std::vector<int>> ranksInGroupedP2P;

  // This is achieved by waiting for the start event. We can only guarantee
  // before this time point kernel has already started, but we cannot guarantee
  // kernel started exactly at this time point.
  std::chrono::time_point<std::chrono::high_resolution_clock> startTs{};
  std::chrono::time_point<std::chrono::high_resolution_clock> enqueueTs{};
  std::chrono::microseconds interCollTime;

  // Conversion to folly::dynamic for JSON serialization
  folly::dynamic toDynamic() const {
    folly::dynamic obj = folly::dynamic::object;
    obj["collId"] = collId;
    obj["opName"] = opName;
    obj["opCount"] = opCount;
    obj["dataType"] = dataType;
    obj["count"] = count;
    obj["latencyMs"] = latencyMs;
    obj["algoName"] = algoName;

    // Handle optional fields
    if (sendbuff.has_value()) {
      obj["sendbuff"] = reinterpret_cast<int64_t>(*sendbuff);
    }
    if (recvbuff.has_value()) {
      obj["recvbuff"] = reinterpret_cast<int64_t>(*recvbuff);
    }
    if (ranksInGroupedP2P.has_value()) {
      obj["ranksInGroupedP2P"] = folly::toDynamic(this->ranksInGroupedP2P);
    }

    obj["startTs"] = std::chrono::duration_cast<std::chrono::microseconds>(
                         startTs.time_since_epoch())
                         .count();
    obj["enqueueTs"] = std::chrono::duration_cast<std::chrono::microseconds>(
                           enqueueTs.time_since_epoch())
                           .count();
    obj["interCollTimeUs"] = interCollTime.count();

    return obj;
  }
};

// Free function for folly::toDynamic conversion
inline folly::dynamic toDynamic(const CollTraceInfo& info) {
  return info.toDynamic();
}

#ifdef BUILD_META_INTERNAL
void reportToScuba(
    const std::deque<std::vector<CollStats>>& stats,
    const std::string& scubaTable,
    const std::string& commHash);
#endif

std::vector<CollStats> aggregateResults(
    const std::deque<std::unique_ptr<CollTraceInfo>>& info,
    const std::vector<float>& latencyAllGather,
    int RANKS_PER_HOST,
    int NCCL_COLLTRACE_RECORD_MAX);

float getSizeMb(const std::string& dataType, int count);

} // namespace meta::colltrace
