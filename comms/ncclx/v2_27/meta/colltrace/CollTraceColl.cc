// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "CollTraceColl.h"

#include "comms/utils/StrUtils.h"
#include "comms/utils/logger/EventMgr.h"
#include "meta/colltrace/CollTraceUtils.h"
#include "meta/logger/DebugExt.h"
#include "meta/wrapper/DataTypeStrUtils.h"

using namespace ncclx::colltrace;

ScubaEntry CollTraceColl::toScubaEntry() const {
  ScubaEntry entry;

  try {
    entry.addIntValue("collId", collId);
    entry.addIntValue("rank", logMetaData.rank);
    entry.addIntValue("commHash", logMetaData.commHash);
    entry.addIntValue("opCount", opCount);
    entry.addIntValue("stream", reinterpret_cast<ScubaEntry::int_type>(stream));
    entry.addIntValue("iteration", iteration);
    if (sendbuff.has_value()) {
      entry.addIntValue(
          "sendbuff", reinterpret_cast<ScubaEntry::int_type>(*sendbuff));
    }
    if (recvbuff.has_value()) {
      entry.addIntValue(
          "recvbuff", reinterpret_cast<ScubaEntry::int_type>(*recvbuff));
    }
    if (count.has_value()) {
      entry.addIntValue("count", *count);
    }
    if (ctranAttr.has_value()) {
      if (ctranAttr->checksum.has_value()) {
        entry.addIntValue("checksum", ctranAttr->checksum.value());
      }
    }
    entry.addIntValue("latencyUs", 1000 * latency);
    entry.addIntValue(
        "startTs",
        std::chrono::duration_cast<std::chrono::microseconds>(
            startTs.time_since_epoch())
            .count());
    entry.addIntValue(
        "enqueueTs",
        std::chrono::duration_cast<std::chrono::microseconds>(
            enqueueTs.time_since_epoch())
            .count());
    auto latencyMs = std::chrono::duration<double, std::milli>(latency);
    entry.addIntValue(
        "ExecutionTimeUs",
        std::chrono::duration_cast<std::chrono::microseconds>(latencyMs)
            .count());
    entry.addIntValue(
        "QueueingTimeUs",
        std::chrono::duration_cast<std::chrono::microseconds>(
            startTs - enqueueTs)
            .count());
    entry.addIntValue("InterCollTimeUs", interCollTime.count());

    std::string codepathStr = "undefined";
    if (codepath == Codepath::BASELINE) {
      codepathStr = "baseline";
    } else if (codepath == Codepath::CTRAN) {
      codepathStr = "ctran";
    } else if (codepath == Codepath::CTRAN_CPU) {
      codepathStr = "ctran_cpu";
    }
    entry.addNormalValue("codepath", codepathStr);

    entry.addNormalValue("opName", opName);
    entry.addNormalValue("dataType", getDatatypeStr(dataType));
    entry.addIntValue("nThreads", nThreads);
    entry.addNormalValue("algoName", algoName);

    if (baselineAttr.has_value()) {
      entry.addIntValue("root", baselineAttr->root);
      entry.addIntValue("nChannels", baselineAttr->nChannels);

      std::string algoStr = (0 <= baselineAttr->algorithm &&
                             baselineAttr->algorithm < NCCL_NUM_ALGORITHMS)
          ? ncclAlgoStr[baselineAttr->algorithm]
          : "N/A";
      std::string protoStr = (0 <= baselineAttr->protocol &&
                              baselineAttr->protocol < NCCL_NUM_PROTOCOLS)
          ? ncclProtoStr[baselineAttr->protocol]
          : "N/A";
      entry.addNormalValue("redOp", getRedOpStr(baselineAttr->op));
      entry.addNormalValue("algorithm", algoStr);
      entry.addNormalValue("protocol", protoStr);
      entry.addNormalValue("pattern", getNcclPatternStr(baselineAttr->pattern));
    }

    if (ranksInGroupedP2P.has_value()) {
      entry.addNormalValue(
          "ranksInGroupedP2P", serializeVec(ranksInGroupedP2P.value()));
    }
  } catch (const std::exception& e) {
    WARN_FIRST_N(
        kDebugRepeatLogCount,
        "CollEvent to Scuba Entry failed: %s\n",
        e.what());
  }
  return entry;
}

CollSignature CollTraceColl::toCollSignature() const {
  uint64_t pgid = LOGGER_PG_ID_DEFAULT;
  std::string commDesc(comm->config_.commDesc);
  if (commDesc != "undefined") {
    size_t pos = commDesc.find(':');
    if (pos != std::string::npos) {
      try {
        pgid = std::stoi(commDesc.substr(pos + 1));
      } catch (const std::exception&) {
        WARN_FIRST_N(
            kDebugRepeatLogCount,
            "CollTrace: Invalid commDesc: %s",
            comm->config_.commDesc);
      }
    }
  }

  int rank = comm->statex_->rank();
  uint64_t commHash = comm->statex_->commHash();

  int dataCount = -1;
  if (count.has_value()) {
    dataCount = *count;
  }

  if (baselineAttr.has_value()) {
    std::string algoStr = (0 <= baselineAttr->algorithm &&
                           baselineAttr->algorithm < NCCL_NUM_ALGORITHMS)
        ? ncclAlgoStr[baselineAttr->algorithm]
        : "N/A";
    std::string protoStr = (0 <= baselineAttr->protocol &&
                            baselineAttr->protocol < NCCL_NUM_PROTOCOLS)
        ? ncclProtoStr[baselineAttr->protocol]
        : "N/A";

    return CollSignature(
        commHash,
        pgid,
        rank,
        opName,
        nThreads,
        getDatatypeStr(dataType),
        dataCount,
        baselineAttr->nChannels,
        baselineAttr->root,
        getRedOpStr(baselineAttr->op),
        protoStr,
        algoStr,
        getNcclPatternStr(baselineAttr->pattern));
  } else {
    return CollSignature(
        commHash,
        pgid,
        rank,
        opName,
        nThreads,
        getDatatypeStr(dataType),
        algoName,
        dataCount);
  }
}

static std::vector<std::string> collKeys = {
    "collId",         "opCount",
    "opName",         "algoName",
    "sendbuff",       "recvbuff",
    "count",          "dataType",
    "redOp",          "root",
    "algorithm",      "protocol",
    "pattern",        "channelId",
    "nChannels",      "nThreads",
    "latencyUs",      "startTs",
    "enqueueTs",      "InterCollTimeUs",
    "QueueingTimeUs", "ExecutionTimeUs",
    "codepath",       "ranksInGroupedP2P",
    "checksum",       "iteration"};

// Always do not quote outcome from nested objects
static std::unordered_set<std::string> collUnquoteKeys = {"ranksInGroupedP2P"};

std::unordered_map<std::string, std::string> CollTraceColl::retrieveMap(
    bool quoted) const {
  std::unordered_map<std::string, std::string> infoMap;

  auto scubaEntry = toScubaEntry();

  const auto& normalMap = scubaEntry.getNormalMap();
  const auto& intMap = scubaEntry.getIntMap();

  for (const auto& key : collKeys) {
    if (normalMap.count(key)) {
      if (quoted && !collUnquoteKeys.contains(key)) {
        infoMap[key] = toQuotedString(normalMap.at(key));
      } else {
        infoMap[key] = normalMap.at(key);
      }
    } else if (intMap.count(key)) {
      infoMap[key] = std::to_string(intMap.at(key));
    }
  }

  return infoMap;
}

std::string CollTraceColl::serialize(bool quoted) const {
  std::unordered_map<std::string, std::string> infoMap = retrieveMap(quoted);
  return serializeMap(collKeys, infoMap, quoted);
}

std::string CollTraceColl::toString() const {
  std::unordered_map<std::string, std::string> infoMap = retrieveMap(false);
  // Convert integer sendbuff and recvbuff to hexadecimal only for display
  if (sendbuff.has_value()) {
    infoMap["sendbuff"] =
        uint64ToHexStr(reinterpret_cast<uint64_t>(*sendbuff), "0x");
  }
  if (recvbuff.has_value()) {
    infoMap["recvbuff"] =
        uint64ToHexStr(reinterpret_cast<uint64_t>(*recvbuff), "0x");
  }
  return mapToString(collKeys, infoMap);
}

folly::dynamic CollTraceColl::toDynamic() const noexcept {
  auto map = toScubaEntry();
  folly::dynamic result = folly::dynamic::object;
  for (const auto& [key, value] : map.getNormalMap()) {
    // Special handling for ranksInGroupedP2P. Because it is actually a vector,
    // we previously serialize it to a string and store it in the normal map.
    // Now since we are using folly::dynamic, we need to store it as a vector.
    if (key == "ranksInGroupedP2P") {
      result[key] = folly::toDynamic(this->ranksInGroupedP2P);
    } else {
      result[key] = value;
    }
  }
  for (const auto& [key, value] : map.getIntMap()) {
    result[key] = value;
  }
  for (const auto& [key, value] : map.getDoubleMap()) {
    result[key] = value;
  }
  return result;
}

uint64_t CollTraceColl::getCollId() const noexcept {
  return collId;
}
