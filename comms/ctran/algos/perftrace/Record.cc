// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/algos/perftrace/Record.h"
#include "comms/utils/cvars/nccl_cvars.h"

#include <fmt/format.h>
#include <folly/String.h>

namespace ctran::perftrace {

Record::Record(const std::string& algo, const int rank)
    : algo(algo), rank_(rank) {
  mapSeqNumTimePoints.clear();
  mapSeqNumTimeIntervals.clear();
  if (NCCL_CTRAN_ENABLE_PERFTRACE) {
    enabled = true;
    start_ = std::chrono::system_clock::now();
  } else {
    enabled = false;
  }
}

void Record::addMetadata(const std::string& key, const std::string& val) {
  if (enabled) {
    metaDataKeys_.push_back(key);
    metaDataVals_.push_back(val);
  }
}

std::string Record::toJsonEntry(int& id, const int pid) const {
  std::vector<std::string> jsonEntries;
  std::vector<std::string> argStrs;
  argStrs.reserve(metaDataKeys_.size());
  for (auto i = 0; i < metaDataKeys_.size(); i++) {
    argStrs.push_back(
        fmt::format(
            "\"{}\": \"{}\"", metaDataKeys_.at(i), metaDataVals_.at(i)));
  }
  std::string argStr = folly::join(", ", argStrs);

  std::string recordEntry = "{\"name\": \"" + algo + "\", " +
      "\"cat\": \"COL\", " + "\"id\": " + std::to_string(id) + ", " +
      "\"ph\": \"X\", " + "\"pid\": " + std::to_string(pid) + ", " +
      "\"dur\": " + std::to_string(durationUs()) + ", " + "\"args\": {" +
      argStr + "}," + "\"tid\": -1, " + "\"ts\": " +
      std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(
                         start_.time_since_epoch())
                         .count()) +
      "}";
  id++;
  jsonEntries.push_back(recordEntry);

  for (const auto& entry : mapSeqNumTimePoints) {
    const auto& [name, seqNum] = entry.first;
    auto& tsp = entry.second;
    id++;
    jsonEntries.push_back(tsp.toJsonEntry(name, id, pid, seqNum));
  }
  for (const auto& entry : mapSeqNumTimeIntervals) {
    const auto& [name, seqNum] = entry.first;
    auto& timeInterval = entry.second;
    id++;
    jsonEntries.push_back(timeInterval.toJsonEntry(name, id, pid, seqNum));
  }
  return folly::join(",", jsonEntries);
}

} // namespace ctran::perftrace
