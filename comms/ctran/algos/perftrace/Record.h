// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <chrono>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "comms/ctran/algos/perftrace/TimeInterval.h"
#include "comms/ctran/algos/perftrace/TimestampPoint.h"
#include "comms/ctran/utils/Checks.h"

namespace ctran::perftrace {

// Record is not thread safe, assume it is only used in one thread
class Record {
 public:
  explicit Record(const std::string& algo, const int rank = -1);
  ~Record() = default;

  // Add a point with sequence number. If the key <name, seqNum> already
  // exists, it will hit an assertion fail. Caller is responsible to ensure
  // the uniqueness.
  inline void addPoint(
      const std::string& name,
      const int seqNum,
      std::optional<const int> peer = std::nullopt,
      std::optional<const std::map<std::string, std::string>> metaData =
          std::nullopt) {
    if (enabled) {
      int p = rank_;
      if (peer.has_value()) {
        p = peer.value();
      }
      auto key = std::make_pair(name, seqNum);

      FB_CHECKABORT(
          !hasInterval(name, seqNum),
          fmt::format("Point name {} seqNum {} already exists", name, seqNum));

      TimestampPoint tp = TimestampPoint(p);
      if (metaData.has_value()) {
        tp.setMetadata(metaData.value());
      }
      mapSeqNumTimePoints[key] = tp;
    }
  }

  // Add a start interval with sequence number. If the key <name, seqNum>
  // already exists, it will hit an assertion fail. Caller is responsible to
  // ensure the uniqueness.
  inline void startInterval(
      const std::string& name,
      const int seqNum,
      std::optional<const int> peer = std::nullopt,
      std::optional<const std::map<std::string, std::string>> metaData =
          std::nullopt) {
    if (enabled) {
      int p = -1;
      if (peer.has_value()) {
        p = peer.value();
      }
      TimeInterval timeInterval = TimeInterval(p);
      timeInterval.start();

      if (metaData.has_value()) {
        timeInterval.setMetadata(metaData.value());
      }
      auto key = std::make_pair(name, seqNum);

      FB_CHECKABORT(
          !hasInterval(name, seqNum),
          fmt::format(
              "Starting interval name {} seqNum {} already exists",
              name,
              seqNum));
      mapSeqNumTimeIntervals[key] = timeInterval;
    }
  }

  // End an existing interval. If the key <name, seqNum> doesn't exist,
  // it will hit an assertion fail. Caller is responsible to ensure the
  // the corresponding starting interval has been recorded.
  void endInterval(const std::string& name, const int seqNum) {
    if (enabled) {
      FB_CHECKABORT(
          hasInterval(name, seqNum),
          fmt::format(
              "Starting interval name {} seqNum {} not found", name, seqNum));
      auto key = std::make_pair(name, seqNum);
      auto& record = mapSeqNumTimeIntervals[key];
      record.end();
    }
  }

  // Check whether a start interval with <name, seqNum> exists.
  inline bool hasInterval(const std::string& name, const int seqNum) {
    if (enabled) {
      auto it = mapSeqNumTimeIntervals.find(std::make_pair(name, seqNum));
      return it != mapSeqNumTimeIntervals.end();
    }
    return false;
  }

  // Record completion of the entire trace record
  inline void end() {
    if (enabled) {
      end_ = std::chrono::system_clock::now();
    }
  }

  void addMetadata(const std::string& key, const std::string& val);

  std::string toJsonEntry(int& id, const int pid) const;

  inline uint64_t durationUs() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(end_ - start_)
        .count();
  }

 private:
  // Report as "X" event in the trace
  // Using <name, seqNum> as key
  std::unordered_map<std::pair<std::string, int>, TimestampPoint>
      mapSeqNumTimePoints;

  // Report "b" and "e" internal in the trace
  // Using <name, seqNum> as key
  std::unordered_map<std::pair<std::string, int>, TimeInterval>
      mapSeqNumTimeIntervals;

  // start/end of the entire trace record
  std::chrono::time_point<std::chrono::system_clock> start_;
  std::chrono::time_point<std::chrono::system_clock> end_;
  // Use vector to maintain the order of metadata following inserting order
  std::vector<std::string> metaDataKeys_;
  std::vector<std::string> metaDataVals_;

  std::string algo;
  bool enabled = false;
  int rank_{-1};
};

} // namespace ctran::perftrace
