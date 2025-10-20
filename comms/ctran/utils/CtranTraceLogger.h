// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <folly/Synchronized.h>
#include <chrono>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace ctran::utils {

class TimestampPoint {
 public:
  explicit TimestampPoint() {
    TimestampPoint(0);
  }

  explicit TimestampPoint(int peer) {
    this->now_ = std::chrono::system_clock::now();
    this->peer_ = peer;
  }
  ~TimestampPoint() = default;

  int64_t durationMs(
      const std::chrono::time_point<std::chrono::system_clock>& begin) const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(now_ - begin)
        .count();
  }

  int64_t durationUs(
      const std::chrono::time_point<std::chrono::system_clock>& begin) const {
    return std::chrono::duration_cast<std::chrono::microseconds>(now_ - begin)
        .count();
  }

  std::string toJsonEntry(
      const std::string& name,
      int id,
      const int pid,
      const int seqNum = -1) const;

  void setMetadata(const std::map<std::string, std::string>& metaData) {
    this->metaData_ = metaData;
  }

 private:
  std::chrono::time_point<std::chrono::system_clock> now_;
  int peer_;
  std::map<std::string, std::string> metaData_;
};

class TimeInterval {
 public:
  explicit TimeInterval() {
    TimeInterval(0);
  }

  explicit TimeInterval(int peer) {
    start_ = std::chrono::system_clock::now();
    peer_ = peer;
    completed_ = false;
  }
  ~TimeInterval() = default;

  std::string toJsonEntry(
      const std::string& name,
      int id,
      const int pid,
      const int seqNum = -1) const;

  void start() {
    this->start_ = std::chrono::system_clock::now();
  }

  void end() {
    if (!completed_) {
      this->end_ = std::chrono::system_clock::now();
      completed_ = true;
    }
  }

  int64_t durationMs() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               end_ - this->start_)
        .count();
  }

  int64_t durationUs() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               end_ - this->start_)
        .count();
  }

  void setMetadata(const std::map<std::string, std::string>& metaData) {
    this->metaData_ = metaData;
  }

 private:
  std::chrono::time_point<std::chrono::system_clock> start_;
  std::chrono::time_point<std::chrono::system_clock> end_;
  bool completed_ = false;
  int peer_;
  std::map<std::string, std::string> metaData_;
};

// Trace record is not thread safe, assume it is only used in one thread
class TraceRecord {
 public:
  explicit TraceRecord(const std::string& algo, const int rank = -1);
  ~TraceRecord() = default;

  // Add a point with sequence number. If the key <name, seqNum> already
  // exists, it will hit an assertion fail. Caller is responsible to ensure
  // the uniqueness.
  void addPoint(
      const std::string& name,
      const int seqNum,
      std::optional<const int> peer = std::nullopt,
      std::optional<const std::map<std::string, std::string>> metaData =
          std::nullopt);

  // Add a start interval with sequence number. If the key <name, seqNum>
  // already exists, it will hit an assertion fail. Caller is responsible to
  // ensure the uniqueness.
  void startInterval(
      const std::string& name,
      const int seqNum,
      std::optional<const int> peer = std::nullopt,
      std::optional<const std::map<std::string, std::string>> metaData =
          std::nullopt);

  // End an existing interval. If the key <name, seqNum> doesn't exist,
  // it will hit an assertion fail. Caller is responsible to ensure the
  // the corresponding starting interval has been recorded.
  void endInterval(const std::string& name, const int seqNum);

  // Check whether a start interval with <name, seqNum> exists.
  bool hasInterval(const std::string& name, const int seqNum);

  // Record completion of the entire trace record
  void end();

  inline void addMetadata(const std::string& key, const std::string& val) {
    if (enabled) {
      metaDataKeys_.push_back(key);
      metaDataVals_.push_back(val);
    }
  }

  std::string toJsonEntry(int& id, const int pid) const;

  uint64_t durationUs() const {
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

// TraceLogger is thread safe, allowing multiple threads to add trace
// records
class TraceLogger {
 public:
  explicit TraceLogger(int rank);
  ~TraceLogger();

  bool isTraceEnabled() {
    return traceEnabled_;
  }

  void addTraceRecord(std::unique_ptr<TraceRecord> traceRecord);

 private:
  void reportTracing();
  bool traceEnabled_ = false;
  int rank_;
  folly::Synchronized<std::vector<std::unique_ptr<TraceRecord>>> traceRecords_;
};

} // namespace ctran::utils
