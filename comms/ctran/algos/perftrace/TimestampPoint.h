// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <chrono>
#include <map>
#include <string>

namespace ctran::perftrace {

class TimestampPoint {
 public:
  TimestampPoint() : TimestampPoint(0) {}

  explicit TimestampPoint(int peer)
      : now_(std::chrono::system_clock::now()), peer_(peer) {}
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

} // namespace ctran::perftrace
