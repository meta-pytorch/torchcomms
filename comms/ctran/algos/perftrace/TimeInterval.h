// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <chrono>
#include <map>
#include <string>

namespace ctran::perftrace {

class TimeInterval {
 public:
  TimeInterval() : TimeInterval(0) {}

  explicit TimeInterval(int peer)
      : start_(std::chrono::system_clock::now()),
        completed_(false),
        peer_(peer) {}
  ~TimeInterval() = default;

  std::string toJsonEntry(
      const std::string& name,
      int id,
      const int pid,
      const int seqNum = -1) const;

  inline void start() {
    this->start_ = std::chrono::system_clock::now();
  }

  inline void end() {
    if (!completed_) {
      this->end_ = std::chrono::system_clock::now();
      completed_ = true;
    }
  }

  inline int64_t durationMs() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               end_ - this->start_)
        .count();
  }

  inline int64_t durationUs() const {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               end_ - this->start_)
        .count();
  }

  inline void setMetadata(const std::map<std::string, std::string>& metaData) {
    this->metaData_ = metaData;
  }

 private:
  std::chrono::time_point<std::chrono::system_clock> start_;
  std::chrono::time_point<std::chrono::system_clock> end_;
  bool completed_ = false;
  int peer_;
  std::map<std::string, std::string> metaData_;
};

} // namespace ctran::perftrace
