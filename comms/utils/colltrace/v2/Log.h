// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdio>
#include <sstream>
#include <string>

namespace meta::comms::colltracev2 {

// stderr only: colltrace v2 is shared by every backend in the process and so
// belongs to none of their loggers.
class LogLine {
 public:
  explicit LogLine(const char* level) : level_(level) {}
  ~LogLine() {
    std::fprintf(stderr, "[colltrace] %s %s\n", level_, os_.str().c_str());
  }
  LogLine(const LogLine&) = delete;
  LogLine& operator=(const LogLine&) = delete;
  LogLine(LogLine&&) = delete;
  LogLine& operator=(LogLine&&) = delete;

  template <typename T>
  LogLine& operator<<(const T& value) {
    os_ << value;
    return *this;
  }

 private:
  const char* level_;
  std::ostringstream os_;
};

#define COLLTRACE_LOG_WARN() ::meta::comms::colltracev2::LogLine("WARN")
#define COLLTRACE_LOG_ERROR() ::meta::comms::colltracev2::LogLine("ERROR")

} // namespace meta::comms::colltracev2
