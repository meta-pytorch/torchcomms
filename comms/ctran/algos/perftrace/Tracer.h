// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <folly/Synchronized.h>
#include <memory>
#include <vector>

#include "comms/ctran/algos/perftrace/Record.h"

namespace ctran::perftrace {

// Tracer is thread safe, allowing multiple threads to add trace
// records
class Tracer {
 public:
  explicit Tracer(int rank);
  ~Tracer();

  bool isTraceEnabled() {
    return traceEnabled_;
  }

  void addRecord(std::unique_ptr<Record> record);

 private:
  void reportTracing();
  bool traceEnabled_ = false;
  int rank_;
  folly::Synchronized<std::vector<std::unique_ptr<Record>>> records_;
};

} // namespace ctran::perftrace
