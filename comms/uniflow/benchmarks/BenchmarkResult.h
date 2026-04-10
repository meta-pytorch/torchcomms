// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <string>

#include "comms/uniflow/benchmarks/Stats.h"

namespace uniflow::benchmark {

struct BenchmarkResult {
  std::string benchmarkName;
  std::string transport;
  std::string direction;
  size_t messageSize{0};
  int iterations{0};
  int batchSize{0};
  size_t chunkSize{0};
  double bandwidthGBs{0};
  Stats latency{}; // in microseconds
  double messageRateMops{0};
  int numStreams{0};
};

} // namespace uniflow::benchmark
