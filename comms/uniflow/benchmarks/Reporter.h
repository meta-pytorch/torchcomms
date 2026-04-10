// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <ostream>
#include <string>
#include <vector>

#include "comms/uniflow/benchmarks/BenchmarkResult.h"
#include "comms/uniflow/benchmarks/Bootstrap.h"

namespace uniflow::benchmark {

class Reporter {
 public:
  static void printTable(
      const std::vector<BenchmarkResult>& results,
      std::ostream& os);

  static void printCSV(
      const std::vector<BenchmarkResult>& results,
      std::ostream& os);

  static void printHeader(
      const BootstrapConfig& config,
      const std::string& transport,
      std::ostream& os);
};

} // namespace uniflow::benchmark
