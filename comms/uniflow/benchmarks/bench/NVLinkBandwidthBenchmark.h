// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <string>
#include <vector>

#include "comms/uniflow/benchmarks/BenchmarkRunner.h"

namespace uniflow::benchmark {

/// Measures NVLink put/get bandwidth across message sizes.
/// Uses cuMem VMM allocations with fabric handles for cross-host MNNVL
/// or FD handles for intra-host P2P.
class NVLinkBandwidthBenchmark : public Benchmark {
 public:
  std::string name() const override {
    return "nvlink_bandwidth";
  }

  std::vector<BenchmarkResult> run(
      const BenchmarkConfig& config,
      std::vector<PeerConnection>& peers,
      const BootstrapConfig& bootstrap) override;
};

} // namespace uniflow::benchmark
