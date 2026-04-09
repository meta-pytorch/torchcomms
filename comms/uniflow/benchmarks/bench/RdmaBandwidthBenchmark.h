// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <string>
#include <vector>

#include "comms/uniflow/benchmarks/BenchmarkRunner.h"

namespace uniflow::benchmark {

/// Measures RDMA put/get bandwidth across message sizes.
class RdmaBandwidthBenchmark : public Benchmark {
 public:
  explicit RdmaBandwidthBenchmark(std::vector<std::string> rdmaDevices)
      : rdmaDevices_(std::move(rdmaDevices)) {}

  std::string name() const override {
    return "rdma_bandwidth";
  }

  std::vector<BenchmarkResult> run(
      const BenchmarkConfig& config,
      std::vector<PeerConnection>& peers,
      const BootstrapConfig& bootstrap) override;

 private:
  std::vector<std::string> rdmaDevices_;
};

} // namespace uniflow::benchmark
