// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/uniflow/benchmarks/BenchmarkRunner.h"

namespace uniflow::benchmark {

/// Measures the time to create a transport factory, exchange topology,
/// create transport, bind, exchange TransportInfo, and connect.
class ConnectionSetupBenchmark : public Benchmark {
 public:
  std::string name() const override {
    return "connection_setup";
  }

  std::vector<BenchmarkResult> run(
      const BenchmarkConfig& config,
      std::vector<PeerConnection>& peers,
      const BootstrapConfig& bootstrap) override;
};

} // namespace uniflow::benchmark
