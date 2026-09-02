// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <string>
#include <vector>

#include "comms/uniflow/benchmarks/BenchmarkRunner.h"

namespace uniflow::benchmark {

/// Measures TCP transport put/get bandwidth across message sizes over a chosen
/// front-end interface (default eth2). Single point-to-point connection; DRAM
/// buffers, or VRAM staged through host memory inside the transport.
class TcpBandwidthBenchmark : public Benchmark {
 public:
  explicit TcpBandwidthBenchmark(std::string iface)
      : iface_(std::move(iface)) {}

  std::string name() const override {
    return "tcp_bandwidth";
  }

  std::vector<BenchmarkResult> run(
      const BenchmarkConfig& config,
      std::vector<PeerConnection>& peers,
      const BootstrapConfig& bootstrap) override;

 private:
  std::string iface_;
};

} // namespace uniflow::benchmark
