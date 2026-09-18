// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <optional>
#include <string>
#include <vector>

#include "comms/uniflow/benchmarks/BenchmarkRunner.h"

namespace uniflow::benchmark {

/// Measures TCP transport put/get bandwidth across message sizes over a chosen
/// front-end interface (default eth2). Single point-to-point connection; DRAM
/// buffers, or VRAM staged through host memory inside the transport.
class TcpBandwidthBenchmark : public Benchmark {
 public:
  /// @p bindDevs are the network devices to pin egress to via SO_BINDTODEVICE,
  /// one listener per device and lane i placed on device i % size(). Empty
  /// leaves egress to the routing table, in which case @p iface only selects a
  /// source address and routing still decides which NIC traffic leaves through.
  TcpBandwidthBenchmark(
      std::string iface,
      bool asyncGetH2d = true,
      size_t socketsPerNic = 4,
      std::vector<std::string> bindDevs = {})
      : iface_(std::move(iface)),
        asyncGetH2d_(asyncGetH2d),
        socketsPerNic_(socketsPerNic),
        bindDevs_(std::move(bindDevs)) {}

  std::string name() const override {
    return "tcp_bandwidth";
  }

  std::vector<BenchmarkResult> run(
      const BenchmarkConfig& config,
      std::vector<PeerConnection>& peers,
      const BootstrapConfig& bootstrap) override;

 private:
  std::string iface_;
  bool asyncGetH2d_{true};
  // Lanes per bound device; total is this times the device count.
  size_t socketsPerNic_{4};
  std::vector<std::string> bindDevs_;
};

} // namespace uniflow::benchmark
