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
  /// @p sockBufSize is the SO_SNDBUF/SO_RCVBUF value for the data connection;
  /// nullopt leaves the kernel's own sizing in place, which is what lets
  /// receive autotuning grow the window past the bandwidth-delay product.
  TcpBandwidthBenchmark(
      std::string iface,
      std::optional<int> sockBufSize,
      bool asyncGetH2d = true)
      : iface_(std::move(iface)),
        sockBufSize_(sockBufSize),
        asyncGetH2d_(asyncGetH2d) {}

  std::string name() const override {
    return "tcp_bandwidth";
  }

  std::vector<BenchmarkResult> run(
      const BenchmarkConfig& config,
      std::vector<PeerConnection>& peers,
      const BootstrapConfig& bootstrap) override;

 private:
  std::string iface_;
  std::optional<int> sockBufSize_;
  bool asyncGetH2d_{true};
};

} // namespace uniflow::benchmark
