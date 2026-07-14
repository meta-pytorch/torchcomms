// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <string>
#include <vector>

#include "comms/uniflow/benchmarks/BenchmarkRunner.h"

namespace uniflow::benchmark {

/// Measures copy-based send/recv bandwidth across message sizes with
/// multi-peer support.  One RdmaTransportFactory creates N-1 transports
/// (one per peer), all sharing NicResources and SlabPool.
///
/// Topologies (--topology):
///   fanout  — rank 0 sends to all peers, others receive
///
/// fanin support will be added in a follow-up diff.
///
/// Bandwidth is aggregate: size * numPeers * iters / time, busBw factor = 1.
class SendRecvBandwidthBenchmark : public Benchmark {
 public:
  explicit SendRecvBandwidthBenchmark(std::vector<std::string> rdmaDevices)
      : rdmaDevices_(std::move(rdmaDevices)) {}

  std::string name() const override {
    return "sendrecv_bandwidth";
  }

  std::vector<BenchmarkResult> run(
      const BenchmarkConfig& config,
      std::vector<PeerConnection>& peers,
      const BootstrapConfig& bootstrap) override;

 private:
  std::vector<std::string> rdmaDevices_;
};

} // namespace uniflow::benchmark
