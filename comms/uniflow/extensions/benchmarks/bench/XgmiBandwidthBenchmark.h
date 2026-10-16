// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <string>
#include <vector>

#include "comms/uniflow/benchmarks/BenchmarkRunner.h"

namespace uniflow::benchmark {

/// Measures intra-node put/get bandwidth across message sizes on AMD, where the
/// intra-node tier is served by HIP IPC over XGMI.
///
/// This is the AMD counterpart of NVLinkBandwidthBenchmark: same tier, same
/// shared IntraNodeTransport, same measurement method -- the difference is the
/// registration backend (IPC instead of CUDA VMM) and therefore the allocator
/// (plain device allocation instead of a cuMem VMM mapping, since IPC exports
/// an ordinary allocation).
///
/// AMD-only: on NVIDIA the intra-node tier is the VMM path and
/// nvlink_bandwidth already covers it, so main.cpp registers this benchmark
/// only under __HIP_PLATFORM_AMD__.
class XgmiBandwidthBenchmark : public Benchmark {
 public:
  std::string name() const override {
    return "xgmi_bandwidth";
  }

  std::vector<BenchmarkResult> run(
      const BenchmarkConfig& config,
      std::vector<PeerConnection>& peers,
      const BootstrapConfig& bootstrap) override;
};

} // namespace uniflow::benchmark
