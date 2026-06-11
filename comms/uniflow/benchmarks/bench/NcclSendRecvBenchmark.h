// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <string>
#include <vector>

#include "comms/uniflow/benchmarks/BenchmarkRunner.h"

namespace uniflow::benchmark {

/// NCCL send/recv benchmark for direct comparison with the uniflow
/// copy-based send/recv benchmark under the same framework.
/// Uses ncclSend/ncclRecv in a ring pattern (same as nccl-tests sendrecv).
class NcclSendRecvBenchmark : public Benchmark {
 public:
  std::string name() const override {
    return "nccl_sendrecv";
  }

  std::vector<BenchmarkResult> run(
      const BenchmarkConfig& config,
      std::vector<PeerConnection>& peers,
      const BootstrapConfig& bootstrap) override;
};

} // namespace uniflow::benchmark
