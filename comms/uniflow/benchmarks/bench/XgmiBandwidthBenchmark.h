// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "comms/uniflow/benchmarks/BenchmarkRunner.h"

namespace uniflow::benchmark {

/// Allocation behind a benchmark buffer, which decides how P2P shares it.
enum class XgmiMemory {
  Device, // cudaMalloc; shared through HIP IPC
  Vmm, // cuMemCreate chunks with a POSIX-fd handle type; shared as fds
};

/// What one xgmi_bandwidth run registers and how P2P may share it. Pairing a
/// Device source with a Vmm destination (or the reverse) separates the cost of
/// the peer's imported mapping from the cost of the local buffer.
struct XgmiOptions {
  // Local buffer: the source of put and the destination of get.
  XgmiMemory srcMemory{XgmiMemory::Device};
  // Buffer each rank registers for its peer to import and copy into or from.
  XgmiMemory dstMemory{XgmiMemory::Device};
  // VMM buffers are built from chunks of this size, each rounded up to the
  // allocation granularity, as PyTorch expandable segments are; 0 makes each
  // buffer one chunk.
  size_t vmmChunkSize{0};
  // P2pTransportFactory's enableVmm, which
  // MultiTransportFactoryOptions::p2pEnableVmm forwards to. False skips the VMM
  // check on every registration, so Device runs with it off measure what that
  // check costs. It must stay true for Vmm buffers: HIP IPC cannot share them.
  bool p2pEnableVmm{true};
};

/// Measures intra-node put/get bandwidth across message sizes on AMD, where the
/// intra-node tier is the P2P transport over XGMI.
///
/// This is the AMD counterpart of NVLinkBandwidthBenchmark: same tier, same
/// measurement method. XgmiOptions picks the allocation of each buffer. Either
/// way the copies run on the mapped peer memory; the sharing modes differ in
/// how registration and import set up that mapping, which the benchmark times
/// and logs per size, with the teardown of the previous size.
///
/// AMD-only: on NVIDIA the intra-node tier is the VMM path and
/// nvlink_bandwidth already covers it, so main.cpp registers this benchmark
/// only under __HIP_PLATFORM_AMD__.
class XgmiBandwidthBenchmark : public Benchmark {
 public:
  explicit XgmiBandwidthBenchmark(XgmiOptions options = {})
      : options_{options} {}

  std::string name() const override {
    return "xgmi_bandwidth";
  }

  std::vector<BenchmarkResult> run(
      const BenchmarkConfig& config,
      std::vector<PeerConnection>& peers,
      const BootstrapConfig& bootstrap) override;

 private:
  XgmiOptions options_;
};

} // namespace uniflow::benchmark
