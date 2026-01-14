// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/pipes/P2pSelfTransportDevice.cuh"
#include "comms/pipes/benchmarks/SelfTransportBench.cuh"

namespace comms::pipes::benchmark {

__global__ void selfTransportWriteKernel(
    char* dst,
    const char* src,
    std::size_t nBytes,
    int nRuns) {
  P2pSelfTransportDevice transport;
  auto group = make_warp_group();

  for (int run = 0; run < nRuns; ++run) {
    transport.write(group, dst, src, nBytes);
  }
}

} // namespace comms::pipes::benchmark
