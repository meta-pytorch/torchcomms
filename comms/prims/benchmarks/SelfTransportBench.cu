// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/prims/benchmarks/SelfTransportBench.cuh"
#include "comms/prims/transport/self/P2pSelfTransportDevice.cuh"

namespace comms::prims::benchmark {

__global__ void selfTransportPutKernel(
    char* dst,
    const char* src,
    std::size_t nBytes,
    int nRuns) {
  P2pSelfTransportDevice transport;
  auto group = make_warp_group();

  for (int run = 0; run < nRuns; ++run) {
    transport.put_group(group, dst, src, nBytes);
  }
}

__global__ void selfTransportPutTileKernel(
    char* dst,
    const char* src,
    std::size_t tileSize,
    int nRuns) {
  P2pSelfTransportDevice transport;
  auto group = make_block_group();
  std::size_t offset = group.group_id * tileSize;

  for (int run = 0; run < nRuns; ++run) {
    transport.put(group, dst + offset, src + offset, tileSize);
  }
}

} // namespace comms::prims::benchmark
