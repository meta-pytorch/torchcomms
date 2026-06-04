// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/prims/P2pSelfTransportDevice.cuh"
#include "comms/ctran/prims/benchmarks/SelfTransportBench.cuh"

namespace ctran::prims::benchmark {

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

} // namespace ctran::prims::benchmark
