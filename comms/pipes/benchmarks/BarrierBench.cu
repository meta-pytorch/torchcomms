// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>

#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/benchmarks/BarrierBench.cuh"

namespace comms::pipes::benchmark {

__global__ void p2pBarrierThreadGroupBenchKernel(
    P2pNvlTransportDevice p2p,
    int nSteps,
    bool useBlockGroups) {
  ThreadGroup group = useBlockGroups ? make_block_group() : make_warp_group();

  // Perform nSteps barrier synchronizations
  for (int step = 0; step < nSteps; ++step) {
    p2p.barrier_threadgroup(group);
  }
}

} // namespace comms::pipes::benchmark
