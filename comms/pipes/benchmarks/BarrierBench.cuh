// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>
#include "comms/pipes/P2pNvlTransportDevice.cuh"

namespace comms::pipes::benchmark {

/**
 * P2P barrier synchronization kernel
 *
 * This kernel performs barrier synchronization between two GPUs using the
 * P2pNvlTransportDevice::barrier() API.
 *
 * @param p2p P2pNvlTransportDevice configured for this GPU
 * @param nSteps Number of barrier iterations to perform
 * @param useBlockGroups If true, use block-level ThreadGroup; if false, use
 * warp
 */
__global__ void p2pBarrierThreadGroupBenchKernel(
    P2pNvlTransportDevice p2p,
    int nSteps,
    bool useBlockGroups);

} // namespace comms::pipes::benchmark
