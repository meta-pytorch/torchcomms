// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/pipes/BarrierState.cuh"

namespace comms::pipes::benchmark {

/**
 * launchBarrierBenchKernel - Launch barrier synchronization benchmark
 *
 * This wrapper function creates P2pNvlTransportDevice objects from raw
 * barrier pointers and launches the benchmark kernel. This allows the
 * .cc file to avoid including P2pNvlTransportDevice.cuh.
 *
 * @param localBarrier Local barrier buffer (waits here)
 * @param remoteBarrier Remote barrier buffer (arrives here)
 * @param numBarriers Number of barriers in the buffer
 * @param localGpu Local GPU ID
 * @param remoteGpu Remote GPU ID
 * @param nBlocks Number of thread blocks to launch
 * @param nThreads Number of threads per block
 * @param nSteps Number of barrier sync steps to perform
 * @param useBlockGroups If true, use block groups; otherwise use warp groups
 * @param stream CUDA stream to use for the kernel launch
 */
void launchBarrierBenchKernel(
    BarrierState* localBarrier,
    BarrierState* remoteBarrier,
    int numBarriers,
    int localGpu,
    int remoteGpu,
    int nBlocks,
    int nThreads,
    int nSteps,
    bool useBlockGroups,
    cudaStream_t stream);

} // namespace comms::pipes::benchmark
