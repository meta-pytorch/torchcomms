// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/pipes/Signal.cuh"

namespace comms::pipes::benchmark {

/**
 * signalAddBenchKernel - Benchmark kernel for P2P signaling using Signal Add
 *
 * @param remote Array of Signal objects (one per block/group)
 * @param local Array of Signal objects (one per block/group)
 * @param nSteps Number of signal/wait steps to perform
 * @param useBlockGroups If true, use block groups; otherwise use warp groups
 */
__global__ void signalAddBenchKernel(
    Signal* remote,
    Signal* local,
    int nSteps,
    bool useBlockGroups);

/**
 * signalSetBenchKernel - Benchmark kernel for P2P signaling using Signal Set
 *
 * @param remote Array of Signal objects (one per block/group)
 * @param local Array of Signal objects (one per block/group)
 * @param nSteps Number of signal/wait steps to perform
 * @param useBlockGroups If true, use block groups; otherwise use warp groups
 */
__global__ void signalSetBenchKernel(
    Signal* remote,
    Signal* local,
    int nSteps,
    bool useBlockGroups);

} // namespace comms::pipes::benchmark
