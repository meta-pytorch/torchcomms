// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/pipes/ChunkState.cuh"

namespace comms::pipes::benchmark {

/**
 * p2pSyncKernel - Benchmark kernel for P2P synchronization using ChunkState
 *
 * Sender and receiver alternate signaling through ChunkState:
 *   - Sender: waitReadyToSend() -> readyToRecv(step)
 *   - Receiver: waitReadyToRecv(step) -> readyToSend()
 *
 * @param chunkStates Array of ChunkState objects (one per block/group)
 * @param isSender True for sender kernel, false for receiver
 * @param nSteps Number of sync steps to perform
 * @param useBlockGroups If true, use block groups; otherwise use warp groups
 */
__global__ void p2pSyncKernel(
    ChunkState* chunkStates,
    bool isSender,
    int nSteps,
    bool useBlockGroups);

} // namespace comms::pipes::benchmark
