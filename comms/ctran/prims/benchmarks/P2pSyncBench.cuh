// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/ctran/prims/ChunkState.cuh"
#include "comms/ctran/prims/ThreadGroup.cuh"

namespace ctran::prims::benchmark {

/**
 * p2pSyncKernel - Benchmark kernel for P2P synchronization using ChunkState
 *
 * Sender and receiver alternate signaling through ChunkState:
 *   - Sender: wait_ready_to_send() -> ready_to_recv(step)
 *   - Receiver: wait_ready_to_recv(step) -> ready_to_send()
 *
 * @param chunkStates Array of ChunkState objects (one per block/group)
 * @param isSender True for sender kernel, false for receiver
 * @param nSteps Number of sync steps to perform
 * @param groupScope Thread group scope (WARP, MULTIWARP, or BLOCK)
 */
__global__ void p2pSyncKernel(
    ChunkState* chunkStates,
    bool isSender,
    int nSteps,
    SyncScope groupScope);

} // namespace ctran::prims::benchmark
