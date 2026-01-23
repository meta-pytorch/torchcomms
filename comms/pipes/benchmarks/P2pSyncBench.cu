// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/benchmarks/P2pSyncBench.cuh"

namespace comms::pipes::benchmark {

__global__ void p2pSyncKernel(
    ChunkState* chunkStates,
    bool isSender,
    int nSteps,
    SyncScope groupScope) {
  auto group = make_thread_group(groupScope);
  auto groupIdx = group.group_id;

  ChunkState* myChunkState = &chunkStates[groupIdx];

  for (int step = 1; step <= nSteps; step++) {
    if (isSender) {
      myChunkState->waitReadyToSend(group);
      myChunkState->readyToRecv(group, step);
    } else {
      myChunkState->waitReadyToRecv(group, step);
      myChunkState->readyToSend(group);
    }
  }
}

} // namespace comms::pipes::benchmark
