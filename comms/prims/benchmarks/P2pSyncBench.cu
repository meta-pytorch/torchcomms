// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/ThreadGroup.cuh"
#include "comms/prims/benchmarks/P2pSyncBench.cuh"

namespace comms::prims::benchmark {

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
      myChunkState->wait_ready_to_send(group);
      myChunkState->ready_to_recv(group, step);
    } else {
      myChunkState->wait_ready_to_recv(group, step);
      myChunkState->ready_to_send(group);
    }
  }
}

} // namespace comms::prims::benchmark
