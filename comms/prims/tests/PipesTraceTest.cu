// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/prims/tests/PipesTraceTest.cuh"

#include "comms/prims/trace/PipesTraceTypes.h"

namespace comms::prims::test {

__global__ void writeAndSpinKernel(
    PipesTraceHandle trace,
    volatile int* releaseFlag,
    int numEvents) {
  for (int i = 0; i < numEvents; i++) {
    write_pipes_trace(
        trace,
        PipesTraceEventType::kHierAgNvlTaskDone,
        static_cast<uint32_t>(i),
        static_cast<uint16_t>(i),
        0);
  }
  __syncthreads();
  __threadfence_system();
  while (*releaseFlag == 0) {
    // Spin until the host releases the kernel.
  }
}

__global__ void writeEventsKernel(PipesTraceHandle trace, int numEvents) {
  for (int i = 0; i < numEvents; i++) {
    write_pipes_trace(
        trace,
        PipesTraceEventType::kHierAgNvlTaskDone,
        static_cast<uint32_t>(i),
        static_cast<uint16_t>(i),
        0);
  }
}

__global__ void writeEventsMultiBlockKernel(
    PipesTraceHandle trace,
    int eventsPerBlock) {
  int blockId = static_cast<int>(blockIdx.x);
  for (int i = 0; i < eventsPerBlock; i++) {
    write_pipes_trace(
        trace,
        PipesTraceEventType::kHierAgNvlTaskDone,
        static_cast<uint32_t>(i),
        static_cast<uint16_t>(blockId),
        static_cast<uint8_t>(blockId));
  }
}

void launchWriteAndSpin(
    PipesTraceHandle trace,
    volatile int* releaseFlag,
    int numEvents,
    cudaStream_t stream) {
  // NOLINTNEXTLINE(facebook-cuda-safe-kernel-call-check)
  writeAndSpinKernel<<<1, 1, 0, stream>>>(trace, releaseFlag, numEvents);
}

void launchWriteEvents(
    PipesTraceHandle trace,
    int numEvents,
    cudaStream_t stream) {
  // NOLINTNEXTLINE(facebook-cuda-safe-kernel-call-check)
  writeEventsKernel<<<1, 1, 0, stream>>>(trace, numEvents);
}

void launchWriteEventsMultiBlock(
    PipesTraceHandle trace,
    int numBlocks,
    int eventsPerBlock,
    cudaStream_t stream) {
  // NOLINTNEXTLINE(facebook-cuda-safe-kernel-call-check)
  writeEventsMultiBlockKernel<<<numBlocks, 1, 0, stream>>>(
      trace, eventsPerBlock);
}

} // namespace comms::prims::test
