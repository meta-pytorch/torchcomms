// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/benchmarks/BarrierBench.cuh"

namespace comms::pipes::benchmark {

/**
 * barrierBenchKernel - Benchmark kernel for P2P barrier synchronization
 *
 * Uses P2pNvlTransportDevice::barrier_sync_threadgroup() for cross-GPU
 * synchronization over NVLink.
 */
__global__ void
barrierBenchKernel(P2pNvlTransportDevice p2p, int nSteps, bool useBlockGroups) {
  auto group = useBlockGroups ? make_block_group() : make_warp_group();
  for (int step = 0; step < nSteps; ++step) {
    p2p.barrier_sync_threadgroup(group, group.group_id);
  }
}

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
    cudaStream_t stream) {
  // Create transport options (only barrier buffers are used)
  P2pNvlTransportOptions options{
      .dataBufferSize = 1024,
      .chunkSize = 512,
      .pipelineDepth = 2,
  };

  // Create local and remote state with barrier buffers
  LocalState localState{
      .dataBuffer = nullptr,
      .stateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(nullptr, 0),
      .barrierBuffer = DeviceSpan<BarrierState>(localBarrier, numBarriers),
  };
  RemoteState remoteState{
      .dataBuffer = nullptr,
      .stateBuffer = DeviceSpan<ChunkState>(nullptr, 0),
      .signalBuffer = DeviceSpan<SignalState>(nullptr, 0),
      .barrierBuffer = DeviceSpan<BarrierState>(remoteBarrier, numBarriers),
  };

  // Create P2pNvlTransportDevice
  P2pNvlTransportDevice transport(
      localGpu, remoteGpu, options, localState, remoteState);

  // Launch kernel
  void* kernArgs[3] = {
      (void*)&transport, (void*)&nSteps, (void*)&useBlockGroups};
  dim3 grid{static_cast<unsigned int>(nBlocks), 1, 1};
  dim3 blocks{static_cast<unsigned int>(nThreads), 1, 1};
  cudaLaunchKernel(
      (const void*)barrierBenchKernel, grid, blocks, kernArgs, 0, stream);
}

} // namespace comms::pipes::benchmark
