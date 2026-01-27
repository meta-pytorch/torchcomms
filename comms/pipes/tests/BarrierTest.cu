// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/pipes/BarrierState.cuh"
#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/pipes/tests/BarrierTest.cuh"
#include "comms/pipes/tests/Checks.h"
#include "comms/pipes/tests/P2pNvlTransportDeviceTest.cuh"

namespace comms::pipes::test {

// Helper to create the appropriate thread group based on type
__device__ inline ThreadGroup make_group(GroupType groupType) {
  switch (groupType) {
    case GroupType::WARP:
      return make_warp_group();
    case GroupType::BLOCK:
      return make_block_group();
    default:
      return make_warp_group();
  }
}

// =============================================================================
// Barrier struct test kernels
// =============================================================================

__global__ void testRawBarrierArriveKernel(
    BarrierState* barrier_d,
    GroupType groupType) {
  auto group = make_group(groupType);
  barrier_d->arrive(group);
}

__global__ void testRawBarrierWaitKernel(
    BarrierState* barrier_d,
    GroupType groupType) {
  auto group = make_group(groupType);
  barrier_d->wait(group);
}

__global__ void testRawBarrierArriveWaitKernel(
    BarrierState* barrier_d,
    GroupType groupType) {
  auto group = make_group(groupType);
  barrier_d->arrive(group);
  barrier_d->wait(group);
}

__global__ void testReadBarrierCurrentCounterKernel(
    BarrierState* barrier_d,
    uint64_t* result_d) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *result_d = barrier_d->current_counter_.load();
  }
}

__global__ void testReadBarrierExpectedCounterKernel(
    BarrierState* barrier_d,
    uint64_t* result_d) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *result_d = barrier_d->expected_counter_.load();
  }
}

void testRawBarrierArrive(
    BarrierState* barrier_d,
    int numBlocks,
    int blockSize,
    GroupType groupType) {
  testRawBarrierArriveKernel<<<numBlocks, blockSize>>>(barrier_d, groupType);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testRawBarrierWait(
    BarrierState* barrier_d,
    int numBlocks,
    int blockSize,
    GroupType groupType) {
  testRawBarrierWaitKernel<<<numBlocks, blockSize>>>(barrier_d, groupType);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testRawBarrierArriveWait(
    BarrierState* barrier_d,
    int numBlocks,
    int blockSize,
    GroupType groupType) {
  testRawBarrierArriveWaitKernel<<<numBlocks, blockSize>>>(
      barrier_d, groupType);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testReadBarrierCurrentCounter(
    BarrierState* barrier_d,
    uint64_t* result_d) {
  testReadBarrierCurrentCounterKernel<<<1, 1>>>(barrier_d, result_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testReadBarrierExpectedCounter(
    BarrierState* barrier_d,
    uint64_t* result_d) {
  testReadBarrierExpectedCounterKernel<<<1, 1>>>(barrier_d, result_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// P2pNvlTransportDevice barrier API test kernels
// =============================================================================

__global__ void testDeviceBarrierSyncKernel(
    P2pNvlTransportDevice p2p,
    uint64_t barrierId,
    GroupType groupType) {
  auto group = make_group(groupType);
  p2p.barrier_sync_threadgroup(group, barrierId);
}

__global__ void testDeviceBarrierSyncMultipleKernel(
    P2pNvlTransportDevice p2p,
    uint64_t barrierId,
    int numSyncs,
    GroupType groupType) {
  auto group = make_group(groupType);
  for (int i = 0; i < numSyncs; ++i) {
    p2p.barrier_sync_threadgroup(group, barrierId);
  }
}

void testDeviceBarrierSync(
    P2pNvlTransportDevice p2p,
    uint64_t barrierId,
    int numBlocks,
    int blockSize,
    GroupType groupType) {
  testDeviceBarrierSyncKernel<<<numBlocks, blockSize>>>(
      p2p, barrierId, groupType);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testDeviceBarrierSyncMultiple(
    P2pNvlTransportDevice p2p,
    uint64_t barrierId,
    int numSyncs,
    int numBlocks,
    int blockSize,
    GroupType groupType) {
  testDeviceBarrierSyncMultipleKernel<<<numBlocks, blockSize>>>(
      p2p, barrierId, numSyncs, groupType);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// Data transfer with barrier verification test kernels
// Tests that data written by one GPU is visible to another after barrier sync
// =============================================================================

__global__ void testBarrierWriteDataKernel(
    P2pNvlTransportDevice p2p,
    char* remoteDataBuffer,
    const char* localSrcBuffer,
    size_t dataSize,
    GroupType groupType) {
  auto group = make_group(groupType);

  // Each thread group uses its own barrier id
  uint64_t barrierId = group.group_id;

  // The put() API distributes work across all thread groups automatically
  p2p.put(group, remoteDataBuffer, localSrcBuffer, dataSize);

  // Each thread group uses its own barrier id
  p2p.barrier_sync_threadgroup(group, barrierId);
}

__global__ void testBarrierVerifyDataKernel(
    P2pNvlTransportDevice p2p,
    uint8_t* localDataBuffer,
    size_t dataSize,
    uint8_t expectedValue,
    uint32_t* errorCount,
    GroupType groupType) {
  auto group = make_group(groupType);

  // Each thread group uses its own barrier id (matches the writer)
  uint64_t barrierId = group.group_id;

  // Barrier sync - arrive on remote, wait on local
  // This ensures writer's data is visible before we read
  p2p.barrier_sync_threadgroup(group, barrierId);

  // Calculate the portion of data this thread group handles
  size_t bytesPerGroup = dataSize / group.total_groups;
  size_t startOffset = group.group_id * bytesPerGroup;
  size_t endOffset = min(startOffset + bytesPerGroup, dataSize);

  // Each thread in the group verifies its portion
  for (size_t i = startOffset + group.thread_id_in_group; i < endOffset;
       i += group.group_size) {
    if (localDataBuffer[i] != expectedValue) {
      comms::device::atomic_fetch_add_relaxed_gpu_global(errorCount, 1);
    }
  }
}

void testBarrierWriteData(
    P2pNvlTransportDevice p2p,
    char* remoteDataBuffer,
    const char* localSrcBuffer,
    size_t dataSize,
    int numBlocks,
    int blockSize,
    GroupType groupType) {
  testBarrierWriteDataKernel<<<numBlocks, blockSize>>>(
      p2p, remoteDataBuffer, localSrcBuffer, dataSize, groupType);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testBarrierVerifyData(
    P2pNvlTransportDevice p2p,
    uint8_t* localDataBuffer,
    size_t dataSize,
    uint8_t expectedValue,
    uint32_t* errorCount,
    int numBlocks,
    int blockSize,
    GroupType groupType) {
  testBarrierVerifyDataKernel<<<numBlocks, blockSize>>>(
      p2p, localDataBuffer, dataSize, expectedValue, errorCount, groupType);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::pipes::test
