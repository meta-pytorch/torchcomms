// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/pipes/tests/P2pNvlTransportTest.cuh"
#include "comms/pipes/tests/Utils.h"

namespace comms::pipes::test {

__global__ void fillBufferKernel(int* buffer, int value, size_t numElements) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    buffer[idx] = value;
  }
}

__global__ void verifyBufferKernel(
    const int* buffer,
    int expectedValue,
    size_t numElements,
    int* errorCount) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    if (buffer[idx] != expectedValue) {
      atomicAdd(errorCount, 1);
    }
  }
}

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

__global__ void testSendKernel(
    P2pNvlTransportDevice p2p,
    void* src_d,
    size_t nbytes,
    GroupType groupType) {
  auto group = make_group(groupType);
  p2p.send(group, src_d, nbytes);
}

__global__ void testRecvKernel(
    P2pNvlTransportDevice p2p,
    void* dst_d,
    size_t nbytes,
    GroupType groupType) {
  auto group = make_group(groupType);
  p2p.recv(group, dst_d, nbytes);
}

// Kernel that performs multiple sequential sends within a single kernel launch
__global__ void testMultiSendKernel(
    P2pNvlTransportDevice p2p,
    void* src_d,
    size_t nbytes,
    int numSends,
    GroupType groupType) {
  auto group = make_group(groupType);
  char* src = reinterpret_cast<char*>(src_d);
  for (int i = 0; i < numSends; i++) {
    p2p.send(group, src + i * nbytes, nbytes);
  }
}

// Kernel that performs multiple sequential recvs within a single kernel launch
__global__ void testMultiRecvKernel(
    P2pNvlTransportDevice p2p,
    void* dst_d,
    size_t nbytes,
    int numRecvs,
    GroupType groupType) {
  auto group = make_group(groupType);
  char* dst = reinterpret_cast<char*>(dst_d);
  for (int i = 0; i < numRecvs; i++) {
    p2p.recv(group, dst + i * nbytes, nbytes);
  }
}

// Kernel that performs both send and recv within a single kernel launch
// Used for pipelined bidirectional communication
__global__ void testSendRecvKernel(
    P2pNvlTransportDevice p2p,
    void* send_d,
    void* recv_d,
    size_t nbytes,
    GroupType groupType) {
  auto group = make_group(groupType);
  p2p.send(group, send_d, nbytes);
  p2p.recv(group, recv_d, nbytes);
}

// Kernel that performs recv then send within a single kernel launch
// Paired with testSendRecvKernel for bidirectional tests
__global__ void testRecvSendKernel(
    P2pNvlTransportDevice p2p,
    void* recv_d,
    void* send_d,
    size_t nbytes,
    GroupType groupType) {
  auto group = make_group(groupType);
  p2p.recv(group, recv_d, nbytes);
  p2p.send(group, send_d, nbytes);
}

// Kernel that performs weighted partition send/recv
// Groups are partitioned according to weights, partition 0 sends, partition 1
// recvs
__global__ void testWeightedSendRecvKernel(
    P2pNvlTransportDevice p2p,
    void* send_d,
    void* recv_d,
    size_t nbytes,
    uint32_t sendWeight,
    uint32_t recvWeight,
    GroupType groupType) {
  auto group = make_group(groupType);
  uint32_t weights[] = {sendWeight, recvWeight};
  auto [partition_id, subgroup] = group.partition(make_device_span(weights, 2));
  if (partition_id == 0) {
    p2p.send(subgroup, send_d, nbytes);
  } else {
    p2p.recv(subgroup, recv_d, nbytes);
  }
}

// Kernel that performs weighted partition recv/send
// Groups are partitioned according to weights, partition 0 recvs, partition 1
// sends
__global__ void testWeightedRecvSendKernel(
    P2pNvlTransportDevice p2p,
    void* recv_d,
    void* send_d,
    size_t nbytes,
    uint32_t recvWeight,
    uint32_t sendWeight,
    GroupType groupType) {
  auto group = make_group(groupType);
  uint32_t weights[] = {recvWeight, sendWeight};
  auto [partition_id, subgroup] = group.partition(make_device_span(weights, 2));
  if (partition_id == 0) {
    p2p.recv(subgroup, recv_d, nbytes);
  } else {
    p2p.send(subgroup, send_d, nbytes);
  }
}

void fillBuffer(int* deviceBuffer, int value, size_t numElements) {
  const int blockSize = 256;
  const int numBlocks = (numElements + blockSize - 1) / blockSize;
  fillBufferKernel<<<numBlocks, blockSize>>>(deviceBuffer, value, numElements);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void verifyBuffer(
    const int* deviceBuffer,
    int expectedValue,
    size_t numElements,
    int* deviceErrorCount) {
  const int blockSize = 256;
  const int numBlocks = (numElements + blockSize - 1) / blockSize;
  verifyBufferKernel<<<numBlocks, blockSize>>>(
      deviceBuffer, expectedValue, numElements, deviceErrorCount);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testSend(
    P2pNvlTransportDevice p2p,
    void* src_d,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    GroupType groupType,
    int /*blocksPerGroup*/) {
  testSendKernel<<<numBlocks, blockSize>>>(p2p, src_d, nbytes, groupType);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testRecv(
    P2pNvlTransportDevice p2p,
    void* dst_d,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    GroupType groupType,
    int /*blocksPerGroup*/) {
  testRecvKernel<<<numBlocks, blockSize>>>(p2p, dst_d, nbytes, groupType);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testMultiSend(
    P2pNvlTransportDevice p2p,
    void* src_d,
    size_t nbytes,
    int numSends,
    int numBlocks,
    int blockSize,
    GroupType groupType,
    int /*blocksPerGroup*/) {
  testMultiSendKernel<<<numBlocks, blockSize>>>(
      p2p, src_d, nbytes, numSends, groupType);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testMultiRecv(
    P2pNvlTransportDevice p2p,
    void* dst_d,
    size_t nbytes,
    int numRecvs,
    int numBlocks,
    int blockSize,
    GroupType groupType,
    int /*blocksPerGroup*/) {
  testMultiRecvKernel<<<numBlocks, blockSize>>>(
      p2p, dst_d, nbytes, numRecvs, groupType);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testSendRecv(
    P2pNvlTransportDevice p2p,
    void* send_d,
    void* recv_d,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    GroupType groupType,
    int /*blocksPerGroup*/) {
  testSendRecvKernel<<<numBlocks, blockSize>>>(
      p2p, send_d, recv_d, nbytes, groupType);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testRecvSend(
    P2pNvlTransportDevice p2p,
    void* recv_d,
    void* send_d,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    GroupType groupType,
    int /*blocksPerGroup*/) {
  testRecvSendKernel<<<numBlocks, blockSize>>>(
      p2p, recv_d, send_d, nbytes, groupType);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testWeightedSendRecv(
    P2pNvlTransportDevice p2p,
    void* send_d,
    void* recv_d,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    uint32_t sendWeight,
    uint32_t recvWeight,
    GroupType groupType) {
  testWeightedSendRecvKernel<<<numBlocks, blockSize>>>(
      p2p, send_d, recv_d, nbytes, sendWeight, recvWeight, groupType);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testWeightedRecvSend(
    P2pNvlTransportDevice p2p,
    void* recv_d,
    void* send_d,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    uint32_t recvWeight,
    uint32_t sendWeight,
    GroupType groupType) {
  testWeightedRecvSendKernel<<<numBlocks, blockSize>>>(
      p2p, recv_d, send_d, nbytes, recvWeight, sendWeight, groupType);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::pipes::test
