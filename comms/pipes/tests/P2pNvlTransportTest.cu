// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/pipes/tests/Checks.h"
#include "comms/pipes/tests/P2pNvlTransportTest.cuh"

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

// =============================================================================
// write() test kernel and wrapper
// =============================================================================

__global__ void testWriteWithSignalKernel(
    P2pNvlTransportDevice p2p,
    char* dst_d,
    const char* src_d,
    uint64_t signal_id,
    size_t nbytes,
    GroupType groupType) {
  auto group = make_group(groupType);
  auto writtenBytes = p2p.write(group, dst_d, src_d, nbytes);
  p2p.signal_threadgroup(group, signal_id, SignalOp::SIGNAL_ADD, writtenBytes);
}

void testWriteWithSignal(
    P2pNvlTransportDevice p2p,
    char* dst_d,
    const char* src_d,
    uint64_t signal_id,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    GroupType groupType) {
  testWriteWithSignalKernel<<<numBlocks, blockSize>>>(
      p2p, dst_d, src_d, signal_id, nbytes, groupType);
  PIPES_KERNEL_LAUNCH_CHECK();
}

// =============================================================================
// wait() test kernel and wrapper
// =============================================================================

__global__ void testWaitKernel(
    P2pNvlTransportDevice p2p,
    CmpOp op,
    uint64_t signal_id,
    uint64_t expected,
    GroupType groupType) {
  auto group = make_group(groupType);
  p2p.wait_signal_until_threadgroup(group, signal_id, op, expected);
}

void testWait(
    P2pNvlTransportDevice p2p,
    CmpOp op,
    uint64_t signal_id,
    uint64_t expected,
    int numBlocks,
    int blockSize,
    GroupType groupType) {
  testWaitKernel<<<numBlocks, blockSize>>>(
      p2p, op, signal_id, expected, groupType);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::pipes::test
