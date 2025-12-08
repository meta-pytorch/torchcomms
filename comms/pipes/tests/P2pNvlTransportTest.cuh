// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstddef>

namespace comms::pipes::test {

// Enum for specifying the thread group type
enum class GroupType {
  WARP, // 32-thread warp groups
  BLOCK // Full block groups (all threads in block)
};

// Kernel to fill buffer with a specific value
// Each thread writes one integer
__global__ void fillBufferKernel(int* buffer, int value, size_t numElements);

// Kernel to verify buffer contents
// Each thread checks one integer and atomically increments error counter if
// mismatch
__global__ void verifyBufferKernel(
    const int* buffer,
    int expectedValue,
    size_t numElements,
    int* errorCount);

// Host wrapper functions
void fillBuffer(int* deviceBuffer, int value, size_t numElements);
void verifyBuffer(
    const int* deviceBuffer,
    int expectedValue,
    size_t numElements,
    int* deviceErrorCount);

void testSend(
    P2pNvlTransportDevice p2p,
    void* src_d,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    GroupType groupType = GroupType::WARP,
    int blocksPerGroup = 1);

void testRecv(
    P2pNvlTransportDevice p2p,
    void* dst_d,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    GroupType groupType = GroupType::WARP,
    int blocksPerGroup = 1);

// Multiple sequential sends within a single kernel
void testMultiSend(
    P2pNvlTransportDevice p2p,
    void* src_d,
    size_t nbytes,
    int numSends,
    int numBlocks,
    int blockSize,
    GroupType groupType = GroupType::WARP,
    int blocksPerGroup = 1);

// Multiple sequential recvs within a single kernel
void testMultiRecv(
    P2pNvlTransportDevice p2p,
    void* dst_d,
    size_t nbytes,
    int numRecvs,
    int numBlocks,
    int blockSize,
    GroupType groupType = GroupType::WARP,
    int blocksPerGroup = 1);

// Send then recv within a single kernel (for pipelined bidirectional)
void testSendRecv(
    P2pNvlTransportDevice p2p,
    void* send_d,
    void* recv_d,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    GroupType groupType = GroupType::WARP,
    int blocksPerGroup = 1);

// Recv then send within a single kernel (paired with testSendRecv)
void testRecvSend(
    P2pNvlTransportDevice p2p,
    void* recv_d,
    void* send_d,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    GroupType groupType = GroupType::WARP,
    int blocksPerGroup = 1);

} // namespace comms::pipes::test
