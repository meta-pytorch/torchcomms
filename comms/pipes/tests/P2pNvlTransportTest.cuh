// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstddef>

#include "comms/pipes/P2pNvlTransportDevice.cuh"

namespace comms::pipes::test {

using comms::pipes::P2pNvlTransportDevice;

// Enum for specifying the thread group type
enum class GroupType {
  WARP, // 32-thread warp groups
  BLOCK // Full block groups (all threads in block)
};

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

// Weighted partition send/recv - partitions groups according to weights
// sendWeight:recvWeight controls the ratio of groups assigned to send vs recv
void testWeightedSendRecv(
    P2pNvlTransportDevice p2p,
    void* send_d,
    void* recv_d,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    uint32_t sendWeight,
    uint32_t recvWeight,
    GroupType groupType = GroupType::WARP);

// Weighted partition recv/send - partitions groups according to weights
// recvWeight:sendWeight controls the ratio of groups assigned to recv vs send
void testWeightedRecvSend(
    P2pNvlTransportDevice p2p,
    void* recv_d,
    void* send_d,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    uint32_t recvWeight,
    uint32_t sendWeight,
    GroupType groupType = GroupType::WARP);

} // namespace comms::pipes::test
