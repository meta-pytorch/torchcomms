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

// call_index must be globally unique across ALL send/recv calls, both within
// a kernel and across multiple kernel launches. The caller is responsible for
// maintaining a global counter. See P2pNvlTransportDevice.cuh for details.

void testSend(
    P2pNvlTransportDevice* p2p,
    void* src_d,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    uint32_t call_index,
    GroupType groupType = GroupType::WARP,
    int blocksPerGroup = 1);

void testRecv(
    P2pNvlTransportDevice* p2p,
    void* dst_d,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    uint32_t call_index,
    GroupType groupType = GroupType::WARP,
    int blocksPerGroup = 1);

// Multiple sequential sends within a single kernel
// call_index_base: starting call_index for this kernel (user tracks globally)
void testMultiSend(
    P2pNvlTransportDevice* p2p,
    void* src_d,
    size_t nbytes,
    int numSends,
    int numBlocks,
    int blockSize,
    uint32_t call_index_base,
    GroupType groupType = GroupType::WARP,
    int blocksPerGroup = 1);

// Multiple sequential recvs within a single kernel
// call_index_base: starting call_index for this kernel (user tracks globally)
void testMultiRecv(
    P2pNvlTransportDevice* p2p,
    void* dst_d,
    size_t nbytes,
    int numRecvs,
    int numBlocks,
    int blockSize,
    uint32_t call_index_base,
    GroupType groupType = GroupType::WARP,
    int blocksPerGroup = 1);

// Send then recv within a single kernel (for pipelined bidirectional)
// call_index_base: starting call_index for this kernel (uses 2 indices)
void testSendRecv(
    P2pNvlTransportDevice* p2p,
    void* send_d,
    void* recv_d,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    uint32_t call_index_base,
    GroupType groupType = GroupType::WARP,
    int blocksPerGroup = 1);

// Recv then send within a single kernel (paired with testSendRecv)
// call_index_base: starting call_index for this kernel (uses 2 indices)
void testRecvSend(
    P2pNvlTransportDevice* p2p,
    void* recv_d,
    void* send_d,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    uint32_t call_index_base,
    GroupType groupType = GroupType::WARP,
    int blocksPerGroup = 1);

// Weighted partition send/recv - partitions groups according to weights
// sendWeight:recvWeight controls the ratio of groups assigned to send vs recv
// call_index: user-tracked global call index
void testWeightedSendRecv(
    P2pNvlTransportDevice* p2p,
    void* send_d,
    void* recv_d,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    uint32_t sendWeight,
    uint32_t recvWeight,
    uint32_t call_index,
    GroupType groupType = GroupType::WARP);

// Weighted partition recv/send - partitions groups according to weights
// recvWeight:sendWeight controls the ratio of groups assigned to recv vs send
// call_index: user-tracked global call index
void testWeightedRecvSend(
    P2pNvlTransportDevice* p2p,
    void* recv_d,
    void* send_d,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    uint32_t recvWeight,
    uint32_t sendWeight,
    uint32_t call_index,
    GroupType groupType = GroupType::WARP);

// Test put() - one-sided direct memory write to peer GPU and signal peer
// Unlike send()/recv(), put() writes directly to dst_d without staging buffers
void testPutWithSignal(
    P2pNvlTransportDevice* p2p,
    char* dst_d, // Destination on peer GPU (must be NVLink-accessible)
    const char* src_d, // Source on local GPU
    uint64_t signal_id,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    GroupType groupType = GroupType::WARP);

// Test wait() - one-sided wait for peer to write to dst_d and signal
void testWait(
    P2pNvlTransportDevice* p2p,
    CmpOp op,
    uint64_t signal_id,
    uint64_t expected,
    int numBlocks,
    int blockSize,
    GroupType groupType = GroupType::WARP);

} // namespace comms::pipes::test
