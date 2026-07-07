// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstddef>

#include "comms/prims/core/TiledBuffer.cuh"
#include "comms/prims/transport/nvl/P2pNvlTransportDevice.cuh"

namespace comms::prims::test {

using comms::prims::P2pNvlTransportDevice;

// Enum for specifying the thread group type
enum class GroupType {
  WARP, // 32-thread warp groups
  BLOCK // Full block groups (all threads in block)
};

void testSend(
    P2pNvlTransportDevice* p2p,
    void* src_d,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    GroupType groupType = GroupType::WARP,
    int blocksPerGroup = 1,
    cudaStream_t stream = nullptr);

void testRecv(
    P2pNvlTransportDevice* p2p,
    void* dst_d,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    GroupType groupType = GroupType::WARP,
    int blocksPerGroup = 1,
    cudaStream_t stream = nullptr);

void testTileSend(
    const P2pNvlTransportDevice& p2p,
    void* src_d,
    size_t nbytes,
    size_t maxSignalBytes,
    Timeout timeout,
    int numBlocks,
    int blockSize,
    cudaStream_t stream = nullptr);

void testTileRecv(
    const P2pNvlTransportDevice& p2p,
    void* dst_d,
    size_t nbytes,
    size_t maxSignalBytes,
    Timeout timeout,
    int numBlocks,
    int blockSize,
    cudaStream_t stream = nullptr);

// Multiple sequential sends within a single kernel
void testMultiSend(
    P2pNvlTransportDevice* p2p,
    void* src_d,
    size_t nbytes,
    int numSends,
    int numBlocks,
    int blockSize,
    GroupType groupType = GroupType::WARP,
    int blocksPerGroup = 1);

// Multiple sequential recvs within a single kernel
void testMultiRecv(
    P2pNvlTransportDevice* p2p,
    void* dst_d,
    size_t nbytes,
    int numRecvs,
    int numBlocks,
    int blockSize,
    GroupType groupType = GroupType::WARP,
    int blocksPerGroup = 1);

// Send then recv within a single kernel (for pipelined bidirectional)
void testSendRecv(
    P2pNvlTransportDevice* p2p,
    void* send_d,
    void* recv_d,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    GroupType groupType = GroupType::WARP,
    int blocksPerGroup = 1);

// Recv then send within a single kernel (paired with testSendRecv)
void testRecvSend(
    P2pNvlTransportDevice* p2p,
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
    P2pNvlTransportDevice* p2p,
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
    P2pNvlTransportDevice* p2p,
    void* recv_d,
    void* send_d,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    uint32_t recvWeight,
    uint32_t sendWeight,
    GroupType groupType = GroupType::WARP);

void testTileMultiCallSendRecv(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> sendTiles,
    TiledBuffer<char> recvTiles,
    int activeBlocks,
    int numCalls,
    size_t bytesPerCall,
    size_t maxSignalBytes,
    bool waitForSecondCallSignal,
    int blockSize,
    cudaStream_t stream = nullptr);

void testTileTwoCallVariableSignalSendRecv(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> sendTiles,
    TiledBuffer<char> recvTiles,
    int activeBlocks,
    size_t firstCallBytes,
    size_t secondCallBytes,
    size_t firstMaxSignalBytes,
    size_t secondMaxSignalBytes,
    bool waitForSecondCallSignal,
    int blockSize,
    Timeout timeout = Timeout(),
    cudaStream_t stream = nullptr);

void testTileMultiCallSendOnly(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> sendTiles,
    int activeBlocks,
    int numCalls,
    size_t bytesPerCall,
    size_t maxSignalBytes,
    int blockSize,
    cudaStream_t stream = nullptr);

void testTileTwoCallSendOnly(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> sendTiles,
    int activeBlocks,
    size_t firstCallBytes,
    size_t secondCallBytes,
    size_t maxSignalBytes,
    int blockSize,
    cudaStream_t stream = nullptr);

void testTileSendWaitsForWrappedSubstepAck(
    P2pNvlTransportDevice p2p,
    const char* sendData,
    size_t nbytes,
    size_t maxSignalBytes,
    unsigned char sentinel,
    int* observedEarlyOverwrite,
    int blockSize,
    cudaStream_t stream = nullptr);

void testTileForwardWaitsForWrappedSubstepAck(
    P2pNvlTransportDevice pred,
    P2pNvlTransportDevice succ,
    char* dst,
    size_t nbytes,
    size_t maxSignalBytes,
    unsigned char sentinel,
    int* observedEarlyOverwrite,
    int blockSize,
    cudaStream_t stream = nullptr);

void testPrepareTileStaging(
    P2pNvlTransportDevice p2p,
    int activeBlocks,
    int numCalls,
    size_t bytesPerCall,
    size_t maxSignalBytes,
    int sourceRank,
    int blockSize,
    cudaStream_t stream = nullptr);

void testPrepareTileTwoCallStaging(
    P2pNvlTransportDevice p2p,
    int activeBlocks,
    size_t firstCallBytes,
    size_t secondCallBytes,
    size_t maxSignalBytes,
    int sourceRank,
    int blockSize,
    cudaStream_t stream = nullptr);

void testTileMultiCallRecvOnly(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> recvTiles,
    int activeBlocks,
    int numCalls,
    size_t bytesPerCall,
    size_t maxSignalBytes,
    int blockSize,
    cudaStream_t stream = nullptr);

void testTileTwoCallRecvOnly(
    P2pNvlTransportDevice p2p,
    TiledBuffer<char> recvTiles,
    int activeBlocks,
    size_t firstCallBytes,
    size_t secondCallBytes,
    size_t maxSignalBytes,
    int blockSize,
    cudaStream_t stream = nullptr);

void testTileMultiCallForward(
    P2pNvlTransportDevice pred,
    P2pNvlTransportDevice succ,
    TiledBuffer<char> dstTiles,
    int activeBlocks,
    int numCalls,
    size_t bytesPerCall,
    size_t maxSignalBytes,
    bool waitForSecondCallSignal,
    int blockSize,
    cudaStream_t stream = nullptr);

void testTileTwoCallForward(
    P2pNvlTransportDevice pred,
    P2pNvlTransportDevice succ,
    TiledBuffer<char> dstTiles,
    int activeBlocks,
    size_t firstCallBytes,
    size_t secondCallBytes,
    size_t maxSignalBytes,
    int blockSize,
    cudaStream_t stream = nullptr);

void testTileTwoCallVariableSignalForward(
    P2pNvlTransportDevice pred,
    P2pNvlTransportDevice succ,
    TiledBuffer<char> dstTiles,
    int activeBlocks,
    size_t firstCallBytes,
    size_t secondCallBytes,
    size_t firstMaxSignalBytes,
    size_t secondMaxSignalBytes,
    int blockSize,
    cudaStream_t stream = nullptr);

void testCopyLocalStaging(
    P2pNvlTransportDevice p2p,
    void* dst,
    size_t nbytes,
    int blockSize,
    cudaStream_t stream = nullptr);

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

// forward_group: fused recv-and-forward (reads from predecessor staging buffer,
// writes to local dst and successor's remote staging buffer simultaneously)
void testForward(
    P2pNvlTransportDevice* pred,
    P2pNvlTransportDevice* succ,
    void* dst_d,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    GroupType groupType = GroupType::WARP,
    cudaStream_t stream = nullptr);

// Test wait() - one-sided wait for peer to write to dst_d and signal
void testWait(
    P2pNvlTransportDevice* p2p,
    CmpOp op,
    uint64_t signal_id,
    uint64_t expected,
    int numBlocks,
    int blockSize,
    GroupType groupType = GroupType::WARP);

} // namespace comms::prims::test
