// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

#include "comms/pipes/IbgdaBuffer.h"

namespace comms::pipes {

// Forward declaration - full definition in P2pIbgdaTransportDevice.cuh
class P2pIbgdaTransportDevice;

} // namespace comms::pipes

namespace comms::pipes::test {

/**
 * Test kernel: Put data + signal remote (thread-scope, slot-index)
 *
 * Uses thread-scope put() with slot-index signal to write data and signal
 * completion via the transport's owned signal buffer.
 */
void testPutAndSignal(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Group-collaborative put + signal (warp group, slot-index)
 */
void testPutAndSignalGroup(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Multi-warp group-collaborative put + signal (slot-index)
 */
void testPutAndSignalGroupMultiWarp(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Block-scope group-collaborative put + signal (slot-index)
 */
void testPutAndSignalGroupBlock(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Wait for signal via slot-index on transport's local inbox
 */
void testWaitSignal(
    P2pIbgdaTransportDevice* transport,
    int signalId,
    uint64_t expectedSignal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Multiple put + signal operations in sequence (slot-index)
 */
void testMultiplePutAndSignal(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t bytesPerPut,
    int signalId,
    int numPuts,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Send signal only (no data, slot-index)
 */
void testSignalOnly(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Put data without signal
 */
void testPutOnly(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int numBlocks,
    int blockSize);

/**
 * Fill a device buffer with a pattern based on index
 */
void fillBufferWithPattern(
    void* buffer,
    std::size_t nbytes,
    uint8_t baseValue,
    int numBlocks,
    int blockSize);

/**
 * Verify a device buffer matches expected pattern
 */
void verifyBufferPattern(
    const void* buffer,
    std::size_t nbytes,
    uint8_t expectedBaseValue,
    int* errorCount,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Wait for ready signal, then put data with signal (slot-index)
 */
void testWaitReadyThenPutAndSignal(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int readySignalId,
    uint64_t readySignalVal,
    int dataSignalId,
    uint64_t dataSignalVal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Bidirectional put and wait in single kernel (slot-index)
 */
void testBidirectionalPutAndWait(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int sendSignalId,
    uint64_t sendSignalVal,
    int recvSignalId,
    uint64_t recvSignalVal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: All-to-all send phase (slot-index)
 */
void testAllToAll(
    P2pIbgdaTransportDevice** peerTransports,
    IbgdaLocalBuffer* localSendBufs,
    IbgdaRemoteBuffer* peerRecvBufs,
    int myRank,
    std::size_t nbytes,
    int numPeers,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: All-to-all wait phase (slot-index)
 */
void testAllToAllWait(
    P2pIbgdaTransportDevice** peerTransports,
    int numPeers,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Put data + signal remote + counter (slot-index)
 */
void testPutSignalCounter(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localDataBuf,
    const IbgdaRemoteBuffer& remoteDataBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal,
    int counterId,
    uint64_t counterVal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Wait for local counter to reach expected value (slot-index)
 */
void testWaitCounter(
    P2pIbgdaTransportDevice* transport,
    int counterId,
    uint64_t expectedVal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Multi-QP put + signal with per-block QP selection
 *
 * Each block selects its QP via blockIdx.x % numQps, puts its chunk
 * of totalBytes, then signals. Tests that independent QPs work correctly
 * when blocks use different QPs.
 *
 * @param transports Base pointer to N contiguous P2pIbgdaTransportDevice
 * @param numQps Number of QPs (transports array length)
 * @param localBuf Local source buffer
 * @param remoteBuf Remote destination buffer
 * @param totalBytes Total bytes (split across blocks)
 * @param remoteSignalBuf Remote signal buffer
 * @param signalId Signal slot index
 * @param signalVal Signal value per block
 * @param numBlocks Grid dimension
 * @param blockSize Block dimension
 */
void testMultiQpPutAndSignal(
    P2pIbgdaTransportDevice* transports,
    int numQps,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t totalBytes,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int blockSize);

} // namespace comms::pipes::test
