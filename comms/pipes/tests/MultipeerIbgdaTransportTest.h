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
 * Test kernel: Put data + signal remote (adaptive-routing safe)
 *
 * Uses put() followed by signal_remote_with_fence() to write data and signal
 * completion to the peer's signal buffer. The NIC fence ensures data is
 * committed before the signal arrives. wait_local() is used for each
 * operation's local completion.
 *
 * @param deviceTransportPtr Pointer to P2pIbgdaTransportDevice in device memory
 * @param localBuf Local source buffer (with lkey)
 * @param remoteBuf Remote destination buffer (with rkey)
 * @param nbytes Number of bytes to transfer
 * @param remoteSignalBuf Remote signal buffer (with rkey)
 * @param signalId Signal slot index
 * @param signalVal Signal value to send
 */
void testPutAndSignal(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Group-collaborative put + signal (warp group)
 *
 * Uses put_group_local() to partition data across warp lanes, then leader
 * calls signal_remote_with_fence().
 *
 * @param deviceTransportPtr Pointer to P2pIbgdaTransportDevice in device
 * memory
 * @param localBuf Local source buffer (with lkey)
 * @param remoteBuf Remote destination buffer (with rkey)
 * @param nbytes Total bytes to transfer (split across group lanes)
 * @param remoteSignalBuf Remote signal buffer (with rkey)
 * @param signalId Signal slot index
 * @param signalVal Signal value to send
 */
void testPutAndSignalGroup(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Multi-warp group-collaborative put + signal
 *
 * Multiple warps call put_group_global(), each leader signals with
 * signalVal. Total accumulated signal is (numWarps * signalVal).
 *
 * @param deviceTransportPtr Pointer to P2pIbgdaTransportDevice in device
 * memory
 * @param localBuf Local source buffer (with lkey)
 * @param remoteBuf Remote destination buffer (with rkey)
 * @param nbytes Total bytes to transfer (split across warps)
 * @param remoteSignalBuf Remote signal buffer (with rkey)
 * @param signalId Signal slot index
 * @param signalVal Signal value per warp
 */
void testPutAndSignalGroupMultiWarp(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Block-scope group-collaborative put + signal
 *
 * Multiple blocks call put_group_global(), each leader signals with
 * signalVal. Total accumulated signal is (numBlocks * signalVal).
 *
 * @param deviceTransportPtr Pointer to P2pIbgdaTransportDevice in device
 * memory
 * @param localBuf Local source buffer (with lkey)
 * @param remoteBuf Remote destination buffer (with rkey)
 * @param nbytes Total bytes to transfer (split across blocks)
 * @param remoteSignalBuf Remote signal buffer (with rkey)
 * @param signalId Signal slot index
 * @param signalVal Signal value per block
 */
void testPutAndSignalGroupBlock(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Wait for signal via volatile spin on local signal buffer
 *
 * Spins on localSignalBuf[signalId] until the value is >= expectedSignal.
 *
 * @param localSignalBuf Local signal buffer pointer (GPU memory)
 * @param signalId Signal slot index
 * @param expectedSignal Signal value to wait for (GE comparison)
 */
void testWaitSignal(
    uint64_t* localSignalBuf,
    int signalId,
    uint64_t expectedSignal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Multiple put + signal operations in sequence
 *
 * Each iteration: put one chunk + signal with value 1. Total signal
 * after numPuts iterations = numPuts.
 *
 * @param deviceTransportPtr Pointer to P2pIbgdaTransportDevice in device
 * memory
 * @param localBuf Local source buffer
 * @param remoteBuf Remote destination buffer
 * @param bytesPerPut Bytes per put operation
 * @param remoteSignalBuf Remote signal buffer (with rkey)
 * @param signalId Signal slot index
 * @param numPuts Number of put operations
 */
void testMultiplePutAndSignal(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t bytesPerPut,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int signalId,
    int numPuts,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Send signal only (no data)
 *
 * Sends an RDMA atomic fetch-add to the remote peer's signal buffer.
 *
 * @param deviceTransportPtr Pointer to P2pIbgdaTransportDevice in device
 * memory
 * @param remoteSignalBuf Remote signal buffer (with rkey)
 * @param signalId Signal slot index
 * @param signalVal Signal value to send
 */
void testSignalOnly(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Put data without signal
 *
 * Performs an RDMA write without signaling.
 *
 * @param deviceTransportPtr Pointer to P2pIbgdaTransportDevice in device
 * memory
 * @param localBuf Local source buffer
 * @param remoteBuf Remote destination buffer
 * @param nbytes Number of bytes to transfer
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
 *
 * Each byte is set to (baseValue + (index % 256))
 *
 * @param buffer Device buffer pointer
 * @param nbytes Number of bytes to fill
 * @param baseValue Base value for the pattern
 */
void fillBufferWithPattern(
    void* buffer,
    std::size_t nbytes,
    uint8_t baseValue,
    int numBlocks,
    int blockSize);

/**
 * Verify a device buffer matches expected pattern
 *
 * Returns the count of mismatched bytes.
 *
 * @param buffer Device buffer pointer
 * @param nbytes Number of bytes to verify
 * @param expectedBaseValue Expected base value of pattern
 * @param errorCount Output: number of errors found (device pointer)
 */
void verifyBufferPattern(
    const void* buffer,
    std::size_t nbytes,
    uint8_t expectedBaseValue,
    int* errorCount,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Reset a remote signal slot to zero
 *
 * Uses RDMA inline write to set the remote signal to zero.
 *
 * @param deviceTransportPtr Pointer to P2pIbgdaTransportDevice in device
 * memory
 * @param remoteSignalBuf Remote signal buffer (with rkey)
 * @param signalId Signal slot index
 */
void testResetSignal(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int signalId,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Wait for ready signal, then put data with signal
 *
 * Sender waits for the receiver's ready signal (volatile spin on local
 * signal buffer), then performs put + signal_remote_with_fence.
 *
 * @param deviceTransportPtr Pointer to P2pIbgdaTransportDevice in device
 * memory
 * @param localBuf Local source buffer (with lkey)
 * @param remoteBuf Remote destination buffer (with rkey)
 * @param nbytes Number of bytes to transfer
 * @param localSignalBuf Local signal buffer for waiting (GPU memory)
 * @param readySignalId Signal slot index to wait on for ready
 * @param readySignalVal Signal value to wait for indicating ready
 * @param remoteSignalBuf Remote signal buffer (with rkey)
 * @param dataSignalId Signal slot index to signal data completion
 * @param dataSignalVal Signal value to send with data
 */
void testWaitReadyThenPutAndSignal(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    uint64_t* localSignalBuf,
    int readySignalId,
    uint64_t readySignalVal,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int dataSignalId,
    uint64_t dataSignalVal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Bidirectional put and wait in single kernel
 *
 * Thread 0: put + signal_remote_with_fence (send)
 * Thread 1: volatile spin wait (receive)
 *
 * @param deviceTransportPtr Pointer to P2pIbgdaTransportDevice in device
 * memory
 * @param localBuf Local source buffer for sending
 * @param remoteBuf Remote destination buffer
 * @param nbytes Number of bytes to transfer
 * @param remoteSignalBuf Remote signal buffer for sending signals
 * @param sendSignalId Signal slot index for sending
 * @param sendSignalVal Signal value to send
 * @param localSignalBuf Local signal buffer for receiving (GPU memory)
 * @param recvSignalId Signal slot index to wait on for receiving
 * @param recvSignalVal Signal value to wait for
 */
void testBidirectionalPutAndWait(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int sendSignalId,
    uint64_t sendSignalVal,
    uint64_t* localSignalBuf,
    int recvSignalId,
    uint64_t recvSignalVal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: All-to-all send phase
 *
 * Uses partition() to parallelize sends across peers.
 * Each peer gets put + signal_remote_with_fence.
 *
 * @param peerTransports Array of transport pointers (device memory)
 * @param localSendBufs Array of local send buffers (device memory)
 * @param peerRecvBufs Array of remote receive buffers (device memory)
 * @param remoteSignalBufs Array of remote signal buffers (device memory)
 * @param myRank This rank's ID (used as signal ID)
 * @param nbytes Number of bytes to transfer per peer
 * @param numPeers Number of peers (nRanks - 1)
 */
void testAllToAll(
    P2pIbgdaTransportDevice** peerTransports,
    IbgdaLocalBuffer* localSendBufs,
    IbgdaRemoteBuffer* peerRecvBufs,
    IbgdaRemoteBuffer* remoteSignalBufs,
    int myRank,
    std::size_t nbytes,
    int numPeers,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: All-to-all wait phase
 *
 * Waits for signals from all peers via volatile spin on local signal
 * buffer.
 *
 * @param localSignalBuf Local signal buffer (GPU memory)
 * @param peerRanks Array of peer ranks (device memory)
 * @param numPeers Number of peers
 */
void testAllToAllWait(
    uint64_t* localSignalBuf,
    int* peerRanks,
    int numPeers,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Put data + signal remote + counter via companion QP
 *
 * Uses put_signal_counter_remote() to write data, signal the remote peer,
 * and increment a local counter via companion QP loopback.
 */
void testPutSignalCounter(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localDataBuf,
    const IbgdaRemoteBuffer& remoteDataBuf,
    std::size_t nbytes,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int signalId,
    uint64_t signalVal,
    const IbgdaLocalBuffer& localCounterBuf,
    int counterId,
    uint64_t counterVal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Wait for local counter to reach expected value
 *
 * GPU thread spins on volatile counter until it reaches expectedVal.
 */
void testWaitCounter(
    uint64_t* counterBuf,
    int counterId,
    uint64_t expectedVal,
    int numBlocks,
    int blockSize);

} // namespace comms::pipes::test
