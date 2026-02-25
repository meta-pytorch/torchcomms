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
 * Test kernel: Group-collaborative put_signal (adaptive-routing safe)
 *
 * Uses the public put_signal_group() API with a warp-sized ThreadGroup.
 * The group partitions the data across lanes and the leader issues
 * a fenced signal.
 *
 * @param deviceTransportPtr Pointer to P2pIbgdaTransportDevice in device memory
 * @param localBuf Local source buffer (with lkey)
 * @param remoteBuf Remote destination buffer (with rkey)
 * @param nbytes Total bytes to transfer (split across group lanes)
 * @param signalId Signal slot index
 * @param signalVal Signal value to send
 */
void testPutSignalGroup(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Multi-warp group-collaborative put_signal
 *
 * Multiple warps each independently call put_signal_group() on their own
 * chunk of data. Each warp signals with signalVal, so the total accumulated
 * signal is (numWarps * signalVal) via atomic fetch-add.
 *
 * @param deviceTransportPtr Pointer to P2pIbgdaTransportDevice in device memory
 * @param localBuf Local source buffer (with lkey)
 * @param remoteBuf Remote destination buffer (with rkey)
 * @param nbytes Total bytes to transfer (split across warps)
 * @param signalId Signal slot index
 * @param signalVal Signal value per warp
 */
void testPutSignalGroupMultiWarp(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Block-scope group-collaborative put_signal
 *
 * Multiple blocks each independently call put_signal_group() with
 * block-scope groups on their own chunk of data. Each block signals
 * with signalVal, so the total accumulated signal is
 * (numBlocks * signalVal) via atomic fetch-add.
 *
 * @param deviceTransportPtr Pointer to P2pIbgdaTransportDevice in device memory
 * @param localBuf Local source buffer (with lkey)
 * @param remoteBuf Remote destination buffer (with rkey)
 * @param nbytes Total bytes to transfer (split across blocks)
 * @param signalId Signal slot index
 * @param signalVal Signal value per block
 */
void testPutSignalGroupBlock(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Send data via put_signal_non_adaptive
 *
 * Uses the fused put+signal operation (single compound WQE) instead of
 * the split put + signal used by testPutSignal. Faster but ordering
 * depends on NIC (not safe with adaptive routing).
 *
 * @param deviceTransportPtr Pointer to P2pIbgdaTransportDevice in device memory
 * @param localBuf Local source buffer (with lkey)
 * @param remoteBuf Remote destination buffer (with rkey)
 * @param nbytes Number of bytes to transfer
 * @param signalId Signal slot index
 * @param signalVal Signal value to send
 */
void testPutSignalNonAdaptive(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Send data via put_signal
 *
 * Sender fills local buffer with pattern and uses put_signal to transfer
 * data to remote peer, signaling completion.
 *
 * @param deviceTransportPtr Pointer to P2pIbgdaTransportDevice in device memory
 * @param localBuf Local source buffer (with lkey)
 * @param remoteBuf Remote destination buffer (with rkey)
 * @param nbytes Number of bytes to transfer
 * @param signalId Signal slot index
 * @param signalVal Signal value to send
 */
void testPutSignal(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Wait for signal
 *
 * Receiver waits for the signal value to arrive, indicating data is ready.
 *
 * @param deviceTransportPtr Pointer to P2pIbgdaTransportDevice in device memory
 * @param signalId Signal slot index
 * @param cmp Comparison operation
 * @param expectedSignal Signal value to wait for
 */
void testWaitSignal(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    int signalId,
    IbgdaCmpOp cmp,
    uint64_t expectedSignal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Multiple put_signal operations in sequence
 *
 * Performs multiple put_signal operations, each with a unique signal value.
 *
 * @param deviceTransportPtr Pointer to P2pIbgdaTransportDevice in device memory
 * @param localBuf Local source buffer
 * @param remoteBuf Remote destination buffer
 * @param bytesPerPut Bytes per put operation
 * @param signalId Signal slot index
 * @param numPuts Number of put operations
 */
void testMultiplePutSignal(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t bytesPerPut,
    int signalId,
    int numPuts,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Send signal only (no data)
 *
 * Sends an atomic signal to the remote peer without any data transfer.
 * Useful for pure synchronization scenarios.
 *
 * @param deviceTransportPtr Pointer to P2pIbgdaTransportDevice in device memory
 * @param signalId Signal slot index
 * @param signalVal Signal value to send
 */
void testSignalOnly(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    int signalId,
    uint64_t signalVal,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Reset remote signal to zero
 *
 * Resets the remote peer's signal buffer at the specified slot to zero.
 * Used to prepare for the next iteration of signaling.
 *
 * @param deviceTransportPtr Pointer to P2pIbgdaTransportDevice in device memory
 * @param signalId Signal slot index
 */
void testResetSignal(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    int signalId,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Put data without signal
 *
 * Performs an RDMA write from local buffer to remote buffer without
 * any signaling. Caller must use other means to synchronize.
 *
 * @param deviceTransportPtr Pointer to P2pIbgdaTransportDevice in device memory
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
 * Test kernel: Read current signal value
 *
 * Non-blocking read of the local signal buffer value.
 *
 * @param deviceTransportPtr Pointer to P2pIbgdaTransportDevice in device memory
 * @param signalId Signal slot index
 * @param d_result Output: signal value read (device pointer)
 */
void testReadSignal(
    P2pIbgdaTransportDevice* deviceTransportPtr,
    int signalId,
    uint64_t* d_result,
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
 * Test kernel: Wait for ready signal, then put data with signal
 *
 * Sender waits for the receiver to signal that its buffer is ready,
 * then performs put_signal to transfer data.
 *
 * @param deviceTransportPtr Pointer to P2pIbgdaTransportDevice in device memory
 * @param localBuf Local source buffer (with lkey)
 * @param remoteBuf Remote destination buffer (with rkey)
 * @param nbytes Number of bytes to transfer
 * @param readySignalId Signal slot index to wait on for ready
 * @param readySignalVal Signal value to wait for indicating ready
 * @param dataSignalId Signal slot index to signal data completion
 * @param dataSignalVal Signal value to send with data
 */
void testWaitReadyThenPutSignal(
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
 * Test kernel: Bidirectional put and wait in single kernel
 *
 * Launches a kernel where thread 0 does put_signal (send) and
 * thread 1 does wait_signal (receive), enabling concurrent
 * bidirectional communication.
 *
 * @param deviceTransportPtr Pointer to P2pIbgdaTransportDevice in device memory
 * @param localBuf Local source buffer for sending
 * @param remoteBuf Remote destination buffer
 * @param nbytes Number of bytes to transfer
 * @param sendSignalId Signal slot index for sending
 * @param sendSignalVal Signal value to send
 * @param recvSignalId Signal slot index to wait on for receiving
 * @param recvSignalVal Signal value to wait for
 */
void testBidirectionalPutWait(
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
 * Test kernel: All-to-all communication pattern
 *
 * Launches a kernel that uses partition() to parallelize communication
 * across multiple peers. Each peer gets a subset of thread groups, and
 * within each peer's groups, threads are divided into senders and receivers.
 *
 * This tests the key multi-peer bidirectional pattern:
 *   [peer_id, per_peer_group] = group.partition(numPeers);
 *
 * @param peerTransports Array of transport pointers (one per peer) in device
 * memory
 * @param localSendBufs Array of local send buffers (one per peer) in device
 * memory
 * @param peerRecvBufs Array of remote receive buffers (one per peer) in device
 * memory
 * @param nbytes Number of bytes to transfer per peer
 * @param numPeers Number of peers (nRanks - 1)
 */
void testAllToAll(
    P2pIbgdaTransportDevice** peerTransports,
    IbgdaLocalBuffer* localSendBufs,
    IbgdaRemoteBuffer* peerRecvBufs,
    int* peerRanks,
    int myRank,
    std::size_t nbytes,
    int numPeers,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: All-to-all wait phase
 *
 * Waits for signals from all peers after the send phase completes.
 * Call this after testAllToAll and an MPI barrier.
 */
void testAllToAllWait(
    P2pIbgdaTransportDevice** peerTransports,
    int* peerRanks,
    int numPeers,
    int numBlocks,
    int blockSize);

} // namespace comms::pipes::test
