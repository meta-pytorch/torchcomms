// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

#include "comms/pipes/MultiPeerDeviceTransport.cuh"

namespace comms::pipes::test {

/**
 * Test kernel: Verify MultiPeerDeviceTransport accessors on device
 *
 * @param transport The MultiPeerDeviceTransport to test
 * @param results Output array: [0]=rank, [1]=nRanks, [2]=numPeers
 */
void testMultiPeerDeviceTransportAccessors(
    MultiPeerDeviceTransport& transport,
    int* results);

/**
 * Test kernel: Send data from this rank to a single peer
 *
 * @param transport The MultiPeerDeviceTransport to use
 * @param targetRank The destination rank
 * @param srcBuff Source buffer containing data to send
 * @param nbytes Number of bytes to send
 * @param numBlocks Number of thread blocks to launch
 * @param blockSize Threads per block
 */
void testSinglePeerSend(
    MultiPeerDeviceTransport& transport,
    int targetRank,
    void* srcBuff,
    std::size_t nbytes,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Receive data from a single peer to this rank
 *
 * @param transport The MultiPeerDeviceTransport to use
 * @param targetRank The source rank
 * @param dstBuff Destination buffer for received data
 * @param nbytes Number of bytes to receive
 * @param numBlocks Number of thread blocks to launch
 * @param blockSize Threads per block
 */
void testSinglePeerRecv(
    MultiPeerDeviceTransport& transport,
    int targetRank,
    void* dstBuff,
    std::size_t nbytes,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Parallel send/recv to all peers using partition
 *
 * Uses partition_interleaved to split warps between send and recv work,
 * then further partitions across peers. This avoids deadlocks that can occur
 * when send and recv are done sequentially.
 *
 * @param transport The MultiPeerDeviceTransport to use
 * @param srcBuffs Array of source buffers, one per peer
 * @param dstBuffs Array of destination buffers, one per peer
 * @param nbytesPerPeer Number of bytes to transfer per peer
 * @param numBlocks Number of thread blocks to launch
 * @param blockSize Threads per block
 */
void testMultiPeerSendRecvAllPeers(
    MultiPeerDeviceTransport& transport,
    void** srcBuffs,
    void** dstBuffs,
    std::size_t nbytesPerPeer,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Verify transport type accessors
 *
 * Checks that get_peer_transport() and get_self_transport() return correct
 * transport types for self vs peer.
 *
 * @param transport The MultiPeerDeviceTransport to use
 * @param results Output array: [0]=numPeers, [1..nRanks]=transport types
 */
void testTransportTypes(
    const MultiPeerDeviceTransport& transport,
    int* results);

} // namespace comms::pipes::test
