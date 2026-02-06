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
    MultiPeerDeviceTransport transport,
    int* results);

/**
 * Test kernel: Send data from this rank to a single peer
 *
 * @param transport The MultiPeerDeviceTransport to use
 * @param peerIndex The destination peer index in [0, num_peers())
 * @param srcBuff Source buffer containing data to send
 * @param nbytes Number of bytes to send
 * @param numBlocks Number of thread blocks to launch
 * @param blockSize Threads per block
 */
void testSinglePeerSend(
    MultiPeerDeviceTransport transport,
    int peerIndex,
    void* srcBuff,
    std::size_t nbytes,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Receive data from a single peer to this rank
 *
 * @param transport The MultiPeerDeviceTransport to use
 * @param peerIndex The source peer index in [0, num_peers())
 * @param dstBuff Destination buffer for received data
 * @param nbytes Number of bytes to receive
 * @param numBlocks Number of thread blocks to launch
 * @param blockSize Threads per block
 */
void testSinglePeerRecv(
    MultiPeerDeviceTransport transport,
    int peerIndex,
    void* dstBuff,
    std::size_t nbytes,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Peer iteration using numPeers() and peerIndexToRank()
 *
 * Sends data to all peers using the peer iteration helpers.
 *
 * @param transport The MultiPeerDeviceTransport to use
 * @param srcBuffs Array of source buffers, one per peer
 * @param nbytesPerPeer Number of bytes to send to each peer
 * @param numBlocks Number of thread blocks to launch
 * @param blockSize Threads per block
 */
void testMultiPeerSendAllPeers(
    MultiPeerDeviceTransport transport,
    void** srcBuffs,
    std::size_t nbytesPerPeer,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Receive from all peers using peer iteration helpers
 *
 * @param transport The MultiPeerDeviceTransport to use
 * @param dstBuffs Array of destination buffers, one per peer
 * @param nbytesPerPeer Number of bytes to receive from each peer
 * @param numBlocks Number of thread blocks to launch
 * @param blockSize Threads per block
 */
void testMultiPeerRecvAllPeers(
    MultiPeerDeviceTransport transport,
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
