// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>

namespace comms::pipes::test {

/**
 * Test kernel: Verify DeviceWindowSignal construction and basic accessors
 *
 * @param myRank Rank ID for the signal object
 * @param nRanks Total number of ranks
 * @param signalCount Number of signal slots per peer
 * @param results Output array for test results [0]=rank, [1]=nRanks,
 * [2]=signalCount
 */
void testDeviceWindowSignalConstruction(
    int myRank,
    int nRanks,
    int signalCount,
    int* results);

/**
 * Test kernel: Verify DeviceWindowBarrier construction and basic accessors
 *
 * @param myRank Rank ID for the barrier object
 * @param nRanks Total number of ranks
 * @param results Output array for test results [0]=rank, [1]=nRanks
 */
void testDeviceWindowBarrierConstruction(int myRank, int nRanks, int* results);

/**
 * Test kernel: Verify MultiPeerDeviceTransport construction and basic accessors
 *
 * @param myRank Rank ID for the transport object
 * @param nRanks Total number of ranks
 * @param results Output array for test results [0]=rank, [1]=nRanks
 */
void testMultiPeerDeviceTransportConstruction(
    int myRank,
    int nRanks,
    int* results);

/**
 * Test kernel: Verify self-transport put() operation via get_self_transport()
 *
 * Tests that MultiPeerDeviceTransport::get_self_transport() returns a
 * valid self-transport and that put() correctly copies data.
 *
 * @param transport_d Device pointer to Transport object
 * @param dst_d Destination buffer (device memory)
 * @param src_d Source buffer (device memory)
 * @param nbytes Number of bytes to copy
 * @param numBlocks Number of blocks to launch
 * @param blockSize Threads per block
 */
void testSelfTransportPut(
    void* transport_d,
    char* dst_d,
    const char* src_d,
    std::size_t nbytes,
    int numBlocks,
    int blockSize);

/**
 * Test kernel: Verify transport returns correct transport type
 *
 * @param transport_d Device pointer to Transport object (self-transport)
 * @param results Output: [0]=1 if SELF type, 0 otherwise
 */
void testGetTransportType(void* transport_d, int* results);

/**
 * Test kernel: Verify peer iteration helpers (numPeers, peerIndexToRank)
 *
 * @param myRank Rank ID for the transport object
 * @param nRanks Total number of ranks
 * @param results Output array for test results:
 *   [0]=numPeers
 *   [1..numPeers]=peerIndexToRank for each index
 */
void testPeerIterationHelpers(int myRank, int nRanks, int* results);

/**
 * Test kernel: Verify peer index conversion roundtrip and transport accessors
 *
 * Tests rank_to_peer_index(), roundtrip identity properties, and
 * get_self_transport()/get_peer_transport() type correctness.
 *
 * @param myRank Rank ID for the transport object
 * @param nRanks Total number of ranks
 * @param results Output array for test results (size = 4*numPeers + 2)
 */
void testPeerIndexConversionRoundtrip(int myRank, int nRanks, int* results);

/**
 * Test kernel: Verify DeviceWindowMemory accessors return correct metadata
 *
 * Constructs DeviceWindowMemory from DeviceWindowSignal + DeviceWindowBarrier,
 * then verifies signal() and barrier() return objects with matching metadata.
 *
 * @param myRank Rank ID
 * @param nRanks Total number of ranks
 * @param signalCount Number of signal slots
 * @param results Output array: [0]=signal.rank, [1]=signal.nRanks,
 *   [2]=signal.signalCount, [3]=barrier.rank, [4]=barrier.nRanks
 */
void testDeviceWindowMemoryAccessors(
    int myRank,
    int nRanks,
    int signalCount,
    int* results);

} // namespace comms::pipes::test
