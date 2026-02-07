// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>

namespace comms::pipes::test {

/**
 * Test kernel: Verify DeviceSignal construction and basic accessors
 *
 * @param myRank Rank ID for the signal object
 * @param nRanks Total number of ranks
 * @param signalCount Number of signal slots per peer
 * @param results Output array for test results [0]=rank, [1]=nRanks,
 * [2]=signalCount
 */
void testDeviceSignalConstruction(
    int myRank,
    int nRanks,
    int signalCount,
    int* results);

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

} // namespace comms::pipes::test
