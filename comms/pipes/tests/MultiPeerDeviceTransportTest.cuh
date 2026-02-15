// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>

namespace comms::pipes::test {

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

} // namespace comms::pipes::test
