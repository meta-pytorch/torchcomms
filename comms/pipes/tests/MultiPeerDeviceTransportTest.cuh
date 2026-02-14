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
 * Test kernel: Verify DeviceCounter construction and basic accessors
 *
 * @param counterCount Number of counters
 * @param results Output array for test results [0]=counterCount
 */
void testDeviceCounterConstruction(int counterCount, uint32_t* results);

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

// =============================================================================
// DeviceCounter Operation Tests
// =============================================================================

/**
 * Test kernel: Verify counter increment and read operations
 *
 * @param counterCount Number of counters
 * @param results Output array: [0]=counter value after increment
 */
void testCounterIncrementAndRead(int counterCount, uint64_t* results);

/**
 * Test kernel: Verify counter value accumulation
 *
 * @param counterCount Number of counters
 * @param results Output array: [0]=accumulated counter value after 3 increments
 */
void testCounterValueAccumulation(int counterCount, uint64_t* results);

/**
 * Test kernel: Verify counter increment with custom value
 *
 * @param counterCount Number of counters
 * @param incrementValue Value to increment by
 * @param results Output array: [0]=counter value after increment
 */
void testCounterIncrementCustomValue(
    int counterCount,
    uint64_t incrementValue,
    uint64_t* results);

/**
 * Test kernel: Verify wait_counter with CMP_GE comparison
 *
 * @param counterCount Number of counters
 * @param results Output array: [0]=1 if wait completed successfully
 */
void testWaitCounterCmpGe(int counterCount, int* results);

/**
 * Test kernel: Verify wait_counter with CMP_EQ comparison
 *
 * @param counterCount Number of counters
 * @param results Output array: [0]=1 if wait completed successfully
 */
void testWaitCounterCmpEq(int counterCount, int* results);

/**
 * Test kernel: Verify reset_counter resets a single counter
 *
 * @param counterCount Number of counters
 * @param results Output array: [0]=counter value after reset
 */
void testResetCounter(int counterCount, uint64_t* results);

/**
 * Test kernel: Verify reset_all_counters resets all counters
 *
 * @param counterCount Number of counters (should be >= 3)
 * @param results Output array: [0..counterCount-1]=counter values after reset
 */
void testResetAllCounters(int counterCount, uint64_t* results);

/**
 * Test kernel: Verify multiple counters are independent
 *
 * @param counterCount Number of counters (should be >= 3)
 * @param results Output array: [0..2]=counter values for counters 0, 1, 2
 */
void testMultipleCounterIndependence(int counterCount, uint64_t* results);
} // namespace comms::pipes::test
