// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>

namespace comms::pipes::test {

/**
 * Test kernel: Verify DeviceWindow construction and basic accessors
 *
 * @param myRank Rank ID for the window object
 * @param nRanks Total number of ranks
 * @param signalCount Number of signal slots per peer
 * @param results Output array: [0]=rank, [1]=nRanks, [2]=numNvlPeers
 */
void testDeviceWindowConstruction(
    int myRank,
    int nRanks,
    int signalCount,
    int* results);

/**
 * Test kernel: Verify DeviceWindow basic accessors (rank, nRanks)
 *
 * @param myRank Rank ID for the window object
 * @param nRanks Total number of ranks
 * @param results Output array: [0]=rank, [1]=nRanks
 */
void testDeviceWindowBasicAccessors(int myRank, int nRanks, int* results);

/**
 * Test kernel: Verify self-transport put() operation via Transport
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
 * @param myRank Rank ID for the window object
 * @param nRanks Total number of ranks
 * @param results Output: [0]=numPeers, [1..numPeers]=peerIndexToRank
 */
void testPeerIterationHelpers(int myRank, int nRanks, int* results);

/**
 * Test kernel: Verify peer index conversion roundtrip and transport accessors
 *
 * @param myRank Rank ID for the window object
 * @param nRanks Total number of ranks
 * @param results Output array (size = 4*numPeers + 2)
 */
void testPeerIndexConversionRoundtrip(int myRank, int nRanks, int* results);

/**
 * Test: DeviceWindow NVL signal write + read
 *
 * Verifies signal_peer() writes to the NVL inbox and
 * read_signal_from() / read_signal() return the correct value.
 */
void testDeviceWindowSignalWriteRead(
    int myRank,
    int nRanks,
    int signalCount,
    int targetPeerRank,
    int signalId,
    uint64_t* results);

/**
 * Test: DeviceWindow read_signal
 *
 * Verifies signal_peer() + read_signal(signal_id) returns the
 * correct aggregate value.
 */
void testDeviceWindowReadSignalGroup(
    int myRank,
    int nRanks,
    int signalCount,
    uint64_t* results);

/**
 * Test: DeviceWindow signal_all + read_signal aggregate
 *
 * Verifies signal_all() signals all peers and read_signal()
 * returns the correct aggregate across multiple peers.
 */
void testDeviceWindowSignalAllAggregate(
    int myRank,
    int nRanks,
    int signalCount,
    int signalId,
    uint64_t* results);

/**
 * Test: IBGDA signal read_signal_from + read_signal
 *
 * Seeds a known value into the IBGDA inbox at (sourceRank, signalId),
 * then verifies read_signal_from() and read_signal() return the
 * correct value. Validates that the inbox layout (peerIdx * signalCount
 * + signalId) is consistent with the pre-offset remote buffer scheme.
 */
void testIbgdaSignalRead(
    int myRank,
    int nRanks,
    int signalCount,
    int sourceRank,
    int signalId,
    uint64_t seedValue,
    uint64_t* results);

/**
 * Test: IBGDA multi-peer aggregate read_signal
 *
 * Seeds per-peer values into the IBGDA inbox for a given signalId,
 * then verifies read_signal() returns the correct sum across all peers.
 */
void testIbgdaSignalAggregateRead(
    int myRank,
    int nRanks,
    int signalCount,
    int signalId,
    const uint64_t* peerValues,
    int nPeers,
    uint64_t* result);

/**
 * Test: DeviceWindow offset-based NVL put()
 *
 * Verifies the offset-based put() overload correctly resolves
 * dst_offset into the window buffer and src_buf + src_offset
 * into a registered source buffer, then copies the data.
 */
void testDeviceWindowNvlOffsetPut(
    int myRank,
    int nRanks,
    char* windowBuf_d,
    const char* srcBuf_d,
    std::size_t srcBufSize,
    std::size_t dst_offset,
    std::size_t src_offset,
    std::size_t nbytes);

/**
 * Test: DeviceWindow offset-based NVL put_signal()
 *
 * Verifies the offset-based put_signal() overload copies data
 * to the correct region and signals the target peer.
 */
void testDeviceWindowNvlOffsetPutSignal(
    int myRank,
    int nRanks,
    char* windowBuf_d,
    const char* srcBuf_d,
    std::size_t srcBufSize,
    std::size_t dst_offset,
    std::size_t src_offset,
    std::size_t nbytes,
    int signalId);

/**
 * Test: Bidirectional offset-based NVL put_signal()
 *
 * Simulates 2 ranks on a single GPU. Each rank does put_signal()
 * to the other's window buffer with different data patterns,
 * verifying both directions land correctly.
 *
 * @param windowBuf0_d Window buffer for rank 0 (rank 1 writes here)
 * @param windowBuf1_d Window buffer for rank 1 (rank 0 writes here)
 * @param srcBuf0_d Source buffer for rank 0
 * @param srcBuf1_d Source buffer for rank 1
 * @param srcBufSize Size of each source buffer in bytes
 * @param dst_offset Byte offset into destination window buffer
 * @param src_offset Byte offset into source buffer
 * @param nbytes Number of bytes to copy
 * @param signalId Signal slot to use
 */
void testDeviceWindowNvlBidirectionalOffsetPutSignal(
    char* windowBuf0_d,
    char* windowBuf1_d,
    const char* srcBuf0_d,
    const char* srcBuf1_d,
    std::size_t srcBufSize,
    std::size_t dst_offset,
    std::size_t src_offset,
    std::size_t nbytes,
    int signalId);

} // namespace comms::pipes::test
