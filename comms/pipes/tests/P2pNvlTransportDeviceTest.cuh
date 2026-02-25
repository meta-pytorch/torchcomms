// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#include "comms/pipes/P2pNvlTransportDevice.cuh"

namespace comms::pipes::test {

// Enum for specifying the thread group type
enum class GroupType {
  WARP, // 32-thread warp groups
  BLOCK // Full block groups (all threads in block)
};

// =============================================================================
// Signal API test helpers for P2pNvlTransportDevice (D90597777)
// These tests use a loopback configuration on a single GPU to test the
// signal/wait APIs without requiring actual NVLink peer communication.
// =============================================================================

// Signal operation on a single transport (signals to its remote state)
void testDeviceSignal(
    P2pNvlTransportDevice p2p,
    uint64_t signalId,
    SignalOp op,
    uint64_t value,
    int numBlocks,
    int blockSize,
    GroupType groupType = GroupType::WARP);

// Wait operation on a single transport (waits on its local state)
void testDeviceWaitSignal(
    P2pNvlTransportDevice p2p,
    uint64_t signalId,
    CmpOp op,
    uint64_t value,
    int numBlocks,
    int blockSize,
    GroupType groupType = GroupType::WARP);

// Signal then wait within a single kernel
void testDeviceSignalThenWait(
    P2pNvlTransportDevice p2p,
    uint64_t signalId,
    SignalOp signalOp,
    uint64_t signalValue,
    CmpOp waitOp,
    uint64_t waitValue,
    int numBlocks,
    int blockSize,
    GroupType groupType = GroupType::WARP);

// =============================================================================
// Direct Signal struct test helpers
// These test the Signal struct directly without P2pNvlTransportDevice
// =============================================================================

// Signal operation on a raw Signal pointer
void testRawSignal(
    SignalState* signal_d,
    SignalOp op,
    uint64_t value,
    int numBlocks,
    int blockSize,
    GroupType groupType = GroupType::WARP);

// Wait operation on a raw SignalState pointer
void testRawWaitSignal(
    SignalState* signal_d,
    CmpOp op,
    uint64_t value,
    int numBlocks,
    int blockSize,
    GroupType groupType = GroupType::WARP);

// Read the signal value (for verification)
void testReadSignal(SignalState* signal_d, uint64_t* result_d);

// =============================================================================
// RecvStream/SendStream test helpers
// These test the streaming primitives for pipelined collectives
// =============================================================================

/**
 * Test RecvStream/SendStream loopback transfer.
 *
 * Uses two transports in a loopback configuration where transport0 sends
 * to transport1. The sender uses SendStream::for_each_slot() and the
 * receiver uses RecvStream::for_each_ready_chunk().
 *
 * @param transport0 Sender transport (writes to transport1's staging)
 * @param transport1 Receiver transport (reads from its own staging)
 * @param srcBuffer0 Source buffer on GPU 0
 * @param dstBuffer1 Destination buffer on GPU 1
 * @param nbytes Number of bytes to transfer
 * @param numBlocks Number of blocks to launch
 * @param blockSize Threads per block
 */
void testRecvSendStreamLoopback(
    P2pNvlTransportDevice transport0,
    P2pNvlTransportDevice transport1,
    char* srcBuffer0,
    char* dstBuffer1,
    std::size_t nbytes,
    int numBlocks,
    int blockSize);

/**
 * Test RecvStream/SendStream forwarding through an intermediate rank.
 *
 * Uses a ring topology: GPU0 (sender) → GPU1 (intermediate) → GPU0 (receiver).
 * The intermediate rank uses the positional API (slot_for + commit_slot) to
 * forward each received chunk to the next hop.
 *
 * @param transport_send_0to1 Sender transport on GPU0 (writes to GPU1 staging)
 * @param transport_recv_1from0 Receiver transport on GPU1 (reads from GPU1
 * staging)
 * @param transport_send_1to0 Sender transport on GPU1 (writes to GPU0 staging)
 * @param transport_recv_0from1 Receiver transport on GPU0 (reads from GPU0
 * staging)
 * @param srcBuffer0 Source buffer on GPU 0
 * @param dstBuffer0 Destination buffer on GPU 0
 * @param nbytes Number of bytes to transfer
 * @param numBlocks Number of blocks to launch
 * @param blockSize Threads per block
 */
void testRecvSendStreamForwarding(
    P2pNvlTransportDevice transport_send_0to1,
    P2pNvlTransportDevice transport_recv_1from0,
    P2pNvlTransportDevice transport_send_1to0,
    P2pNvlTransportDevice transport_recv_0from1,
    char* srcBuffer0,
    char* dstBuffer0,
    std::size_t nbytes,
    int numBlocks,
    int blockSize);

} // namespace comms::pipes::test
