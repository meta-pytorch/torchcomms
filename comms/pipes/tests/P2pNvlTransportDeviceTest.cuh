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
    P2pNvlTransportDevice* p2p,
    uint64_t signalId,
    SignalOp op,
    uint64_t value,
    int numBlocks,
    int blockSize,
    GroupType groupType = GroupType::WARP);

// Wait operation on a single transport (waits on its local state)
void testDeviceWaitSignal(
    P2pNvlTransportDevice* p2p,
    uint64_t signalId,
    CmpOp op,
    uint64_t value,
    int numBlocks,
    int blockSize,
    GroupType groupType = GroupType::WARP);

// Signal then wait within a single kernel
void testDeviceSignalThenWait(
    P2pNvlTransportDevice* p2p,
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
// LL128 transport send/recv test helpers
// These test the ll128_send()/ll128_recv() methods on P2pNvlTransportDevice
// =============================================================================

/**
 * Test LL128 send/recv via P2pNvlTransportDevice transport wrappers.
 *
 * Uses two transports in a loopback configuration where transport0 sends
 * to transport1. The sender calls p2p.ll128_send() and the receiver calls
 * p2p.ll128_recv().
 *
 * @param sender Sender transport (writes to receiver's LL128 buffer)
 * @param receiver Receiver transport (reads from its local LL128 buffer)
 * @param src_d Source buffer on sender GPU
 * @param dst_d Destination buffer on receiver GPU
 * @param nbytes Number of bytes to transfer (must be multiple of 16)
 * @param numBlocks Number of blocks to launch
 * @param blockSize Threads per block
 */
void testLl128SendRecv(
    P2pNvlTransportDevice sender,
    P2pNvlTransportDevice receiver,
    const char* src_d,
    char* dst_d,
    size_t nbytes,
    int numBlocks,
    int blockSize);

} // namespace comms::pipes::test
