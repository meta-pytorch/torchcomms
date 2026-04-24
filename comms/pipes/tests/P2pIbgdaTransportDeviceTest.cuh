// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>

#include "comms/pipes/IbgdaBuffer.h"

namespace comms::pipes::tests {

// Test transport construction on device (null QP)
void runTestP2pTransportConstruction(bool* d_success);

// Test default construction - all members should be initialized to null/zero
void runTestP2pTransportDefaultConstruction(bool* d_success);

// Test read_signal returns the value at the correct signal slot.
// Constructs transport with ownedLocalSignalBuf pointing to d_signalBuf,
// then verifies read_signal(slotId) returns the correct value for each slot.
void runTestP2pTransportReadSignal(
    uint64_t* d_signalBuf,
    int numSignals,
    bool* d_success);

// =============================================================================
// wait_signal tests - Test GE (>=) comparison (only supported operation)
// These tests pre-set the signal buffer to a value that satisfies the condition
// so wait_signal returns immediately without blocking.
// =============================================================================

// Test wait_signal with GE (greater or equal) comparison
// Pre-sets signal to a value >= targetValue, waits for GE targetValue
void runTestWaitSignalGE(
    uint64_t* d_signalBuf,
    uint64_t targetValue,
    bool* d_success);

// Test wait_signal with multiple signal slots
// Verifies that wait_signal operates on the correct slot
void runTestWaitSignalMultipleSlots(
    uint64_t* d_signalBuf,
    int numSignals,
    bool* d_success);

// =============================================================================
// Group-level API tests
// =============================================================================

// Test put_group_local data partitioning across warp lanes.
void runTestPutGroupPartitioning(bool* d_success);

// Test put_signal_group_local signal broadcast.
void runTestPutSignalGroupBroadcast(bool* d_success);

// =============================================================================
// broadcast tests for non-warp scopes
// =============================================================================

// Test broadcast<uint64_t> with BLOCK scope (<<<4, 256>>>)
void runTestBroadcast64Block(bool* d_success);

// Test broadcast<uint64_t> with MULTIWARP scope (<<<2, 512>>>)
void runTestBroadcast64Multiwarp(bool* d_success);

// Test double-broadcast safety: two consecutive broadcasts with different
// values should not race (verifies the double-sync pattern)
void runTestBroadcast64DoubleSafety(bool* d_success);

// Test put_group_local partitioning logic with block-sized groups
void runTestPutGroupPartitioningBlock(bool* d_success);

// =============================================================================
// wait_signal timeout tests
// =============================================================================

// Test that wait_signal traps when timeout expires (signal never satisfies
// condition). After calling this, check cudaGetLastError() for trap error.
void runTestWaitSignalTimeout(
    uint64_t* d_signalBuf,
    int device,
    uint32_t timeout_ms);

// Test that wait_signal completes normally when signal is already satisfied
// even when timeout is enabled (positive test case).
void runTestWaitSignalNoTimeout(
    uint64_t* d_signalBuf,
    int device,
    uint32_t timeout_ms,
    bool* d_success);

} // namespace comms::pipes::tests
