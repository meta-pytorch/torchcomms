// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>

#include "comms/pipes/IbgdaBuffer.h"

namespace comms::pipes::tests {

// Wrapper function to launch test kernel (defined in .cu, called from .cc)
// This function creates a P2pIbgdaTransportDevice on the device with the given
// buffers and verifies its accessors work correctly.
void runTestP2pTransportConstruction(
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    bool* d_success);

// Test default construction - all members should be initialized to null/zero
void runTestP2pTransportDefaultConstruction(bool* d_success);

// Test getNumSignals accessor with various values
void runTestP2pTransportNumSignals(
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    int numSignals,
    bool* d_success);

// Test signal pointer arithmetic - verify correct offsets for multi-signal
// setup This tests the internal getLocalSignalPtr/getRemoteSignalPtr logic by
// checking buffer accessors return pointers at expected offsets.
void runTestP2pTransportSignalPointerArithmetic(
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    int numSignals,
    bool* d_success);

// Test read_signal returns the value at the correct signal slot.
// This writes known values to the signal buffer and verifies read_signal
// returns the correct value for each slot.
void runTestP2pTransportReadSignal(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    int numSignals,
    bool* d_success);

// Test IbgdaWork struct construction and value access
void runTestIbgdaWork(bool* d_success);

// =============================================================================
// wait_signal tests - Test each comparison operation
// These tests pre-set the signal buffer to a value that satisfies the condition
// so wait_signal returns immediately without blocking.
// =============================================================================

// Test wait_signal with EQ (equal) comparison
// Pre-sets signal to targetValue, waits for EQ targetValue
void runTestWaitSignalEQ(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    uint64_t targetValue,
    bool* d_success);

// Test wait_signal with NE (not equal) comparison
// Pre-sets signal to a value != targetValue, waits for NE targetValue
void runTestWaitSignalNE(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    uint64_t signalValue,
    uint64_t targetValue,
    bool* d_success);

// Test wait_signal with GE (greater or equal) comparison
// Pre-sets signal to a value >= targetValue, waits for GE targetValue
void runTestWaitSignalGE(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    uint64_t signalValue,
    uint64_t targetValue,
    bool* d_success);

// Test wait_signal with GT (greater than) comparison
// Pre-sets signal to a value > targetValue, waits for GT targetValue
void runTestWaitSignalGT(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    uint64_t signalValue,
    uint64_t targetValue,
    bool* d_success);

// Test wait_signal with LE (less or equal) comparison
// Pre-sets signal to a value <= targetValue, waits for LE targetValue
void runTestWaitSignalLE(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    uint64_t signalValue,
    uint64_t targetValue,
    bool* d_success);

// Test wait_signal with LT (less than) comparison
// Pre-sets signal to a value < targetValue, waits for LT targetValue
void runTestWaitSignalLT(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    uint64_t signalValue,
    uint64_t targetValue,
    bool* d_success);

// Test wait_signal with multiple signal slots
// Verifies that wait_signal operates on the correct slot
void runTestWaitSignalMultipleSlots(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    int numSignals,
    bool* d_success);

// =============================================================================
// Group-level API tests
// These tests verify the put_group_local and put_signal_group_local APIs that
// accept a ThreadGroup, partition a single data chunk across group threads,
// and have the leader issue the signal.
// =============================================================================

// Test put_group_local data partitioning across warp lanes.
// Verifies that each lane computes the correct offset and chunk size.
void runTestPutGroupPartitioning(bool* d_success);

// Test put_signal_group_local signal broadcast.
// Verifies that the leader issues the signal and broadcasts the ticket
// to all lanes via broadcast<uint64_t>.
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
// These tests verify that the Timeout parameter on wait_signal works correctly.
// =============================================================================

// Test that wait_signal traps when timeout expires (signal never satisfies
// condition). After calling this, check cudaGetLastError() for trap error.
void runTestWaitSignalTimeout(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    int device,
    uint32_t timeout_ms);

// Test that wait_signal completes normally when signal is already satisfied
// even when timeout is enabled (positive test case).
void runTestWaitSignalNoTimeout(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    int device,
    uint32_t timeout_ms,
    bool* d_success);

} // namespace comms::pipes::tests
