// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace comms::pipes::test {

// =============================================================================
// Passing case tests - expression evaluates to true, kernel completes normally
// =============================================================================

// Tests PIPES_DEVICE_CHECK with a true expression - should complete without
// trap
void testDeviceCheckPassing(int numBlocks, int blockSize);

// Tests PIPES_DEVICE_CHECK_MSG with a true expression - should complete without
// trap
void testDeviceCheckMsgPassing(int numBlocks, int blockSize);

// Tests multiple PIPES_DEVICE_CHECK calls in same kernel - all passing
void testMultipleDeviceChecksPassing(
    uint32_t value,
    uint32_t threshold,
    int numBlocks,
    int blockSize);

// =============================================================================
// Trap tests - expression evaluates to false, triggers __trap()
// =============================================================================

// Tests PIPES_DEVICE_CHECK with a false expression - should trigger trap
void testDeviceCheckFailing(int numBlocks, int blockSize);

// Tests PIPES_DEVICE_CHECK_MSG with a false expression - should trigger trap
void testDeviceCheckMsgFailing(int numBlocks, int blockSize);

} // namespace comms::pipes::test
