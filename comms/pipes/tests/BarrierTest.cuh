// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#include "comms/pipes/BarrierState.cuh"
#include "comms/pipes/P2pNvlTransportDevice.cuh"

namespace comms::pipes::test {

// Enum for specifying the thread group type
enum class GroupType;

// =============================================================================
// Barrier struct test helpers
// These test the Barrier struct directly without P2pNvlTransportDevice
// =============================================================================

// Arrive operation on a raw Barrier pointer
void testRawBarrierArrive(
    BarrierState* barrier_d,
    int numBlocks,
    int blockSize,
    GroupType groupType);

// Wait operation on a raw Barrier pointer
void testRawBarrierWait(
    BarrierState* barrier_d,
    int numBlocks,
    int blockSize,
    GroupType groupType);

// Arrive then wait on a raw Barrier pointer (same kernel)
void testRawBarrierArriveWait(
    BarrierState* barrier_d,
    int numBlocks,
    int blockSize,
    GroupType groupType);

// Read the barrier current counter value (for verification)
void testReadBarrierCurrentCounter(BarrierState* barrier_d, uint64_t* result_d);

// Read the barrier expected counter value (for verification)
void testReadBarrierExpectedCounter(
    BarrierState* barrier_d,
    uint64_t* result_d);

// =============================================================================
// P2pNvlTransportDevice barrier API test helpers
// These test barrier_sync_threadgroup() with 2-GPU P2P communication
// =============================================================================

// Barrier sync operation using P2pNvlTransportDevice
void testDeviceBarrierSync(
    P2pNvlTransportDevice p2p,
    uint64_t barrierId,
    int numBlocks,
    int blockSize,
    GroupType groupType);

// Multiple barrier syncs in sequence
void testDeviceBarrierSyncMultiple(
    P2pNvlTransportDevice p2p,
    uint64_t barrierId,
    int numSyncs,
    int numBlocks,
    int blockSize,
    GroupType groupType);

// =============================================================================
// Data transfer with barrier verification test helpers
// Tests that data written by one GPU is visible to another after barrier sync
// Each thread group uses its own barrier id (group.group_id)
// =============================================================================

// Write data to remote buffer using put() API, then barrier sync
void testBarrierWriteData(
    P2pNvlTransportDevice p2p,
    char* remoteDataBuffer,
    const char* localSrcBuffer,
    size_t dataSize,
    int numBlocks,
    int blockSize,
    GroupType groupType);

// Barrier sync, then verify local data matches expected value
void testBarrierVerifyData(
    P2pNvlTransportDevice p2p,
    uint8_t* localDataBuffer,
    size_t dataSize,
    uint8_t expectedValue,
    uint32_t* errorCount,
    int numBlocks,
    int blockSize,
    GroupType groupType);

} // namespace comms::pipes::test
