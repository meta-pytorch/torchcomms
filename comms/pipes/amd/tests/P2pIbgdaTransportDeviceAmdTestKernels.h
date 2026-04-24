// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// AMD/HIP port of comms/pipes/tests/P2pIbgdaTransportDeviceTest.cuh
// Same test wrapper function declarations, adapted for pipes_gda types.

#pragma once

#include <cstdint>

#include "PipesGdaShared.h"

// Forward declaration
struct pipes_gda_gpu_dev_verbs_qp;

namespace pipes_gda::tests {

using pipes_gda::IbgdaLocalBuffer;
using pipes_gda::IbgdaRemoteBuffer;

void runTestP2pTransportConstruction(bool* d_success);

void runTestP2pTransportDefaultConstruction(bool* d_success);

void runTestP2pTransportReadSignal(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    int numSignals,
    bool* d_success);

void runTestIbgdaWork(bool* d_success);

// wait_signal tests (GE-only comparison)
void runTestWaitSignalGE(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    uint64_t targetValue,
    bool* d_success);

void runTestWaitSignalMultipleSlots(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    int numSignals,
    bool* d_success);

// wait_signal timeout tests
void runTestWaitSignalTimeout(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    int device,
    uint32_t timeout_ms);

void runTestWaitSignalNoTimeout(
    uint64_t* d_signalBuf,
    IbgdaLocalBuffer localBuf,
    int device,
    uint32_t timeout_ms,
    bool* d_success);

// ThreadGroup / group-level API tests
void runTestPutGroupPartitioning(bool* d_success);

void runTestPutSignalGroupBroadcast(bool* d_success);

void runTestBroadcast64Block(bool* d_success);

void runTestBroadcast64Multiwarp(bool* d_success);

void runTestBroadcast64DoubleSafety(bool* d_success);

void runTestPutGroupPartitioningBlock(bool* d_success);

// =============================================================================
// put() latency benchmark
// =============================================================================

void runTestPutLatency(
    std::size_t nbytes,
    uint32_t numWarmup,
    uint32_t numIters,
    uint64_t* d_latencies);

void runTestPutLatencyReal(
    pipes_gda_gpu_dev_verbs_qp* qp,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes,
    uint32_t numWarmup,
    uint32_t numIters,
    uint64_t* d_latencies);

} // namespace pipes_gda::tests
