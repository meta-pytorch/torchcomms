// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

#include "comms/pipes/IbgdaBuffer.h"

// Forward declarations for DOCA types (opaque in host code)
struct doca_gpu_dev_verbs_qp;

namespace comms::pipes {
// Forward declaration - full definition in .cuh file
class P2pIbgdaTransportDevice;
} // namespace comms::pipes

namespace comms::pipes::tests {

/**
 * Allocate and initialize a P2pIbgdaTransportDevice on the GPU.
 *
 * This helper function allows .cc files to create device transport objects
 * without needing to include CUDA headers. The device transport is allocated
 * in GPU memory and initialized with the provided parameters.
 *
 * @param qp GPU QP handle for RDMA operations
 * @param localSignalBuf Local signal buffer descriptor
 * @param remoteSignalBuf Remote signal buffer descriptor
 * @param numSignals Number of signal slots
 * @return Pointer to device-allocated transport, or nullptr on failure
 */
P2pIbgdaTransportDevice* allocateDeviceTransport(
    doca_gpu_dev_verbs_qp* qp,
    const IbgdaLocalBuffer& localSignalBuf,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int numSignals);

/**
 * Free a device-allocated P2pIbgdaTransportDevice.
 *
 * @param d_transport Device pointer to free (null-safe)
 */
void freeDeviceTransport(P2pIbgdaTransportDevice* d_transport);

// =============================================================================
// Kernel Wrapper Declarations
// =============================================================================

/**
 * Test put_signal: Sender writes data and signals receiver.
 *
 * @param d_transport Device transport handle
 * @param localDataBuf Source data buffer
 * @param remoteDataBuf Destination data buffer on peer
 * @param nbytes Number of bytes to transfer
 * @param signalId Signal slot to use
 * @param signalVal Value to add to signal
 */
void runPutSignalKernel(
    P2pIbgdaTransportDevice* d_transport,
    const IbgdaLocalBuffer& localDataBuf,
    const IbgdaRemoteBuffer& remoteDataBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal);

/**
 * Test put_signal_non_adaptive: Sender writes data and signals receiver.
 * Uses fused put_signal operation - faster but unsafe with adaptive routing.
 *
 * @param d_transport Device transport handle
 * @param localDataBuf Source data buffer
 * @param remoteDataBuf Destination data buffer on peer
 * @param nbytes Number of bytes to transfer
 * @param signalId Signal slot to use
 * @param signalVal Value to add to signal
 */
void runPutSignalNonAdaptiveKernel(
    P2pIbgdaTransportDevice* d_transport,
    const IbgdaLocalBuffer& localDataBuf,
    const IbgdaRemoteBuffer& remoteDataBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal);

/**
 * Test wait_signal: Receiver waits for signal, then verifies data.
 *
 * @param d_transport Device transport handle
 * @param signalId Signal slot to wait on
 * @param expectedSignal Expected signal value (GE comparison)
 * @param d_success Device pointer to success flag
 */
void runWaitSignalKernel(
    P2pIbgdaTransportDevice* d_transport,
    int signalId,
    uint64_t expectedSignal,
    bool* d_success);

/**
 * Test signal-only operation.
 *
 * @param d_transport Device transport handle
 * @param signalId Signal slot to use
 * @param signalVal Value to add to signal
 */
void runSignalOnlyKernel(
    P2pIbgdaTransportDevice* d_transport,
    int signalId,
    uint64_t signalVal);

/**
 * Verify data matches expected pattern after transfer.
 *
 * @param d_data Device data buffer to verify
 * @param nbytes Number of bytes to check
 * @param expectedPattern Expected byte pattern
 * @param d_success Device pointer to success flag
 */
void runVerifyDataKernel(
    void* d_data,
    std::size_t nbytes,
    uint8_t expectedPattern,
    bool* d_success);

/**
 * Fill data buffer with a pattern.
 *
 * @param d_data Device data buffer to fill
 * @param nbytes Number of bytes to fill
 * @param pattern Byte pattern to write
 */
void runFillDataKernel(void* d_data, std::size_t nbytes, uint8_t pattern);

} // namespace comms::pipes::tests
