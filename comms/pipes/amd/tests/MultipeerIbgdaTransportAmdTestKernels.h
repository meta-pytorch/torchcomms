// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// AMD/HIP port of comms/pipes/tests/MultipeerIbgdaTransportTest.h
// Declares kernel wrapper functions for the multipeer IBGDA integration tests.

#pragma once

#include <cstddef>
#include <cstdint>

#include "comms/pipes/IbgdaBuffer.h"

namespace pipes_gda {
template <typename NicBackend>
class P2pIbgdaTransportDeviceImpl;
struct Mlx5NicBackend;
using P2pIbgdaTransportDevice = P2pIbgdaTransportDeviceImpl<Mlx5NicBackend>;
} // namespace pipes_gda

namespace pipes_gda::tests {

using comms::pipes::IbgdaLocalBuffer;
using comms::pipes::IbgdaRemoteBuffer;
using comms::pipes::NetworkLKey;

// =============================================================================
// Data verification utilities
// =============================================================================

void fillBufferWithPattern(void* d_buf, std::size_t nbytes, uint8_t seed);

void verifyBufferPattern(
    const void* d_buf,
    std::size_t nbytes,
    uint8_t seed,
    bool* d_success);

// =============================================================================
// Sender-side kernel wrappers
// =============================================================================

// put_signal() + wait_local()
void runTestPutAndSignal(
    pipes_gda::P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    uint64_t signalVal);

// put_group_local() + put_signal_group_local() (wavefront group)
void runTestPutAndSignalGroup(
    pipes_gda::P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    uint64_t signalVal);

// put_signal_group_global() with multi-wavefront groups
void runTestPutAndSignalGroupMultiWarp(
    pipes_gda::P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    uint64_t signalVal);

// put_signal_group_global() with block-scope groups
void runTestPutAndSignalGroupBlock(
    pipes_gda::P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    uint64_t signalVal);

// Sequential put_signal() iterations with wait_local() each
void runTestMultiplePutAndSignal(
    pipes_gda::P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    uint64_t signalValPerIter,
    uint32_t numIters);

// signal_remote() + wait_local() (no data)
void runTestSignalOnly(
    pipes_gda::P2pIbgdaTransportDevice* transport,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    uint64_t signalVal);

// put() + wait_local() (no signal)
void runTestPutOnly(
    pipes_gda::P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes);

// reset_signal() — write 0 to remote signal via RDMA inline write
void runTestResetSignal(
    pipes_gda::P2pIbgdaTransportDevice* transport,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId);

// =============================================================================
// Receiver-side kernel wrappers
// =============================================================================

// Volatile spin on local signal buffer until value >= expected
void runTestWaitSignal(
    IbgdaLocalBuffer localSignalBuf,
    int signalId,
    uint64_t expectedVal,
    bool* d_success);

// Volatile spin on local counter buffer until value >= expected
void runTestWaitCounter(
    IbgdaLocalBuffer localCounterBuf,
    int counterId,
    uint64_t expectedVal,
    bool* d_success);

// =============================================================================
// Compound kernel wrappers
// =============================================================================

// Wait for ready signal, then put + signal data
void runTestWaitReadyThenPutAndSignal(
    pipes_gda::P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes,
    IbgdaLocalBuffer localSignalBuf,
    int readySignalId,
    uint64_t readyExpected,
    IbgdaRemoteBuffer remoteSignalBuf,
    int dataSignalId,
    uint64_t dataSignalVal);

// Bidirectional: put + signal to remote AND wait for incoming signal
void runTestBidirectionalPutAndWait(
    pipes_gda::P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes,
    IbgdaRemoteBuffer remoteSignalBuf,
    int sendSignalId,
    uint64_t sendSignalVal,
    IbgdaLocalBuffer localSignalBuf,
    int recvSignalId,
    uint64_t recvExpected,
    bool* d_success);

// put_signal_counter_remote() via companion QP
void runTestPutSignalCounter(
    pipes_gda::P2pIbgdaTransportDevice* transport,
    pipes_gda::P2pIbgdaTransportDevice* companionTransport,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    uint64_t signalVal,
    IbgdaLocalBuffer localCounterBuf,
    int counterId,
    uint64_t counterVal);

} // namespace pipes_gda::tests
