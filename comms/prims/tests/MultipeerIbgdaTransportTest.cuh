// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

#include "comms/prims/transport/P2pIbTransportDevice.cuh"
#include "comms/prims/transport/ibgda/IbgdaBuffer.h"

// Include the host-safe header for the public API
#include "comms/prims/tests/MultipeerIbgdaTransportTest.h"

namespace comms::prims::test {

// Internal kernel declarations - only visible to CUDA compilation units

__global__ void putAndSignalKernel(
    P2pIbTransportDevice transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal);

__global__ void putAndSignalGroupKernel(
    P2pIbTransportDevice transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal);

__global__ void putAndSignalGroupMultiWarpKernel(
    P2pIbTransportDevice transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal);

__global__ void putAndSignalGroupBlockKernel(
    P2pIbTransportDevice transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal);

__global__ void waitSignalKernel(
    P2pIbTransportDevice transport,
    int signalId,
    uint64_t expectedSignal);

__global__ void multiplePutAndSignalKernel(
    P2pIbTransportDevice transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t bytesPerPut,
    int signalId,
    int numPuts);

__global__ void signalOnlyKernel(
    P2pIbTransportDevice transport,
    int signalId,
    uint64_t signalVal);

__global__ void putOnlyKernel(
    P2pIbTransportDevice transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes);

__global__ void sendRecvKernel(
    P2pIbgdaTransportDevice* transport,
    void* buffer,
    std::size_t nbytes,
    std::size_t maxSignalBytes,
    bool send);

#ifndef __HIP_PLATFORM_AMD__
__global__ void progressSendRecvKernel(
    P2pIbgdaTransportDevice* transport,
    void* buffer,
    std::size_t nbytes,
    std::size_t maxSignalBytes,
    bool send);

__global__ void progressReservationKernel(
    P2pIbgdaTransportDevice* transport,
    int64_t* output,
    std::size_t sendBytes,
    std::size_t recvBytes);

__global__ void sendRecvReuseCreditKernel(
    P2pIbgdaTransportDevice* transport,
    void* buffer,
    std::size_t nbytes,
    int activeBlocks,
    int iterations,
    bool send,
    bool useProgress,
    uint64_t waitExpectedNicDoneCredit,
    uint64_t waitExpectedSlotFreeCredit,
    uint64_t* output);
#endif

__global__ void
fillPatternKernel(uint8_t* buffer, std::size_t nbytes, uint8_t baseValue);

__global__ void verifyPatternKernel(
    const uint8_t* buffer,
    std::size_t nbytes,
    uint8_t expectedBaseValue,
    int* errorCount);

__global__ void waitReadyThenPutAndSignalKernel(
    P2pIbTransportDevice transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    int readySignalId,
    uint64_t readySignalVal,
    int dataSignalId,
    uint64_t dataSignalVal);

__global__ void bidirectionalPutAndWaitKernel(
    P2pIbTransportDevice transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    int sendSignalId,
    uint64_t sendSignalVal,
    int recvSignalId,
    uint64_t recvSignalVal);

__global__ void allToAllSendKernel(
    P2pIbTransportDevice* peerTransports,
    IbgdaLocalBuffer* localSendBufs,
    IbgdaRemoteBuffer* peerRecvBufs,
    int myRank,
    std::size_t nbytes,
    int numPeers);

__global__ void allToAllWaitKernel(
    P2pIbTransportDevice* peerTransports,
    int numPeers);

__global__ void putSignalCounterKernel(
    P2pIbTransportDevice transport,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal,
    int counterId,
    uint64_t counterVal);

__global__ void waitCounterKernel(
    P2pIbTransportDevice transport,
    int counterId,
    uint64_t expectedVal);

// Multi-QP kernel: QP selection is transparent via active_qp() inside transport
__global__ void multiQpPutAndSignalKernel(
    P2pIbTransportDevice transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t totalBytes,
    int signalId,
    uint64_t signalVal);

} // namespace comms::prims::test
