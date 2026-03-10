// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

#include "comms/pipes/IbgdaBuffer.h"
#include "comms/pipes/P2pIbgdaTransportDevice.cuh"

// Include the host-safe header for the public API
#include "comms/pipes/tests/MultipeerIbgdaTransportTest.h"

namespace comms::pipes::test {

// Internal kernel declarations - only visible to CUDA compilation units

__global__ void putAndSignalKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    uint64_t signalVal);

__global__ void putAndSignalGroupKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    uint64_t signalVal);

__global__ void putAndSignalGroupMultiWarpKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    uint64_t signalVal);

__global__ void putAndSignalGroupBlockKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    uint64_t signalVal);

__global__ void waitSignalKernel(
    volatile uint64_t* localSignalBuf,
    int signalId,
    uint64_t expectedSignal);

__global__ void multiplePutAndSignalKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t bytesPerPut,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    int numPuts);

__global__ void signalOnlyKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    uint64_t signalVal);

__global__ void putOnlyKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes);

__global__ void
fillPatternKernel(uint8_t* buffer, std::size_t nbytes, uint8_t baseValue);

__global__ void verifyPatternKernel(
    const uint8_t* buffer,
    std::size_t nbytes,
    uint8_t expectedBaseValue,
    int* errorCount);

__global__ void waitReadyThenPutAndSignalKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    volatile uint64_t* localSignalBuf,
    int readySignalId,
    uint64_t readySignalVal,
    IbgdaRemoteBuffer remoteSignalBuf,
    int dataSignalId,
    uint64_t dataSignalVal);

__global__ void bidirectionalPutAndWaitKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    IbgdaRemoteBuffer remoteSignalBuf,
    int sendSignalId,
    uint64_t sendSignalVal,
    volatile uint64_t* localSignalBuf,
    int recvSignalId,
    uint64_t recvSignalVal);

__global__ void allToAllSendKernel(
    P2pIbgdaTransportDevice** peerTransports,
    IbgdaLocalBuffer* localSendBufs,
    IbgdaRemoteBuffer* peerRecvBufs,
    IbgdaRemoteBuffer* remoteSignalBufs,
    int myRank,
    std::size_t nbytes,
    int numPeers);

__global__ void allToAllWaitKernel(
    volatile uint64_t* localSignalBuf,
    int* peerRanks,
    int numPeers);

__global__ void putSignalCounterKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localDataBuf,
    IbgdaRemoteBuffer remoteDataBuf,
    std::size_t nbytes,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    uint64_t signalVal,
    IbgdaLocalBuffer localCounterBuf,
    int counterId,
    uint64_t counterVal);

__global__ void resetSignalKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId);

__global__ void waitCounterKernel(
    volatile uint64_t* counterBuf,
    int counterId,
    uint64_t expectedVal);

} // namespace comms::pipes::test
