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

__global__ void putSignalKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    uint64_t signalVal);

__global__ void waitSignalKernel(
    P2pIbgdaTransportDevice* transport,
    uint64_t expectedSignal);

__global__ void multiplePutSignalKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t bytesPerPut,
    int numPuts);

__global__ void
fillPatternKernel(uint8_t* buffer, std::size_t nbytes, uint8_t baseValue);

__global__ void verifyPatternKernel(
    const uint8_t* buffer,
    std::size_t nbytes,
    uint8_t expectedBaseValue,
    int* errorCount);

} // namespace comms::pipes::test
