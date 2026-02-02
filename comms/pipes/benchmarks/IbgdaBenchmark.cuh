// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

#include "comms/pipes/IbgdaBuffer.h"
#include "comms/pipes/P2pIbgdaTransportDevice.cuh"

// Include host-safe header for the public API
#include "comms/pipes/benchmarks/IbgdaBenchmark.h"

namespace comms::pipes::benchmark {

// Internal kernel declarations - only visible to CUDA compilation units

__global__ void ibgdaPutSignalWaitLocalKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    uint64_t signalVal);

__global__ void ibgdaWaitSignalKernel(
    P2pIbgdaTransportDevice* transport,
    uint64_t expectedSignal);

__global__ void ibgdaSignalOnlyKernel(
    P2pIbgdaTransportDevice* transport,
    uint64_t signalVal);

__global__ void ibgdaResetSignalKernel(P2pIbgdaTransportDevice* transport);

} // namespace comms::pipes::benchmark
