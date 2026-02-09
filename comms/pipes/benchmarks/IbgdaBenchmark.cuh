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
    int signalId,
    uint64_t signalVal);

__global__ void ibgdaPutSignalNonAdaptiveWaitLocalKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    int signalId,
    uint64_t signalVal);

__global__ void ibgdaWaitSignalKernel(
    P2pIbgdaTransportDevice* transport,
    int signalId,
    IbgdaCmpOp cmpOp,
    uint64_t expectedSignal);

__global__ void ibgdaSignalOnlyKernel(
    P2pIbgdaTransportDevice* transport,
    int signalId,
    uint64_t signalVal);

__global__ void ibgdaResetSignalKernel(
    P2pIbgdaTransportDevice* transport,
    int signalId);

__global__ void ibgdaPutWaitLocalKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes);

__global__ void ibgdaPutWaitLocalBatchKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    int numIters,
    unsigned long long* totalCycles);

__global__ void ibgdaPutSignalWaitLocalBatchKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    int signalId,
    int numIters,
    unsigned long long* totalCycles);

__global__ void ibgdaSignalOnlyBatchKernel(
    P2pIbgdaTransportDevice* transport,
    int signalId,
    int numIters,
    unsigned long long* totalCycles);

} // namespace comms::pipes::benchmark
