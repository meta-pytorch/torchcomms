// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

#include "comms/prims/transport/P2pIbTransportDevice.cuh"
#include "comms/prims/transport/ibgda/IbgdaBuffer.h"

// Include host-safe header for the public API
#include "comms/prims/benchmarks/IbgdaBenchmark.h"

namespace comms::prims::benchmark {

// Internal kernel declarations - only visible to CUDA compilation units.
//
// Kernels take the backend-agnostic P2pIbTransportDevice handle by value so the
// same kernel runs over either IBGDA (GPU-initiated) or IBRC (CPU-proxy); the
// handle dispatches each device call on its embedded backend tag.

__global__ void ibgdaPutSignalWaitCounterKernel(
    P2pIbTransportDevice transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    int signalId,
    int counterId);

__global__ void ibgdaPutWaitCounterKernel(
    P2pIbTransportDevice transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes);

__global__ void ibgdaPutWaitCounterBatchKernel(
    P2pIbTransportDevice transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    int counterId,
    int numIters,
    unsigned long long* totalCycles);

__global__ void ibgdaPutWaitLocalBatchKernel(
    P2pIbTransportDevice transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    int numIters,
    unsigned long long* totalCycles);

__global__ void ibgdaPutFlushBatchKernel(
    P2pIbTransportDevice transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    int numIters,
    unsigned long long* totalCycles);

__global__ void ibgdaThreadScopeMultiBlockPutFlushBatchKernel(
    P2pIbgdaTransportDevice* transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytesPerBlock,
    int numIters,
    unsigned long long* blockCycles);

__global__ void ibgdaPutSignalWaitCounterBatchKernel(
    P2pIbTransportDevice transport,
    IbgdaLocalBuffer localBuf,
    IbgdaRemoteBuffer remoteBuf,
    std::size_t nbytes,
    int signalId,
    int counterId,
    int numIters,
    unsigned long long* totalCycles);

__global__ void ibgdaSignalOnlyBatchKernel(
    P2pIbTransportDevice transport,
    IbgdaRemoteBuffer remoteSignalBuf,
    int signalId,
    int numIters,
    unsigned long long* totalCycles);

// Multi-peer kernels for counter fan-out validation.
//
// `transports` is a contiguous device array of one P2pIbTransportDevice handle
// per peer (peer p at index p). This replaces the IBGDA-only
// getDeviceTransportPtr()+stride layout so the multi-peer path works for both
// backends (IBRC exposes only per-peer getP2pTransportDevice()).

__global__ void ibgdaMultiPeerSerialCounterFanOutBatchKernel(
    const P2pIbTransportDevice* transports,
    int numPeers,
    IbgdaLocalBuffer localBuf,
    const IbgdaRemoteBuffer* remoteDataBufs,
    std::size_t nbytes,
    const IbgdaRemoteBuffer* remoteSignalBufs,
    int signalId,
    IbgdaLocalBuffer localCounterBuf,
    int numIters,
    unsigned long long* totalCycles);

__global__ void ibgdaMultiPeerCounterFanOutBatchKernel(
    const P2pIbTransportDevice* transports,
    int numPeers,
    IbgdaLocalBuffer localBuf,
    const IbgdaRemoteBuffer* remoteDataBufs,
    std::size_t nbytes,
    const IbgdaRemoteBuffer* remoteSignalBufs,
    int signalId,
    IbgdaLocalBuffer localCounterBuf,
    int counterId,
    int numIters,
    unsigned long long* totalCycles);

} // namespace comms::prims::benchmark
