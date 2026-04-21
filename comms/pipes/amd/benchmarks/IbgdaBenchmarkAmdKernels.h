// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// AMD port of comms/pipes/benchmarks/IbgdaBenchmark.h
// Same API but uses pipes_gda::P2pIbgdaTransportDevice and hipStream_t.

#pragma once

#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>

#include "PipesGdaShared.h"

namespace pipes_gda {
// Forward declaration
template <typename NicBackend>
class P2pIbgdaTransportDeviceImpl;
struct Mlx5NicBackend;
using P2pIbgdaTransportDevice = P2pIbgdaTransportDeviceImpl<Mlx5NicBackend>;
} // namespace pipes_gda

namespace pipes_gda::benchmark {

void launchIbgdaPutSignalSingle(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int signalId,
    hipStream_t stream);

void launchIbgdaPutWaitLocalBatch(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int numIters,
    unsigned long long* totalCycles,
    hipStream_t stream);

void launchIbgdaPutSignalWaitLocalBatch(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int signalId,
    int numIters,
    unsigned long long* totalCycles,
    hipStream_t stream);

void launchIbgdaSignalOnlyBatch(
    P2pIbgdaTransportDevice* transport,
    const IbgdaRemoteBuffer& remoteSignalBuf,
    int signalId,
    int numIters,
    unsigned long long* totalCycles,
    hipStream_t stream);

void launchIbgdaPutCqPollWaitBatch(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& localBuf,
    const IbgdaRemoteBuffer& remoteBuf,
    std::size_t nbytes,
    int numIters,
    unsigned long long* totalCycles,
    hipStream_t stream);

} // namespace pipes_gda::benchmark
