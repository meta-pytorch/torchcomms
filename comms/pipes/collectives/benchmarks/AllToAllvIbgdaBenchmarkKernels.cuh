// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>

#include <cuda_runtime.h>

namespace comms::pipes {
class P2pIbgdaTransportDevice;
} // namespace comms::pipes

namespace comms::pipes::benchmark {

void launchIbgdaSendBench(
    P2pIbgdaTransportDevice* transport,
    void* sendBuf,
    std::size_t nbytes,
    int numIters,
    cudaStream_t stream,
    int numBlocks = 1,
    int numThreads = 256);

void launchIbgdaRecvBench(
    P2pIbgdaTransportDevice* transport,
    void* recvBuf,
    std::size_t nbytes,
    int numIters,
    cudaStream_t stream,
    int numBlocks = 1,
    int numThreads = 256);

} // namespace comms::pipes::benchmark
