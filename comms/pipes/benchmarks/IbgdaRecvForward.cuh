// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/pipes/CopyOp.cuh"
#include "comms/pipes/P2pIbgdaTransportDevice.cuh"
#include "comms/pipes/TiledBuffer.cuh"
#include "comms/pipes/Timeout.cuh"

namespace comms::pipes::benchmark {

/**
 * 3-rank chain kernel using recv_forward on rank 1.
 * rank 0: send, rank 1: recv_forward, rank 2: recv.
 */
__global__ void ibgda_recv_forward_kernel(
    P2pIbgdaTransportDevice* prev_transport,
    P2pIbgdaTransportDevice* next_transport,
    char* src,
    char* dst,
    std::size_t totalBytes,
    int numBlocks,
    int my_rank,
    Timeout timeout);

/**
 * 3-rank chain kernel using recv + send on rank 1 (baseline).
 * rank 0: send, rank 1: recv then send, rank 2: recv.
 */
__global__ void ibgda_recv_send_kernel(
    P2pIbgdaTransportDevice* prev_transport,
    P2pIbgdaTransportDevice* next_transport,
    char* src,
    char* dst,
    std::size_t totalBytes,
    int numBlocks,
    int my_rank,
    Timeout timeout);

} // namespace comms::pipes::benchmark
