// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/prims/core/CopyOp.cuh"
#include "comms/prims/core/TiledBuffer.cuh"
#include "comms/prims/core/Timeout.cuh"
#include "comms/prims/transport/ibgda/P2pIbgdaTransportDevice.cuh"

namespace comms::prims::benchmark {

/**
 * 3-rank chain kernel using forward on rank 1.
 * rank 0: send, rank 1: forward, rank 2: recv.
 */
__global__ void ibgda_forward_kernel(
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

} // namespace comms::prims::benchmark
