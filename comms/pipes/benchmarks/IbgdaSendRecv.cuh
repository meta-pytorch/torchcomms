// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/pipes/P2pIbgdaTransportDevice.cuh"
#include "comms/pipes/TiledBuffer.cuh"
#include "comms/pipes/Timeout.cuh"

namespace comms::pipes::benchmark {

/**
 * Bidirectional tile sendrecv kernel for IBGDA transport.
 *
 * Grid: 2 * numBlocks (first half sends, second half receives).
 * Block: 512 threads.
 * Each sender block i is paired with receiver block i on the remote GPU.
 * Uses transport-managed staging buffers via send/recv.
 *
 * The kernel loops over sections of totalBytes, each dataBufferSize in size
 * (read from the transport's tile state). Within each section, TiledBuffer
 * partitions data into per-block tiles. Each tile fits in one perBlockSlotSize.
 */
__global__ void ibgda_send_recv_kernel(
    P2pIbgdaTransportDevice* transport,
    char* src,
    char* dst,
    std::size_t totalBytes,
    int numBlocks,
    Timeout timeout);

/**
 * Unidirectional tile send kernel. All blocks send.
 * Grid: numBlocks. Block: 512 threads.
 */
__global__ void ibgda_send_kernel(
    P2pIbgdaTransportDevice* transport,
    char* src,
    std::size_t totalBytes,
    int numBlocks,
    Timeout timeout);

/**
 * Unidirectional tile recv kernel. All blocks receive.
 * Grid: numBlocks. Block: 512 threads.
 */
__global__ void ibgda_recv_kernel(
    P2pIbgdaTransportDevice* transport,
    char* dst,
    std::size_t totalBytes,
    int numBlocks,
    Timeout timeout);

} // namespace comms::pipes::benchmark
