// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include "comms/prims/core/TiledBuffer.cuh"
#include "comms/prims/core/Timeout.cuh"
#include "comms/prims/transport/ibgda/P2pIbgdaTransportDevice.cuh"

namespace comms::prims::benchmark {

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
    std::size_t maxSignalBytes,
    Timeout timeout);

#ifndef __HIP_PLATFORM_AMD__
__global__ void ibgda_progress_send_recv_kernel(
    P2pIbgdaTransportDevice* transport,
    char* src,
    char* dst,
    std::size_t totalBytes,
    int numBlocks,
    std::size_t maxSignalBytes,
    Timeout timeout);
#endif

__global__ void ibgda_send_recv_two_call_kernel(
    P2pIbgdaTransportDevice* transport,
    char* src,
    char* dst,
    std::size_t firstBytes,
    std::size_t secondBytes,
    int numBlocks,
    std::size_t firstMaxSignalBytes,
    std::size_t secondMaxSignalBytes,
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
    std::size_t maxSignalBytes,
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
    std::size_t maxSignalBytes,
    Timeout timeout);

#ifndef __HIP_PLATFORM_AMD__
/**
 * Unidirectional progress send kernel. All blocks send.
 * Grid: numBlocks. Block: 512 threads.
 */
__global__ void ibgda_progress_send_kernel(
    P2pIbgdaTransportDevice* transport,
    char* src,
    std::size_t totalBytes,
    int numBlocks,
    std::size_t maxSignalBytes,
    Timeout timeout);

/**
 * Unidirectional progress recv kernel. All blocks receive.
 * Grid: numBlocks. Block: 512 threads.
 */
__global__ void ibgda_progress_recv_kernel(
    P2pIbgdaTransportDevice* transport,
    char* dst,
    std::size_t totalBytes,
    int numBlocks,
    std::size_t maxSignalBytes,
    Timeout timeout);
#endif

} // namespace comms::prims::benchmark
