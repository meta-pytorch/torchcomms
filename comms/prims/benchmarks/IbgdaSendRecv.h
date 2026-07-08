// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#include "comms/prims/core/Timeout.cuh"
#include "comms/prims/transport/P2pIbTransportDeviceDecl.cuh"

namespace comms::prims::benchmark {

/**
 * Launch bidirectional tile sendrecv kernel for IB transport.
 *
 * Grid: 2 * numBlocks (first half sends, second half receives).
 * Block: 512 threads.
 *
 * @param transport  Backend-agnostic P2pIbTransportDevice handle
 * @param src        Source data buffer (device memory)
 * @param dst        Destination data buffer (device memory)
 * @param nbytes     Total bytes to transfer
 * @param numBlocks  Number of send blocks (= number of recv blocks)
 * @param stream     CUDA stream
 * @param maxSignalBytes Max bytes per signaled sub-chunk
 * @param timeout    Optional timeout for wait operations
 */
void launch_ibgda_send_recv(
    P2pIbTransportDevice transport,
    char* src,
    char* dst,
    std::size_t nbytes,
    int numBlocks,
    cudaStream_t stream,
    std::size_t maxSignalBytes = 0,
    Timeout timeout = Timeout());

/**
 * Launch bidirectional progress-send/recv kernel for IB transport.
 *
 * Uses the resumable init/progress API and loops until each initialized
 * transfer reaches Done.
 */
void launch_ibgda_progress_send_recv(
    P2pIbTransportDevice transport,
    char* src,
    char* dst,
    std::size_t nbytes,
    int numBlocks,
    cudaStream_t stream,
    std::size_t maxSignalBytes = 0,
    Timeout timeout = Timeout());

/**
 * Launch bidirectional tile sendrecv kernel that performs two back-to-back
 * send()/recv() calls with independent maxSignalBytes values.
 */
void launch_ibgda_send_recv_two_call(
    P2pIbTransportDevice transport,
    char* src,
    char* dst,
    std::size_t firstBytes,
    std::size_t secondBytes,
    int numBlocks,
    std::size_t firstMaxSignalBytes,
    std::size_t secondMaxSignalBytes,
    cudaStream_t stream,
    Timeout timeout = Timeout());

/**
 * Launch unidirectional tile send kernel. All blocks send.
 */
void launch_ibgda_send(
    P2pIbTransportDevice transport,
    char* src,
    std::size_t nbytes,
    int numBlocks,
    cudaStream_t stream,
    std::size_t maxSignalBytes = 0,
    Timeout timeout = Timeout());

/**
 * Launch unidirectional tile recv kernel. All blocks receive.
 */
void launch_ibgda_recv(
    P2pIbTransportDevice transport,
    char* dst,
    std::size_t nbytes,
    int numBlocks,
    cudaStream_t stream,
    std::size_t maxSignalBytes = 0,
    Timeout timeout = Timeout());

/**
 * Drain outstanding bidirectional send/recv transport work for benchmark
 * measurement and safe teardown.
 */
void launch_ibgda_drain_send_recv(
    P2pIbgdaTransportDevice* transport,
    int activeBlocks,
    std::size_t totalBytes,
    int iterations,
    cudaStream_t stream,
    Timeout timeout = Timeout());

/**
 * Reset benchmark-owned send/recv transport state after outstanding work has
 * been drained.
 */
void launch_ibgda_reset_send_recv(
    P2pIbgdaTransportDevice* transport,
    int maxGroups,
    cudaStream_t stream);

/**
 * Launch unidirectional progress send kernel. All blocks send.
 */
void launch_ibgda_progress_send(
    P2pIbTransportDevice transport,
    char* src,
    std::size_t nbytes,
    int numBlocks,
    cudaStream_t stream,
    std::size_t maxSignalBytes = 0,
    Timeout timeout = Timeout());

/**
 * Launch unidirectional progress recv kernel. All blocks receive.
 */
void launch_ibgda_progress_recv(
    P2pIbTransportDevice transport,
    char* dst,
    std::size_t nbytes,
    int numBlocks,
    cudaStream_t stream,
    std::size_t maxSignalBytes = 0,
    Timeout timeout = Timeout());

/**
 * Snapshot the transport send/recv byte cursors into device memory.
 *
 * @param transport  Backend-agnostic P2pIbTransportDevice handle
 * @param dst        Device buffer receiving `count` int64_t values
 * @param count      Number of byte cursor entries to copy
 * @param stream     CUDA stream
 */
void launch_ibgda_snapshot_step_state(
    P2pIbTransportDevice transport,
    int64_t* dst,
    int count,
    cudaStream_t stream);

} // namespace comms::prims::benchmark
