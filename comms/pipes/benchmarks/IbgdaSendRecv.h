// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#include "comms/pipes/Timeout.cuh"

namespace comms::pipes {
class P2pIbgdaTransportDevice;
} // namespace comms::pipes

namespace comms::pipes::benchmark {

/**
 * Launch bidirectional tile sendrecv kernel for IBGDA transport.
 *
 * Grid: 2 * numBlocks (first half sends, second half receives).
 * Block: 512 threads.
 *
 * @param transport  GPU-resident P2pIbgdaTransportDevice pointer
 * @param src        Source data buffer (device memory)
 * @param dst        Destination data buffer (device memory)
 * @param nbytes     Total bytes to transfer
 * @param numBlocks  Number of send blocks (= number of recv blocks)
 * @param stream     CUDA stream
 * @param timeout    Optional timeout for wait operations
 */
void launch_ibgda_send_recv(
    P2pIbgdaTransportDevice* transport,
    char* src,
    char* dst,
    std::size_t nbytes,
    int numBlocks,
    cudaStream_t stream,
    Timeout timeout = Timeout());

/**
 * Launch unidirectional tile send kernel. All blocks send.
 */
void launch_ibgda_send(
    P2pIbgdaTransportDevice* transport,
    char* src,
    std::size_t nbytes,
    int numBlocks,
    cudaStream_t stream,
    Timeout timeout = Timeout());

/**
 * Launch unidirectional tile recv kernel. All blocks receive.
 */
void launch_ibgda_recv(
    P2pIbgdaTransportDevice* transport,
    char* dst,
    std::size_t nbytes,
    int numBlocks,
    cudaStream_t stream,
    Timeout timeout = Timeout());

/**
 * Snapshot the transport send/recv step-state array into device memory.
 *
 * @param transport  GPU-resident P2pIbgdaTransportDevice pointer
 * @param dst        Device buffer receiving `count` int64_t values
 * @param count      Number of step-state entries to copy
 * @param stream     CUDA stream
 */
void launch_ibgda_snapshot_step_state(
    P2pIbgdaTransportDevice* transport,
    int64_t* dst,
    int count,
    cudaStream_t stream);

} // namespace comms::pipes::benchmark
