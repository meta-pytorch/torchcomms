// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#include "comms/prims/core/AbortCheck.cuh"
#include "comms/prims/transport/ibgda/IbgdaBuffer.h"

namespace comms::prims {
class P2pIbgdaTransportDevice;
} // namespace comms::prims

namespace comms::prims::benchmark {

inline constexpr uint32_t kDefaultIbgdaWarpProxyQueueDepth = 16;

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
 * @param maxSignalBytes Max bytes per signaled sub-chunk
 * @param abortDevice    Optional abortDevice for wait operations
 */
void launch_ibgda_send_recv(
    P2pIbgdaTransportDevice* transport,
    char* src,
    char* dst,
    std::size_t nbytes,
    int numBlocks,
    cudaStream_t stream,
    std::size_t maxSignalBytes = 0,
    AbortDevice abortDevice = AbortDevice());

/**
 * Launch bidirectional progress-send/recv kernel for IBGDA transport.
 *
 * Uses the resumable init/progress API and loops until each initialized
 * transfer reaches Done.
 */
void launch_ibgda_progress_send_recv(
    P2pIbgdaTransportDevice* transport,
    char* src,
    char* dst,
    std::size_t nbytes,
    int numBlocks,
    cudaStream_t stream,
    std::size_t maxSignalBytes = 0,
    AbortDevice abortDevice = AbortDevice());

/**
 * Launch bidirectional tile sendrecv kernel that performs two back-to-back
 * send()/recv() calls with independent maxSignalBytes values.
 */
void launch_ibgda_send_recv_two_call(
    P2pIbgdaTransportDevice* transport,
    char* src,
    char* dst,
    std::size_t firstBytes,
    std::size_t secondBytes,
    int numBlocks,
    std::size_t firstMaxSignalBytes,
    std::size_t secondMaxSignalBytes,
    cudaStream_t stream,
    AbortDevice abortDevice = AbortDevice());

/**
 * Launch unidirectional tile send kernel. All blocks send.
 */
void launch_ibgda_send(
    P2pIbgdaTransportDevice* transport,
    char* src,
    std::size_t nbytes,
    int numBlocks,
    cudaStream_t stream,
    std::size_t maxSignalBytes = 0,
    AbortDevice abortDevice = AbortDevice());

/**
 * Launch NVIDIA-only unidirectional send with one IB service warp per block.
 */
void launch_ibgda_warp_proxy_send(
    P2pIbgdaTransportDevice* transport,
    char* src,
    std::size_t nbytes,
    int numBlocks,
    cudaStream_t stream,
    std::size_t maxSignalBytes = 0,
    AbortDevice abortDevice = AbortDevice(),
    uint32_t queueDepth = kDefaultIbgdaWarpProxyQueueDepth);

/**
 * Launch unidirectional tile recv kernel. All blocks receive.
 */
void launch_ibgda_recv(
    P2pIbgdaTransportDevice* transport,
    char* dst,
    std::size_t nbytes,
    int numBlocks,
    cudaStream_t stream,
    std::size_t maxSignalBytes = 0,
    AbortDevice abortDevice = AbortDevice());

/**
 * Launch NVIDIA-only unidirectional receive with one IB service warp per block.
 */
void launch_ibgda_warp_proxy_recv(
    P2pIbgdaTransportDevice* transport,
    char* dst,
    std::size_t nbytes,
    int numBlocks,
    cudaStream_t stream,
    std::size_t maxSignalBytes = 0,
    AbortDevice abortDevice = AbortDevice(),
    uint32_t queueDepth = kDefaultIbgdaWarpProxyQueueDepth);

/**
 * Low-latency (LL) protocol counterparts of launch_ibgda_send_recv/send/recv.
 * Same grid layout; dispatch to the transport's send_ll/recv_ll (data + inline
 * flag, 2x wire, no DATA_READY wait; Memcpy only).
 */
void launch_ibgda_send_recv_ll(
    P2pIbgdaTransportDevice* transport,
    char* src,
    char* dst,
    std::size_t nbytes,
    int numBlocks,
    cudaStream_t stream,
    std::size_t maxSignalBytes = 0,
    AbortDevice abortDevice = AbortDevice());

void launch_ibgda_send_ll(
    P2pIbgdaTransportDevice* transport,
    char* src,
    std::size_t nbytes,
    int numBlocks,
    cudaStream_t stream,
    std::size_t maxSignalBytes = 0,
    AbortDevice abortDevice = AbortDevice());

void launch_ibgda_recv_ll(
    P2pIbgdaTransportDevice* transport,
    char* dst,
    std::size_t nbytes,
    int numBlocks,
    cudaStream_t stream,
    std::size_t maxSignalBytes = 0,
    AbortDevice abortDevice = AbortDevice());

/**
 * Drain outstanding bidirectional send/recv transport work for benchmark
 * measurement and safe teardown.
 */
void launch_ibgda_drain_send_recv(
    P2pIbgdaTransportDevice* transport,
    int numBlocks,
    std::size_t totalBytes,
    int iterations,
    cudaStream_t stream,
    AbortDevice abortDevice = AbortDevice());

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
    P2pIbgdaTransportDevice* transport,
    char* src,
    std::size_t nbytes,
    int numBlocks,
    cudaStream_t stream,
    std::size_t maxSignalBytes = 0,
    AbortDevice abortDevice = AbortDevice());

/**
 * Launch the staged progress sender and wait for the receiver's final credit.
 */
void launch_ibgda_progress_send_complete(
    P2pIbgdaTransportDevice* transport,
    char* src,
    std::size_t nbytes,
    int numBlocks,
    cudaStream_t stream,
    std::size_t maxSignalBytes = 0,
    AbortDevice abortDevice = AbortDevice());

/**
 * Launch a unidirectional registered-source progress send kernel.
 *
 * The source is read directly by the NIC. The kernel drains local NIC reads
 * and waits for the receiver's final slot-free credit before returning.
 */
void launch_ibgda_registered_progress_send(
    P2pIbgdaTransportDevice* transport,
    const IbgdaLocalBuffer& src,
    std::size_t nbytes,
    int numBlocks,
    cudaStream_t stream,
    std::size_t maxSignalBytes = 0,
    AbortDevice abortDevice = AbortDevice());

/**
 * Launch unidirectional progress recv kernel. All blocks receive.
 */
void launch_ibgda_progress_recv(
    P2pIbgdaTransportDevice* transport,
    char* dst,
    std::size_t nbytes,
    int numBlocks,
    cudaStream_t stream,
    std::size_t maxSignalBytes = 0,
    AbortDevice abortDevice = AbortDevice());

/**
 * Launch unidirectional LL progress send kernel. All blocks send.
 */
void launch_ibgda_progress_send_ll(
    P2pIbgdaTransportDevice* transport,
    char* src,
    std::size_t nbytes,
    int numBlocks,
    cudaStream_t stream,
    std::size_t maxSignalBytes = 0,
    AbortDevice abortDevice = AbortDevice());

/**
 * Launch unidirectional LL progress recv kernel. All blocks receive.
 */
void launch_ibgda_progress_recv_ll(
    P2pIbgdaTransportDevice* transport,
    char* dst,
    std::size_t nbytes,
    int numBlocks,
    cudaStream_t stream,
    std::size_t maxSignalBytes = 0,
    AbortDevice abortDevice = AbortDevice());

/**
 * Snapshot the transport send/recv byte cursors into device memory.
 *
 * @param transport  GPU-resident P2pIbgdaTransportDevice pointer
 * @param dst        Device buffer receiving `count` int64_t values
 * @param count      Number of byte cursor entries to copy
 * @param stream     CUDA stream
 */
void launch_ibgda_snapshot_step_state(
    P2pIbgdaTransportDevice* transport,
    int64_t* dst,
    int count,
    cudaStream_t stream);

} // namespace comms::prims::benchmark
