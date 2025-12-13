// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>

#include "comms/pipes/AllToAllv.cuh"
#include "comms/pipes/P2pNvlTransportDevice.cuh"

namespace comms::pipes::benchmark {

/**
 * TimingStats - GPU-side timing statistics using clock64()
 *
 * Records cycle counts inside the kernel for accurate timing without
 * host-side overhead. Use with cudaDevAttrClockRate to convert to time.
 */
struct TimingStats {
  unsigned long long startCycle;
  unsigned long long endCycle;
  unsigned long long totalCycles;
};

// =============================================================================
// Benchmark kernels with configurable parallelism
// =============================================================================

// Send kernel - useBlockGroups selects warp vs block level parallelism
__global__ void p2pSend(
    P2pNvlTransportDevice p2p,
    void* srcBuff,
    std::size_t nBytes,
    bool useBlockGroups = false);

// Recv kernel
__global__ void p2pRecv(
    P2pNvlTransportDevice p2p,
    void* dstBuff,
    std::size_t nBytes,
    bool useBlockGroups = false);

// Timed versions that export GPU-side clock64() timing stats
__global__ void p2pSendTimed(
    P2pNvlTransportDevice p2p,
    void* srcBuff,
    std::size_t nBytes,
    TimingStats* stats,
    bool useBlockGroups = false);

__global__ void p2pRecvTimed(
    P2pNvlTransportDevice p2p,
    void* dstBuff,
    std::size_t nBytes,
    TimingStats* stats,
    bool useBlockGroups = false);

// Bidirectional kernel - half groups send, half groups receive
__global__ void p2pBidirectional(
    P2pNvlTransportDevice p2p,
    void* sendBuff,
    void* recvBuff,
    std::size_t nBytes,
    bool useBlockGroups = false);

/**
 * AllToAllv benchmark kernel.
 * All ranks participate in all-to-all communication with variable chunk sizes.
 */
__global__ void allToAllvKernel(
    void* recvbuff_d,
    const void* sendbuff_d,
    int my_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    DeviceSpan<ChunkInfo> send_chunk_infos,
    DeviceSpan<ChunkInfo> recv_chunk_infos);

} // namespace comms::pipes::benchmark
