// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>

#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/pipes/ThreadGroup.cuh"

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

// Send kernel - groupScope selects warp vs block level parallelism
__global__ void p2pSend(
    P2pNvlTransportDevice* p2p,
    void* srcBuff,
    std::size_t nBytes,
    SyncScope groupScope = SyncScope::WARP,
    Timeout timeout = Timeout());

// Recv kernel
__global__ void p2pRecv(
    P2pNvlTransportDevice* p2p,
    void* dstBuff,
    std::size_t nBytes,
    SyncScope groupScope = SyncScope::WARP,
    Timeout timeout = Timeout());

// Timed versions that export GPU-side clock64() timing stats
__global__ void p2pSendTimed(
    P2pNvlTransportDevice* p2p,
    void* srcBuff,
    std::size_t nBytes,
    TimingStats* stats,
    SyncScope groupScope = SyncScope::WARP);

__global__ void p2pRecvTimed(
    P2pNvlTransportDevice* p2p,
    void* dstBuff,
    std::size_t nBytes,
    TimingStats* stats,
    SyncScope groupScope = SyncScope::WARP);

// Bidirectional kernel - half groups send, half groups receive
__global__ void p2pBidirectional(
    P2pNvlTransportDevice* p2p,
    void* sendBuff,
    void* recvBuff,
    std::size_t nBytes,
    SyncScope groupScope = SyncScope::WARP,
    Timeout timeout = Timeout());

// Signal benchmark kernel - ping-pong signaling pattern
__global__ void p2pSignalBenchKernel(
    P2pNvlTransportDevice* p2p,
    int nSteps,
    SyncScope groupScope = SyncScope::WARP);

// Send one kernel - single chunk transfer with metadata
__global__ void p2pSendOne(
    P2pNvlTransportDevice p2p,
    void* srcBuff,
    std::size_t nBytes,
    SyncScope groupScope = SyncScope::WARP);

// Recv one kernel - single chunk receive with metadata
__global__ void p2pRecvOne(
    P2pNvlTransportDevice p2p,
    void* dstBuff,
    SyncScope groupScope = SyncScope::WARP);

// Send multiple kernel - transfer multiple chunks with metadata
__global__ void p2pSendMultiple(
    P2pNvlTransportDevice p2p,
    void* srcBuff,
    DeviceSpan<const std::size_t> chunkSizes,
    DeviceSpan<const std::size_t> chunkIndices,
    SyncScope groupScope = SyncScope::WARP);

// Recv multiple kernel - receive multiple chunks with metadata
__global__ void p2pRecvMultiple(
    P2pNvlTransportDevice p2p,
    void* dstBuff,
    DeviceSpan<std::size_t> chunkSizes,
    SyncScope groupScope = SyncScope::WARP);

// =============================================================================
// LL128 benchmark kernels - warp-based LL128 protocol with inline flag
// signaling
// =============================================================================

// LL128 send kernel
__global__ void p2pLl128Send(
    P2pNvlTransportDevice p2p,
    void* srcBuff,
    std::size_t nBytes,
    int64_t flagValue,
    Timeout timeout = Timeout());

// LL128 recv kernel
__global__ void p2pLl128Recv(
    P2pNvlTransportDevice p2p,
    void* dstBuff,
    std::size_t nBytes,
    int64_t flagValue,
    Timeout timeout = Timeout());

// LL128 bidirectional kernel - half warps send, half recv
__global__ void p2pLl128Bidirectional(
    P2pNvlTransportDevice p2p,
    void* sendBuff,
    void* recvBuff,
    std::size_t nBytes,
    int64_t flagValue,
    Timeout timeout = Timeout());

} // namespace comms::pipes::benchmark
