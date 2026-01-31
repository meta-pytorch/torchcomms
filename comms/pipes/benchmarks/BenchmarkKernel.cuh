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
// call_index must be globally unique across all send/recv calls (see
// P2pNvlTransportDevice.cuh)
__global__ void p2pSend(
    P2pNvlTransportDevice* p2p,
    void* srcBuff,
    std::size_t nBytes,
    uint32_t call_index,
    SyncScope groupScope = SyncScope::WARP,
    Timeout timeout = Timeout());

// Recv kernel
// call_index must be globally unique across all send/recv calls (see
// P2pNvlTransportDevice.cuh)
__global__ void p2pRecv(
    P2pNvlTransportDevice* p2p,
    void* dstBuff,
    std::size_t nBytes,
    uint32_t call_index,
    SyncScope groupScope = SyncScope::WARP,
    Timeout timeout = Timeout());

// Timed versions that export GPU-side clock64() timing stats
// call_index must be globally unique across all send/recv calls (see
// P2pNvlTransportDevice.cuh)
__global__ void p2pSendTimed(
    P2pNvlTransportDevice* p2p,
    void* srcBuff,
    std::size_t nBytes,
    uint32_t call_index,
    TimingStats* stats,
    SyncScope groupScope = SyncScope::WARP);

__global__ void p2pRecvTimed(
    P2pNvlTransportDevice* p2p,
    void* dstBuff,
    std::size_t nBytes,
    uint32_t call_index,
    TimingStats* stats,
    SyncScope groupScope = SyncScope::WARP);

// Bidirectional kernel - half groups send, half groups receive
// call_index must be globally unique across all send/recv calls (see
// P2pNvlTransportDevice.cuh)
__global__ void p2pBidirectional(
    P2pNvlTransportDevice* p2p,
    void* sendBuff,
    void* recvBuff,
    std::size_t nBytes,
    uint32_t call_index,
    SyncScope groupScope = SyncScope::WARP,
    Timeout timeout = Timeout());

// Signal benchmark kernel - ping-pong signaling pattern
__global__ void p2pSignalBenchKernel(
    P2pNvlTransportDevice* p2p,
    int nSteps,
    SyncScope groupScope = SyncScope::WARP);

} // namespace comms::pipes::benchmark
