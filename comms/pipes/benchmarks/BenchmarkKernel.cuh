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
    P2pNvlTransportDevice p2p,
    void* srcBuff,
    std::size_t nBytes,
    SyncScope groupScope = SyncScope::WARP,
    Timeout timeout = Timeout());

// Recv kernel
__global__ void p2pRecv(
    P2pNvlTransportDevice p2p,
    void* dstBuff,
    std::size_t nBytes,
    SyncScope groupScope = SyncScope::WARP,
    Timeout timeout = Timeout());

// Timed versions that export GPU-side clock64() timing stats
__global__ void p2pSendTimed(
    P2pNvlTransportDevice p2p,
    void* srcBuff,
    std::size_t nBytes,
    TimingStats* stats,
    SyncScope groupScope = SyncScope::WARP);

__global__ void p2pRecvTimed(
    P2pNvlTransportDevice p2p,
    void* dstBuff,
    std::size_t nBytes,
    TimingStats* stats,
    SyncScope groupScope = SyncScope::WARP);

// Bidirectional kernel - half groups send, half groups receive
__global__ void p2pBidirectional(
    P2pNvlTransportDevice p2p,
    void* sendBuff,
    void* recvBuff,
    std::size_t nBytes,
    SyncScope groupScope = SyncScope::WARP,
    Timeout timeout = Timeout());

// Asymmetric recv - batches fine-grained ChunkStates into coarser recv chunks.
// recvChunkSize must be an exact multiple of the transport's chunkSize.
__global__ void p2pAsymmetricRecv(
    P2pNvlTransportDevice p2p,
    void* dstBuff,
    std::size_t nBytes,
    std::size_t recvChunkSize,
    SyncScope groupScope = SyncScope::WARP,
    Timeout timeout = Timeout());

// Asymmetric bidirectional - normal send (transport chunkSize) + batched recv
// (recvChunkSize). recvChunkSize must be an exact multiple of transport
// chunkSize.
__global__ void p2pAsymmetricBidirectional(
    P2pNvlTransportDevice p2p,
    void* sendBuff,
    void* recvBuff,
    std::size_t nBytes,
    std::size_t recvChunkSize,
    SyncScope groupScope = SyncScope::WARP,
    Timeout timeout = Timeout());

// Asymmetric send - batches fine-grained ChunkStates into coarser send chunks.
// sendChunkSize must be an exact multiple of the transport's chunkSize.
__global__ void p2pAsymmetricSend(
    P2pNvlTransportDevice p2p,
    void* srcBuff,
    std::size_t nBytes,
    std::size_t sendChunkSize,
    SyncScope groupScope = SyncScope::WARP,
    Timeout timeout = Timeout());

// Asymmetric send bidirectional - batched send (sendChunkSize) + normal recv
// (transport chunkSize). sendChunkSize must be an exact multiple of transport
// chunkSize.
__global__ void p2pAsymmetricSendBidirectional(
    P2pNvlTransportDevice p2p,
    void* sendBuff,
    void* recvBuff,
    std::size_t nBytes,
    std::size_t sendChunkSize,
    SyncScope groupScope = SyncScope::WARP,
    Timeout timeout = Timeout());

// Asymmetric both bidirectional - batched send (sendChunkSize) + batched recv
// (recvChunkSize). Both must be exact multiples of transport chunkSize.
__global__ void p2pAsymmetricBothBidirectional(
    P2pNvlTransportDevice p2p,
    void* sendBuff,
    void* recvBuff,
    std::size_t nBytes,
    std::size_t sendChunkSize,
    std::size_t recvChunkSize,
    SyncScope groupScope = SyncScope::WARP,
    Timeout timeout = Timeout());

// Signal benchmark kernel - ping-pong signaling between two peers
__global__ void p2pSignalBenchKernel(
    P2pNvlTransportDevice p2p,
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

// Stream send kernel - uses SendStream::for_each_slot API
__global__ void p2pStreamSend(
    P2pNvlTransportDevice p2p,
    void* srcBuff,
    std::size_t nBytes,
    SyncScope groupScope = SyncScope::WARP,
    Timeout timeout = Timeout());

// Stream recv kernel - uses RecvStream::for_each_ready_chunk API
__global__ void p2pStreamRecv(
    P2pNvlTransportDevice p2p,
    void* dstBuff,
    std::size_t nBytes,
    SyncScope groupScope = SyncScope::WARP,
    Timeout timeout = Timeout());

} // namespace comms::pipes::benchmark
