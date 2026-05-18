// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// AMD hybrid AllToAllv: NVL (intra-node) + IBGDA (inter-node).
// Separate from the shared AllToAllv.h to avoid modifying shared Pipes code.

#pragma once

#include <hip/hip_runtime.h>
#include <chrono>

#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/IbgdaBuffer.h"
#include "comms/pipes/Transport.cuh"
#include "comms/pipes/collectives/AllToAllv.cuh" // for ChunkInfo

namespace comms::pipes {

/**
 * AMD hybrid AllToAllv — NVL for local peers, IBGDA for remote peers.
 *
 * For single-node (num_ibgda_peers == 0), behaves identically to the
 * standard all_to_allv. For multi-node, thread 0 posts IBGDA puts to
 * remote peers while all threads handle NVL send/recv for local peers.
 *
 * @param ibgda_transport_base  Base pointer to IBGDA device transports array
 * @param ibgda_transport_stride  Byte stride between IBGDA transport entries
 * @param ibgda_peer_ranks  Device array of global rank IDs for IBGDA peers
 * @param num_ibgda_peers  Number of IBGDA (remote) peers
 * @param ibgda_send_buf  Registered send buffer for RDMA
 * @param ibgda_recv_bufs  Device array of remote recv buffer descriptors
 */
void all_to_allv_amd(
    void* recvbuff_d,
    const void* sendbuff_d,
    int my_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    DeviceSpan<ChunkInfo> send_chunk_infos,
    DeviceSpan<ChunkInfo> recv_chunk_infos,
    // IBGDA parameters for remote peers
    void* ibgda_transport_base,
    std::size_t ibgda_transport_stride,
    int* ibgda_peer_ranks,
    int num_ibgda_peers,
    IbgdaLocalBuffer ibgda_send_buf,
    IbgdaRemoteBuffer* ibgda_recv_bufs,
    // Launch config
    hipStream_t stream = nullptr,
    int num_blocks = 16,
    int num_threads = 512);

} // namespace comms::pipes
