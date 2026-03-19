// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include "comms/pipes/DeviceCheck.cuh"
#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/Timeout.cuh"
#include "comms/pipes/Transport.cuh"
#include "comms/pipes/collectives/AllToAllv.cuh" // ChunkInfo

namespace comms::pipes {

/**
 * AllToAllv collective using LL128 protocol.
 *
 * Uses fine-grained (128B packet) pipelining with inline flag signaling,
 * optimized for small/medium messages (<= 256KB per peer).
 *
 * Two differences from all_to_allv() (Simple protocol):
 * 1. Calls transport.p2p_nvl.ll128_send() / ll128_recv()
 * 2. No cluster launch (volatile stores bypass L1 cache)
 *
 * The sender always polls for READY_TO_WRITE before overwriting each buffer
 * slot. On first use this passes instantly (buffers are initialized to
 * READY_TO_WRITE). On subsequent iterations the sender blocks until the
 * receiver ACKs each slot.
 *
 * @param recvbuff_d Device pointer to receive buffer
 * @param sendbuff_d Device pointer to send buffer
 * @param my_rank_id Current rank ID
 * @param transports_per_rank Array of transport objects per rank
 * @param send_chunk_infos Array of send chunk metadata per destination rank
 * @param recv_chunk_infos Array of recv chunk metadata per source rank
 * @param timeout Timeout configuration
 */
__device__ __forceinline__ void all_to_allv_ll128(
    void* recvbuff_d,
    const void* sendbuff_d,
    int my_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    DeviceSpan<ChunkInfo> send_chunk_infos,
    DeviceSpan<ChunkInfo> recv_chunk_infos,
    Timeout timeout) {
#ifdef __CUDA_ARCH__
  auto group = make_warp_group();
  const auto nranks = transports_per_rank.size();
  PIPES_DEVICE_CHECK(nranks == send_chunk_infos.size());
  PIPES_DEVICE_CHECK(nranks == recv_chunk_infos.size());

  // Single rank case - just do self-copy
  if (nranks == 1) {
    const auto& send_info = send_chunk_infos[my_rank_id];
    const auto& recv_info = recv_chunk_infos[my_rank_id];
    PIPES_DEVICE_CHECK(send_info.nbytes == recv_info.nbytes);
    auto& transport = transports_per_rank[my_rank_id];
    PIPES_DEVICE_CHECK(transport.type == TransportType::SELF);
    transport.self.put(
        group,
        static_cast<char*>(recvbuff_d) + recv_info.offset,
        static_cast<const char*>(sendbuff_d) + send_info.offset,
        send_info.nbytes);
    return;
  }

  // 1. First partition by SEND/RECV using interleaved partitioning
  // partition_id: 0 = send, 1 = recv
  auto [partition_id, send_recv_group] = group.partition_interleaved(2);

  // 2. Then partition by PEERS using interleaved partitioning
  auto [peer_rank_id, group_per_peer] =
      send_recv_group.partition_interleaved(nranks);

  if (peer_rank_id == my_rank_id) {
    // Self partition - both send and recv groups participate in copying
    auto& transport = transports_per_rank[my_rank_id];
    PIPES_DEVICE_CHECK(transport.type == TransportType::SELF);
    const auto& send_info = send_chunk_infos[my_rank_id];
    const auto& recv_info = recv_chunk_infos[my_rank_id];
    PIPES_DEVICE_CHECK(send_info.nbytes == recv_info.nbytes);
    // Only one partition does the self-copy (match Simple protocol)
    if (partition_id == 0) {
      transport.self.put(
          group_per_peer,
          static_cast<char*>(recvbuff_d) + recv_info.offset,
          static_cast<const char*>(sendbuff_d) + send_info.offset,
          send_info.nbytes);
    }
    return;
  }

  // Peer communication via LL128
  const auto& send_info = send_chunk_infos[peer_rank_id];
  const auto& recv_info = recv_chunk_infos[peer_rank_id];

  // Extract to local pointer to avoid aliasing: compiler can't prove that
  // operations on transport won't modify transports_per_rank.data_, forcing
  // reloads. Local variable is provably independent. See DeviceSpan.cuh:228.
  auto transports = transports_per_rank.data();
  auto& transport = transports[peer_rank_id];
  PIPES_DEVICE_CHECK(transport.type == TransportType::P2P_NVL);

  if (partition_id == 0) {
    transport.p2p_nvl.ll128_send(
        group_per_peer,
        static_cast<const char*>(sendbuff_d) + send_info.offset,
        send_info.nbytes,
        timeout);
  } else {
    transport.p2p_nvl.ll128_recv(
        group_per_peer,
        static_cast<char*>(recvbuff_d) + recv_info.offset,
        recv_info.nbytes,
        timeout);
  }
#endif
}

} // namespace comms::pipes
