// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// AMD hybrid AllToAllv device kernel.
// Handles SELF + P2P_NVL (intra-node) + P2P_IBGDA_AMD (inter-node).
//
// Three-phase algorithm:
//   Phase 1: Leader posts IBGDA puts for all remote peers (non-blocking RDMA)
//   Phase 2: All threads handle NVL send/recv + self-copy for local peers
//   Phase 3: Leader fences IBGDA completions

#pragma once

#include <cstdint>
#include <cstdio>

#include "P2pIbgdaTransportDeviceAmd.h" // @manual
#include "comms/pipes/DeviceCheck.cuh"
#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/IbgdaBuffer.h"
#include "comms/pipes/Timeout.cuh"
#include "comms/pipes/Transport.cuh"
#include "comms/pipes/collectives/AllToAllv.cuh" // for ChunkInfo

namespace comms::pipes {

/**
 * AMD hybrid AllToAllv device function.
 *
 * Phase 1: IBGDA puts (leader thread only, non-blocking RDMA writes)
 * Phase 2: NVL send/recv + self-copy (all threads, pipelined)
 * Phase 3: IBGDA fence (leader thread only, wait for RDMA completion)
 */
__device__ __forceinline__ void all_to_allv_hybrid_amd(
    void* recvbuff_d,
    const void* sendbuff_d,
    int my_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    DeviceSpan<ChunkInfo> send_chunk_infos,
    DeviceSpan<ChunkInfo> recv_chunk_infos,
    Timeout timeout,
    // IBGDA parameters
    void* ibgda_transport_base,
    std::size_t ibgda_transport_stride,
    int* ibgda_peer_ranks,
    int num_ibgda_peers,
    IbgdaLocalBuffer ibgda_send_buf,
    IbgdaRemoteBuffer* ibgda_recv_bufs) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  auto group = make_warp_group();
  const auto nranks = transports_per_rank.size();
  PIPES_DEVICE_CHECK(nranks == send_chunk_infos.size());
  PIPES_DEVICE_CHECK(nranks == recv_chunk_infos.size());

  // ─── Phase 1: Post IBGDA puts for remote peers (leader only) ───────────
  // IBGDA puts are non-blocking: the NIC handles the RDMA transfer while
  // we proceed to Phase 2. This overlaps inter-node RDMA with intra-node
  // NVLink copies.
  if (group.is_global_leader() && num_ibgda_peers > 0) {
    char* sendbuf = static_cast<char*>(const_cast<void*>(sendbuff_d));
    for (int i = 0; i < num_ibgda_peers; i++) {
      int peer_rank = ibgda_peer_ranks[i];
      const auto& send_info = send_chunk_infos[peer_rank];

      auto* transport = reinterpret_cast<pipes_gda::P2pIbgdaTransportDevice*>(
          static_cast<char*>(ibgda_transport_base) +
          i * ibgda_transport_stride);

      IbgdaLocalBuffer src(
          sendbuf + send_info.offset, ibgda_send_buf.lkey_per_device);

      // Remote recv buf points to peer's recvbuff base. Add offset for
      // where this rank's data goes (= recv_chunk_infos[my_rank_id] on
      // the remote side, symmetric with our send_chunk_infos[my_rank_id]).
      IbgdaRemoteBuffer dst =
          ibgda_recv_bufs[i].subBuffer(send_chunk_infos[my_rank_id].offset);

      transport->put(src, dst, send_info.nbytes);
    }
  }

  // ─── Phase 2: NVL send/recv + self-copy for local peers ────────────────
  // Build local peer mapping (SELF + P2P_NVL only, max 9 entries)
  constexpr int kMaxLocalPeers = 9; // self + max 8 NVL peers
  int local_ranks[kMaxLocalPeers];
  int num_local = 0;
  for (uint32_t r = 0; r < nranks && num_local < kMaxLocalPeers; r++) {
    if (transports_per_rank[r].type == TransportType::SELF ||
        transports_per_rank[r].type == TransportType::P2P_NVL) {
      local_ranks[num_local++] = r;
    }
  }

  if (num_local == 1) {
    // Only self — just do self-copy
    int self_rank = local_ranks[0];
    const auto& send_info = send_chunk_infos[self_rank];
    const auto& recv_info = recv_chunk_infos[self_rank];
    const char* src = static_cast<const char*>(sendbuff_d) + send_info.offset;
    char* dst = static_cast<char*>(recvbuff_d) + recv_info.offset;

    auto& transport = transports_per_rank[self_rank];
    PIPES_DEVICE_CHECK(transport.type == TransportType::SELF);
    transport.self.put(group, dst, src, send_info.nbytes);
  } else if (num_local > 1) {
    // Multiple local peers — partition into send/recv, then by peer
    auto [partition_id, send_recv_group] = group.partition_interleaved(2);
    auto [local_peer_idx, group_per_peer] =
        send_recv_group.partition_interleaved(num_local);

    // Map local partition index back to global rank
    int peer_rank_id = local_ranks[local_peer_idx];

    if (peer_rank_id == my_rank_id) {
      // Self-copy (only partition 0 participates)
      auto& transport = transports_per_rank[my_rank_id];
      PIPES_DEVICE_CHECK(transport.type == TransportType::SELF);
      const auto& send_info = send_chunk_infos[my_rank_id];
      const auto& recv_info = recv_chunk_infos[my_rank_id];
      PIPES_DEVICE_CHECK(send_info.nbytes == recv_info.nbytes);

      const char* src = static_cast<const char*>(sendbuff_d) + send_info.offset;
      char* dst = static_cast<char*>(recvbuff_d) + recv_info.offset;
      if (partition_id == 0) {
        transport.self.put(group_per_peer, dst, src, send_info.nbytes);
      }
    } else {
      // NVL peer communication
      const auto& send_info = send_chunk_infos[peer_rank_id];
      const auto& recv_info = recv_chunk_infos[peer_rank_id];

      auto transports = transports_per_rank.data();
      auto& transport = transports[peer_rank_id];
      PIPES_DEVICE_CHECK(transport.type == TransportType::P2P_NVL);

      bool is_send = (partition_id == 0);
      if (is_send) {
        transport.p2p_nvl.send_group(
            group_per_peer,
            static_cast<char*>(const_cast<void*>(sendbuff_d)) +
                send_info.offset,
            send_info.nbytes,
            timeout);
      } else {
        transport.p2p_nvl.recv_group(
            group_per_peer,
            static_cast<char*>(recvbuff_d) + recv_info.offset,
            recv_info.nbytes,
            timeout);
      }
    }
  }

  // ─── Phase 3: Fence IBGDA completions (leader only) ────────────────────
  // Ensures all RDMA puts from Phase 1 are complete before kernel returns.
  if (group.is_global_leader() && num_ibgda_peers > 0) {
    for (int i = 0; i < num_ibgda_peers; i++) {
      auto* transport = reinterpret_cast<pipes_gda::P2pIbgdaTransportDevice*>(
          static_cast<char*>(ibgda_transport_base) +
          i * ibgda_transport_stride);
      transport->fence();
    }
  }

#endif
}

} // namespace comms::pipes
