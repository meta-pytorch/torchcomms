// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/Transport.cuh"

namespace comms::pipes {

namespace {
/**
 * Debug helper to print allToAllv communication information.
 * Automatically detects self-copy vs peer communication based on my_rank ==
 * peer_rank.
 */
__device__ __forceinline__ void printPerPeerOperation(
    int my_rank_id,
    int peer_rank_id,
    int partition_id,
    uint32_t total_groups,
    size_t send_offset,
    size_t recv_offset,
    size_t send_nbytes,
    size_t recv_nbytes) {
  if (my_rank_id == peer_rank_id) {
    // Self-copy
    printf(
        "Rank=%d pid=%d total-groups=%d: self-copy send-offset=%lu, recv-offset=%lu nbytes=%lu\n",
        my_rank_id,
        partition_id,
        total_groups,
        send_offset,
        recv_offset,
        send_nbytes);
  } else {
    // Peer communication
    bool is_send = (partition_id == 0);
    size_t offset = is_send ? send_offset : recv_offset;
    size_t nbytes = is_send ? send_nbytes : recv_nbytes;
    printf(
        "Rank=%d pid=%d total-groups=%d: %s rank=%d offset=%lu nbytes=%lu\n",
        my_rank_id,
        partition_id,
        total_groups,
        is_send ? "send to" : "recv from",
        peer_rank_id,
        offset,
        nbytes);
  }
}
} // namespace

/**
 * Chunk metadata for allToAllv operation.
 * Describes a contiguous chunk of data to send or receive for a specific peer.
 */
struct ChunkInfo {
  std::size_t offset; // offset in bytes from buffer base address
  std::size_t nbytes; // number of bytes to send or recv

  __host__ __device__ __forceinline__
  ChunkInfo(std::size_t offset, std::size_t nbytes)
      : offset(offset), nbytes(nbytes) {}
};

/**
 * AllToAllv collective communication primitive.
 *
 * Performs variable-sized all-to-all data exchange among multiple ranks.
 * Each rank sends a potentially different amount of data to every other rank,
 * and receives a potentially different amount of data from every other rank.
 *
 * Algorithm:
 * 1. First weighted partition: Distribute warps across ranks based on total
 *    communication workload (send+recv bytes for peers, send only for self)
 * 2. For self-rank: Perform local memory copy within the same GPU
 * 3. For peer ranks: Second weighted partition to split warps between send
 *    and recv operations based on their respective data sizes
 * 4. Execute send or recv using P2P NVL transport
 *
 * Parameters:
 *   @param recvbuff_d: Device pointer to receive buffer
 *   @param sendbuff_d: Device pointer to send buffer
 *   @param my_rank_id: Current rank ID
 *   @param transports_per_rank: Array of transport objects, one per rank
 *                        (self-transport for my_rank, P2P for others)
 *   @param send_chunk_infos: Array of send chunk metadata, one per destination
 * rank
 *   @param recv_chunk_infos: Array of recv chunk metadata, one per source rank
 *
 * Requirements:
 * - Must be called from device code with sufficient threads
 * - transports_per_rank.size() == send_chunk_infos.size() ==
 *   recv_chunk_infos.size()
 * - send_chunk_infos[i].nbytes == recv_chunk_infos[i].nbytes for i ==
 *   my_rank_id
 * - Max 8 ranks supported (stack-allocated weights)
 */
__device__ __forceinline__ void allToAllv(
    void* recvbuff_d,
    const void* sendbuff_d,
    int my_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    DeviceSpan<ChunkInfo> send_chunk_infos,
    DeviceSpan<ChunkInfo> recv_chunk_infos
    // all arguments below will eventually come from communicator
) {
#ifdef __CUDA_ARCH__
  auto group = make_warp_group();
  const auto nranks = transports_per_rank.size();
  assert(nranks == send_chunk_infos.size());
  assert(nranks == recv_chunk_infos.size());

  // Single rank case - just do self-copy
  if (nranks == 1) {
    const auto& send_info = send_chunk_infos[my_rank_id];
    const auto& recv_info = recv_chunk_infos[my_rank_id];
    const char* src = static_cast<const char*>(sendbuff_d) + send_info.offset;
    char* dst = static_cast<char*>(recvbuff_d) + recv_info.offset;

    auto& transport = transports_per_rank[my_rank_id];
    assert(transport.type == TransportType::SELF);
    transport.self.write(group, dst, src, send_info.nbytes);
    return;
  }

  // 1. partition into per rank groups using partition
  auto [peer_rank_id, group_per_rank] = group.partition(nranks);

  if (peer_rank_id == my_rank_id) {
    // Self partition
    auto& transport = transports_per_rank[my_rank_id];
    assert(transport.type == TransportType::SELF);

    const auto& send_info = send_chunk_infos[my_rank_id];
    const auto& recv_info = recv_chunk_infos[my_rank_id];
    assert(send_info.nbytes == recv_info.nbytes);

    const char* src = static_cast<const char*>(sendbuff_d) + send_info.offset;
    char* dst = static_cast<char*>(recvbuff_d) + recv_info.offset;

#ifdef DEBUG_ALLTOALLV
    if (group_per_rank.is_global_leader()) {
      printPerPeerOperation(
          my_rank_id,
          peer_rank_id,
          peer_rank_id, // partition_id same as peer_rank_id for self
          group_per_rank.total_groups,
          send_info.offset,
          recv_info.offset,
          send_info.nbytes,
          recv_info.nbytes);
    }
#endif

    transport.self.write(group_per_rank, dst, src, send_info.nbytes);
    return;
  }

  // Peer communication
  // Partition into send/recv groups based on send/recv workload
  const auto& send_info = send_chunk_infos[peer_rank_id];
  const auto& recv_info = recv_chunk_infos[peer_rank_id];

  auto [partition_id, send_recv_group] = group_per_rank.partition(2);

  // Extract to local pointer to avoid aliasing: compiler can't prove that
  // operations on transport won't modify transports_per_rank.data_, forcing
  // reloads. Local variable is provably independent. See DeviceSpan.cuh:228.
  auto transports = transports_per_rank.data();
  auto& transport = transports[peer_rank_id];
  assert(transport.type == TransportType::P2P_NVL);

#ifdef DEBUG_ALLTOALLV
  if (send_recv_group.is_global_leader()) {
    printPerPeerOperation(
        my_rank_id,
        peer_rank_id,
        partition_id,
        send_recv_group.total_groups,
        send_info.offset,
        recv_info.offset,
        send_info.nbytes,
        recv_info.nbytes);
  }
#endif

  // Perform peer send/recv
  bool is_send = (partition_id == 0);
  if (is_send) {
    transport.p2p_nvl.send(
        send_recv_group,
        static_cast<char*>(const_cast<void*>(sendbuff_d)) + send_info.offset,
        send_info.nbytes);
  } else {
    transport.p2p_nvl.recv(
        send_recv_group,
        static_cast<char*>(recvbuff_d) + recv_info.offset,
        recv_info.nbytes);
  }

#endif
}

} // namespace comms::pipes
