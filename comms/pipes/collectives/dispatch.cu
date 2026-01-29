// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/collectives/dispatch.cuh"
#include "comms/pipes/collectives/dispatch.h"

#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/tests/Checks.h"

namespace comms::pipes::collectives {

// Round-robin tournament pairing: compute peer for a given round.
// Uses rotation method: fix rank 0, rotate ranks 1..n-1 counterclockwise.
// For round r: rank 0 pairs with rank at position 0, others fold/pair
// opposites.
__device__ __forceinline__ int
getPeerForRound(int my_rank, int n_ranks, int round) {
  int n_minus_1 = n_ranks - 1;

  if (my_rank == 0) {
    // Rank 0 pairs with rank at position 0 after rotation
    return ((n_minus_1 - round) % n_minus_1) + 1;
  }

  // Find my position in the rotated list
  // Position p contains rank ((p - round + n_minus_1) % n_minus_1) + 1
  // So for rank r: position = (r - 1 + round) % n_minus_1
  int my_pos = (my_rank - 1 + round) % n_minus_1;

  if (my_pos == 0) {
    // I'm at position 0, pair with rank 0
    return 0;
  }

  // Find opposite position in the fold (positions 1..n-2 pair with n-1-pos)
  int opposite_pos = n_minus_1 - my_pos;

  // Find rank at opposite position
  int peer_rank = ((opposite_pos - round + n_minus_1) % n_minus_1) + 1;

  return peer_rank;
}

// Copy chunks destined for self from send buffer to recv buffer.
template <typename WarpGroup>
__device__ __forceinline__ void handleSelfCopy(
    WarpGroup& group,
    Transport& selfTransport,
    const void* sendbuff_d,
    void* recv_buffer,
    const std::size_t* input_chunk_sizes_d,
    std::size_t input_chunk_sizes_count,
    const std::size_t* self_indices,
    std::size_t self_count,
    std::size_t* self_output_sizes) {
  const char* src_base = static_cast<const char*>(sendbuff_d);
  char* dst_base = static_cast<char*>(recv_buffer);

  for (std::size_t i = 0; i < self_count; i++) {
    std::size_t chunk_idx = self_indices[i];
    std::size_t chunk_size = input_chunk_sizes_d[chunk_idx];

    // Calculate offset by summing sizes of chunks before this one
    std::size_t chunk_offset = 0;
    for (std::size_t j = 0; j < chunk_idx; j++) {
      chunk_offset += input_chunk_sizes_d[j];
    }

    selfTransport.self.put(
        group, dst_base + chunk_offset, src_base + chunk_offset, chunk_size);
  }

  // Write output chunk sizes for self (copy all chunk sizes)
  selfTransport.self.put(
      group,
      reinterpret_cast<char*>(self_output_sizes),
      reinterpret_cast<const char*>(input_chunk_sizes_d),
      input_chunk_sizes_count * sizeof(std::size_t));
}

/**
 * Vertical sharding dispatch kernel.
 *
 * Uses vertical sharding with round-robin tournament scheduling:
 * All warps work on one peer at a time, cycling through all peers sequentially.
 * This maximizes SM utilization per operation but processes peers sequentially.
 *
 * Best for: Large messages, imbalanced workloads where full SM power per
 *           operation is beneficial.
 */
__global__ void dispatchKernelVertical(
    DeviceSpan<Transport> transports,
    int my_rank,
    const void* sendbuff_d,
    DeviceSpan<void* const> recvbuffs,
    DeviceSpan<const std::size_t> input_chunk_sizes,
    const std::size_t* input_chunk_indices_d,
    DeviceSpan<const std::size_t> input_chunk_indices_count_per_rank,
    DeviceSpan<std::size_t> output_chunk_sizes_per_rank) {
  auto group = make_warp_group();

  // Extract raw pointers to avoid aliasing issues (see DeviceSpan.cuh)
  Transport* const transports_ptr = transports.data();
  void* const* const recvbuffs_ptr = recvbuffs.data();
  const std::size_t* const input_chunk_sizes_ptr = input_chunk_sizes.data();
  const std::size_t* const indices_count_per_rank_ptr =
      input_chunk_indices_count_per_rank.data();
  std::size_t* const output_chunk_sizes_ptr =
      output_chunk_sizes_per_rank.data();
  const int n_ranks = static_cast<int>(transports.size());
  const std::size_t input_chunk_sizes_count = input_chunk_sizes.size();

  // VERTICAL SHARDING + ROUND-ROBIN SCHEDULING: Maximizes SM utilization.
  //
  // Vertical sharding: All warps process the same peer operation at each step,
  // ensuring all SMs contribute to the current send/recv. In contrast,
  // horizontal sharding (partitioning warps across different peer operations)
  // can leave warps idle when workloads are imbalanced.
  //
  // Round-robin tournament: Ensures all ranks are active at every step with
  // matched send/recv pairs. For n ranks: 2*(n-1) steps, n/2 parallel pairs.
  //
  // Example with 4 ranks (all ranks active every step, 2 parallel pairs):
  //   Step 0: R0->R1, R2->R3   Step 1: R1->R0, R3->R2   (round 0)
  //   Step 2: R0->R3, R1->R2   Step 3: R3->R0, R2->R1   (round 1)
  //   Step 4: R0->R2, R1->R3   Step 5: R2->R0, R3->R1   (round 2)

  int num_rounds = n_ranks - 1;
  int num_steps = 2 * num_rounds;

  for (int step = 0; step < num_steps; step++) {
    int round = step / 2;
    bool first_direction = (step % 2 == 0);

    int peer = getPeerForRound(my_rank, n_ranks, round);
    Transport& peerTransport = transports_ptr[peer];

    // Within each pair, lower rank sends first (on even step), higher recvs
    // On odd step, roles are reversed
    bool is_send = (my_rank < peer) ? first_direction : !first_direction;

    if (is_send) {
      // Compute offset for peer by summing counts for ranks 0..peer-1
      std::size_t send_offset = 0;
      for (int r = 0; r < peer; r++) {
        send_offset += indices_count_per_rank_ptr[r];
      }
      std::size_t send_count = indices_count_per_rank_ptr[peer];
      const std::size_t* send_indices = input_chunk_indices_d + send_offset;

      peerTransport.p2p_nvl.send_multiple(
          group,
          sendbuff_d,
          DeviceSpan<const std::size_t>(
              input_chunk_sizes_ptr, input_chunk_sizes_count),
          DeviceSpan<const std::size_t>(send_indices, send_count));
    } else {
      void* recv_buffer = recvbuffs_ptr[peer];
      std::size_t* recv_output_sizes =
          output_chunk_sizes_ptr + peer * input_chunk_sizes_count;

      peerTransport.p2p_nvl.recv_multiple(
          group,
          recv_buffer,
          DeviceSpan<std::size_t>(recv_output_sizes, input_chunk_sizes_count));
    }
  }

  // Handle self-copy using self transport (after peer communication)
  Transport& selfTransport = transports_ptr[my_rank];
  std::size_t self_offset = 0;
  for (int r = 0; r < my_rank; r++) {
    self_offset += indices_count_per_rank_ptr[r];
  }
  std::size_t self_count = indices_count_per_rank_ptr[my_rank];
  const std::size_t* self_indices = input_chunk_indices_d + self_offset;

  handleSelfCopy(
      group,
      selfTransport,
      sendbuff_d,
      recvbuffs_ptr[my_rank],
      input_chunk_sizes_ptr,
      input_chunk_sizes_count,
      self_indices,
      self_count,
      output_chunk_sizes_ptr + my_rank * input_chunk_sizes_count);
}

/**
 * Horizontal sharding dispatch kernel.
 *
 * Uses horizontal sharding like alltoallv: partition warps across all peers
 * first, then partition each peer's warps into send/recv groups for
 * simultaneous operation. This enables all peer communications to happen
 * in parallel with smaller thread groups per peer.
 *
 * Comparison with vertical sharding:
 * - Vertical: All warps work on one peer at a time (sequential across peers)
 * - Horizontal: Warps distributed across all peers (parallel across peers)
 */
__global__ void dispatchKernelHorizontal(
    DeviceSpan<Transport> transports,
    int my_rank,
    const void* sendbuff_d,
    DeviceSpan<void* const> recvbuffs,
    DeviceSpan<const std::size_t> input_chunk_sizes,
    const std::size_t* input_chunk_indices_d,
    DeviceSpan<const std::size_t> input_chunk_indices_count_per_rank,
    DeviceSpan<std::size_t> output_chunk_sizes_per_rank) {
  auto group = make_warp_group();

  // Extract raw pointers to avoid aliasing issues (see DeviceSpan.cuh)
  Transport* const transports_ptr = transports.data();
  void* const* const recvbuffs_ptr = recvbuffs.data();
  const std::size_t* const input_chunk_sizes_ptr = input_chunk_sizes.data();
  const std::size_t* const indices_count_per_rank_ptr =
      input_chunk_indices_count_per_rank.data();
  std::size_t* const output_chunk_sizes_ptr =
      output_chunk_sizes_per_rank.data();
  const int n_ranks = static_cast<int>(transports.size());
  const std::size_t input_chunk_sizes_count = input_chunk_sizes.size();

  // HORIZONTAL SHARDING: Partition warps across all peers for parallel
  // processing.
  //
  // Step 1: Partition warps across n_ranks peers
  // Step 2: For self-rank: handle self-copy
  // Step 3: For peer ranks: partition into send/recv groups (simultaneous)
  //
  // Example with 8 ranks, 32 warps:
  //   - First partition: 4 warps per rank
  //   - For peer ranks: 2 warps send, 2 warps recv (simultaneous)

  auto [peer_rank, group_per_rank] = group.partition(n_ranks);

  if (peer_rank == my_rank) {
    // Self partition: handle self-copy
    Transport& selfTransport = transports_ptr[my_rank];

    // Compute offset for self by summing counts for ranks 0..my_rank-1
    std::size_t self_offset = 0;
    for (int r = 0; r < my_rank; r++) {
      self_offset += indices_count_per_rank_ptr[r];
    }
    std::size_t self_count = indices_count_per_rank_ptr[my_rank];
    const std::size_t* self_indices = input_chunk_indices_d + self_offset;

    handleSelfCopy(
        group_per_rank,
        selfTransport,
        sendbuff_d,
        recvbuffs_ptr[my_rank],
        input_chunk_sizes_ptr,
        input_chunk_sizes_count,
        self_indices,
        self_count,
        output_chunk_sizes_ptr + my_rank * input_chunk_sizes_count);
    return;
  }

  // Peer communication: partition into send/recv groups
  auto [partition_id, send_recv_group] = group_per_rank.partition(2);

  Transport& peerTransport = transports_ptr[peer_rank];

  if (partition_id == 0) {
    // Send group
    // Compute offset for peer by summing counts for ranks 0..peer-1
    std::size_t send_offset = 0;
    for (int r = 0; r < peer_rank; r++) {
      send_offset += indices_count_per_rank_ptr[r];
    }
    std::size_t send_count = indices_count_per_rank_ptr[peer_rank];
    const std::size_t* send_indices = input_chunk_indices_d + send_offset;

    peerTransport.p2p_nvl.send_multiple(
        send_recv_group,
        sendbuff_d,
        DeviceSpan<const std::size_t>(
            input_chunk_sizes_ptr, input_chunk_sizes_count),
        DeviceSpan<const std::size_t>(send_indices, send_count));
  } else {
    // Recv group
    void* recv_buffer = recvbuffs_ptr[peer_rank];
    std::size_t* recv_output_sizes =
        output_chunk_sizes_ptr + peer_rank * input_chunk_sizes_count;

    peerTransport.p2p_nvl.recv_multiple(
        send_recv_group,
        recv_buffer,
        DeviceSpan<std::size_t>(recv_output_sizes, input_chunk_sizes_count));
  }
}

void dispatch(
    // Outputs
    DeviceSpan<void* const> recvbuffs,
    DeviceSpan<std::size_t> output_chunk_sizes_per_rank,
    // Inputs
    DeviceSpan<Transport> transports,
    int my_rank,
    const void* sendbuff_d,
    DeviceSpan<const std::size_t> input_chunk_sizes,
    const std::size_t* input_chunk_indices_d,
    DeviceSpan<const std::size_t> input_chunk_indices_count_per_rank,
    cudaStream_t stream,
    int num_blocks,
    int num_threads,
    ShardingMode mode) {
  switch (mode) {
    case ShardingMode::VERTICAL:
      dispatchKernelVertical<<<num_blocks, num_threads, 0, stream>>>(
          transports,
          my_rank,
          sendbuff_d,
          recvbuffs,
          input_chunk_sizes,
          input_chunk_indices_d,
          input_chunk_indices_count_per_rank,
          output_chunk_sizes_per_rank);
      break;
    case ShardingMode::HORIZONTAL:
      dispatchKernelHorizontal<<<num_blocks, num_threads, 0, stream>>>(
          transports,
          my_rank,
          sendbuff_d,
          recvbuffs,
          input_chunk_sizes,
          input_chunk_indices_d,
          input_chunk_indices_count_per_rank,
          output_chunk_sizes_per_rank);
      break;
  }
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::pipes::collectives
