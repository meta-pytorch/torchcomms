// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>

#include "comms/prims/core/CopyOp.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/transport/ibgda/P2pIbgdaTransportDevice.cuh"

namespace comms::prims::test {

// Chain kernel: rank 0 sends, intermediates recv_forward, last rank receives.
__global__ void recv_forward_chain_kernel(
    P2pIbgdaTransportDevice** transports,
    const char* send_buf,
    char* recv_buf,
    std::size_t nbytes,
    int my_rank,
    int world_size,
    bool use_dst) {
  auto group = make_block_group();
  const auto num_blocks = gridDim.x;

  const std::size_t per_block = (nbytes / num_blocks) & ~15ULL;
  const std::size_t my_off = group.group_id * per_block;
  const std::size_t my_bytes =
      (group.group_id == num_blocks - 1) ? (nbytes - my_off) : per_block;

  const int prev_rank = (my_rank - 1 + world_size) % world_size;
  const int next_rank = (my_rank + 1) % world_size;

  if (my_rank == 0) {
    // First rank: send to next
    P2pIbgdaTransportDevice& next = *transports[next_rank];
    next.send(group, send_buf + my_off, my_bytes, num_blocks);
  } else if (my_rank == world_size - 1) {
    // Last rank: receive from prev
    P2pIbgdaTransportDevice& prev = *transports[prev_rank];
    prev.recv(group, recv_buf + my_off, my_bytes, num_blocks);
  } else {
    P2pIbgdaTransportDevice& prev = *transports[prev_rank];
    P2pIbgdaTransportDevice& next = *transports[next_rank];
    char* dst = use_dst ? (recv_buf + my_off) : nullptr;
    prev.forward(group, dst, next, my_bytes, num_blocks);
  }
}

void launch_recv_forward_chain(
    P2pIbgdaTransportDevice** transports,
    const char* send_buf,
    char* recv_buf,
    std::size_t nbytes,
    int my_rank,
    int world_size,
    int num_blocks,
    cudaStream_t stream) {
  recv_forward_chain_kernel<<<num_blocks, 128, 0, stream>>>(
      transports,
      send_buf,
      recv_buf,
      nbytes,
      my_rank,
      world_size,
      /*use_dst=*/true);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf(
        "recv_forward_chain kernel launch failed: %s\n",
        cudaGetErrorString(err));
  }
}

void launch_recv_forward_chain_no_dst(
    P2pIbgdaTransportDevice** transports,
    const char* send_buf,
    char* recv_buf,
    std::size_t nbytes,
    int my_rank,
    int world_size,
    int num_blocks,
    cudaStream_t stream) {
  recv_forward_chain_kernel<<<num_blocks, 128, 0, stream>>>(
      transports,
      send_buf,
      recv_buf,
      nbytes,
      my_rank,
      world_size,
      /*use_dst=*/false);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf(
        "recv_forward_chain_no_dst kernel launch failed: %s\n",
        cudaGetErrorString(err));
  }
}

// Resumable-forward variant of the chain. Intermediate ranks drive the
// yield-able init_forward_progress / progress_forward_once to completion on a
// single lane (loop until Done); endpoints keep blocking send()/recv(), which
// are wire-compatible (a send -> forward* -> recv chain is valid regardless of
// whether each forward is blocking or resumable). Byte-parity with the blocking
// chain validates the resumable forward's fused recv+reduce+send residency, the
// step-4-before-step-5 ordering, and the replay-safe yields (the dependency
// waits on later pipeline chunks exercise the real signal/counter/put paths).
__global__ void recv_forward_chain_progress_kernel(
    P2pIbgdaTransportDevice** transports,
    const char* send_buf,
    char* recv_buf,
    std::size_t nbytes,
    int my_rank,
    int world_size,
    bool use_dst) {
  auto group = make_block_group();
  const auto num_blocks = gridDim.x;

  const std::size_t per_block = (nbytes / num_blocks) & ~15ULL;
  const std::size_t my_off = group.group_id * per_block;
  const std::size_t my_bytes =
      (group.group_id == num_blocks - 1) ? (nbytes - my_off) : per_block;

  const int prev_rank = (my_rank - 1 + world_size) % world_size;
  const int next_rank = (my_rank + 1) % world_size;

  if (my_rank == 0) {
    // First rank: send to next (blocking endpoint).
    P2pIbgdaTransportDevice& next = *transports[next_rank];
    next.send(group, send_buf + my_off, my_bytes, num_blocks);
  } else if (my_rank == world_size - 1) {
    // Last rank: receive from prev (blocking endpoint).
    P2pIbgdaTransportDevice& prev = *transports[prev_rank];
    prev.recv(group, recv_buf + my_off, my_bytes, num_blocks);
  } else {
    // Intermediate rank: resumable forward (this=prev/recv, fwd=next/send).
    P2pIbgdaTransportDevice& prev = *transports[prev_rank];
    P2pIbgdaTransportDevice& next = *transports[next_rank];
    char* dst = use_dst ? (recv_buf + my_off) : nullptr;
    prev.init_forward_progress(group, next, my_bytes, num_blocks);
    IbgdaSendRecvProgressStatus status = IbgdaSendRecvProgressStatus::Waiting;
    do {
      status =
          prev.progress_forward_once(group, dst, next, my_bytes, num_blocks);
    } while (status != IbgdaSendRecvProgressStatus::Done);
  }
}

void launch_recv_forward_chain_progress(
    P2pIbgdaTransportDevice** transports,
    const char* send_buf,
    char* recv_buf,
    std::size_t nbytes,
    int my_rank,
    int world_size,
    int num_blocks,
    cudaStream_t stream) {
  recv_forward_chain_progress_kernel<<<num_blocks, 128, 0, stream>>>(
      transports,
      send_buf,
      recv_buf,
      nbytes,
      my_rank,
      world_size,
      /*use_dst=*/true);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf(
        "recv_forward_chain_progress kernel launch failed: %s\n",
        cudaGetErrorString(err));
  }
}

void launch_recv_forward_chain_progress_no_dst(
    P2pIbgdaTransportDevice** transports,
    const char* send_buf,
    char* recv_buf,
    std::size_t nbytes,
    int my_rank,
    int world_size,
    int num_blocks,
    cudaStream_t stream) {
  recv_forward_chain_progress_kernel<<<num_blocks, 128, 0, stream>>>(
      transports,
      send_buf,
      recv_buf,
      nbytes,
      my_rank,
      world_size,
      /*use_dst=*/false);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf(
        "recv_forward_chain_progress_no_dst kernel launch failed: %s\n",
        cudaGetErrorString(err));
  }
}

// Interleaved multi-lane resumable-forward chain. A SINGLE CUDA block per rank
// drives kLanes (2) independent transfers over distinct transport group_ids
// (0 and 1), round-robining progress_forward_once so a stalled lane yields to
// the other — the multiplexing that motivates the resumable API. Endpoints use
// blocking send/recv per lane (wire-compatible). active_blocks = kLanes. Data
// is split into 2 lane slices; byte-parity validates that two concurrent
// forwards on distinct group_ids do not corrupt each other's staging/cursors.
__global__ void recv_forward_chain_2lane_progress_kernel(
    P2pIbgdaTransportDevice** transports,
    const char* send_buf,
    char* recv_buf,
    std::size_t nbytes,
    int my_rank,
    int world_size,
    bool use_dst) {
  constexpr int kLanes = 2;
  const std::size_t lane0 = (nbytes / kLanes) & ~15ULL;
  const std::size_t laneBytes[kLanes] = {lane0, nbytes - lane0};
  const std::size_t laneOff[kLanes] = {0, lane0};

  // One block-scope group per lane, with distinct group_ids 0..kLanes-1 and
  // total_groups = kLanes (= active_blocks passed to the transport).
  ThreadGroup lanes[kLanes];
  for (int L = 0; L < kLanes; ++L) {
    lanes[L] = ThreadGroup{
        .thread_id_in_group = threadIdx.x,
        .group_size = blockDim.x,
        .group_id = static_cast<uint32_t>(L),
        .total_groups = static_cast<uint32_t>(kLanes),
        .scope = SyncScope::BLOCK};
  }

  const int prev_rank = (my_rank - 1 + world_size) % world_size;
  const int next_rank = (my_rank + 1) % world_size;

  if (my_rank == 0) {
    P2pIbgdaTransportDevice& next = *transports[next_rank];
    for (int L = 0; L < kLanes; ++L) {
      next.send(lanes[L], send_buf + laneOff[L], laneBytes[L], kLanes);
    }
  } else if (my_rank == world_size - 1) {
    P2pIbgdaTransportDevice& prev = *transports[prev_rank];
    for (int L = 0; L < kLanes; ++L) {
      prev.recv(lanes[L], recv_buf + laneOff[L], laneBytes[L], kLanes);
    }
  } else {
    P2pIbgdaTransportDevice& prev = *transports[prev_rank];
    P2pIbgdaTransportDevice& next = *transports[next_rank];
    char* dst[kLanes] = {
        use_dst ? recv_buf + laneOff[0] : nullptr,
        use_dst ? recv_buf + laneOff[1] : nullptr};
    for (int L = 0; L < kLanes; ++L) {
      prev.init_forward_progress(lanes[L], next, laneBytes[L], kLanes);
    }
    bool done[kLanes] = {laneBytes[0] == 0, laneBytes[1] == 0};
    while (!done[0] || !done[1]) {
      for (int L = 0; L < kLanes; ++L) {
        if (!done[L]) {
          done[L] = prev.progress_forward_once(
                        lanes[L], dst[L], next, laneBytes[L], kLanes) ==
              IbgdaSendRecvProgressStatus::Done;
        }
      }
    }
  }
}

void launch_recv_forward_chain_2lane_progress(
    P2pIbgdaTransportDevice** transports,
    const char* send_buf,
    char* recv_buf,
    std::size_t nbytes,
    int my_rank,
    int world_size,
    cudaStream_t stream) {
  // Single block per rank: the 2 lanes are multiplexed within one block.
  recv_forward_chain_2lane_progress_kernel<<<1, 128, 0, stream>>>(
      transports,
      send_buf,
      recv_buf,
      nbytes,
      my_rank,
      world_size,
      /*use_dst=*/true);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf(
        "recv_forward_chain_2lane_progress kernel launch failed: %s\n",
        cudaGetErrorString(err));
  }
}

} // namespace comms::prims::test
