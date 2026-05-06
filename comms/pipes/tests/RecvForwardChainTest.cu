// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>

#include "comms/pipes/CopyOp.cuh"
#include "comms/pipes/P2pIbgdaTransportDevice.cuh"
#include "comms/pipes/ThreadGroup.cuh"

namespace comms::pipes::test {

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
    next.send<Memcpy>(group, send_buf + my_off, my_bytes, num_blocks);
  } else if (my_rank == world_size - 1) {
    // Last rank: receive from prev
    P2pIbgdaTransportDevice& prev = *transports[prev_rank];
    prev.recv<Memcpy>(group, recv_buf + my_off, my_bytes, num_blocks);
  } else {
    // Intermediate: recv_forward
    P2pIbgdaTransportDevice& prev = *transports[prev_rank];
    P2pIbgdaTransportDevice& next = *transports[next_rank];
    char* dst = use_dst ? (recv_buf + my_off) : nullptr;
    prev.recv_forward<Memcpy>(group, dst, next, my_bytes, num_blocks);
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

} // namespace comms::pipes::test
