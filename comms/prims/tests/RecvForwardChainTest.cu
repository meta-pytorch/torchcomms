// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/tests/RecvForwardChainTest.h"

#include <cuda_runtime.h>

#include <cstdint>

#include "comms/prims/core/CopyOp.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/transport/ibgda/P2pIbgdaTransportDevice.cuh"

namespace comms::prims::test {

// Chain kernel: rank 0 sends, intermediates recv_forward, last rank receives.
// `Proto` selects the wire format; every rank must instantiate the same one.
template <typename Proto>
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
    next.send<Memcpy, Proto>(group, send_buf + my_off, my_bytes);
  } else if (my_rank == world_size - 1) {
    // Last rank: receive from prev
    P2pIbgdaTransportDevice& prev = *transports[prev_rank];
    prev.recv<Memcpy, Proto>(group, recv_buf + my_off, my_bytes);
  } else {
    P2pIbgdaTransportDevice& prev = *transports[prev_rank];
    P2pIbgdaTransportDevice& next = *transports[next_rank];
    char* dst = use_dst ? (recv_buf + my_off) : nullptr;
    prev.forward<Memcpy, Proto>(group, dst, next, my_bytes);
  }
}

namespace {

template <typename Proto>
void launch_typed(
    P2pIbgdaTransportDevice** transports,
    const char* send_buf,
    char* recv_buf,
    std::size_t nbytes,
    int my_rank,
    int world_size,
    int num_blocks,
    cudaStream_t stream,
    bool use_dst) {
  recv_forward_chain_kernel<Proto><<<num_blocks, 128, 0, stream>>>(
      transports, send_buf, recv_buf, nbytes, my_rank, world_size, use_dst);
}

// Runtime enum -> compile-time tag. The protocol is a template parameter all
// the way down to the transport seam, so the choice has to be resolved here.
void launch_chain(
    P2pIbgdaTransportDevice** transports,
    const char* send_buf,
    char* recv_buf,
    std::size_t nbytes,
    int my_rank,
    int world_size,
    int num_blocks,
    cudaStream_t stream,
    ChainProto proto,
    bool use_dst,
    const char* what) {
  if (proto == ChainProto::LL) {
    launch_typed<protocol::LL>(
        transports,
        send_buf,
        recv_buf,
        nbytes,
        my_rank,
        world_size,
        num_blocks,
        stream,
        use_dst);
  } else {
    launch_typed<protocol::Simple>(
        transports,
        send_buf,
        recv_buf,
        nbytes,
        my_rank,
        world_size,
        num_blocks,
        stream,
        use_dst);
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("%s kernel launch failed: %s\n", what, cudaGetErrorString(err));
  }
}

} // namespace

void launch_recv_forward_chain(
    P2pIbgdaTransportDevice** transports,
    const char* send_buf,
    char* recv_buf,
    std::size_t nbytes,
    int my_rank,
    int world_size,
    int num_blocks,
    cudaStream_t stream,
    ChainProto proto) {
  launch_chain(
      transports,
      send_buf,
      recv_buf,
      nbytes,
      my_rank,
      world_size,
      num_blocks,
      stream,
      proto,
      /*use_dst=*/true,
      "recv_forward_chain");
}

void launch_recv_forward_chain_no_dst(
    P2pIbgdaTransportDevice** transports,
    const char* send_buf,
    char* recv_buf,
    std::size_t nbytes,
    int my_rank,
    int world_size,
    int num_blocks,
    cudaStream_t stream,
    ChainProto proto) {
  launch_chain(
      transports,
      send_buf,
      recv_buf,
      nbytes,
      my_rank,
      world_size,
      num_blocks,
      stream,
      proto,
      /*use_dst=*/false,
      "recv_forward_chain_no_dst");
}

} // namespace comms::prims::test
