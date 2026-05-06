// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/benchmarks/IbgdaRecvForward.cuh"

#include "comms/pipes/ThreadGroup.cuh"

namespace comms::pipes::benchmark {

__global__ void __launch_bounds__(512, 1) ibgda_recv_forward_kernel(
    P2pIbgdaTransportDevice* prev_transport,
    P2pIbgdaTransportDevice* next_transport,
    char* src,
    char* dst,
    std::size_t totalBytes,
    int numBlocks,
    int my_rank,
    Timeout timeout) {
  auto group = make_block_group();

  const std::size_t sectionBytes = (my_rank == 1)
      ? min(prev_transport->send_recv_state().dataBufferSize, totalBytes)
      : min((my_rank == 0 ? next_transport : prev_transport)
                ->send_recv_state()
                .dataBufferSize,
            totalBytes);
  const std::size_t totalSections = totalBytes / sectionBytes;

  for (std::size_t s = 0; s < totalSections; ++s) {
    const std::size_t offset = s * sectionBytes;

    if (my_rank == 0) {
      TiledBuffer<char> tiles(src + offset, sectionBytes, group);
      next_transport->send(
          group, tiles.data(), tiles.bytes(), numBlocks, 0, timeout);
    } else if (my_rank == 2) {
      TiledBuffer<char> tiles(dst + offset, sectionBytes, group);
      prev_transport->recv(
          group, tiles.data(), tiles.bytes(), numBlocks, 0, timeout);
    } else {
      TiledBuffer<char> tiles(dst + offset, sectionBytes, group);
      prev_transport->recv_forward(
          group,
          tiles.data(),
          *next_transport,
          tiles.bytes(),
          numBlocks,
          0,
          timeout);
    }
  }
}

__global__ void __launch_bounds__(512, 1) ibgda_recv_send_kernel(
    P2pIbgdaTransportDevice* prev_transport,
    P2pIbgdaTransportDevice* next_transport,
    char* src,
    char* dst,
    std::size_t totalBytes,
    int numBlocks,
    int my_rank,
    Timeout timeout) {
  auto group = make_block_group();

  const std::size_t sectionBytes = (my_rank == 1)
      ? min(prev_transport->send_recv_state().dataBufferSize, totalBytes)
      : min((my_rank == 0 ? next_transport : prev_transport)
                ->send_recv_state()
                .dataBufferSize,
            totalBytes);
  const std::size_t totalSections = totalBytes / sectionBytes;

  for (std::size_t s = 0; s < totalSections; ++s) {
    const std::size_t offset = s * sectionBytes;

    if (my_rank == 0) {
      TiledBuffer<char> tiles(src + offset, sectionBytes, group);
      next_transport->send(
          group, tiles.data(), tiles.bytes(), numBlocks, 0, timeout);
    } else if (my_rank == 2) {
      TiledBuffer<char> tiles(dst + offset, sectionBytes, group);
      prev_transport->recv(
          group, tiles.data(), tiles.bytes(), numBlocks, 0, timeout);
    } else {
      TiledBuffer<char> tiles(dst + offset, sectionBytes, group);
      prev_transport->recv(
          group, tiles.data(), tiles.bytes(), numBlocks, 0, timeout);
      next_transport->send(
          group, tiles.data(), tiles.bytes(), numBlocks, 0, timeout);
    }
  }
}

void launch_ibgda_recv_forward_chain(
    P2pIbgdaTransportDevice* prev_transport,
    P2pIbgdaTransportDevice* next_transport,
    char* src,
    char* dst,
    std::size_t nbytes,
    int numBlocks,
    int my_rank,
    cudaStream_t stream,
    Timeout timeout) {
  ibgda_recv_forward_kernel<<<numBlocks, 512, 0, stream>>>(
      prev_transport,
      next_transport,
      src,
      dst,
      nbytes,
      numBlocks,
      my_rank,
      timeout);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf(
        "[PIPES] recv_forward kernel launch failed: %s\n",
        cudaGetErrorString(err));
  }
}

void launch_ibgda_recv_send_chain(
    P2pIbgdaTransportDevice* prev_transport,
    P2pIbgdaTransportDevice* next_transport,
    char* src,
    char* dst,
    std::size_t nbytes,
    int numBlocks,
    int my_rank,
    cudaStream_t stream,
    Timeout timeout) {
  ibgda_recv_send_kernel<<<numBlocks, 512, 0, stream>>>(
      prev_transport,
      next_transport,
      src,
      dst,
      nbytes,
      numBlocks,
      my_rank,
      timeout);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf(
        "[PIPES] recv_send kernel launch failed: %s\n",
        cudaGetErrorString(err));
  }
}

} // namespace comms::pipes::benchmark
