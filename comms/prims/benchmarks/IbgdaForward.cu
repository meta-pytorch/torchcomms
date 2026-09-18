// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/benchmarks/IbgdaForward.cuh"

#include "comms/prims/core/ThreadGroup.cuh"

namespace comms::prims::benchmark {

__global__ void __launch_bounds__(512, 1) ibgda_forward_kernel(
    P2pIbgdaTransportDevice* prev_transport,
    P2pIbgdaTransportDevice* next_transport,
    char* src,
    char* dst,
    std::size_t totalBytes,
    int numBlocks,
    int my_rank,
    AbortDevice abortDevice) {
  auto group = make_block_group();

  const std::size_t sectionBytes = (my_rank == 1)
      ? min(prev_transport->channel_layout().data_buffer_size(), totalBytes)
      : min((my_rank == 0 ? next_transport : prev_transport)
                ->channel_layout()
                .data_buffer_size(),
            totalBytes);
  const std::size_t totalSections = totalBytes / sectionBytes;

  for (std::size_t s = 0; s < totalSections; ++s) {
    const std::size_t offset = s * sectionBytes;

    if (my_rank == 0) {
      TiledBuffer<char> tiles(src + offset, sectionBytes, group);
      next_transport->send(group, tiles.data(), tiles.bytes(), 0, abortDevice);
    } else if (my_rank == 2) {
      TiledBuffer<char> tiles(dst + offset, sectionBytes, group);
      prev_transport->recv(group, tiles.data(), tiles.bytes(), 0, abortDevice);
    } else {
      TiledBuffer<char> tiles(dst + offset, sectionBytes, group);
      prev_transport->forward(
          group, tiles.data(), *next_transport, tiles.bytes(), 0, abortDevice);
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
    AbortDevice abortDevice) {
  auto group = make_block_group();

  const std::size_t sectionBytes = (my_rank == 1)
      ? min(prev_transport->channel_layout().data_buffer_size(), totalBytes)
      : min((my_rank == 0 ? next_transport : prev_transport)
                ->channel_layout()
                .data_buffer_size(),
            totalBytes);
  const std::size_t totalSections = totalBytes / sectionBytes;

  for (std::size_t s = 0; s < totalSections; ++s) {
    const std::size_t offset = s * sectionBytes;

    if (my_rank == 0) {
      TiledBuffer<char> tiles(src + offset, sectionBytes, group);
      next_transport->send(group, tiles.data(), tiles.bytes(), 0, abortDevice);
    } else if (my_rank == 2) {
      TiledBuffer<char> tiles(dst + offset, sectionBytes, group);
      prev_transport->recv(group, tiles.data(), tiles.bytes(), 0, abortDevice);
    } else {
      TiledBuffer<char> tiles(dst + offset, sectionBytes, group);
      prev_transport->recv(group, tiles.data(), tiles.bytes(), 0, abortDevice);
      next_transport->send(group, tiles.data(), tiles.bytes(), 0, abortDevice);
    }
  }
}

void launch_ibgda_forward_chain(
    P2pIbgdaTransportDevice* prev_transport,
    P2pIbgdaTransportDevice* next_transport,
    char* src,
    char* dst,
    std::size_t nbytes,
    int numBlocks,
    int my_rank,
    cudaStream_t stream,
    AbortDevice abortDevice) {
  ibgda_forward_kernel<<<numBlocks, 512, 0, stream>>>(
      prev_transport,
      next_transport,
      src,
      dst,
      nbytes,
      numBlocks,
      my_rank,
      abortDevice);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf(
        "[PIPES] forward kernel launch failed: %s\n", cudaGetErrorString(err));
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
    AbortDevice abortDevice) {
  ibgda_recv_send_kernel<<<numBlocks, 512, 0, stream>>>(
      prev_transport,
      next_transport,
      src,
      dst,
      nbytes,
      numBlocks,
      my_rank,
      abortDevice);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf(
        "[PIPES] recv_send kernel launch failed: %s\n",
        cudaGetErrorString(err));
  }
}

} // namespace comms::prims::benchmark
