// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/benchmarks/IbgdaSendRecv.cuh"

#include <algorithm>

#include "comms/pipes/CopyUtils.cuh"
#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/TiledBuffer.cuh"
#include "comms/pipes/Timeout.cuh"

namespace comms::pipes::benchmark {

__global__ void __launch_bounds__(512, 1) ibgda_send_recv_kernel(
    P2pIbgdaTransportDevice* transport,
    char* src,
    char* dst,
    std::size_t totalBytes,
    int numBlocks,
    Timeout timeout) {
  auto group = make_block_group();

  // Partition blocks: first half sends, second half receives.
  auto [role, sub] = group.partition(2);
  const bool isSender = (role == 0);

  // Section size = transport's staging slot size (dataBufferSize).
  // Clamp to totalBytes for small transfers.
  const std::size_t sectionBytes =
      min(transport->send_recv_state().dataBufferSize, totalBytes);
  const std::size_t totalSections = totalBytes / sectionBytes;

  for (std::size_t s = 0; s < totalSections; ++s) {
    const std::size_t offset = s * sectionBytes;

    if (isSender) {
      TiledBuffer<char> tiles(src + offset, sectionBytes, sub);
      transport->send(sub, tiles.data(), tiles.bytes(), numBlocks, 0, timeout);
    } else {
      TiledBuffer<char> tiles(dst + offset, sectionBytes, sub);
      transport->recv(sub, tiles.data(), tiles.bytes(), numBlocks, 0, timeout);
    }
  }
}

void launch_ibgda_send_recv(
    P2pIbgdaTransportDevice* transport,
    char* src,
    char* dst,
    std::size_t nbytes,
    int numBlocks,
    cudaStream_t stream,
    Timeout timeout) {
  ibgda_send_recv_kernel<<<2 * numBlocks, 512, 0, stream>>>(
      transport, src, dst, nbytes, numBlocks, timeout);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("[PIPES] Kernel launch failed: %s\n", cudaGetErrorString(err));
  }
}

__global__ void __launch_bounds__(512, 1) ibgda_send_kernel(
    P2pIbgdaTransportDevice* transport,
    char* src,
    std::size_t totalBytes,
    int numBlocks,
    Timeout timeout) {
  auto group = make_block_group();

  const std::size_t sectionBytes =
      min(transport->send_recv_state().dataBufferSize, totalBytes);
  const std::size_t totalSections = totalBytes / sectionBytes;

  for (std::size_t s = 0; s < totalSections; ++s) {
    TiledBuffer<char> tiles(src + s * sectionBytes, sectionBytes, group);
    transport->send(group, tiles.data(), tiles.bytes(), numBlocks, 0, timeout);
  }
}

__global__ void __launch_bounds__(512, 1) ibgda_recv_kernel(
    P2pIbgdaTransportDevice* transport,
    char* dst,
    std::size_t totalBytes,
    int numBlocks,
    Timeout timeout) {
  auto group = make_block_group();

  const std::size_t sectionBytes =
      min(transport->send_recv_state().dataBufferSize, totalBytes);
  const std::size_t totalSections = totalBytes / sectionBytes;

  for (std::size_t s = 0; s < totalSections; ++s) {
    TiledBuffer<char> tiles(dst + s * sectionBytes, sectionBytes, group);
    transport->recv(group, tiles.data(), tiles.bytes(), numBlocks, 0, timeout);
  }
}

void launch_ibgda_send(
    P2pIbgdaTransportDevice* transport,
    char* src,
    std::size_t nbytes,
    int numBlocks,
    cudaStream_t stream,
    Timeout timeout) {
  ibgda_send_kernel<<<numBlocks, 512, 0, stream>>>(
      transport, src, nbytes, numBlocks, timeout);
}

void launch_ibgda_recv(
    P2pIbgdaTransportDevice* transport,
    char* dst,
    std::size_t nbytes,
    int numBlocks,
    cudaStream_t stream,
    Timeout timeout) {
  ibgda_recv_kernel<<<numBlocks, 512, 0, stream>>>(
      transport, dst, nbytes, numBlocks, timeout);
}

__global__ void ibgda_snapshot_step_state_kernel(
    P2pIbgdaTransportDevice* transport,
    int64_t* dst,
    int count) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    dst[idx] = transport->send_recv_state().stepState[idx];
  }
}

void launch_ibgda_snapshot_step_state(
    P2pIbgdaTransportDevice* transport,
    int64_t* dst,
    int count,
    cudaStream_t stream) {
  constexpr int kThreads = 256;
  const int blocks = (count + kThreads - 1) / kThreads;
  ibgda_snapshot_step_state_kernel<<<blocks, kThreads, 0, stream>>>(
      transport, dst, count);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf(
        "[PIPES] Step-state snapshot launch failed: %s\n",
        cudaGetErrorString(err));
  }
}

} // namespace comms::pipes::benchmark
