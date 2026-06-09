// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/benchmarks/IbgdaSendRecv.cuh"

#include <algorithm>

#include "comms/prims/core/CopyUtils.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/TiledBuffer.cuh"
#include "comms/prims/core/Timeout.cuh"

namespace comms::prims::benchmark {

__global__ void __launch_bounds__(512, 1) ibgda_send_recv_kernel(
    P2pIbgdaTransportDevice* transport,
    char* src,
    char* dst,
    std::size_t totalBytes,
    int numBlocks,
    std::size_t maxSignalBytes,
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
      transport->send(
          sub, tiles.data(), tiles.bytes(), numBlocks, maxSignalBytes, timeout);
    } else {
      TiledBuffer<char> tiles(dst + offset, sectionBytes, sub);
      transport->recv(
          sub, tiles.data(), tiles.bytes(), numBlocks, maxSignalBytes, timeout);
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
    std::size_t maxSignalBytes,
    Timeout timeout) {
  ibgda_send_recv_kernel<<<2 * numBlocks, 512, 0, stream>>>(
      transport, src, dst, nbytes, numBlocks, maxSignalBytes, timeout);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("[PIPES] Kernel launch failed: %s\n", cudaGetErrorString(err));
  }
}

__global__ void __launch_bounds__(512, 1) ibgda_send_recv_two_call_kernel(
    P2pIbgdaTransportDevice* transport,
    char* src,
    char* dst,
    std::size_t firstBytes,
    std::size_t secondBytes,
    int numBlocks,
    std::size_t firstMaxSignalBytes,
    std::size_t secondMaxSignalBytes,
    Timeout timeout) {
  auto group = make_block_group();

  auto [role, sub] = group.partition(2);
  const bool isSender = (role == 0);

  if (isSender) {
    TiledBuffer<char> first(src, firstBytes, sub);
    transport->send(
        sub,
        first.data(),
        first.bytes(),
        numBlocks,
        firstMaxSignalBytes,
        timeout);
    TiledBuffer<char> second(src + firstBytes, secondBytes, sub);
    transport->send(
        sub,
        second.data(),
        second.bytes(),
        numBlocks,
        secondMaxSignalBytes,
        timeout);
  } else {
    TiledBuffer<char> first(dst, firstBytes, sub);
    transport->recv(
        sub,
        first.data(),
        first.bytes(),
        numBlocks,
        firstMaxSignalBytes,
        timeout);
    TiledBuffer<char> second(dst + firstBytes, secondBytes, sub);
    transport->recv(
        sub,
        second.data(),
        second.bytes(),
        numBlocks,
        secondMaxSignalBytes,
        timeout);
  }
}

void launch_ibgda_send_recv_two_call(
    P2pIbgdaTransportDevice* transport,
    char* src,
    char* dst,
    std::size_t firstBytes,
    std::size_t secondBytes,
    int numBlocks,
    std::size_t firstMaxSignalBytes,
    std::size_t secondMaxSignalBytes,
    cudaStream_t stream,
    Timeout timeout) {
  ibgda_send_recv_two_call_kernel<<<2 * numBlocks, 512, 0, stream>>>(
      transport,
      src,
      dst,
      firstBytes,
      secondBytes,
      numBlocks,
      firstMaxSignalBytes,
      secondMaxSignalBytes,
      timeout);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf(
        "[PIPES] two-call kernel launch failed: %s\n", cudaGetErrorString(err));
  }
}

__global__ void __launch_bounds__(512, 1) ibgda_send_kernel(
    P2pIbgdaTransportDevice* transport,
    char* src,
    std::size_t totalBytes,
    int numBlocks,
    std::size_t maxSignalBytes,
    Timeout timeout) {
  auto group = make_block_group();

  const std::size_t sectionBytes =
      min(transport->send_recv_state().dataBufferSize, totalBytes);
  const std::size_t totalSections = totalBytes / sectionBytes;

  for (std::size_t s = 0; s < totalSections; ++s) {
    TiledBuffer<char> tiles(src + s * sectionBytes, sectionBytes, group);
    transport->send(
        group, tiles.data(), tiles.bytes(), numBlocks, maxSignalBytes, timeout);
  }
}

__global__ void __launch_bounds__(512, 1) ibgda_recv_kernel(
    P2pIbgdaTransportDevice* transport,
    char* dst,
    std::size_t totalBytes,
    int numBlocks,
    std::size_t maxSignalBytes,
    Timeout timeout) {
  auto group = make_block_group();

  const std::size_t sectionBytes =
      min(transport->send_recv_state().dataBufferSize, totalBytes);
  const std::size_t totalSections = totalBytes / sectionBytes;

  for (std::size_t s = 0; s < totalSections; ++s) {
    TiledBuffer<char> tiles(dst + s * sectionBytes, sectionBytes, group);
    transport->recv(
        group, tiles.data(), tiles.bytes(), numBlocks, maxSignalBytes, timeout);
  }
}

void launch_ibgda_send(
    P2pIbgdaTransportDevice* transport,
    char* src,
    std::size_t nbytes,
    int numBlocks,
    cudaStream_t stream,
    std::size_t maxSignalBytes,
    Timeout timeout) {
  ibgda_send_kernel<<<numBlocks, 512, 0, stream>>>(
      transport, src, nbytes, numBlocks, maxSignalBytes, timeout);
}

void launch_ibgda_recv(
    P2pIbgdaTransportDevice* transport,
    char* dst,
    std::size_t nbytes,
    int numBlocks,
    cudaStream_t stream,
    std::size_t maxSignalBytes,
    Timeout timeout) {
  ibgda_recv_kernel<<<numBlocks, 512, 0, stream>>>(
      transport, dst, nbytes, numBlocks, maxSignalBytes, timeout);
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

} // namespace comms::prims::benchmark
