// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/prims/benchmarks/IbgdaSendRecv.cuh"

#include <algorithm>

#include "comms/prims/core/CopyUtils.cuh"
#include "comms/prims/core/ThreadGroup.cuh"
#include "comms/prims/core/TiledBuffer.cuh"
#include "comms/prims/core/Timeout.cuh"

namespace comms::prims::benchmark {

namespace {

__device__ __forceinline__ std::size_t benchmark_align_protocol_bytes(
    std::size_t nbytes) {
  return (nbytes + 15ULL) & ~15ULL;
}

__device__ __forceinline__ std::size_t section_bytes(
    P2pIbgdaTransportDevice* transport,
    std::size_t totalBytes) {
  return min(transport->channel_layout().data_buffer_size(), totalBytes);
}

__device__ __forceinline__ std::size_t
protocol_bytes_for_tiled_group_per_launch(
    std::size_t totalBytes,
    int numBlocks,
    std::size_t slotSize,
    int groupId) {
  const std::size_t sectionBytes = min(slotSize, totalBytes);
  if (sectionBytes == 0) {
    return 0;
  }

  const std::size_t totalSections = totalBytes / sectionBytes;
  const std::size_t tileElements =
      (((sectionBytes + numBlocks - 1) / numBlocks) + 15ULL) & ~15ULL;
  const std::size_t tileOffset = groupId * tileElements;
  if (tileOffset >= sectionBytes) {
    return 0;
  }

  const std::size_t tileBytes = min(tileElements, sectionBytes - tileOffset);
  return totalSections * benchmark_align_protocol_bytes(tileBytes);
}

} // namespace

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
  const std::size_t sectionBytes = section_bytes(transport, totalBytes);
  const std::size_t totalSections = totalBytes / sectionBytes;

  for (std::size_t s = 0; s < totalSections; ++s) {
    const std::size_t offset = s * sectionBytes;

    if (isSender) {
      TiledBuffer<char> tiles(src + offset, sectionBytes, sub);
      transport->send(
          sub, tiles.data(), tiles.bytes(), maxSignalBytes, timeout);
    } else {
      TiledBuffer<char> tiles(dst + offset, sectionBytes, sub);
      transport->recv(
          sub, tiles.data(), tiles.bytes(), maxSignalBytes, timeout);
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

#ifndef __HIP_PLATFORM_AMD__
__global__ void __launch_bounds__(512, 1) ibgda_progress_send_recv_kernel(
    P2pIbgdaTransportDevice* transport,
    char* src,
    char* dst,
    std::size_t totalBytes,
    int numBlocks,
    std::size_t maxSignalBytes,
    Timeout timeout) {
  auto group = make_block_group();
  auto [role, sub] = group.partition(2);
  const bool isSender = (role == 0);

  const std::size_t sectionBytes = section_bytes(transport, totalBytes);
  const std::size_t totalSections = totalBytes / sectionBytes;

  for (std::size_t s = 0; s < totalSections; ++s) {
    const std::size_t offset = s * sectionBytes;

    if (isSender) {
      TiledBuffer<char> tiles(src + offset, sectionBytes, sub);
      transport->init_send_progress(sub, tiles.bytes(), maxSignalBytes);
      while (transport->progress_send_once(
                 sub, tiles.data(), tiles.bytes(), maxSignalBytes, timeout) !=
             IbgdaSendRecvProgressStatus::Done) {
      }
    } else {
      TiledBuffer<char> tiles(dst + offset, sectionBytes, sub);
      transport->init_recv_progress(sub, tiles.bytes(), maxSignalBytes);
      while (transport->progress_recv_once(
                 sub, tiles.data(), tiles.bytes(), maxSignalBytes, timeout) !=
             IbgdaSendRecvProgressStatus::Done) {
      }
    }
  }
}
#endif

void launch_ibgda_progress_send_recv(
    P2pIbgdaTransportDevice* transport,
    char* src,
    char* dst,
    std::size_t nbytes,
    int numBlocks,
    cudaStream_t stream,
    std::size_t maxSignalBytes,
    Timeout timeout) {
#ifdef __HIP_PLATFORM_AMD__
  (void)transport;
  (void)src;
  (void)dst;
  (void)nbytes;
  (void)numBlocks;
  (void)stream;
  (void)maxSignalBytes;
  (void)timeout;
  printf("[PIPES] progress send/recv benchmark is NVIDIA-only\n");
#else
  ibgda_progress_send_recv_kernel<<<2 * numBlocks, 512, 0, stream>>>(
      transport, src, dst, nbytes, numBlocks, maxSignalBytes, timeout);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf(
        "[PIPES] progress sendrecv kernel launch failed: %s\n",
        cudaGetErrorString(err));
  }
#endif
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
        sub, first.data(), first.bytes(), firstMaxSignalBytes, timeout);
    TiledBuffer<char> second(src + firstBytes, secondBytes, sub);
    transport->send(
        sub, second.data(), second.bytes(), secondMaxSignalBytes, timeout);
  } else {
    TiledBuffer<char> first(dst, firstBytes, sub);
    transport->recv(
        sub, first.data(), first.bytes(), firstMaxSignalBytes, timeout);
    TiledBuffer<char> second(dst + firstBytes, secondBytes, sub);
    transport->recv(
        sub, second.data(), second.bytes(), secondMaxSignalBytes, timeout);
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

  const std::size_t sectionBytes = section_bytes(transport, totalBytes);
  const std::size_t totalSections = totalBytes / sectionBytes;

  for (std::size_t s = 0; s < totalSections; ++s) {
    TiledBuffer<char> tiles(src + s * sectionBytes, sectionBytes, group);
    transport->send(
        group, tiles.data(), tiles.bytes(), maxSignalBytes, timeout);
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

  const std::size_t sectionBytes = section_bytes(transport, totalBytes);
  const std::size_t totalSections = totalBytes / sectionBytes;

  for (std::size_t s = 0; s < totalSections; ++s) {
    TiledBuffer<char> tiles(dst + s * sectionBytes, sectionBytes, group);
    transport->recv(
        group, tiles.data(), tiles.bytes(), maxSignalBytes, timeout);
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
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("[PIPES] send kernel launch failed: %s\n", cudaGetErrorString(err));
  }
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
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("[PIPES] recv kernel launch failed: %s\n", cudaGetErrorString(err));
  }
}

__global__ void __launch_bounds__(512, 1) ibgda_drain_send_recv_kernel(
    P2pIbgdaTransportDevice* transport,
    int numBlocks,
    std::size_t totalBytes,
    int iterations,
    Timeout timeout) {
  auto group = make_block_group();
  if (group.group_id >= static_cast<uint32_t>(numBlocks)) {
    return;
  }

  const int groupId = static_cast<int>(group.group_id);
  const auto& layout = transport->channel_layout();
  const std::size_t expectedBytes =
      protocol_bytes_for_tiled_group_per_launch(
          totalBytes, numBlocks, layout.data_buffer_size(), groupId) *
      iterations;
  if (expectedBytes == 0) {
    return;
  }
  auto& channel = transport->local_channel(static_cast<uint32_t>(groupId));
  transport->wait_counter(group, channel.nicDoneWait, expectedBytes, timeout);
  transport->wait_signal(group, channel.slotFree, expectedBytes, timeout);
}

void launch_ibgda_drain_send_recv(
    P2pIbgdaTransportDevice* transport,
    int numBlocks,
    std::size_t totalBytes,
    int iterations,
    cudaStream_t stream,
    Timeout timeout) {
  if (totalBytes == 0 || iterations == 0) {
    return;
  }
  ibgda_drain_send_recv_kernel<<<numBlocks, 512, 0, stream>>>(
      transport, numBlocks, totalBytes, iterations, timeout);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("[PIPES] Drain launch failed: %s\n", cudaGetErrorString(err));
  }
}

__global__ void __launch_bounds__(256, 1) ibgda_reset_send_recv_kernel(
    P2pIbgdaTransportDevice* transport,
    int maxGroups) {
  const auto& layout = transport->channel_layout();
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto stride = blockDim.x * gridDim.x;

  // Zero the whole signal region: a numLanes-per-channel DATA_READY block plus
  // a one-per-channel SLOT_FREE block. Grid-stride so the loop covers it for
  // any launch grid size.
  const uint32_t dataReadySlots =
      static_cast<uint32_t>(layout.numLanes * maxGroups);
  const uint32_t slotFreeSlots = static_cast<uint32_t>(maxGroups);
  const uint32_t totalSignalSlots = dataReadySlots + slotFreeSlots;
  for (auto slot = idx; slot < totalSignalSlots; slot += stride) {
    if (SignalState* signal = layout.localSignalState(static_cast<int>(slot))) {
      signal->signal_ = 0;
    }
    if (slot < static_cast<uint32_t>(maxGroups)) {
      auto& channel = transport->local_channel(slot);
      channel.sendProgress = IbChannelProgress{};
      channel.recvProgress = IbChannelProgress{};
      // Zero the per-lane receiver DATA_READY expectations so they stay aligned
      // with the DATA_READY slots zeroed above. recvDataReadyLaneCursor is
      // deliberately NOT reset here: it mirrors the sender's free-running
      // IbQpState::cursor, which this kernel also leaves untouched, so zeroing
      // it would desync the round-robin lane mapping on the next stream.
      for (int lane = 0; lane < kIbMaxQpLanesPerChannelDirection; ++lane) {
        channel.recvLaneExpected[lane] = 0;
      }
    }
  }

  for (auto slot = idx; slot < static_cast<uint32_t>(maxGroups);
       slot += stride) {
    if (SignalState* counter =
            layout.localCounterState(static_cast<int>(slot))) {
      counter->signal_ = 0;
    }
  }
}

void launch_ibgda_reset_send_recv(
    P2pIbgdaTransportDevice* transport,
    int maxGroups,
    cudaStream_t stream) {
  if (maxGroups == 0) {
    return;
  }
  constexpr int kThreads = 256;
  const int blocks = (2 * maxGroups + kThreads - 1) / kThreads;
  ibgda_reset_send_recv_kernel<<<blocks, kThreads, 0, stream>>>(
      transport, maxGroups);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("[PIPES] Reset launch failed: %s\n", cudaGetErrorString(err));
  }
}

#ifndef __HIP_PLATFORM_AMD__
__global__ void __launch_bounds__(512, 1) ibgda_progress_send_kernel(
    P2pIbgdaTransportDevice* transport,
    char* src,
    std::size_t totalBytes,
    int numBlocks,
    std::size_t maxSignalBytes,
    Timeout timeout) {
  auto group = make_block_group();

  const std::size_t sectionBytes = section_bytes(transport, totalBytes);
  const std::size_t totalSections = totalBytes / sectionBytes;

  for (std::size_t s = 0; s < totalSections; ++s) {
    TiledBuffer<char> tiles(src + s * sectionBytes, sectionBytes, group);
    transport->init_send_progress(group, tiles.bytes(), maxSignalBytes);
    while (transport->progress_send_once(
               group, tiles.data(), tiles.bytes(), maxSignalBytes, timeout) !=
           IbgdaSendRecvProgressStatus::Done) {
    }
  }
}

__global__ void __launch_bounds__(512, 1) ibgda_progress_recv_kernel(
    P2pIbgdaTransportDevice* transport,
    char* dst,
    std::size_t totalBytes,
    int numBlocks,
    std::size_t maxSignalBytes,
    Timeout timeout) {
  auto group = make_block_group();

  const std::size_t sectionBytes = section_bytes(transport, totalBytes);
  const std::size_t totalSections = totalBytes / sectionBytes;

  for (std::size_t s = 0; s < totalSections; ++s) {
    TiledBuffer<char> tiles(dst + s * sectionBytes, sectionBytes, group);
    transport->init_recv_progress(group, tiles.bytes(), maxSignalBytes);
    while (transport->progress_recv_once(
               group, tiles.data(), tiles.bytes(), maxSignalBytes, timeout) !=
           IbgdaSendRecvProgressStatus::Done) {
    }
  }
}
#endif

void launch_ibgda_progress_send(
    P2pIbgdaTransportDevice* transport,
    char* src,
    std::size_t nbytes,
    int numBlocks,
    cudaStream_t stream,
    std::size_t maxSignalBytes,
    Timeout timeout) {
#ifdef __HIP_PLATFORM_AMD__
  (void)transport;
  (void)src;
  (void)nbytes;
  (void)numBlocks;
  (void)stream;
  (void)maxSignalBytes;
  (void)timeout;
  printf("[PIPES] progress send benchmark is NVIDIA-only\n");
#else
  ibgda_progress_send_kernel<<<numBlocks, 512, 0, stream>>>(
      transport, src, nbytes, numBlocks, maxSignalBytes, timeout);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf(
        "[PIPES] progress send kernel launch failed: %s\n",
        cudaGetErrorString(err));
  }
#endif
}

void launch_ibgda_progress_recv(
    P2pIbgdaTransportDevice* transport,
    char* dst,
    std::size_t nbytes,
    int numBlocks,
    cudaStream_t stream,
    std::size_t maxSignalBytes,
    Timeout timeout) {
#ifdef __HIP_PLATFORM_AMD__
  (void)transport;
  (void)dst;
  (void)nbytes;
  (void)numBlocks;
  (void)stream;
  (void)maxSignalBytes;
  (void)timeout;
  printf("[PIPES] progress recv benchmark is NVIDIA-only\n");
#else
  ibgda_progress_recv_kernel<<<numBlocks, 512, 0, stream>>>(
      transport, dst, nbytes, numBlocks, maxSignalBytes, timeout);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf(
        "[PIPES] progress recv kernel launch failed: %s\n",
        cudaGetErrorString(err));
  }
#endif
}

__global__ void ibgda_snapshot_step_state_kernel(
    P2pIbgdaTransportDevice* transport,
    int64_t* dst,
    int count) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    const auto& layout = transport->channel_layout();
    const auto maxChannels = static_cast<uint32_t>(layout.maxChannels);
    if (idx < maxChannels) {
      dst[idx] = transport->local_channel(idx).sendProgress.nextStep;
    } else {
      dst[idx] =
          transport->local_channel(idx - maxChannels).recvProgress.nextStep;
    }
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
