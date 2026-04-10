// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Include .cuh (not .h) so the __global__ kernel below only sees the __device__
// overload of all_to_allv.  Including .h would bring in the host overload
// whose first 7 parameter types are identical, causing an ambiguous-overload
// error in NVCC (it resolves C++ overloads before __host__/__device__
// filtering).
#include "comms/pipes/collectives/AllToAllv.cuh"

#include <chrono>
#include <optional>

#include "comms/common/CudaWrap.h"
#include "comms/pipes/Checks.h"
#include "comms/pipes/TimeoutUtils.h"

namespace comms::pipes {

/**
 * Unified AllToAllv kernel.
 * Handles both NVL-only and IBGDA transports — the device-side all_to_allv()
 * dispatches per-peer based on transport type. IBGDA staging state is now
 * embedded in each P2pIbgdaTransportDevice (via send()/recv() APIs).
 */
__global__ void allToAllvKernel(
    void* recvbuff_d,
    const void* sendbuff_d,
    int my_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    DeviceSpan<ChunkInfo> send_chunk_infos,
    DeviceSpan<ChunkInfo> recv_chunk_infos,
    Timeout timeout) {
  timeout.start();
  all_to_allv(
      recvbuff_d,
      sendbuff_d,
      my_rank_id,
      transports_per_rank,
      send_chunk_infos,
      recv_chunk_infos,
      timeout);
}

// --- Unified host wrappers ---

void all_to_allv(
    void* recvbuff_d,
    const void* sendbuff_d,
    int my_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    DeviceSpan<ChunkInfo> send_chunk_infos,
    DeviceSpan<ChunkInfo> recv_chunk_infos,
    Timeout timeout_config,
    cudaStream_t stream,
    int num_blocks,
    int num_threads,
    std::optional<dim3> cluster_dim) {
  void* args[] = {
      &recvbuff_d,
      &sendbuff_d,
      &my_rank_id,
      &transports_per_rank,
      &send_chunk_infos,
      &recv_chunk_infos,
      &timeout_config};

  comms::common::launchKernel(
      (void*)allToAllvKernel,
      dim3(num_blocks),
      dim3(num_threads),
      args,
      stream,
      cluster_dim);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void all_to_allv(
    void* recvbuff_d,
    const void* sendbuff_d,
    int my_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    DeviceSpan<ChunkInfo> send_chunk_infos,
    DeviceSpan<ChunkInfo> recv_chunk_infos,
    std::chrono::milliseconds timeout,
    cudaStream_t stream,
    int num_blocks,
    int num_threads,
    std::optional<dim3> cluster_dim) {
  int device = 0;
  PIPES_CUDA_CHECK(cudaGetDevice(&device));
  Timeout timeout_config =
      makeTimeout(static_cast<uint32_t>(timeout.count()), device);
  all_to_allv(
      recvbuff_d,
      sendbuff_d,
      my_rank_id,
      transports_per_rank,
      send_chunk_infos,
      recv_chunk_infos,
      timeout_config,
      stream,
      num_blocks,
      num_threads,
      cluster_dim);
}

WarpReserveDeviceConfig resolveWarpReserve(
    const WarpReserveConfig& config,
    int numNvlPeers,
    int numIbgdaPeers,
    const int* d_nvlPeerRanks,
    const int* d_ibgdaPeerRanks,
    int numThreadsPerBlock,
    int maxChannelsPerPeer) {
  if (numNvlPeers == 0 && numIbgdaPeers == 0) {
    return {};
  }

  bool anyExplicit = config.nvlSendWarps > 0 || config.nvlRecvWarps > 0 ||
      config.ibgdaSendWarps > 0 || config.ibgdaRecvWarps > 0 ||
      config.selfWarps > 0;
  if (!anyExplicit) {
    return {};
  }

  int selfW = config.selfWarps > 0 ? config.selfWarps : 1;
  // Clamp transport-specific warps to 0 when no peers of that type exist,
  // AND round down to nearest multiple of peer count to ensure symmetric
  // channel assignment. Without rounding, partition_interleaved gives
  // uneven per-peer warp counts, creating send/recv channel mismatches
  // that deadlock (e.g., ibgdaSendWarps=8 with 7 peers → peer 0 gets
  // 2 warps/channels, but the remote rank only sends 1 → recv hangs).
  int nvlSendW = numNvlPeers > 0
      ? (config.nvlSendWarps > 0
             ? (config.nvlSendWarps / numNvlPeers) * numNvlPeers
             : 2 * numNvlPeers)
      : 0;
  int nvlRecvW = numNvlPeers > 0
      ? (config.nvlRecvWarps > 0
             ? (config.nvlRecvWarps / numNvlPeers) * numNvlPeers
             : 2 * numNvlPeers)
      : 0;
  int ibgdaSendW = numIbgdaPeers > 0
      ? (config.ibgdaSendWarps > 0
             ? (config.ibgdaSendWarps / numIbgdaPeers) * numIbgdaPeers
             : 1 * numIbgdaPeers)
      : 0;
  int ibgdaRecvW = numIbgdaPeers > 0
      ? (config.ibgdaRecvWarps > 0
             ? (config.ibgdaRecvWarps / numIbgdaPeers) * numIbgdaPeers
             : 1 * numIbgdaPeers)
      : 0;

  // Validate IBGDA warp counts when explicitly set:
  //   - send == recv: protocol symmetry (sender channel K must have matching
  //     receiver channel K)
  //   - divisible by numIbgdaPeers: even distribution across peers
  if (numIbgdaPeers > 0 && config.ibgdaSendWarps > 0 &&
      config.ibgdaRecvWarps > 0) {
    if (ibgdaSendW != ibgdaRecvW) {
      throw std::runtime_error(
          "ibgdaSendWarps (" + std::to_string(ibgdaSendW) +
          ") must equal ibgdaRecvWarps (" + std::to_string(ibgdaRecvW) +
          ") for IBGDA protocol symmetry");
    }
    if (ibgdaSendW % numIbgdaPeers != 0) {
      throw std::runtime_error(
          "ibgdaSendWarps (" + std::to_string(ibgdaSendW) +
          ") must be divisible by numIbgdaPeers (" +
          std::to_string(numIbgdaPeers) + ") for even warp distribution");
    }
  }

  // Validate IBGDA warps per peer fits within a single block.
  // Named barrier synchronization (SyncScope::MULTIWARP) is block-local —
  // warps for the same IBGDA peer/channel cannot span multiple blocks.
  if (numIbgdaPeers > 0 && ibgdaSendW > 0) {
    int warpsPerBlock = numThreadsPerBlock / 32;
    int ibgdaWarpsPerPeer = ibgdaSendW / numIbgdaPeers;
    if (ibgdaWarpsPerPeer > warpsPerBlock) {
      throw std::runtime_error(
          "IBGDA warps per peer (" + std::to_string(ibgdaWarpsPerPeer) +
          ") exceeds warps per block (" + std::to_string(warpsPerBlock) +
          "). Cannot guarantee same-block affinity for cooperative memcpy. "
          "Reduce ibgdaSendWarps or increase threads per block.");
    }
  }

  WarpReserveDeviceConfig dc;
  dc.selfEnd = static_cast<uint32_t>(selfW);
  dc.nvlSendEnd = dc.selfEnd + static_cast<uint32_t>(nvlSendW);
  dc.nvlRecvEnd = dc.nvlSendEnd + static_cast<uint32_t>(nvlRecvW);

  // Block-align IBGDA category starts so that all warps for a given IBGDA
  // peer are guaranteed to be in the same block. Named barrier sync
  // (SyncScope::MULTIWARP) is block-local — cross-block spanning is fatal.
  // Padding warps between nvlRecvEnd and ibgdaSendBase silently early-return.
  int warpsPerBlock = numThreadsPerBlock / 32;
  uint32_t ibgdaSendBase = (ibgdaSendW > 0)
      ? ((dc.nvlRecvEnd + warpsPerBlock - 1) / warpsPerBlock) * warpsPerBlock
      : dc.nvlRecvEnd;
  dc.ibgdaSendBase = ibgdaSendBase;
  dc.ibgdaSendEnd = ibgdaSendBase + static_cast<uint32_t>(ibgdaSendW);

  uint32_t ibgdaRecvBase = (ibgdaRecvW > 0)
      ? ((dc.ibgdaSendEnd + warpsPerBlock - 1) / warpsPerBlock) * warpsPerBlock
      : dc.ibgdaSendEnd;
  dc.ibgdaRecvBase = ibgdaRecvBase;
  dc.ibgdaRecvEnd = ibgdaRecvBase + static_cast<uint32_t>(ibgdaRecvW);

  dc.nvlPeerRanks = d_nvlPeerRanks;
  dc.numNvlPeers = static_cast<uint32_t>(numNvlPeers);
  dc.ibgdaPeerRanks = d_ibgdaPeerRanks;
  dc.numIbgdaPeers = static_cast<uint32_t>(numIbgdaPeers);
  dc.maxChannelsPerPeer = static_cast<uint32_t>(maxChannelsPerPeer);

  return dc;
}

} // namespace comms::pipes
