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
    Timeout timeout,
    WarpReserveDeviceConfig reserve_config) {
  timeout.start();
  all_to_allv(
      recvbuff_d,
      sendbuff_d,
      my_rank_id,
      transports_per_rank,
      send_chunk_infos,
      recv_chunk_infos,
      timeout,
      reserve_config);
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
    std::optional<dim3> cluster_dim,
    WarpReserveDeviceConfig reserve_config) {
  void* args[] = {
      &recvbuff_d,
      &sendbuff_d,
      &my_rank_id,
      &transports_per_rank,
      &send_chunk_infos,
      &recv_chunk_infos,
      &timeout_config,
      &reserve_config};

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
    std::optional<dim3> cluster_dim,
    WarpReserveDeviceConfig reserve_config) {
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
      cluster_dim,
      reserve_config);
}

WarpReserveDeviceConfig resolveWarpReserve(
    const WarpReserveConfig& config,
    int numNvlPeers,
    int numIbgdaPeers,
    const int* d_nvlPeerRanks,
    const int* d_ibgdaPeerRanks) {
  if (numNvlPeers == 0 && numIbgdaPeers == 0) {
    return {};
  }

  // Only activate warp reserve when at least one field is explicitly set.
  // When all fields are 0 (auto), return unconfigured so the kernel uses
  // the uniform partition_interleaved path which dispatches per-transport-type.
  // This is critical for TEST_IBGDA_SINGLE_NODE=1 where nvlPeerRanks reflects
  // NVLink topology but the actual transport is forced to IBGDA.
  bool anyExplicit = config.nvlSendWarps > 0 || config.nvlRecvWarps > 0 ||
      config.ibgdaSendWarps > 0 || config.ibgdaRecvWarps > 0 ||
      config.selfWarps > 0;
  if (!anyExplicit) {
    return {};
  }

  int selfW = config.selfWarps > 0 ? config.selfWarps : 1;
  int nvlSendW =
      config.nvlSendWarps > 0 ? config.nvlSendWarps : 2 * numNvlPeers;
  int nvlRecvW =
      config.nvlRecvWarps > 0 ? config.nvlRecvWarps : 2 * numNvlPeers;
  int ibgdaSendW =
      config.ibgdaSendWarps > 0 ? config.ibgdaSendWarps : 1 * numIbgdaPeers;
  int ibgdaRecvW =
      config.ibgdaRecvWarps > 0 ? config.ibgdaRecvWarps : 1 * numIbgdaPeers;

  WarpReserveDeviceConfig dc;
  dc.selfEnd = static_cast<uint32_t>(selfW);
  dc.nvlSendEnd = dc.selfEnd + static_cast<uint32_t>(nvlSendW);
  dc.nvlRecvEnd = dc.nvlSendEnd + static_cast<uint32_t>(nvlRecvW);
  dc.ibgdaSendEnd = dc.nvlRecvEnd + static_cast<uint32_t>(ibgdaSendW);

  dc.nvlPeerRanks = d_nvlPeerRanks;
  dc.numNvlPeers = static_cast<uint32_t>(numNvlPeers);
  dc.ibgdaPeerRanks = d_ibgdaPeerRanks;
  dc.numIbgdaPeers = static_cast<uint32_t>(numIbgdaPeers);

  return dc;
}

} // namespace comms::pipes
