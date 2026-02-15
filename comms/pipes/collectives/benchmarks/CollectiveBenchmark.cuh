// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>

#include "comms/pipes/Timeout.cuh"
#include "comms/pipes/collectives/AllGather.cuh"
#include "comms/pipes/collectives/AllToAllv.cuh"

namespace comms::pipes::benchmark {

/**
 * AllToAllv benchmark kernel.
 * All ranks participate in all-to-all communication with variable chunk sizes.
 */
__global__ void all_to_allv_kernel(
    void* recvbuff_d,
    const void* sendbuff_d,
    int my_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    DeviceSpan<ChunkInfo> send_chunk_infos,
    DeviceSpan<ChunkInfo> recv_chunk_infos,
    Timeout timeout);

/**
 * AllGather benchmark kernel.
 * All ranks participate in all-gather communication.
 * Each rank contributes sendcount bytes, and after the operation,
 * each rank has nranks * sendcount bytes.
 */
__global__ void all_gather_kernel(
    void* recvbuff_d,
    const void* sendbuff_d,
    std::size_t sendcount,
    int my_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    Timeout timeout);

} // namespace comms::pipes::benchmark
