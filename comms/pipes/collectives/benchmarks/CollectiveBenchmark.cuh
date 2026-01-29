// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda_runtime.h>

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
    DeviceSpan<ChunkInfo> recv_chunk_infos);

} // namespace comms::pipes::benchmark
