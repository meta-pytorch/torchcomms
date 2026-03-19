// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "comms/pipes/collectives/AllToAllvLl128.cuh"

namespace comms::pipes::test {

// Test all_to_allv_ll128 with transports
void test_all_to_allv_ll128(
    void* recvbuff_d,
    const void* sendbuff_d,
    int my_rank_id,
    int nranks,
    DeviceSpan<Transport> transports,
    DeviceSpan<ChunkInfo> send_chunk_infos,
    DeviceSpan<ChunkInfo> recv_chunk_infos,
    int numBlocks,
    int blockSize,
    Timeout timeout = Timeout());

} // namespace comms::pipes::test
