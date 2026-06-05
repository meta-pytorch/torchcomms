// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#ifndef __HIP_PLATFORM_AMD__
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include "comms/prims/collectives/AllToAllv.cuh"

namespace comms::prims::test {

// Test all_to_allv with transports
void testAllToAllv(
    void* recvbuff_d,
    const void* sendbuff_d,
    int my_rank_id,
    int nranks,
    DeviceSpan<Transport> transports,
    DeviceSpan<ChunkInfo> send_chunk_infos,
    DeviceSpan<ChunkInfo> recv_chunk_infos,
    int numBlocks,
    int blockSize);

} // namespace comms::prims::test
