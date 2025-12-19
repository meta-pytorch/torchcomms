// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "comms/pipes/AllToAllv.cuh"

namespace comms::pipes::test {

// Test allToAllv with transports
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

} // namespace comms::pipes::test
