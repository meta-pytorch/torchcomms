// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "comms/pipes/collectives/AllGather.cuh"

namespace comms::pipes::test {

/**
 * Test all_gather with transports.
 * Launches a kernel that performs the AllGather collective operation.
 *
 * @param recvbuff_d Device pointer to receive buffer (nranks * sendcount bytes)
 * @param sendbuff_d Device pointer to send buffer (sendcount bytes)
 * @param sendcount Number of bytes each rank contributes
 * @param my_rank_id Current rank ID
 * @param nranks Total number of ranks
 * @param transports Array of transport objects
 * @param numBlocks Number of thread blocks to launch
 * @param blockSize Number of threads per block
 */
void testAllGather(
    void* recvbuff_d,
    const void* sendbuff_d,
    std::size_t sendcount,
    int my_rank_id,
    int nranks,
    DeviceSpan<Transport> transports,
    int numBlocks,
    int blockSize);

} // namespace comms::pipes::test
