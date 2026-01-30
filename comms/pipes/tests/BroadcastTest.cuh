// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include <cuda_runtime.h>

#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/Transport.cuh"

namespace comms::pipes::collectives::test {

/**
 * Host-callable wrapper to launch the broadcast test kernel (flat-tree).
 *
 * @param buff_d Device buffer (source for root, destination for non-root)
 * @param my_rank_id Current rank ID
 * @param root_rank_id Rank that broadcasts data
 * @param transports_per_rank Array of transport objects
 * @param nbytes Number of bytes to broadcast
 * @param numBlocks Number of CUDA blocks to launch
 * @param blockSize Number of threads per block
 * @param clusterDim Optional cluster dimension for spread cluster launch.
 *                   When provided, uses cudaLaunchKernelExC with
 *                   cudaClusterSchedulingPolicySpread for better SM
 * distribution.
 * @param stream CUDA stream for kernel execution (default: 0)
 */
void testBroadcastFlat(
    void* buff_d,
    int my_rank_id,
    int root_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::size_t nbytes,
    int numBlocks,
    int blockSize,
    std::optional<dim3> clusterDim = std::nullopt,
    cudaStream_t stream = 0);

} // namespace comms::pipes::collectives::test
