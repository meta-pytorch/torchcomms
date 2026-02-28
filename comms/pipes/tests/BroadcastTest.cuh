// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include <cuda_runtime.h>

#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/Transport.cuh"
#include "comms/pipes/collectives/broadcast/BroadcastContext.cuh"

namespace comms::pipes::collectives::test {

// =============================================================================
// Template API Test Wrappers
// =============================================================================

/**
 * Host-callable wrapper to launch broadcast with explicit TopologyTag.
 *
 * This is the preferred test interface that directly exercises the template
 * API: broadcast<TopologyTag>(...). Use this to verify the template dispatch
 * mechanism works correctly.
 *
 * Usage:
 *   - testBroadcast<FlatTag>(buff, rank, root, transports, nbytes, blocks,
 * threads);
 *   - testBroadcast<RingTag>(buff, rank, root, transports, nbytes, blocks,
 * threads);
 *   - testBroadcast<BinomialTreeTag>(buff, rank, root, transports, nbytes,
 * blocks, threads);
 *
 * @tparam TopologyTag One of: FlatTag, RingTag, BinomialTreeTag
 */
template <typename TopologyTag>
void testBroadcast(
    void* buff_d,
    int my_rank_id,
    int root_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::size_t nbytes,
    int numBlocks,
    int blockSize,
    std::optional<dim3> clusterDim = std::nullopt,
    cudaStream_t stream = 0);

// Explicit instantiation declarations for the three topology tags
extern template void testBroadcast<FlatTag>(
    void* buff_d,
    int my_rank_id,
    int root_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::size_t nbytes,
    int numBlocks,
    int blockSize,
    std::optional<dim3> clusterDim,
    cudaStream_t stream);

extern template void testBroadcast<RingTag>(
    void* buff_d,
    int my_rank_id,
    int root_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::size_t nbytes,
    int numBlocks,
    int blockSize,
    std::optional<dim3> clusterDim,
    cudaStream_t stream);

extern template void testBroadcast<BinomialTreeTag>(
    void* buff_d,
    int my_rank_id,
    int root_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::size_t nbytes,
    int numBlocks,
    int blockSize,
    std::optional<dim3> clusterDim,
    cudaStream_t stream);

// =============================================================================
// Legacy Test Wrappers (for backward compatibility with existing tests)
// =============================================================================

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
    cudaStream_t stream = nullptr);

/**
 * Host-callable wrapper to launch the binomial tree broadcast test kernel.
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
void testBroadcastBinomialTree(
    void* buff_d,
    int my_rank_id,
    int root_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::size_t nbytes,
    int numBlocks,
    int blockSize,
    std::optional<dim3> clusterDim = std::nullopt,
    cudaStream_t stream = nullptr);

/**
 * Host-callable wrapper to launch the ring broadcast test kernel.
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
void testBroadcastRing(
    void* buff_d,
    int my_rank_id,
    int root_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::size_t nbytes,
    int numBlocks,
    int blockSize,
    std::optional<dim3> clusterDim = std::nullopt,
    cudaStream_t stream = nullptr);

/**
 * Host-callable wrapper to launch the adaptive broadcast test kernel.
 *
 * The adaptive algorithm automatically selects between flat-tree and ring
 * based on message size:
 * - For messages < 4MB: uses flat-tree (lower latency)
 * - For messages >= 4MB with nranks > 2: uses ring (better bandwidth)
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
void testBroadcastAdaptive(
    void* buff_d,
    int my_rank_id,
    int root_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::size_t nbytes,
    int numBlocks,
    int blockSize,
    std::optional<dim3> clusterDim = std::nullopt,
    cudaStream_t stream = nullptr);

} // namespace comms::pipes::collectives::test
