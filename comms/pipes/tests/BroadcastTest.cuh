// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstddef>
#include <cstdint>

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
 */
void testBroadcast(
    void* buff_d,
    int my_rank_id,
    int root_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::size_t nbytes,
    int numBlocks,
    int blockSize);

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
 */
void testBroadcastBinomialTree(
    void* buff_d,
    int my_rank_id,
    int root_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::size_t nbytes,
    int numBlocks,
    int blockSize);

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
 */
void testBroadcastRing(
    void* buff_d,
    int my_rank_id,
    int root_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::size_t nbytes,
    int numBlocks,
    int blockSize);

} // namespace comms::pipes::collectives::test
