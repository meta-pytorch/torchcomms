// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/pipes/tests/BroadcastTest.cuh"

#include "comms/pipes/collectives/Broadcast.cuh"
#include "comms/pipes/collectives/BroadcastBinomialTree.cuh"
#include "comms/pipes/collectives/BroadcastRing.cuh"
#include "comms/pipes/tests/Checks.h"

namespace comms::pipes::collectives::test {

/**
 * Device kernel that invokes the flat-tree broadcast collective.
 * Minimal wrapper - all logic is in Broadcast.cuh.
 */
__global__ void testBroadcastKernel(
    void* buff_d,
    int my_rank_id,
    int root_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::size_t nbytes) {
  broadcast(buff_d, my_rank_id, root_rank_id, transports_per_rank, nbytes);
}

/**
 * Device kernel that invokes the binomial tree broadcast collective.
 * Minimal wrapper - all logic is in BroadcastBinomialTree.cuh.
 */
__global__ void testBroadcastBinomialTreeKernel(
    void* buff_d,
    int my_rank_id,
    int root_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::size_t nbytes) {
  broadcast_binomial_tree(
      buff_d, my_rank_id, root_rank_id, transports_per_rank, nbytes);
}

/**
 * Device kernel that invokes the ring broadcast collective.
 * Minimal wrapper - all logic is in BroadcastRing.cuh.
 */
__global__ void testBroadcastRingKernel(
    void* buff_d,
    int my_rank_id,
    int root_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::size_t nbytes) {
  broadcast_ring(buff_d, my_rank_id, root_rank_id, transports_per_rank, nbytes);
}

void testBroadcast(
    void* buff_d,
    int my_rank_id,
    int root_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::size_t nbytes,
    int numBlocks,
    int blockSize) {
  testBroadcastKernel<<<numBlocks, blockSize>>>(
      buff_d, my_rank_id, root_rank_id, transports_per_rank, nbytes);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testBroadcastBinomialTree(
    void* buff_d,
    int my_rank_id,
    int root_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::size_t nbytes,
    int numBlocks,
    int blockSize) {
  testBroadcastBinomialTreeKernel<<<numBlocks, blockSize>>>(
      buff_d, my_rank_id, root_rank_id, transports_per_rank, nbytes);
  PIPES_KERNEL_LAUNCH_CHECK();
}

void testBroadcastRing(
    void* buff_d,
    int my_rank_id,
    int root_rank_id,
    DeviceSpan<Transport> transports_per_rank,
    std::size_t nbytes,
    int numBlocks,
    int blockSize) {
  testBroadcastRingKernel<<<numBlocks, blockSize>>>(
      buff_d, my_rank_id, root_rank_id, transports_per_rank, nbytes);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::pipes::collectives::test
