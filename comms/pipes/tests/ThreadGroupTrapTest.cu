// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>
#include <cstdint>

#include "comms/pipes/ThreadGroup.cuh"

namespace comms::pipes::test {

using namespace comms::pipes;

// =============================================================================
// Invalid Partition Test (more partitions than groups)
// =============================================================================

__global__ void testWeightedPartitionMorePartitionsThanGroupsKernel(
    const uint32_t* weights,
    uint32_t numPartitions) {
  auto warp = make_warp_group();

  // This should trigger an assertion failure because numPartitions >
  // total_groups The assertion in partition(weights) checks: num_partitions <=
  // total_groups
  auto [partition_id, subgroup] =
      warp.partition(make_device_span(weights, numPartitions));

  // Suppress unused variable warnings (we won't reach here due to assertion)
  (void)partition_id;
  (void)subgroup;
}

void testWeightedPartitionMorePartitionsThanGroups(
    const uint32_t* weights_d,
    uint32_t numPartitions,
    int numBlocks,
    int blockSize) {
  testWeightedPartitionMorePartitionsThanGroupsKernel<<<numBlocks, blockSize>>>( // NOLINT(facebook-cuda-safe-kernel-call-check)
      weights_d, numPartitions);
  // No kernel launch check here - this test expects the kernel to trap
}

// =============================================================================
// Invalid Partition Test (more partitions than groups) - Non-weighted version
// =============================================================================

__global__ void testPartitionMorePartitionsThanGroupsKernel(
    uint32_t numPartitions) {
  auto warp = make_warp_group();

  // This should trigger a trap because numPartitions > total_groups
  // The check in partition(num_partitions) validates: num_partitions <=
  // total_groups
  auto [partition_id, subgroup] = warp.partition(numPartitions);

  // Suppress unused variable warnings (we won't reach here due to trap)
  (void)partition_id;
  (void)subgroup;
}

void testPartitionMorePartitionsThanGroups(
    uint32_t numPartitions,
    int numBlocks,
    int blockSize) {
  testPartitionMorePartitionsThanGroupsKernel<<< // NOLINT(facebook-cuda-safe-kernel-call-check)
      numBlocks,
      blockSize>>>(
      numPartitions);
  // No kernel launch check here - this test expects the kernel to trap
}

} // namespace comms::pipes::test
