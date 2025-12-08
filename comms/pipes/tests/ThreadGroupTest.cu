// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

#include "comms/pipes/ThreadGroup.cuh"
#include "comms/pipes/tests/Utils.h"

namespace comms::pipes::test {

using namespace comms::pipes;

__global__ void testContiguousLocalityKernel(
    uint32_t* groupIds,
    uint32_t numItems,
    uint32_t* errorCount) {
  auto warp = make_warp_group();

  // Each warp writes its group_id to its assigned work items
  warp.for_each_item_contiguous(numItems, [&](uint32_t item_id) {
    if (item_id >= numItems) {
      atomicAdd(errorCount, 1);
      return;
    }

    groupIds[item_id] = warp.group_id;
  });

  __syncthreads();

  // Leader of first warp verifies the contiguous pattern on GPU
  // (CPU will also verify this for thoroughness)
  if (warp.is_global_leader()) {
    uint32_t items_per_group =
        (numItems + warp.total_groups - 1) / warp.total_groups;

    for (uint32_t group_id = 0; group_id < warp.total_groups; group_id++) {
      uint32_t start_item = group_id * items_per_group;
      uint32_t end_item = (start_item + items_per_group < numItems)
          ? start_item + items_per_group
          : numItems;

      // Verify all items in [start, end) belong to this group
      for (uint32_t item_id = start_item; item_id < end_item; item_id++) {
        if (groupIds[item_id] != group_id) {
          atomicAdd(errorCount, 1);
        }
      }
    }
  }
}

void testContiguousLocality(
    uint32_t* groupIds_d,
    uint32_t numItems,
    uint32_t* errorCount_d,
    int numBlocks,
    int blockSize) {
  testContiguousLocalityKernel<<<numBlocks, blockSize>>>(
      groupIds_d, numItems, errorCount_d);
  PIPES_KERNEL_LAUNCH_CHECK();
}

} // namespace comms::pipes::test
