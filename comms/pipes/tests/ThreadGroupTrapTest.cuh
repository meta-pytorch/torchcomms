// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace comms::pipes::test {

// Tests that partition(weights) asserts when num_partitions > total_groups
// This is invalid usage since some partitions would be empty
void testWeightedPartitionMorePartitionsThanGroups(
    const uint32_t* weights_d,
    uint32_t numPartitions,
    int numBlocks,
    int blockSize);

// Tests that partition(num_partitions) traps when num_partitions > total_groups
// This is invalid usage since some partitions would be empty
void testPartitionMorePartitionsThanGroups(
    uint32_t numPartitions,
    int numBlocks,
    int blockSize);

} // namespace comms::pipes::test
