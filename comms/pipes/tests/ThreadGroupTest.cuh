// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace comms::pipes::test {

// Kernel: testContiguousLocalityKernel
// Tests that for_each_item_contiguous assigns CONTIGUOUS blocks of work items
// to each warp. Each warp writes its group_id to all work items it processes.
// The CPU then verifies that work items [start, end) all have the same
// group_id, confirming contiguous-based assignment.
void testContiguousLocality(
    uint32_t* groupIds_d,
    uint32_t numItems,
    uint32_t* errorCount_d,
    int numBlocks,
    int blockSize);

} // namespace comms::pipes::test
