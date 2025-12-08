// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace comms::pipes::test {

void testCopyChunkVectorized(
    char* dst_d,
    const char* src_d,
    std::size_t chunk_bytes,
    uint32_t* errorCount_d,
    int numBlocks,
    int blockSize);

} // namespace comms::pipes::test
