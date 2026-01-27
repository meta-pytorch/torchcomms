// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstddef>

namespace comms::pipes::test {

// Test kernel that calls SelfTransportDevice::write()
void testSelfPut(
    char* dst_d,
    const char* src_d,
    size_t nbytes,
    int numBlocks,
    int blockSize);

} // namespace comms::pipes::test
