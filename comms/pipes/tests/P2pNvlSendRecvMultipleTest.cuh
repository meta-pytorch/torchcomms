// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstddef>

#include "comms/pipes/P2pNvlTransportDevice.cuh"

namespace comms::pipes::test {

using comms::pipes::P2pNvlTransportDevice;

// Test send_multiple: sends selected chunks from input buffer
void testSendMultiple(
    P2pNvlTransportDevice* p2p,
    const void* src_d,
    const size_t* chunk_sizes_d,
    size_t chunk_sizes_count,
    const size_t* chunk_indices_d,
    size_t chunk_indices_count,
    int numBlocks,
    int blockSize);

// Test recv_multiple: receives chunks into output buffer
void testRecvMultiple(
    P2pNvlTransportDevice* p2p,
    void* dst_d,
    size_t* chunk_sizes_d,
    size_t chunk_sizes_count,
    int numBlocks,
    int blockSize);

} // namespace comms::pipes::test
