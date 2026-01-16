// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstddef>

#include "comms/pipes/P2pNvlTransportDevice.cuh"

namespace comms::pipes::test {

using comms::pipes::P2pNvlTransportDevice;

// Test send_one: sends a single chunk with metadata
void testSendOne(
    P2pNvlTransportDevice p2p,
    const void* src_d,
    size_t nbytes,
    size_t offset_in_output,
    size_t total_chunks,
    int numBlocks,
    int blockSize);

// Test recv_one: receives a single chunk with metadata
void testRecvOne(
    P2pNvlTransportDevice p2p,
    void* dst_base_d,
    size_t* nbytes_d,
    size_t* offset_d,
    size_t* total_chunks_d,
    int numBlocks,
    int blockSize);

// Test send_one called multiple times in one kernel
void testSendOneMultipleTimes(
    P2pNvlTransportDevice p2p,
    const void* const* src_d_array,
    const size_t* nbytes_array,
    const size_t* offset_array,
    const size_t* total_chunks_array,
    size_t num_calls,
    int numBlocks,
    int blockSize);

// Test recv_one called multiple times in one kernel
void testRecvOneMultipleTimes(
    P2pNvlTransportDevice p2p,
    void* dst_base_d,
    size_t* nbytes_array_d,
    size_t* offset_array_d,
    size_t* total_chunks_array_d,
    size_t num_calls,
    int numBlocks,
    int blockSize);

} // namespace comms::pipes::test
