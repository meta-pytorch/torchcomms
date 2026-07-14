// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstddef>

#include "comms/prims/transport/MultiPeerDeviceHandle.cuh"

namespace comms::prims::test {

/**
 * Verify that the device handle type map is correctly populated on GPU.
 * Writes per-rank transport type values to output array.
 *
 * @param handle Device handle to test.
 * @param output_d GPU buffer to write per-rank type values.
 * @param numBlocks Number of CUDA blocks.
 * @param blockSize Number of threads per block.
 */
void test_device_handle_type_map(
    MultiPeerDeviceHandle handle,
    int* output_d,
    int numBlocks,
    int blockSize);

/**
 * Test self-transport put via MultiPeerDeviceHandle.
 *
 * @param handle Device handle.
 * @param dst_d Destination buffer in GPU memory.
 * @param src_d Source buffer in GPU memory.
 * @param nbytes Number of bytes to copy.
 * @param numBlocks Number of CUDA blocks.
 * @param blockSize Number of threads per block.
 */
void test_multi_peer_self_put(
    MultiPeerDeviceHandle handle,
    void* dst_d,
    const void* src_d,
    size_t nbytes,
    int numBlocks,
    int blockSize);

} // namespace comms::prims::test
