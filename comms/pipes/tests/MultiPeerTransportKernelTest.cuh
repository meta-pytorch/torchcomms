// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstddef>

#include "comms/pipes/MultiPeerDeviceHandle.cuh"

namespace comms::pipes::test {

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
 * End-to-end NVL send via MultiPeerDeviceHandle.
 *
 * @param handle Device handle with NVL transports.
 * @param peerRank Rank of the NVL peer to send to.
 * @param src_d Source buffer in GPU memory.
 * @param nbytes Number of bytes to send.
 * @param numBlocks Number of CUDA blocks.
 * @param blockSize Number of threads per block.
 */
void test_multi_peer_nvl_send(
    MultiPeerDeviceHandle handle,
    int peerRank,
    void* src_d,
    size_t nbytes,
    int numBlocks,
    int blockSize);

/**
 * End-to-end NVL recv via MultiPeerDeviceHandle.
 *
 * @param handle Device handle with NVL transports.
 * @param peerRank Rank of the NVL peer to receive from.
 * @param dst_d Destination buffer in GPU memory.
 * @param nbytes Number of bytes to receive.
 * @param numBlocks Number of CUDA blocks.
 * @param blockSize Number of threads per block.
 */
void test_multi_peer_nvl_recv(
    MultiPeerDeviceHandle handle,
    int peerRank,
    void* dst_d,
    size_t nbytes,
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

/**
 * Verify IBGDA transport accessor via MultiPeerDeviceHandle.
 *
 * For each rank, if the transport type is P2P_IBGDA, reads the signal
 * count from the device-side P2pIbgdaTransportDevice and writes it to
 * output_d[rank]. Non-IBGDA ranks get -1.
 *
 * @param handle Device handle with IBGDA transports.
 * @param output_d GPU buffer to write per-rank signal counts (size = nRanks).
 * @param numBlocks Number of CUDA blocks.
 * @param blockSize Number of threads per block.
 */
void test_ibgda_accessor(
    MultiPeerDeviceHandle handle,
    int* output_d,
    int numBlocks,
    int blockSize);

} // namespace comms::pipes::test
