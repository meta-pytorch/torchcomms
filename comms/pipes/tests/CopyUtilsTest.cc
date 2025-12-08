// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "comms/pipes/tests/CopyUtilsTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

namespace comms::pipes::test {

using meta::comms::DeviceBuffer;

class CopyUtilsTestFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    int device;
    CUDACHECK_TEST(cudaGetDevice(&device));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaDeviceSynchronize());
  }
};

struct CopyChunkVectorizedParams {
  int numBlocks;
  int numThreads;
  std::size_t nBytes;
  std::size_t dstOffset;
  std::size_t srcOffset;
};

class CopyUtilsTestParameterized
    : public CopyUtilsTestFixture,
      public ::testing::WithParamInterface<CopyChunkVectorizedParams> {};

// Test copy_chunk_vectorized() with uint4 vectorized loads/stores.
// The function uses 16-byte (uint4) vectorized memory operations with 4x
// unrolling to efficiently copy data. Threads within a warp cooperate to
// process the chunk, with each thread handling a strided subset of the data.
// This test verifies that all bytes are copied correctly from source to
// destination, including handling of unaligned remainders.
TEST_P(CopyUtilsTestParameterized, CopyChunkVectorized) {
  const auto& params = GetParam();
  const std::size_t bufferSize =
      params.nBytes + std::max(params.dstOffset, params.srcOffset);

  // Allocate device buffers for source, destination, and error counting
  DeviceBuffer srcBuffer(bufferSize);
  DeviceBuffer dstBuffer(bufferSize);
  DeviceBuffer errorCountBuffer(sizeof(uint32_t));

  auto src_d = static_cast<char*>(srcBuffer.get());
  auto dst_d = static_cast<char*>(dstBuffer.get());
  auto errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());

  // Initialize source with test pattern: repeating sequence 0-255
  std::vector<char> src_h(bufferSize);
  for (std::size_t i = 0; i < bufferSize; i++) {
    src_h[i] = static_cast<char>(i % 256);
  }

  // Transfer test data to device and zero out destination
  CUDACHECK_TEST(
      cudaMemcpy(src_d, src_h.data(), bufferSize, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(dst_d, 0, bufferSize));
  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  // Execute vectorized copy: warp threads cooperate to copy the chunk
  testCopyChunkVectorized(
      dst_d + params.dstOffset,
      src_d + params.srcOffset,
      params.nBytes,
      errorCount_d,
      params.numBlocks,
      params.numThreads);

  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Verify correctness: kernel compares src and dst byte-by-byte on device
  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));

  EXPECT_EQ(errorCount_h, 0)
      << "Copy failed with " << errorCount_h << " mismatches"
      << " (numBlocks=" << params.numBlocks
      << ", numThreads=" << params.numThreads << ", nBytes=" << params.nBytes
      << ", dstOffset=" << params.dstOffset
      << ", srcOffset=" << params.srcOffset << ")";
}

INSTANTIATE_TEST_SUITE_P(
    CopyUtilsTests,
    CopyUtilsTestParameterized,
    ::testing::Values(
        // Basic aligned case: 4KB, no offsets
        CopyChunkVectorizedParams{1, 256, 4096, 0, 0},
        // Unaligned size (not multiple of 64 bytes)
        CopyChunkVectorizedParams{1, 256, 4097, 0, 0},
        // Small size (less than warp size * vector size)
        CopyChunkVectorizedParams{1, 32, 128, 0, 0},
        // Large size with multiple blocks
        CopyChunkVectorizedParams{4, 256, 65536, 0, 0},
        // Non-zero destination offset (16-byte aligned for uint4)
        CopyChunkVectorizedParams{1, 256, 4096, 64, 0},
        // Non-zero source offset (16-byte aligned for uint4)
        CopyChunkVectorizedParams{1, 256, 4096, 0, 128},
        // Both offsets non-zero (16-byte aligned for uint4)
        CopyChunkVectorizedParams{1, 256, 4096, 32, 48},
        // Different thread count
        CopyChunkVectorizedParams{2, 128, 8192, 0, 0},
        // Edge case: single warp
        CopyChunkVectorizedParams{1, 32, 2048, 0, 0}));

} // namespace comms::pipes::test
