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

// Test memcpy_vectorized() with uint4 vectorized loads/stores.
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

class CopyDualTestParameterized
    : public CopyUtilsTestFixture,
      public ::testing::WithParamInterface<CopyChunkVectorizedParams> {};

TEST_P(CopyDualTestParameterized, CopyDual) {
  const auto& params = GetParam();
  const std::size_t bufferSize =
      params.nBytes + std::max(params.dstOffset, params.srcOffset);

  DeviceBuffer srcBuffer(bufferSize);
  DeviceBuffer dst1Buffer(bufferSize);
  DeviceBuffer dst2Buffer(bufferSize);
  DeviceBuffer errorCountBuffer(sizeof(uint32_t));

  auto src_d = static_cast<char*>(srcBuffer.get());
  auto dst1_d = static_cast<char*>(dst1Buffer.get());
  auto dst2_d = static_cast<char*>(dst2Buffer.get());
  auto errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());

  std::vector<char> src_h(bufferSize);
  for (std::size_t i = 0; i < bufferSize; i++) {
    src_h[i] = static_cast<char>(i % 256);
  }

  CUDACHECK_TEST(
      cudaMemcpy(src_d, src_h.data(), bufferSize, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(dst1_d, 0, bufferSize));
  CUDACHECK_TEST(cudaMemset(dst2_d, 0, bufferSize));
  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  testCopyDual(
      dst1_d + params.dstOffset,
      dst2_d + params.dstOffset,
      src_d + params.srcOffset,
      params.nBytes,
      errorCount_d,
      params.numBlocks,
      params.numThreads);

  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));

  EXPECT_EQ(errorCount_h, 0)
      << "Dual copy failed with " << errorCount_h << " mismatches"
      << " (numBlocks=" << params.numBlocks
      << ", numThreads=" << params.numThreads << ", nBytes=" << params.nBytes
      << ", dstOffset=" << params.dstOffset
      << ", srcOffset=" << params.srcOffset << ")";
}

INSTANTIATE_TEST_SUITE_P(
    CopyDualTests,
    CopyDualTestParameterized,
    ::testing::Values(
        CopyChunkVectorizedParams{1, 256, 4096, 0, 0},
        CopyChunkVectorizedParams{1, 256, 4097, 0, 0},
        CopyChunkVectorizedParams{1, 32, 128, 0, 0},
        CopyChunkVectorizedParams{4, 256, 65536, 0, 0},
        CopyChunkVectorizedParams{1, 256, 4096, 64, 0},
        CopyChunkVectorizedParams{1, 256, 4096, 0, 128},
        CopyChunkVectorizedParams{1, 256, 4096, 32, 48},
        CopyChunkVectorizedParams{2, 128, 8192, 0, 0},
        CopyChunkVectorizedParams{1, 32, 2048, 0, 0}));

struct CopyDualMisalignedParams {
  int numBlocks;
  int numThreads;
  std::size_t nBytes;
  std::size_t dst1Offset;
  std::size_t dst2Offset;
  std::size_t srcOffset;
};

class CopyDualMisalignedTestParameterized
    : public CopyUtilsTestFixture,
      public ::testing::WithParamInterface<CopyDualMisalignedParams> {};

TEST_P(CopyDualMisalignedTestParameterized, CopyDualMisaligned) {
  const auto& params = GetParam();
  const std::size_t maxOffset =
      std::max({params.dst1Offset, params.dst2Offset, params.srcOffset});
  const std::size_t bufferSize = params.nBytes + maxOffset;

  DeviceBuffer srcBuffer(bufferSize);
  DeviceBuffer dst1Buffer(bufferSize);
  DeviceBuffer dst2Buffer(bufferSize);
  DeviceBuffer errorCountBuffer(sizeof(uint32_t));

  auto src_d = static_cast<char*>(srcBuffer.get());
  auto dst1_d = static_cast<char*>(dst1Buffer.get());
  auto dst2_d = static_cast<char*>(dst2Buffer.get());
  auto errorCount_d = static_cast<uint32_t*>(errorCountBuffer.get());

  std::vector<char> src_h(bufferSize);
  for (std::size_t i = 0; i < bufferSize; i++) {
    src_h[i] = static_cast<char>(i % 256);
  }

  CUDACHECK_TEST(
      cudaMemcpy(src_d, src_h.data(), bufferSize, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(dst1_d, 0, bufferSize));
  CUDACHECK_TEST(cudaMemset(dst2_d, 0, bufferSize));
  CUDACHECK_TEST(cudaMemset(errorCount_d, 0, sizeof(uint32_t)));

  testCopyDual(
      dst1_d + params.dst1Offset,
      dst2_d + params.dst2Offset,
      src_d + params.srcOffset,
      params.nBytes,
      errorCount_d,
      params.numBlocks,
      params.numThreads);

  CUDACHECK_TEST(cudaDeviceSynchronize());

  uint32_t errorCount_h = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &errorCount_h, errorCount_d, sizeof(uint32_t), cudaMemcpyDeviceToHost));

  EXPECT_EQ(errorCount_h, 0)
      << "Dual copy (misaligned) failed with " << errorCount_h << " mismatches"
      << " (numBlocks=" << params.numBlocks
      << ", numThreads=" << params.numThreads << ", nBytes=" << params.nBytes
      << ", dst1Offset=" << params.dst1Offset
      << ", dst2Offset=" << params.dst2Offset
      << ", srcOffset=" << params.srcOffset << ")";
}

INSTANTIATE_TEST_SUITE_P(
    CopyDualMisalignedTests,
    CopyDualMisalignedTestParameterized,
    ::testing::Values(
        // src misaligned by 1 byte
        CopyDualMisalignedParams{1, 256, 4096, 0, 0, 1},
        // dst1 misaligned by 3 bytes
        CopyDualMisalignedParams{1, 256, 4096, 3, 0, 0},
        // dst2 misaligned by 7 bytes
        CopyDualMisalignedParams{1, 256, 4096, 0, 7, 0},
        // all three misaligned differently
        CopyDualMisalignedParams{1, 256, 4096, 1, 5, 3},
        // dst1 aligned, dst2 misaligned (forces char fallback)
        CopyDualMisalignedParams{1, 256, 4096, 16, 13, 0},
        // large size, all misaligned
        CopyDualMisalignedParams{4, 256, 65536, 7, 11, 3},
        // non-power-of-2 size + misalignment
        CopyDualMisalignedParams{1, 256, 4097, 5, 9, 1},
        // small size, misaligned
        CopyDualMisalignedParams{1, 32, 128, 3, 7, 1},
        // single warp, misaligned
        CopyDualMisalignedParams{1, 32, 2048, 1, 1, 1}));

} // namespace comms::pipes::test
