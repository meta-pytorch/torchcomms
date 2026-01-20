// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include <algorithm>
#include <string>
#include <vector>

#include "comms/pipes/P2pSelfTransportDevice.cuh"
#include "comms/pipes/tests/P2pSelfTransportDeviceTest.cuh"
#include "comms/pipes/tests/Utils.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

using namespace meta::comms;

namespace comms::pipes {

// Parameters for transfer size tests: (nbytes, name)
struct TransferSizeParams {
  size_t nbytes;
  std::string name;
};

class SelfTransportDeviceTestFixture : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDACHECK_TEST(cudaSetDevice(0));
  }

  void TearDown() override {}
};

// Helper to run a single write test with verification
// only support no overlaop copy for now
void runWriteNoOverlapTest(size_t nbytes, const std::string& testName) {
  const size_t numInts = nbytes / sizeof(int);
  const int testValue = 42;

  // Allocate send and receive buffers
  DeviceBuffer sendBuffer(nbytes);
  DeviceBuffer recvBuffer(nbytes);

  auto send_d = static_cast<int*>(sendBuffer.get());
  auto recv_d = static_cast<int*>(recvBuffer.get());

  // Initialize send buffer with test value
  test::fillBuffer(send_d, testValue, numInts);
  CUDACHECK_TEST(cudaMemset(recv_d, 0, nbytes)); // Clear recv buffer
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Launch write kernel
  // 32 warps
  const int numBlocks = 4;
  const int blockSize = 256;
  test::testSelfWrite(
      reinterpret_cast<char*>(recv_d),
      reinterpret_cast<const char*>(send_d),
      nbytes,
      numBlocks,
      blockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Verify received data
  DeviceBuffer errorCountBuffer(sizeof(int));
  auto d_errorCount = static_cast<int*>(errorCountBuffer.get());
  CUDACHECK_TEST(cudaMemset(d_errorCount, 0, sizeof(int)));

  test::verifyBuffer(recv_d, testValue, numInts, d_errorCount);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Copy error count back to host
  int h_errorCount = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &h_errorCount, d_errorCount, sizeof(int), cudaMemcpyDeviceToHost));

  // Assert no errors
  ASSERT_EQ(h_errorCount, 0)
      << "Test '" << testName << "' found " << h_errorCount << " errors";
}

// Parameterized test fixture for variable sizes
class TransferSizeTestFixture
    : public SelfTransportDeviceTestFixture,
      public ::testing::WithParamInterface<TransferSizeParams> {};

TEST_P(TransferSizeTestFixture, Write) {
  const auto& params = GetParam();
  runWriteNoOverlapTest(params.nbytes, params.name);
}

std::string transferSizeParamName(
    const ::testing::TestParamInfo<TransferSizeParams>& info) {
  return info.param.name;
}

INSTANTIATE_TEST_SUITE_P(
    TransferSizeVariations,
    TransferSizeTestFixture,
    ::testing::Values(
        // Very small transfers (test alignment edge cases)
        TransferSizeParams{.nbytes = 16, .name = "Size_16B"},
        TransferSizeParams{.nbytes = 32, .name = "Size_32B"},
        // Small transfer (less than vector size)
        TransferSizeParams{.nbytes = 64, .name = "Size_64B"},
        // Medium transfer (1KB)
        TransferSizeParams{.nbytes = 1024, .name = "Size_1KB"},
        // Non-aligned to vector size (16 bytes)
        TransferSizeParams{
            .nbytes = 1000,
            .name = "NonVectorAligned_1000Bytes"},
        // 64 KB transfer
        TransferSizeParams{.nbytes = 64 * 1024, .name = "Size_64KB"},
        // 512 KB transfer
        TransferSizeParams{.nbytes = 512 * 1024, .name = "Size_512KB"},
        // 1 MB transfer
        TransferSizeParams{.nbytes = 1 * 1024 * 1024, .name = "Size_1MB"},
        // 512 MB transfer
        TransferSizeParams{.nbytes = 512 * 1024 * 1024, .name = "Size_512MB"},
        // 1G transfer
        TransferSizeParams{
            .nbytes = 1 * 1024 * 1024 * 1024,
            .name = "Size_1GB"}),
    transferSizeParamName);

// Chunk boundary edge case tests (dynamic chunk sizing)
INSTANTIATE_TEST_SUITE_P(
    ChunkBoundaryEdgeCases,
    TransferSizeTestFixture,
    ::testing::Values(
        // Just below 64KB chunk boundary
        TransferSizeParams{.nbytes = 64 * 1024 - 1, .name = "Size_64KB_Minus1"},
        // Just above 64KB chunk boundary
        TransferSizeParams{.nbytes = 64 * 1024 + 1, .name = "Size_64KB_Plus1"},
        // Just below 128KB (2 chunks)
        TransferSizeParams{
            .nbytes = 128 * 1024 - 1,
            .name = "Size_128KB_Minus1"},
        // Just above 128KB (2 chunks)
        TransferSizeParams{
            .nbytes = 128 * 1024 + 1,
            .name = "Size_128KB_Plus1"},
        // Just below 256KB (4 chunks)
        TransferSizeParams{
            .nbytes = 256 * 1024 - 1,
            .name = "Size_256KB_Minus1"},
        // 3 full chunks
        TransferSizeParams{.nbytes = 192 * 1024, .name = "Size_192KB"},
        // Non-power-of-two near chunk boundary
        TransferSizeParams{
            .nbytes = 64 * 1024 - 17,
            .name = "Size_64KB_Minus17"}),
    transferSizeParamName);

// Test zero bytes edge case
TEST_F(SelfTransportDeviceTestFixture, WriteZeroBytes) {
  const size_t nbytes = 1024;

  // Allocate buffers
  DeviceBuffer sendBuffer(nbytes);
  DeviceBuffer recvBuffer(nbytes);

  auto send_d = static_cast<int*>(sendBuffer.get());
  auto recv_d = static_cast<int*>(recvBuffer.get());

  // Initialize send buffer with a test value and recv buffer with zeros
  test::fillBuffer(send_d, 42, nbytes / sizeof(int));
  CUDACHECK_TEST(cudaMemset(recv_d, 0, nbytes));
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Call write with zero bytes - should be a no-op
  const int numBlocks = 4;
  const int blockSize = 256;
  test::testSelfWrite(
      reinterpret_cast<char*>(recv_d),
      reinterpret_cast<const char*>(send_d),
      0, // zero bytes
      numBlocks,
      blockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Verify recv buffer is still all zeros (nothing was copied)
  DeviceBuffer errorCountBuffer(sizeof(int));
  auto d_errorCount = static_cast<int*>(errorCountBuffer.get());
  CUDACHECK_TEST(cudaMemset(d_errorCount, 0, sizeof(int)));

  test::verifyBuffer(recv_d, 0, nbytes / sizeof(int), d_errorCount);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  int h_errorCount = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &h_errorCount, d_errorCount, sizeof(int), cudaMemcpyDeviceToHost));

  ASSERT_EQ(h_errorCount, 0)
      << "Zero bytes test: recv buffer should remain unchanged";
}

// Test same pointer edge case (dst == src)
TEST_F(SelfTransportDeviceTestFixture, WriteSamePointer) {
  const size_t nbytes = 1024;
  const int testValue = 42;
  const size_t numInts = nbytes / sizeof(int);

  // Allocate a single buffer
  DeviceBuffer buffer(nbytes);
  auto buf_d = static_cast<int*>(buffer.get());

  // Initialize buffer with test value
  test::fillBuffer(buf_d, testValue, numInts);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Call write with same src and dst pointer - should be a no-op
  const int numBlocks = 4;
  const int blockSize = 256;
  test::testSelfWrite(
      reinterpret_cast<char*>(buf_d),
      reinterpret_cast<const char*>(buf_d),
      nbytes,
      numBlocks,
      blockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Verify buffer still has original values
  DeviceBuffer errorCountBuffer(sizeof(int));
  auto d_errorCount = static_cast<int*>(errorCountBuffer.get());
  CUDACHECK_TEST(cudaMemset(d_errorCount, 0, sizeof(int)));

  test::verifyBuffer(buf_d, testValue, numInts, d_errorCount);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  int h_errorCount = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &h_errorCount, d_errorCount, sizeof(int), cudaMemcpyDeviceToHost));

  ASSERT_EQ(h_errorCount, 0)
      << "Same pointer test: buffer should remain unchanged";
}

// Test very small sizes (smaller than vectorized size of 16 bytes)
// Uses byte-level verification since int-based helpers require >= 4 bytes
class SmallSizeTestFixture : public SelfTransportDeviceTestFixture,
                             public ::testing::WithParamInterface<size_t> {};

TEST_P(SmallSizeTestFixture, WriteSmallSize) {
  const size_t nbytes = GetParam();
  const char testValue = 0x42;

  // Allocate buffers (minimum allocation to ensure valid pointers)
  const size_t allocSize = std::max(nbytes, size_t(16));
  DeviceBuffer sendBuffer(allocSize);
  DeviceBuffer recvBuffer(allocSize);

  auto send_d = static_cast<char*>(sendBuffer.get());
  auto recv_d = static_cast<char*>(recvBuffer.get());

  // Initialize send buffer with test value using cudaMemset
  CUDACHECK_TEST(cudaMemset(send_d, testValue, nbytes));
  CUDACHECK_TEST(cudaMemset(recv_d, 0, nbytes));
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Launch write kernel
  const int numBlocks = 4;
  const int blockSize = 256;
  test::testSelfWrite(recv_d, send_d, nbytes, numBlocks, blockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Copy back to host and verify
  std::vector<char> h_recv(nbytes);
  CUDACHECK_TEST(
      cudaMemcpy(h_recv.data(), recv_d, nbytes, cudaMemcpyDeviceToHost));

  int errorCount = 0;
  for (size_t i = 0; i < nbytes; ++i) {
    if (h_recv[i] != testValue) {
      ++errorCount;
    }
  }

  ASSERT_EQ(errorCount, 0) << "Small size test (" << nbytes << " bytes) found "
                           << errorCount << " errors";
}

INSTANTIATE_TEST_SUITE_P(
    SmallSizeVariations,
    SmallSizeTestFixture,
    ::testing::Values(
        1, // Single byte
        2, // Two bytes
        3, // Three bytes (odd)
        4, // One int
        7, // Prime number
        8, // Two ints
        15, // Just below vectorized size
        17 // Just above vectorized size
        ));

// Parameters for thread configuration tests
struct ThreadConfigParams {
  int numBlocks;
  int blockSize;
  std::string name;
};

class ThreadConfigTestFixture
    : public SelfTransportDeviceTestFixture,
      public ::testing::WithParamInterface<ThreadConfigParams> {};

TEST_P(ThreadConfigTestFixture, WriteWithDifferentConfig) {
  const auto& params = GetParam();
  const size_t nbytes = 256 * 1024; // 256KB - ensures multiple chunks
  const size_t numInts = nbytes / sizeof(int);
  const int testValue = 42;

  // Allocate send and receive buffers
  DeviceBuffer sendBuffer(nbytes);
  DeviceBuffer recvBuffer(nbytes);

  auto send_d = static_cast<int*>(sendBuffer.get());
  auto recv_d = static_cast<int*>(recvBuffer.get());

  // Initialize send buffer with test value
  test::fillBuffer(send_d, testValue, numInts);
  CUDACHECK_TEST(cudaMemset(recv_d, 0, nbytes));
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Launch write kernel with parameterized configuration
  test::testSelfWrite(
      reinterpret_cast<char*>(recv_d),
      reinterpret_cast<const char*>(send_d),
      nbytes,
      params.numBlocks,
      params.blockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Verify received data
  DeviceBuffer errorCountBuffer(sizeof(int));
  auto d_errorCount = static_cast<int*>(errorCountBuffer.get());
  CUDACHECK_TEST(cudaMemset(d_errorCount, 0, sizeof(int)));

  test::verifyBuffer(recv_d, testValue, numInts, d_errorCount);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  int h_errorCount = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &h_errorCount, d_errorCount, sizeof(int), cudaMemcpyDeviceToHost));

  ASSERT_EQ(h_errorCount, 0) << "Thread config test '" << params.name
                             << "' found " << h_errorCount << " errors";
}

std::string threadConfigParamName(
    const ::testing::TestParamInfo<ThreadConfigParams>& info) {
  return info.param.name;
}

INSTANTIATE_TEST_SUITE_P(
    ThreadConfigVariations,
    ThreadConfigTestFixture,
    ::testing::Values(
        // Single block, single warp
        ThreadConfigParams{.numBlocks = 1, .blockSize = 32, .name = "1B_32T"},
        // Single block, multiple warps
        ThreadConfigParams{.numBlocks = 1, .blockSize = 128, .name = "1B_128T"},
        ThreadConfigParams{.numBlocks = 1, .blockSize = 256, .name = "1B_256T"},
        // Multiple blocks, small block size
        ThreadConfigParams{.numBlocks = 2, .blockSize = 64, .name = "2B_64T"},
        ThreadConfigParams{.numBlocks = 8, .blockSize = 32, .name = "8B_32T"},
        // Larger configurations
        ThreadConfigParams{.numBlocks = 8, .blockSize = 256, .name = "8B_256T"},
        ThreadConfigParams{
            .numBlocks = 16,
            .blockSize = 256,
            .name = "16B_256T"},
        // Large block size (512 threads)
        ThreadConfigParams{.numBlocks = 4, .blockSize = 512, .name = "4B_512T"},
        // High block counts to stress dynamic chunk sizing
        ThreadConfigParams{
            .numBlocks = 32,
            .blockSize = 256,
            .name = "32B_256T"},
        ThreadConfigParams{
            .numBlocks = 64,
            .blockSize = 256,
            .name = "64B_256T"},
        // Many blocks with small block size (tests many small chunks)
        ThreadConfigParams{
            .numBlocks = 64,
            .blockSize = 32,
            .name = "64B_32T"}),
    threadConfigParamName);

// Helper for unaligned pointer tests - uses byte-level operations instead of
// int* because unaligned int* casts are undefined behavior
void runUnalignedWriteTest(
    char* dst_d,
    char* src_d,
    size_t nbytes,
    const std::string& testName) {
  const char testValue = 0x42;

  // Initialize using cudaMemset (works with any alignment)
  CUDACHECK_TEST(cudaMemset(src_d, testValue, nbytes));
  CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Launch write kernel
  const int numBlocks = 4;
  const int blockSize = 256;
  test::testSelfWrite(dst_d, src_d, nbytes, numBlocks, blockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Copy back to host and verify byte-by-byte
  std::vector<char> h_recv(nbytes);
  CUDACHECK_TEST(
      cudaMemcpy(h_recv.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

  int errorCount = 0;
  for (size_t i = 0; i < nbytes; ++i) {
    if (h_recv[i] != testValue) {
      ++errorCount;
    }
  }

  ASSERT_EQ(errorCount, 0) << "Test '" << testName << "' found " << errorCount
                           << " errors out of " << nbytes << " bytes";
}

class UnalignedPointerTestFixture
    : public SelfTransportDeviceTestFixture,
      public ::testing::WithParamInterface<size_t> {};

TEST_P(UnalignedPointerTestFixture, WriteWithUnalignedSrcPointer) {
  const size_t offset = GetParam();
  const size_t nbytes = 1024;

  // Allocate buffers
  DeviceBuffer sendBuffer(nbytes);
  DeviceBuffer recvBuffer(nbytes + offset);

  auto send_d = static_cast<char*>(sendBuffer.get());
  auto recv_d = static_cast<char*>(recvBuffer.get()) + offset;

  runUnalignedWriteTest(
      send_d, recv_d, nbytes, "UnalignedSrc_" + std::to_string(offset));
}

TEST_P(UnalignedPointerTestFixture, WriteWithUnalignedDstPointer) {
  const size_t offset = GetParam();
  const size_t nbytes = 1024;

  // Allocate buffers
  DeviceBuffer sendBuffer(nbytes + offset);
  DeviceBuffer recvBuffer(nbytes);

  auto send_d = static_cast<char*>(sendBuffer.get()) + offset;
  auto recv_d = static_cast<char*>(recvBuffer.get());

  runUnalignedWriteTest(
      send_d, recv_d, nbytes, "UnalignedDst_" + std::to_string(offset));
}

TEST_P(UnalignedPointerTestFixture, WriteWithBothPointersUnaligned) {
  const size_t offset = GetParam();
  const size_t nbytes = 1024;

  // Allocate buffers
  DeviceBuffer sendBuffer(nbytes + offset);
  DeviceBuffer recvBuffer(nbytes + offset);

  auto send_d = static_cast<char*>(sendBuffer.get()) + offset;
  auto recv_d = static_cast<char*>(recvBuffer.get()) + offset;

  runUnalignedWriteTest(
      send_d, recv_d, nbytes, "BothUnaligned_" + std::to_string(offset));
}

INSTANTIATE_TEST_SUITE_P(
    UnalignedPointerEdgeCases,
    UnalignedPointerTestFixture,
    ::testing::Values(1, 2, 3, 4, 5, 7, 8, 9, 13, 15));

} // namespace comms::pipes

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
