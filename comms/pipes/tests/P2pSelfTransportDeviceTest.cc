// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include <string>

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
        // Very small transfer (less than vector size)
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

} // namespace comms::pipes

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
