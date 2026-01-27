// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include "comms/pipes/GpuMemHandler.h"
#include "comms/pipes/tests/Utils.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

using comms::pipes::GpuMemHandler;
using comms::pipes::MemSharingMode;
using meta::comms::DeviceBuffer;
using meta::comms::MpiBaseTestFixture;
using meta::comms::MpiBootstrap;
using meta::comms::MPIEnvironmentBase;

namespace comms::pipes::tests {

class GpuMemHandlerTestFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }

  void TearDown() override {
    MpiBaseTestFixture::TearDown();
  }
};

/**
 * Test basic IPC memory access via GpuMemHandler.
 *
 * Each rank allocates memory, exchanges handles, then:
 * - Writes its rank value to local buffer
 * - Reads from peer's buffer and verifies peer's rank value
 */
TEST_F(GpuMemHandlerTestFixture, RemoteWriteLocalRead) {
  // Only test with 2 ranks
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;

  const size_t numElements = 256;
  const size_t bufferSize = sizeof(int) * numElements;

  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  GpuMemHandler handler(bootstrap, globalRank, numRanks, bufferSize);

  XLOGF(
      INFO,
      "Rank {} created handler in {} mode",
      globalRank,
      handler.getMode() == MemSharingMode::kFabric ? "fabric" : "cudaIpc");

  handler.exchangeMemPtrs();
  XLOGF(INFO, "Rank {} exchanged memory handles", globalRank);

  auto localAddr = static_cast<int*>(handler.getLocalDeviceMemPtr());
  auto remoteAddr = static_cast<int*>(handler.getPeerDeviceMemPtr(peerRank));

  XLOGF(
      INFO,
      "Rank {}: localAddr: {}, remoteAddr: {}",
      globalRank,
      static_cast<void*>(localAddr),
      static_cast<void*>(remoteAddr));

  // Each rank writes its rank value to local buffer
  // rank0 writes all 0s, rank1 writes all 1s
  int writeValue = globalRank;
  test::fillBuffer(localAddr, writeValue, numElements);
  CUDACHECK_TEST(cudaDeviceSynchronize());
  XLOGF(INFO, "Rank {} filled local buffer with {}", globalRank, writeValue);

  // Barrier to ensure both ranks have written their data
  MPI_Barrier(MPI_COMM_WORLD);
  XLOGF(INFO, "Rank {} passed barrier", globalRank);

  // Each rank reads from peer's buffer and verifies
  // rank0 should read all 1s from rank1
  // rank1 should read all 0s from rank0
  int expectedValue = peerRank;

  // Allocate error counter on device using DeviceBuffer
  DeviceBuffer errorCountBuffer(sizeof(int));
  auto d_errorCount = static_cast<int*>(errorCountBuffer.get());
  CUDACHECK_TEST(cudaMemset(d_errorCount, 0, sizeof(int)));

  test::verifyBuffer(remoteAddr, expectedValue, numElements, d_errorCount);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Copy error count back to host
  int h_errorCount = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &h_errorCount, d_errorCount, sizeof(int), cudaMemcpyDeviceToHost));

  XLOGF(
      INFO,
      "Rank {} verified peer buffer, errors: {}",
      globalRank,
      h_errorCount);

  // Assert no errors
  ASSERT_EQ(h_errorCount, 0)
      << "Rank " << globalRank << " found " << h_errorCount
      << " errors when reading from peer rank " << peerRank;
}

/**
 * Test that self rank returns local pointer.
 */
TEST_F(GpuMemHandlerTestFixture, SelfRankReturnsLocalPtr) {
  // Only test with 2 ranks
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const size_t bufferSize = 1024;

  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  GpuMemHandler handler(bootstrap, globalRank, numRanks, bufferSize);
  handler.exchangeMemPtrs();

  // getPeerDeviceMemPtr(selfRank) should return the local pointer
  void* localPtr = handler.getLocalDeviceMemPtr();
  void* selfPtr = handler.getPeerDeviceMemPtr(globalRank);

  EXPECT_EQ(localPtr, selfPtr)
      << "getPeerDeviceMemPtr(selfRank) should return local pointer";
}

/**
 * Test explicit cudaIpc mode.
 */
TEST_F(GpuMemHandlerTestFixture, ExplicitCudaIpcMode) {
  // Only test with 2 ranks
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;

  const size_t numElements = 128;
  const size_t bufferSize = sizeof(int) * numElements;

  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  // Explicitly request cudaIpc mode
  GpuMemHandler handler(
      bootstrap, globalRank, numRanks, bufferSize, MemSharingMode::kCudaIpc);

  EXPECT_EQ(handler.getMode(), MemSharingMode::kCudaIpc);
  XLOGF(INFO, "Rank {} created handler with explicit cudaIpc mode", globalRank);

  handler.exchangeMemPtrs();

  auto localAddr = static_cast<int*>(handler.getLocalDeviceMemPtr());
  auto remoteAddr = static_cast<int*>(handler.getPeerDeviceMemPtr(peerRank));

  // Write and verify
  int writeValue = globalRank + 100;
  test::fillBuffer(localAddr, writeValue, numElements);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  MPI_Barrier(MPI_COMM_WORLD);

  int expectedValue = peerRank + 100;

  DeviceBuffer errorCountBuffer(sizeof(int));
  auto d_errorCount = static_cast<int*>(errorCountBuffer.get());
  CUDACHECK_TEST(cudaMemset(d_errorCount, 0, sizeof(int)));

  test::verifyBuffer(remoteAddr, expectedValue, numElements, d_errorCount);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  int h_errorCount = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &h_errorCount, d_errorCount, sizeof(int), cudaMemcpyDeviceToHost));

  ASSERT_EQ(h_errorCount, 0) << "Rank " << globalRank << " found "
                             << h_errorCount << " errors in cudaIpc mode test";
}

/**
 * Test single rank exchange (nRanks=1).
 *
 * Each MPI rank independently tests the single-rank scenario where
 * we only exchange with ourselves. This verifies that GpuMemHandler
 * works correctly when there are no peers.
 */
TEST_F(GpuMemHandlerTestFixture, SingleRankExchange) {
  const size_t numElements = 256;
  const size_t bufferSize = sizeof(int) * numElements;

  // Create a single-rank handler (nRanks=1, selfRank=0)
  // Each MPI rank tests this independently
  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  GpuMemHandler handler(
      bootstrap, 0 /* selfRank */, 1 /* nRanks */, bufferSize);

  XLOGF(
      INFO,
      "MPI Rank {} testing single-rank exchange in {} mode",
      globalRank,
      handler.getMode() == MemSharingMode::kFabric ? "fabric" : "cudaIpc");

  handler.exchangeMemPtrs();
  XLOGF(INFO, "MPI Rank {} completed single-rank exchange", globalRank);

  // Get local pointer
  auto localAddr = static_cast<int*>(handler.getLocalDeviceMemPtr());
  ASSERT_NE(localAddr, nullptr) << "Local pointer should not be null";

  // getPeerDeviceMemPtr(0) should return the same as local pointer
  auto selfPtr = static_cast<int*>(handler.getPeerDeviceMemPtr(0));
  EXPECT_EQ(localAddr, selfPtr)
      << "getPeerDeviceMemPtr(0) should return local pointer in single-rank "
         "mode";

  // Write to local buffer and verify we can read it back
  int writeValue = 42;
  test::fillBuffer(localAddr, writeValue, numElements);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Verify the data through the "peer" pointer (which is the same as local)
  DeviceBuffer errorCountBuffer(sizeof(int));
  auto d_errorCount = static_cast<int*>(errorCountBuffer.get());
  CUDACHECK_TEST(cudaMemset(d_errorCount, 0, sizeof(int)));

  test::verifyBuffer(selfPtr, writeValue, numElements, d_errorCount);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  int h_errorCount = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &h_errorCount, d_errorCount, sizeof(int), cudaMemcpyDeviceToHost));

  ASSERT_EQ(h_errorCount, 0) << "Single-rank exchange verification failed";

  XLOGF(INFO, "MPI Rank {} single-rank exchange test passed", globalRank);
}

} // namespace comms::pipes::tests

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
