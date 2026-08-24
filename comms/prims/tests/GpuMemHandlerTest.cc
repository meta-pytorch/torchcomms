// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include "comms/utils/logger/SpdlogLogger.h"

#include <cstddef>
#include <cstring>
#include <string>

#include "comms/common/bootstrap/tests/MockBootstrap.h"
#include "comms/prims/memory/CuMemAllocation.h"
#include "comms/prims/memory/GpuMemHandler.h"
#include "comms/prims/memory/NvlMemExchange.h"
#include "comms/prims/tests/Utils.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

using comms::prims::GpuMemHandler;
using comms::prims::MemSharingMode;
using meta::comms::DeviceBuffer;
using meta::comms::MpiBaseTestFixture;
using meta::comms::MpiBootstrap;
using meta::comms::MPIEnvironmentBase;

namespace comms::prims::tests {

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
    COMMS_LOG(
        WARN, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;

  const size_t numElements = 256;
  const size_t bufferSize = sizeof(int) * numElements;

  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  GpuMemHandler handler(bootstrap, globalRank, numRanks, bufferSize);

  COMMS_LOG(
      INFO,
      "Rank {} created handler in {} mode",
      globalRank,
      handler.getMode() == MemSharingMode::kFabric ? "fabric" : "cudaIpc");

  handler.exchangeMemPtrs();
  COMMS_LOG(INFO, "Rank {} exchanged memory handles", globalRank);

  auto localAddr = static_cast<int*>(handler.getLocalDeviceMemPtr());
  auto remoteAddr = static_cast<int*>(handler.getPeerDeviceMemPtr(peerRank));

  COMMS_LOG(
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
  COMMS_LOG(
      INFO, "Rank {} filled local buffer with {}", globalRank, writeValue);

  // Barrier to ensure both ranks have written their data
  MPI_Barrier(MPI_COMM_WORLD);
  COMMS_LOG(INFO, "Rank {} passed barrier", globalRank);

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

  COMMS_LOG(
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
    COMMS_LOG(
        WARN, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
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
    COMMS_LOG(
        WARN, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
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
  COMMS_LOG(
      INFO, "Rank {} created handler with explicit cudaIpc mode", globalRank);

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

  COMMS_LOG(
      INFO,
      "MPI Rank {} testing single-rank exchange in {} mode",
      globalRank,
      handler.getMode() == MemSharingMode::kFabric ? "fabric" : "cudaIpc");

  handler.exchangeMemPtrs();
  COMMS_LOG(INFO, "MPI Rank {} completed single-rank exchange", globalRank);

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

  COMMS_LOG(INFO, "MPI Rank {} single-rank exchange test passed", globalRank);
}

TEST_F(GpuMemHandlerTestFixture, VmmImportFailurePropagatesToEveryRank) {
  if (numRanks != 2) {
    GTEST_SKIP() << "Test requires exactly two ranks";
  }

  using meta::comms::testing::MockBootstrap;
  using ::testing::_;

  auto realBootstrap = std::make_shared<MpiBootstrap>();
  auto bootstrap = std::make_shared<::testing::NiceMock<MockBootstrap>>();
  int allGatherCalls = 0;

  // The first allGather exchanges one VMM record per rank. After the real
  // collective completes, rank 0 changes rank 1's received handle type to
  // kUnsupported in rank 0's local receive buffer. Rank 0 therefore fails
  // while importing rank 1's handle. The following status allGather propagates
  // that rank-local import failure to rank 1. Later calls are not corrupted,
  // allowing the final exchangeMemPtrs() call to verify recovery.
  ON_CALL(*bootstrap, allGather(_, _, _, _))
      .WillByDefault([&](void* buf, int len, int rank, int nRanks) {
        // MpiBootstrap completes MPI_Allgather synchronously before returning
        // its ready future, so the received records are stable here.
        auto result = realBootstrap->allGather(buf, len, rank, nRanks);
        if (allGatherCalls++ == 0 && globalRank == 0) {
          if (len < static_cast<int>(sizeof(ShareableHandle))) {
            ADD_FAILURE() << "VMM exchange record is smaller than its handle";
            return result;
          }
          const auto unsupported = ShareableHandleType::kUnsupported;
          // len is the per-rank record size, so this selects rank 1's record.
          auto* peerRecord = static_cast<std::byte*>(buf) + len;
          std::memcpy(peerRecord, &unsupported, sizeof(unsupported));
        }
        return result;
      });
  EXPECT_CALL(*bootstrap, barrier(_, _)).Times(0);

  GpuMemHandler handler(bootstrap, globalRank, numRanks, 4096);
  if (handler.getMode() == MemSharingMode::kCudaIpc ||
      handler.getMode() == MemSharingMode::kCudaIpcUncached) {
    GTEST_SKIP() << "Test requires a VMM-backed sharing mode";
  }

  std::string error;
  try {
    handler.exchangeMemPtrs();
  } catch (const std::exception& ex) {
    error = ex.what();
  }

  EXPECT_THAT(error, ::testing::HasSubstr("peer import failed on rank 0"));
  if (globalRank == 0) {
    EXPECT_THAT(
        error,
        ::testing::HasSubstr(
            "NvlMemExchange: cannot import unsupported handle type"));
  }
  EXPECT_NO_THROW(handler.exchangeMemPtrs());
}

TEST_F(GpuMemHandlerTestFixture, VmmExportFailurePropagatesToEveryRank) {
  if (numRanks != 2) {
    GTEST_SKIP() << "Test requires exactly two ranks";
  }

#if CUDART_VERSION < 12030
  GTEST_SKIP() << "VMM exchange requires CUDA 12.3+";
#else
  int cudaDevice = 0;
  CUDACHECK_TEST(cudaGetDevice(&cudaDevice));
  const auto cuDevice = static_cast<CUdevice>(cudaDevice);
  auto allocation = CuMemAllocation::create(
      cuDevice, 4096, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);

  // Rank 0 passes an invalid zero allocation handle while rank 1 passes its
  // valid allocation handle. Rank 0 consequently fails inside
  // cuMemExportToShareableHandle. The handle/status allGather then reports
  // rank 0's local export failure to both ranks.
  auto bootstrap = std::make_shared<MpiBootstrap>();
  std::string error;
  try {
    (void)nvlMemExchangeVmm(
        *bootstrap,
        globalRank,
        numRanks,
        cuDevice,
        globalRank == 0 ? 0 : allocation->handle(),
        nullptr,
        allocation->size(),
        /*preferFabric=*/false);
  } catch (const std::exception& ex) {
    error = ex.what();
  }

  EXPECT_THAT(error, ::testing::HasSubstr("handle export failed on rank 0"));
  if (globalRank == 0) {
    EXPECT_THAT(
        error,
        ::testing::HasSubstr(
            "cuMemExportToShareableHandle for POSIX FD failed"));
  }
#endif
}

} // namespace comms::prims::tests

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
