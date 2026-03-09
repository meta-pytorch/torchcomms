// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include <memory>
#include <string>
#include <vector>

#include "comms/pipes/MultipeerIbgdaTransport.h"
#include "comms/pipes/tests/MultipeerIbgdaTransportTest.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::DeviceBuffer;
using meta::comms::MpiBaseTestFixture;
using meta::comms::MpiBootstrap;
using meta::comms::MPIEnvironmentBase;

namespace comms::pipes::tests {

// =============================================================================
// Test Fixture
// =============================================================================

class MultipeerIbgdaTransportTestFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }

  void TearDown() override {
    MpiBaseTestFixture::TearDown();
  }

  // Helper: Create transport with default config
  std::unique_ptr<MultipeerIbgdaTransport> createTransport() {
    MultipeerIbgdaTransportConfig config{
        .cudaDevice = localRank,
    };

    auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
    auto transport = std::make_unique<MultipeerIbgdaTransport>(
        globalRank, numRanks, bootstrap, config);
    transport->exchange();
    return transport;
  }
};

// =============================================================================
// Basic Construction and Exchange Test
// =============================================================================

TEST_F(MultipeerIbgdaTransportTestFixture, ConstructAndExchange) {
  if (numRanks < 2) {
    XLOGF(
        WARNING, "Skipping test: requires at least 2 ranks, got {}", numRanks);
    return;
  }

  try {
    auto transport = createTransport();

    EXPECT_EQ(transport->myRank(), globalRank);
    EXPECT_EQ(transport->nRanks(), numRanks);
    EXPECT_EQ(transport->numPeers(), numRanks - 1);
    EXPECT_NE(transport->getDeviceTransportPtr(), nullptr);

    XLOGF(
        INFO,
        "Rank {}: Transport created with GID index {}",
        globalRank,
        transport->getGidIndex());
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  XLOGF(INFO, "Rank {}: ConstructAndExchange test completed", globalRank);
}

// =============================================================================
// Put/Signal Basic Test - Verifies RDMA data transfer correctness
// =============================================================================

TEST_F(MultipeerIbgdaTransportTestFixture, PutSignalBasic) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const std::size_t nbytes = 64 * 1024; // 64KB transfer
  const int numBlocks = 1;
  const int blockSize = 32;
  const int peerRank = (globalRank == 0) ? 1 : 0;
  const uint8_t testPattern = 0x42;

  try {
    auto transport = createTransport();

    // Allocate and register user-owned data buffer
    DeviceBuffer dataBuffer(nbytes);
    auto localDataBuf = transport->registerBuffer(dataBuffer.get(), nbytes);

    // Collectively exchange data buffer info
    auto remoteDataBufs = transport->exchangeBuffer(localDataBuf);
    int peerIndex = (peerRank < globalRank) ? peerRank : (peerRank - 1);
    auto remoteDataBuf = remoteDataBufs[peerIndex];

    // Allocate and register signal buffer (1 slot)
    const std::size_t signalBufSize = sizeof(uint64_t);
    DeviceBuffer signalBuffer(signalBufSize);
    CUDACHECK_TEST(cudaMemset(signalBuffer.get(), 0, signalBufSize));
    auto localSignalBuf =
        transport->registerBuffer(signalBuffer.get(), signalBufSize);
    auto remoteSignalBufs = transport->exchangeBuffer(localSignalBuf);
    auto remoteSignalBuf = remoteSignalBufs[peerIndex];

    XLOGF(
        INFO,
        "Rank {}: localDataBuf ptr={} lkey={}, remoteDataBuf ptr={} rkey={}",
        globalRank,
        localDataBuf.ptr,
        localDataBuf.lkey.value,
        remoteDataBuf.ptr,
        remoteDataBuf.rkey.value);

    // Get peer transport for explicit peer selection
    P2pIbgdaTransportDevice* peerTransportPtr =
        transport->getP2pTransportDevice(peerRank);

    if (globalRank == 0) {
      // Rank 0: Sender
      // Fill local buffer with test pattern
      test::fillBufferWithPattern(
          localDataBuf.ptr, nbytes, testPattern, numBlocks, blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      // Perform RDMA put with signal
      test::testPutAndSignal(
          peerTransportPtr,
          localDataBuf,
          remoteDataBuf,
          nbytes,
          remoteSignalBuf,
          0, // signal id
          1, // signal value
          numBlocks,
          blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    } else {
      // Rank 1: Receiver
      // Clear local buffer
      CUDACHECK_TEST(cudaMemset(localDataBuf.ptr, 0, nbytes));
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      // Wait for signal from sender
      test::testWaitSignal(
          static_cast<uint64_t*>(signalBuffer.get()),
          0, // signal id
          1, // expected signal
          numBlocks,
          blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      // Verify received data
      DeviceBuffer errorCountBuf(sizeof(int));
      auto* d_errorCount = static_cast<int*>(errorCountBuf.get());
      CUDACHECK_TEST(cudaMemset(d_errorCount, 0, sizeof(int)));

      test::verifyBufferPattern(
          localDataBuf.ptr,
          nbytes,
          testPattern,
          d_errorCount,
          numBlocks,
          blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      int h_errorCount = 0;
      CUDACHECK_TEST(cudaMemcpy(
          &h_errorCount, d_errorCount, sizeof(int), cudaMemcpyDeviceToHost));

      EXPECT_EQ(h_errorCount, 0)
          << "Rank " << globalRank << ": Found " << h_errorCount
          << " byte mismatches out of " << nbytes << " bytes";

      if (h_errorCount > 0) {
        // Print first few mismatches for debugging
        std::vector<uint8_t> hostBuf(std::min(nbytes, std::size_t(256)));
        CUDACHECK_TEST(cudaMemcpy(
            hostBuf.data(),
            localDataBuf.ptr,
            hostBuf.size(),
            cudaMemcpyDeviceToHost));
        XLOGF(ERR, "First bytes received:");
        for (size_t i = 0; i < std::min(size_t(16), hostBuf.size()); i++) {
          XLOGF(
              ERR,
              "  [{}] expected=0x{:02x} got=0x{:02x}",
              i,
              static_cast<uint8_t>(testPattern + (i % 256)),
              hostBuf[i]);
        }
      }
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }

  XLOGF(INFO, "Rank {}: PutSignalBasic test completed", globalRank);
}

// =============================================================================
// Group-Level Put/Signal Basic Test - Verifies group-collaborative RDMA
// =============================================================================

TEST_F(MultipeerIbgdaTransportTestFixture, PutSignalGroupBasic) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const std::size_t nbytes = 64 * 1024;
  const int numBlocks = 1;
  const int blockSize = 32;
  const int peerRank = (globalRank == 0) ? 1 : 0;
  const uint8_t testPattern = 0x47;

  try {
    auto transport = createTransport();

    DeviceBuffer dataBuffer(nbytes);
    auto localDataBuf = transport->registerBuffer(dataBuffer.get(), nbytes);
    auto remoteDataBufs = transport->exchangeBuffer(localDataBuf);
    int peerIndex = (peerRank < globalRank) ? peerRank : (peerRank - 1);
    auto remoteDataBuf = remoteDataBufs[peerIndex];

    const std::size_t signalBufSize = sizeof(uint64_t);
    DeviceBuffer signalBuffer(signalBufSize);
    CUDACHECK_TEST(cudaMemset(signalBuffer.get(), 0, signalBufSize));
    auto localSignalBuf =
        transport->registerBuffer(signalBuffer.get(), signalBufSize);
    auto remoteSignalBufs = transport->exchangeBuffer(localSignalBuf);
    auto remoteSignalBuf = remoteSignalBufs[peerIndex];

    P2pIbgdaTransportDevice* peerTransportPtr =
        transport->getP2pTransportDevice(peerRank);

    if (globalRank == 0) {
      test::fillBufferWithPattern(
          localDataBuf.ptr, nbytes, testPattern, numBlocks, blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      test::testPutAndSignalGroup(
          peerTransportPtr,
          localDataBuf,
          remoteDataBuf,
          nbytes,
          remoteSignalBuf,
          0,
          1,
          numBlocks,
          blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    } else {
      CUDACHECK_TEST(cudaMemset(localDataBuf.ptr, 0, nbytes));
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      test::testWaitSignal(
          static_cast<uint64_t*>(signalBuffer.get()),
          0,
          1,
          numBlocks,
          blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      DeviceBuffer errorCountBuf(sizeof(int));
      auto* d_errorCount = static_cast<int*>(errorCountBuf.get());
      CUDACHECK_TEST(cudaMemset(d_errorCount, 0, sizeof(int)));

      test::verifyBufferPattern(
          localDataBuf.ptr,
          nbytes,
          testPattern,
          d_errorCount,
          numBlocks,
          blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      int h_errorCount = 0;
      CUDACHECK_TEST(cudaMemcpy(
          &h_errorCount, d_errorCount, sizeof(int), cudaMemcpyDeviceToHost));

      EXPECT_EQ(h_errorCount, 0) << "Rank " << globalRank << ": Found "
                                 << h_errorCount << " byte mismatches out of "
                                 << nbytes << " bytes (group put+signal)";
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }

  XLOGF(INFO, "Rank {}: PutSignalGroupBasic test completed", globalRank);
}

// =============================================================================
// Multi-Warp Group Put/Signal Test
// =============================================================================

TEST_F(MultipeerIbgdaTransportTestFixture, PutSignalGroupMultiWarp) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const std::size_t nbytes = 65000;
  const int numBlocks = 4;
  const int blockSize = 128;
  const int numWarps = numBlocks * (blockSize / 32);
  const int peerRank = (globalRank == 0) ? 1 : 0;
  const uint8_t testPattern = 0x4D;

  try {
    auto transport = createTransport();

    DeviceBuffer dataBuffer(nbytes);
    auto localDataBuf = transport->registerBuffer(dataBuffer.get(), nbytes);
    auto remoteDataBufs = transport->exchangeBuffer(localDataBuf);
    int peerIndex = (peerRank < globalRank) ? peerRank : (peerRank - 1);
    auto remoteDataBuf = remoteDataBufs[peerIndex];

    const std::size_t signalBufSize = sizeof(uint64_t);
    DeviceBuffer signalBuffer(signalBufSize);
    CUDACHECK_TEST(cudaMemset(signalBuffer.get(), 0, signalBufSize));
    auto localSignalBuf =
        transport->registerBuffer(signalBuffer.get(), signalBufSize);
    auto remoteSignalBufs = transport->exchangeBuffer(localSignalBuf);
    auto remoteSignalBuf = remoteSignalBufs[peerIndex];

    P2pIbgdaTransportDevice* peerTransportPtr =
        transport->getP2pTransportDevice(peerRank);

    if (globalRank == 0) {
      test::fillBufferWithPattern(
          localDataBuf.ptr, nbytes, testPattern, numBlocks, blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      // Each warp signals with 1, total = numWarps
      test::testPutAndSignalGroupMultiWarp(
          peerTransportPtr,
          localDataBuf,
          remoteDataBuf,
          nbytes,
          remoteSignalBuf,
          0,
          1,
          numBlocks,
          blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    } else {
      CUDACHECK_TEST(cudaMemset(localDataBuf.ptr, 0, nbytes));
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      // Wait for accumulated signal from all warps
      test::testWaitSignal(
          static_cast<uint64_t*>(signalBuffer.get()),
          0,
          numWarps, // expected signal: 16 warps × 1
          1,
          32);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      DeviceBuffer errorCountBuf(sizeof(int));
      auto* d_errorCount = static_cast<int*>(errorCountBuf.get());
      CUDACHECK_TEST(cudaMemset(d_errorCount, 0, sizeof(int)));

      test::verifyBufferPattern(
          localDataBuf.ptr,
          nbytes,
          testPattern,
          d_errorCount,
          numBlocks,
          blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      int h_errorCount = 0;
      CUDACHECK_TEST(cudaMemcpy(
          &h_errorCount, d_errorCount, sizeof(int), cudaMemcpyDeviceToHost));

      EXPECT_EQ(h_errorCount, 0)
          << "Rank " << globalRank << ": Found " << h_errorCount
          << " byte mismatches out of " << nbytes
          << " bytes (multi-warp put_group_global)";
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }

  XLOGF(INFO, "Rank {}: PutSignalGroupMultiWarp test completed", globalRank);
}

// =============================================================================
// Block-Scope Group Put/Signal Test
// =============================================================================

TEST_F(MultipeerIbgdaTransportTestFixture, PutSignalGroupBlock) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const std::size_t nbytes = 65003;
  const int numBlocks = 4;
  const int blockSize = 256;
  const int peerRank = (globalRank == 0) ? 1 : 0;
  const uint8_t testPattern = 0x62;

  try {
    auto transport = createTransport();

    DeviceBuffer dataBuffer(nbytes);
    auto localDataBuf = transport->registerBuffer(dataBuffer.get(), nbytes);
    auto remoteDataBufs = transport->exchangeBuffer(localDataBuf);
    int peerIndex = (peerRank < globalRank) ? peerRank : (peerRank - 1);
    auto remoteDataBuf = remoteDataBufs[peerIndex];

    const std::size_t signalBufSize = sizeof(uint64_t);
    DeviceBuffer signalBuffer(signalBufSize);
    CUDACHECK_TEST(cudaMemset(signalBuffer.get(), 0, signalBufSize));
    auto localSignalBuf =
        transport->registerBuffer(signalBuffer.get(), signalBufSize);
    auto remoteSignalBufs = transport->exchangeBuffer(localSignalBuf);
    auto remoteSignalBuf = remoteSignalBufs[peerIndex];

    P2pIbgdaTransportDevice* peerTransportPtr =
        transport->getP2pTransportDevice(peerRank);

    if (globalRank == 0) {
      test::fillBufferWithPattern(
          localDataBuf.ptr, nbytes, testPattern, numBlocks, blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      // Each block signals with 1, total = numBlocks
      test::testPutAndSignalGroupBlock(
          peerTransportPtr,
          localDataBuf,
          remoteDataBuf,
          nbytes,
          remoteSignalBuf,
          0,
          1,
          numBlocks,
          blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    } else {
      CUDACHECK_TEST(cudaMemset(localDataBuf.ptr, 0, nbytes));
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      // Wait for accumulated signal from all blocks
      test::testWaitSignal(
          static_cast<uint64_t*>(signalBuffer.get()),
          0,
          numBlocks, // expected: 4 blocks × 1
          1,
          32);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      DeviceBuffer errorCountBuf(sizeof(int));
      auto* d_errorCount = static_cast<int*>(errorCountBuf.get());
      CUDACHECK_TEST(cudaMemset(d_errorCount, 0, sizeof(int)));

      test::verifyBufferPattern(
          localDataBuf.ptr,
          nbytes,
          testPattern,
          d_errorCount,
          numBlocks,
          blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      int h_errorCount = 0;
      CUDACHECK_TEST(cudaMemcpy(
          &h_errorCount, d_errorCount, sizeof(int), cudaMemcpyDeviceToHost));

      EXPECT_EQ(h_errorCount, 0)
          << "Rank " << globalRank << ": Found " << h_errorCount
          << " byte mismatches out of " << nbytes
          << " bytes (block-scope put_group_global)";
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }

  XLOGF(INFO, "Rank {}: PutSignalGroupBlock test completed", globalRank);
}

// =============================================================================
// Multiple Transfer Sizes Test
// =============================================================================

struct TransferSizeParams {
  std::size_t nbytes;
  std::string name;
};

class TransferSizeTestFixture
    : public MpiBaseTestFixture,
      public ::testing::WithParamInterface<TransferSizeParams> {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }
};

TEST_P(TransferSizeTestFixture, PutSignal) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const auto& params = GetParam();
  const std::size_t nbytes = params.nbytes;
  const int numBlocks = 4;
  const int blockSize = 128;
  const int peerRank = (globalRank == 0) ? 1 : 0;
  const uint8_t testPattern = static_cast<uint8_t>(globalRank + 0x10);

  XLOGF(
      INFO,
      "Rank {}: Running transfer size test {} with {} bytes",
      globalRank,
      params.name,
      nbytes);

  try {
    MultipeerIbgdaTransportConfig config{
        .cudaDevice = localRank,
    };

    auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
    MultipeerIbgdaTransport transport(globalRank, numRanks, bootstrap, config);
    transport.exchange();

    DeviceBuffer dataBuffer(nbytes);
    auto localDataBuf = transport.registerBuffer(dataBuffer.get(), nbytes);
    auto remoteDataBufs = transport.exchangeBuffer(localDataBuf);
    int peerIndex = (peerRank < globalRank) ? peerRank : (peerRank - 1);
    auto remoteDataBuf = remoteDataBufs[peerIndex];

    const std::size_t signalBufSize = sizeof(uint64_t);
    DeviceBuffer signalBuffer(signalBufSize);
    CUDACHECK_TEST(cudaMemset(signalBuffer.get(), 0, signalBufSize));
    auto localSignalBuf =
        transport.registerBuffer(signalBuffer.get(), signalBufSize);
    auto remoteSignalBufs = transport.exchangeBuffer(localSignalBuf);
    auto remoteSignalBuf = remoteSignalBufs[peerIndex];

    P2pIbgdaTransportDevice* peerTransportPtr =
        transport.getP2pTransportDevice(peerRank);

    if (globalRank == 0) {
      test::fillBufferWithPattern(
          localDataBuf.ptr, nbytes, testPattern, numBlocks, blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      test::testPutAndSignal(
          peerTransportPtr,
          localDataBuf,
          remoteDataBuf,
          nbytes,
          remoteSignalBuf,
          0,
          1,
          numBlocks,
          blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    } else {
      CUDACHECK_TEST(cudaMemset(localDataBuf.ptr, 0, nbytes));
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      test::testWaitSignal(
          static_cast<uint64_t*>(signalBuffer.get()),
          0,
          1,
          numBlocks,
          blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      DeviceBuffer errorCountBuf(sizeof(int));
      auto* d_errorCount = static_cast<int*>(errorCountBuf.get());
      CUDACHECK_TEST(cudaMemset(d_errorCount, 0, sizeof(int)));

      // Sender's pattern (rank 0 = 0x10)
      test::verifyBufferPattern(
          localDataBuf.ptr, nbytes, 0x10, d_errorCount, numBlocks, blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      int h_errorCount = 0;
      CUDACHECK_TEST(cudaMemcpy(
          &h_errorCount, d_errorCount, sizeof(int), cudaMemcpyDeviceToHost));

      EXPECT_EQ(h_errorCount, 0)
          << "Test " << params.name << ": Found " << h_errorCount
          << " byte mismatches out of " << nbytes << " bytes";
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }

  XLOGF(
      INFO,
      "Rank {}: Transfer size test {} completed",
      globalRank,
      params.name);
}

std::string transferSizeParamName(
    const ::testing::TestParamInfo<TransferSizeParams>& info) {
  return info.param.name;
}

INSTANTIATE_TEST_SUITE_P(
    TransferSizeVariations,
    TransferSizeTestFixture,
    ::testing::Values(
        TransferSizeParams{.nbytes = 1024, .name = "Size_1KB"},
        TransferSizeParams{.nbytes = 4 * 1024, .name = "Size_4KB"},
        TransferSizeParams{.nbytes = 64 * 1024, .name = "Size_64KB"},
        TransferSizeParams{.nbytes = 256 * 1024, .name = "Size_256KB"},
        TransferSizeParams{.nbytes = 1024 * 1024, .name = "Size_1MB"},
        TransferSizeParams{.nbytes = 4 * 1024 * 1024, .name = "Size_4MB"},
        TransferSizeParams{.nbytes = 16 * 1024 * 1024, .name = "Size_16MB"}),
    transferSizeParamName);

// =============================================================================
// Bidirectional Transfer Test
// =============================================================================

TEST_F(MultipeerIbgdaTransportTestFixture, Bidirectional) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const std::size_t nbytes = 256 * 1024;
  const int numBlocks = 2;
  const int blockSize = 64;
  const int peerRank = (globalRank == 0) ? 1 : 0;
  const uint8_t rank0Pattern = 0x20;
  const uint8_t rank1Pattern = 0x21;

  try {
    auto transport = createTransport();

    DeviceBuffer dataBuffer(nbytes);
    auto localDataBuf = transport->registerBuffer(dataBuffer.get(), nbytes);
    auto remoteDataBufs = transport->exchangeBuffer(localDataBuf);
    int peerIndex = (peerRank < globalRank) ? peerRank : (peerRank - 1);
    auto remoteDataBuf = remoteDataBufs[peerIndex];

    const std::size_t signalBufSize = sizeof(uint64_t);
    DeviceBuffer signalBuffer(signalBufSize);
    CUDACHECK_TEST(cudaMemset(signalBuffer.get(), 0, signalBufSize));
    auto localSignalBuf =
        transport->registerBuffer(signalBuffer.get(), signalBufSize);
    auto remoteSignalBufs = transport->exchangeBuffer(localSignalBuf);
    auto remoteSignalBuf = remoteSignalBufs[peerIndex];

    P2pIbgdaTransportDevice* peerTransportPtr =
        transport->getP2pTransportDevice(peerRank);

    DeviceBuffer errorCountBuf(sizeof(int));
    auto* d_errorCount = static_cast<int*>(errorCountBuf.get());

    // Phase 1: Rank 0 sends to Rank 1
    if (globalRank == 0) {
      test::fillBufferWithPattern(
          localDataBuf.ptr, nbytes, rank0Pattern, numBlocks, blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());
    } else {
      CUDACHECK_TEST(cudaMemset(localDataBuf.ptr, 0, nbytes));
      CUDACHECK_TEST(cudaDeviceSynchronize());
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (globalRank == 0) {
      test::testPutAndSignal(
          peerTransportPtr,
          localDataBuf,
          remoteDataBuf,
          nbytes,
          remoteSignalBuf,
          0,
          1,
          numBlocks,
          blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());
    } else {
      test::testWaitSignal(
          static_cast<uint64_t*>(signalBuffer.get()),
          0,
          1,
          numBlocks,
          blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      CUDACHECK_TEST(cudaMemset(d_errorCount, 0, sizeof(int)));
      test::verifyBufferPattern(
          localDataBuf.ptr,
          nbytes,
          rank0Pattern,
          d_errorCount,
          numBlocks,
          blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      int h_errorCount = 0;
      CUDACHECK_TEST(cudaMemcpy(
          &h_errorCount, d_errorCount, sizeof(int), cudaMemcpyDeviceToHost));
      EXPECT_EQ(h_errorCount, 0)
          << "Rank 1: Phase 1 verification found " << h_errorCount
          << " byte mismatches receiving from Rank 0";
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Phase 2: Rank 1 sends to Rank 0
    // Reset signal buffer for phase 2 (signal value was 1, now we wait for 2)
    if (globalRank == 1) {
      test::fillBufferWithPattern(
          localDataBuf.ptr, nbytes, rank1Pattern, numBlocks, blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());
    } else {
      CUDACHECK_TEST(cudaMemset(localDataBuf.ptr, 0, nbytes));
      CUDACHECK_TEST(cudaDeviceSynchronize());
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (globalRank == 1) {
      test::testPutAndSignal(
          peerTransportPtr,
          localDataBuf,
          remoteDataBuf,
          nbytes,
          remoteSignalBuf,
          0,
          1, // cumulative: now signal = 2
          numBlocks,
          blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());
    } else {
      test::testWaitSignal(
          static_cast<uint64_t*>(signalBuffer.get()),
          0,
          2, // wait for cumulative signal >= 2
          numBlocks,
          blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      CUDACHECK_TEST(cudaMemset(d_errorCount, 0, sizeof(int)));
      test::verifyBufferPattern(
          localDataBuf.ptr,
          nbytes,
          rank1Pattern,
          d_errorCount,
          numBlocks,
          blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      int h_errorCount = 0;
      CUDACHECK_TEST(cudaMemcpy(
          &h_errorCount, d_errorCount, sizeof(int), cudaMemcpyDeviceToHost));
      EXPECT_EQ(h_errorCount, 0)
          << "Rank 0: Phase 2 verification found " << h_errorCount
          << " byte mismatches receiving from Rank 1";
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }

  XLOGF(INFO, "Rank {}: Bidirectional test completed", globalRank);
}

// =============================================================================
// Stress Test - Multiple iterations
// =============================================================================

TEST_F(MultipeerIbgdaTransportTestFixture, StressTest) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const std::size_t nbytes = 128 * 1024;
  const int numIterations = 100;
  const int numBlocks = 2;
  const int blockSize = 64;
  const int peerRank = (globalRank == 0) ? 1 : 0;

  try {
    auto transport = createTransport();

    DeviceBuffer dataBuffer(nbytes);
    auto localDataBuf = transport->registerBuffer(dataBuffer.get(), nbytes);
    auto remoteDataBufs = transport->exchangeBuffer(localDataBuf);
    int peerIndex = (peerRank < globalRank) ? peerRank : (peerRank - 1);
    auto remoteDataBuf = remoteDataBufs[peerIndex];

    const std::size_t signalBufSize = sizeof(uint64_t);
    DeviceBuffer signalBuffer(signalBufSize);
    CUDACHECK_TEST(cudaMemset(signalBuffer.get(), 0, signalBufSize));
    auto localSignalBuf =
        transport->registerBuffer(signalBuffer.get(), signalBufSize);
    auto remoteSignalBufs = transport->exchangeBuffer(localSignalBuf);
    auto remoteSignalBuf = remoteSignalBufs[peerIndex];

    P2pIbgdaTransportDevice* peerTransportPtr =
        transport->getP2pTransportDevice(peerRank);

    int totalErrors = 0;

    for (int iter = 0; iter < numIterations; iter++) {
      const uint8_t testPattern = static_cast<uint8_t>(iter % 256);

      if (globalRank == 0) {
        test::fillBufferWithPattern(
            localDataBuf.ptr, nbytes, testPattern, numBlocks, blockSize);
        CUDACHECK_TEST(cudaDeviceSynchronize());

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        // Signal value = iter+1 (cumulative via atomic fetch-add)
        test::testPutAndSignal(
            peerTransportPtr,
            localDataBuf,
            remoteDataBuf,
            nbytes,
            remoteSignalBuf,
            0,
            iter + 1,
            numBlocks,
            blockSize);
        CUDACHECK_TEST(cudaDeviceSynchronize());

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
      } else {
        CUDACHECK_TEST(cudaMemset(localDataBuf.ptr, 0xFF, nbytes));
        CUDACHECK_TEST(cudaDeviceSynchronize());

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        // Wait for cumulative signal >= iter+1
        test::testWaitSignal(
            static_cast<uint64_t*>(signalBuffer.get()),
            0,
            iter + 1,
            numBlocks,
            blockSize);
        CUDACHECK_TEST(cudaDeviceSynchronize());

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        DeviceBuffer errorCountBuf(sizeof(int));
        auto* d_errorCount = static_cast<int*>(errorCountBuf.get());
        CUDACHECK_TEST(cudaMemset(d_errorCount, 0, sizeof(int)));

        test::verifyBufferPattern(
            localDataBuf.ptr,
            nbytes,
            testPattern,
            d_errorCount,
            numBlocks,
            blockSize);
        CUDACHECK_TEST(cudaDeviceSynchronize());

        int h_errorCount = 0;
        CUDACHECK_TEST(cudaMemcpy(
            &h_errorCount, d_errorCount, sizeof(int), cudaMemcpyDeviceToHost));

        if (h_errorCount > 0) {
          totalErrors += h_errorCount;
          XLOGF(ERR, "Iteration {}: Found {} errors", iter, h_errorCount);
        }
      }
    }

    if (globalRank == 1) {
      EXPECT_EQ(totalErrors, 0)
          << "Stress test found " << totalErrors << " total byte errors "
          << "across " << numIterations << " iterations";
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }

  XLOGF(
      INFO,
      "Rank {}: Stress test completed ({} iterations)",
      globalRank,
      numIterations);
}

// =============================================================================
// Signal Only Test - Tests sending signals without data
// =============================================================================

TEST_F(MultipeerIbgdaTransportTestFixture, SignalOnly) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const int numBlocks = 1;
  const int blockSize = 32;

  try {
    auto transport = createTransport();

    int peerRank = (globalRank == 0) ? 1 : 0;
    int peerIndex = (peerRank < globalRank) ? peerRank : (peerRank - 1);

    // Allocate and register signal buffer
    const std::size_t signalBufSize = sizeof(uint64_t);
    DeviceBuffer signalBuffer(signalBufSize);
    CUDACHECK_TEST(cudaMemset(signalBuffer.get(), 0, signalBufSize));
    auto localSignalBuf =
        transport->registerBuffer(signalBuffer.get(), signalBufSize);
    auto remoteSignalBufs = transport->exchangeBuffer(localSignalBuf);
    auto remoteSignalBuf = remoteSignalBufs[peerIndex];

    P2pIbgdaTransportDevice* peerTransportPtr =
        transport->getP2pTransportDevice(peerRank);

    if (globalRank == 0) {
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      test::testSignalOnly(
          peerTransportPtr, remoteSignalBuf, 0, 42, numBlocks, blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    } else {
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      test::testWaitSignal(
          static_cast<uint64_t*>(signalBuffer.get()),
          0,
          42,
          numBlocks,
          blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      // Verify signal value via host-side read
      uint64_t h_result = 0;
      CUDACHECK_TEST(cudaMemcpy(
          &h_result,
          signalBuffer.get(),
          sizeof(uint64_t),
          cudaMemcpyDeviceToHost));

      EXPECT_GE(h_result, 42u) << "Signal value should be >= 42";
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }

  XLOGF(INFO, "Rank {}: SignalOnly test completed", globalRank);
}

// =============================================================================
// Put + Signal + Counter Test - Tests companion QP counter-based completion
// =============================================================================

TEST_F(MultipeerIbgdaTransportTestFixture, PutSignalCounter) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const std::size_t nbytes = 64 * 1024;
  const int numBlocks = 1;
  const int blockSize = 32;

  try {
    auto transport = createTransport();

    int peerRank = (globalRank == 0) ? 1 : 0;
    int peerIndex = (peerRank < globalRank) ? peerRank : (peerRank - 1);

    // Data buffer
    DeviceBuffer dataBuffer(nbytes);
    auto localDataBuf = transport->registerBuffer(dataBuffer.get(), nbytes);
    auto remoteDataBufs = transport->exchangeBuffer(localDataBuf);
    auto remoteDataBuf = remoteDataBufs[peerIndex];

    // Signal buffer (remote, exchanged)
    DeviceBuffer signalBuffer(sizeof(uint64_t));
    CUDACHECK_TEST(cudaMemset(signalBuffer.get(), 0, sizeof(uint64_t)));
    auto localSignalBuf =
        transport->registerBuffer(signalBuffer.get(), sizeof(uint64_t));
    auto remoteSignalBufs = transport->exchangeBuffer(localSignalBuf);
    auto remoteSignalBuf = remoteSignalBufs[peerIndex];

    // Counter buffer (local only, no exchange — companion QP writes to self)
    DeviceBuffer counterBuffer(sizeof(uint64_t));
    CUDACHECK_TEST(cudaMemset(counterBuffer.get(), 0, sizeof(uint64_t)));
    auto localCounterBuf =
        transport->registerBuffer(counterBuffer.get(), sizeof(uint64_t));

    P2pIbgdaTransportDevice* peerTransportPtr =
        transport->getP2pTransportDevice(peerRank);

    if (globalRank == 0) {
      // Sender: fill buffer with pattern, barrier, put+signal+counter
      test::fillBufferWithPattern(
          dataBuffer.get(),
          nbytes,
          static_cast<uint8_t>(0xAB),
          numBlocks,
          blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      test::testPutSignalCounter(
          peerTransportPtr,
          localDataBuf,
          remoteDataBuf,
          nbytes,
          remoteSignalBuf,
          0,
          1, // signalId=0, signalVal=1
          localCounterBuf,
          0,
          1, // counterId=0, counterVal=1
          numBlocks,
          blockSize);

      // Wait for local counter to confirm NIC completion
      test::testWaitCounter(
          static_cast<uint64_t*>(counterBuffer.get()),
          0,
          1, // counterId=0, expectedVal=1
          numBlocks,
          blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      // Verify counter value on host
      uint64_t h_counter = 0;
      CUDACHECK_TEST(cudaMemcpy(
          &h_counter,
          counterBuffer.get(),
          sizeof(uint64_t),
          cudaMemcpyDeviceToHost));
      EXPECT_GE(h_counter, 1u)
          << "Counter should be >= 1 after companion QP loopback";

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    } else {
      CUDACHECK_TEST(cudaMemset(dataBuffer.get(), 0, nbytes));
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      // Receiver: wait for signal, verify data
      test::testWaitSignal(
          static_cast<uint64_t*>(signalBuffer.get()),
          0,
          1,
          numBlocks,
          blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      // Verify data arrived correctly
      int errorCount = 0;
      test::verifyBufferPattern(
          dataBuffer.get(),
          nbytes,
          static_cast<uint8_t>(0xAB),
          &errorCount,
          numBlocks,
          blockSize);
      EXPECT_EQ(errorCount, 0) << "PutSignalCounter: data corruption detected";
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }
}

// =============================================================================
// Reset Signal Test - Tests resetting signals for reuse
// =============================================================================

TEST_F(MultipeerIbgdaTransportTestFixture, ResetSignal) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const int numBlocks = 1;
  const int blockSize = 32;
  const int numIterations = 3;

  try {
    auto transport = createTransport();

    int peerRank = (globalRank == 0) ? 1 : 0;
    int peerIndex = (peerRank < globalRank) ? peerRank : (peerRank - 1);

    const std::size_t signalBufSize = sizeof(uint64_t);
    DeviceBuffer signalBuffer(signalBufSize);
    CUDACHECK_TEST(cudaMemset(signalBuffer.get(), 0, signalBufSize));
    auto localSignalBuf =
        transport->registerBuffer(signalBuffer.get(), signalBufSize);
    auto remoteSignalBufs = transport->exchangeBuffer(localSignalBuf);
    auto remoteSignalBuf = remoteSignalBufs[peerIndex];

    P2pIbgdaTransportDevice* peerTransportPtr =
        transport->getP2pTransportDevice(peerRank);

    for (int iter = 0; iter < numIterations; iter++) {
      if (globalRank == 0) {
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        // Send signal
        test::testSignalOnly(
            peerTransportPtr, remoteSignalBuf, 0, 1, numBlocks, blockSize);
        CUDACHECK_TEST(cudaDeviceSynchronize());

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
      } else {
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        // Wait for signal (always expecting 1 since we reset each iteration)
        test::testWaitSignal(
            static_cast<uint64_t*>(signalBuffer.get()),
            0,
            1,
            numBlocks,
            blockSize);
        CUDACHECK_TEST(cudaDeviceSynchronize());

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        // Reset own signal buffer for next iteration
        CUDACHECK_TEST(cudaMemset(signalBuffer.get(), 0, signalBufSize));
        CUDACHECK_TEST(cudaDeviceSynchronize());

        // Verify signal was reset to 0
        uint64_t h_result = 1; // Initialize to non-zero
        CUDACHECK_TEST(cudaMemcpy(
            &h_result,
            signalBuffer.get(),
            sizeof(uint64_t),
            cudaMemcpyDeviceToHost));

        EXPECT_EQ(h_result, 0u)
            << "Iteration " << iter << ": Signal should be reset to 0";

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
      }
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }

  XLOGF(
      INFO,
      "Rank {}: ResetSignal test completed ({} iterations)",
      globalRank,
      numIterations);
}

// =============================================================================
// Multiple Signal Slots Test - Tests using multiple signal IDs
// =============================================================================

TEST_F(MultipeerIbgdaTransportTestFixture, MultipleSignalSlots) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const int numSignals = 4;
  const int numBlocks = 1;
  const int blockSize = 32;

  try {
    auto transport = createTransport();

    int peerRank = (globalRank == 0) ? 1 : 0;
    int peerIndex = (peerRank < globalRank) ? peerRank : (peerRank - 1);

    // Allocate signal buffer with multiple slots
    const std::size_t signalBufSize = numSignals * sizeof(uint64_t);
    DeviceBuffer signalBuffer(signalBufSize);
    CUDACHECK_TEST(cudaMemset(signalBuffer.get(), 0, signalBufSize));
    auto localSignalBuf =
        transport->registerBuffer(signalBuffer.get(), signalBufSize);
    auto remoteSignalBufs = transport->exchangeBuffer(localSignalBuf);
    auto remoteSignalBuf = remoteSignalBufs[peerIndex];

    P2pIbgdaTransportDevice* peerTransportPtr =
        transport->getP2pTransportDevice(peerRank);

    if (globalRank == 0) {
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      for (int i = 0; i < numSignals; i++) {
        test::testSignalOnly(
            peerTransportPtr,
            remoteSignalBuf,
            i,
            static_cast<uint64_t>(i + 1) * 10,
            numBlocks,
            blockSize);
      }
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    } else {
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      for (int i = 0; i < numSignals; i++) {
        test::testWaitSignal(
            static_cast<uint64_t*>(signalBuffer.get()),
            i,
            static_cast<uint64_t>(i + 1) * 10,
            numBlocks,
            blockSize);
      }
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      // Verify each signal slot via host-side read
      std::vector<uint64_t> h_signals(numSignals);
      CUDACHECK_TEST(cudaMemcpy(
          h_signals.data(),
          signalBuffer.get(),
          signalBufSize,
          cudaMemcpyDeviceToHost));

      for (int i = 0; i < numSignals; i++) {
        uint64_t expected = static_cast<uint64_t>(i + 1) * 10;
        EXPECT_GE(h_signals[i], expected)
            << "Signal slot " << i << ": expected >= " << expected << ", got "
            << h_signals[i];
      }
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }

  XLOGF(INFO, "Rank {}: MultipleSignalSlots test completed", globalRank);
}

// =============================================================================
// Put Signal Wait For Ready Test
// =============================================================================

TEST_F(MultipeerIbgdaTransportTestFixture, PutSignalWaitForReady) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const std::size_t nbytes = 64 * 1024;
  const int numBlocks = 1;
  const int blockSize = 32;
  const int peerRank = (globalRank == 0) ? 1 : 0;
  const uint8_t testPattern = 0x55;

  try {
    auto transport = createTransport();

    DeviceBuffer dataBuffer(nbytes);
    auto localDataBuf = transport->registerBuffer(dataBuffer.get(), nbytes);
    auto remoteDataBufs = transport->exchangeBuffer(localDataBuf);
    int peerIndex = (peerRank < globalRank) ? peerRank : (peerRank - 1);
    auto remoteDataBuf = remoteDataBufs[peerIndex];

    // 2 signal slots: slot 0 = ready, slot 1 = data
    const std::size_t signalBufSize = 2 * sizeof(uint64_t);
    DeviceBuffer signalBuffer(signalBufSize);
    CUDACHECK_TEST(cudaMemset(signalBuffer.get(), 0, signalBufSize));
    auto localSignalBuf =
        transport->registerBuffer(signalBuffer.get(), signalBufSize);
    auto remoteSignalBufs = transport->exchangeBuffer(localSignalBuf);
    auto remoteSignalBuf = remoteSignalBufs[peerIndex];

    P2pIbgdaTransportDevice* peerTransportPtr =
        transport->getP2pTransportDevice(peerRank);

    if (globalRank == 0) {
      // Rank 0: Sender
      test::fillBufferWithPattern(
          localDataBuf.ptr, nbytes, testPattern, numBlocks, blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      // Wait for receiver's "ready" signal on local slot 0,
      // then put data + signal completion on remote slot 1
      test::testWaitReadyThenPutAndSignal(
          peerTransportPtr,
          localDataBuf,
          remoteDataBuf,
          nbytes,
          static_cast<uint64_t*>(signalBuffer.get()),
          0,
          1, // readySignalId, readySignalVal
          remoteSignalBuf,
          1,
          1, // dataSignalId, dataSignalVal
          numBlocks,
          blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    } else {
      // Rank 1: Receiver
      CUDACHECK_TEST(cudaMemset(localDataBuf.ptr, 0, nbytes));
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      // Send "ready" signal to sender (remote slot 0)
      test::testSignalOnly(
          peerTransportPtr, remoteSignalBuf, 0, 1, numBlocks, blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      // Wait for "data" signal from sender (local slot 1)
      test::testWaitSignal(
          static_cast<uint64_t*>(signalBuffer.get()),
          1,
          1,
          numBlocks,
          blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      DeviceBuffer errorCountBuf(sizeof(int));
      auto* d_errorCount = static_cast<int*>(errorCountBuf.get());
      CUDACHECK_TEST(cudaMemset(d_errorCount, 0, sizeof(int)));

      test::verifyBufferPattern(
          localDataBuf.ptr,
          nbytes,
          testPattern,
          d_errorCount,
          numBlocks,
          blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      int h_errorCount = 0;
      CUDACHECK_TEST(cudaMemcpy(
          &h_errorCount, d_errorCount, sizeof(int), cudaMemcpyDeviceToHost));

      EXPECT_EQ(h_errorCount, 0)
          << "Rank " << globalRank << ": Found " << h_errorCount
          << " byte mismatches out of " << nbytes << " bytes";
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }

  XLOGF(INFO, "Rank {}: PutSignalWaitForReady test completed", globalRank);
}

// =============================================================================
// Bidirectional Concurrent Test
// =============================================================================

TEST_F(MultipeerIbgdaTransportTestFixture, BidirectionalConcurrent) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const std::size_t nbytes = 64 * 1024;
  const int numBlocks = 1;
  const int blockSize = 2;
  const int peerRank = (globalRank == 0) ? 1 : 0;
  const uint8_t rank0Pattern = 0xAA;
  const uint8_t rank1Pattern = 0xBB;
  const uint8_t myPattern = (globalRank == 0) ? rank0Pattern : rank1Pattern;
  const uint8_t peerPattern = (globalRank == 0) ? rank1Pattern : rank0Pattern;

  try {
    auto transport = createTransport();

    int peerIndex = (peerRank < globalRank) ? peerRank : (peerRank - 1);

    DeviceBuffer sendBuffer(nbytes);
    DeviceBuffer recvBuffer(nbytes);

    auto localSendBuf = transport->registerBuffer(sendBuffer.get(), nbytes);
    auto localRecvBuf = transport->registerBuffer(recvBuffer.get(), nbytes);
    auto remoteRecvBufs = transport->exchangeBuffer(localRecvBuf);
    auto peerRecvBuf = remoteRecvBufs[peerIndex];

    // 2 signal slots
    const std::size_t signalBufSize = 2 * sizeof(uint64_t);
    DeviceBuffer signalBuffer(signalBufSize);
    CUDACHECK_TEST(cudaMemset(signalBuffer.get(), 0, signalBufSize));
    auto localSignalBuf =
        transport->registerBuffer(signalBuffer.get(), signalBufSize);
    auto remoteSignalBufs = transport->exchangeBuffer(localSignalBuf);
    auto remoteSignalBuf = remoteSignalBufs[peerIndex];

    P2pIbgdaTransportDevice* peerTransportPtr =
        transport->getP2pTransportDevice(peerRank);

    test::fillBufferWithPattern(
        localSendBuf.ptr, nbytes, myPattern, numBlocks, 32);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    CUDACHECK_TEST(cudaMemset(localRecvBuf.ptr, 0, nbytes));
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Rank 0 sends on signal 0, receives on signal 1
    // Rank 1 sends on signal 1, receives on signal 0
    int sendSignalId = globalRank;
    int recvSignalId = peerRank;

    test::testBidirectionalPutAndWait(
        peerTransportPtr,
        localSendBuf,
        peerRecvBuf,
        nbytes,
        remoteSignalBuf,
        sendSignalId,
        1,
        static_cast<uint64_t*>(signalBuffer.get()),
        recvSignalId,
        1,
        numBlocks,
        blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    DeviceBuffer errorCountBuf(sizeof(int));
    auto* d_errorCount = static_cast<int*>(errorCountBuf.get());
    CUDACHECK_TEST(cudaMemset(d_errorCount, 0, sizeof(int)));

    test::verifyBufferPattern(
        localRecvBuf.ptr, nbytes, peerPattern, d_errorCount, numBlocks, 32);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    int h_errorCount = 0;
    CUDACHECK_TEST(cudaMemcpy(
        &h_errorCount, d_errorCount, sizeof(int), cudaMemcpyDeviceToHost));

    EXPECT_EQ(h_errorCount, 0)
        << "Rank " << globalRank << ": Found " << h_errorCount
        << " byte mismatches receiving from rank " << peerRank;

  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }

  XLOGF(INFO, "Rank {}: BidirectionalConcurrent test completed", globalRank);
}

// =============================================================================
// AlltoAll Test - Uses partition API for parallel peer comm
// =============================================================================

TEST_F(MultipeerIbgdaTransportTestFixture, AllToAll) {
  if (numRanks < 2) {
    XLOGF(
        WARNING, "Skipping test: requires at least 2 ranks, got {}", numRanks);
    return;
  }

  const int numPeers = numRanks - 1;
  const std::size_t bytesPerPeer = 64 * 1024;
  const std::size_t totalBytes = bytesPerPeer * numPeers;
  const int numBlocks = numPeers;
  const int blockSize = 32;

  try {
    auto transport = createTransport();

    DeviceBuffer sendBuffer(totalBytes);
    DeviceBuffer recvBuffer(totalBytes);

    auto localSendBuf = transport->registerBuffer(sendBuffer.get(), totalBytes);
    auto localRecvBuf = transport->registerBuffer(recvBuffer.get(), totalBytes);
    auto remoteRecvBufs = transport->exchangeBuffer(localRecvBuf);

    // Allocate signal buffer: numRanks slots (peerRank as signal ID)
    const std::size_t signalBufSize = numRanks * sizeof(uint64_t);
    DeviceBuffer signalBuffer(signalBufSize);
    CUDACHECK_TEST(cudaMemset(signalBuffer.get(), 0, signalBufSize));
    auto localSignalBuf =
        transport->registerBuffer(signalBuffer.get(), signalBufSize);
    auto remoteSignalBufs = transport->exchangeBuffer(localSignalBuf);

    // Build per-peer arrays
    std::vector<IbgdaLocalBuffer> localSendBufsPerPeer(numPeers);
    std::vector<IbgdaRemoteBuffer> peerRecvBufs(numPeers);
    std::vector<IbgdaRemoteBuffer> peerRemoteSignalBufs(numPeers);
    std::vector<P2pIbgdaTransportDevice*> peerTransports(numPeers);
    std::vector<int> peerRanksVec(numPeers);

    for (int peerIndex = 0; peerIndex < numPeers; peerIndex++) {
      int peerRank = (peerIndex < globalRank) ? peerIndex : (peerIndex + 1);
      peerRanksVec[peerIndex] = peerRank;

      localSendBufsPerPeer[peerIndex] =
          localSendBuf.subBuffer(peerIndex * bytesPerPeer);

      int ourIndexOnPeer =
          (globalRank < peerRank) ? globalRank : (globalRank - 1);
      peerRecvBufs[peerIndex] =
          remoteRecvBufs[peerIndex].subBuffer(ourIndexOnPeer * bytesPerPeer);

      peerRemoteSignalBufs[peerIndex] = remoteSignalBufs[peerIndex];
      peerTransports[peerIndex] = transport->getP2pTransportDevice(peerRank);
    }

    const uint8_t myPattern = static_cast<uint8_t>(0x30 + globalRank);
    test::fillBufferWithPattern(
        localSendBuf.ptr, totalBytes, myPattern, 4, 128);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    CUDACHECK_TEST(cudaMemset(localRecvBuf.ptr, 0, totalBytes));
    CUDACHECK_TEST(cudaDeviceSynchronize());

    // Copy arrays to device memory
    DeviceBuffer d_peerTransports(numPeers * sizeof(P2pIbgdaTransportDevice*));
    DeviceBuffer d_localSendBufs(numPeers * sizeof(IbgdaLocalBuffer));
    DeviceBuffer d_peerRecvBufs(numPeers * sizeof(IbgdaRemoteBuffer));
    DeviceBuffer d_remoteSignalBufs(numPeers * sizeof(IbgdaRemoteBuffer));
    DeviceBuffer d_peerRanks(numPeers * sizeof(int));

    CUDACHECK_TEST(cudaMemcpy(
        d_peerTransports.get(),
        peerTransports.data(),
        numPeers * sizeof(P2pIbgdaTransportDevice*),
        cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemcpy(
        d_localSendBufs.get(),
        localSendBufsPerPeer.data(),
        numPeers * sizeof(IbgdaLocalBuffer),
        cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemcpy(
        d_peerRecvBufs.get(),
        peerRecvBufs.data(),
        numPeers * sizeof(IbgdaRemoteBuffer),
        cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemcpy(
        d_remoteSignalBufs.get(),
        peerRemoteSignalBufs.data(),
        numPeers * sizeof(IbgdaRemoteBuffer),
        cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemcpy(
        d_peerRanks.get(),
        peerRanksVec.data(),
        numPeers * sizeof(int),
        cudaMemcpyHostToDevice));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    test::testAllToAll(
        static_cast<P2pIbgdaTransportDevice**>(d_peerTransports.get()),
        static_cast<IbgdaLocalBuffer*>(d_localSendBufs.get()),
        static_cast<IbgdaRemoteBuffer*>(d_peerRecvBufs.get()),
        static_cast<IbgdaRemoteBuffer*>(d_remoteSignalBufs.get()),
        globalRank,
        bytesPerPeer,
        numPeers,
        numBlocks,
        blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Verify received data from each peer
    int totalErrors = 0;
    for (int peerIndex = 0; peerIndex < numPeers; peerIndex++) {
      int peerRank = (peerIndex < globalRank) ? peerIndex : (peerIndex + 1);
      uint8_t expectedPattern = static_cast<uint8_t>(0x30 + peerRank);

      DeviceBuffer errorCountBuf(sizeof(int));
      auto* d_errorCount = static_cast<int*>(errorCountBuf.get());
      CUDACHECK_TEST(cudaMemset(d_errorCount, 0, sizeof(int)));

      void* recvChunk =
          static_cast<char*>(localRecvBuf.ptr) + peerIndex * bytesPerPeer;
      test::verifyBufferPattern(
          recvChunk, bytesPerPeer, expectedPattern, d_errorCount, 4, 128);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      int h_errorCount = 0;
      CUDACHECK_TEST(cudaMemcpy(
          &h_errorCount, d_errorCount, sizeof(int), cudaMemcpyDeviceToHost));

      if (h_errorCount > 0) {
        totalErrors += h_errorCount;
        XLOGF(
            ERR,
            "Rank {}: {} byte mismatches receiving from rank {}",
            globalRank,
            h_errorCount,
            peerRank);
      }
    }

    EXPECT_EQ(totalErrors, 0)
        << "Rank " << globalRank << ": Found " << totalErrors
        << " total byte mismatches across " << numPeers << " peers";

  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }

  XLOGF(
      INFO,
      "Rank {}: AllToAll test completed with {} peers",
      globalRank,
      numPeers);
}

} // namespace comms::pipes::tests

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  auto mpi_env = std::make_unique<MPIEnvironmentBase>();
  ::testing::AddGlobalTestEnvironment(mpi_env.get());
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
