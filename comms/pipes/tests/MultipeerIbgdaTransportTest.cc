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

using namespace meta::comms;

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
  std::unique_ptr<MultipeerIbgdaTransport> createTransport(
      std::size_t dataBufferSize,
      std::size_t signalCount = 1) {
    MultipeerIbgdaTransportConfig config{
        .cudaDevice = localRank,
        .dataBufferSize = dataBufferSize,
        .signalCount = signalCount,
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

  const std::size_t dataBufferSize = 1024 * 1024; // 1MB per peer

  try {
    auto transport = createTransport(dataBufferSize);

    EXPECT_EQ(transport->myRank(), globalRank);
    EXPECT_EQ(transport->nRanks(), numRanks);
    EXPECT_EQ(transport->numPeers(), numRanks - 1);
    EXPECT_NE(transport->getDeviceTransportPtr(), nullptr);

    XLOGF(
        INFO,
        "Rank {}: Transport created with NIC {} GID index {}",
        globalRank,
        transport->getNicDeviceName(),
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
    auto transport = createTransport(nbytes);

    // Get data buffers for the peer
    auto localDataBuf = transport->getDataBuffer(peerRank);
    auto remoteDataBuf = transport->getRemoteDataBuffer(peerRank);

    XLOGF(
        INFO,
        "Rank {}: localDataBuf ptr={} lkey={}, remoteDataBuf ptr={} rkey={}",
        globalRank,
        localDataBuf.ptr,
        localDataBuf.lkey,
        remoteDataBuf.ptr,
        remoteDataBuf.rkey);

    // Get per-peer transport from device transport array
    // For a 2-rank setup, peer index 0 = the other rank
    P2pIbgdaTransportDevice* deviceTransportPtr =
        transport->getDeviceTransportPtr();

    if (globalRank == 0) {
      // Rank 0: Sender
      // Fill local buffer with test pattern
      test::fillBufferWithPattern(
          localDataBuf.ptr, nbytes, testPattern, numBlocks, blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      // Perform RDMA put with signal
      test::testPutSignal(
          deviceTransportPtr,
          localDataBuf,
          remoteDataBuf,
          nbytes,
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
          deviceTransportPtr,
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
// Multiple Transfers Test - Tests repeated put_signal operations
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
        .dataBufferSize = nbytes,
        .signalCount = 1,
    };

    auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
    MultipeerIbgdaTransport transport(globalRank, numRanks, bootstrap, config);
    transport.exchange();

    auto localDataBuf = transport.getDataBuffer(peerRank);
    auto remoteDataBuf = transport.getRemoteDataBuffer(peerRank);

    P2pIbgdaTransportDevice* deviceTransportPtr =
        transport.getDeviceTransportPtr();

    if (globalRank == 0) {
      // Sender
      test::fillBufferWithPattern(
          localDataBuf.ptr, nbytes, testPattern, numBlocks, blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      test::testPutSignal(
          deviceTransportPtr,
          localDataBuf,
          remoteDataBuf,
          nbytes,
          1,
          numBlocks,
          blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    } else {
      // Receiver
      CUDACHECK_TEST(cudaMemset(localDataBuf.ptr, 0, nbytes));
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      test::testWaitSignal(deviceTransportPtr, 1, numBlocks, blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      // Verify all bytes
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

  const std::size_t nbytes = 256 * 1024; // 256KB
  const int numBlocks = 2;
  const int blockSize = 64;
  const int peerRank = (globalRank == 0) ? 1 : 0;
  const uint8_t myPattern = static_cast<uint8_t>(globalRank + 0x20);
  const uint8_t peerPattern = static_cast<uint8_t>(peerRank + 0x20);

  try {
    auto transport = createTransport(nbytes);

    auto localDataBuf = transport->getDataBuffer(peerRank);
    auto remoteDataBuf = transport->getRemoteDataBuffer(peerRank);

    P2pIbgdaTransportDevice* deviceTransportPtr =
        transport->getDeviceTransportPtr();

    // Fill local buffer with my pattern
    test::fillBufferWithPattern(
        localDataBuf.ptr, nbytes, myPattern, numBlocks, blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Both ranks send to each other, in order to avoid deadlock:
    // Rank 0: send then wait
    // Rank 1: wait then send
    if (globalRank == 0) {
      test::testPutSignal(
          deviceTransportPtr,
          localDataBuf,
          remoteDataBuf,
          nbytes,
          1,
          numBlocks,
          blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      test::testWaitSignal(deviceTransportPtr, 1, numBlocks, blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());
    } else {
      test::testWaitSignal(deviceTransportPtr, 1, numBlocks, blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      test::testPutSignal(
          deviceTransportPtr,
          localDataBuf,
          remoteDataBuf,
          nbytes,
          1,
          numBlocks,
          blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Both ranks verify received data from peer
    DeviceBuffer errorCountBuf(sizeof(int));
    auto* d_errorCount = static_cast<int*>(errorCountBuf.get());
    CUDACHECK_TEST(cudaMemset(d_errorCount, 0, sizeof(int)));

    test::verifyBufferPattern(
        localDataBuf.ptr,
        nbytes,
        peerPattern,
        d_errorCount,
        numBlocks,
        blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    int h_errorCount = 0;
    CUDACHECK_TEST(cudaMemcpy(
        &h_errorCount, d_errorCount, sizeof(int), cudaMemcpyDeviceToHost));

    EXPECT_EQ(h_errorCount, 0)
        << "Rank " << globalRank << ": Bidirectional test found "
        << h_errorCount << " byte mismatches";

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

  const std::size_t nbytes = 128 * 1024; // 128KB
  const int numIterations = 100;
  const int numBlocks = 2;
  const int blockSize = 64;
  const int peerRank = (globalRank == 0) ? 1 : 0;

  try {
    auto transport = createTransport(nbytes);

    auto localDataBuf = transport->getDataBuffer(peerRank);
    auto remoteDataBuf = transport->getRemoteDataBuffer(peerRank);

    P2pIbgdaTransportDevice* deviceTransportPtr =
        transport->getDeviceTransportPtr();

    int totalErrors = 0;

    for (int iter = 0; iter < numIterations; iter++) {
      const uint8_t testPattern = static_cast<uint8_t>(iter % 256);

      if (globalRank == 0) {
        // Sender
        test::fillBufferWithPattern(
            localDataBuf.ptr, nbytes, testPattern, numBlocks, blockSize);
        CUDACHECK_TEST(cudaDeviceSynchronize());

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        test::testPutSignal(
            deviceTransportPtr,
            localDataBuf,
            remoteDataBuf,
            nbytes,
            iter + 1,
            numBlocks,
            blockSize);
        CUDACHECK_TEST(cudaDeviceSynchronize());

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
      } else {
        // Receiver
        CUDACHECK_TEST(cudaMemset(localDataBuf.ptr, 0xFF, nbytes));
        CUDACHECK_TEST(cudaDeviceSynchronize());

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        test::testWaitSignal(
            deviceTransportPtr, iter + 1, numBlocks, blockSize);
        CUDACHECK_TEST(cudaDeviceSynchronize());

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        // Verify
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

} // namespace comms::pipes::tests

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  auto mpi_env = std::make_unique<MPIEnvironmentBase>();
  ::testing::AddGlobalTestEnvironment(mpi_env.get());
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
