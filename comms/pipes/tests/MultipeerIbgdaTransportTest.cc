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
  std::unique_ptr<MultipeerIbgdaTransport> createTransport(
      std::size_t signalCount = 1) {
    MultipeerIbgdaTransportConfig config{
        .cudaDevice = localRank,
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

  try {
    auto transport = createTransport();

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
    auto transport = createTransport();

    // Allocate and register user-owned data buffer
    DeviceBuffer dataBuffer(nbytes);
    auto localDataBuf = transport->registerBuffer(dataBuffer.get(), nbytes);

    // Collectively exchange buffer info to get remote buffer handles
    auto remoteDataBufs = transport->exchangeBuffer(localDataBuf);
    int peerIndex = (peerRank < globalRank) ? peerRank : (peerRank - 1);
    auto remoteDataBuf = remoteDataBufs[peerIndex];

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
      test::testPutSignal(
          peerTransportPtr,
          localDataBuf,
          remoteDataBuf,
          nbytes,
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
          peerTransportPtr,
          0, // signal id
          IbgdaCmpOp::GE, // comparison operation
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
        .signalCount = 1,
    };

    auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
    MultipeerIbgdaTransport transport(globalRank, numRanks, bootstrap, config);
    transport.exchange();

    // Allocate and register user-owned data buffer
    DeviceBuffer dataBuffer(nbytes);
    auto localDataBuf = transport.registerBuffer(dataBuffer.get(), nbytes);

    // Collectively exchange buffer info to get remote buffer handles
    auto remoteDataBufs = transport.exchangeBuffer(localDataBuf);
    int peerIndex = (peerRank < globalRank) ? peerRank : (peerRank - 1);
    auto remoteDataBuf = remoteDataBufs[peerIndex];

    // Get peer transport for explicit peer selection
    P2pIbgdaTransportDevice* peerTransportPtr =
        transport.getP2pTransportDevice(peerRank);

    if (globalRank == 0) {
      // Sender
      test::fillBufferWithPattern(
          localDataBuf.ptr, nbytes, testPattern, numBlocks, blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      test::testPutSignal(
          peerTransportPtr,
          localDataBuf,
          remoteDataBuf,
          nbytes,
          0, // signal id
          1, // signal value
          numBlocks,
          blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    } else {
      // Receiver
      CUDACHECK_TEST(cudaMemset(localDataBuf.ptr, 0, nbytes));
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      test::testWaitSignal(
          peerTransportPtr,
          0, // signal id
          IbgdaCmpOp::GE, // comparison operation
          1, // expected signal
          numBlocks,
          blockSize);
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

  // For bidirectional transfer using a single shared buffer per peer,
  // we need to sequence the transfers carefully:
  // 1. Rank 0 sends first, Rank 1 receives and verifies
  // 2. Then Rank 1 sends, Rank 0 receives and verifies
  // This avoids the race condition where RDMA overwrites destroy data
  // before it can be sent back.

  const std::size_t nbytes = 256 * 1024; // 256KB
  const int numBlocks = 2;
  const int blockSize = 64;
  const int peerRank = (globalRank == 0) ? 1 : 0;
  const uint8_t rank0Pattern = 0x20;
  const uint8_t rank1Pattern = 0x21;

  try {
    auto transport = createTransport();

    // Allocate and register user-owned data buffer
    DeviceBuffer dataBuffer(nbytes);
    auto localDataBuf = transport->registerBuffer(dataBuffer.get(), nbytes);

    // Collectively exchange buffer info to get remote buffer handles
    auto remoteDataBufs = transport->exchangeBuffer(localDataBuf);
    int peerIndex = (peerRank < globalRank) ? peerRank : (peerRank - 1);
    auto remoteDataBuf = remoteDataBufs[peerIndex];

    // Get peer transport for explicit peer selection
    P2pIbgdaTransportDevice* peerTransportPtr =
        transport->getP2pTransportDevice(peerRank);

    DeviceBuffer errorCountBuf(sizeof(int));
    auto* d_errorCount = static_cast<int*>(errorCountBuf.get());

    // Phase 1: Rank 0 sends to Rank 1
    if (globalRank == 0) {
      // Fill local buffer with rank 0's pattern
      test::fillBufferWithPattern(
          localDataBuf.ptr, nbytes, rank0Pattern, numBlocks, blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());
    } else {
      // Clear buffer to receive
      CUDACHECK_TEST(cudaMemset(localDataBuf.ptr, 0, nbytes));
      CUDACHECK_TEST(cudaDeviceSynchronize());
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (globalRank == 0) {
      test::testPutSignal(
          peerTransportPtr,
          localDataBuf,
          remoteDataBuf,
          nbytes,
          0, // signal id
          1, // signal value
          numBlocks,
          blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());
    } else {
      test::testWaitSignal(
          peerTransportPtr,
          0, // signal id
          IbgdaCmpOp::GE, // comparison operation
          1, // expected signal
          numBlocks,
          blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      // Verify Rank 1 received Rank 0's pattern
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
    if (globalRank == 1) {
      // Fill local buffer with rank 1's pattern
      test::fillBufferWithPattern(
          localDataBuf.ptr, nbytes, rank1Pattern, numBlocks, blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());
    } else {
      // Clear buffer to receive
      CUDACHECK_TEST(cudaMemset(localDataBuf.ptr, 0, nbytes));
      CUDACHECK_TEST(cudaDeviceSynchronize());
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (globalRank == 1) {
      test::testPutSignal(
          peerTransportPtr,
          localDataBuf,
          remoteDataBuf,
          nbytes,
          0, // signal id
          2, // Use signal value 2 for phase 2
          numBlocks,
          blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());
    } else {
      test::testWaitSignal(
          peerTransportPtr,
          0, // signal id
          IbgdaCmpOp::GE, // comparison operation
          2, // expected signal
          numBlocks,
          blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      // Verify Rank 0 received Rank 1's pattern
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

  const std::size_t nbytes = 128 * 1024; // 128KB
  const int numIterations = 100;
  const int numBlocks = 2;
  const int blockSize = 64;
  const int peerRank = (globalRank == 0) ? 1 : 0;

  try {
    auto transport = createTransport();

    // Allocate and register user-owned data buffer
    DeviceBuffer dataBuffer(nbytes);
    auto localDataBuf = transport->registerBuffer(dataBuffer.get(), nbytes);

    // Collectively exchange buffer info to get remote buffer handles
    auto remoteDataBufs = transport->exchangeBuffer(localDataBuf);
    int peerIndex = (peerRank < globalRank) ? peerRank : (peerRank - 1);
    auto remoteDataBuf = remoteDataBufs[peerIndex];

    // Get peer transport for explicit peer selection
    P2pIbgdaTransportDevice* peerTransportPtr =
        transport->getP2pTransportDevice(peerRank);

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
            peerTransportPtr,
            localDataBuf,
            remoteDataBuf,
            nbytes,
            0, // signal id
            iter + 1, // signal value
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
            peerTransportPtr,
            0, // signal id
            IbgdaCmpOp::GE, // comparison operation
            iter + 1, // expected signal
            numBlocks,
            blockSize);
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

    // For SignalOnly test, use explicit peer selection
    int peerRank = (globalRank == 0) ? 1 : 0;
    P2pIbgdaTransportDevice* peerTransportPtr =
        transport->getP2pTransportDevice(peerRank);

    if (globalRank == 0) {
      // Sender: Send signal only (no data)
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      test::testSignalOnly(
          peerTransportPtr,
          0, // signal id
          42, // signal value
          numBlocks,
          blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    } else {
      // Receiver: Wait for signal
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      test::testWaitSignal(
          peerTransportPtr,
          0, // signal id
          IbgdaCmpOp::GE, // comparison
          42, // expected signal
          numBlocks,
          blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      // Verify signal value
      DeviceBuffer resultBuf(sizeof(uint64_t));
      auto* d_result = static_cast<uint64_t*>(resultBuf.get());

      test::testReadSignal(peerTransportPtr, 0, d_result, numBlocks, blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      uint64_t h_result = 0;
      CUDACHECK_TEST(cudaMemcpy(
          &h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost));

      EXPECT_GE(h_result, 42u) << "Signal value should be >= 42";
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }

  XLOGF(INFO, "Rank {}: SignalOnly test completed", globalRank);
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

    // For ResetSignal test, use explicit peer selection
    int peerRank = (globalRank == 0) ? 1 : 0;
    P2pIbgdaTransportDevice* peerTransportPtr =
        transport->getP2pTransportDevice(peerRank);

    for (int iter = 0; iter < numIterations; iter++) {
      if (globalRank == 0) {
        // Sender
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        // Send signal
        test::testSignalOnly(
            peerTransportPtr,
            0, // signal id
            1, // signal value (always 1)
            numBlocks,
            blockSize);
        CUDACHECK_TEST(cudaDeviceSynchronize());

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        // Reset the receiver's signal for next iteration
        test::testResetSignal(peerTransportPtr, 0, numBlocks, blockSize);
        CUDACHECK_TEST(cudaDeviceSynchronize());

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
      } else {
        // Receiver
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        // Wait for signal (always expecting 1 since sender resets)
        test::testWaitSignal(
            peerTransportPtr,
            0, // signal id
            IbgdaCmpOp::GE,
            1, // expected value
            numBlocks,
            blockSize);
        CUDACHECK_TEST(cudaDeviceSynchronize());

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        // Wait for sender to reset our signal
        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

        // Verify signal was reset to 0
        DeviceBuffer resultBuf(sizeof(uint64_t));
        auto* d_result = static_cast<uint64_t*>(resultBuf.get());

        test::testReadSignal(
            peerTransportPtr, 0, d_result, numBlocks, blockSize);
        CUDACHECK_TEST(cudaDeviceSynchronize());

        uint64_t h_result = 1; // Initialize to non-zero
        CUDACHECK_TEST(cudaMemcpy(
            &h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost));

        EXPECT_EQ(h_result, 0u)
            << "Iteration " << iter << ": Signal should be reset to 0";
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
    // Create transport with multiple signal slots
    MultipeerIbgdaTransportConfig config{
        .cudaDevice = localRank,
        .signalCount = static_cast<std::size_t>(numSignals),
    };

    auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
    MultipeerIbgdaTransport transport(globalRank, numRanks, bootstrap, config);
    transport.exchange();

    // For MultipleSignalSlots test, use explicit peer selection
    int peerRank = (globalRank == 0) ? 1 : 0;
    P2pIbgdaTransportDevice* peerTransportPtr =
        transport.getP2pTransportDevice(peerRank);

    if (globalRank == 0) {
      // Sender: Send signals to different slots
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      for (int i = 0; i < numSignals; i++) {
        test::testSignalOnly(
            peerTransportPtr,
            i, // signal id
            static_cast<uint64_t>(i + 1) * 10, // unique value per slot
            numBlocks,
            blockSize);
      }
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    } else {
      // Receiver: Wait for signals on each slot
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      for (int i = 0; i < numSignals; i++) {
        test::testWaitSignal(
            peerTransportPtr,
            i, // signal id
            IbgdaCmpOp::GE,
            static_cast<uint64_t>(i + 1) * 10, // expected value
            numBlocks,
            blockSize);
      }
      CUDACHECK_TEST(cudaDeviceSynchronize());

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      // Verify each signal slot has the correct value
      for (int i = 0; i < numSignals; i++) {
        DeviceBuffer resultBuf(sizeof(uint64_t));
        auto* d_result = static_cast<uint64_t*>(resultBuf.get());

        test::testReadSignal(
            peerTransportPtr, i, d_result, numBlocks, blockSize);
        CUDACHECK_TEST(cudaDeviceSynchronize());

        uint64_t h_result = 0;
        CUDACHECK_TEST(cudaMemcpy(
            &h_result, d_result, sizeof(uint64_t), cudaMemcpyDeviceToHost));

        uint64_t expected = static_cast<uint64_t>(i + 1) * 10;
        EXPECT_GE(h_result, expected)
            << "Signal slot " << i << ": expected >= " << expected << ", got "
            << h_result;
      }
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }

  XLOGF(INFO, "Rank {}: MultipleSignalSlots test completed", globalRank);
}

// =============================================================================
// Put Signal Wait For Ready Test - Sender waits for receiver's ready signal
// =============================================================================

TEST_F(MultipeerIbgdaTransportTestFixture, PutSignalWaitForReady) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  // This test uses 2 signal slots:
  // - Signal 0: "ready" signal from receiver to sender
  // - Signal 1: "data" signal from sender to receiver
  const std::size_t nbytes = 64 * 1024; // 64KB transfer
  const int numBlocks = 1;
  const int blockSize = 32;
  const int peerRank = (globalRank == 0) ? 1 : 0;
  const uint8_t testPattern = 0x55;

  try {
    // Need 2 signal slots for ready + data signals
    auto transport = createTransport(2);

    // Allocate and register user-owned data buffer
    DeviceBuffer dataBuffer(nbytes);
    auto localDataBuf = transport->registerBuffer(dataBuffer.get(), nbytes);

    // Collectively exchange buffer info to get remote buffer handles
    auto remoteDataBufs = transport->exchangeBuffer(localDataBuf);
    int peerIndex = (peerRank < globalRank) ? peerRank : (peerRank - 1);
    auto remoteDataBuf = remoteDataBufs[peerIndex];

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

      // Wait for receiver's "ready" signal, then put data with "data" signal
      test::testWaitReadyThenPutSignal(
          peerTransportPtr,
          localDataBuf,
          remoteDataBuf,
          nbytes,
          0, // readySignalId - wait on this
          1, // readySignalVal - wait for this value
          1, // dataSignalId - signal on this
          1, // dataSignalVal - send this value
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

      // Send "ready" signal to sender indicating buffer is ready
      test::testSignalOnly(
          peerTransportPtr,
          0, // signal id (ready signal)
          1, // signal value
          numBlocks,
          blockSize);
      CUDACHECK_TEST(cudaDeviceSynchronize());

      // Wait for "data" signal from sender
      test::testWaitSignal(
          peerTransportPtr,
          1, // signal id (data signal)
          IbgdaCmpOp::GE,
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
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }

  XLOGF(INFO, "Rank {}: PutSignalWaitForReady test completed", globalRank);
}

// =============================================================================
// Bidirectional Concurrent Test - Both ranks do put_signal and wait_signal
// =============================================================================

TEST_F(MultipeerIbgdaTransportTestFixture, BidirectionalConcurrent) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  // This test launches a kernel with 2 threads on each rank:
  // - Thread 0: does put_signal to send data to peer
  // - Thread 1: does wait_signal to receive data from peer
  //
  // Both ranks send and receive concurrently, testing true bidirectional
  // communication.
  //
  // Signal layout (2 slots per peer):
  // - Signal 0: used by rank 0 to signal rank 1
  // - Signal 1: used by rank 1 to signal rank 0

  const std::size_t nbytes = 64 * 1024; // 64KB transfer
  const int numBlocks = 1;
  const int blockSize = 2; // 2 threads: one for put, one for wait
  const int peerRank = (globalRank == 0) ? 1 : 0;
  const uint8_t rank0Pattern = 0xAA;
  const uint8_t rank1Pattern = 0xBB;
  const uint8_t myPattern = (globalRank == 0) ? rank0Pattern : rank1Pattern;
  const uint8_t peerPattern = (globalRank == 0) ? rank1Pattern : rank0Pattern;

  try {
    // Need 2 signal slots
    auto transport = createTransport(2);

    // Allocate separate send and receive buffers
    DeviceBuffer sendBuffer(nbytes);
    DeviceBuffer recvBuffer(nbytes);

    auto localSendBuf = transport->registerBuffer(sendBuffer.get(), nbytes);
    auto localRecvBuf = transport->registerBuffer(recvBuffer.get(), nbytes);

    // Exchange buffer info - each rank registers its recv buffer for remote
    // writes
    auto remoteRecvBufs = transport->exchangeBuffer(localRecvBuf);
    int peerIndex = (peerRank < globalRank) ? peerRank : (peerRank - 1);
    auto peerRecvBuf = remoteRecvBufs[peerIndex];

    // Get peer transport
    P2pIbgdaTransportDevice* peerTransportPtr =
        transport->getP2pTransportDevice(peerRank);

    // Fill send buffer with my pattern
    test::fillBufferWithPattern(
        localSendBuf.ptr, nbytes, myPattern, numBlocks, 32);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    // Clear receive buffer
    CUDACHECK_TEST(cudaMemset(localRecvBuf.ptr, 0, nbytes));
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Determine signal IDs based on rank
    // Rank 0 sends on signal 0, receives on signal 1
    // Rank 1 sends on signal 1, receives on signal 0
    int sendSignalId = globalRank;
    int recvSignalId = peerRank;

    // Launch bidirectional kernel:
    // - Thread 0 does put_signal (sends data to peer's recv buffer)
    // - Thread 1 does wait_signal (waits for data from peer)
    test::testBidirectionalPutWait(
        peerTransportPtr,
        localSendBuf,
        peerRecvBuf,
        nbytes,
        sendSignalId,
        1, // sendSignalVal
        recvSignalId,
        1, // recvSignalVal
        numBlocks,
        blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Verify received data matches peer's pattern
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

  // This test uses the partition API to parallelize communication across peers:
  //   [peer_id, per_peer_group] = group.partition(numPeers);
  //
  // Each rank sends unique data to all peers and receives data from all peers
  // concurrently using a single kernel launch.
  //
  // Buffer layout: Single large buffer split into per-peer chunks
  // - sendBuffer: [chunk_for_peer0 | chunk_for_peer1 | ... ]
  // - recvBuffer: [chunk_from_peer0 | chunk_from_peer1 | ... ]

  const int numPeers = numRanks - 1;
  const std::size_t bytesPerPeer = 64 * 1024; // 64KB per peer
  const std::size_t totalBytes = bytesPerPeer * numPeers;
  // Need at least numPeers blocks so partition() works (num_partitions <=
  // total_groups)
  const int numBlocks = numPeers;
  const int blockSize = 32; // At least 2 threads per block for send/recv

  try {
    // Create transport with enough signal slots for all ranks
    // (we use peerRank as signal ID, which ranges from 0 to numRanks-1)
    auto transport = createTransport(static_cast<std::size_t>(numRanks));

    // Allocate single send and receive buffers (split into per-peer chunks)
    DeviceBuffer sendBuffer(totalBytes);
    DeviceBuffer recvBuffer(totalBytes);

    auto localSendBuf = transport->registerBuffer(sendBuffer.get(), totalBytes);
    auto localRecvBuf = transport->registerBuffer(recvBuffer.get(), totalBytes);

    // Single collective exchange for the receive buffer
    auto remoteRecvBufs = transport->exchangeBuffer(localRecvBuf);

    // Build per-peer buffer views and transport pointers
    std::vector<IbgdaLocalBuffer> localSendBufsPerPeer(numPeers);
    std::vector<IbgdaRemoteBuffer> peerRecvBufs(numPeers);
    std::vector<P2pIbgdaTransportDevice*> peerTransports(numPeers);
    std::vector<int> peerRanksVec(numPeers);

    for (int peerIndex = 0; peerIndex < numPeers; peerIndex++) {
      int peerRank = (peerIndex < globalRank) ? peerIndex : (peerIndex + 1);
      peerRanksVec[peerIndex] = peerRank;

      // Local send buffer chunk for this peer
      localSendBufsPerPeer[peerIndex] =
          localSendBuf.subBuffer(peerIndex * bytesPerPeer);

      // Remote receive buffer chunk - peer writes to their slot in our buffer
      // Peer's peerIndex for us determines where they write
      int ourIndexOnPeer =
          (globalRank < peerRank) ? globalRank : (globalRank - 1);
      peerRecvBufs[peerIndex] =
          remoteRecvBufs[peerIndex].subBuffer(ourIndexOnPeer * bytesPerPeer);

      peerTransports[peerIndex] = transport->getP2pTransportDevice(peerRank);
    }

    // Fill send buffer with unique pattern per rank
    const uint8_t myPattern = static_cast<uint8_t>(0x30 + globalRank);
    test::fillBufferWithPattern(
        localSendBuf.ptr, totalBytes, myPattern, 4, 128);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    // Clear receive buffer
    CUDACHECK_TEST(cudaMemset(localRecvBuf.ptr, 0, totalBytes));
    CUDACHECK_TEST(cudaDeviceSynchronize());

    // Copy arrays to device memory
    DeviceBuffer d_peerTransports(numPeers * sizeof(P2pIbgdaTransportDevice*));
    DeviceBuffer d_localSendBufs(numPeers * sizeof(IbgdaLocalBuffer));
    DeviceBuffer d_peerRecvBufs(numPeers * sizeof(IbgdaRemoteBuffer));
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
        d_peerRanks.get(),
        peerRanksVec.data(),
        numPeers * sizeof(int),
        cudaMemcpyHostToDevice));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // All ranks send data to all peers using partition() API
    test::testAllToAll(
        static_cast<P2pIbgdaTransportDevice**>(d_peerTransports.get()),
        static_cast<IbgdaLocalBuffer*>(d_localSendBufs.get()),
        static_cast<IbgdaRemoteBuffer*>(d_peerRecvBufs.get()),
        static_cast<int*>(d_peerRanks.get()),
        globalRank,
        bytesPerPeer,
        numPeers,
        numBlocks,
        blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    // Barrier to ensure all sends are complete before verifying data
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Verify received data from each peer
    int totalErrors = 0;
    for (int peerIndex = 0; peerIndex < numPeers; peerIndex++) {
      int peerRank = (peerIndex < globalRank) ? peerIndex : (peerIndex + 1);
      uint8_t expectedPattern = static_cast<uint8_t>(0x30 + peerRank);

      DeviceBuffer errorCountBuf(sizeof(int));
      auto* d_errorCount = static_cast<int*>(errorCountBuf.get());
      CUDACHECK_TEST(cudaMemset(d_errorCount, 0, sizeof(int)));

      // Verify the chunk in our receive buffer from this peer
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
