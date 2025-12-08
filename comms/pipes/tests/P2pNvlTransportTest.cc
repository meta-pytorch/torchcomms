// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#define P2pNvlTransport_TEST_FRIENDS \
  FRIEND_TEST(P2pNvlTransportTestFixture, IpcMemAccess);

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include <string>

#include "comms/pipes/P2pNvlTransport.h"
#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/pipes/tests/P2pNvlTransportTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

using namespace meta::comms;

namespace comms::pipes {

// Parameters for transfer size tests: (nbytes, dataBufferSize, chunkSize, name)
struct TransferSizeParams {
  size_t nbytes;
  size_t dataBufferSize;
  size_t chunkSize;
  std::string name;
};

// Parameters for group type tests: (groupType, numBlocks, blockSize,
// blocksPerGroup, name)
struct GroupTypeParams {
  test::GroupType groupType;
  int numBlocks;
  int blockSize;
  int blocksPerGroup;
  std::string name;
};

class P2pNvlTransportTestFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    cudaSetDevice(localRank);
  }

  void TearDown() override {
    MpiBaseTestFixture::TearDown();
  }
};

TEST_F(P2pNvlTransportTestFixture, IpcMemAccess) {
  // Only test with 2 ranks
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;

  const size_t numElements = 256;
  P2pNvlTransportConfig config{
      .dataBufferSize = sizeof(int) * numElements,
      .chunkSize = 256, // 256 bytes
  };

  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  P2pNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.exchange();
  XLOGF(INFO, "Rank {} created transport and exchanged IPC", globalRank);

  auto p2p = transport.getTransportDevice(peerRank);

  auto localAddr =
      static_cast<int*>(static_cast<void*>(p2p.localState_.dataBuffer));
  auto remoteAddr =
      static_cast<int*>(static_cast<void*>(p2p.remoteState_.dataBuffer));
  XLOGF(
      INFO,
      "Rank {}: localAddr: {}, remoteAddr: {}",
      globalRank,
      static_cast<void*>(localAddr),
      static_cast<void*>(remoteAddr));

  // Each rank writes its pattern to local buffer
  // rank0 writes all 0s, rank1 writes all 1s
  int writeValue = globalRank;
  test::fillBuffer(localAddr, writeValue, numElements);
  CUDACHECK_TEST(cudaDeviceSynchronize());
  XLOGF(INFO, "Rank {} filled local buffer with {}", globalRank, writeValue);

  // Barrier to ensure both ranks have written their data
  MPI_Barrier(MPI_COMM_WORLD);
  XLOGF(INFO, "Rank {} passed barrier", globalRank);

  // Now each rank reads from peer buffer and verifies
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

void runSendRecvIteration(
    int globalRank,
    P2pNvlTransportDevice& p2p,
    int* src_d,
    int* dst_d,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    int iter) {
  const size_t numInts = nbytes / sizeof(int);
  const int testValue = 42 + iter;

  if (globalRank == 0) {
    test::fillBuffer(src_d, testValue, numInts);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    XLOGF(
        INFO,
        "Rank 0 [iter {}]: filled source buffer with {}",
        iter,
        testValue);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    test::testSend(p2p, src_d, nbytes, numBlocks, blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    XLOGF(INFO, "Rank 0 [iter {}]: send completed", iter);
  } else {
    CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));
    XLOGF(INFO, "Rank 1 [iter {}]: cleared destination buffer", iter);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    test::testRecv(p2p, dst_d, nbytes, numBlocks, blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    XLOGF(INFO, "Rank 1 [iter {}]: recv completed", iter);

    std::vector<int> hostBuffer(numInts);
    CUDACHECK_TEST(
        cudaMemcpy(hostBuffer.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < numInts; i++) {
      EXPECT_EQ(hostBuffer[i], testValue)
          << "Iter " << iter << ": Mismatch at index " << i << ": expected "
          << testValue << ", got " << hostBuffer[i];
      if (hostBuffer[i] != testValue) {
        break;
      }
    }

    XLOGF(INFO, "Rank 1 [iter {}]: verification completed", iter);
  }
}

// =============================================================================
// Parameterized Test Fixture for Transfer Size Variations
// =============================================================================

class TransferSizeTestFixture
    : public MpiBaseTestFixture,
      public ::testing::WithParamInterface<TransferSizeParams> {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    cudaSetDevice(localRank);
  }
};

TEST_P(TransferSizeTestFixture, SendRecv) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const auto& params = GetParam();
  XLOGF(
      INFO,
      "Running transfer size test: {} (nbytes={}, bufferSize={}, chunkSize={})",
      params.name,
      params.nbytes,
      params.dataBufferSize,
      params.chunkSize);

  int peerRank = (globalRank == 0) ? 1 : 0;

  P2pNvlTransportConfig config{
      .dataBufferSize = params.dataBufferSize,
      .chunkSize = params.chunkSize,
  };

  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  P2pNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.exchange();

  auto p2p = transport.getTransportDevice(peerRank);

  DeviceBuffer srcBuffer(params.nbytes);
  DeviceBuffer dstBuffer(params.nbytes);

  auto src_d = static_cast<int*>(srcBuffer.get());
  auto dst_d = static_cast<int*>(dstBuffer.get());

  const int numBlocks = 2;
  const int blockSize = 64;

  runSendRecvIteration(
      globalRank, p2p, src_d, dst_d, params.nbytes, numBlocks, blockSize, 0);

  XLOGF(
      INFO,
      "Rank {}: Transfer size test '{}' completed",
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
        // Small transfer: nbytes < chunkSize
        TransferSizeParams{
            .nbytes = 512,
            .dataBufferSize = 4096,
            .chunkSize = 1024,
            .name = "SmallTransfer_LessThanChunk"},
        // Single chunk: nbytes == chunkSize
        TransferSizeParams{
            .nbytes = 1024,
            .dataBufferSize = 4096,
            .chunkSize = 1024,
            .name = "SingleChunk_ExactMatch"},
        // Transfer not aligned to chunk size
        TransferSizeParams{
            .nbytes = 1000,
            .dataBufferSize = 4096,
            .chunkSize = 256,
            .name = "UnalignedToChunk"},
        // Transfer not aligned to vector size (16 bytes)
        TransferSizeParams{
            .nbytes = 1000,
            .dataBufferSize = 4096,
            .chunkSize = 256,
            .name = "NonVectorAligned_1000bytes"},
        // Another non-vector-aligned size
        TransferSizeParams{
            .nbytes = 100,
            .dataBufferSize = 1024,
            .chunkSize = 64,
            .name = "NonVectorAligned_100bytes"},
        // Transfer exactly equals buffer size (single step)
        TransferSizeParams{
            .nbytes = 4096,
            .dataBufferSize = 4096,
            .chunkSize = 256,
            .name = "ExactBufferSize"},
        // Multiple steps: transfer > buffer size
        TransferSizeParams{
            .nbytes = 16 * 1024,
            .dataBufferSize = 4096,
            .chunkSize = 256,
            .name = "MultipleSteps_4x"},
        // Large transfer with multiple steps
        TransferSizeParams{
            .nbytes = 4 * 1024 * 1024,
            .dataBufferSize = 1024 * 1024,
            .chunkSize = 4096,
            .name = "LargeMultiStep_4MB"},
        // Very large transfer (64MB with 8MB buffer = 8 steps)
        TransferSizeParams{
            .nbytes = 64 * 1024 * 1024,
            .dataBufferSize = 8 * 1024 * 1024,
            .chunkSize = 1024,
            .name = "VeryLargeMultiStep_64MB"}),
    transferSizeParamName);

// =============================================================================
// Parameterized Test Fixture for Group Type Variations
// =============================================================================

class GroupTypeTestFixture
    : public MpiBaseTestFixture,
      public ::testing::WithParamInterface<GroupTypeParams> {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    cudaSetDevice(localRank);
  }
};

TEST_P(GroupTypeTestFixture, SendRecv) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const auto& params = GetParam();
  XLOGF(
      INFO,
      "Running group type test: {} (numBlocks={}, blockSize={}, blocksPerGroup={})",
      params.name,
      params.numBlocks,
      params.blockSize,
      params.blocksPerGroup);

  int peerRank = (globalRank == 0) ? 1 : 0;

  const size_t dataBufferSize = 1024 * 1024; // 1MB staging buffer
  const size_t nbytes = 4 * 1024 * 1024; // 4MB total transfer
  P2pNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 1024,
  };

  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  P2pNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.exchange();

  auto p2p = transport.getTransportDevice(peerRank);

  const size_t numInts = nbytes / sizeof(int);
  DeviceBuffer srcBuffer(nbytes);
  DeviceBuffer dstBuffer(nbytes);

  auto src_d = static_cast<int*>(srcBuffer.get());
  auto dst_d = static_cast<int*>(dstBuffer.get());

  const int testValue = 42;

  if (globalRank == 0) {
    test::fillBuffer(src_d, testValue, numInts);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    test::testSend(
        p2p,
        src_d,
        nbytes,
        params.numBlocks,
        params.blockSize,
        params.groupType,
        params.blocksPerGroup);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  } else {
    CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    test::testRecv(
        p2p,
        dst_d,
        nbytes,
        params.numBlocks,
        params.blockSize,
        params.groupType,
        params.blocksPerGroup);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    std::vector<int> hostBuffer(numInts);
    CUDACHECK_TEST(
        cudaMemcpy(hostBuffer.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < numInts; i++) {
      EXPECT_EQ(hostBuffer[i], testValue)
          << "Mismatch at index " << i << ": expected " << testValue << ", got "
          << hostBuffer[i];
      if (hostBuffer[i] != testValue) {
        break;
      }
    }
  }

  XLOGF(
      INFO, "Rank {}: Group type test '{}' completed", globalRank, params.name);
}

std::string groupTypeParamName(
    const ::testing::TestParamInfo<GroupTypeParams>& info) {
  return info.param.name;
}

INSTANTIATE_TEST_SUITE_P(
    GroupTypeVariations,
    GroupTypeTestFixture,
    ::testing::Values(
        // Warp-based groups (32 threads per group)
        GroupTypeParams{
            .groupType = test::GroupType::WARP,
            .numBlocks = 4,
            .blockSize = 128,
            .blocksPerGroup = 1,
            .name = "Warp_4Blocks_128Threads"},
        GroupTypeParams{
            .groupType = test::GroupType::WARP,
            .numBlocks = 8,
            .blockSize = 256,
            .blocksPerGroup = 1,
            .name = "Warp_8Blocks_256Threads"},
        // Block-based groups (all threads in block form one group)
        GroupTypeParams{
            .groupType = test::GroupType::BLOCK,
            .numBlocks = 4,
            .blockSize = 128,
            .blocksPerGroup = 1,
            .name = "Block_4Groups_128Threads"},
        GroupTypeParams{
            .groupType = test::GroupType::BLOCK,
            .numBlocks = 8,
            .blockSize = 256,
            .blocksPerGroup = 1,
            .name = "Block_8Groups_256Threads"},
        GroupTypeParams{
            .groupType = test::GroupType::BLOCK,
            .numBlocks = 2,
            .blockSize = 512,
            .blocksPerGroup = 1,
            .name = "Block_2Groups_512Threads"}),
    groupTypeParamName);

// =============================================================================
// Bidirectional Send/Recv Test
// =============================================================================

TEST_F(P2pNvlTransportTestFixture, BidirectionalSendRecv) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;

  const size_t dataBufferSize = 1024 * 1024; // 1MB staging buffer
  const size_t nbytes = 4 * 1024 * 1024; // 4MB total transfer
  P2pNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 1024,
  };

  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  P2pNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.exchange();

  auto p2p = transport.getTransportDevice(peerRank);

  const size_t numInts = nbytes / sizeof(int);

  // Each rank has both send and receive buffers
  DeviceBuffer sendBuffer(nbytes);
  DeviceBuffer recvBuffer(nbytes);

  auto send_d = static_cast<int*>(sendBuffer.get());
  auto recv_d = static_cast<int*>(recvBuffer.get());

  const int numBlocks = 4;
  const int blockSize = 128;

  // Each rank uses a different test value
  const int sendValue = 100 + globalRank;
  const int expectedRecvValue = 100 + peerRank;

  // Fill send buffer with this rank's value
  test::fillBuffer(send_d, sendValue, numInts);
  CUDACHECK_TEST(cudaMemset(recv_d, 0, nbytes));
  CUDACHECK_TEST(cudaDeviceSynchronize());

  XLOGF(
      INFO,
      "Rank {}: filled send buffer with {}, expecting to receive {}",
      globalRank,
      sendValue,
      expectedRecvValue);

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Both ranks send and receive simultaneously
  // Rank 0 sends first, then receives
  // Rank 1 receives first, then sends
  // This tests that the state buffers are managed correctly for bidirectional
  if (globalRank == 0) {
    test::testSend(p2p, send_d, nbytes, numBlocks, blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    test::testRecv(p2p, recv_d, nbytes, numBlocks, blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  } else {
    test::testRecv(p2p, recv_d, nbytes, numBlocks, blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    test::testSend(p2p, send_d, nbytes, numBlocks, blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  }

  // Verify received data
  std::vector<int> hostBuffer(numInts);
  CUDACHECK_TEST(
      cudaMemcpy(hostBuffer.data(), recv_d, nbytes, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < numInts; i++) {
    EXPECT_EQ(hostBuffer[i], expectedRecvValue)
        << "Rank " << globalRank << ": Mismatch at index " << i << ": expected "
        << expectedRecvValue << ", got " << hostBuffer[i];
    if (hostBuffer[i] != expectedRecvValue) {
      break;
    }
  }

  XLOGF(INFO, "Rank {}: Bidirectional test completed", globalRank);
}

// =============================================================================
// Stress Test with Many Iterations
// =============================================================================

TEST_F(P2pNvlTransportTestFixture, SendRecvStress) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;

  const size_t dataBufferSize = 512 * 1024; // 512KB staging buffer
  const size_t nbytes = 2 * 1024 * 1024; // 2MB total transfer
  P2pNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 512,
  };

  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  P2pNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.exchange();

  auto p2p = transport.getTransportDevice(peerRank);

  DeviceBuffer srcBuffer(nbytes);
  DeviceBuffer dstBuffer(nbytes);

  auto src_d = static_cast<int*>(srcBuffer.get());
  auto dst_d = static_cast<int*>(dstBuffer.get());

  const int numBlocks = 4;
  const int blockSize = 128;
  const int nIter = 100; // Stress test with many iterations

  XLOGF(
      INFO,
      "Rank {}: Starting stress test with {} iterations",
      globalRank,
      nIter);

  for (int iter = 0; iter < nIter; iter++) {
    runSendRecvIteration(
        globalRank, p2p, src_d, dst_d, nbytes, numBlocks, blockSize, iter);

    if ((iter + 1) % 25 == 0) {
      XLOGF(
          INFO,
          "Rank {}: Completed {}/{} iterations",
          globalRank,
          iter + 1,
          nIter);
    }
  }

  XLOGF(
      INFO,
      "Rank {}: Stress test completed ({} iterations)",
      globalRank,
      nIter);
}

// =============================================================================
// Zero-Byte Transfer Test
// =============================================================================

TEST_F(P2pNvlTransportTestFixture, SendRecvZeroBytes) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;

  const size_t dataBufferSize = 4096;
  P2pNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 256,
  };

  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  P2pNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.exchange();

  auto p2p = transport.getTransportDevice(peerRank);

  // Allocate small buffers for the zero-byte transfer test
  const size_t bufferSize = 64;
  const size_t numInts = bufferSize / sizeof(int);
  DeviceBuffer srcBuffer(bufferSize);
  DeviceBuffer dstBuffer(bufferSize);

  auto src_d = static_cast<int*>(srcBuffer.get());
  auto dst_d = static_cast<int*>(dstBuffer.get());

  const int numBlocks = 1;
  const int blockSize = 32;
  const size_t nbytes = 0; // Zero-byte transfer

  // Initialize destination buffer with a known pattern to verify it remains
  // unchanged
  const int initialValue = 999;
  test::fillBuffer(dst_d, initialValue, numInts);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  if (globalRank == 0) {
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    test::testSend(p2p, src_d, nbytes, numBlocks, blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  } else {
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    test::testRecv(p2p, dst_d, nbytes, numBlocks, blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Verify that the destination buffer was NOT modified (since zero bytes
    // transferred)
    std::vector<int> hostBuffer(numInts);
    CUDACHECK_TEST(cudaMemcpy(
        hostBuffer.data(), dst_d, bufferSize, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < numInts; i++) {
      EXPECT_EQ(hostBuffer[i], initialValue)
          << "Zero-byte transfer modified buffer at index " << i
          << ": expected " << initialValue << ", got " << hostBuffer[i];
      if (hostBuffer[i] != initialValue) {
        break;
      }
    }
  }

  XLOGF(INFO, "Rank {}: Zero-byte transfer test completed", globalRank);
}

// =============================================================================
// Multiple Sends in Single Kernel Test
// =============================================================================

TEST_F(P2pNvlTransportTestFixture, MultiSendInKernel) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;

  const size_t dataBufferSize = 512 * 1024; // 512KB staging buffer
  const size_t nbytesPerSend = 256 * 1024; // 256KB per send
  const int numSends = 4;
  const size_t totalBytes = nbytesPerSend * numSends;

  P2pNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 1024,
  };

  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  P2pNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.exchange();

  auto p2p = transport.getTransportDevice(peerRank);

  DeviceBuffer srcBuffer(totalBytes);
  DeviceBuffer dstBuffer(totalBytes);

  auto src_d = static_cast<int*>(srcBuffer.get());
  auto dst_d = static_cast<int*>(dstBuffer.get());

  const int numBlocks = 4;
  const int blockSize = 128;

  // Fill source buffer with different values for each segment
  const size_t intsPerSend = nbytesPerSend / sizeof(int);
  for (int i = 0; i < numSends; i++) {
    test::fillBuffer(src_d + i * intsPerSend, 100 + i, intsPerSend);
  }
  CUDACHECK_TEST(cudaDeviceSynchronize());

  if (globalRank == 0) {
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    // Single kernel launch that does multiple sends
    test::testMultiSend(
        p2p, src_d, nbytesPerSend, numSends, numBlocks, blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  } else {
    CUDACHECK_TEST(cudaMemset(dst_d, 0, totalBytes));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    // Single kernel launch that does multiple recvs
    test::testMultiRecv(
        p2p, dst_d, nbytesPerSend, numSends, numBlocks, blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Verify each segment
    std::vector<int> hostBuffer(intsPerSend);
    for (int i = 0; i < numSends; i++) {
      CUDACHECK_TEST(cudaMemcpy(
          hostBuffer.data(),
          dst_d + i * intsPerSend,
          nbytesPerSend,
          cudaMemcpyDeviceToHost));

      const int expectedValue = 100 + i;
      for (size_t j = 0; j < intsPerSend; j++) {
        EXPECT_EQ(hostBuffer[j], expectedValue)
            << "Segment " << i << ", index " << j << ": expected "
            << expectedValue << ", got " << hostBuffer[j];
        if (hostBuffer[j] != expectedValue) {
          break;
        }
      }
    }
  }

  XLOGF(INFO, "Rank {}: MultiSendInKernel test completed", globalRank);
}

// =============================================================================
// Multiple Recvs in Single Kernel Test
// =============================================================================

TEST_F(P2pNvlTransportTestFixture, MultiRecvInKernel) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;

  const size_t dataBufferSize = 512 * 1024; // 512KB staging buffer
  const size_t nbytesPerRecv = 128 * 1024; // 128KB per recv
  const int numRecvs = 8;
  const size_t totalBytes = nbytesPerRecv * numRecvs;

  P2pNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 512,
  };

  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  P2pNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.exchange();

  auto p2p = transport.getTransportDevice(peerRank);

  DeviceBuffer srcBuffer(totalBytes);
  DeviceBuffer dstBuffer(totalBytes);

  auto src_d = static_cast<int*>(srcBuffer.get());
  auto dst_d = static_cast<int*>(dstBuffer.get());

  const int numBlocks = 2;
  const int blockSize = 64;

  // Fill source buffer with unique pattern
  const size_t intsPerRecv = nbytesPerRecv / sizeof(int);
  for (int i = 0; i < numRecvs; i++) {
    test::fillBuffer(src_d + i * intsPerRecv, 200 + i, intsPerRecv);
  }
  CUDACHECK_TEST(cudaDeviceSynchronize());

  if (globalRank == 0) {
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    test::testMultiSend(
        p2p, src_d, nbytesPerRecv, numRecvs, numBlocks, blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  } else {
    CUDACHECK_TEST(cudaMemset(dst_d, 0, totalBytes));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    test::testMultiRecv(
        p2p, dst_d, nbytesPerRecv, numRecvs, numBlocks, blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Verify each segment
    std::vector<int> hostBuffer(intsPerRecv);
    for (int i = 0; i < numRecvs; i++) {
      CUDACHECK_TEST(cudaMemcpy(
          hostBuffer.data(),
          dst_d + i * intsPerRecv,
          nbytesPerRecv,
          cudaMemcpyDeviceToHost));

      const int expectedValue = 200 + i;
      for (size_t j = 0; j < intsPerRecv; j++) {
        EXPECT_EQ(hostBuffer[j], expectedValue)
            << "Segment " << i << ", index " << j << ": expected "
            << expectedValue << ", got " << hostBuffer[j];
        if (hostBuffer[j] != expectedValue) {
          break;
        }
      }
    }
  }

  XLOGF(INFO, "Rank {}: MultiRecvInKernel test completed", globalRank);
}

// =============================================================================
// Simultaneous Send+Recv in Single Kernel Test
// =============================================================================

TEST_F(P2pNvlTransportTestFixture, SimultaneousSendRecvInKernel) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;

  const size_t dataBufferSize = 1024 * 1024; // 1MB staging buffer
  const size_t nbytes = 2 * 1024 * 1024; // 2MB transfer each direction

  P2pNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 1024,
  };

  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  P2pNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.exchange();

  auto p2p = transport.getTransportDevice(peerRank);

  const size_t numInts = nbytes / sizeof(int);

  // Each rank has send and receive buffers
  DeviceBuffer sendBuffer(nbytes);
  DeviceBuffer recvBuffer(nbytes);

  auto send_d = static_cast<int*>(sendBuffer.get());
  auto recv_d = static_cast<int*>(recvBuffer.get());

  const int numBlocks = 4;
  const int blockSize = 128;

  // Each rank uses unique values
  const int sendValue = 300 + globalRank;
  const int expectedRecvValue = 300 + peerRank;

  test::fillBuffer(send_d, sendValue, numInts);
  CUDACHECK_TEST(cudaMemset(recv_d, 0, nbytes));
  CUDACHECK_TEST(cudaDeviceSynchronize());

  XLOGF(
      INFO,
      "Rank {}: Simulatenous test - sending {}, expecting {}",
      globalRank,
      sendValue,
      expectedRecvValue);

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Both ranks do send+recv in a single kernel, but in opposite order
  // to avoid deadlock: rank 0 sends then recvs, rank 1 recvs then sends
  if (globalRank == 0) {
    test::testSendRecv(p2p, send_d, recv_d, nbytes, numBlocks, blockSize);
  } else {
    test::testRecvSend(p2p, recv_d, send_d, nbytes, numBlocks, blockSize);
  }
  CUDACHECK_TEST(cudaDeviceSynchronize());

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Verify received data
  std::vector<int> hostBuffer(numInts);
  CUDACHECK_TEST(
      cudaMemcpy(hostBuffer.data(), recv_d, nbytes, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < numInts; i++) {
    EXPECT_EQ(hostBuffer[i], expectedRecvValue)
        << "Rank " << globalRank << ": Mismatch at index " << i << ": expected "
        << expectedRecvValue << ", got " << hostBuffer[i];
    if (hostBuffer[i] != expectedRecvValue) {
      break;
    }
  }

  XLOGF(
      INFO, "Rank {}: SimultaneousSendRecvInKernel test completed", globalRank);
}

} // namespace comms::pipes

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
