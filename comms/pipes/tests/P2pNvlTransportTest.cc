// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include <string>

#include "comms/pipes/MultiPeerNvlTransport.h"
#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/pipes/tests/P2pNvlTransportTest.cuh"
#include "comms/pipes/tests/Utils.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

using namespace meta::comms;

namespace comms::pipes::tests {

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
    CUDACHECK_TEST(cudaSetDevice(localRank));
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
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = sizeof(int) * numElements,
      .chunkSize = 256, // 256 bytes
      .pipelineDepth = 4,
  };

  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  MultiPeerNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.exchange();
  XLOGF(INFO, "Rank {} created transport and exchanged IPC", globalRank);

  auto p2p = transport.getP2pTransportDevice(peerRank);

  auto localAddr =
      static_cast<int*>(static_cast<void*>(p2p.getLocalState().dataBuffer));
  auto remoteAddr =
      static_cast<int*>(static_cast<void*>(p2p.getRemoteState().dataBuffer));
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

// Helper to verify received data with early exit on first mismatch
void verifyReceivedData(
    const int* dst_d,
    size_t nbytes,
    int expectedValue,
    const std::string& context = "") {
  const size_t numInts = nbytes / sizeof(int);
  std::vector<int> hostBuffer(numInts);
  CUDACHECK_TEST(
      cudaMemcpy(hostBuffer.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < numInts; i++) {
    EXPECT_EQ(hostBuffer[i], expectedValue)
        << context << "Mismatch at index " << i << ": expected "
        << expectedValue << ", got " << hostBuffer[i];
    if (hostBuffer[i] != expectedValue) {
      break;
    }
  }
}
// Helper to run a single send/recv iteration with verification
void runSendRecvIteration(
    int globalRank,
    P2pNvlTransportDevice& p2p,
    int* src_d,
    int* dst_d,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    int iter,
    test::GroupType groupType = test::GroupType::WARP) {
  const size_t numInts = nbytes / sizeof(int);
  const int testValue = 42 + iter;

  if (globalRank == 0) {
    test::fillBuffer(src_d, testValue, numInts);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    test::testSend(p2p, src_d, nbytes, numBlocks, blockSize, groupType, 1);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  } else {
    CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    test::testRecv(p2p, dst_d, nbytes, numBlocks, blockSize, groupType, 1);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

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
  }
}

// =============================================================================
// TransportTestHelper - Reduces boilerplate for creating transport objects
// =============================================================================

class TransportTestHelper {
 public:
  TransportTestHelper(
      int globalRank,
      int numRanks,
      int localRank,
      const MultiPeerNvlTransportConfig& config)
      : globalRank_(globalRank),
        numRanks_(numRanks),
        peerRank_((globalRank == 0) ? 1 : 0),
        bootstrap_(std::make_shared<meta::comms::MpiBootstrap>()),
        transport_(
            std::make_unique<MultiPeerNvlTransport>(
                globalRank,
                numRanks,
                bootstrap_,
                config)) {
    CUDACHECK_TEST(cudaSetDevice(localRank));
    transport_->exchange();
  }

  P2pNvlTransportDevice getDevice() {
    return transport_->getP2pTransportDevice(peerRank_);
  }

  int peerRank() const {
    return peerRank_;
  }

  int globalRank() const {
    return globalRank_;
  }

 private:
  int globalRank_;
  int numRanks_;
  int peerRank_;
  std::shared_ptr<meta::comms::MpiBootstrap> bootstrap_;
  std::unique_ptr<MultiPeerNvlTransport> transport_;
};

// =============================================================================
// runBasicSendRecvTest - Common test pattern for send/recv verification
// =============================================================================

void runBasicSendRecvTest(
    TransportTestHelper& helper,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    int nIter = 1,
    test::GroupType groupType = test::GroupType::WARP) {
  auto p2p = helper.getDevice();

  DeviceBuffer srcBuffer(nbytes);
  DeviceBuffer dstBuffer(nbytes);

  auto src_d = static_cast<int*>(srcBuffer.get());
  auto dst_d = static_cast<int*>(dstBuffer.get());

  for (int iter = 0; iter < nIter; iter++) {
    runSendRecvIteration(
        helper.globalRank(),
        p2p,
        src_d,
        dst_d,
        nbytes,
        numBlocks,
        blockSize,
        iter,
        groupType);
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
    CUDACHECK_TEST(cudaSetDevice(localRank));
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

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = params.dataBufferSize,
      .chunkSize = params.chunkSize,
      .pipelineDepth = 4,
  };

  TransportTestHelper helper(globalRank, numRanks, localRank, config);
  runBasicSendRecvTest(helper, params.nbytes, 2, 64);

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
            .name = "VeryLargeMultiStep_64MB"},
        // Edge case: stepBytes exactly divisible by chunkSize (no partial
        // chunk) Tests that we don't process any 0-byte chunks
        TransferSizeParams{
            .nbytes = 4096,
            .dataBufferSize = 4096,
            .chunkSize = 1024,
            .name = "ExactChunkBoundary_4Chunks"},
        // Edge case: last chunk has minimal bytes (1 byte remainder)
        // stepBytes=4097, chunkSize=1024 → 5 chunks, last chunk = 1 byte
        TransferSizeParams{
            .nbytes = 4097,
            .dataBufferSize = 8192,
            .chunkSize = 1024,
            .name = "MinimalLastChunk_1Byte"},
        // Edge case: multiple steps where each step ends exactly on chunk
        // boundary 8KB transfer, 4KB buffer, 1KB chunks → 2 steps × 4 chunks
        // each
        TransferSizeParams{
            .nbytes = 8 * 1024,
            .dataBufferSize = 4 * 1024,
            .chunkSize = 1024,
            .name = "MultiStep_ExactChunkBoundaries"},
        // Edge case: chunkSize larger than stepBytes
        // Forces single chunk per step with partial fill
        TransferSizeParams{
            .nbytes = 2048,
            .dataBufferSize = 1024,
            .chunkSize = 2048,
            .name = "ChunkLargerThanStep"}),
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
    CUDACHECK_TEST(cudaSetDevice(localRank));
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
      "Running group type test: {} (numBlocks={}, blockSize={})",
      params.name,
      params.numBlocks,
      params.blockSize);

  const size_t dataBufferSize = 1024 * 1024; // 1MB staging buffer
  const size_t nbytes = 4 * 1024 * 1024; // 4MB total transfer
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 1024,
      .pipelineDepth = 4,
  };

  TransportTestHelper helper(globalRank, numRanks, localRank, config);
  runBasicSendRecvTest(
      helper, nbytes, params.numBlocks, params.blockSize, 1, params.groupType);

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

  const size_t dataBufferSize = 1024 * 1024; // 1MB staging buffer
  const size_t nbytes = 4 * 1024 * 1024; // 4MB total transfer
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 1024,
      .pipelineDepth = 4,
  };

  TransportTestHelper helper(globalRank, numRanks, localRank, config);
  auto p2p = helper.getDevice();

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
  const int expectedRecvValue = 100 + helper.peerRank();

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

  const size_t dataBufferSize = 512 * 1024; // 512KB staging buffer
  const size_t nbytes = 2 * 1024 * 1024; // 2MB total transfer
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 512,
      .pipelineDepth = 4,
  };

  TransportTestHelper helper(globalRank, numRanks, localRank, config);
  const int nIter = 100;

  XLOGF(
      INFO,
      "Rank {}: Starting stress test with {} iterations",
      globalRank,
      nIter);

  runBasicSendRecvTest(helper, nbytes, 4, 128, nIter);

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

  const size_t dataBufferSize = 4096;
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 256,
      .pipelineDepth = 4,
  };

  TransportTestHelper helper(globalRank, numRanks, localRank, config);
  auto p2p = helper.getDevice();

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

  const size_t dataBufferSize = 512 * 1024; // 512KB staging buffer
  const size_t nbytesPerSend = 256 * 1024; // 256KB per send
  const int numSends = 4;
  const size_t totalBytes = nbytesPerSend * numSends;

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 1024,
      .pipelineDepth = 4,
  };

  TransportTestHelper helper(globalRank, numRanks, localRank, config);
  auto p2p = helper.getDevice();

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

  if (helper.globalRank() == 0) {
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

  XLOGF(INFO, "Rank {}: MultiSendInKernel test completed", helper.globalRank());
}

// =============================================================================
// Multiple Recvs in Single Kernel Test
// =============================================================================

TEST_F(P2pNvlTransportTestFixture, MultiRecvInKernel) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const size_t dataBufferSize = 512 * 1024; // 512KB staging buffer
  const size_t nbytesPerRecv = 128 * 1024; // 128KB per recv
  const int numRecvs = 8;
  const size_t totalBytes = nbytesPerRecv * numRecvs;

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 512,
      .pipelineDepth = 4,
  };

  TransportTestHelper helper(globalRank, numRanks, localRank, config);
  auto p2p = helper.getDevice();

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

  if (helper.globalRank() == 0) {
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

  XLOGF(INFO, "Rank {}: MultiRecvInKernel test completed", helper.globalRank());
}

// =============================================================================
// Simultaneous Send+Recv in Single Kernel Test
// =============================================================================

TEST_F(P2pNvlTransportTestFixture, SimultaneousSendRecvInKernel) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const size_t dataBufferSize = 1024 * 1024; // 1MB staging buffer
  const size_t nbytes = 2 * 1024 * 1024; // 2MB transfer each direction

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 1024,
      .pipelineDepth = 4,
  };

  TransportTestHelper helper(globalRank, numRanks, localRank, config);
  auto p2p = helper.getDevice();

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
  const int expectedRecvValue = 300 + helper.peerRank();

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
  if (helper.globalRank() == 0) {
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
      INFO,
      "Rank {}: SimultaneousSendRecvInKernel test completed",
      helper.globalRank());
}

// =============================================================================
// Parameterized Test Fixture for Weighted Partition Send/Recv
// =============================================================================
// Tests unequal send/recv partitioning with weighted splits

struct WeightedPartitionParams {
  uint32_t sendWeight;
  uint32_t recvWeight;
  std::string name;
};

class WeightedPartitionTestFixture
    : public MpiBaseTestFixture,
      public ::testing::WithParamInterface<WeightedPartitionParams> {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }
};

TEST_P(WeightedPartitionTestFixture, SendRecv) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const auto& params = GetParam();
  const size_t nbytes = 2 * 1024 * 1024; // 2MB
  const int numBlocks = 4;
  const int blockSize = 128;

  const size_t dataBufferSize = 1024 * 1024; // 1MB staging buffer
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 1024,
      .pipelineDepth = 4,
  };

  TransportTestHelper helper(globalRank, numRanks, localRank, config);
  auto p2p = helper.getDevice();

  const size_t numInts = nbytes / sizeof(int);

  DeviceBuffer sendBuffer(nbytes);
  DeviceBuffer recvBuffer(nbytes);

  auto send_d = static_cast<int*>(sendBuffer.get());
  auto recv_d = static_cast<int*>(recvBuffer.get());

  const int sendValue = 400 + globalRank;
  const int expectedRecvValue = 400 + helper.peerRank();

  test::fillBuffer(send_d, sendValue, numInts);
  CUDACHECK_TEST(cudaMemset(recv_d, 0, nbytes));
  CUDACHECK_TEST(cudaDeviceSynchronize());

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Rank 0 sends then recvs, rank 1 recvs then sends
  if (helper.globalRank() == 0) {
    test::testWeightedSendRecv(
        p2p,
        send_d,
        recv_d,
        nbytes,
        numBlocks,
        blockSize,
        params.sendWeight,
        params.recvWeight);
  } else {
    test::testWeightedRecvSend(
        p2p,
        recv_d,
        send_d,
        nbytes,
        numBlocks,
        blockSize,
        params.sendWeight,
        params.recvWeight);
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
      INFO,
      "Rank {}: Weighted partition test '{}' completed",
      globalRank,
      params.name);
}

std::string weightedPartitionParamName(
    const ::testing::TestParamInfo<WeightedPartitionParams>& info) {
  return info.param.name;
}

INSTANTIATE_TEST_SUITE_P(
    WeightedPartitionVariations,
    WeightedPartitionTestFixture,
    ::testing::Values(
        WeightedPartitionParams{
            .sendWeight = 3,
            .recvWeight = 1,
            .name = "Send3_Recv1"},
        WeightedPartitionParams{
            .sendWeight = 1,
            .recvWeight = 3,
            .name = "Send1_Recv3"},
        // Extreme case: 1:99 split - tests that at least 1 warp is assigned to
        // send
        WeightedPartitionParams{
            .sendWeight = 1,
            .recvWeight = 99,
            .name = "Send1_Recv99"}),
    weightedPartitionParamName);

// =============================================================================
// Parameterized Test Fixture for Pipeline Depth Variation
// =============================================================================
// Test different pipelineDepth values to verify pipelining works correctly:
// - pipelineDepth = 1 (no pipelining, sequential)
// - pipelineDepth = 2 (minimal pipelining)
// - pipelineDepth = 4 (default)
// - pipelineDepth = 8 (deep pipelining)

struct PipelineDepthParams {
  size_t pipelineDepth;
  size_t nbytes;
  size_t dataBufferSize;
  size_t chunkSize;
  std::string name;
};

class PipelineDepthTestFixture
    : public MpiBaseTestFixture,
      public ::testing::WithParamInterface<PipelineDepthParams> {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }
};

TEST_P(PipelineDepthTestFixture, SendRecv) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const auto& params = GetParam();
  XLOGF(
      INFO,
      "Running pipeline depth test: {} (pipelineDepth={}, nbytes={}, bufferSize={}, chunkSize={})",
      params.name,
      params.pipelineDepth,
      params.nbytes,
      params.dataBufferSize,
      params.chunkSize);

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = params.dataBufferSize,
      .chunkSize = params.chunkSize,
      .pipelineDepth = params.pipelineDepth,
  };

  // Calculate expected number of steps to verify pipelining
  const size_t totalSteps =
      (params.nbytes + params.dataBufferSize - 1) / params.dataBufferSize;
  XLOGF(
      INFO,
      "Rank {}: Transfer will use {} steps with pipeline depth {}",
      globalRank,
      totalSteps,
      params.pipelineDepth);

  TransportTestHelper helper(globalRank, numRanks, localRank, config);
  runBasicSendRecvTest(helper, params.nbytes, 4, 128);

  XLOGF(
      INFO,
      "Rank {}: Pipeline depth test '{}' completed",
      globalRank,
      params.name);
}

std::string pipelineDepthParamName(
    const ::testing::TestParamInfo<PipelineDepthParams>& info) {
  return info.param.name;
}

INSTANTIATE_TEST_SUITE_P(
    PipelineDepthVariations,
    PipelineDepthTestFixture,
    ::testing::Values(
        // pipelineDepth=1: No pipelining, sequential execution
        // 4MB transfer with 512KB buffer = 8 steps, all executed sequentially
        PipelineDepthParams{
            .pipelineDepth = 1,
            .nbytes = 4 * 1024 * 1024,
            .dataBufferSize = 512 * 1024,
            .chunkSize = 1024,
            .name = "Depth1_Sequential_8Steps"},
        // pipelineDepth=2: Minimal pipelining
        // 4MB transfer with 512KB buffer = 8 steps, using 2 slots
        PipelineDepthParams{
            .pipelineDepth = 2,
            .nbytes = 4 * 1024 * 1024,
            .dataBufferSize = 512 * 1024,
            .chunkSize = 1024,
            .name = "Depth2_MinimalPipeline_8Steps"},
        // pipelineDepth=4: Default pipelining
        // 4MB transfer with 512KB buffer = 8 steps, using 4 slots
        PipelineDepthParams{
            .pipelineDepth = 4,
            .nbytes = 4 * 1024 * 1024,
            .dataBufferSize = 512 * 1024,
            .chunkSize = 1024,
            .name = "Depth4_DefaultPipeline_8Steps"},
        // pipelineDepth=8: Deep pipelining
        // 4MB transfer with 512KB buffer = 8 steps, using all 8 slots
        PipelineDepthParams{
            .pipelineDepth = 8,
            .nbytes = 4 * 1024 * 1024,
            .dataBufferSize = 512 * 1024,
            .chunkSize = 1024,
            .name = "Depth8_DeepPipeline_8Steps"},
        // pipelineDepth=8 with more steps than depth
        // 8MB transfer with 512KB buffer = 16 steps, using 8 slots (slot reuse)
        PipelineDepthParams{
            .pipelineDepth = 8,
            .nbytes = 8 * 1024 * 1024,
            .dataBufferSize = 512 * 1024,
            .chunkSize = 1024,
            .name = "Depth8_SlotReuse_16Steps"}),
    pipelineDepthParamName);

// =============================================================================
// Parameterized Test Fixture for Pipeline Slot Reuse
// =============================================================================
// Tests that pipeline slots are correctly reused when totalSteps >
// pipelineDepth:
// - Verifies stepId % pipelineDepth indexing works correctly
// - Verifies state buffer is properly reset when slots are reused
// - Ensures data integrity across multiple reuses of the same slot

struct PipelineSaturationParams {
  size_t pipelineDepth;
  size_t totalSteps;
  size_t chunkSize;
  std::string name;
};

class PipelineSaturationTestFixture
    : public MpiBaseTestFixture,
      public ::testing::WithParamInterface<PipelineSaturationParams> {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }
};

TEST_P(PipelineSaturationTestFixture, SendRecv) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const auto& params = GetParam();
  // Calculate buffer size and total bytes to achieve desired number of steps
  const size_t dataBufferSize = 256 * 1024; // 256KB per step
  const size_t nbytes = params.totalSteps * dataBufferSize;

  XLOGF(
      INFO,
      "Running pipeline saturation test: {} (pipelineDepth={}, steps={}, nbytes={})",
      params.name,
      params.pipelineDepth,
      params.totalSteps,
      nbytes);

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = params.chunkSize,
      .pipelineDepth = params.pipelineDepth,
  };

  TransportTestHelper helper(globalRank, numRanks, localRank, config);
  runBasicSendRecvTest(helper, nbytes, 4, 128);

  XLOGF(
      INFO,
      "Rank {}: Pipeline saturation test '{}' completed",
      globalRank,
      params.name);
}

std::string pipelineSaturationParamName(
    const ::testing::TestParamInfo<PipelineSaturationParams>& info) {
  return info.param.name;
}

INSTANTIATE_TEST_SUITE_P(
    PipelineSlotReuseVariations,
    PipelineSaturationTestFixture,
    ::testing::Values(
        // pipelineDepth=2, 10 steps: each slot used 5 times (steps 0,2,4,6,8
        // and 1,3,5,7,9)
        PipelineSaturationParams{
            .pipelineDepth = 2,
            .totalSteps = 10,
            .chunkSize = 1024,
            .name = "Depth2_10Steps_5xReuse"},
        // pipelineDepth=2, 16 steps: each slot used 8 times
        PipelineSaturationParams{
            .pipelineDepth = 2,
            .totalSteps = 16,
            .chunkSize = 1024,
            .name = "Depth2_16Steps_8xReuse"},
        // pipelineDepth=3, 12 steps: each slot used 4 times
        PipelineSaturationParams{
            .pipelineDepth = 3,
            .totalSteps = 12,
            .chunkSize = 1024,
            .name = "Depth3_12Steps_4xReuse"},
        // pipelineDepth=4, 20 steps: each slot used 5 times
        PipelineSaturationParams{
            .pipelineDepth = 4,
            .totalSteps = 20,
            .chunkSize = 512,
            .name = "Depth4_20Steps_5xReuse"}),
    pipelineSaturationParamName);

// =============================================================================
// Parameterized Test Fixture for Chunk Count Edge Cases
// =============================================================================
// Test edge cases in chunk distribution:
// - numChunks < numWarps (some warps have no work)
// - numChunks = 1 (single chunk)
// - numChunks = numWarps (exactly 1 chunk per warp)
// - Very small transfer (< chunkSize)

struct ChunkCountEdgeCaseParams {
  size_t nbytes;
  size_t dataBufferSize;
  size_t chunkSize;
  int numBlocks;
  int blockSize;
  std::string name;
};

class ChunkCountEdgeCaseTestFixture
    : public MpiBaseTestFixture,
      public ::testing::WithParamInterface<ChunkCountEdgeCaseParams> {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }
};

TEST_P(ChunkCountEdgeCaseTestFixture, SendRecv) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const auto& params = GetParam();

  // Calculate chunk distribution info
  const size_t chunksPerStep =
      (params.dataBufferSize + params.chunkSize - 1) / params.chunkSize;
  const int numWarps =
      (params.numBlocks * params.blockSize + 31) / 32; // Approximate

  XLOGF(
      INFO,
      "Running chunk edge case test: {} (nbytes={}, chunkSize={}, ~{} chunks, ~{} warps)",
      params.name,
      params.nbytes,
      params.chunkSize,
      chunksPerStep,
      numWarps);

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = params.dataBufferSize,
      .chunkSize = params.chunkSize,
      .pipelineDepth = 4,
  };

  TransportTestHelper helper(globalRank, numRanks, localRank, config);
  runBasicSendRecvTest(
      helper, params.nbytes, params.numBlocks, params.blockSize);

  XLOGF(
      INFO,
      "Rank {}: Chunk edge case test '{}' completed",
      globalRank,
      params.name);
}

std::string chunkCountEdgeCaseParamName(
    const ::testing::TestParamInfo<ChunkCountEdgeCaseParams>& info) {
  return info.param.name;
}

INSTANTIATE_TEST_SUITE_P(
    ChunkCountEdgeCases,
    ChunkCountEdgeCaseTestFixture,
    ::testing::Values(
        // numChunks < numWarps: 4 chunks with 64 warps (8 blocks × 256 threads)
        // Many warps will have no work
        ChunkCountEdgeCaseParams{
            .nbytes = 4 * 1024, // 4KB
            .dataBufferSize = 4 * 1024,
            .chunkSize = 1024, // 4 chunks
            .numBlocks = 8,
            .blockSize = 256, // 64 warps
            .name = "FewChunks_4Chunks_64Warps"},
        // numChunks = 1: Single chunk transfer
        ChunkCountEdgeCaseParams{
            .nbytes = 512, // 512 bytes
            .dataBufferSize = 1024,
            .chunkSize = 1024, // 1 chunk (transfer < chunkSize)
            .numBlocks = 4,
            .blockSize = 128,
            .name = "SingleChunk_512Bytes"},
        // numChunks = 1 with larger chunk
        ChunkCountEdgeCaseParams{
            .nbytes = 4 * 1024, // 4KB
            .dataBufferSize = 4 * 1024,
            .chunkSize = 4 * 1024, // 1 chunk
            .numBlocks = 4,
            .blockSize = 128,
            .name = "SingleChunk_4KB"},
        // numChunks = numWarps: Exactly 1 chunk per warp
        // 16 chunks with 16 warps (4 blocks × 128 threads = 16 warps)
        ChunkCountEdgeCaseParams{
            .nbytes = 16 * 1024, // 16KB
            .dataBufferSize = 16 * 1024,
            .chunkSize = 1024, // 16 chunks
            .numBlocks = 4,
            .blockSize = 128, // 16 warps
            .name = "ExactMatch_16Chunks_16Warps"},
        // Very small transfer (< chunkSize)
        ChunkCountEdgeCaseParams{
            .nbytes = 64, // 64 bytes (much smaller than chunk)
            .dataBufferSize = 1024,
            .chunkSize = 256,
            .numBlocks = 2,
            .blockSize = 64,
            .name = "VerySmall_64Bytes"},
        // Another very small transfer
        ChunkCountEdgeCaseParams{
            .nbytes = 128, // 128 bytes
            .dataBufferSize = 1024,
            .chunkSize = 512,
            .numBlocks = 2,
            .blockSize = 64,
            .name = "VerySmall_128Bytes"},
        // Edge case: nbytes not aligned to chunk or vector size
        ChunkCountEdgeCaseParams{
            .nbytes = 100, // Non-aligned size
            .dataBufferSize = 1024,
            .chunkSize = 64,
            .numBlocks = 2,
            .blockSize = 64,
            .name = "NonAligned_100Bytes"}),
    chunkCountEdgeCaseParamName);

// =============================================================================
// Parameterized Test Fixture for Large Transfers (Stress Test)
// =============================================================================
// Stress test with large transfers:
// - 64MB, 128MB, 256MB transfers
// - Exercises full pipeline depth, many steps, many chunks

struct LargeTransferParams {
  size_t nbytes;
  size_t dataBufferSize;
  size_t chunkSize;
  size_t pipelineDepth;
  std::string name;
};

class LargeTransferTestFixture
    : public MpiBaseTestFixture,
      public ::testing::WithParamInterface<LargeTransferParams> {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }
};

TEST_P(LargeTransferTestFixture, SendRecv) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const auto& params = GetParam();

  // Calculate transfer statistics
  const size_t totalSteps =
      (params.nbytes + params.dataBufferSize - 1) / params.dataBufferSize;
  const size_t chunksPerStep =
      (params.dataBufferSize + params.chunkSize - 1) / params.chunkSize;

  XLOGF(
      INFO,
      "Running large transfer test: {} (nbytes={}MB, {} steps, {} chunks/step)",
      params.name,
      params.nbytes / (1024 * 1024),
      totalSteps,
      chunksPerStep);

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = params.dataBufferSize,
      .chunkSize = params.chunkSize,
      .pipelineDepth = params.pipelineDepth,
  };

  TransportTestHelper helper(globalRank, numRanks, localRank, config);
  runBasicSendRecvTest(helper, params.nbytes, 8, 256);

  XLOGF(
      INFO,
      "Rank {}: Large transfer test '{}' completed",
      globalRank,
      params.name);
}

std::string largeTransferParamName(
    const ::testing::TestParamInfo<LargeTransferParams>& info) {
  return info.param.name;
}

INSTANTIATE_TEST_SUITE_P(
    LargeTransferVariations,
    LargeTransferTestFixture,
    ::testing::Values(
        // 64MB transfer with 8MB buffer = 8 steps
        LargeTransferParams{
            .nbytes = 64 * 1024 * 1024,
            .dataBufferSize = 8 * 1024 * 1024,
            .chunkSize = 4 * 1024,
            .pipelineDepth = 4,
            .name = "Large_64MB_8MBBuffer"},
        // 128MB transfer with 8MB buffer = 16 steps
        LargeTransferParams{
            .nbytes = 128 * 1024 * 1024,
            .dataBufferSize = 8 * 1024 * 1024,
            .chunkSize = 4 * 1024,
            .pipelineDepth = 4,
            .name = "Large_128MB_8MBBuffer"},
        // 256MB transfer with 8MB buffer = 32 steps
        LargeTransferParams{
            .nbytes = 256 * 1024 * 1024,
            .dataBufferSize = 8 * 1024 * 1024,
            .chunkSize = 4 * 1024,
            .pipelineDepth = 4,
            .name = "Large_256MB_8MBBuffer"},
        // 64MB transfer with smaller buffer = more steps
        LargeTransferParams{
            .nbytes = 64 * 1024 * 1024,
            .dataBufferSize = 4 * 1024 * 1024,
            .chunkSize = 2 * 1024,
            .pipelineDepth = 8,
            .name = "Large_64MB_4MBBuffer_DeepPipeline"},
        // 128MB transfer with deep pipeline
        LargeTransferParams{
            .nbytes = 128 * 1024 * 1024,
            .dataBufferSize = 4 * 1024 * 1024,
            .chunkSize = 2 * 1024,
            .pipelineDepth = 8,
            .name = "Large_128MB_4MBBuffer_DeepPipeline"}),
    largeTransferParamName);

// =============================================================================
// Parameterized Test Fixture for Asymmetric Group Configurations
// =============================================================================
// Tests that sender and receiver can use different thread group configurations
// This validates that the protocol works across asymmetric kernel launches.

struct AsymmetricGroupParams {
  test::GroupType senderGroupType;
  int senderNumBlocks;
  int senderBlockSize;
  test::GroupType receiverGroupType;
  int receiverNumBlocks;
  int receiverBlockSize;
  std::string name;
};

class AsymmetricGroupTestFixture
    : public MpiBaseTestFixture,
      public ::testing::WithParamInterface<AsymmetricGroupParams> {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }
};

TEST_P(AsymmetricGroupTestFixture, SendRecv) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const auto& params = GetParam();
  XLOGF(
      INFO,
      "Running asymmetric group test: {} (sender: {} {}x{}, receiver: {} {}x{})",
      params.name,
      params.senderGroupType == test::GroupType::WARP ? "WARP" : "BLOCK",
      params.senderNumBlocks,
      params.senderBlockSize,
      params.receiverGroupType == test::GroupType::WARP ? "WARP" : "BLOCK",
      params.receiverNumBlocks,
      params.receiverBlockSize);

  const size_t dataBufferSize = 1024 * 1024; // 1MB staging buffer
  const size_t nbytes = 4 * 1024 * 1024; // 4MB total transfer
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 1024,
      .pipelineDepth = 4,
  };

  TransportTestHelper helper(globalRank, numRanks, localRank, config);
  auto p2p = helper.getDevice();

  const size_t numInts = nbytes / sizeof(int);

  DeviceBuffer srcBuffer(nbytes);
  DeviceBuffer dstBuffer(nbytes);

  auto src_d = static_cast<int*>(srcBuffer.get());
  auto dst_d = static_cast<int*>(dstBuffer.get());

  if (globalRank == 0) {
    // Sender
    test::fillBuffer(src_d, 42, numInts);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    test::testSend(
        p2p,
        src_d,
        nbytes,
        params.senderNumBlocks,
        params.senderBlockSize,
        params.senderGroupType);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  } else {
    // Receiver
    CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    test::testRecv(
        p2p,
        dst_d,
        nbytes,
        params.receiverNumBlocks,
        params.receiverBlockSize,
        params.receiverGroupType);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Verify received data
    std::vector<int> hostBuffer(numInts);
    CUDACHECK_TEST(
        cudaMemcpy(hostBuffer.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < numInts; i++) {
      EXPECT_EQ(hostBuffer[i], 42)
          << "Rank " << globalRank << ": Mismatch at index " << i
          << ": expected 42, got " << hostBuffer[i];
      if (hostBuffer[i] != 42) {
        break;
      }
    }
  }

  XLOGF(
      INFO,
      "Rank {}: Asymmetric group test '{}' completed",
      globalRank,
      params.name);
}

std::string asymmetricGroupParamName(
    const ::testing::TestParamInfo<AsymmetricGroupParams>& info) {
  return info.param.name;
}

INSTANTIATE_TEST_SUITE_P(
    AsymmetricGroupVariations,
    AsymmetricGroupTestFixture,
    ::testing::Values(
        // Sender uses WARP groups, receiver uses BLOCK groups
        AsymmetricGroupParams{
            .senderGroupType = test::GroupType::WARP,
            .senderNumBlocks = 4,
            .senderBlockSize = 128,
            .receiverGroupType = test::GroupType::BLOCK,
            .receiverNumBlocks = 4,
            .receiverBlockSize = 128,
            .name = "SenderWarp_ReceiverBlock"},
        // Sender uses BLOCK groups, receiver uses WARP groups
        AsymmetricGroupParams{
            .senderGroupType = test::GroupType::BLOCK,
            .senderNumBlocks = 4,
            .senderBlockSize = 128,
            .receiverGroupType = test::GroupType::WARP,
            .receiverNumBlocks = 4,
            .receiverBlockSize = 128,
            .name = "SenderBlock_ReceiverWarp"},
        // Different block configurations
        AsymmetricGroupParams{
            .senderGroupType = test::GroupType::WARP,
            .senderNumBlocks = 8,
            .senderBlockSize = 256,
            .receiverGroupType = test::GroupType::BLOCK,
            .receiverNumBlocks = 2,
            .receiverBlockSize = 512,
            .name = "SenderWarp8x256_ReceiverBlock2x512"},
        // Same group type but different configurations
        AsymmetricGroupParams{
            .senderGroupType = test::GroupType::WARP,
            .senderNumBlocks = 2,
            .senderBlockSize = 64,
            .receiverGroupType = test::GroupType::WARP,
            .receiverNumBlocks = 8,
            .receiverBlockSize = 256,
            .name = "SenderWarp2x64_ReceiverWarp8x256"}),
    asymmetricGroupParamName);

// =============================================================================
// P2pNvlTransportDevice::put() Tests
// =============================================================================
// Tests for the one-sided put() API that writes directly to peer memory
// via NVLink without using staging buffers.

// Helper to run a write() test with verification
void runPutTest(
    int globalRank,
    P2pNvlTransportDevice& p2p,
    char* localSrc,
    char* remoteDst,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    const std::string& testName) {
  const char testValue = 0x42;
  const uint64_t signal_id = 0;

  if (globalRank == 0) {
    // Rank 0: Initialize source buffer and call write()
    CUDACHECK_TEST(cudaMemset(localSrc, testValue, nbytes));
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Write data to peer's buffer
    test::testPutWithSignal(
        p2p, remoteDst, localSrc, signal_id, nbytes, numBlocks, blockSize);

    CUDACHECK_TEST(cudaDeviceSynchronize());
  } else {
    // Rank 1: Clear destination buffer and verify after write()
    CUDACHECK_TEST(cudaMemset(localSrc, 0, nbytes));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    test::testWait(p2p, CmpOp::CMP_GE, signal_id, nbytes, numBlocks, blockSize);

    CUDACHECK_TEST(cudaDeviceSynchronize());

    // Verify the data was written correctly
    std::vector<char> hostBuffer(nbytes);
    CUDACHECK_TEST(cudaMemcpy(
        hostBuffer.data(), localSrc, nbytes, cudaMemcpyDeviceToHost));

    int errorCount = 0;
    for (size_t i = 0; i < nbytes; i++) {
      if (hostBuffer[i] != testValue) {
        ++errorCount;
        if (errorCount <= 5) {
          XLOGF(
              ERR,
              "{}: Mismatch at index {}: expected 0x{:02x}, got 0x{:02x}",
              testName,
              i,
              static_cast<unsigned char>(testValue),
              static_cast<unsigned char>(hostBuffer[i]));
        }
      }
    }

    ASSERT_EQ(errorCount, 0) << testName << " found " << errorCount
                             << " errors out of " << nbytes << " bytes";
  }
}

// Basic write() test with aligned pointers
TEST_F(P2pNvlTransportTestFixture, PutBasic) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const size_t nbytes = 1024 * 1024; // 1MB
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = nbytes,
      .chunkSize = 1,
      .pipelineDepth = 1,
  };

  TransportTestHelper helper(globalRank, numRanks, localRank, config);
  auto p2p = helper.getDevice();

  // Get remote destination (peer's local data buffer)
  char* localSrc = p2p.getLocalState().dataBuffer;
  char* remoteDst = p2p.getRemoteState().dataBuffer;

  runPutTest(globalRank, p2p, localSrc, remoteDst, nbytes, 4, 128, "PutBasic");

  XLOGF(INFO, "Rank {}: PutBasic test completed", globalRank);
}

// Parameterized test for write() with various transfer sizes
struct PutTransferSizeParams {
  size_t nbytes;
  std::string name;
};

class PutTransferSizeTestFixture
    : public MpiBaseTestFixture,
      public ::testing::WithParamInterface<PutTransferSizeParams> {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }
};

TEST_P(PutTransferSizeTestFixture, Put) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const auto& params = GetParam();
  XLOGF(
      INFO,
      "Running write transfer size test: {} (nbytes={})",
      params.name,
      params.nbytes);

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = params.nbytes,
      .chunkSize = 1,
      .pipelineDepth = 1,
  };

  TransportTestHelper helper(globalRank, numRanks, localRank, config);
  auto p2p = helper.getDevice();

  char* localSrc = p2p.getLocalState().dataBuffer;
  char* remoteDst = p2p.getRemoteState().dataBuffer;

  runPutTest(
      globalRank, p2p, localSrc, remoteDst, params.nbytes, 4, 128, params.name);

  XLOGF(
      INFO,
      "Rank {}: Put transfer size test '{}' completed",
      globalRank,
      params.name);
}

std::string putTransferSizeParamName(
    const ::testing::TestParamInfo<PutTransferSizeParams>& info) {
  return info.param.name;
}

INSTANTIATE_TEST_SUITE_P(
    PutTransferSizeVariations,
    PutTransferSizeTestFixture,
    ::testing::Values(
        // Small sizes (smaller than vector size of 16 bytes)
        PutTransferSizeParams{.nbytes = 1, .name = "Put_1Byte"},
        PutTransferSizeParams{.nbytes = 7, .name = "Put_7Bytes"},
        PutTransferSizeParams{.nbytes = 15, .name = "Put_15Bytes"},
        // Around vector size boundary
        PutTransferSizeParams{.nbytes = 16, .name = "Put_16Bytes"},
        PutTransferSizeParams{.nbytes = 17, .name = "Put_17Bytes"},
        PutTransferSizeParams{.nbytes = 31, .name = "Put_31Bytes"},
        PutTransferSizeParams{.nbytes = 32, .name = "Put_32Bytes"},
        // Non-aligned sizes
        PutTransferSizeParams{.nbytes = 100, .name = "Put_100Bytes"},
        PutTransferSizeParams{.nbytes = 1000, .name = "Put_1000Bytes"},
        PutTransferSizeParams{.nbytes = 4097, .name = "Put_4097Bytes"},
        // Aligned sizes
        PutTransferSizeParams{.nbytes = 1024, .name = "Put_1KB"},
        PutTransferSizeParams{.nbytes = 64 * 1024, .name = "Put_64KB"},
        PutTransferSizeParams{.nbytes = 256 * 1024, .name = "Put_256KB"},
        PutTransferSizeParams{.nbytes = 1024 * 1024, .name = "Put_1MB"},
        // Large sizes
        PutTransferSizeParams{.nbytes = 4 * 1024 * 1024, .name = "Put_4MB"},
        PutTransferSizeParams{.nbytes = 16 * 1024 * 1024, .name = "Put_16MB"}),
    putTransferSizeParamName);

// Parameterized test for write() with unaligned pointers
struct PutUnalignedParams {
  size_t srcOffset; // Offset from 16-byte alignment for source
  size_t dstOffset; // Offset from 16-byte alignment for destination
  size_t nbytes;
  std::string name;
};

class PutUnalignedTestFixture
    : public MpiBaseTestFixture,
      public ::testing::WithParamInterface<PutUnalignedParams> {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }
};

TEST_P(PutUnalignedTestFixture, Put) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const auto& params = GetParam();
  XLOGF(
      INFO,
      "Running write unaligned test: {} (srcOffset={}, dstOffset={}, nbytes={})",
      params.name,
      params.srcOffset,
      params.dstOffset,
      params.nbytes);

  // Allocate larger staging buffers to accommodate offsets
  const size_t dataBufferSize =
      params.nbytes + std::max(params.srcOffset, params.dstOffset);
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 1024,
      .pipelineDepth = 4,
  };

  TransportTestHelper helper(globalRank, numRanks, localRank, config);
  auto p2p = helper.getDevice();

  // Get remote and local destination with offset applied
  char* localSrc = p2p.getLocalState().dataBuffer;
  char* remoteDst = p2p.getRemoteState().dataBuffer;
  if (globalRank == 0) {
    localSrc += params.srcOffset;
    remoteDst += params.dstOffset;
  } else {
    localSrc += params.dstOffset;
    remoteDst += params.srcOffset;
  }

  runPutTest(
      globalRank, p2p, localSrc, remoteDst, params.nbytes, 4, 128, params.name);

  XLOGF(
      INFO,
      "Rank {}: Put unaligned test '{}' completed",
      globalRank,
      params.name);
}

std::string putUnalignedParamName(
    const ::testing::TestParamInfo<PutUnalignedParams>& info) {
  return info.param.name;
}

INSTANTIATE_TEST_SUITE_P(
    PutUnalignedVariations,
    PutUnalignedTestFixture,
    ::testing::Values(
        // Same misalignment (can use vectorized copy after aligning)
        PutUnalignedParams{
            .srcOffset = 1,
            .dstOffset = 1,
            .nbytes = 1024,
            .name = "SameMisalign_1"},
        PutUnalignedParams{
            .srcOffset = 7,
            .dstOffset = 7,
            .nbytes = 1024,
            .name = "SameMisalign_7"},
        PutUnalignedParams{
            .srcOffset = 8,
            .dstOffset = 8,
            .nbytes = 1024,
            .name = "SameMisalign_8"},
        PutUnalignedParams{
            .srcOffset = 13,
            .dstOffset = 13,
            .nbytes = 1024,
            .name = "SameMisalign_13"},
        PutUnalignedParams{
            .srcOffset = 15,
            .dstOffset = 15,
            .nbytes = 1024,
            .name = "SameMisalign_15"},
        // Different misalignment (fallback to byte-by-byte)
        PutUnalignedParams{
            .srcOffset = 1,
            .dstOffset = 3,
            .nbytes = 1024,
            .name = "DiffMisalign_1_3"},
        PutUnalignedParams{
            .srcOffset = 0,
            .dstOffset = 7,
            .nbytes = 1024,
            .name = "DiffMisalign_0_7"},
        PutUnalignedParams{
            .srcOffset = 5,
            .dstOffset = 0,
            .nbytes = 1024,
            .name = "DiffMisalign_5_0"},
        PutUnalignedParams{
            .srcOffset = 4,
            .dstOffset = 8,
            .nbytes = 1024,
            .name = "DiffMisalign_4_8"},
        // Larger transfers with misalignment
        PutUnalignedParams{
            .srcOffset = 3,
            .dstOffset = 3,
            .nbytes = 64 * 1024,
            .name = "SameMisalign_3_64KB"},
        PutUnalignedParams{
            .srcOffset = 5,
            .dstOffset = 11,
            .nbytes = 64 * 1024,
            .name = "DiffMisalign_5_11_64KB"},
        // Small transfers with misalignment
        PutUnalignedParams{
            .srcOffset = 7,
            .dstOffset = 7,
            .nbytes = 100,
            .name = "SameMisalign_7_100Bytes"},
        PutUnalignedParams{
            .srcOffset = 1,
            .dstOffset = 9,
            .nbytes = 100,
            .name = "DiffMisalign_1_9_100Bytes"}),
    putUnalignedParamName);

} // namespace comms::pipes::tests

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
