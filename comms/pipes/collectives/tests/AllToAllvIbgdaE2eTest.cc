// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>
#include <mpi.h>

#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/MultiPeerIbgdaTransportSetup.h"
#include "comms/pipes/MultipeerIbgdaTransport.h"
#include "comms/pipes/P2pIbgdaTransportState.h"
#include "comms/pipes/P2pSelfTransportDevice.cuh"
#include "comms/pipes/Transport.cuh"
#include "comms/pipes/collectives/AllToAllv.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::DeviceBuffer;
using meta::comms::MpiBaseTestFixture;

namespace comms::pipes {

class AllToAllvIbgdaE2eTestFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }

  // Collectively create IBGDA transport. All ranks must agree on success
  // or failure to avoid MPI collective divergence (one rank skipping
  // while others enter exchangeBuffers/MPI_Barrier → deadlock).
  std::unique_ptr<MultipeerIbgdaTransport> createTransport() {
    std::unique_ptr<MultipeerIbgdaTransport> transport;
    try {
      MultipeerIbgdaTransportConfig config{
          .cudaDevice = localRank,
      };
      auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
      transport = std::make_unique<MultipeerIbgdaTransport>(
          globalRank, numRanks, bootstrap, config);
      transport->exchange();
    } catch (const std::exception& e) {
      XLOGF(
          ERR, "Rank {}: transport creation failed: {}", globalRank, e.what());
      transport.reset();
    }

    // Collective agreement: all ranks must succeed or all must skip.
    int localOk = transport ? 1 : 0;
    int globalOk = 0;
    MPI_Allreduce(&localOk, &globalOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (globalOk == 0) {
      transport.reset();
      return nullptr;
    }
    return transport;
  }

  bool shouldSkip(const std::unique_ptr<MultipeerIbgdaTransport>& transport) {
    return transport == nullptr;
  }

  // Build a device Transport array with SELF for myRank and IBGDA for all
  // peers. Uses buildP2pTransportDevice() to create fully-formed devices
  // with embedded staging state (required for send()/recv() APIs).
  Transport* buildTransportsArray(
      MultipeerIbgdaTransport& ibgdaTransport,
      MultiPeerIbgdaTransportSetup& setup,
      int myRank,
      int nRanks) {
    const size_t arrayBytes = nRanks * sizeof(Transport);
    auto* h_transports = static_cast<Transport*>(
        std::aligned_alloc(alignof(Transport), arrayBytes));
    const auto& hostStates = setup.getHostPeerStates();
    uint64_t* iterCounter = setup.getIterationCounter();
    for (int r = 0; r < nRanks; ++r) {
      if (r == myRank) {
        new (&h_transports[r]) Transport(P2pSelfTransportDevice{});
      } else {
        int maxCh = hostStates[r].maxChannelsPerPeer;
        P2pIbgdaTransportDevice* devPtr =
            ibgdaTransport.buildP2pTransportDevice(
                r,
                hostStates[r],
                &iterCounter[r * maxCh * 2],
                &iterCounter[r * maxCh * 2 + maxCh]);
        new (&h_transports[r]) Transport(devPtr);
      }
    }
    Transport* d_transports = nullptr;
    cudaMalloc(&d_transports, arrayBytes);
    cudaMemcpy(d_transports, h_transports, arrayBytes, cudaMemcpyHostToDevice);
    for (int r = 0; r < nRanks; ++r) {
      h_transports[r].~Transport();
    }
    std::free(h_transports);
    return d_transports;
  }

  // Helper to run an equal-size alltoallv over IBGDA and verify correctness
  void runEqualSizeTest(
      MultipeerIbgdaTransport& ibgdaTransport,
      MultiPeerIbgdaTransportSetup& setup,
      size_t numIntsPerRank,
      int numBlocks = 4,
      int numThreads = 256) {
    const size_t totalInts = numIntsPerRank * numRanks;
    const size_t bufferSize = totalInts * sizeof(int32_t);
    const size_t perPeerBytes = numIntsPerRank * sizeof(int32_t);

    Transport* d_transports =
        buildTransportsArray(ibgdaTransport, setup, globalRank, numRanks);

    DeviceBuffer sendBuffer(bufferSize);
    DeviceBuffer recvBuffer(bufferSize);

    // Fill send buffer: rank R sending to peer P: R*1000 + P*100 + i
    std::vector<int32_t> h_send(totalInts);
    for (int peer = 0; peer < numRanks; peer++) {
      for (size_t i = 0; i < numIntsPerRank; i++) {
        h_send[peer * numIntsPerRank + i] =
            globalRank * 1000 + peer * 100 + static_cast<int32_t>(i);
      }
    }
    CUDACHECK_TEST(cudaMemcpy(
        sendBuffer.get(), h_send.data(), bufferSize, cudaMemcpyHostToDevice));

    // Initialize recv with -1
    std::vector<int32_t> h_recv_init(totalInts, -1);
    CUDACHECK_TEST(cudaMemcpy(
        recvBuffer.get(),
        h_recv_init.data(),
        bufferSize,
        cudaMemcpyHostToDevice));

    // Build ChunkInfo arrays on device
    std::vector<ChunkInfo> h_send_chunks, h_recv_chunks;
    for (int rank = 0; rank < numRanks; rank++) {
      size_t offset = rank * perPeerBytes;
      h_send_chunks.emplace_back(offset, perPeerBytes);
      h_recv_chunks.emplace_back(offset, perPeerBytes);
    }

    DeviceBuffer d_send_chunks(sizeof(ChunkInfo) * numRanks);
    DeviceBuffer d_recv_chunks(sizeof(ChunkInfo) * numRanks);
    CUDACHECK_TEST(cudaMemcpy(
        d_send_chunks.get(),
        h_send_chunks.data(),
        sizeof(ChunkInfo) * numRanks,
        cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemcpy(
        d_recv_chunks.get(),
        h_recv_chunks.data(),
        sizeof(ChunkInfo) * numRanks,
        cudaMemcpyHostToDevice));

    DeviceSpan<Transport> transports_span(
        d_transports, static_cast<uint32_t>(numRanks));
    DeviceSpan<ChunkInfo> send_chunk_infos(
        static_cast<ChunkInfo*>(d_send_chunks.get()), numRanks);
    DeviceSpan<ChunkInfo> recv_chunk_infos(
        static_cast<ChunkInfo*>(d_recv_chunks.get()), numRanks);

    MPI_Barrier(MPI_COMM_WORLD);

    all_to_allv(
        recvBuffer.get(),
        sendBuffer.get(),
        globalRank,
        transports_span,
        send_chunk_infos,
        recv_chunk_infos,
        std::chrono::milliseconds{30000},
        nullptr,
        numBlocks,
        numThreads,
        std::nullopt);

    CUDACHECK_TEST(cudaDeviceSynchronize());

    // Verify received data
    std::vector<int32_t> h_recv(totalInts);
    CUDACHECK_TEST(cudaMemcpy(
        h_recv.data(), recvBuffer.get(), bufferSize, cudaMemcpyDeviceToHost));

    int errorCount = 0;
    for (int peer = 0; peer < numRanks; peer++) {
      for (size_t i = 0; i < numIntsPerRank; i++) {
        int32_t expected =
            peer * 1000 + globalRank * 100 + static_cast<int32_t>(i);
        int32_t actual = h_recv[peer * numIntsPerRank + i];
        if (expected != actual) {
          errorCount++;
          if (errorCount <= 10) {
            XLOGF(
                ERR,
                "Rank {}: Error at peer {} pos {}: expected {}, got {}",
                globalRank,
                peer,
                i,
                expected,
                actual);
          }
        }
      }
    }

    EXPECT_EQ(errorCount, 0) << "Rank " << globalRank << " found " << errorCount
                             << " verification errors";

    MPI_Barrier(MPI_COMM_WORLD);
    cudaFree(d_transports);
  }
};

// =============================================================================
// E2E Tests: Full pipeline with IBGDA transport
// =============================================================================

TEST_F(AllToAllvIbgdaE2eTestFixture, FullPipeline_EqualSize_Small) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Need at least 2 ranks";
  }
  auto transport = createTransport();
  if (shouldSkip(transport)) {
    GTEST_SKIP() << "IBGDA transport not available (collective skip)";
  }

  MultiPeerIbgdaTransportSetupConfig setupConfig{
      .dataBufferSize = 4096,
      .chunkSize = 4096,
      .pipelineDepth = 4,
  };

  MultiPeerIbgdaTransportSetup setup(
      *transport, globalRank, numRanks, setupConfig);
  setup.exchangeBuffers();

  // 64 ints = 256 bytes per peer (fits in one pipeline step)
  runEqualSizeTest(*transport, setup, 64);
}

TEST_F(AllToAllvIbgdaE2eTestFixture, FullPipeline_EqualSize_LargeMessage) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Need at least 2 ranks";
  }
  auto transport = createTransport();
  if (shouldSkip(transport)) {
    GTEST_SKIP() << "IBGDA transport not available (collective skip)";
  }

  // 256KB staging buffer → 1MB message requires 4 pipeline steps
  MultiPeerIbgdaTransportSetupConfig setupConfig{
      .dataBufferSize = 256 * 1024,
      .chunkSize = 256 * 1024,
      .pipelineDepth = 4,
  };

  MultiPeerIbgdaTransportSetup setup(
      *transport, globalRank, numRanks, setupConfig);
  setup.exchangeBuffers();

  // 262144 ints = 1MB per peer
  runEqualSizeTest(*transport, setup, 262144);
}

TEST_F(AllToAllvIbgdaE2eTestFixture, RepeatedCalls) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Need at least 2 ranks";
  }
  auto transport = createTransport();
  if (shouldSkip(transport)) {
    GTEST_SKIP() << "IBGDA transport not available (collective skip)";
  }

  MultiPeerIbgdaTransportSetupConfig setupConfig{
      .dataBufferSize = 4096,
      .chunkSize = 4096,
      .pipelineDepth = 4,
  };

  MultiPeerIbgdaTransportSetup setup(
      *transport, globalRank, numRanks, setupConfig);
  setup.exchangeBuffers();

  // Run 5 iterations — tests monotonic signal counter correctness
  for (int iter = 0; iter < 5; iter++) {
    runEqualSizeTest(*transport, setup, 256);
  }
}

TEST_F(AllToAllvIbgdaE2eTestFixture, MixedSizes_AllPeers) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Need at least 2 ranks";
  }
  auto transport = createTransport();
  if (shouldSkip(transport)) {
    GTEST_SKIP() << "IBGDA transport not available (collective skip)";
  }

  MultiPeerIbgdaTransportSetupConfig setupConfig{
      .dataBufferSize = 8192,
      .chunkSize = 8192,
      .pipelineDepth = 4,
  };

  MultiPeerIbgdaTransportSetup setup(
      *transport, globalRank, numRanks, setupConfig);
  setup.exchangeBuffers();

  Transport* d_transports =
      buildTransportsArray(*transport, setup, globalRank, numRanks);

  // Variable sizes: (globalRank + rank + 1) * 64 ints per peer
  const size_t baseInts = 64;
  std::vector<ChunkInfo> h_send_chunks, h_recv_chunks;
  size_t sendOffset = 0, recvOffset = 0;

  for (int rank = 0; rank < numRanks; rank++) {
    size_t numInts = (globalRank + rank + 1) * baseInts;
    size_t nbytes = numInts * sizeof(int32_t);
    h_send_chunks.emplace_back(sendOffset, nbytes);
    h_recv_chunks.emplace_back(recvOffset, nbytes);
    sendOffset += nbytes;
    recvOffset += nbytes;
  }

  DeviceBuffer sendBuffer(std::max(sendOffset, size_t(16)));
  DeviceBuffer recvBuffer(std::max(recvOffset, size_t(16)));

  // Fill send buffer
  std::vector<int32_t> h_send(sendOffset / sizeof(int32_t));
  size_t intOff = 0;
  for (int peer = 0; peer < numRanks; peer++) {
    size_t numInts = (globalRank + peer + 1) * baseInts;
    for (size_t i = 0; i < numInts; i++) {
      h_send[intOff + i] =
          globalRank * 1000 + peer * 100 + static_cast<int32_t>(i);
    }
    intOff += numInts;
  }
  CUDACHECK_TEST(cudaMemcpy(
      sendBuffer.get(), h_send.data(), sendOffset, cudaMemcpyHostToDevice));

  // Init recv with -1
  std::vector<int32_t> h_recv_init(recvOffset / sizeof(int32_t), -1);
  CUDACHECK_TEST(cudaMemcpy(
      recvBuffer.get(),
      h_recv_init.data(),
      recvOffset,
      cudaMemcpyHostToDevice));

  // Copy chunks to device
  DeviceBuffer d_send_chunks(sizeof(ChunkInfo) * numRanks);
  DeviceBuffer d_recv_chunks(sizeof(ChunkInfo) * numRanks);
  CUDACHECK_TEST(cudaMemcpy(
      d_send_chunks.get(),
      h_send_chunks.data(),
      sizeof(ChunkInfo) * numRanks,
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemcpy(
      d_recv_chunks.get(),
      h_recv_chunks.data(),
      sizeof(ChunkInfo) * numRanks,
      cudaMemcpyHostToDevice));

  MPI_Barrier(MPI_COMM_WORLD);

  all_to_allv(
      recvBuffer.get(),
      sendBuffer.get(),
      globalRank,
      DeviceSpan<Transport>(d_transports, static_cast<uint32_t>(numRanks)),
      DeviceSpan<ChunkInfo>(
          static_cast<ChunkInfo*>(d_send_chunks.get()), numRanks),
      DeviceSpan<ChunkInfo>(
          static_cast<ChunkInfo*>(d_recv_chunks.get()), numRanks),
      std::chrono::milliseconds{30000},
      nullptr,
      4,
      256,
      std::nullopt);

  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Verify
  std::vector<int32_t> h_recv(recvOffset / sizeof(int32_t));
  CUDACHECK_TEST(cudaMemcpy(
      h_recv.data(), recvBuffer.get(), recvOffset, cudaMemcpyDeviceToHost));

  int errorCount = 0;
  intOff = 0;
  for (int peer = 0; peer < numRanks; peer++) {
    size_t numInts = (globalRank + peer + 1) * baseInts;
    for (size_t i = 0; i < numInts; i++) {
      int32_t expected =
          peer * 1000 + globalRank * 100 + static_cast<int32_t>(i);
      int32_t actual = h_recv[intOff + i];
      if (expected != actual) {
        errorCount++;
        if (errorCount <= 10) {
          XLOGF(
              ERR,
              "Rank {}: MixedSizes error at peer {} pos {}: expected {}, got {}",
              globalRank,
              peer,
              i,
              expected,
              actual);
        }
      }
    }
    intOff += numInts;
  }

  EXPECT_EQ(errorCount, 0) << "Rank " << globalRank << " found " << errorCount
                           << " verification errors";
  MPI_Barrier(MPI_COMM_WORLD);
  cudaFree(d_transports);
}

// =============================================================================
// Parameterized Tests: Block and Thread count sweeps
// =============================================================================

class AllToAllvIbgdaE2eBlockSweepTest
    : public AllToAllvIbgdaE2eTestFixture,
      public ::testing::WithParamInterface<int> {};

TEST_P(AllToAllvIbgdaE2eBlockSweepTest, EqualSize_VaryingBlocks) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Need at least 2 ranks";
  }
  auto transport = createTransport();
  if (shouldSkip(transport)) {
    GTEST_SKIP() << "IBGDA transport not available (collective skip)";
  }

  MultiPeerIbgdaTransportSetupConfig setupConfig{
      .dataBufferSize = 4096,
      .chunkSize = 4096,
      .pipelineDepth = 4,
  };

  MultiPeerIbgdaTransportSetup setup(
      *transport, globalRank, numRanks, setupConfig);
  setup.exchangeBuffers();

  const int numBlocks = GetParam();
  runEqualSizeTest(*transport, setup, 256, numBlocks, 256);
}

INSTANTIATE_TEST_SUITE_P(
    BlockSweep,
    AllToAllvIbgdaE2eBlockSweepTest,
    ::testing::Values(2, 4, 8, 16, 32));

class AllToAllvIbgdaE2eThreadSweepTest
    : public AllToAllvIbgdaE2eTestFixture,
      public ::testing::WithParamInterface<int> {};

TEST_P(AllToAllvIbgdaE2eThreadSweepTest, EqualSize_VaryingThreads) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Need at least 2 ranks";
  }
  auto transport = createTransport();
  if (shouldSkip(transport)) {
    GTEST_SKIP() << "IBGDA transport not available (collective skip)";
  }

  MultiPeerIbgdaTransportSetupConfig setupConfig{
      .dataBufferSize = 4096,
      .chunkSize = 4096,
      .pipelineDepth = 4,
  };

  MultiPeerIbgdaTransportSetup setup(
      *transport, globalRank, numRanks, setupConfig);
  setup.exchangeBuffers();

  const int totalThreads = GetParam();
  const int maxThreadsPerBlock = 256;
  const int threadsPerBlock = std::min(totalThreads, maxThreadsPerBlock);
  // Block-group scheduling requires at least 2 blocks for
  // partition_interleaved(2) send/recv split.
  const int numBlocks = std::max(2, totalThreads / threadsPerBlock);
  runEqualSizeTest(*transport, setup, 256, numBlocks, threadsPerBlock);
}

INSTANTIATE_TEST_SUITE_P(
    ThreadSweep,
    AllToAllvIbgdaE2eThreadSweepTest,
    ::testing::Values(128, 256, 512));

// =============================================================================
// Zero bytes test: one peer sends 0 bytes
// =============================================================================

TEST_F(AllToAllvIbgdaE2eTestFixture, ZeroBytes) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Need at least 2 ranks";
  }
  auto transport = createTransport();
  if (shouldSkip(transport)) {
    GTEST_SKIP() << "IBGDA transport not available (collective skip)";
  }

  MultiPeerIbgdaTransportSetupConfig setupConfig{
      .dataBufferSize = 4096,
      .chunkSize = 4096,
      .pipelineDepth = 4,
  };

  MultiPeerIbgdaTransportSetup setup(
      *transport, globalRank, numRanks, setupConfig);
  setup.exchangeBuffers();

  Transport* d_transports =
      buildTransportsArray(*transport, setup, globalRank, numRanks);

  // Per-peer sizes: any pair involving rank 0 is 0 bytes, others 256 bytes.
  // Both send and recv must agree: if rank A sends 0 to rank B, then
  // rank B must also expect 0 from rank A (alltoallv symmetry).
  const size_t normalInts = 64;
  const size_t normalBytes = normalInts * sizeof(int32_t);
  std::vector<ChunkInfo> h_send_chunks, h_recv_chunks;
  size_t sendOffset = 0, recvOffset = 0;

  for (int rank = 0; rank < numRanks; rank++) {
    size_t nbytes = (rank == 0 || globalRank == 0) ? 0 : normalBytes;
    h_send_chunks.emplace_back(sendOffset, nbytes);
    h_recv_chunks.emplace_back(recvOffset, nbytes);
    sendOffset += nbytes;
    recvOffset += nbytes;
  }

  DeviceBuffer sendBuffer(std::max(sendOffset, size_t(16)));
  DeviceBuffer recvBuffer(std::max(recvOffset, size_t(16)));

  // Fill send buffer
  size_t totalSendInts = sendOffset / sizeof(int32_t);
  std::vector<int32_t> h_send(totalSendInts);
  size_t intOff = 0;
  for (int peer = 0; peer < numRanks; peer++) {
    size_t numInts = (peer == 0 || globalRank == 0) ? 0 : normalInts;
    for (size_t i = 0; i < numInts; i++) {
      h_send[intOff + i] =
          globalRank * 1000 + peer * 100 + static_cast<int32_t>(i);
    }
    intOff += numInts;
  }
  if (sendOffset > 0) {
    CUDACHECK_TEST(cudaMemcpy(
        sendBuffer.get(), h_send.data(), sendOffset, cudaMemcpyHostToDevice));
  }

  // Init recv with -1
  size_t totalRecvInts = recvOffset / sizeof(int32_t);
  std::vector<int32_t> h_recv_init(totalRecvInts, -1);
  if (recvOffset > 0) {
    CUDACHECK_TEST(cudaMemcpy(
        recvBuffer.get(),
        h_recv_init.data(),
        recvOffset,
        cudaMemcpyHostToDevice));
  }

  DeviceBuffer d_send_chunks(sizeof(ChunkInfo) * numRanks);
  DeviceBuffer d_recv_chunks(sizeof(ChunkInfo) * numRanks);
  CUDACHECK_TEST(cudaMemcpy(
      d_send_chunks.get(),
      h_send_chunks.data(),
      sizeof(ChunkInfo) * numRanks,
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemcpy(
      d_recv_chunks.get(),
      h_recv_chunks.data(),
      sizeof(ChunkInfo) * numRanks,
      cudaMemcpyHostToDevice));

  MPI_Barrier(MPI_COMM_WORLD);

  all_to_allv(
      recvBuffer.get(),
      sendBuffer.get(),
      globalRank,
      DeviceSpan<Transport>(d_transports, static_cast<uint32_t>(numRanks)),
      DeviceSpan<ChunkInfo>(
          static_cast<ChunkInfo*>(d_send_chunks.get()), numRanks),
      DeviceSpan<ChunkInfo>(
          static_cast<ChunkInfo*>(d_recv_chunks.get()), numRanks),
      std::chrono::milliseconds{30000},
      nullptr,
      4,
      256,
      std::nullopt);

  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Verify: peer 0 should have 0 bytes, others should be correct
  if (recvOffset > 0) {
    std::vector<int32_t> h_recv(totalRecvInts);
    CUDACHECK_TEST(cudaMemcpy(
        h_recv.data(), recvBuffer.get(), recvOffset, cudaMemcpyDeviceToHost));

    int errorCount = 0;
    intOff = 0;
    for (int peer = 0; peer < numRanks; peer++) {
      size_t numInts = (peer == 0 || globalRank == 0) ? 0 : normalInts;
      for (size_t i = 0; i < numInts; i++) {
        int32_t expected =
            peer * 1000 + globalRank * 100 + static_cast<int32_t>(i);
        int32_t actual = h_recv[intOff + i];
        if (expected != actual) {
          errorCount++;
          if (errorCount <= 10) {
            XLOGF(
                ERR,
                "Rank {}: ZeroBytes error at peer {} pos {}: expected {}, got {}",
                globalRank,
                peer,
                i,
                expected,
                actual);
          }
        }
      }
      intOff += numInts;
    }

    EXPECT_EQ(errorCount, 0) << "Rank " << globalRank << " found " << errorCount
                             << " verification errors";
  }

  MPI_Barrier(MPI_COMM_WORLD);
  cudaFree(d_transports);
}

// =============================================================================
// Pipeline depth variations
// =============================================================================

class AllToAllvIbgdaE2ePipelineDepthTest
    : public AllToAllvIbgdaE2eTestFixture,
      public ::testing::WithParamInterface<int> {};

TEST_P(AllToAllvIbgdaE2ePipelineDepthTest, PipelineDepthVariations) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Need at least 2 ranks";
  }

  const int pipelineDepth = GetParam();

  // Fresh transport per depth to avoid NIC state conflicts
  auto transport = createTransport();
  if (shouldSkip(transport)) {
    GTEST_SKIP() << "IBGDA transport not available (collective skip)";
  }

  // 16KB staging buffer, 1MB message → exercises pipelining
  MultiPeerIbgdaTransportSetupConfig setupConfig{
      .dataBufferSize = 16 * 1024,
      .chunkSize = 16 * 1024,
      .pipelineDepth = pipelineDepth,
  };

  MultiPeerIbgdaTransportSetup setup(
      *transport, globalRank, numRanks, setupConfig);
  setup.exchangeBuffers();

  // 262144 ints = 1MB per peer → multiple pipeline steps needed
  runEqualSizeTest(*transport, setup, 262144);
}

INSTANTIATE_TEST_SUITE_P(
    PipelineDepth,
    AllToAllvIbgdaE2ePipelineDepthTest,
    ::testing::Values(1, 2, 4, 8));

// =============================================================================
// Block-scope cooperative memcpy tests
//
// These tests verify that the block-group scheduling path works correctly.
// When IBGDA peers are present, all_to_allv() uses make_block_group() where
// all threads in a block cooperatively memcpy to staging via named barriers.
// =============================================================================

TEST_F(AllToAllvIbgdaE2eTestFixture, BlockScope_MinBlocks) {
  // Launch with exactly 2*nranks blocks (minimum for block-group scheduling)
  if (numRanks < 2) {
    GTEST_SKIP() << "Need at least 2 ranks";
  }
  auto transport = createTransport();
  if (shouldSkip(transport)) {
    GTEST_SKIP() << "IBGDA transport not available";
  }
  MultiPeerIbgdaTransportSetupConfig setupConfig;
  MultiPeerIbgdaTransportSetup setup(
      *transport, globalRank, numRanks, setupConfig);
  setup.exchangeBuffers();

  // 256 ints = 1024 bytes per peer
  int minBlocks = 2 * numRanks;
  runEqualSizeTest(*transport, setup, 256, minBlocks, 256);
}

TEST_F(AllToAllvIbgdaE2eTestFixture, BlockScope_ExcessBlocks) {
  // Launch with many more blocks than peers to test excess block early-return
  if (numRanks < 2) {
    GTEST_SKIP() << "Need at least 2 ranks";
  }
  auto transport = createTransport();
  if (shouldSkip(transport)) {
    GTEST_SKIP() << "IBGDA transport not available";
  }
  MultiPeerIbgdaTransportSetupConfig setupConfig;
  MultiPeerIbgdaTransportSetup setup(
      *transport, globalRank, numRanks, setupConfig);
  setup.exchangeBuffers();

  // 1024 ints per peer, 64 blocks (many excess blocks per peer)
  runEqualSizeTest(*transport, setup, 1024, 64, 256);
}

TEST_F(AllToAllvIbgdaE2eTestFixture, BlockScope_LargeMessage) {
  // Large message with block-group scheduling to exercise multi-step pipeline
  if (numRanks < 2) {
    GTEST_SKIP() << "Need at least 2 ranks";
  }
  auto transport = createTransport();
  if (shouldSkip(transport)) {
    GTEST_SKIP() << "IBGDA transport not available";
  }
  MultiPeerIbgdaTransportSetupConfig setupConfig;
  MultiPeerIbgdaTransportSetup setup(
      *transport, globalRank, numRanks, setupConfig);
  setup.exchangeBuffers();

  // 262144 ints = 1MB per peer, 32 blocks
  runEqualSizeTest(*transport, setup, 262144, 32, 256);
}

// =============================================================================
// Multi-block per peer tests: verify correctness when multiple blocks service
// the same peer, each with a disjoint channel range and data slice.
// =============================================================================

TEST_F(AllToAllvIbgdaE2eTestFixture, BlockScope_MultiBlockPerPeer_2Blocks) {
  // 2 active blocks per peer: maxChannelsPerPeer=2, 4*nranks total blocks.
  // Each active block handles 1 channel with a disjoint data slice.
  if (numRanks < 2) {
    GTEST_SKIP() << "Need at least 2 ranks";
  }
  auto transport = createTransport();
  if (shouldSkip(transport)) {
    GTEST_SKIP() << "IBGDA transport not available";
  }

  const int maxChannels = 2;
  MultiPeerIbgdaTransportSetupConfig setupConfig{
      .dataBufferSize = 4096,
      .chunkSize = 4096,
      .pipelineDepth = 4,
  };
  MultiPeerIbgdaTransportSetup setup(
      *transport, globalRank, numRanks, setupConfig, maxChannels);
  setup.exchangeBuffers();

  // 4*nranks blocks: after send/recv split → 2*nranks per direction.
  // With maxChannelsPerPeer=2, numActiveBlocks=2 per peer.
  runEqualSizeTest(*transport, setup, 1024, 4 * numRanks, 256);
}

TEST_F(AllToAllvIbgdaE2eTestFixture, BlockScope_MultiBlockPerPeer_4Blocks) {
  // 4 active blocks per peer: maxChannelsPerPeer=4, 8*nranks total blocks.
  if (numRanks < 2) {
    GTEST_SKIP() << "Need at least 2 ranks";
  }
  auto transport = createTransport();
  if (shouldSkip(transport)) {
    GTEST_SKIP() << "IBGDA transport not available";
  }

  const int maxChannels = 4;
  MultiPeerIbgdaTransportSetupConfig setupConfig{
      .dataBufferSize = 4096,
      .chunkSize = 4096,
      .pipelineDepth = 4,
  };
  MultiPeerIbgdaTransportSetup setup(
      *transport, globalRank, numRanks, setupConfig, maxChannels);
  setup.exchangeBuffers();

  // 8*nranks blocks: after send/recv split → 4*nranks per direction.
  // With maxChannelsPerPeer=4, numActiveBlocks=4 per peer.
  runEqualSizeTest(*transport, setup, 1024, 8 * numRanks, 256);
}

TEST_F(
    AllToAllvIbgdaE2eTestFixture,
    BlockScope_MultiBlockPerPeer_WithThreadChannels) {
  // Combines multi-block and multi-channel-within-block:
  //   maxChannelsPerPeer=8, 256 threads → 8 warps/block → channelsPerBlock=8.
  //   numActiveBlocks=1 (maxChannelsPerPeer / channelsPerBlock = 1).
  // This is the thread-channel path exercised across realistic block counts.
  if (numRanks < 2) {
    GTEST_SKIP() << "Need at least 2 ranks";
  }
  auto transport = createTransport();
  if (shouldSkip(transport)) {
    GTEST_SKIP() << "IBGDA transport not available";
  }

  const int maxChannels = 8;
  MultiPeerIbgdaTransportSetupConfig setupConfig{
      .dataBufferSize = 4096,
      .chunkSize = 4096,
      .pipelineDepth = 4,
  };
  MultiPeerIbgdaTransportSetup setup(
      *transport, globalRank, numRanks, setupConfig, maxChannels);
  setup.exchangeBuffers();

  // 4*nranks blocks, 256 threads/block: 8 warps → channelsPerBlock=8,
  // maxBlocks=1 → all threads in one block split across 8 channels.
  runEqualSizeTest(*transport, setup, 1024, 4 * numRanks, 256);
}

// =============================================================================
// True multi-block per peer tests: exercise numActiveBlocks > 1.
//
// The tests above use 256 threads/block (8 warps), so channelsPerBlock =
// min(8, maxCh) = maxCh, giving numActiveBlocks = 1 (single block handles
// all channels). These tests use fewer threads per block (32 or 64) so
// channelsPerBlock < maxChannelsPerPeer, forcing the two-level partitioning
// to actually split work across multiple blocks per peer.
// =============================================================================

TEST_F(AllToAllvIbgdaE2eTestFixture, BlockScope_TrueMultiBlock_2Blocks) {
  // 32 threads/block = 1 warp → channelsPerBlock = min(1, 2) = 1.
  // maxBlocks = 2/1 = 2 → numActiveBlocks = min(2, 2) = 2.
  // Each of 2 active blocks handles 1 channel with half the data.
  if (numRanks < 2) {
    GTEST_SKIP() << "Need at least 2 ranks";
  }
  auto transport = createTransport();
  if (shouldSkip(transport)) {
    GTEST_SKIP() << "IBGDA transport not available";
  }

  const int maxChannels = 2;
  MultiPeerIbgdaTransportSetupConfig setupConfig{
      .dataBufferSize = 4096,
      .chunkSize = 4096,
      .pipelineDepth = 4,
  };
  MultiPeerIbgdaTransportSetup setup(
      *transport, globalRank, numRanks, setupConfig, maxChannels);
  setup.exchangeBuffers();

  // 4*nranks blocks × 32 threads: after send/recv split → 2 blocks per peer.
  // Both blocks active with channelsPerBlock=1: block 0 → ch0, block 1 → ch1.
  runEqualSizeTest(*transport, setup, 1024, 4 * numRanks, 32);
}

TEST_F(AllToAllvIbgdaE2eTestFixture, BlockScope_TrueMultiBlock_4Blocks) {
  // 32 threads/block = 1 warp → channelsPerBlock = min(1, 4) = 1.
  // maxBlocks = 4/1 = 4 → numActiveBlocks = min(4, 4) = 4.
  // Each of 4 active blocks handles 1 channel with quarter of the data.
  if (numRanks < 2) {
    GTEST_SKIP() << "Need at least 2 ranks";
  }
  auto transport = createTransport();
  if (shouldSkip(transport)) {
    GTEST_SKIP() << "IBGDA transport not available";
  }

  const int maxChannels = 4;
  MultiPeerIbgdaTransportSetupConfig setupConfig{
      .dataBufferSize = 4096,
      .chunkSize = 4096,
      .pipelineDepth = 4,
  };
  MultiPeerIbgdaTransportSetup setup(
      *transport, globalRank, numRanks, setupConfig, maxChannels);
  setup.exchangeBuffers();

  // 8*nranks blocks × 32 threads: after send/recv split → 4 blocks per peer.
  // All 4 blocks active: block i → channel i, with data slice i/4.
  runEqualSizeTest(*transport, setup, 1024, 8 * numRanks, 32);
}

TEST_F(
    AllToAllvIbgdaE2eTestFixture,
    BlockScope_TrueMultiBlock_WithThreadChannels) {
  // 64 threads/block = 2 warps → channelsPerBlock = min(2, 8) = 2.
  // maxBlocks = 8/2 = 4 → numActiveBlocks = min(4, 4) = 4.
  // Each of 4 active blocks handles 2 channels (via thread-channel split),
  // covering all 8 channels across 4 blocks.
  if (numRanks < 2) {
    GTEST_SKIP() << "Need at least 2 ranks";
  }
  auto transport = createTransport();
  if (shouldSkip(transport)) {
    GTEST_SKIP() << "IBGDA transport not available";
  }

  const int maxChannels = 8;
  MultiPeerIbgdaTransportSetupConfig setupConfig{
      .dataBufferSize = 4096,
      .chunkSize = 4096,
      .pipelineDepth = 4,
  };
  MultiPeerIbgdaTransportSetup setup(
      *transport, globalRank, numRanks, setupConfig, maxChannels);
  setup.exchangeBuffers();

  // 8*nranks blocks × 64 threads: after send/recv split → 4 blocks per peer.
  // Each block has 2 warps → channelsPerBlock=2. 4 blocks × 2 ch = 8 channels.
  runEqualSizeTest(*transport, setup, 1024, 8 * numRanks, 64);
}

TEST_F(AllToAllvIbgdaE2eTestFixture, BlockScope_TrueMultiBlock_ExcessBlocks) {
  // Tests excess-block early-return in the multi-block path.
  // 32 threads/block = 1 warp → channelsPerBlock = 1, maxBlocks = 2.
  // 8*nranks blocks → 4 blocks per peer, but numActiveBlocks = min(4, 2) = 2.
  // Blocks 2-3 per peer early-return; only blocks 0-1 do work.
  if (numRanks < 2) {
    GTEST_SKIP() << "Need at least 2 ranks";
  }
  auto transport = createTransport();
  if (shouldSkip(transport)) {
    GTEST_SKIP() << "IBGDA transport not available";
  }

  const int maxChannels = 2;
  MultiPeerIbgdaTransportSetupConfig setupConfig{
      .dataBufferSize = 4096,
      .chunkSize = 4096,
      .pipelineDepth = 4,
  };
  MultiPeerIbgdaTransportSetup setup(
      *transport, globalRank, numRanks, setupConfig, maxChannels);
  setup.exchangeBuffers();

  // 8*nranks blocks × 32 threads: 4 blocks/peer, but only 2 active.
  runEqualSizeTest(*transport, setup, 1024, 8 * numRanks, 32);
}

} // namespace comms::pipes

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  // GTest takes ownership of the environment pointer via
  // AddGlobalTestEnvironment, so we must NOT wrap it in unique_ptr.
  ::testing::AddGlobalTestEnvironment(new meta::comms::MPIEnvironmentBase());
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
