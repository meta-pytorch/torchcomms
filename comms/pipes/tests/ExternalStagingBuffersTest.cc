// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include <memory>
#include <vector>

#include "comms/pipes/GpuMemHandler.h"
#include "comms/pipes/MultiPeerNvlTransport.h"
#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/pipes/tests/Utils.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::DeviceBuffer;
using meta::comms::MpiBaseTestFixture;
using meta::comms::MpiBootstrap;
using meta::comms::MPIEnvironmentBase;

namespace comms::pipes::tests {

namespace {
constexpr std::size_t kDataBufferSize = 1024 * 1024; // 1MB per peer
constexpr std::size_t kChunkSize = 1024;
constexpr std::size_t kPipelineDepth = 4;
constexpr std::size_t kNumElements = 256;
} // namespace

class ExternalStagingBuffersTestFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MpiBaseTestFixture::TearDown();
  }

  MultiPeerNvlTransportConfig defaultConfig() {
    return MultiPeerNvlTransportConfig{
        .dataBufferSize = kDataBufferSize,
        .chunkSize = kChunkSize,
        .pipelineDepth = kPipelineDepth,
    };
  }

  // Allocate external data buffers using GpuMemHandler to handle IPC sharing.
  // Returns the GpuMemHandler (for lifetime management) and the
  // ExternalStagingBuffers struct with per-peer buffer pointers.
  struct ExternalBufferBundle {
    std::unique_ptr<GpuMemHandler> handler;
    ExternalStagingBuffers stagingBuffers;
  };

  ExternalBufferBundle allocateExternalBuffers(
      std::shared_ptr<meta::comms::IBootstrap> bootstrap,
      std::size_t perPeerSize) {
    const std::size_t totalSize = perPeerSize * (numRanks - 1);
    auto memSharingMode = GpuMemHandler::detectBestMode();
    auto handler = std::make_unique<GpuMemHandler>(
        bootstrap, globalRank, numRanks, totalSize, memSharingMode);
    handler->exchangeMemPtrs();

    // DeviceSpan is non-assignable (const pointer member), so build vectors
    // sequentially via emplace_back.
    std::vector<DeviceSpan<char>> localSpans;
    std::vector<DeviceSpan<char>> remoteSpans;
    localSpans.reserve(numRanks);
    remoteSpans.reserve(numRanks);
    const auto size = static_cast<uint32_t>(perPeerSize);

    for (int peer = 0; peer < numRanks; ++peer) {
      if (peer == globalRank) {
        localSpans.emplace_back(nullptr, 0u);
        remoteSpans.emplace_back(nullptr, 0u);
        continue;
      }
      // Calculate offsets using same logic as MultiPeerNvlTransport
      const int localPeerIndex = (peer < globalRank) ? peer : (peer - 1);
      const int remotePeerIndex =
          (globalRank < peer) ? globalRank : (globalRank - 1);

      char* localPtr = static_cast<char*>(handler->getLocalDeviceMemPtr()) +
          localPeerIndex * perPeerSize;
      char* remotePtr = static_cast<char*>(handler->getPeerDeviceMemPtr(peer)) +
          remotePeerIndex * perPeerSize;
      localSpans.emplace_back(localPtr, size);
      remoteSpans.emplace_back(remotePtr, size);
    }

    ExternalStagingBuffers staging;
    staging.localBuffers = std::move(localSpans);
    staging.remoteBuffers = std::move(remoteSpans);

    return {std::move(handler), std::move(staging)};
  }
};

// Verify that transport devices use external buffer pointers when set.
// buildP2pTransportDevice() should return devices whose data buffer pointers
// match the external buffers we provided.
TEST_F(ExternalStagingBuffersTestFixture, TransportUsesExternalBuffers) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks, got " << numRanks;
  }

  auto config = defaultConfig();
  const std::size_t perPeerDataSize =
      config.pipelineDepth * config.dataBufferSize;
  auto bootstrap = std::make_shared<MpiBootstrap>();

  // Allocate external buffers via IPC
  auto extBuf = allocateExternalBuffers(bootstrap, perPeerDataSize);

  // Create transport with external buffers
  MultiPeerNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.setExternalDataBuffers(std::move(extBuf.stagingBuffers));
  transport.exchange();

  // Verify that the transport device uses the external buffer pointers
  for (int peer = 0; peer < numRanks; ++peer) {
    if (peer == globalRank) {
      continue;
    }
    auto p2p = transport.buildP2pTransportDevice(peer);
    auto localData = p2p.getLocalState().dataBuffer;
    auto remoteData = p2p.getRemoteState().dataBuffer;

    // Re-compute expected pointers from the handler
    const int localPeerIndex = (peer < globalRank) ? peer : (peer - 1);
    const int remotePeerIndex =
        (globalRank < peer) ? globalRank : (globalRank - 1);

    char* expectedLocal =
        static_cast<char*>(extBuf.handler->getLocalDeviceMemPtr()) +
        localPeerIndex * perPeerDataSize;
    char* expectedRemote =
        static_cast<char*>(extBuf.handler->getPeerDeviceMemPtr(peer)) +
        remotePeerIndex * perPeerDataSize;

    EXPECT_EQ(localData, expectedLocal)
        << "Rank " << globalRank << ": local data buffer for peer " << peer
        << " does not match external buffer";
    EXPECT_EQ(remoteData, expectedRemote)
        << "Rank " << globalRank << ": remote data buffer for peer " << peer
        << " does not match external buffer";
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

// Verify that IPC memory access works through external data buffers.
// Each rank writes a pattern to its local external buffer, then the peer
// reads from the remote external buffer and verifies correctness.
TEST_F(ExternalStagingBuffersTestFixture, IpcMemAccessThroughExternalBuffers) {
  if (numRanks != 2) {
    GTEST_SKIP() << "Requires exactly 2 ranks, got " << numRanks;
  }

  const int peerRank = (globalRank == 0) ? 1 : 0;

  auto config = defaultConfig();
  config.dataBufferSize = sizeof(int) * kNumElements;
  const std::size_t perPeerDataSize =
      config.pipelineDepth * config.dataBufferSize;
  auto bootstrap = std::make_shared<MpiBootstrap>();

  // Allocate external buffers via IPC
  auto extBuf = allocateExternalBuffers(bootstrap, perPeerDataSize);

  // Create transport with external buffers
  MultiPeerNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.setExternalDataBuffers(std::move(extBuf.stagingBuffers));
  transport.exchange();

  // Get host-side copy of transport device to access buffer pointers
  auto p2p = transport.buildP2pTransportDevice(peerRank);

  auto localAddr =
      static_cast<int*>(static_cast<void*>(p2p.getLocalState().dataBuffer));
  auto remoteAddr =
      static_cast<int*>(static_cast<void*>(p2p.getRemoteState().dataBuffer));

  XLOGF(
      INFO,
      "Rank {}: localAddr={}, remoteAddr={}",
      globalRank,
      static_cast<void*>(localAddr),
      static_cast<void*>(remoteAddr));

  // Each rank writes its rank ID to its local data buffer
  test::fillBuffer(localAddr, globalRank, kNumElements);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Barrier to ensure both ranks have written
  MPI_Barrier(MPI_COMM_WORLD);

  // Read from peer's buffer via IPC and verify
  DeviceBuffer errorCountBuffer(sizeof(int));
  auto d_errorCount = static_cast<int*>(errorCountBuffer.get());
  CUDACHECK_TEST(cudaMemset(d_errorCount, 0, sizeof(int)));

  test::verifyBuffer(remoteAddr, peerRank, kNumElements, d_errorCount);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  int h_errorCount = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &h_errorCount, d_errorCount, sizeof(int), cudaMemcpyDeviceToHost));

  ASSERT_EQ(h_errorCount, 0)
      << "Rank " << globalRank << " found " << h_errorCount
      << " errors reading peer's external buffer";

  MPI_Barrier(MPI_COMM_WORLD);
}

// Verify that the default path (no external buffers) still allocates internal
// data buffers and works correctly.
TEST_F(ExternalStagingBuffersTestFixture, DefaultPathAllocatesInternalBuffers) {
  if (numRanks != 2) {
    GTEST_SKIP() << "Requires exactly 2 ranks, got " << numRanks;
  }

  const int peerRank = (globalRank == 0) ? 1 : 0;

  auto config = defaultConfig();
  config.dataBufferSize = sizeof(int) * kNumElements;
  auto bootstrap = std::make_shared<MpiBootstrap>();

  // Create transport WITHOUT external buffers (default path)
  MultiPeerNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.exchange();

  // Verify data buffers are accessible and functional
  auto p2p = transport.buildP2pTransportDevice(peerRank);

  auto localAddr =
      static_cast<int*>(static_cast<void*>(p2p.getLocalState().dataBuffer));
  auto remoteAddr =
      static_cast<int*>(static_cast<void*>(p2p.getRemoteState().dataBuffer));

  // Buffer pointers should be non-null (internal allocation happened)
  ASSERT_NE(localAddr, nullptr)
      << "Default path: local data buffer should be non-null";
  ASSERT_NE(remoteAddr, nullptr)
      << "Default path: remote data buffer should be non-null";

  // Write and verify through IPC to confirm functional
  test::fillBuffer(localAddr, globalRank, kNumElements);
  CUDACHECK_TEST(cudaDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  DeviceBuffer errorCountBuffer(sizeof(int));
  auto d_errorCount = static_cast<int*>(errorCountBuffer.get());
  CUDACHECK_TEST(cudaMemset(d_errorCount, 0, sizeof(int)));
  test::verifyBuffer(remoteAddr, peerRank, kNumElements, d_errorCount);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  int h_errorCount = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &h_errorCount, d_errorCount, sizeof(int), cudaMemcpyDeviceToHost));
  ASSERT_EQ(h_errorCount, 0)
      << "Default path: IPC read through internal buffers failed";

  MPI_Barrier(MPI_COMM_WORLD);
}

// Verify that state and signal buffers are still internally allocated when
// external data buffers are provided.
TEST_F(ExternalStagingBuffersTestFixture, StateAndSignalBuffersStillAllocated) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks, got " << numRanks;
  }

  auto config = defaultConfig();
  const std::size_t perPeerDataSize =
      config.pipelineDepth * config.dataBufferSize;
  auto bootstrap = std::make_shared<MpiBootstrap>();

  auto extBuf = allocateExternalBuffers(bootstrap, perPeerDataSize);

  MultiPeerNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.setExternalDataBuffers(std::move(extBuf.stagingBuffers));
  transport.exchange();

  // Verify that state and signal buffers are non-null for each peer
  for (int peer = 0; peer < numRanks; ++peer) {
    if (peer == globalRank) {
      continue;
    }
    auto p2p = transport.buildP2pTransportDevice(peer);

    // State buffer spans should be non-empty (internally allocated)
    EXPECT_FALSE(p2p.getLocalState().receiverStateBuffer.empty())
        << "Rank " << globalRank << ": local state buffer for peer " << peer
        << " should not be empty";
    EXPECT_FALSE(p2p.getRemoteState().receiverStateBuffer.empty())
        << "Rank " << globalRank << ": remote state buffer for peer " << peer
        << " should not be empty";

    // Signal buffer spans should be non-empty (internally allocated)
    EXPECT_FALSE(p2p.getLocalState().signalBuffer.empty())
        << "Rank " << globalRank << ": local signal buffer for peer " << peer
        << " should not be empty";
    EXPECT_FALSE(p2p.getRemoteState().signalBuffer.empty())
        << "Rank " << globalRank << ": remote signal buffer for peer " << peer
        << " should not be empty";
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace comms::pipes::tests

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
