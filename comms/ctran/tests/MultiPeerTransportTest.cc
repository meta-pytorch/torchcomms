// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda.h>
#include <cuda_runtime.h>

#include <folly/init/Init.h>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/algos/CtranAlgo.h"
#include "comms/ctran/algos/CtranAlgoDev.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/pipes/MultiPeerDeviceHandle.cuh"
#include "comms/pipes/MultiPeerTransport.h"
#include "comms/pipes/P2pNvlTransportDevice.cuh"
#include "comms/pipes/tests/Utils.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

using namespace meta::comms;

class MultiPeerTransportEnvironment : public ctran::CtranEnvironmentBase {
 public:
  void SetUp() override {
    ctran::CtranEnvironmentBase::SetUp();

    // Enable Pipes MultiPeerTransport
    setenv("NCCL_CTRAN_USE_PIPES", "1", 1);
    setenv("NCCL_DEBUG", "INFO", 1);
  }
};

class MultiPeerTransportTest : public ctran::CtranDistTestFixture {
 public:
  void SetUp() override {
    CtranDistTestFixture::SetUp();
  }

  void TearDown() override {
    CtranDistTestFixture::TearDown();
  }
};

TEST_F(MultiPeerTransportTest, InitAndExchange) {
  // Create ctran comm - this should initialize MultiPeerTransport
  auto comm = makeCtranComm();

  ASSERT_NE(comm, nullptr);
  ASSERT_NE(comm->ctran_, nullptr);
  ASSERT_TRUE(comm->ctran_->isInitialized());

  // Verify MultiPeerTransport was created
  ASSERT_NE(comm->multiPeerTransport_, nullptr)
      << "MultiPeerTransport should be initialized when NCCL_CTRAN_USE_PIPES=1";

  // Verify rank and nRanks match
  EXPECT_EQ(comm->multiPeerTransport_->my_rank(), globalRank)
      << "MultiPeerTransport rank should match globalRank";
  EXPECT_EQ(comm->multiPeerTransport_->n_ranks(), numRanks)
      << "MultiPeerTransport nRanks should match numRanks";

  XLOG(INFO) << "Rank " << globalRank << "/" << numRanks
             << ": MultiPeerTransport initialized successfully";

  // Verify transport types are set for all peers
  for (int peer = 0; peer < numRanks; peer++) {
    auto transportType = comm->multiPeerTransport_->get_transport_type(peer);
    if (peer == globalRank) {
      EXPECT_EQ(transportType, comms::pipes::TransportType::SELF)
          << "Transport to self should be SELF";
    } else {
      // Should be either NVL or IBGDA depending on topology
      EXPECT_TRUE(
          transportType == comms::pipes::TransportType::P2P_NVL ||
          transportType == comms::pipes::TransportType::P2P_IBGDA)
          << "Transport to peer " << peer << " should be P2P_NVL or P2P_IBGDA";
    }
    XLOG(INFO) << "Rank " << globalRank << ": transport to peer " << peer
               << " is type " << static_cast<int>(transportType);
  }

  // Verify NVL peer info
  int nvlNRanks = comm->multiPeerTransport_->nvl_n_ranks();
  int nvlLocalRank = comm->multiPeerTransport_->nvl_local_rank();
  XLOG(INFO) << "Rank " << globalRank << ": NVL local rank " << nvlLocalRank
             << "/" << nvlNRanks;

  EXPECT_GE(nvlNRanks, 1) << "Should have at least 1 NVL rank (self)";
  EXPECT_GE(nvlLocalRank, 0) << "NVL local rank should be >= 0";
  EXPECT_LT(nvlLocalRank, nvlNRanks) << "NVL local rank should be < nvlNRanks";
}

TEST_F(MultiPeerTransportTest, DeviceHandle) {
  auto comm = makeCtranComm();

  ASSERT_NE(comm->multiPeerTransport_, nullptr);

  // Get device handle - this should work after exchange()
  auto deviceHandle = comm->multiPeerTransport_->get_device_handle();

  // Verify device handle has valid data
  EXPECT_EQ(deviceHandle.myRank, globalRank);
  EXPECT_EQ(deviceHandle.nRanks, numRanks);
  EXPECT_NE(deviceHandle.transports.data(), nullptr)
      << "Device handle should have valid transports array";
  EXPECT_EQ(deviceHandle.transports.size(), static_cast<size_t>(numRanks))
      << "Device handle transports should have nRanks entries";

  XLOG(INFO) << "Rank " << globalRank
             << ": MultiPeerTransport device handle created successfully"
             << ", numNvlPeers=" << deviceHandle.numNvlPeers
             << ", numIbPeers=" << deviceHandle.numIbPeers;
}

// Verify that the P2pNvlTransportDevice objects constructed by CtranAlgo
// use buffer pointers that match the SharedResource staging buffers.
// This catches buffer cross-wiring bugs where the transport would reference
// invalid memory.
TEST_F(MultiPeerTransportTest, TransportBufferPointersMatchStagingBuffers) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks, got " << numRanks;
  }

  auto comm = makeCtranComm();
  ASSERT_NE(comm->multiPeerTransport_, nullptr);

  auto* algo = comm->ctran_->algo.get();
  ASSERT_NE(algo, nullptr);

  auto* nvlTransportsBase = algo->getNvlTransportsBase();
  ASSERT_NE(nvlTransportsBase, nullptr)
      << "nvlTransports should be allocated after initKernelResources";

  auto* statex = comm->statex_.get();
  int myLocalRank = statex->localRank();
  int nLocalRanks = statex->nLocalRanks();

  // Copy CtranAlgoDeviceState from device to host to get staging buffer
  // pointers (devState_ is private, but getDevState() returns device ptr).
  CtranAlgoDeviceState devStateHost;
  CUDACHECK_TEST(cudaMemcpy(
      &devStateHost,
      algo->getDevState(),
      sizeof(CtranAlgoDeviceState),
      cudaMemcpyDeviceToHost));

  for (int peer = 0; peer < nLocalRanks; peer++) {
    if (peer == myLocalRank) {
      continue;
    }

    // Copy the P2pNvlTransportDevice from device memory back to host.
    // P2pNvlTransportDevice has const members so default ctor is deleted;
    // use a raw byte buffer and reinterpret_cast.
    alignas(comms::pipes::P2pNvlTransportDevice) char
        buf[sizeof(comms::pipes::P2pNvlTransportDevice)];
    CUDACHECK_TEST(cudaMemcpy(
        buf,
        &nvlTransportsBase[peer],
        sizeof(comms::pipes::P2pNvlTransportDevice),
        cudaMemcpyDeviceToHost));
    auto& transportHost =
        *reinterpret_cast<comms::pipes::P2pNvlTransportDevice*>(buf);

    // Verify data buffer pointers match SharedResource staging buffers
    char* expectedLocalData =
        static_cast<char*>(devStateHost.localStagingBufsMap[peer]);
    char* expectedRemoteData =
        static_cast<char*>(devStateHost.remoteStagingBufsMap[peer]);

    ASSERT_NE(expectedLocalData, nullptr)
        << "localStagingBufsMap[" << peer << "] should not be null";
    ASSERT_NE(expectedRemoteData, nullptr)
        << "remoteStagingBufsMap[" << peer << "] should not be null";

    EXPECT_EQ(transportHost.getLocalState().dataBuffer, expectedLocalData)
        << "Rank " << globalRank
        << ": P2pNvlTransportDevice local data buffer for peer " << peer
        << " does not match SharedResource staging buffer";
    EXPECT_EQ(transportHost.getRemoteState().dataBuffer, expectedRemoteData)
        << "Rank " << globalRank
        << ": P2pNvlTransportDevice remote data buffer for peer " << peer
        << " does not match SharedResource staging buffer";

    XLOG(INFO) << "Rank " << globalRank << ": peer " << peer
               << " buffer pointers verified"
               << " localData=" << static_cast<void*>(expectedLocalData)
               << " remoteData=" << static_cast<void*>(expectedRemoteData);
  }
}

// Verify that the staging buffers wired into the transport are actually
// accessible via IPC by writing a pattern and reading it from the peer.
// This catches issues like stale IPC handles, unmapped memory, or
// incorrect offset calculations that would cause segfaults or silent
// data corruption.
TEST_F(MultiPeerTransportTest, StagingBufferIpcAccessible) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks, got " << numRanks;
  }

  auto comm = makeCtranComm();
  ASSERT_NE(comm->multiPeerTransport_, nullptr);

  auto* statex = comm->statex_.get();
  int myLocalRank = statex->localRank();
  int nLocalRanks = statex->nLocalRanks();

  // For simplicity, test with 1 peer (the next local rank)
  int peerLocalRank = (myLocalRank + 1) % nLocalRanks;

  // Copy CtranAlgoDeviceState from device to host to get staging buffer ptrs.
  auto* algo = comm->ctran_->algo.get();
  CtranAlgoDeviceState devStateHost;
  CUDACHECK_TEST(cudaMemcpy(
      &devStateHost,
      algo->getDevState(),
      sizeof(CtranAlgoDeviceState),
      cudaMemcpyDeviceToHost));

  // Each rank writes its globalRank into its local staging buffer for this
  // peer. The peer should see this value through its remote staging buffer
  // pointer.
  constexpr size_t kNumElements = 256;
  auto* localDataBuffer =
      static_cast<int*>(devStateHost.localStagingBufsMap[peerLocalRank]);
  ASSERT_NE(localDataBuffer, nullptr);

  comms::pipes::test::fillBuffer(localDataBuffer, globalRank, kNumElements);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Barrier to ensure all ranks have written their patterns
  comm->bootstrap_->barrier(comm->statex_->rank(), comm->statex_->nRanks())
      .get();

  // Now read from the remote staging buffer (IPC pointer to peer's local
  // buffer) and verify we see the peer's globalRank value.
  auto* remoteDataBuffer =
      static_cast<int*>(devStateHost.remoteStagingBufsMap[peerLocalRank]);
  ASSERT_NE(remoteDataBuffer, nullptr);

  int peerGlobalRank = statex->localRankToRanks()[peerLocalRank];

  DeviceBuffer errorCountBuffer(sizeof(int));
  auto* d_errorCount = static_cast<int*>(errorCountBuffer.get());
  CUDACHECK_TEST(cudaMemset(d_errorCount, 0, sizeof(int)));

  comms::pipes::test::verifyBuffer(
      remoteDataBuffer, peerGlobalRank, kNumElements, d_errorCount);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  int h_errorCount = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &h_errorCount, d_errorCount, sizeof(int), cudaMemcpyDeviceToHost));

  ASSERT_EQ(h_errorCount, 0)
      << "Rank " << globalRank << " found " << h_errorCount
      << " errors reading peer " << peerGlobalRank
      << "'s staging buffer through IPC. "
      << "This indicates the external buffer wiring is incorrect.";

  XLOG(INFO) << "Rank " << globalRank << ": IPC read from peer "
             << peerGlobalRank << "'s staging buffer verified (" << kNumElements
             << " elements)";

  comm->bootstrap_->barrier(comm->statex_->rank(), comm->statex_->nRanks())
      .get();
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MultiPeerTransportEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
