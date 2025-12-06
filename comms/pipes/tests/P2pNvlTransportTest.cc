// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#define P2pNvlTransport_TEST_FRIENDS \
  FRIEND_TEST(P2pNvlTransportTestFixture, IpcMemAccess);

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include "comms/pipes/P2pNvlTransport.h"
#include "comms/pipes/tests/P2pNvlTransportTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

using namespace meta::comms;

namespace comms::pipes {

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
      .stateBufferSize = sizeof(int) * numElements,
  };

  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  P2pNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.exchange();
  XLOGF(INFO, "Rank {} created transport and exchanged IPC", globalRank);

  auto p2p = transport.getTransportDevice(peerRank);

  auto localAddr = static_cast<int*>(p2p.myState_.dataBuffer_d);
  auto remoteAddr = static_cast<int*>(p2p.peerState_.dataBuffer_d);
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

} // namespace comms::pipes

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
