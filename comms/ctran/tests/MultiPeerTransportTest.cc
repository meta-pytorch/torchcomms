// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda.h>
#include <cuda_runtime.h>

#include <folly/init/Init.h>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/CtranComm.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/pipes/MultiPeerDeviceHandle.cuh"
#include "comms/pipes/MultiPeerTransport.h"

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

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MultiPeerTransportEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
