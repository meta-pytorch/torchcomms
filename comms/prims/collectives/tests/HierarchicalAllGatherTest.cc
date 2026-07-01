// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <folly/init/Init.h>

#include <memory>
#include <new>
#include <vector>

#include "comms/prims/bootstrap/NvlBootstrapAdapter.h"
#include "comms/prims/collectives/AllGatherLauncher.h"
#include "comms/prims/collectives/RingUtils.h"
#include "comms/prims/collectives/tests/AllGatherTestHarness.h"
#include "comms/prims/transport/ibgda/MultipeerIbgdaTransport.h"
#include "comms/prims/transport/nvl/MultiPeerNvlTransport.h"

using meta::comms::DeviceBuffer;

namespace comms::prims::test {

namespace {

class HierarchicalAllGatherTest : public AllGatherTestBase {
 protected:
  void runCorrectness(bool inPlace) {
    constexpr int kNvlSize = 2;
    constexpr int kIbSize = 2;
    constexpr std::size_t kSendcount = 64 * 1024;
    constexpr int kNumBlocks = 4;

    if (worldSize != kNvlSize * kIbSize) {
      GTEST_SKIP() << "Hierarchical test expects " << kNvlSize * kIbSize
                   << " ranks";
    }

    const int ibRank = globalRank / kNvlSize;
    const int nvlRank = globalRank % kNvlSize;

    std::unique_ptr<MultipeerIbgdaTransport> ibTransport;
    try {
      MultipeerIbgdaTransportConfig ibConfig{
          .cudaDevice = localRank,
          .dataBufferSize = 1024 * 1024,
          .sendRecv =
              MultipeerIbgdaTransportConfig::SendRecvConfig{
                  .maxGroups = kNumBlocks,
                  .pipelineDepth = 2,
              },
      };
      ibTransport = std::make_unique<MultipeerIbgdaTransport>(
          globalRank, worldSize, bootstrap, ibConfig);
      ibTransport->exchange();
    } catch (const std::exception& e) {
      GTEST_SKIP() << "IBGDA transport not available: " << e.what();
    }

    std::unique_ptr<MultiPeerNvlTransport> nvlTransport;
    try {
      MultiPeerNvlTransportConfig nvlConfig{
          .dataBufferSize = 1024 * 1024,
          .chunkSize = 1024 * 1024,
          .pipelineDepth = 2,
          .p2pSignalCount = static_cast<std::size_t>(kNumBlocks),
          .max_num_channels = kNumBlocks,
          .perChannelSize = (1024 * 1024) / kNumBlocks,
          .memSharingMode = MemSharingMode::kCudaIpc,
      };
      std::vector<int> nvlRankToGlobal(kNvlSize);
      for (int peer = 0; peer < kNvlSize; ++peer) {
        nvlRankToGlobal[peer] = ibRank * kNvlSize + peer;
      }
      auto nvlBootstrap = std::make_shared<NvlBootstrapAdapter>(
          bootstrap, std::move(nvlRankToGlobal));
      nvlTransport = std::make_unique<MultiPeerNvlTransport>(
          nvlRank, kNvlSize, nvlBootstrap, nvlConfig);
      nvlTransport->exchange();
    } catch (const std::exception& e) {
      GTEST_SKIP() << "NVLink transport not available: " << e.what();
    }

    DeviceBuffer sendbuf(kSendcount);
    DeviceBuffer recvbuf(kSendcount * worldSize);
    CUDACHECK_TEST(cudaMemset(recvbuf.get(), 0, kSendcount * worldSize));

    const auto ownOffset = static_cast<std::size_t>(globalRank) * kSendcount;
    char* ownRecvSlot = static_cast<char*>(recvbuf.get()) + ownOffset;
    char* sendbuf_d = inPlace ? ownRecvSlot : static_cast<char*>(sendbuf.get());
    fill_sendbuf(sendbuf_d, kSendcount);

    auto ibRingsOpt = make_standard_rings(kIbSize, ibRank, 1);
    ASSERT_TRUE(ibRingsOpt.has_value());
    const auto& ibRings = *ibRingsOpt;

    HierarchicalAllgatherLaunchParams launchParams{};
    launchParams.num_ranks = worldSize;
    launchParams.ib_rank = ibRank;
    launchParams.ib_size = kIbSize;
    launchParams.nvl_rank = nvlRank;
    launchParams.nvl_size = kNvlSize;
    launchParams.sendcount = kSendcount;
    launchParams.sendbuf = sendbuf_d;
    launchParams.recvbuf = static_cast<char*>(recvbuf.get());
    launchParams.in_place = inPlace;
    launchParams.ib_num_blocks = kNumBlocks;
    launchParams.timeout_ms = 30000.0f;

    const auto& ibRing = ibRings[0];
    const int prevGlobal = ibRing.prev_rank * kNvlSize + nvlRank;
    const int nextGlobal = ibRing.next_rank * kNvlSize + nvlRank;
    launchParams.ib_ring.prev_rank = ibRing.prev_rank;
    launchParams.ib_ring.next_rank = ibRing.next_rank;
    launchParams.ib_ring.prev = ibTransport->getP2pTransportDevice(prevGlobal);
    launchParams.ib_ring.next = ibTransport->getP2pTransportDevice(nextGlobal);

    for (int peer = 0; peer < kNvlSize; ++peer) {
      if (peer == nvlRank) {
        continue;
      }
      new (&launchParams.nvl_peers[peer])
          P2pNvlTransportDevice(nvlTransport->getP2pTransportDevice(peer));
    }

    bootstrap->barrierAll();
    launch_hierarchical_allgather_fused(launchParams);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    bootstrap->barrierAll();

    verify_allgather(static_cast<const char*>(recvbuf.get()), kSendcount);
  }
};

TEST_F(HierarchicalAllGatherTest, Correctness) {
  runCorrectness(false);
}

TEST_F(HierarchicalAllGatherTest, InPlaceCorrectness) {
  runCorrectness(true);
}

} // namespace

} // namespace comms::prims::test

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  ::testing::AddGlobalTestEnvironment(new meta::comms::BenchmarkEnvironment());
  return RUN_ALL_TESTS();
}
