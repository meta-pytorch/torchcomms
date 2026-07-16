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
          .perChannelSize = (1024 * 1024) / kNumBlocks,
          .max_num_channels = kNumBlocks,
          .pipelineDepth = 2,
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
          .pipelineDepth = 2,
          .p2pSignalCount = static_cast<std::size_t>(kNumBlocks),
          .maxNumChannels = kNumBlocks,
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

/*
 * Regression for the blocking-payload-window padding deadlock in the overlap
 * kernel (caller #5 of blocking_payload_window()). The overlap kernel's IB
 * blocks do a blocking, in-order send/forward/recv ring (sub-tiling each chunk
 * by next.blocking_payload_window()), concurrently with NVL blocks that wait on
 * ready_counters. A short first launch leaves the IB channel cursor mid-slot; a
 * second full-window launch atop that cursor deadlocks the IB block pre-fix
 * (full raw pipeline_window send waits SLOT_FREE credit only the not-yet-run
 * matching recv can post), which in turn hangs the NVL waiters -> whole-kernel
 * hang. The fix reserves one slot of headroom so the padded send always fits.
 *
 * This is also the only coverage of launch_hierarchical_allgather_overlap, and
 * it validates the ready_sequence reuse contract: ready_counters is zeroed
 * once, then the two launches use strictly-increasing ready_sequence (1, then
 * 2) with NO re-zero -- publish_ready writes the absolute sequence value, so a
 * reused value would let the second launch's NVL waiters observe stale
 * readiness.
 *
 * chunk_bytes == sendcount gives total_chunks == 1 (ready_counters = ib_size
 * slots); ib_num_blocks == 1 makes the per-block IB payload equal sendcount
 * exactly, so short < perBlockSlot and full >= perBlockSlot * pipelineDepth.
 */
TEST_F(HierarchicalAllGatherTest, OverlapPaddingWindowNoDeadlock) {
  constexpr int kNvlSize = 2;
  constexpr int kIbSize = 2;
  constexpr int kIbNumBlocks = 1;
  constexpr int kNvlNumBlocks = 1;
  constexpr int kPipelineDepth = 2;
  constexpr std::size_t kDataBufferSize = 1024 * 1024;

  if (worldSize != kNvlSize * kIbSize) {
    GTEST_SKIP() << "Hierarchical overlap test expects " << kNvlSize * kIbSize
                 << " ranks";
  }

  const int ibRank = globalRank / kNvlSize;
  const int nvlRank = globalRank % kNvlSize;

  const std::size_t ibPerChannel =
      kDataBufferSize / static_cast<std::size_t>(kIbNumBlocks);
  const std::size_t perBlockSlot = ibPerChannel & ~std::size_t{15};
  ASSERT_GT(perBlockSlot, 0u);

  // Short leaves the IB cursor mid-slot; full clamps to the raw
  // pipeline_window.
  const std::size_t shortBytes = perBlockSlot / 2;
  const std::size_t fullBytes = perBlockSlot * kPipelineDepth * 2;

  std::unique_ptr<MultipeerIbgdaTransport> ibTransport;
  try {
    MultipeerIbgdaTransportConfig ibConfig{
        .cudaDevice = localRank,
        .perChannelSize = ibPerChannel,
        .max_num_channels = kIbNumBlocks,
        .pipelineDepth = kPipelineDepth,
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
        .pipelineDepth = kPipelineDepth,
        .p2pSignalCount = static_cast<std::size_t>(kNvlNumBlocks),
        .maxNumChannels = kNvlNumBlocks,
        .perChannelSize = fullBytes,
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

  auto ibRingsOpt = make_standard_rings(kIbSize, ibRank, 1);
  ASSERT_TRUE(ibRingsOpt.has_value());
  const auto& ibRing = (*ibRingsOpt)[0];
  const int prevGlobal = ibRing.prev_rank * kNvlSize + nvlRank;
  const int nextGlobal = ibRing.next_rank * kNvlSize + nvlRank;

  // chunk_bytes == sendcount => total_chunks == 1 => ib_size counter slots.
  // Zeroed once; the two launches advance ready_sequence (1, 2) without
  // re-zero.
  DeviceBuffer readyCounters(
      static_cast<std::size_t>(kIbSize) * sizeof(uint64_t));
  CUDACHECK_TEST(
      cudaMemset(readyCounters.get(), 0, kIbSize * sizeof(uint64_t)));

  uint64_t sequence = 0;
  auto doLaunch = [&](std::size_t sendcount) {
    ++sequence;
    DeviceBuffer sendbuf(sendcount);
    DeviceBuffer recvbuf(sendcount * worldSize);
    CUDACHECK_TEST(cudaMemset(recvbuf.get(), 0, sendcount * worldSize));
    fill_sendbuf(static_cast<char*>(sendbuf.get()), sendcount);

    HierarchicalAllgatherOverlapLaunchParams launchParams{};
    launchParams.num_ranks = worldSize;
    launchParams.ib_rank = ibRank;
    launchParams.ib_size = kIbSize;
    launchParams.nvl_rank = nvlRank;
    launchParams.nvl_size = kNvlSize;
    launchParams.sendcount = sendcount;
    launchParams.chunk_bytes = sendcount;
    launchParams.ready_sequence = sequence;
    launchParams.ready_counters = static_cast<uint64_t*>(readyCounters.get());
    launchParams.sendbuf = static_cast<const char*>(sendbuf.get());
    launchParams.recvbuf = static_cast<char*>(recvbuf.get());
    launchParams.in_place = false;
    launchParams.ib_num_blocks = kIbNumBlocks;
    launchParams.nvl_num_blocks = kNvlNumBlocks;
    launchParams.timeout_ms = 30000.0f;
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
    launch_hierarchical_allgather_overlap(launchParams);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    bootstrap->barrierAll();
    verify_allgather(static_cast<const char*>(recvbuf.get()), sendcount);
  };

  doLaunch(shortBytes);
  doLaunch(fullBytes);
}

} // namespace

} // namespace comms::prims::test

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  ::testing::AddGlobalTestEnvironment(new meta::comms::BenchmarkEnvironment());
  return RUN_ALL_TESTS();
}
