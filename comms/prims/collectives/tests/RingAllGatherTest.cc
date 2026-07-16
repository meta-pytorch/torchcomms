// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include "comms/prims/collectives/RingAllgatherLauncher.h"
#include "comms/prims/collectives/RingUtils.h"
#include "comms/prims/collectives/tests/AllGatherTestHarness.h"
#include "comms/prims/transport/ibgda/MultipeerIbgdaTransport.h"

using meta::comms::DeviceBuffer;

namespace comms::prims::test {

namespace {

struct RingAllGatherTestParams {
  AllGatherTestConfig config;
  int num_rings;
  std::size_t data_buffer_size;
  int pipeline_depth;
  bool ibLazyConnect{false};
};

std::string param_name(
    const ::testing::TestParamInfo<RingAllGatherTestParams>& info) {
  return info.param.config.name;
}

class RingAllGatherTest
    : public AllGatherTestBase,
      public ::testing::WithParamInterface<RingAllGatherTestParams> {};

TEST_P(RingAllGatherTest, Correctness) {
  const auto& params = GetParam();
  const auto& config = params.config;

  if (worldSize < 2) {
    GTEST_SKIP() << "Ring allgather requires at least 2 ranks";
  }

  std::unique_ptr<MultipeerIbgdaTransport> transport;
  try {
    const int maxChannels = config.num_blocks * params.num_rings;
    MultipeerIbgdaTransportConfig transportConfig{
        .cudaDevice = localRank,
        .perChannelSize =
            params.data_buffer_size / static_cast<std::size_t>(maxChannels),
        .max_num_channels = maxChannels,
        .pipelineDepth = params.pipeline_depth,
        .ibLazyConnect = params.ibLazyConnect,
    };
    transport = std::make_unique<MultipeerIbgdaTransport>(
        globalRank, worldSize, bootstrap, transportConfig);
    transport->exchange();
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }

  DeviceBuffer sendbuf(config.sendcount);
  DeviceBuffer recvbuf(config.sendcount * worldSize);
  CUDACHECK_TEST(cudaMemset(recvbuf.get(), 0, config.sendcount * worldSize));

  fill_sendbuf(static_cast<char*>(sendbuf.get()), config.sendcount);

  auto rings_opt = make_standard_rings(worldSize, globalRank, params.num_rings);
  ASSERT_TRUE(rings_opt.has_value())
      << "Cannot construct " << params.num_rings << " distinct rings for "
      << worldSize << " ranks";
  auto& rings = *rings_opt;

  RingAllgatherLaunchParams launchParams{};
  launchParams.my_rank = globalRank;
  launchParams.num_ranks = worldSize;
  launchParams.sendcount = config.sendcount;
  launchParams.signaling_data_size = 0;
  launchParams.sendbuf = static_cast<const char*>(sendbuf.get());
  launchParams.recvbuf = static_cast<char*>(recvbuf.get());
  launchParams.num_blocks = config.num_blocks * params.num_rings;
  launchParams.num_rings = params.num_rings;
  launchParams.timeout_ms = 30000.0f;

  for (int r = 0; r < params.num_rings; r++) {
    auto& ringParams = launchParams.rings[r];
    ringParams.prev_rank = rings[r].prev_rank;
    ringParams.next_rank = rings[r].next_rank;
    transport->queuePeerForMaterialization(ringParams.prev_rank);
    transport->queuePeerForMaterialization(ringParams.next_rank);
  }
  transport->connectPeers();
  for (int r = 0; r < params.num_rings; r++) {
    auto& ringParams = launchParams.rings[r];
    ringParams.prev = P2pIbTransportDevice(
        transport->getP2pTransportDevice(ringParams.prev_rank));
    ringParams.next = P2pIbTransportDevice(
        transport->getP2pTransportDevice(ringParams.next_rank));
  }

  bootstrap->barrierAll();
  launch_ring_allgather(launchParams);
  CUDACHECK_TEST(cudaDeviceSynchronize());
  bootstrap->barrierAll();

  verify_allgather(static_cast<const char*>(recvbuf.get()), config.sendcount);
}

INSTANTIATE_TEST_SUITE_P(
    Ring1,
    RingAllGatherTest,
    ::testing::Values(
        RingAllGatherTestParams{
            .config =
                {.sendcount = 64 * 1024, .num_blocks = 4, .name = "64KB_4B"},
            .num_rings = 1,
            .data_buffer_size = 1024 * 1024,
            .pipeline_depth = 2,
        },
        RingAllGatherTestParams{
            .config =
                {.sendcount = 256 * 1024, .num_blocks = 8, .name = "256KB_8B"},
            .num_rings = 1,
            .data_buffer_size = 1024 * 1024,
            .pipeline_depth = 2,
        },
        RingAllGatherTestParams{
            .config =
                {.sendcount = 1024 * 1024, .num_blocks = 16, .name = "1MB_16B"},
            .num_rings = 1,
            .data_buffer_size = 2 * 1024 * 1024,
            .pipeline_depth = 2,
        },
        RingAllGatherTestParams{
            .config =
                {.sendcount = 4 * 1024 * 1024,
                 .num_blocks = 16,
                 .name = "4MB_16B"},
            .num_rings = 1,
            .data_buffer_size = 4 * 1024 * 1024,
            .pipeline_depth = 2,
        },
        RingAllGatherTestParams{
            .config =
                {.sendcount = 16 * 1024 * 1024,
                 .num_blocks = 16,
                 .name = "16MB_16B"},
            .num_rings = 1,
            .data_buffer_size = 8 * 1024 * 1024,
            .pipeline_depth = 2,
        }),
    param_name);

INSTANTIATE_TEST_SUITE_P(
    Ring2,
    RingAllGatherTest,
    ::testing::Values(
        RingAllGatherTestParams{
            .config =
                {.sendcount = 256 * 1024,
                 .num_blocks = 8,
                 .name = "256KB_8B_2R"},
            .num_rings = 2,
            .data_buffer_size = 1024 * 1024,
            .pipeline_depth = 2,
        },
        RingAllGatherTestParams{
            .config =
                {.sendcount = 4 * 1024 * 1024,
                 .num_blocks = 16,
                 .name = "4MB_16B_2R"},
            .num_rings = 2,
            .data_buffer_size = 4 * 1024 * 1024,
            .pipeline_depth = 2,
        }),
    param_name);

INSTANTIATE_TEST_SUITE_P(
    LazyMode,
    RingAllGatherTest,
    ::testing::Values(
        RingAllGatherTestParams{
            .config =
                {.sendcount = 64 * 1024,
                 .num_blocks = 8,
                 .name = "256KB_8B_lazy"},
            .num_rings = 1,
            .data_buffer_size = 1024 * 1024,
            .pipeline_depth = 2,
            .ibLazyConnect = true,
        }),
    param_name);

/*
 * Regression for the blocking-payload-window padding deadlock.
 *
 * The blocking, in-order ring must chunk sends by blocking_payload_window()
 * (perBlockSlot * (pipelineDepth - 1)), NOT the raw pipeline_window(). A short
 * launch whose per-block payload is not slot-aligned leaves the persistent
 * channel cursor mid-slot; a second full-window launch then inserts leading
 * protocol padding to realign. Pre-fix, a full raw-pipeline_window send atop
 * that padding waits SLOT_FREE credit that only the not-yet-run matching recv
 * can post -> single-lane deadlock.
 *
 * Two back-to-back launches on the SAME transport reproduce the mid-slot
 * cursor: a short launch (sendcount < perBlockSlot) then a full launch
 * (sendcount >= perBlockSlot * pipelineDepth, so the window clamps to the raw
 * pipeline_window that deadlocks pre-fix). Both must complete without hanging.
 * num_blocks == num_rings == 1 makes the per-block payload equal sendcount
 * exactly.
 */
class RingAllGatherDeadlockRegressionTest : public AllGatherTestBase {};

TEST_F(RingAllGatherDeadlockRegressionTest, PaddingWindowNoDeadlock) {
  if (worldSize < 2) {
    GTEST_SKIP() << "Ring allgather requires at least 2 ranks";
  }

  constexpr int kNumBlocks = 1;
  constexpr int kNumRings = 1;
  constexpr int kPipelineDepth = 2;
  constexpr std::size_t kDataBufferSize = 1024 * 1024;

  const int maxChannels = kNumBlocks * kNumRings;
  const std::size_t perChannelSize =
      kDataBufferSize / static_cast<std::size_t>(maxChannels);
  const std::size_t perBlockSlot = perChannelSize & ~std::size_t{15};
  ASSERT_GT(perBlockSlot, 0u);

  std::unique_ptr<MultipeerIbgdaTransport> transport;
  try {
    MultipeerIbgdaTransportConfig transportConfig{
        .cudaDevice = localRank,
        .perChannelSize = perChannelSize,
        .max_num_channels = maxChannels,
        .pipelineDepth = kPipelineDepth,
    };
    transport = std::make_unique<MultipeerIbgdaTransport>(
        globalRank, worldSize, bootstrap, transportConfig);
    transport->exchange();
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }

  auto rings_opt = make_standard_rings(worldSize, globalRank, kNumRings);
  ASSERT_TRUE(rings_opt.has_value());
  auto& rings = *rings_opt;
  transport->queuePeerForMaterialization(rings[0].prev_rank);
  transport->queuePeerForMaterialization(rings[0].next_rank);
  transport->connectPeers();

  // Short leaves the cursor mid-slot; full clamps to the raw pipeline_window.
  const std::size_t shortBytes = perBlockSlot / 2;
  const std::size_t fullBytes = perBlockSlot * kPipelineDepth * 2;

  auto doLaunch = [&](std::size_t sendcount) {
    DeviceBuffer sendbuf(sendcount);
    DeviceBuffer recvbuf(sendcount * worldSize);
    CUDACHECK_TEST(cudaMemset(recvbuf.get(), 0, sendcount * worldSize));
    fill_sendbuf(static_cast<char*>(sendbuf.get()), sendcount);

    RingAllgatherLaunchParams launchParams{};
    launchParams.my_rank = globalRank;
    launchParams.num_ranks = worldSize;
    launchParams.sendcount = sendcount;
    launchParams.signaling_data_size = 0;
    launchParams.sendbuf = static_cast<const char*>(sendbuf.get());
    launchParams.recvbuf = static_cast<char*>(recvbuf.get());
    launchParams.num_blocks = kNumBlocks * kNumRings;
    launchParams.num_rings = kNumRings;
    launchParams.timeout_ms = 30000.0f;
    auto& ringParams = launchParams.rings[0];
    ringParams.prev_rank = rings[0].prev_rank;
    ringParams.next_rank = rings[0].next_rank;
    ringParams.prev = P2pIbTransportDevice(
        transport->getP2pTransportDevice(ringParams.prev_rank));
    ringParams.next = P2pIbTransportDevice(
        transport->getP2pTransportDevice(ringParams.next_rank));

    bootstrap->barrierAll();
    launch_ring_allgather(launchParams);
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
