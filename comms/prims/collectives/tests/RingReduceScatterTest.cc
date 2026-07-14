// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include <memory>
#include <vector>

#include "comms/prims/collectives/RingReduceScatterLauncher.h"
#include "comms/prims/collectives/RingUtils.h"
#include "comms/prims/collectives/tests/ReduceScatterTestHarness.h"
#include "comms/prims/transport/ibgda/MultipeerIbgdaTransport.h"
#include "comms/prims/transport/ibrc/MultipeerIbrcTransport.h"

using meta::comms::DeviceBuffer;

namespace comms::prims::test {

namespace {

struct RingReduceScatterTestParams {
  ReduceScatterTestConfig config;
  int num_rings;
  std::size_t data_buffer_size;
  int pipeline_depth;
  bool ibLazyConnect{false};
  bool useIbrc{false};
};

std::string param_name(
    const ::testing::TestParamInfo<RingReduceScatterTestParams>& info) {
  return info.param.config.name;
}

class RingReduceScatterIbTransport {
 public:
  RingReduceScatterIbTransport(
      bool useIbrc,
      int globalRank,
      int worldSize,
      const std::shared_ptr<meta::comms::IBootstrap>& bootstrap,
      const MultipeerIbgdaTransportConfig& transportConfig) {
    if (useIbrc) {
      ibrcTransport_ = std::make_unique<MultipeerIbrcTransport>(
          globalRank, worldSize, bootstrap, transportConfig);
    } else {
      ibgdaTransport_ = std::make_unique<MultipeerIbgdaTransport>(
          globalRank, worldSize, bootstrap, transportConfig);
    }
  }

  void exchange() {
    if (ibrcTransport_) {
      ibrcTransport_->exchange();
    } else {
      ibgdaTransport_->exchange();
    }
  }

  void queuePeerForMaterialization(int peerRank) {
    if (ibrcTransport_) {
      ibrcTransport_->queuePeerForMaterialization(peerRank);
    } else {
      ibgdaTransport_->queuePeerForMaterialization(peerRank);
    }
  }

  void connectPeers() {
    if (ibrcTransport_) {
      ibrcTransport_->connectPeers();
    } else {
      ibgdaTransport_->connectPeers();
    }
  }

  P2pIbTransportDevice getP2pTransportDevice(int peerRank) {
    if (ibrcTransport_) {
      return P2pIbTransportDevice(
          ibrcTransport_->getP2pTransportDevice(peerRank));
    }
    return P2pIbTransportDevice(
        ibgdaTransport_->getP2pTransportDevice(peerRank));
  }

 private:
  std::unique_ptr<MultipeerIbgdaTransport> ibgdaTransport_;
  std::unique_ptr<MultipeerIbrcTransport> ibrcTransport_;
};

class RingReduceScatterTest
    : public ReduceScatterTestBase,
      public ::testing::WithParamInterface<RingReduceScatterTestParams> {};

TEST_P(RingReduceScatterTest, Correctness) {
  const auto& params = GetParam();
  const auto& config = params.config;

  if (worldSize < 2) {
    GTEST_SKIP() << "Ring reduce-scatter requires at least 2 ranks";
  }

  const int maxChannels = config.num_blocks * params.num_rings;
  MultipeerIbgdaTransportConfig transportConfig{
      .cudaDevice = localRank,
      .perChannelSize =
          params.data_buffer_size / static_cast<std::size_t>(maxChannels),
      .max_num_channels = maxChannels,
      .pipelineDepth = params.pipeline_depth,
      .ibLazyConnect = params.ibLazyConnect,
  };
  auto transport = std::make_unique<RingReduceScatterIbTransport>(
      params.useIbrc, globalRank, worldSize, bootstrap, transportConfig);
  transport->exchange();

  const std::size_t total_elements = config.chunk_elements * worldSize;
  DeviceBuffer inputBuf(total_elements * sizeof(float));
  DeviceBuffer outputBuf(config.chunk_elements * sizeof(float));
  CUDACHECK_TEST(
      cudaMemset(outputBuf.get(), 0, config.chunk_elements * sizeof(float)));

  fill_input(static_cast<float*>(inputBuf.get()), total_elements);

  auto rings_opt = make_standard_rings(worldSize, globalRank, params.num_rings);
  ASSERT_TRUE(rings_opt.has_value())
      << "Cannot construct " << params.num_rings << " distinct rings for "
      << worldSize << " ranks";
  auto& rings = *rings_opt;

  RingReduceScatterLaunchParams launchParams{};
  launchParams.my_rank = globalRank;
  launchParams.num_ranks = worldSize;
  launchParams.chunk_elements = config.chunk_elements;
  launchParams.signaling_data_size = 0;
  launchParams.input = static_cast<const float*>(inputBuf.get());
  launchParams.output = static_cast<float*>(outputBuf.get());
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
    ringParams.prev = transport->getP2pTransportDevice(ringParams.prev_rank);
    ringParams.next = transport->getP2pTransportDevice(ringParams.next_rank);
  }

  bootstrap->barrierAll();
  launch_ring_reduce_scatter(launchParams);
  CUDACHECK_TEST(cudaDeviceSynchronize());
  bootstrap->barrierAll();

  verify_reduce_scatter(
      static_cast<const float*>(outputBuf.get()), config.chunk_elements);
}

std::vector<RingReduceScatterTestParams> all_test_params() {
  std::vector<RingReduceScatterTestParams> base{
      RingReduceScatterTestParams{
          .config =
              {.chunk_elements = 16 * 1024, .num_blocks = 4, .name = "64KB_4B"},
          .num_rings = 1,
          .data_buffer_size = 1024 * 1024,
          .pipeline_depth = 2,
      },
      RingReduceScatterTestParams{
          .config =
              {.chunk_elements = 64 * 1024,
               .num_blocks = 8,
               .name = "256KB_8B"},
          .num_rings = 1,
          .data_buffer_size = 1024 * 1024,
          .pipeline_depth = 2,
      },
      RingReduceScatterTestParams{
          .config =
              {.chunk_elements = 256 * 1024,
               .num_blocks = 16,
               .name = "1MB_16B"},
          .num_rings = 1,
          .data_buffer_size = 2 * 1024 * 1024,
          .pipeline_depth = 2,
      },
      RingReduceScatterTestParams{
          .config =
              {.chunk_elements = 1024 * 1024,
               .num_blocks = 16,
               .name = "4MB_16B"},
          .num_rings = 1,
          .data_buffer_size = 4 * 1024 * 1024,
          .pipeline_depth = 2,
      },
      RingReduceScatterTestParams{
          .config =
              {.chunk_elements = 4 * 1024 * 1024,
               .num_blocks = 16,
               .name = "16MB_16B"},
          .num_rings = 1,
          .data_buffer_size = 8 * 1024 * 1024,
          .pipeline_depth = 2,
      },
      RingReduceScatterTestParams{
          .config =
              {.chunk_elements = 64 * 1024,
               .num_blocks = 8,
               .name = "256KB_8B_2R"},
          .num_rings = 2,
          .data_buffer_size = 1024 * 1024,
          .pipeline_depth = 2,
      },
      RingReduceScatterTestParams{
          .config =
              {.chunk_elements = 1024 * 1024,
               .num_blocks = 16,
               .name = "4MB_16B_2R"},
          .num_rings = 2,
          .data_buffer_size = 4 * 1024 * 1024,
          .pipeline_depth = 2,
      },
      RingReduceScatterTestParams{
          .config =
              {.chunk_elements = 64 * 1024,
               .num_blocks = 8,
               .name = "256KB_8B_lazy"},
          .num_rings = 1,
          .data_buffer_size = 1024 * 1024,
          .pipeline_depth = 2,
          .ibLazyConnect = true,
      },
  };

  std::vector<RingReduceScatterTestParams> expanded;
  expanded.reserve(base.size() * 2);
  for (auto params : base) {
    expanded.push_back(params);
    params.useIbrc = true;
    params.config.name += "_ibrc";
    expanded.push_back(params);
  }
  return expanded;
}

INSTANTIATE_TEST_SUITE_P(
    AllBackends,
    RingReduceScatterTest,
    ::testing::ValuesIn(all_test_params()),
    param_name);

} // namespace

} // namespace comms::prims::test

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  ::testing::AddGlobalTestEnvironment(new meta::comms::BenchmarkEnvironment());
  return RUN_ALL_TESTS();
}
