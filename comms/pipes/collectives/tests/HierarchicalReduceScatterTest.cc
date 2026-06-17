// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <folly/init/Init.h>

#include <memory>
#include <new>
#include <string>
#include <vector>

#include "comms/pipes/MultiPeerNvlTransport.h"
#include "comms/pipes/MultipeerIbgdaTransport.h"
#include "comms/pipes/bootstrap/NvlBootstrapAdapter.h"
#include "comms/pipes/collectives/HierarchicalReduceScatterLauncher.h"
#include "comms/pipes/collectives/RingUtils.h"
#include "comms/pipes/collectives/tests/ReduceScatterTestHarness.h"

using meta::comms::DeviceBuffer;

namespace comms::pipes::test {

namespace {

struct HierarchicalReduceScatterTestParams {
  ReduceScatterTestConfig config;
  int ib_size;
  int nvl_size;
};

std::string param_name(
    const ::testing::TestParamInfo<HierarchicalReduceScatterTestParams>& info) {
  return info.param.config.name;
}

class HierarchicalReduceScatterTest : public ReduceScatterTestBase,
                                      public ::testing::WithParamInterface<
                                          HierarchicalReduceScatterTestParams> {
};

TEST_P(HierarchicalReduceScatterTest, Correctness) {
  const auto& params = GetParam();
  const auto& config = params.config;

  if (worldSize != params.ib_size * params.nvl_size) {
    GTEST_SKIP() << "Hierarchical reduce-scatter test expects "
                 << params.ib_size * params.nvl_size << " ranks";
  }
  if (localSize < params.nvl_size) {
    GTEST_SKIP() << "Hierarchical reduce-scatter test needs " << params.nvl_size
                 << " local ranks per NVLink group";
  }

  const int ibRank = globalRank / params.nvl_size;
  const int nvlRank = globalRank % params.nvl_size;

  std::unique_ptr<MultipeerIbgdaTransport> ibTransport;
  if (params.ib_size > 1) {
    try {
      MultipeerIbgdaTransportConfig ibConfig{
          .cudaDevice = localRank,
          .dataBufferSize = 1024 * 1024,
          .sendRecv =
              MultipeerIbgdaTransportConfig::SendRecvConfig{
                  .maxGroups = config.num_blocks,
                  .pipelineDepth = 2,
              },
      };
      ibTransport = std::make_unique<MultipeerIbgdaTransport>(
          globalRank, worldSize, bootstrap, ibConfig);
      ibTransport->exchange();
    } catch (const std::exception& e) {
      GTEST_SKIP() << "IBGDA transport not available: " << e.what();
    }
  }

  std::unique_ptr<MultiPeerNvlTransport> nvlTransport;
  if (params.nvl_size > 1) {
    try {
      MultiPeerNvlTransportConfig nvlConfig{
          .dataBufferSize = 1024 * 1024,
          .chunkSize = 1024 * 1024,
          .pipelineDepth = 2,
          .p2pSignalCount = static_cast<std::size_t>(config.num_blocks),
          .tile_max_groups = config.num_blocks,
          .memSharingMode = MemSharingMode::kCudaIpc,
      };
      std::vector<int> nvlRankToGlobal(params.nvl_size);
      for (int peer = 0; peer < params.nvl_size; ++peer) {
        nvlRankToGlobal[peer] = ibRank * params.nvl_size + peer;
      }
      auto nvlBootstrap = std::make_shared<NvlBootstrapAdapter>(
          bootstrap, std::move(nvlRankToGlobal));
      nvlTransport = std::make_unique<MultiPeerNvlTransport>(
          nvlRank, params.nvl_size, nvlBootstrap, nvlConfig);
      nvlTransport->exchange();
    } catch (const std::exception& e) {
      GTEST_SKIP() << "NVLink transport not available: " << e.what();
    }
  }

  const std::size_t totalElements = config.chunk_elements * worldSize;
  const std::size_t chunkBytes = config.chunk_elements * sizeof(float);
  const std::size_t workspaceBytes =
      static_cast<std::size_t>(params.ib_size) * chunkBytes;
  DeviceBuffer inputBuf(totalElements * sizeof(float));
  DeviceBuffer outputBuf(chunkBytes);
  DeviceBuffer workspaceBuf(workspaceBytes);
  CUDACHECK_TEST(cudaMemset(outputBuf.get(), 0, chunkBytes));
  CUDACHECK_TEST(cudaMemset(workspaceBuf.get(), 0, workspaceBytes));

  fill_input(static_cast<float*>(inputBuf.get()), totalElements);

  HierarchicalReduceScatterLaunchParams launchParams{};
  launchParams.num_ranks = worldSize;
  launchParams.ib_rank = ibRank;
  launchParams.ib_size = params.ib_size;
  launchParams.nvl_rank = nvlRank;
  launchParams.nvl_size = params.nvl_size;
  launchParams.chunk_elements = config.chunk_elements;
  launchParams.input = static_cast<const float*>(inputBuf.get());
  launchParams.output = static_cast<float*>(outputBuf.get());
  launchParams.workspace = static_cast<float*>(workspaceBuf.get());
  launchParams.num_blocks = config.num_blocks;
  launchParams.timeout_ms = 30000.0f;

  if (params.ib_size > 1) {
    auto ibRingsOpt = make_standard_rings(params.ib_size, ibRank, 1);
    ASSERT_TRUE(ibRingsOpt.has_value());
    const auto& ibRing = (*ibRingsOpt)[0];
    const int prevGlobal = ibRing.prev_rank * params.nvl_size + nvlRank;
    const int nextGlobal = ibRing.next_rank * params.nvl_size + nvlRank;
    launchParams.ib_ring.prev_rank = ibRing.prev_rank;
    launchParams.ib_ring.next_rank = ibRing.next_rank;
    launchParams.ib_ring.prev = ibTransport->getP2pTransportDevice(prevGlobal);
    launchParams.ib_ring.next = ibTransport->getP2pTransportDevice(nextGlobal);
  }

  for (int peer = 0; peer < params.nvl_size; ++peer) {
    if (peer == nvlRank) {
      continue;
    }
    new (&launchParams.nvl_peers[peer])
        P2pNvlTransportDevice(nvlTransport->getP2pTransportDevice(peer));
  }

  bootstrap->barrierAll();
  launch_hierarchical_reduce_scatter_fused(launchParams);
  CUDACHECK_TEST(cudaDeviceSynchronize());
  bootstrap->barrierAll();

  verify_reduce_scatter(
      static_cast<const float*>(outputBuf.get()), config.chunk_elements);
}

INSTANTIATE_TEST_SUITE_P(
    Hierarchical,
    HierarchicalReduceScatterTest,
    ::testing::Values(
        HierarchicalReduceScatterTestParams{
            .config =
                {
                    .chunk_elements = 16 * 1024,
                    .num_blocks = 4,
                    .name = "Ib2Nvl2_64KB_4B",
                },
            .ib_size = 2,
            .nvl_size = 2,
        },
        HierarchicalReduceScatterTestParams{
            .config =
                {
                    .chunk_elements = 16 * 1024,
                    .num_blocks = 4,
                    .name = "Ib1Nvl4_64KB_4B",
                },
            .ib_size = 1,
            .nvl_size = 4,
        },
        HierarchicalReduceScatterTestParams{
            .config =
                {
                    .chunk_elements = 16 * 1024,
                    .num_blocks = 4,
                    .name = "Ib4Nvl1_64KB_4B",
                },
            .ib_size = 4,
            .nvl_size = 1,
        },
        HierarchicalReduceScatterTestParams{
            .config =
                {
                    .chunk_elements = 1024 * 1024,
                    .num_blocks = 4,
                    .name = "Ib4Nvl2_4MB_4B",
                },
            .ib_size = 4,
            .nvl_size = 2,
        }),
    param_name);

} // namespace

} // namespace comms::pipes::test

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  ::testing::AddGlobalTestEnvironment(new meta::comms::BenchmarkEnvironment());
  return RUN_ALL_TESTS();
}
