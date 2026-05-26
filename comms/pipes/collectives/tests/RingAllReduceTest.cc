// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include "comms/pipes/MultipeerIbgdaTransport.h"
#include "comms/pipes/collectives/RingAllReduceLauncher.h"
#include "comms/pipes/collectives/RingUtils.h"
#include "comms/pipes/collectives/tests/AllReduceTestHarness.h"

using meta::comms::DeviceBuffer;

namespace comms::pipes::test {

namespace {

struct RingAllReduceTestParams {
  AllReduceTestConfig config;
  int num_rings;
  std::size_t data_buffer_size;
  int pipeline_depth;
};

std::string param_name(
    const ::testing::TestParamInfo<RingAllReduceTestParams>& info) {
  return info.param.config.name;
}

struct LaunchContext {
  std::unique_ptr<MultipeerIbgdaTransport> transport;
  RingAllReduceLaunchParams params{};
};

class RingAllReduceTest
    : public AllReduceTestBase,
      public ::testing::WithParamInterface<RingAllReduceTestParams> {
 protected:
  std::optional<LaunchContext> setup_launch(
      const RingAllReduceTestParams& params) {
    const auto& config = params.config;

    LaunchContext ctx;
    try {
      MultipeerIbgdaTransportConfig transportConfig{
          .cudaDevice = localRank,
          .dataBufferSize = params.data_buffer_size,
          .sendRecv =
              MultipeerIbgdaTransportConfig::SendRecvConfig{
                  .maxGroups = config.num_blocks * params.num_rings,
                  .pipelineDepth = params.pipeline_depth,
              },
      };
      ctx.transport = std::make_unique<MultipeerIbgdaTransport>(
          globalRank, worldSize, bootstrap, transportConfig);
      ctx.transport->exchange();
    } catch (const std::exception&) {
      return std::nullopt;
    }

    auto rings_opt =
        make_standard_rings(worldSize, globalRank, params.num_rings);
    if (!rings_opt.has_value()) {
      return std::nullopt;
    }
    auto& rings = *rings_opt;

    ctx.params.my_rank = globalRank;
    ctx.params.num_ranks = worldSize;
    ctx.params.count = config.total_elements;
    ctx.params.num_blocks = config.num_blocks * params.num_rings;
    ctx.params.num_rings = params.num_rings;
    ctx.params.timeout_ms = 30000.0f;

    for (int r = 0; r < params.num_rings; r++) {
      auto& rp = ctx.params.rings[r];
      rp.prev_rank = rings[r].prev_rank;
      rp.next_rank = rings[r].next_rank;
      rp.prev = ctx.transport->getP2pTransportDevice(rings[r].prev_rank);
      rp.next = ctx.transport->getP2pTransportDevice(rings[r].next_rank);
    }

    return ctx;
  }
};

TEST_P(RingAllReduceTest, Correctness) {
  const auto& params = GetParam();
  const auto& config = params.config;

  if (worldSize < 2) {
    GTEST_SKIP() << "Ring allreduce requires at least 2 ranks";
  }
  if (config.total_elements % worldSize != 0) {
    GTEST_SKIP() << "total_elements must be divisible by worldSize";
  }

  auto ctx = setup_launch(params);
  if (!ctx) {
    GTEST_SKIP() << "Transport or ring setup unavailable";
  }

  DeviceBuffer inputBuf(config.total_elements * sizeof(float));
  DeviceBuffer outputBuf(config.total_elements * sizeof(float));
  CUDACHECK_TEST(
      cudaMemset(outputBuf.get(), 0, config.total_elements * sizeof(float)));

  fill_input(static_cast<float*>(inputBuf.get()), config.total_elements);

  ctx->params.input = static_cast<const float*>(inputBuf.get());
  ctx->params.output = static_cast<float*>(outputBuf.get());

  bootstrap->barrierAll();
  launch_ring_allreduce(ctx->params);
  CUDACHECK_TEST(cudaDeviceSynchronize());
  bootstrap->barrierAll();

  verify_allreduce(
      static_cast<const float*>(outputBuf.get()), config.total_elements);
}

TEST_P(RingAllReduceTest, InPlaceCorrectness) {
  const auto& params = GetParam();
  const auto& config = params.config;

  if (worldSize < 2) {
    GTEST_SKIP() << "Ring allreduce requires at least 2 ranks";
  }
  if (config.total_elements % worldSize != 0) {
    GTEST_SKIP() << "total_elements must be divisible by worldSize";
  }

  auto ctx = setup_launch(params);
  if (!ctx) {
    GTEST_SKIP() << "Transport or ring setup unavailable";
  }

  DeviceBuffer buf(config.total_elements * sizeof(float));
  fill_input(static_cast<float*>(buf.get()), config.total_elements);

  ctx->params.input = static_cast<const float*>(buf.get());
  ctx->params.output = static_cast<float*>(buf.get());

  bootstrap->barrierAll();
  launch_ring_allreduce(ctx->params);
  CUDACHECK_TEST(cudaDeviceSynchronize());
  bootstrap->barrierAll();

  verify_allreduce(static_cast<const float*>(buf.get()), config.total_elements);
}

INSTANTIATE_TEST_SUITE_P(
    Ring1,
    RingAllReduceTest,
    ::testing::Values(
        RingAllReduceTestParams{
            .config =
                {.total_elements = 16 * 1024,
                 .num_blocks = 4,
                 .name = "64KB_4B"},
            .num_rings = 1,
            .data_buffer_size = 1024 * 1024,
            .pipeline_depth = 2,
        },
        RingAllReduceTestParams{
            .config =
                {.total_elements = 64 * 1024,
                 .num_blocks = 8,
                 .name = "256KB_8B"},
            .num_rings = 1,
            .data_buffer_size = 1024 * 1024,
            .pipeline_depth = 2,
        },
        RingAllReduceTestParams{
            .config =
                {.total_elements = 256 * 1024,
                 .num_blocks = 16,
                 .name = "1MB_16B"},
            .num_rings = 1,
            .data_buffer_size = 2 * 1024 * 1024,
            .pipeline_depth = 2,
        },
        RingAllReduceTestParams{
            .config =
                {.total_elements = 1024 * 1024,
                 .num_blocks = 16,
                 .name = "4MB_16B"},
            .num_rings = 1,
            .data_buffer_size = 4 * 1024 * 1024,
            .pipeline_depth = 2,
        },
        RingAllReduceTestParams{
            .config =
                {.total_elements = 4 * 1024 * 1024,
                 .num_blocks = 16,
                 .name = "16MB_16B"},
            .num_rings = 1,
            .data_buffer_size = 8 * 1024 * 1024,
            .pipeline_depth = 2,
        }),
    param_name);

INSTANTIATE_TEST_SUITE_P(
    Ring2,
    RingAllReduceTest,
    ::testing::Values(
        RingAllReduceTestParams{
            .config =
                {.total_elements = 64 * 1024,
                 .num_blocks = 8,
                 .name = "256KB_8B_2R"},
            .num_rings = 2,
            .data_buffer_size = 1024 * 1024,
            .pipeline_depth = 2,
        },
        RingAllReduceTestParams{
            .config =
                {.total_elements = 1024 * 1024,
                 .num_blocks = 16,
                 .name = "4MB_16B_2R"},
            .num_rings = 2,
            .data_buffer_size = 4 * 1024 * 1024,
            .pipeline_depth = 2,
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
