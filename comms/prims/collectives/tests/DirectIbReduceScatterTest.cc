// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include <memory>
#include <string>
#include <vector>

#include "comms/prims/collectives/ReduceScatterDirectIbLauncher.h"
#include "comms/prims/collectives/tests/ReduceScatterTestHarness.h"
#include "comms/prims/transport/ibgda/MultipeerIbgdaTransport.h"

using meta::comms::DeviceBuffer;

namespace comms::prims::test {

namespace {

struct DirectIbReduceScatterTestParams {
  std::size_t chunk_elements;
  bool quantized;
  std::string name;
};

std::string param_name(
    const ::testing::TestParamInfo<DirectIbReduceScatterTestParams>& info) {
  return info.param.name;
}

class DirectIbReduceScatterTest
    : public ReduceScatterTestBase,
      public ::testing::WithParamInterface<DirectIbReduceScatterTestParams> {};

TEST_P(DirectIbReduceScatterTest, Correctness) {
  const auto& params = GetParam();

  if (worldSize < 2) {
    GTEST_SKIP() << "Direct IB reduce-scatter requires at least 2 ranks";
  }

  // The direct IB kernel launches one CTA per logical channel
  // (total_groups == gridDim.x), so max_num_channels must cover num_blocks.
  const int num_blocks = 8;
  const int channels = num_blocks;
  const std::size_t per_channel_size = 512UL * 1024;

  MultipeerIbgdaTransportConfig transport_config{
      .cudaDevice = localRank,
      .perChannelSize = per_channel_size,
      .max_num_channels = channels,
      .pipelineDepth = 4,
      .qpsPerConnection = 2,
  };
  MultipeerIbgdaTransport transport(
      globalRank, worldSize, bootstrap, transport_config);
  transport.exchange();
  for (int peer = 0; peer < worldSize; ++peer) {
    if (peer == globalRank) {
      continue;
    }
    transport.queuePeerForMaterialization(peer);
  }
  transport.connectPeers();

  const std::size_t total_elements = params.chunk_elements * worldSize;
  DeviceBuffer inputBuf(total_elements * sizeof(float));
  DeviceBuffer outputBuf(params.chunk_elements * sizeof(float));
  DeviceBuffer repeatOutputBuf(params.chunk_elements * sizeof(float));
  DeviceBuffer seedBuf(sizeof(std::uint64_t));
  CUDACHECK_TEST(
      cudaMemset(outputBuf.get(), 0, params.chunk_elements * sizeof(float)));

  fill_input(static_cast<float*>(inputBuf.get()), total_elements);
  const std::uint64_t seed = 0x123456789abcdef0ULL;
  CUDACHECK_TEST(
      cudaMemcpy(seedBuf.get(), &seed, sizeof(seed), cudaMemcpyHostToDevice));

  DirectReduceScatterIbLaunchParams launchParams{};
  launchParams.my_rank = globalRank;
  launchParams.num_ranks = worldSize;
  launchParams.chunk_elements = params.chunk_elements;
  launchParams.signaling_data_size = 0;
  launchParams.input = static_cast<const float*>(inputBuf.get());
  launchParams.output = static_cast<float*>(outputBuf.get());
  launchParams.quantized = params.quantized;
  launchParams.seed_ptr = params.quantized
      ? static_cast<const std::uint64_t*>(seedBuf.get())
      : nullptr;
  launchParams.num_blocks = num_blocks;
  launchParams.timeout_ms = 30000.0f;
  for (int peer = 0; peer < worldSize; ++peer) {
    if (peer == globalRank) {
      continue;
    }
    launchParams.peers[peer] =
        P2pIbTransportDevice(transport.getP2pTransportDevice(peer));
  }

  bootstrap->barrierAll();
  launch_direct_reduce_scatter_ib(launchParams);
  CUDACHECK_TEST(cudaDeviceSynchronize());
  bootstrap->barrierAll();

  if (params.quantized) {
    launchParams.output = static_cast<float*>(repeatOutputBuf.get());
    bootstrap->barrierAll();
    launch_direct_reduce_scatter_ib(launchParams);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    bootstrap->barrierAll();

    std::vector<float> output(params.chunk_elements);
    std::vector<float> repeatOutput(params.chunk_elements);
    CUDACHECK_TEST(cudaMemcpy(
        output.data(),
        outputBuf.get(),
        params.chunk_elements * sizeof(float),
        cudaMemcpyDeviceToHost));
    CUDACHECK_TEST(cudaMemcpy(
        repeatOutput.data(),
        repeatOutputBuf.get(),
        params.chunk_elements * sizeof(float),
        cudaMemcpyDeviceToHost));
    EXPECT_EQ(output, repeatOutput);
  }

  verify_reduce_scatter(
      static_cast<const float*>(outputBuf.get()),
      params.chunk_elements,
      params.quantized ? 5e-2f : 1e-2f);
}

std::vector<DirectIbReduceScatterTestParams> all_test_params() {
  // 8 MB and 32 MB total; chunk_elements = total_bytes / worldSize / 4. Use a
  // fixed worldSize of 8 (ppn) for the label math; chunk_elements below assume
  // an 8-rank job, matching the test configuration.
  const std::vector<std::pair<std::string, std::size_t>> sizes{
      {"8MB", 8UL * 1024 * 1024},
      {"32MB", 32UL * 1024 * 1024},
  };

  constexpr int kExpectedRanks = 8;
  std::vector<DirectIbReduceScatterTestParams> out;
  for (const auto& [size_label, total_bytes] : sizes) {
    const std::size_t chunk_elements =
        total_bytes / sizeof(float) / kExpectedRanks;
    out.push_back(
        {.chunk_elements = chunk_elements,
         .quantized = false,
         .name = size_label + "Fp32"});
    out.push_back(
        {.chunk_elements = chunk_elements,
         .quantized = true,
         .name = size_label + "Quantized"});
  }
  out.push_back(
      {.chunk_elements = 1025, .quantized = true, .name = "OddTailQuantized"});
  return out;
}

INSTANTIATE_TEST_SUITE_P(
    Sizes,
    DirectIbReduceScatterTest,
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
