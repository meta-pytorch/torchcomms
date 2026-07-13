// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <folly/init/Init.h>

#include <memory>
#include <new>
#include <string>
#include <vector>

#include "comms/prims/collectives/AllGatherLauncher.h"
#include "comms/prims/collectives/ReduceScatterLauncher.h"
#include "comms/prims/collectives/tests/AllGatherTestHarness.h"
#include "comms/prims/collectives/tests/ReduceScatterTestHarness.h"
#include "comms/prims/transport/nvl/MultiPeerNvlTransport.h"

using meta::comms::DeviceBuffer;

namespace comms::prims::test {

namespace {

constexpr std::size_t kNvlPerChannelSize = 128 * 1024;

struct DirectNvlTestParams {
  std::size_t bytes{0};
  int num_blocks{8};
  bool in_place{false};
  std::string name;
};

auto param_name = [](const ::testing::TestParamInfo<DirectNvlTestParams>& p) {
  return p.param.name;
};

std::unique_ptr<MultiPeerNvlTransport> make_nvl_transport(
    int globalRank,
    int worldSize,
    std::shared_ptr<meta::comms::IBootstrap> bootstrap,
    std::size_t perChannelSize,
    int numBlocks) {
  MultiPeerNvlTransportConfig config{
      .pipelineDepth = 2,
      .p2pSignalCount = static_cast<std::size_t>(numBlocks),
      .maxNumChannels = numBlocks,
      .perChannelSize = perChannelSize,
      .memSharingMode = MemSharingMode::kCudaIpc,
  };
  auto transport = std::make_unique<MultiPeerNvlTransport>(
      globalRank, worldSize, bootstrap, config);
  transport->exchange();
  return transport;
}

class DirectAllGatherNvlTest
    : public AllGatherTestBase,
      public ::testing::WithParamInterface<DirectNvlTestParams> {};

class DirectReduceScatterNvlTest
    : public ReduceScatterTestBase,
      public ::testing::WithParamInterface<DirectNvlTestParams> {};

TEST_P(DirectAllGatherNvlTest, Correctness) {
  const auto& params = GetParam();

  if (worldSize < 2 || worldSize > kDirectNvlMaxRanks) {
    GTEST_SKIP() << "Direct NVLink allgather requires 2.." << kDirectNvlMaxRanks
                 << " ranks";
  }

  std::unique_ptr<MultiPeerNvlTransport> transport;
  try {
    transport = make_nvl_transport(
        globalRank,
        worldSize,
        bootstrap,
        kNvlPerChannelSize,
        params.num_blocks);
  } catch (const std::exception& e) {
    GTEST_SKIP() << "NVLink transport not available: " << e.what();
  }

  DeviceBuffer recvbuf(params.bytes * worldSize);
  CUDACHECK_TEST(cudaMemset(recvbuf.get(), 0, params.bytes * worldSize));

  DeviceBuffer sendbuf(params.bytes);
  const auto ownOffset = static_cast<std::size_t>(globalRank) * params.bytes;
  char* ownRecvSlot = static_cast<char*>(recvbuf.get()) + ownOffset;
  char* sendbuf_d =
      params.in_place ? ownRecvSlot : static_cast<char*>(sendbuf.get());
  fill_sendbuf(sendbuf_d, params.bytes);

  DirectAllgatherNvlLaunchParams launchParams{};
  launchParams.my_rank = globalRank;
  launchParams.num_ranks = worldSize;
  launchParams.sendcount = params.bytes;
  launchParams.signaling_data_size = 0;
  launchParams.sendbuf = sendbuf_d;
  launchParams.recvbuf = static_cast<char*>(recvbuf.get());
  launchParams.in_place = params.in_place;
  launchParams.num_blocks = params.num_blocks;
  launchParams.timeout_ms = 30000.0f;

  for (int peer = 0; peer < worldSize; ++peer) {
    if (peer == globalRank) {
      continue;
    }
    new (&launchParams.peers[peer])
        P2pNvlTransportDevice(transport->getP2pTransportDevice(peer));
  }

  bootstrap->barrierAll();
  launch_direct_allgather_nvl(launchParams);
  CUDACHECK_TEST(cudaDeviceSynchronize());
  bootstrap->barrierAll();

  verify_allgather(static_cast<const char*>(recvbuf.get()), params.bytes);
}

TEST_P(DirectReduceScatterNvlTest, Correctness) {
  const auto& params = GetParam();

  if (worldSize < 2 || worldSize > kDirectNvlMaxRanks) {
    GTEST_SKIP() << "Direct NVLink reduce-scatter requires 2.."
                 << kDirectNvlMaxRanks << " ranks";
  }

  std::unique_ptr<MultiPeerNvlTransport> transport;
  try {
    transport = make_nvl_transport(
        globalRank,
        worldSize,
        bootstrap,
        kNvlPerChannelSize,
        params.num_blocks);
  } catch (const std::exception& e) {
    GTEST_SKIP() << "NVLink transport not available: " << e.what();
  }

  const std::size_t chunk_elements = params.bytes / sizeof(float);
  const std::size_t total_elements = chunk_elements * worldSize;
  DeviceBuffer inputBuf(total_elements * sizeof(float));
  DeviceBuffer outputBuf(params.bytes);
  CUDACHECK_TEST(cudaMemset(outputBuf.get(), 0, params.bytes));

  fill_input(static_cast<float*>(inputBuf.get()), total_elements);

  DirectReduceScatterNvlLaunchParams launchParams{};
  launchParams.my_rank = globalRank;
  launchParams.num_ranks = worldSize;
  launchParams.chunk_elements = chunk_elements;
  launchParams.signaling_data_size = 0;
  launchParams.input = static_cast<const float*>(inputBuf.get());
  launchParams.output = static_cast<float*>(outputBuf.get());
  launchParams.num_blocks = params.num_blocks;
  launchParams.timeout_ms = 30000.0f;

  for (int peer = 0; peer < worldSize; ++peer) {
    if (peer == globalRank) {
      continue;
    }
    new (&launchParams.peers[peer])
        P2pNvlTransportDevice(transport->getP2pTransportDevice(peer));
  }

  bootstrap->barrierAll();
  launch_direct_reduce_scatter_nvl(launchParams);
  CUDACHECK_TEST(cudaDeviceSynchronize());
  bootstrap->barrierAll();

  verify_reduce_scatter(
      static_cast<const float*>(outputBuf.get()), chunk_elements);
}

INSTANTIATE_TEST_SUITE_P(
    Direct,
    DirectAllGatherNvlTest,
    ::testing::Values(
        DirectNvlTestParams{
            .bytes = 64 * 1024,
            .num_blocks = 4,
            .name = "64KB_4B"},
        DirectNvlTestParams{
            .bytes = 1024 * 1024,
            .num_blocks = 8,
            .name = "1MB_8B"},
        DirectNvlTestParams{
            .bytes = 64 * 1024,
            .num_blocks = 4,
            .in_place = true,
            .name = "64KB_4B_InPlace"}),
    param_name);

INSTANTIATE_TEST_SUITE_P(
    Direct,
    DirectReduceScatterNvlTest,
    ::testing::Values(
        DirectNvlTestParams{
            .bytes = 64 * 1024,
            .num_blocks = 4,
            .name = "64KB_4B"},
        DirectNvlTestParams{
            .bytes = 1024 * 1024,
            .num_blocks = 8,
            .name = "1MB_8B"}),
    param_name);

} // namespace

} // namespace comms::prims::test

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  ::testing::AddGlobalTestEnvironment(new meta::comms::BenchmarkEnvironment());
  return RUN_ALL_TESTS();
}
