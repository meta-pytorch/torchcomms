// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include "comms/prims/tests/MultiPeerTransportKernelTest.cuh"
#include "comms/prims/transport/MultiPeerDeviceHandle.cuh"
#include "comms/prims/transport/MultiPeerTransport.h"
#include "comms/prims/transport/Transport.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

using namespace meta::comms;

namespace comms::prims::tests {

static constexpr int kNumBlocks = 1;
static constexpr int kBlockSize = 32;

class MultiPeerIntegrationTestFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }

  std::unique_ptr<MultiPeerTransport> create_and_exchange() {
    MultiPeerTransportConfig config{
        .nvlConfig =
            {
                .pipelineDepth = 4,
                .p2pSignalCount = 4,
                .maxNumChannels = 64,
                .perChannelSize = 4 * 1024,
            },
        .ibConfig =
            {
                .cudaDevice = localRank,
            },
    };
    auto bootstrap = std::make_shared<MpiBootstrap>();
    auto states = std::make_unique<MultiPeerTransport>(
        globalRank, numRanks, localRank, bootstrap, config);
    states->exchange();
    return states;
  }
};

// =============================================================================
// Device-side type map verification
// =============================================================================

// Verify that the per-rank transport type array on GPU matches the host-side
// topology. This is the most fundamental device-side test.
TEST_F(MultiPeerIntegrationTestFixture, DeviceHandleTypeMap) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks, got " << numRanks;
  }

  auto states = create_and_exchange();
  auto handle = states->get_device_handle();

  // Allocate output array on GPU
  int* output_d = nullptr;
  CUDACHECK_TEST(cudaMalloc(&output_d, numRanks * sizeof(int)));
  CUDACHECK_TEST(cudaMemset(output_d, -1, numRanks * sizeof(int)));

  test::test_device_handle_type_map(handle, output_d, kNumBlocks, kBlockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Copy back to host and verify
  std::vector<int> output_h(numRanks);
  CUDACHECK_TEST(cudaMemcpy(
      output_h.data(),
      output_d,
      numRanks * sizeof(int),
      cudaMemcpyDeviceToHost));

  for (int r = 0; r < numRanks; ++r) {
    auto expected = static_cast<int>(states->get_transport_type(r));
    EXPECT_EQ(output_h[r], expected)
        << "Rank " << globalRank << ": type mismatch at rank " << r
        << " (device=" << output_h[r] << ", host=" << expected << ")";
  }

  CUDACHECK_TEST(cudaFree(output_d));
  MPI_Barrier(MPI_COMM_WORLD);
}

// =============================================================================
// Self transport via MultiPeerDeviceHandle
// =============================================================================

// Verify self-transport memcpy via the device handle.
TEST_F(MultiPeerIntegrationTestFixture, SelfTransportPut) {
  auto states = create_and_exchange();

  ASSERT_EQ(states->get_transport_type(globalRank), TransportType::SELF);

  auto handle = states->get_device_handle();

  const size_t nbytes = 4096;
  void* src_d = nullptr;
  void* dst_d = nullptr;
  CUDACHECK_TEST(cudaMalloc(&src_d, nbytes));
  CUDACHECK_TEST(cudaMalloc(&dst_d, nbytes));

  // Fill src with pattern
  std::vector<char> pattern(nbytes, static_cast<char>(0xAB));
  CUDACHECK_TEST(
      cudaMemcpy(src_d, pattern.data(), nbytes, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));

  test::test_multi_peer_self_put(
      handle, dst_d, src_d, nbytes, kNumBlocks, kBlockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Verify
  std::vector<char> hostBuf(nbytes);
  CUDACHECK_TEST(
      cudaMemcpy(hostBuf.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

  EXPECT_EQ(hostBuf, pattern) << "Self-transport put data mismatch";

  CUDACHECK_TEST(cudaFree(src_d));
  CUDACHECK_TEST(cudaFree(dst_d));

  MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace comms::prims::tests

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
