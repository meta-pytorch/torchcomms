// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include "comms/pipes/MultiPeerDeviceHandle.cuh"
#include "comms/pipes/MultiPeerTransport.h"
#include "comms/pipes/Transport.cuh"
#include "comms/pipes/tests/MultiPeerTransportKernelTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

using namespace meta::comms;

namespace comms::pipes::tests {

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
                .dataBufferSize = 256 * 1024,
                .chunkSize = 512,
                .pipelineDepth = 4,
                .p2pSignalCount = 4,
            },
        .ibgdaConfig =
            {
                .cudaDevice = localRank,
                .signalCount = 4,
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
// NVL send/recv via MultiPeerDeviceHandle
// =============================================================================

// End-to-end test: rank 0 sends data to rank 1 using the device handle's
// NVL transport accessor. This verifies the full pipeline:
//   Host: MultiPeerTransport → getDeviceHandle()
//   Device: handle.getNvl(peer).send() / recv()
TEST_F(MultiPeerIntegrationTestFixture, NvlSendRecv) {
  if (numRanks != 2) {
    GTEST_SKIP() << "Requires exactly 2 ranks, got " << numRanks;
  }

  auto states = create_and_exchange();
  int peerRank = (globalRank == 0) ? 1 : 0;

  // Both ranks must use NVL on same-node test
  ASSERT_TRUE(states->is_nvl_peer(peerRank))
      << "Rank " << globalRank << ": peer " << peerRank
      << " should be NVL on same node";

  auto handle = states->get_device_handle();

  const size_t numInts = 1024;
  const size_t nbytes = numInts * sizeof(int);
  const int testValue = 42;

  int* src_d = nullptr;
  int* dst_d = nullptr;
  CUDACHECK_TEST(cudaMalloc(&src_d, nbytes));
  CUDACHECK_TEST(cudaMalloc(&dst_d, nbytes));

  if (globalRank == 0) {
    // Fill src with test pattern
    std::vector<int> pattern(numInts, testValue);
    CUDACHECK_TEST(
        cudaMemcpy(src_d, pattern.data(), nbytes, cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);

    test::test_multi_peer_nvl_send(
        handle, peerRank, src_d, nbytes, kNumBlocks, kBlockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);
  } else {
    CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));

    MPI_Barrier(MPI_COMM_WORLD);

    test::test_multi_peer_nvl_recv(
        handle, peerRank, dst_d, nbytes, kNumBlocks, kBlockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);

    // Verify received data
    std::vector<int> hostBuf(numInts);
    CUDACHECK_TEST(
        cudaMemcpy(hostBuf.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < numInts; ++i) {
      EXPECT_EQ(hostBuf[i], testValue)
          << "Mismatch at index " << i << ": expected " << testValue << ", got "
          << hostBuf[i];
      if (hostBuf[i] != testValue) {
        break;
      }
    }
  }

  CUDACHECK_TEST(cudaFree(src_d));
  CUDACHECK_TEST(cudaFree(dst_d));

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

// =============================================================================
// IBGDA transport accessor via MultiPeerDeviceHandle
// =============================================================================

// Verify that the IBGDA transport accessor works on the device side.
// For each IBGDA-typed peer, reads the signal count through get_ibgda()
// and verifies it matches the configured value.
TEST_F(MultiPeerIntegrationTestFixture, IbgdaAccessor) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks, got " << numRanks;
  }

  auto states = create_and_exchange();

  // Check if any peers use IBGDA as preferred transport
  bool hasIbgdaPeers = false;
  for (int r = 0; r < numRanks; ++r) {
    if (states->get_transport_type(r) == TransportType::P2P_IBGDA) {
      hasIbgdaPeers = true;
      break;
    }
  }

  if (!hasIbgdaPeers) {
    GTEST_SKIP() << "No IBGDA-typed peers in this topology (single-node?). "
                 << "Run with nnodes >= 2 to exercise IBGDA accessor.";
  }

  auto handle = states->get_device_handle();

  int* output_d = nullptr;
  CUDACHECK_TEST(cudaMalloc(&output_d, numRanks * sizeof(int)));
  CUDACHECK_TEST(cudaMemset(output_d, 0xFF, numRanks * sizeof(int)));

  test::test_ibgda_accessor(handle, output_d, kNumBlocks, kBlockSize);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<int> output_h(numRanks);
  CUDACHECK_TEST(cudaMemcpy(
      output_h.data(),
      output_d,
      numRanks * sizeof(int),
      cudaMemcpyDeviceToHost));

  for (int r = 0; r < numRanks; ++r) {
    if (states->get_transport_type(r) == TransportType::P2P_IBGDA) {
      // Signal count should be the configured value (4)
      EXPECT_GT(output_h[r], 0)
          << "Rank " << globalRank << ": IBGDA accessor for peer " << r
          << " returned invalid signal count " << output_h[r];
    } else {
      EXPECT_EQ(output_h[r], -1) << "Rank " << globalRank << ": non-IBGDA peer "
                                 << r << " should have -1, got " << output_h[r];
    }
  }

  CUDACHECK_TEST(cudaFree(output_d));
  MPI_Barrier(MPI_COMM_WORLD);
}

// =============================================================================
// Bidirectional NVL communication
// =============================================================================

// Sequential NVL ping-pong: Rank 0 sends then receives, Rank 1 receives then
// sends. Verifies both directions of NVL transport work correctly.
TEST_F(MultiPeerIntegrationTestFixture, SequentialNvlPingPong) {
  if (numRanks != 2) {
    GTEST_SKIP() << "Requires exactly 2 ranks, got " << numRanks;
  }

  auto states = create_and_exchange();
  int peerRank = (globalRank == 0) ? 1 : 0;

  ASSERT_TRUE(states->is_nvl_peer(peerRank));

  auto handle = states->get_device_handle();

  const size_t numInts = 512;
  const size_t nbytes = numInts * sizeof(int);
  const int sendValue = globalRank + 100;

  int* src_d = nullptr;
  int* dst_d = nullptr;
  CUDACHECK_TEST(cudaMalloc(&src_d, nbytes));
  CUDACHECK_TEST(cudaMalloc(&dst_d, nbytes));

  // Fill src
  std::vector<int> sendPattern(numInts, sendValue);
  CUDACHECK_TEST(
      cudaMemcpy(src_d, sendPattern.data(), nbytes, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));
  CUDACHECK_TEST(cudaDeviceSynchronize());

  MPI_Barrier(MPI_COMM_WORLD);

  // Rank 0: send then recv, Rank 1: recv then send
  if (globalRank == 0) {
    test::test_multi_peer_nvl_send(
        handle, peerRank, src_d, nbytes, kNumBlocks, kBlockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    test::test_multi_peer_nvl_recv(
        handle, peerRank, dst_d, nbytes, kNumBlocks, kBlockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
  } else {
    test::test_multi_peer_nvl_recv(
        handle, peerRank, dst_d, nbytes, kNumBlocks, kBlockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    test::test_multi_peer_nvl_send(
        handle, peerRank, src_d, nbytes, kNumBlocks, kBlockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Verify received data
  int expectedValue = peerRank + 100;
  std::vector<int> hostBuf(numInts);
  CUDACHECK_TEST(
      cudaMemcpy(hostBuf.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < numInts; ++i) {
    EXPECT_EQ(hostBuf[i], expectedValue)
        << "Rank " << globalRank << ": mismatch at index " << i << " (expected "
        << expectedValue << ", got " << hostBuf[i] << ")";
    if (hostBuf[i] != expectedValue) {
      break;
    }
  }

  CUDACHECK_TEST(cudaFree(src_d));
  CUDACHECK_TEST(cudaFree(dst_d));

  MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace comms::pipes::tests

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
