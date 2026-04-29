// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include "comms/common/CudaWrap.h"
#include "comms/pipes/MultiPeerTransport.h"
#include "comms/pipes/TimeoutUtils.h"
#include "comms/pipes/collectives/AllToAllvTile.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

using namespace meta::comms;

namespace comms::pipes::tests {

class AllToAllvTileTestFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }

  std::unique_ptr<MultiPeerTransport> create_and_exchange(
      std::size_t data_buffer_size = 8 * 1024 * 1024) {
    MultiPeerTransportConfig config{
        .nvlConfig =
            {
                .dataBufferSize = data_buffer_size,
                .chunkSize = 64 * 1024,
                .pipelineDepth = 4,
                .tile_max_groups = 32,
            },
        .ibgdaConfig =
            {
                .cudaDevice = localRank,
            },
    };
    auto bootstrap = std::make_shared<MpiBootstrap>();
    auto transport = std::make_unique<MultiPeerTransport>(
        globalRank, numRanks, localRank, bootstrap, config);
    transport->exchange();
    return transport;
  }

  void run_alltoallv_tile_test(
      std::size_t bytes_per_peer,
      int num_blocks,
      std::size_t max_signal_bytes = 0) {
    auto transport = create_and_exchange();
    auto handle = transport->get_device_handle();

    const int nranks = numRanks;
    const std::size_t total_send = bytes_per_peer * nranks;
    const std::size_t total_recv = bytes_per_peer * nranks;

    DeviceBuffer send_buf(total_send);
    DeviceBuffer recv_buf(total_recv);

    // Fill send buffer: byte[peer * bytesPerPeer + i] = (globalRank * 10 +
    // peer) % 256
    std::vector<char> h_send(total_send);
    for (int peer = 0; peer < nranks; peer++) {
      for (std::size_t i = 0; i < bytes_per_peer; i++) {
        h_send[peer * bytes_per_peer + i] =
            static_cast<char>((globalRank * 10 + peer + i) % 256);
      }
    }
    CUDACHECK_TEST(cudaMemcpy(
        send_buf.get(), h_send.data(), total_send, cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemset(recv_buf.get(), 0, total_recv));

    // Build per-peer pointer and count arrays on device
    std::vector<char*> h_send_ptrs(nranks);
    std::vector<char*> h_recv_ptrs(nranks);
    std::vector<std::size_t> h_send_counts(nranks, bytes_per_peer);
    std::vector<std::size_t> h_recv_counts(nranks, bytes_per_peer);

    for (int r = 0; r < nranks; r++) {
      h_send_ptrs[r] = static_cast<char*>(send_buf.get()) + r * bytes_per_peer;
      h_recv_ptrs[r] = static_cast<char*>(recv_buf.get()) + r * bytes_per_peer;
    }

    DeviceBuffer d_send_ptrs(nranks * sizeof(char*));
    DeviceBuffer d_recv_ptrs(nranks * sizeof(char*));
    DeviceBuffer d_send_counts(nranks * sizeof(std::size_t));
    DeviceBuffer d_recv_counts(nranks * sizeof(std::size_t));

    CUDACHECK_TEST(cudaMemcpy(
        d_send_ptrs.get(),
        h_send_ptrs.data(),
        nranks * sizeof(char*),
        cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemcpy(
        d_recv_ptrs.get(),
        h_recv_ptrs.data(),
        nranks * sizeof(char*),
        cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemcpy(
        d_send_counts.get(),
        h_send_counts.data(),
        nranks * sizeof(std::size_t),
        cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemcpy(
        d_recv_counts.get(),
        h_recv_counts.data(),
        nranks * sizeof(std::size_t),
        cudaMemcpyHostToDevice));

    int num_blocks_nvl = num_blocks;
    int num_blocks_ib = num_blocks;
    int total_blocks = 2 * num_blocks;

    AllToAllvTileArgs args{
        .handle = handle,
        .send_ptrs = static_cast<char**>(d_send_ptrs.get()),
        .send_counts = static_cast<std::size_t*>(d_send_counts.get()),
        .recv_ptrs = static_cast<char**>(d_recv_ptrs.get()),
        .recv_counts = static_cast<std::size_t*>(d_recv_counts.get()),
        .num_blocks_nvl = num_blocks_nvl,
        .num_blocks_ib = num_blocks_ib,
        .max_signal_bytes = max_signal_bytes,
    };

    int device = 0;
    CUDACHECK_TEST(cudaGetDevice(&device));
    Timeout timeout = makeTimeout(30000, device);

    MPI_Barrier(MPI_COMM_WORLD);

    dim3 clus(comms::common::kDefaultClusterSize, 1, 1);
    void* kernel_args[] = {&args, &timeout};
    comms::common::launchKernel(
        (void*)alltoallv_tile_1d_kernel,
        dim3(total_blocks),
        dim3(512),
        kernel_args,
        nullptr,
        clus);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);

    // Verify received data
    std::vector<char> h_recv(total_recv);
    CUDACHECK_TEST(cudaMemcpy(
        h_recv.data(), recv_buf.get(), total_recv, cudaMemcpyDeviceToHost));

    for (int peer = 0; peer < nranks; peer++) {
      for (std::size_t i = 0; i < bytes_per_peer; i++) {
        char expected = static_cast<char>((peer * 10 + globalRank + i) % 256);
        char actual = h_recv[peer * bytes_per_peer + i];
        EXPECT_EQ(actual, expected)
            << "Rank " << globalRank << ": mismatch receiving from peer "
            << peer << " at byte " << i << " (expected "
            << static_cast<int>(expected) << ", got "
            << static_cast<int>(actual) << ")";
        if (actual != expected) {
          return;
        }
      }
    }
  }
  void run_alltoallv_tile_ib_test(
      std::size_t bytes_per_peer,
      int num_blocks_ib) {
    MultiPeerTransportConfig config{
        .nvlConfig =
            {
                .dataBufferSize = 8 * 1024 * 1024,
                .chunkSize = 64 * 1024,
                .pipelineDepth = 4,
                .tile_max_groups = 32,
            },
        .ibgdaConfig =
            {
                .cudaDevice = localRank,
                .dataBufferSize = 8 * 1024 * 1024,
                .sendRecv =
                    MultipeerIbgdaTransportConfig::SendRecvConfig{
                        .maxGroups = 32,
                        .pipelineDepth = 2,
                    },
                .numQpsPerPeer = 2,
            },
        .topoConfig =
            {
                .p2pDisable = true,
            },
    };
    auto bootstrap_ptr = std::make_shared<MpiBootstrap>();
    auto transport = std::make_unique<MultiPeerTransport>(
        globalRank, numRanks, localRank, bootstrap_ptr, config);
    transport->exchange();
    auto handle = transport->get_device_handle();

    const int nranks = numRanks;
    const std::size_t total_send = bytes_per_peer * nranks;
    const std::size_t total_recv = bytes_per_peer * nranks;

    DeviceBuffer send_buf(total_send);
    DeviceBuffer recv_buf(total_recv);

    std::vector<char> h_send(total_send);
    for (int peer = 0; peer < nranks; peer++) {
      for (std::size_t i = 0; i < bytes_per_peer; i++) {
        h_send[peer * bytes_per_peer + i] =
            static_cast<char>((globalRank * 10 + peer + i) % 256);
      }
    }
    CUDACHECK_TEST(cudaMemcpy(
        send_buf.get(), h_send.data(), total_send, cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemset(recv_buf.get(), 0, total_recv));

    std::vector<char*> h_send_ptrs(nranks);
    std::vector<char*> h_recv_ptrs(nranks);
    std::vector<std::size_t> h_counts(nranks, bytes_per_peer);

    for (int r = 0; r < nranks; r++) {
      h_send_ptrs[r] = static_cast<char*>(send_buf.get()) + r * bytes_per_peer;
      h_recv_ptrs[r] = static_cast<char*>(recv_buf.get()) + r * bytes_per_peer;
    }

    DeviceBuffer d_send_ptrs(nranks * sizeof(char*));
    DeviceBuffer d_recv_ptrs(nranks * sizeof(char*));
    DeviceBuffer d_counts(nranks * sizeof(std::size_t));

    CUDACHECK_TEST(cudaMemcpy(
        d_send_ptrs.get(),
        h_send_ptrs.data(),
        nranks * sizeof(char*),
        cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemcpy(
        d_recv_ptrs.get(),
        h_recv_ptrs.data(),
        nranks * sizeof(char*),
        cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemcpy(
        d_counts.get(),
        h_counts.data(),
        nranks * sizeof(std::size_t),
        cudaMemcpyHostToDevice));

    // IB-only: num_blocks_nvl=0
    int total_blocks = 2 * num_blocks_ib;

    AllToAllvTileArgs args{
        .handle = handle,
        .send_ptrs = static_cast<char**>(d_send_ptrs.get()),
        .send_counts = static_cast<std::size_t*>(d_counts.get()),
        .recv_ptrs = static_cast<char**>(d_recv_ptrs.get()),
        .recv_counts = static_cast<std::size_t*>(d_counts.get()),
        .num_blocks_nvl = 0,
        .num_blocks_ib = num_blocks_ib,
        .max_signal_bytes = 0,
    };

    int device = 0;
    CUDACHECK_TEST(cudaGetDevice(&device));
    Timeout timeout = makeTimeout(30000, device);

    MPI_Barrier(MPI_COMM_WORLD);

    dim3 clus(comms::common::kDefaultClusterSize, 1, 1);
    void* kernel_args[] = {&args, &timeout};
    comms::common::launchKernel(
        (void*)alltoallv_tile_1d_kernel,
        dim3(total_blocks),
        dim3(512),
        kernel_args,
        nullptr,
        clus);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<char> h_recv(total_recv);
    CUDACHECK_TEST(cudaMemcpy(
        h_recv.data(), recv_buf.get(), total_recv, cudaMemcpyDeviceToHost));

    for (int peer = 0; peer < nranks; peer++) {
      for (std::size_t i = 0; i < bytes_per_peer; i++) {
        char expected = static_cast<char>((peer * 10 + globalRank + i) % 256);
        char actual = h_recv[peer * bytes_per_peer + i];
        EXPECT_EQ(actual, expected)
            << "Rank " << globalRank << ": mismatch receiving from peer "
            << peer << " at byte " << i << " (expected "
            << static_cast<int>(expected) << ", got "
            << static_cast<int>(actual) << ")";
        if (actual != expected) {
          return;
        }
      }
    }
  }
};

// ============================================================================
// NVL tests
// ============================================================================

TEST_F(AllToAllvTileTestFixture, Small) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks";
  }
  run_alltoallv_tile_test(4096, 4);
}

TEST_F(AllToAllvTileTestFixture, Medium) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks";
  }
  run_alltoallv_tile_test(256 * 1024, 8);
}

TEST_F(AllToAllvTileTestFixture, Large) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks";
  }
  run_alltoallv_tile_test(4 * 1024 * 1024, 16);
}

TEST_F(AllToAllvTileTestFixture, WithSignalBytes) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks";
  }
  run_alltoallv_tile_test(1024 * 1024, 8, 64 * 1024);
}

// ============================================================================
// NVL: blocks > peers (16 blocks, 7 peers → 2+ blocks per peer)
// ============================================================================

TEST_F(AllToAllvTileTestFixture, NvlBlocksGreaterThanPeers) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks";
  }
  run_alltoallv_tile_test(256 * 1024, 16);
}

// ============================================================================
// NVL: blocks < peers (2 blocks, 7 peers → each block handles ~4 peers)
// ============================================================================

TEST_F(AllToAllvTileTestFixture, NvlBlocksLessThanPeers) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks";
  }
  run_alltoallv_tile_test(256 * 1024, 2);
}

// ============================================================================
// IB tests (p2pDisable forces all peers to IBGDA)
// ============================================================================

// IB: blocks > peers (14 blocks, 7 IB peers → 2 blocks per peer)
TEST_F(AllToAllvTileTestFixture, IbBlocksGreaterThanPeers) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks";
  }
  run_alltoallv_tile_ib_test(256 * 1024, 14);
}

// IB: blocks == peers (7 blocks, 7 IB peers → 1 block per peer)
TEST_F(AllToAllvTileTestFixture, IbBlocksEqualPeers) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks";
  }
  run_alltoallv_tile_ib_test(256 * 1024, numRanks - 1);
}

// IB: blocks < peers (2 blocks, 7 IB peers → each block handles ~4 peers)
TEST_F(AllToAllvTileTestFixture, IbBlocksLessThanPeers) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks";
  }
  run_alltoallv_tile_ib_test(256 * 1024, 2);
}

} // namespace comms::pipes::tests

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
