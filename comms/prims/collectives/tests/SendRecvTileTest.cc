// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include "comms/common/CudaWrap.h"
#include "comms/prims/collectives/SendRecvTile.cuh"
#include "comms/prims/core/TimeoutUtils.h"
#include "comms/prims/transport/MultiPeerTransport.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

using namespace meta::comms;

namespace comms::prims::tests {

class SendRecvTileTestFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }

  // Ranks are paired (0<->1, 2<->3, ...). Even ranks send to their odd
  // partner; odd ranks receive. Returns the partner rank.
  int partner() const {
    return globalRank ^ 1;
  }
  bool is_sender() const {
    return (globalRank % 2) == 0;
  }

  std::unique_ptr<MultiPeerTransport> create_and_exchange(
      bool ib_only,
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
                .dataBufferSize = 8 * 1024 * 1024,
                .sendRecv =
                    MultipeerIbgdaTransportConfig::SendRecvConfig{
                        .maxGroups = 32,
                        .pipelineDepth = 2,
                    },
                .numQpsPerPeerPerNic = 2,
            },
        .topoConfig =
            {
                .p2pDisable = ib_only,
            },
    };
    auto bootstrap = std::make_shared<MpiBootstrap>();
    auto transport = std::make_unique<MultiPeerTransport>(
        globalRank, numRanks, localRank, bootstrap, config);
    transport->exchange();
    return transport;
  }

  void run_sendrecv_tile_test(
      std::size_t bytes,
      int num_blocks,
      bool ib_only = false,
      std::size_t max_signal_bytes = 0) {
    auto transport = create_and_exchange(ib_only);
    auto handle = transport->get_device_handle();

    const int peer = partner();
    const bool send = is_sender();

    DeviceBuffer buf(bytes);

    // Sender fills with a deterministic pattern keyed on its own rank;
    // receiver checks the bytes match the sender's (peer's) pattern.
    std::vector<char> h_buf(bytes);
    if (send) {
      for (std::size_t i = 0; i < bytes; i++) {
        h_buf[i] = static_cast<char>((globalRank * 7 + i) % 256);
      }
      CUDACHECK_TEST(
          cudaMemcpy(buf.get(), h_buf.data(), bytes, cudaMemcpyHostToDevice));
    } else {
      CUDACHECK_TEST(cudaMemset(buf.get(), 0, bytes));
    }

    SendRecvTileArgs args{
        .handle = handle,
        .is_send = send,
        .is_recv = !send,
        .send_peer = peer,
        .recv_peer = peer,
        .send_data = send ? static_cast<char*>(buf.get()) : nullptr,
        .send_count = send ? bytes : 0,
        .recv_data = send ? nullptr : static_cast<char*>(buf.get()),
        .recv_count = send ? 0 : bytes,
        .max_signal_bytes = max_signal_bytes,
    };

    int device = 0;
    CUDACHECK_TEST(cudaGetDevice(&device));
    Timeout timeout = makeTimeout(30000, device);

    MPI_Barrier(MPI_COMM_WORLD);

    dim3 clus(comms::common::kDefaultClusterSize, 1, 1);
    void* kernel_args[] = {&args, &timeout};
    comms::common::launchKernel(
        (void*)sendrecv_tile_kernel,
        dim3(num_blocks),
        dim3(512),
        kernel_args,
        nullptr,
        clus);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);

    if (!send) {
      std::vector<char> h_recv(bytes);
      CUDACHECK_TEST(
          cudaMemcpy(h_recv.data(), buf.get(), bytes, cudaMemcpyDeviceToHost));
      for (std::size_t i = 0; i < bytes; i++) {
        char expected = static_cast<char>((peer * 7 + i) % 256);
        char actual = h_recv[i];
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

  // Bidirectional: every rank simultaneously sends to and receives from
  // its partner. The grid is split in half inside the kernel (send half +
  // recv half). `num_blocks` must be even.
  void run_sendrecv_tile_twoway_test(
      std::size_t bytes,
      int num_blocks,
      bool ib_only = false,
      std::size_t max_signal_bytes = 0) {
    auto transport = create_and_exchange(ib_only);
    auto handle = transport->get_device_handle();

    const int peer = partner();

    DeviceBuffer send_buf(bytes);
    DeviceBuffer recv_buf(bytes);

    // Send buffer carries a pattern keyed on this rank; recv buffer is
    // zeroed and must end up matching the peer's send pattern.
    std::vector<char> h_send(bytes);
    for (std::size_t i = 0; i < bytes; i++) {
      h_send[i] = static_cast<char>((globalRank * 7 + i) % 256);
    }
    CUDACHECK_TEST(cudaMemcpy(
        send_buf.get(), h_send.data(), bytes, cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemset(recv_buf.get(), 0, bytes));

    SendRecvTileArgs args{
        .handle = handle,
        .is_send = true,
        .is_recv = true,
        .send_peer = peer,
        .recv_peer = peer,
        .send_data = static_cast<char*>(send_buf.get()),
        .send_count = bytes,
        .recv_data = static_cast<char*>(recv_buf.get()),
        .recv_count = bytes,
        .max_signal_bytes = max_signal_bytes,
    };

    int device = 0;
    CUDACHECK_TEST(cudaGetDevice(&device));
    Timeout timeout = makeTimeout(30000, device);

    MPI_Barrier(MPI_COMM_WORLD);

    dim3 clus(comms::common::kDefaultClusterSize, 1, 1);
    void* kernel_args[] = {&args, &timeout};
    comms::common::launchKernel(
        (void*)sendrecv_tile_kernel,
        dim3(num_blocks),
        dim3(512),
        kernel_args,
        nullptr,
        clus);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<char> h_recv(bytes);
    CUDACHECK_TEST(cudaMemcpy(
        h_recv.data(), recv_buf.get(), bytes, cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < bytes; i++) {
      char expected = static_cast<char>((peer * 7 + i) % 256);
      char actual = h_recv[i];
      EXPECT_EQ(actual, expected)
          << "Rank " << globalRank << ": two-way mismatch from peer " << peer
          << " at byte " << i << " (expected " << static_cast<int>(expected)
          << ", got " << static_cast<int>(actual) << ")";
      if (actual != expected) {
        return;
      }
    }
  }
};

// ============================================================================
// NVL tests
// ============================================================================

TEST_F(SendRecvTileTestFixture, NvlSmall) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks";
  }
  run_sendrecv_tile_test(4096, 4);
}

TEST_F(SendRecvTileTestFixture, NvlMedium) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks";
  }
  run_sendrecv_tile_test(256 * 1024, 8);
}

TEST_F(SendRecvTileTestFixture, NvlLarge) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks";
  }
  run_sendrecv_tile_test(4 * 1024 * 1024, 16);
}

TEST_F(SendRecvTileTestFixture, NvlWithSignalBytes) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks";
  }
  run_sendrecv_tile_test(1024 * 1024, 8, /*ib_only=*/false, 64 * 1024);
}

// ============================================================================
// IB tests (p2pDisable forces the peer onto IBGDA)
// ============================================================================

TEST_F(SendRecvTileTestFixture, IbSmall) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks";
  }
  run_sendrecv_tile_test(4096, 4, /*ib_only=*/true);
}

TEST_F(SendRecvTileTestFixture, IbMedium) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks";
  }
  run_sendrecv_tile_test(256 * 1024, 8, /*ib_only=*/true);
}

TEST_F(SendRecvTileTestFixture, IbLarge) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks";
  }
  run_sendrecv_tile_test(4 * 1024 * 1024, 14, /*ib_only=*/true);
}

// ============================================================================
// Two-way (bidirectional) tests: each rank sends to and receives from its
// partner at the same time; the grid is split half-send / half-recv.
// ============================================================================

TEST_F(SendRecvTileTestFixture, NvlTwoWaySmall) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks";
  }
  run_sendrecv_tile_twoway_test(4096, 8);
}

TEST_F(SendRecvTileTestFixture, NvlTwoWayLarge) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks";
  }
  run_sendrecv_tile_twoway_test(4 * 1024 * 1024, 16);
}

TEST_F(SendRecvTileTestFixture, IbTwoWaySmall) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks";
  }
  run_sendrecv_tile_twoway_test(4096, 8, /*ib_only=*/true);
}

TEST_F(SendRecvTileTestFixture, IbTwoWayLarge) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks";
  }
  run_sendrecv_tile_twoway_test(4 * 1024 * 1024, 16, /*ib_only=*/true);
}

} // namespace comms::prims::tests

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
