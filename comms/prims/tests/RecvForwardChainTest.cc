// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "comms/prims/tests/RecvForwardChainTest.h"
#include "comms/prims/transport/ibgda/MultipeerIbgdaTransport.h"
#include "comms/testinfra/BenchmarkTestFixture.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::BenchmarkEnvironment;
using meta::comms::BenchmarkTestFixture;
using meta::comms::DeviceBuffer;

namespace comms::prims::tests {

using SendRecvConfig = MultipeerIbgdaTransportConfig::SendRecvConfig;

class RecvForwardChainTest : public BenchmarkTestFixture {
 protected:
  void SetUp() override {
    BenchmarkTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
    CUDACHECK_TEST(cudaStreamCreate(&stream_));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaStreamDestroy(stream_));
    BenchmarkTestFixture::TearDown();
  }

  std::unique_ptr<MultipeerIbgdaTransport> create_transport(
      std::size_t slot_size,
      int max_groups,
      int pipeline_depth = 2) {
    MultipeerIbgdaTransportConfig config{
        .cudaDevice = localRank,
        .dataBufferSize = slot_size,
        .sendRecv =
            SendRecvConfig{
                .maxGroups = max_groups,
                .pipelineDepth = pipeline_depth,
            },
    };
    auto transport = std::make_unique<MultipeerIbgdaTransport>(
        globalRank, worldSize, bootstrap, config);
    transport->exchange();
    return transport;
  }

  cudaStream_t stream_{};
};

// Chain test: rank 0 → rank 1 → ... → rank N-1
// Each intermediate uses recv_forward with dst (copies to local buffer).
// Verify all ranks that receive data see the correct pattern.
TEST_F(RecvForwardChainTest, ForwardWithCopy) {
  if (worldSize < 2) {
    XLOGF(INFO, "Skipping: requires >= 2 ranks, got {}", worldSize);
    return;
  }

  constexpr std::size_t kDataBytes = 1 * 1024 * 1024;
  constexpr int kNumBlocks = 4;

  try {
    auto transport = create_transport(kDataBytes, kNumBlocks);

    DeviceBuffer sendBuf(kDataBytes);
    DeviceBuffer recvBuf(kDataBytes);

    const uint8_t fillPattern = 0xCA;
    CUDACHECK_TEST(cudaMemset(sendBuf.get(), fillPattern, kDataBytes));
    CUDACHECK_TEST(cudaMemset(recvBuf.get(), 0, kDataBytes));
    CUDACHECK_TEST(cudaDeviceSynchronize());

    // Build per-rank transport pointer array
    std::vector<P2pIbgdaTransportDevice*> peer_transports(worldSize, nullptr);
    for (int r = 0; r < worldSize; ++r) {
      if (r != globalRank) {
        peer_transports[r] = transport->getP2pTransportDevice(r);
      }
    }
    DeviceBuffer d_transports(worldSize * sizeof(P2pIbgdaTransportDevice*));
    CUDACHECK_TEST(cudaMemcpy(
        d_transports.get(),
        peer_transports.data(),
        worldSize * sizeof(P2pIbgdaTransportDevice*),
        cudaMemcpyHostToDevice));

    bootstrap->barrierAll();

    test::launch_recv_forward_chain(
        static_cast<P2pIbgdaTransportDevice**>(d_transports.get()),
        static_cast<const char*>(sendBuf.get()),
        static_cast<char*>(recvBuf.get()),
        kDataBytes,
        globalRank,
        worldSize,
        kNumBlocks,
        stream_);

    cudaError_t err = cudaStreamSynchronize(stream_);
    ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);

    bootstrap->barrierAll();

    // All ranks except rank 0 should have received the data
    if (globalRank != 0) {
      std::vector<uint8_t> hostBuf(kDataBytes);
      CUDACHECK_TEST(cudaMemcpy(
          hostBuf.data(), recvBuf.get(), kDataBytes, cudaMemcpyDeviceToHost));

      int errors = 0;
      for (std::size_t i = 0; i < kDataBytes; ++i) {
        if (hostBuf[i] != fillPattern) {
          if (errors < 10) {
            XLOGF(
                ERR,
                "Rank {}: byte {} expected 0x{:02X} got 0x{:02X}",
                globalRank,
                i,
                fillPattern,
                hostBuf[i]);
          }
          ++errors;
        }
      }
      EXPECT_EQ(errors, 0) << "Rank " << globalRank << ": " << errors
                           << " byte mismatches";
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }
}

// Chain test with dst=nullptr for intermediates (forward-only).
// Only the last rank should have the data.
TEST_F(RecvForwardChainTest, ForwardOnly) {
  if (worldSize < 3) {
    XLOGF(INFO, "Skipping: requires >= 3 ranks for forward-only test");
    return;
  }

  constexpr std::size_t kDataBytes = 1 * 1024 * 1024;
  constexpr int kNumBlocks = 4;

  try {
    auto transport = create_transport(kDataBytes, kNumBlocks);

    DeviceBuffer sendBuf(kDataBytes);
    DeviceBuffer recvBuf(kDataBytes);

    const uint8_t fillPattern = 0xBE;
    CUDACHECK_TEST(cudaMemset(sendBuf.get(), fillPattern, kDataBytes));
    CUDACHECK_TEST(cudaMemset(recvBuf.get(), 0, kDataBytes));
    CUDACHECK_TEST(cudaDeviceSynchronize());

    std::vector<P2pIbgdaTransportDevice*> peer_transports(worldSize, nullptr);
    for (int r = 0; r < worldSize; ++r) {
      if (r != globalRank) {
        peer_transports[r] = transport->getP2pTransportDevice(r);
      }
    }
    DeviceBuffer d_transports(worldSize * sizeof(P2pIbgdaTransportDevice*));
    CUDACHECK_TEST(cudaMemcpy(
        d_transports.get(),
        peer_transports.data(),
        worldSize * sizeof(P2pIbgdaTransportDevice*),
        cudaMemcpyHostToDevice));

    bootstrap->barrierAll();

    test::launch_recv_forward_chain_no_dst(
        static_cast<P2pIbgdaTransportDevice**>(d_transports.get()),
        static_cast<const char*>(sendBuf.get()),
        static_cast<char*>(recvBuf.get()),
        kDataBytes,
        globalRank,
        worldSize,
        kNumBlocks,
        stream_);

    cudaError_t err = cudaStreamSynchronize(stream_);
    ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);

    bootstrap->barrierAll();

    // Only the last rank gets data
    if (globalRank == worldSize - 1) {
      std::vector<uint8_t> hostBuf(kDataBytes);
      CUDACHECK_TEST(cudaMemcpy(
          hostBuf.data(), recvBuf.get(), kDataBytes, cudaMemcpyDeviceToHost));

      int errors = 0;
      for (std::size_t i = 0; i < kDataBytes; ++i) {
        if (hostBuf[i] != fillPattern) {
          ++errors;
        }
      }
      EXPECT_EQ(errors, 0) << "Last rank: " << errors << " byte mismatches";
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }
}

// 2-rank test: send → recv (no intermediates). Validates that the protocol
// is compatible when recv_forward is not involved — just send + recv.
TEST_F(RecvForwardChainTest, SendRecvDirect) {
  if (worldSize != 2) {
    XLOGF(INFO, "Skipping: requires exactly 2 ranks, got {}", worldSize);
    return;
  }

  constexpr std::size_t kDataBytes = 2 * 1024 * 1024;
  constexpr int kNumBlocks = 4;

  try {
    auto transport = create_transport(kDataBytes, kNumBlocks);

    DeviceBuffer sendBuf(kDataBytes);
    DeviceBuffer recvBuf(kDataBytes);

    const uint8_t fillPattern = 0x55;
    CUDACHECK_TEST(cudaMemset(sendBuf.get(), fillPattern, kDataBytes));
    CUDACHECK_TEST(cudaMemset(recvBuf.get(), 0, kDataBytes));
    CUDACHECK_TEST(cudaDeviceSynchronize());

    std::vector<P2pIbgdaTransportDevice*> peer_transports(worldSize, nullptr);
    for (int r = 0; r < worldSize; ++r) {
      if (r != globalRank) {
        peer_transports[r] = transport->getP2pTransportDevice(r);
      }
    }
    DeviceBuffer d_transports(worldSize * sizeof(P2pIbgdaTransportDevice*));
    CUDACHECK_TEST(cudaMemcpy(
        d_transports.get(),
        peer_transports.data(),
        worldSize * sizeof(P2pIbgdaTransportDevice*),
        cudaMemcpyHostToDevice));

    bootstrap->barrierAll();

    // Chain with worldSize=2: rank 0 sends, rank 1 receives — no recv_forward
    test::launch_recv_forward_chain(
        static_cast<P2pIbgdaTransportDevice**>(d_transports.get()),
        static_cast<const char*>(sendBuf.get()),
        static_cast<char*>(recvBuf.get()),
        kDataBytes,
        globalRank,
        worldSize,
        kNumBlocks,
        stream_);

    cudaError_t err = cudaStreamSynchronize(stream_);
    ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);

    bootstrap->barrierAll();

    if (globalRank == 1) {
      std::vector<uint8_t> hostBuf(kDataBytes);
      CUDACHECK_TEST(cudaMemcpy(
          hostBuf.data(), recvBuf.get(), kDataBytes, cudaMemcpyDeviceToHost));

      int errors = 0;
      for (std::size_t i = 0; i < kDataBytes; ++i) {
        if (hostBuf[i] != fillPattern) {
          ++errors;
        }
      }
      EXPECT_EQ(errors, 0) << "Rank 1: " << errors << " byte mismatches";
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }
}

// Multi-section test: transfer more data than one slot to exercise pipelining.
TEST_F(RecvForwardChainTest, MultiSection) {
  if (worldSize < 2) {
    XLOGF(INFO, "Skipping: requires >= 2 ranks, got {}", worldSize);
    return;
  }

  constexpr std::size_t kSlotSize = 512 * 1024;
  constexpr std::size_t kDataBytes = 4 * kSlotSize; // 4 sections
  constexpr int kNumBlocks = 2;

  try {
    auto transport = create_transport(kSlotSize, kNumBlocks);

    DeviceBuffer sendBuf(kDataBytes);
    DeviceBuffer recvBuf(kDataBytes);

    const uint8_t fillPattern = 0xDD;
    CUDACHECK_TEST(cudaMemset(sendBuf.get(), fillPattern, kDataBytes));
    CUDACHECK_TEST(cudaMemset(recvBuf.get(), 0, kDataBytes));
    CUDACHECK_TEST(cudaDeviceSynchronize());

    std::vector<P2pIbgdaTransportDevice*> peer_transports(worldSize, nullptr);
    for (int r = 0; r < worldSize; ++r) {
      if (r != globalRank) {
        peer_transports[r] = transport->getP2pTransportDevice(r);
      }
    }
    DeviceBuffer d_transports(worldSize * sizeof(P2pIbgdaTransportDevice*));
    CUDACHECK_TEST(cudaMemcpy(
        d_transports.get(),
        peer_transports.data(),
        worldSize * sizeof(P2pIbgdaTransportDevice*),
        cudaMemcpyHostToDevice));

    bootstrap->barrierAll();

    test::launch_recv_forward_chain(
        static_cast<P2pIbgdaTransportDevice**>(d_transports.get()),
        static_cast<const char*>(sendBuf.get()),
        static_cast<char*>(recvBuf.get()),
        kDataBytes,
        globalRank,
        worldSize,
        kNumBlocks,
        stream_);

    cudaError_t err = cudaStreamSynchronize(stream_);
    ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);

    bootstrap->barrierAll();

    if (globalRank != 0) {
      std::vector<uint8_t> hostBuf(kDataBytes);
      CUDACHECK_TEST(cudaMemcpy(
          hostBuf.data(), recvBuf.get(), kDataBytes, cudaMemcpyDeviceToHost));

      int errors = 0;
      for (std::size_t i = 0; i < kDataBytes; ++i) {
        if (hostBuf[i] != fillPattern) {
          ++errors;
        }
      }
      EXPECT_EQ(errors, 0) << "Rank " << globalRank << ": " << errors
                           << " byte mismatches in multi-section transfer";
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }
}

// =============================================================================
// Resumable-forward parity tests: identical setup/checks to the blocking
// chain tests above, but intermediates drive init_forward_progress /
// progress_forward_once. A green result is byte-parity with blocking forward.
// =============================================================================

// Parity with ForwardWithCopy (dst != nullptr; every rank but rank 0 receives).
TEST_F(RecvForwardChainTest, ForwardWithCopyProgress) {
  if (worldSize < 2) {
    XLOGF(INFO, "Skipping: requires >= 2 ranks, got {}", worldSize);
    return;
  }

  constexpr std::size_t kDataBytes = 1 * 1024 * 1024;
  constexpr int kNumBlocks = 4;

  try {
    auto transport = create_transport(kDataBytes, kNumBlocks);

    DeviceBuffer sendBuf(kDataBytes);
    DeviceBuffer recvBuf(kDataBytes);

    const uint8_t fillPattern = 0xCA;
    CUDACHECK_TEST(cudaMemset(sendBuf.get(), fillPattern, kDataBytes));
    CUDACHECK_TEST(cudaMemset(recvBuf.get(), 0, kDataBytes));
    CUDACHECK_TEST(cudaDeviceSynchronize());

    std::vector<P2pIbgdaTransportDevice*> peer_transports(worldSize, nullptr);
    for (int r = 0; r < worldSize; ++r) {
      if (r != globalRank) {
        peer_transports[r] = transport->getP2pTransportDevice(r);
      }
    }
    DeviceBuffer d_transports(worldSize * sizeof(P2pIbgdaTransportDevice*));
    CUDACHECK_TEST(cudaMemcpy(
        d_transports.get(),
        peer_transports.data(),
        worldSize * sizeof(P2pIbgdaTransportDevice*),
        cudaMemcpyHostToDevice));

    bootstrap->barrierAll();

    test::launch_recv_forward_chain_progress(
        static_cast<P2pIbgdaTransportDevice**>(d_transports.get()),
        static_cast<const char*>(sendBuf.get()),
        static_cast<char*>(recvBuf.get()),
        kDataBytes,
        globalRank,
        worldSize,
        kNumBlocks,
        stream_);

    cudaError_t err = cudaStreamSynchronize(stream_);
    ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);

    bootstrap->barrierAll();

    if (globalRank != 0) {
      std::vector<uint8_t> hostBuf(kDataBytes);
      CUDACHECK_TEST(cudaMemcpy(
          hostBuf.data(), recvBuf.get(), kDataBytes, cudaMemcpyDeviceToHost));

      int errors = 0;
      for (std::size_t i = 0; i < kDataBytes; ++i) {
        if (hostBuf[i] != fillPattern) {
          ++errors;
        }
      }
      EXPECT_EQ(errors, 0) << "Rank " << globalRank << ": " << errors
                           << " byte mismatches (resumable forward)";
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }
}

// Parity with ForwardOnly (dst == nullptr; only the last rank receives).
TEST_F(RecvForwardChainTest, ForwardOnlyProgress) {
  if (worldSize < 3) {
    XLOGF(INFO, "Skipping: requires >= 3 ranks for forward-only test");
    return;
  }

  constexpr std::size_t kDataBytes = 1 * 1024 * 1024;
  constexpr int kNumBlocks = 4;

  try {
    auto transport = create_transport(kDataBytes, kNumBlocks);

    DeviceBuffer sendBuf(kDataBytes);
    DeviceBuffer recvBuf(kDataBytes);

    const uint8_t fillPattern = 0xBE;
    CUDACHECK_TEST(cudaMemset(sendBuf.get(), fillPattern, kDataBytes));
    CUDACHECK_TEST(cudaMemset(recvBuf.get(), 0, kDataBytes));
    CUDACHECK_TEST(cudaDeviceSynchronize());

    std::vector<P2pIbgdaTransportDevice*> peer_transports(worldSize, nullptr);
    for (int r = 0; r < worldSize; ++r) {
      if (r != globalRank) {
        peer_transports[r] = transport->getP2pTransportDevice(r);
      }
    }
    DeviceBuffer d_transports(worldSize * sizeof(P2pIbgdaTransportDevice*));
    CUDACHECK_TEST(cudaMemcpy(
        d_transports.get(),
        peer_transports.data(),
        worldSize * sizeof(P2pIbgdaTransportDevice*),
        cudaMemcpyHostToDevice));

    bootstrap->barrierAll();

    test::launch_recv_forward_chain_progress_no_dst(
        static_cast<P2pIbgdaTransportDevice**>(d_transports.get()),
        static_cast<const char*>(sendBuf.get()),
        static_cast<char*>(recvBuf.get()),
        kDataBytes,
        globalRank,
        worldSize,
        kNumBlocks,
        stream_);

    cudaError_t err = cudaStreamSynchronize(stream_);
    ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);

    bootstrap->barrierAll();

    if (globalRank == worldSize - 1) {
      std::vector<uint8_t> hostBuf(kDataBytes);
      CUDACHECK_TEST(cudaMemcpy(
          hostBuf.data(), recvBuf.get(), kDataBytes, cudaMemcpyDeviceToHost));

      int errors = 0;
      for (std::size_t i = 0; i < kDataBytes; ++i) {
        if (hostBuf[i] != fillPattern) {
          ++errors;
        }
      }
      EXPECT_EQ(errors, 0) << "Last rank: " << errors
                           << " byte mismatches (resumable forward-only)";
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }
}

// Parity with MultiSection: transfer spans several pipeline windows so the
// resumable forward exercises the NIC_DONE / SLOT_FREE backpressure waits and
// multi-chunk replay (the no-QP unit tests cannot reach these).
TEST_F(RecvForwardChainTest, MultiSectionProgress) {
  if (worldSize < 2) {
    XLOGF(INFO, "Skipping: requires >= 2 ranks, got {}", worldSize);
    return;
  }

  constexpr std::size_t kSlotSize = 512 * 1024;
  constexpr std::size_t kDataBytes = 4 * kSlotSize; // 4 sections
  constexpr int kNumBlocks = 2;

  try {
    auto transport = create_transport(kSlotSize, kNumBlocks);

    DeviceBuffer sendBuf(kDataBytes);
    DeviceBuffer recvBuf(kDataBytes);

    const uint8_t fillPattern = 0xDD;
    CUDACHECK_TEST(cudaMemset(sendBuf.get(), fillPattern, kDataBytes));
    CUDACHECK_TEST(cudaMemset(recvBuf.get(), 0, kDataBytes));
    CUDACHECK_TEST(cudaDeviceSynchronize());

    std::vector<P2pIbgdaTransportDevice*> peer_transports(worldSize, nullptr);
    for (int r = 0; r < worldSize; ++r) {
      if (r != globalRank) {
        peer_transports[r] = transport->getP2pTransportDevice(r);
      }
    }
    DeviceBuffer d_transports(worldSize * sizeof(P2pIbgdaTransportDevice*));
    CUDACHECK_TEST(cudaMemcpy(
        d_transports.get(),
        peer_transports.data(),
        worldSize * sizeof(P2pIbgdaTransportDevice*),
        cudaMemcpyHostToDevice));

    bootstrap->barrierAll();

    test::launch_recv_forward_chain_progress(
        static_cast<P2pIbgdaTransportDevice**>(d_transports.get()),
        static_cast<const char*>(sendBuf.get()),
        static_cast<char*>(recvBuf.get()),
        kDataBytes,
        globalRank,
        worldSize,
        kNumBlocks,
        stream_);

    cudaError_t err = cudaStreamSynchronize(stream_);
    ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);

    bootstrap->barrierAll();

    if (globalRank != 0) {
      std::vector<uint8_t> hostBuf(kDataBytes);
      CUDACHECK_TEST(cudaMemcpy(
          hostBuf.data(), recvBuf.get(), kDataBytes, cudaMemcpyDeviceToHost));

      int errors = 0;
      for (std::size_t i = 0; i < kDataBytes; ++i) {
        if (hostBuf[i] != fillPattern) {
          ++errors;
        }
      }
      EXPECT_EQ(errors, 0) << "Rank " << globalRank << ": " << errors
                           << " byte mismatches (resumable multi-section)";
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }
}

// Non-16B-aligned resumable forward: a single-group (num_blocks=1) chain with a
// transfer size that is not a multiple of 16 exercises the non-aligned tail of
// CopyOp::forward / the RDMA put through the resumable path. Byte-parity with
// the source pattern is the check.
TEST_F(RecvForwardChainTest, ForwardWithCopyProgressNonAligned) {
  if (worldSize < 3) {
    XLOGF(INFO, "Skipping: requires >= 3 ranks to exercise a forward");
    return;
  }

  constexpr std::size_t kSlotSize = 256 * 1024;
  constexpr std::size_t kDataBytes = 65537; // not a multiple of 16
  constexpr int kNumBlocks = 1;

  try {
    auto transport = create_transport(kSlotSize, kNumBlocks);

    DeviceBuffer sendBuf(kDataBytes);
    DeviceBuffer recvBuf(kDataBytes);

    const uint8_t fillPattern = 0xA7;
    CUDACHECK_TEST(cudaMemset(sendBuf.get(), fillPattern, kDataBytes));
    CUDACHECK_TEST(cudaMemset(recvBuf.get(), 0, kDataBytes));
    CUDACHECK_TEST(cudaDeviceSynchronize());

    std::vector<P2pIbgdaTransportDevice*> peer_transports(worldSize, nullptr);
    for (int r = 0; r < worldSize; ++r) {
      if (r != globalRank) {
        peer_transports[r] = transport->getP2pTransportDevice(r);
      }
    }
    DeviceBuffer d_transports(worldSize * sizeof(P2pIbgdaTransportDevice*));
    CUDACHECK_TEST(cudaMemcpy(
        d_transports.get(),
        peer_transports.data(),
        worldSize * sizeof(P2pIbgdaTransportDevice*),
        cudaMemcpyHostToDevice));

    bootstrap->barrierAll();

    test::launch_recv_forward_chain_progress(
        static_cast<P2pIbgdaTransportDevice**>(d_transports.get()),
        static_cast<const char*>(sendBuf.get()),
        static_cast<char*>(recvBuf.get()),
        kDataBytes,
        globalRank,
        worldSize,
        kNumBlocks,
        stream_);

    cudaError_t err = cudaStreamSynchronize(stream_);
    ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);

    bootstrap->barrierAll();

    if (globalRank != 0) {
      std::vector<uint8_t> hostBuf(kDataBytes);
      CUDACHECK_TEST(cudaMemcpy(
          hostBuf.data(), recvBuf.get(), kDataBytes, cudaMemcpyDeviceToHost));

      int errors = 0;
      for (std::size_t i = 0; i < kDataBytes; ++i) {
        if (hostBuf[i] != fillPattern) {
          ++errors;
        }
      }
      EXPECT_EQ(errors, 0) << "Rank " << globalRank << ": " << errors
                           << " byte mismatches (resumable non-16B forward)";
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }
}

// Interleaved multi-lane: a single block per rank multiplexes 2 forwards over
// distinct group_ids (the resumable API's purpose). Byte-parity confirms the
// two lanes do not corrupt each other's staging/cursors.
TEST_F(RecvForwardChainTest, ForwardInterleaved2LaneProgress) {
  if (worldSize < 3) {
    XLOGF(INFO, "Skipping: requires >= 3 ranks to exercise a forward");
    return;
  }

  constexpr std::size_t kSlotSize = 256 * 1024;
  constexpr std::size_t kDataBytes = 1 * 1024 * 1024; // 2 lanes x multi-window
  constexpr int kMaxGroups = 2; // == kLanes in the kernel

  try {
    auto transport = create_transport(kSlotSize, kMaxGroups);

    DeviceBuffer sendBuf(kDataBytes);
    DeviceBuffer recvBuf(kDataBytes);

    const uint8_t fillPattern = 0x3C;
    CUDACHECK_TEST(cudaMemset(sendBuf.get(), fillPattern, kDataBytes));
    CUDACHECK_TEST(cudaMemset(recvBuf.get(), 0, kDataBytes));
    CUDACHECK_TEST(cudaDeviceSynchronize());

    std::vector<P2pIbgdaTransportDevice*> peer_transports(worldSize, nullptr);
    for (int r = 0; r < worldSize; ++r) {
      if (r != globalRank) {
        peer_transports[r] = transport->getP2pTransportDevice(r);
      }
    }
    DeviceBuffer d_transports(worldSize * sizeof(P2pIbgdaTransportDevice*));
    CUDACHECK_TEST(cudaMemcpy(
        d_transports.get(),
        peer_transports.data(),
        worldSize * sizeof(P2pIbgdaTransportDevice*),
        cudaMemcpyHostToDevice));

    bootstrap->barrierAll();

    test::launch_recv_forward_chain_2lane_progress(
        static_cast<P2pIbgdaTransportDevice**>(d_transports.get()),
        static_cast<const char*>(sendBuf.get()),
        static_cast<char*>(recvBuf.get()),
        kDataBytes,
        globalRank,
        worldSize,
        stream_);

    cudaError_t err = cudaStreamSynchronize(stream_);
    ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);

    bootstrap->barrierAll();

    if (globalRank != 0) {
      std::vector<uint8_t> hostBuf(kDataBytes);
      CUDACHECK_TEST(cudaMemcpy(
          hostBuf.data(), recvBuf.get(), kDataBytes, cudaMemcpyDeviceToHost));

      int errors = 0;
      for (std::size_t i = 0; i < kDataBytes; ++i) {
        if (hostBuf[i] != fillPattern) {
          ++errors;
        }
      }
      EXPECT_EQ(errors, 0) << "Rank " << globalRank << ": " << errors
                           << " byte mismatches (interleaved 2-lane forward)";
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }
}

} // namespace comms::prims::tests

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  auto* env = new BenchmarkEnvironment();
  ::testing::AddGlobalTestEnvironment(env);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
