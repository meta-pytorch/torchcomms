// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include <array>
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
        .perChannelSize = slot_size / static_cast<std::size_t>(max_groups),
        .max_num_channels = max_groups,
        .pipelineDepth = pipeline_depth,
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

// ===========================================================================
// Resumable-forward parity: same chains as above, but intermediates drive
// init_forward_progress / progress_forward_once to completion. These exercise
// the new resumable forward against blocking send/recv endpoints on the real
// wire protocol. Multi-block / multi-section geometries run the forward across
// multiple chunks, validating the fwd cursor advancing (in lockstep) between
// puts.
// ===========================================================================

TEST_F(RecvForwardChainTest, ForwardWithCopyProgress) {
  // >= 3: a 2-rank chain has no intermediate rank, so the forward under test
  // never runs and the case would report a vacuous pass.
  if (worldSize < 3) {
    GTEST_SKIP()
        << "requires >= 3 ranks for an intermediate forwarding rank, got "
        << worldSize;
  }

  constexpr std::size_t kDataBytes = 1 * 1024 * 1024;
  constexpr int kNumBlocks = 4;

  try {
    auto transport = create_transport(kDataBytes, kNumBlocks);

    DeviceBuffer sendBuf(kDataBytes);
    DeviceBuffer recvBuf(kDataBytes);

    const uint8_t fillPattern = 0xAC;
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

    ASSERT_EQ(
        test::launch_recv_forward_chain_progress(
            static_cast<P2pIbgdaTransportDevice**>(d_transports.get()),
            static_cast<const char*>(sendBuf.get()),
            static_cast<char*>(recvBuf.get()),
            kDataBytes,
            globalRank,
            worldSize,
            kNumBlocks,
            stream_),
        cudaSuccess)
        << "kernel launch failed";

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
                           << " byte mismatches (resumable forward)";
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }
}

TEST_F(RecvForwardChainTest, ForwardOnlyProgress) {
  if (worldSize < 3) {
    GTEST_SKIP() << "requires >= 3 ranks for forward-only test, got "
                 << worldSize;
  }

  constexpr std::size_t kDataBytes = 1 * 1024 * 1024;
  constexpr int kNumBlocks = 4;

  try {
    auto transport = create_transport(kDataBytes, kNumBlocks);

    DeviceBuffer sendBuf(kDataBytes);
    DeviceBuffer recvBuf(kDataBytes);

    const uint8_t fillPattern = 0xEF;
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

    ASSERT_EQ(
        test::launch_recv_forward_chain_progress_no_dst(
            static_cast<P2pIbgdaTransportDevice**>(d_transports.get()),
            static_cast<const char*>(sendBuf.get()),
            static_cast<char*>(recvBuf.get()),
            kDataBytes,
            globalRank,
            worldSize,
            kNumBlocks,
            stream_),
        cudaSuccess)
        << "kernel launch failed";

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

// Resumable forward across more data than one slot: per-block bytes span
// several pipeline windows, so each forward runs many chunks and the fwd cursor
// is advanced/validated between puts.
TEST_F(RecvForwardChainTest, MultiSectionProgress) {
  // >= 3: a 2-rank chain has no intermediate rank, so the forward under test
  // never runs and the case would report a vacuous pass.
  if (worldSize < 3) {
    GTEST_SKIP() << "requires >= 3 ranks for an intermediate forwarding rank, "
                    "got "
                 << worldSize;
  }

  constexpr std::size_t kSlotSize = 512 * 1024;
  constexpr std::size_t kDataBytes = 4 * kSlotSize; // 4 sections
  constexpr int kNumBlocks = 2;

  try {
    auto transport = create_transport(kSlotSize, kNumBlocks);

    DeviceBuffer sendBuf(kDataBytes);
    DeviceBuffer recvBuf(kDataBytes);

    const uint8_t fillPattern = 0x9C;
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

    ASSERT_EQ(
        test::launch_recv_forward_chain_progress(
            static_cast<P2pIbgdaTransportDevice**>(d_transports.get()),
            static_cast<const char*>(sendBuf.get()),
            static_cast<char*>(recvBuf.get()),
            kDataBytes,
            globalRank,
            worldSize,
            kNumBlocks,
            stream_),
        cudaSuccess)
        << "kernel launch failed";

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
      EXPECT_EQ(errors, 0)
          << "Rank " << globalRank << ": " << errors
          << " byte mismatches (resumable multi-section forward)";
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }
}

// ===========================================================================
// Guards resumable-forward slot completion when a forward reuses a send slot an
// ordinary resumable send just released: rank 1 does a resumable send (op A)
// then a reduce-forward (op B) to the SAME next transport, reusing the send
// slot across a pipeline-window boundary. This is the send->forward same-slot
// pattern the step-halved ring uses. Requires >= 3 ranks.
// ===========================================================================
TEST_F(RecvForwardChainTest, SendThenReduceForward) {
  if (worldSize < 3) {
    GTEST_SKIP() << "send-then-reduce-forward requires >= 3 ranks, got "
                 << worldSize;
  }

  constexpr std::size_t kSlotSize = 1 * 1024 * 1024; // one pipeline window
  constexpr std::size_t kDataBytes = kSlotSize; // one window per op
  constexpr int kNumBlocks = 1;
  const std::size_t nElems = kDataBytes / sizeof(float);

  try {
    auto transport = create_transport(kSlotSize, kNumBlocks);

    DeviceBuffer sendBuf(kDataBytes); // rank0: op B data; rank1: op A data
    DeviceBuffer localBuf(kDataBytes); // rank1: reduce input for op B
    DeviceBuffer recvA(kDataBytes); // rank2: op A result
    DeviceBuffer recvB(kDataBytes); // rank2: op B result

    // rank0 sends 10; rank1 op A sends 5, op B reduces rank0(10)+local(20)=30.
    const float rank0Value = 10.0f;
    const float rank1SendValue = 5.0f;
    const float rank1LocalValue = 20.0f;
    const float mySend =
        (globalRank == 0) ? rank0Value : rank1SendValue; // used by rank0/rank1
    std::vector<float> hostSend(nElems, mySend);
    std::vector<float> hostLocal(nElems, rank1LocalValue);
    CUDACHECK_TEST(cudaMemcpy(
        sendBuf.get(), hostSend.data(), kDataBytes, cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemcpy(
        localBuf.get(), hostLocal.data(), kDataBytes, cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemset(recvA.get(), 0, kDataBytes));
    CUDACHECK_TEST(cudaMemset(recvB.get(), 0, kDataBytes));
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

    ASSERT_EQ(
        test::launch_send_then_reduce_forward(
            static_cast<P2pIbgdaTransportDevice**>(d_transports.get()),
            static_cast<const char*>(sendBuf.get()),
            static_cast<const char*>(localBuf.get()),
            static_cast<char*>(recvA.get()),
            static_cast<char*>(recvB.get()),
            kDataBytes,
            globalRank,
            worldSize,
            kNumBlocks,
            stream_),
        cudaSuccess)
        << "kernel launch failed";

    cudaError_t err = cudaStreamSynchronize(stream_);
    ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);

    bootstrap->barrierAll();

    // rank 2 holds op A (rank1's send=5) and op B (rank0 10 + rank1 local 20).
    if (globalRank == 2) {
      auto check = [&](DeviceBuffer& buf, float expect, const char* which) {
        std::vector<float> hostBuf(nElems);
        CUDACHECK_TEST(cudaMemcpy(
            hostBuf.data(), buf.get(), kDataBytes, cudaMemcpyDeviceToHost));
        int errors = 0;
        for (std::size_t i = 0; i < nElems; ++i) {
          if (std::abs(hostBuf[i] - expect) > 1e-3f) {
            ++errors;
          }
        }
        EXPECT_EQ(errors, 0)
            << which << ": " << errors << " element mismatches (expected "
            << expect << ")";
      };
      check(recvA, rank1SendValue, "op A (send)");
      check(recvB, rank0Value + rank1LocalValue, "op B (reduce-forward)");
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
