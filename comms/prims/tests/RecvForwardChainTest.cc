// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <folly/init/Init.h>
#include "comms/utils/logger/SpdlogLogger.h"

#include <cstdint>
#include <memory>
#include <string>
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

namespace {

struct ChainMode {
  test::ChainProto proto;
};

// Every wire format the chain supports. Cases sweep both on ONE transport,
// which also gives the cross-protocol coverage that matters most here: Simple
// and LL share a channel's QPs and its recvDataReadyLaneCursor, and an LL path
// that forgets to advance that mirror passes every all-LL and all-Simple run
// yet deadlocks the next Simple recv.
constexpr std::array<ChainMode, 2> kChainModes = {{
    {test::ChainProto::Simple},
    {test::ChainProto::LL},
}};

std::string modeName(const ChainMode& mode) {
  return std::string(mode.proto == test::ChainProto::LL ? "ll" : "simple") +
      "_blocking";
}

// Position-dependent so a stale-slot or stale-flag accept cannot pass. A
// uniform fill would let LL return the previous ring pass's bytes unnoticed --
// and it would return them EARLY, so it shows up as a better latency number
// rather than a failure.
uint8_t expectedByte(uint8_t base, std::size_t i) {
  return static_cast<uint8_t>(base + (i % 251));
}

std::vector<uint8_t> makePattern(uint8_t base, std::size_t nbytes) {
  std::vector<uint8_t> data(nbytes);
  for (std::size_t i = 0; i < nbytes; ++i) {
    data[i] = expectedByte(base, i);
  }
  return data;
}

} // namespace

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

  // Device-resident array of per-rank transport pointers, indexed by rank.
  DeviceBuffer makePeerTable(MultipeerIbgdaTransport& transport) {
    std::vector<P2pIbgdaTransportDevice*> peers(worldSize, nullptr);
    for (int r = 0; r < worldSize; ++r) {
      if (r != globalRank) {
        peers[r] = transport.getP2pTransportDevice(r);
      }
    }
    DeviceBuffer table(worldSize * sizeof(P2pIbgdaTransportDevice*));
    CUDACHECK_TEST(cudaMemcpy(
        table.get(),
        peers.data(),
        worldSize * sizeof(P2pIbgdaTransportDevice*),
        cudaMemcpyHostToDevice));
    return table;
  }

  /**
   * Run one chain pass and verify it.
   *
   * `use_dst` selects whether intermediates also copy into recv_buf. Ranks that
   * are expected to hold the data are checked byte-for-byte; `expectData`
   * decides which those are (all but rank 0 with a dst, only the last rank
   * without one).
   */
  void runChainCase(
      DeviceBuffer& peerTable,
      DeviceBuffer& sendBuf,
      DeviceBuffer& recvBuf,
      std::size_t nbytes,
      int numBlocks,
      const ChainMode& mode,
      bool use_dst,
      uint8_t base) {
    const std::vector<uint8_t> pattern = makePattern(base, nbytes);
    CUDACHECK_TEST(cudaMemcpy(
        sendBuf.get(), pattern.data(), nbytes, cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemset(recvBuf.get(), 0, nbytes));
    CUDACHECK_TEST(cudaDeviceSynchronize());
    bootstrap->barrierAll();

    auto* peers = static_cast<P2pIbgdaTransportDevice**>(peerTable.get());
    auto* src = static_cast<const char*>(sendBuf.get());
    auto* dst = static_cast<char*>(recvBuf.get());
    if (use_dst) {
      test::launch_recv_forward_chain(
          peers,
          src,
          dst,
          nbytes,
          globalRank,
          worldSize,
          numBlocks,
          stream_,
          mode.proto);
    } else {
      test::launch_recv_forward_chain_no_dst(
          peers,
          src,
          dst,
          nbytes,
          globalRank,
          worldSize,
          numBlocks,
          stream_,
          mode.proto);
    }

    cudaError_t err = cudaStreamSynchronize(stream_);
    ASSERT_EQ(err, cudaSuccess)
        << modeName(mode) << " kernel failed: " << cudaGetErrorString(err);
    bootstrap->barrierAll();

    const bool expectData =
        use_dst ? (globalRank != 0) : (globalRank == worldSize - 1);
    if (!expectData) {
      return;
    }
    std::vector<uint8_t> hostBuf(nbytes);
    CUDACHECK_TEST(cudaMemcpy(
        hostBuf.data(), recvBuf.get(), nbytes, cudaMemcpyDeviceToHost));

    std::size_t errors = 0;
    for (std::size_t i = 0; i < nbytes; ++i) {
      if (hostBuf[i] != expectedByte(base, i)) {
        if (errors < 10) {
          COMMS_LOG(
              ERR,
              "[{}] rank {}: byte {} expected 0x{:02X} got 0x{:02X}",
              modeName(mode),
              globalRank,
              i,
              expectedByte(base, i),
              hostBuf[i]);
        }
        ++errors;
      }
    }
    EXPECT_EQ(errors, 0u) << modeName(mode) << ": rank " << globalRank
                          << " saw " << errors << " byte mismatches";
  }

  cudaStream_t stream_{};
};

// Chain test: rank 0 → rank 1 → ... → rank N-1
// Each intermediate uses recv_forward with dst (copies to local buffer).
// Verify all ranks that receive data see the correct pattern.
TEST_F(RecvForwardChainTest, ForwardWithCopy) {
  if (worldSize < 2) {
    COMMS_LOG(INFO, "Skipping: requires >= 2 ranks, got {}", worldSize);
    return;
  }

  constexpr std::size_t kDataBytes = 1 * 1024 * 1024;
  constexpr int kNumBlocks = 4;

  try {
    auto transport = create_transport(kDataBytes, kNumBlocks);
    DeviceBuffer sendBuf(kDataBytes);
    DeviceBuffer recvBuf(kDataBytes);
    DeviceBuffer peerTable = makePeerTable(*transport);

    uint8_t base = 0xCA;
    for (const auto& mode : kChainModes) {
      runChainCase(
          peerTable,
          sendBuf,
          recvBuf,
          kDataBytes,
          kNumBlocks,
          mode,
          /*use_dst=*/true,
          base++);
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }
}

// Chain test with dst=nullptr for intermediates (forward-only).
// Only the last rank should have the data.
TEST_F(RecvForwardChainTest, ForwardOnly) {
  if (worldSize < 3) {
    COMMS_LOG(INFO, "Skipping: requires >= 3 ranks for forward-only test");
    return;
  }

  constexpr std::size_t kDataBytes = 1 * 1024 * 1024;
  constexpr int kNumBlocks = 4;

  try {
    auto transport = create_transport(kDataBytes, kNumBlocks);
    DeviceBuffer sendBuf(kDataBytes);
    DeviceBuffer recvBuf(kDataBytes);
    DeviceBuffer peerTable = makePeerTable(*transport);

    uint8_t base = 0xBE;
    for (const auto& mode : kChainModes) {
      runChainCase(
          peerTable,
          sendBuf,
          recvBuf,
          kDataBytes,
          kNumBlocks,
          mode,
          /*use_dst=*/false,
          base++);
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }
}

// 2-rank test: send → recv (no intermediates). Validates that the protocol
// is compatible when recv_forward is not involved — just send + recv.
TEST_F(RecvForwardChainTest, SendRecvDirect) {
  if (worldSize != 2) {
    COMMS_LOG(INFO, "Skipping: requires exactly 2 ranks, got {}", worldSize);
    return;
  }

  constexpr std::size_t kDataBytes = 2 * 1024 * 1024;
  constexpr int kNumBlocks = 4;

  try {
    auto transport = create_transport(kDataBytes, kNumBlocks);
    DeviceBuffer sendBuf(kDataBytes);
    DeviceBuffer recvBuf(kDataBytes);
    DeviceBuffer peerTable = makePeerTable(*transport);

    uint8_t base = 0x55;
    for (const auto& mode : kChainModes) {
      runChainCase(
          peerTable,
          sendBuf,
          recvBuf,
          kDataBytes,
          kNumBlocks,
          mode,
          /*use_dst=*/true,
          base++);
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }
}

// Multi-section test: transfer more data than one slot to exercise pipelining,
// slot reuse, and the generation advance that slot reuse implies. Without a
// transfer long enough to wrap the staging ring, every packet in an LL run
// carries generation 1 and a missing re-stamp cannot show up.
TEST_F(RecvForwardChainTest, MultiSection) {
  if (worldSize < 2) {
    COMMS_LOG(INFO, "Skipping: requires >= 2 ranks, got {}", worldSize);
    return;
  }

  constexpr std::size_t kSlotSize = 512 * 1024;
  constexpr std::size_t kDataBytes = 4 * kSlotSize; // 4 sections
  constexpr int kNumBlocks = 2;

  try {
    auto transport = create_transport(kSlotSize, kNumBlocks);
    DeviceBuffer sendBuf(kDataBytes);
    DeviceBuffer recvBuf(kDataBytes);
    DeviceBuffer peerTable = makePeerTable(*transport);

    uint8_t base = 0xDD;
    for (const auto& mode : kChainModes) {
      runChainCase(
          peerTable,
          sendBuf,
          recvBuf,
          kDataBytes,
          kNumBlocks,
          mode,
          /*use_dst=*/true,
          base++);
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }
}

// A transfer whose per-block size is not a multiple of the LL packet payload
// (kData = 4), so the final packet of each stream carries fewer than kData
// valid bytes alongside the packer's zero padding. Every size above is a clean
// multiple, which never exercises the valid_payload mask on the relay hop.
TEST_F(RecvForwardChainTest, PartialFinalPacket) {
  if (worldSize < 2) {
    COMMS_LOG(INFO, "Skipping: requires >= 2 ranks, got {}", worldSize);
    return;
  }

  // The slot size stays 16 B-aligned (the transport requires it of
  // perChannelSize); only the payload is odd. Block 0 takes a 16 B-aligned
  // share and the last block takes the remainder, so its stream ends
  // mid-packet: 131083 % 4 == 3.
  constexpr std::size_t kSlotSize = 256 * 1024;
  constexpr std::size_t kDataBytes = kSlotSize - 5;
  constexpr int kNumBlocks = 2;

  try {
    auto transport = create_transport(kSlotSize, kNumBlocks);
    DeviceBuffer sendBuf(kDataBytes);
    DeviceBuffer recvBuf(kDataBytes);
    DeviceBuffer peerTable = makePeerTable(*transport);

    uint8_t base = 0x33;
    for (const auto& mode : kChainModes) {
      runChainCase(
          peerTable,
          sendBuf,
          recvBuf,
          kDataBytes,
          kNumBlocks,
          mode,
          /*use_dst=*/true,
          base++);
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }
}

/**
 * Alternate Simple and LL forwards repeatedly on one channel.
 *
 * The two protocols own separate progress cursors and staging banks but SHARE
 * the channel's QPs and its recvDataReadyLaneCursor. select_put_lane_ordinal()
 * bumps the sender's cursor on every put regardless of protocol, so every LL
 * recv-side path has to bump the receiver's mirror to match. Skipping it
 * leaves the two one chunk apart per LL transfer, and the NEXT Simple receive
 * resolves to a lane the sender never wrote and blocks forever.
 *
 * All-Simple and all-LL runs cannot catch that -- the first keeps both
 * counters in step, the second never consults the lane mapping. Only an
 * interleaving does, which is why this case exists separately from the sweeps
 * above and alternates several times rather than once.
 */
TEST_F(RecvForwardChainTest, MixedProtocolAlternating) {
  if (worldSize < 2) {
    COMMS_LOG(INFO, "Skipping: requires >= 2 ranks, got {}", worldSize);
    return;
  }

  constexpr std::size_t kDataBytes = 256 * 1024;
  constexpr int kNumBlocks = 2;
  constexpr int kRounds = 3;

  try {
    auto transport = create_transport(kDataBytes, kNumBlocks);
    DeviceBuffer sendBuf(kDataBytes);
    DeviceBuffer recvBuf(kDataBytes);
    DeviceBuffer peerTable = makePeerTable(*transport);

    uint8_t base = 0x10;
    for (int round = 0; round < kRounds; ++round) {
      for (const auto proto :
           {test::ChainProto::Simple,
            test::ChainProto::LL,
            test::ChainProto::Simple}) {
        runChainCase(
            peerTable,
            sendBuf,
            recvBuf,
            kDataBytes,
            kNumBlocks,
            ChainMode{proto},
            /*use_dst=*/true,
            base++);
      }
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
