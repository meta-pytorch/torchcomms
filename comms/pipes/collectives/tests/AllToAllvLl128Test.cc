// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include <algorithm>

#include "comms/pipes/MultiPeerNvlTransport.h"
#include "comms/pipes/collectives/AllToAllvAuto.h"
#include "comms/pipes/collectives/AllToAllvLl128.cuh"
#include "comms/pipes/collectives/tests/AllToAllvLl128Test.cuh"
#include "comms/pipes/ll128/Ll128Packet.cuh"
#include "comms/pipes/tests/Utils.cuh"
#include "comms/testinfra/BenchmarkTestFixture.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::BenchmarkTestFixture;
using meta::comms::DeviceBuffer;

namespace comms::pipes {

class AllToAllvLl128TestFixture : public BenchmarkTestFixture {
 protected:
  void SetUp() override {
    BenchmarkTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }

  void TearDown() override {
    BenchmarkTestFixture::TearDown();
  }
};

// =============================================================================
// Equal-size tests
// =============================================================================

struct AllToAllvLl128EqualParams {
  int numBlocks;
  int blockSize;
  size_t numIntsPerRank; // Must be a multiple of 4 (16B alignment)
  size_t ll128BufferNumPackets{0}; // 0 = full-size buffer (no chunking)
  std::string testName;
};

class AllToAllvLl128EqualSizeTest
    : public AllToAllvLl128TestFixture,
      public ::testing::WithParamInterface<AllToAllvLl128EqualParams> {};

TEST_P(AllToAllvLl128EqualSizeTest, AllToAllvLl128EqualSize) {
  const auto& params = GetParam();
  const size_t numIntsPerRank = params.numIntsPerRank;
  const int numBlocks = params.numBlocks;
  const int blockSize = params.blockSize;

  XLOGF(
      DBG1,
      "Rank {}: Running {} with numBlocks={}, blockSize={}, numIntsPerRank={}",
      globalRank,
      params.testName,
      numBlocks,
      blockSize,
      numIntsPerRank);

  const size_t totalInts = numIntsPerRank * worldSize;
  const size_t bufferSize = totalInts * sizeof(int32_t);
  const size_t perPeerBytes = numIntsPerRank * sizeof(int32_t);

  // Transport config with LL128 buffers enabled
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = std::max(size_t(2048), bufferSize),
      .chunkSize = 512,
      .pipelineDepth = 4,
      .ll128BufferSize = (params.ll128BufferNumPackets > 0)
          ? params.ll128BufferNumPackets * kLl128PacketSize
          : ll128_buffer_size(perPeerBytes),
  };

  std::unique_ptr<MultiPeerNvlTransport> transport;
  try {
    transport = std::make_unique<MultiPeerNvlTransport>(
        globalRank, worldSize, bootstrap, config);
    transport->exchange();
  } catch (const std::runtime_error& e) {
    XLOGF(ERR, "Rank {}: transport init failed: {}", globalRank, e.what());
    std::abort();
  }

  auto transports_span = transport->getDeviceTransports();

  DeviceBuffer sendBuffer(bufferSize);
  DeviceBuffer recvBuffer(bufferSize);

  // Initialize recv buffer with -1
  test::fillBuffer(reinterpret_cast<int*>(recvBuffer.get()), -1, totalInts);

  // Fill send buffer: rank R sending to peer P at position i: R*1000 + P*100 +i
  std::vector<int32_t> h_send_init(totalInts);
  for (int peer = 0; peer < worldSize; peer++) {
    for (size_t i = 0; i < numIntsPerRank; i++) {
      h_send_init[peer * numIntsPerRank + i] =
          globalRank * 1000 + peer * 100 + static_cast<int32_t>(i);
    }
  }
  CUDACHECK_TEST(cudaMemcpy(
      sendBuffer.get(),
      h_send_init.data(),
      bufferSize,
      cudaMemcpyHostToDevice));

  // Setup ChunkInfo
  std::vector<ChunkInfo> h_send_chunk_infos;
  std::vector<ChunkInfo> h_recv_chunk_infos;
  for (int rank = 0; rank < worldSize; rank++) {
    size_t offset = rank * perPeerBytes;
    h_send_chunk_infos.emplace_back(offset, perPeerBytes);
    h_recv_chunk_infos.emplace_back(offset, perPeerBytes);
  }

  DeviceBuffer d_send_chunk_infos(sizeof(ChunkInfo) * worldSize);
  DeviceBuffer d_recv_chunk_infos(sizeof(ChunkInfo) * worldSize);
  CUDACHECK_TEST(cudaMemcpy(
      d_send_chunk_infos.get(),
      h_send_chunk_infos.data(),
      sizeof(ChunkInfo) * worldSize,
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemcpy(
      d_recv_chunk_infos.get(),
      h_recv_chunk_infos.data(),
      sizeof(ChunkInfo) * worldSize,
      cudaMemcpyHostToDevice));

  DeviceSpan<ChunkInfo> send_chunk_infos(
      static_cast<ChunkInfo*>(d_send_chunk_infos.get()), worldSize);
  DeviceSpan<ChunkInfo> recv_chunk_infos(
      static_cast<ChunkInfo*>(d_recv_chunk_infos.get()), worldSize);

  bootstrap->barrierAll();

  test::test_all_to_allv_ll128(
      recvBuffer.get(),
      sendBuffer.get(),
      globalRank,
      worldSize,
      transports_span,
      send_chunk_infos,
      recv_chunk_infos,
      numBlocks,
      blockSize);

  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Verify received data
  std::vector<int32_t> h_recv_after(totalInts);
  CUDACHECK_TEST(cudaMemcpy(
      h_recv_after.data(),
      recvBuffer.get(),
      bufferSize,
      cudaMemcpyDeviceToHost));

  int h_errorCount = 0;
  for (int peer = 0; peer < worldSize; peer++) {
    for (size_t i = 0; i < numIntsPerRank; i++) {
      int32_t expected =
          peer * 1000 + globalRank * 100 + static_cast<int32_t>(i);
      int32_t actual = h_recv_after[peer * numIntsPerRank + i];
      if (expected != actual) {
        h_errorCount++;
        if (h_errorCount <= 10) {
          XLOGF(
              ERR,
              "Rank {}: Error at peer {} position {}: expected {}, got {}",
              globalRank,
              peer,
              i,
              expected,
              actual);
        }
      }
    }
  }

  EXPECT_EQ(h_errorCount, 0) << "Rank " << globalRank << " found "
                             << h_errorCount << " verification errors";
  bootstrap->barrierAll();
}

INSTANTIATE_TEST_SUITE_P(
    EqualSizeConfigs,
    AllToAllvLl128EqualSizeTest,
    ::testing::Values(
        // 64B per peer (16 ints)
        AllToAllvLl128EqualParams{
            .numBlocks = 18,
            .blockSize = 256,
            .numIntsPerRank = 16,
            .testName = "18b_256t_64B"},
        // 256B per peer (64 ints)
        AllToAllvLl128EqualParams{
            .numBlocks = 18,
            .blockSize = 512,
            .numIntsPerRank = 64,
            .testName = "18b_256t_256B"},
        // 1KB per peer (256 ints)
        AllToAllvLl128EqualParams{
            .numBlocks = 18,
            .blockSize = 512,
            .numIntsPerRank = 256,
            .testName = "18b_256t_1KB"},
        // 4KB per peer (1024 ints)
        AllToAllvLl128EqualParams{
            .numBlocks = 18,
            .blockSize = 512,
            .numIntsPerRank = 1024,
            .testName = "18b_256t_4KB"},
        // 16KB per peer (4096 ints)
        AllToAllvLl128EqualParams{
            .numBlocks = 18,
            .blockSize = 512,
            .numIntsPerRank = 4096,
            .testName = "18b_256t_16KB"},
        // 64KB per peer (16384 ints)
        AllToAllvLl128EqualParams{
            .numBlocks = 18,
            .blockSize = 512,
            .numIntsPerRank = 16384,
            .testName = "18b_256t_64KB"},
        // 256KB per peer (65536 ints)
        AllToAllvLl128EqualParams{
            .numBlocks = 18,
            .blockSize = 512,
            .numIntsPerRank = 65536,
            .testName = "18b_256t_256KB"}),
    [](const ::testing::TestParamInfo<AllToAllvLl128EqualParams>& info) {
      return info.param.testName;
    });

// =============================================================================
// Edge case tests: partial packets, exact boundaries, zero bytes
// =============================================================================

INSTANTIATE_TEST_SUITE_P(
    EdgeCaseConfigs,
    AllToAllvLl128EqualSizeTest,
    ::testing::Values(
        // 48B per peer (12 ints) — partial packet (< 120B payload)
        AllToAllvLl128EqualParams{
            .numBlocks = 18,
            .blockSize = 256,
            .numIntsPerRank = 12,
            .testName = "18b_256t_48B_partial"},
        // 128B per peer (32 ints) — not a multiple of 120B payload
        AllToAllvLl128EqualParams{
            .numBlocks = 18,
            .blockSize = 512,
            .numIntsPerRank = 32,
            .testName = "18b_256t_128B_partial"},
        // 192B per peer (48 ints) — partial last packet
        AllToAllvLl128EqualParams{
            .numBlocks = 18,
            .blockSize = 512,
            .numIntsPerRank = 48,
            .testName = "18b_256t_192B_partial"},
        // 240B per peer (60 ints) — exactly 2 full packets (2 * 120B payload)
        AllToAllvLl128EqualParams{
            .numBlocks = 18,
            .blockSize = 512,
            .numIntsPerRank = 60,
            .testName = "18b_256t_240B_exact"}),
    [](const ::testing::TestParamInfo<AllToAllvLl128EqualParams>& info) {
      return info.param.testName;
    });

// =============================================================================
// Chunked buffer tests (ll128BufferSize < message size)
// =============================================================================

INSTANTIATE_TEST_SUITE_P(
    ChunkedConfigs,
    AllToAllvLl128EqualSizeTest,
    ::testing::Values(
        // 4KB per peer (35 packets) with 4-packet buffer → heavy chunking
        // (1 active warp, 9 rounds through full collective stack)
        AllToAllvLl128EqualParams{
            .numBlocks = 18,
            .blockSize = 512,
            .numIntsPerRank = 1024,
            .ll128BufferNumPackets = 4,
            .testName = "18b_256t_4KB_chunked_4pkt"},
        // 64KB per peer (547 packets) with 8-packet buffer → many rounds
        // (2 active warps, large message with small buffer, multi-peer)
        AllToAllvLl128EqualParams{
            .numBlocks = 18,
            .blockSize = 512,
            .numIntsPerRank = 16384,
            .ll128BufferNumPackets = 8,
            .testName = "18b_256t_64KB_chunked_8pkt"},
        // 16KB per peer (137 packets) with 32-packet buffer → moderate chunking
        // (8 active warps, warp clamping with multi-peer collective)
        AllToAllvLl128EqualParams{
            .numBlocks = 18,
            .blockSize = 512,
            .numIntsPerRank = 4096,
            .ll128BufferNumPackets = 32,
            .testName = "18b_256t_16KB_chunked_32pkt"}),
    [](const ::testing::TestParamInfo<AllToAllvLl128EqualParams>& info) {
      return info.param.testName;
    });

// =============================================================================
// Chunked buffer + low block count tests
// Exercise LL128 buffer wrapping with few blocks. Warp clamping limits active
// warps to buffer capacity, so these test correctness of the clamping logic
// and per-packet flag protocol under constrained configurations.
// =============================================================================

INSTANTIATE_TEST_SUITE_P(
    ChunkedLowBlockConfigs,
    AllToAllvLl128EqualSizeTest,
    ::testing::Values(
        // 2 blocks, 8-packet buffer — minimal warps per peer, moderate wrapping
        AllToAllvLl128EqualParams{
            .numBlocks = 2,
            .blockSize = 512,
            .numIntsPerRank = 16384,
            .ll128BufferNumPackets = 8,
            .testName = "2b_512t_64KB_chunked_8pkt"},
        // 2 blocks, larger message — more buffer wrap-arounds
        AllToAllvLl128EqualParams{
            .numBlocks = 2,
            .blockSize = 512,
            .numIntsPerRank = 65536,
            .ll128BufferNumPackets = 8,
            .testName = "2b_512t_256KB_chunked_8pkt"},
        // 4 blocks — slightly more warps, still constrained
        AllToAllvLl128EqualParams{
            .numBlocks = 4,
            .blockSize = 512,
            .numIntsPerRank = 16384,
            .ll128BufferNumPackets = 8,
            .testName = "4b_512t_64KB_chunked_8pkt"},
        // 2 blocks, 4-packet buffer — smallest buffer, most wrapping
        AllToAllvLl128EqualParams{
            .numBlocks = 2,
            .blockSize = 512,
            .numIntsPerRank = 16384,
            .ll128BufferNumPackets = 4,
            .testName = "2b_512t_64KB_chunked_4pkt"}),
    [](const ::testing::TestParamInfo<AllToAllvLl128EqualParams>& info) {
      return info.param.testName;
    });

// =============================================================================
// Chunked buffer + high block count tests
// Exercise LL128 chunking with many blocks, matching the Ll128BlockThreadSweep
// benchmark configuration. High block counts create many idle warps due to warp
// clamping — tests that the idle warps correctly skip all buffer operations and
// that active warps complete all multi-step rounds.
// =============================================================================

INSTANTIATE_TEST_SUITE_P(
    ChunkedHighBlockConfigs,
    AllToAllvLl128EqualSizeTest,
    ::testing::Values(
        // 64 blocks, 8-packet buffer, 64KB — many idle warps
        AllToAllvLl128EqualParams{
            .numBlocks = 64,
            .blockSize = 512,
            .numIntsPerRank = 16384,
            .ll128BufferNumPackets = 8,
            .testName = "64b_512t_64KB_chunked_8pkt"},
        // 128 blocks, 8-packet buffer, 64KB — extreme warp over-provisioning
        AllToAllvLl128EqualParams{
            .numBlocks = 128,
            .blockSize = 512,
            .numIntsPerRank = 16384,
            .ll128BufferNumPackets = 8,
            .testName = "128b_512t_64KB_chunked_8pkt"},
        // 64 blocks, 32-packet buffer, 64KB — more active warps, less rounds
        AllToAllvLl128EqualParams{
            .numBlocks = 64,
            .blockSize = 512,
            .numIntsPerRank = 16384,
            .ll128BufferNumPackets = 32,
            .testName = "64b_512t_64KB_chunked_32pkt"},
        // 64 blocks, 8-packet buffer, 4KB — small message, high block count
        AllToAllvLl128EqualParams{
            .numBlocks = 64,
            .blockSize = 512,
            .numIntsPerRank = 1024,
            .ll128BufferNumPackets = 8,
            .testName = "64b_512t_4KB_chunked_8pkt"}),
    [](const ::testing::TestParamInfo<AllToAllvLl128EqualParams>& info) {
      return info.param.testName;
    });

// =============================================================================
// Chunked buffer boundary tests
// Exercise exact boundary conditions: buffer is one packet smaller than the
// message (minimal chunking) or exactly half the message (even split).
// =============================================================================

INSTANTIATE_TEST_SUITE_P(
    ChunkedBoundaryConfigs,
    AllToAllvLl128EqualSizeTest,
    ::testing::Values(
        // 4KB per peer = 35 packets, buffer = 34 packets — minimal chunking
        // (one extra round for the last packet, tests wrap-around at boundary)
        AllToAllvLl128EqualParams{
            .numBlocks = 18,
            .blockSize = 512,
            .numIntsPerRank = 1024,
            .ll128BufferNumPackets = 34,
            .testName = "18b_512t_4KB_chunked_34pkt_minimal"},
        // 4KB per peer = 35 packets, buffer = 16 packets — ~2.2x wrap
        AllToAllvLl128EqualParams{
            .numBlocks = 18,
            .blockSize = 512,
            .numIntsPerRank = 1024,
            .ll128BufferNumPackets = 16,
            .testName = "18b_512t_4KB_chunked_16pkt_half"},
        // 1KB per peer = 9 packets, buffer = 8 packets — one-over boundary
        AllToAllvLl128EqualParams{
            .numBlocks = 18,
            .blockSize = 512,
            .numIntsPerRank = 256,
            .ll128BufferNumPackets = 8,
            .testName = "18b_512t_1KB_chunked_8pkt_one_over"},
        // 1KB per peer = 9 packets, buffer = 4 packets — 3 rounds
        AllToAllvLl128EqualParams{
            .numBlocks = 4,
            .blockSize = 512,
            .numIntsPerRank = 256,
            .ll128BufferNumPackets = 4,
            .testName = "4b_512t_1KB_chunked_4pkt_3rounds"}),
    [](const ::testing::TestParamInfo<AllToAllvLl128EqualParams>& info) {
      return info.param.testName;
    });

// =============================================================================
// Unequal-size tests
// =============================================================================

struct AllToAllvLl128UnequalParams {
  int numBlocks;
  int blockSize;
  size_t base_ints; // Must be a multiple of 4 (16B alignment)
  std::string testName;
};

class AllToAllvLl128UnequalSizeTest
    : public AllToAllvLl128TestFixture,
      public ::testing::WithParamInterface<AllToAllvLl128UnequalParams> {};

TEST_P(AllToAllvLl128UnequalSizeTest, AllToAllvLl128UnequalSize) {
  const auto& params = GetParam();
  const int numBlocks = params.numBlocks;
  const int blockSize = params.blockSize;
  const size_t base_ints = params.base_ints;

  XLOGF(
      DBG1,
      "Rank {}: Running {} with numBlocks={}, blockSize={}, base_ints={}",
      globalRank,
      params.testName,
      numBlocks,
      blockSize,
      base_ints);

  // Max per-peer size: (2*worldSize - 1) * base_ints * sizeof(int32_t)
  size_t maxPerPeerInts = (2 * worldSize - 1) * base_ints;
  size_t maxPerPeerBytes = maxPerPeerInts * sizeof(int32_t);

  // Calculate max total buffer needed for any rank
  size_t max_total_ints = worldSize * (2 * worldSize - 1 + 1) / 2 * base_ints;
  size_t max_buffer_size = max_total_ints * sizeof(int32_t);

  // Transport config with LL128 buffers
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = std::max(size_t(2048), max_buffer_size),
      .chunkSize = 512,
      .pipelineDepth = 4,
      .ll128BufferSize = ll128_buffer_size(maxPerPeerBytes),
  };

  std::unique_ptr<MultiPeerNvlTransport> transport;
  try {
    transport = std::make_unique<MultiPeerNvlTransport>(
        globalRank, worldSize, bootstrap, config);
    transport->exchange();
  } catch (const std::runtime_error& e) {
    XLOGF(ERR, "Rank {}: transport init failed: {}", globalRank, e.what());
    std::abort();
  }

  auto transports_span = transport->getDeviceTransports();

  // Calculate variable chunk sizes: (globalRank + rank + 1) * base_ints
  std::vector<ChunkInfo> h_send_chunk_infos;
  std::vector<ChunkInfo> h_recv_chunk_infos;

  size_t send_offset = 0;
  size_t recv_offset = 0;

  for (int rank = 0; rank < worldSize; rank++) {
    size_t num_ints = (globalRank + rank + 1) * base_ints;
    size_t nbytes = num_ints * sizeof(int32_t);
    h_send_chunk_infos.emplace_back(send_offset, nbytes);
    h_recv_chunk_infos.emplace_back(recv_offset, nbytes);
    send_offset += nbytes;
    recv_offset += nbytes;
  }

  const size_t sendBufferSize = send_offset;
  const size_t recvBufferSize = recv_offset;

  DeviceBuffer sendBuffer(sendBufferSize);
  DeviceBuffer recvBuffer(recvBufferSize);

  // Initialize recv buffer with -1
  std::vector<int32_t> h_recv_init(recvBufferSize / sizeof(int32_t), -1);
  CUDACHECK_TEST(cudaMemcpy(
      recvBuffer.get(),
      h_recv_init.data(),
      recvBufferSize,
      cudaMemcpyHostToDevice));

  // Fill send buffer with pattern
  std::vector<int32_t> h_send_init(sendBufferSize / sizeof(int32_t));
  size_t int_offset = 0;
  for (int peer = 0; peer < worldSize; peer++) {
    size_t num_ints = (globalRank + peer + 1) * base_ints;
    for (size_t i = 0; i < num_ints; i++) {
      h_send_init[int_offset + i] =
          globalRank * 1000 + peer * 100 + static_cast<int32_t>(i);
    }
    int_offset += num_ints;
  }
  CUDACHECK_TEST(cudaMemcpy(
      sendBuffer.get(),
      h_send_init.data(),
      sendBufferSize,
      cudaMemcpyHostToDevice));

  // Copy chunk infos to device
  DeviceBuffer d_send_chunk_infos(sizeof(ChunkInfo) * worldSize);
  DeviceBuffer d_recv_chunk_infos(sizeof(ChunkInfo) * worldSize);
  CUDACHECK_TEST(cudaMemcpy(
      d_send_chunk_infos.get(),
      h_send_chunk_infos.data(),
      sizeof(ChunkInfo) * worldSize,
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemcpy(
      d_recv_chunk_infos.get(),
      h_recv_chunk_infos.data(),
      sizeof(ChunkInfo) * worldSize,
      cudaMemcpyHostToDevice));

  DeviceSpan<ChunkInfo> send_chunk_infos(
      static_cast<ChunkInfo*>(d_send_chunk_infos.get()), worldSize);
  DeviceSpan<ChunkInfo> recv_chunk_infos(
      static_cast<ChunkInfo*>(d_recv_chunk_infos.get()), worldSize);

  bootstrap->barrierAll();

  test::test_all_to_allv_ll128(
      recvBuffer.get(),
      sendBuffer.get(),
      globalRank,
      worldSize,
      transports_span,
      send_chunk_infos,
      recv_chunk_infos,
      numBlocks,
      blockSize);

  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Verify received data
  std::vector<int32_t> h_recv_after(recvBufferSize / sizeof(int32_t));
  CUDACHECK_TEST(cudaMemcpy(
      h_recv_after.data(),
      recvBuffer.get(),
      recvBufferSize,
      cudaMemcpyDeviceToHost));

  int h_errorCount = 0;
  int_offset = 0;

  for (int peer = 0; peer < worldSize; peer++) {
    size_t num_ints = (globalRank + peer + 1) * base_ints;
    for (size_t i = 0; i < num_ints; i++) {
      int32_t expected =
          peer * 1000 + globalRank * 100 + static_cast<int32_t>(i);
      int32_t actual = h_recv_after[int_offset + i];
      if (expected != actual) {
        h_errorCount++;
        if (h_errorCount <= 10) {
          XLOGF(
              ERR,
              "Rank {}: Error at peer {} position {}: expected {}, got {}",
              globalRank,
              peer,
              i,
              expected,
              actual);
        }
      }
    }
    int_offset += num_ints;
  }

  EXPECT_EQ(h_errorCount, 0) << "Rank " << globalRank << " found "
                             << h_errorCount << " verification errors";
  bootstrap->barrierAll();
}

INSTANTIATE_TEST_SUITE_P(
    UnequalSizeConfigs,
    AllToAllvLl128UnequalSizeTest,
    ::testing::Values(
        // base_ints=16, max=(2*8-1)*16*4 = 960B
        AllToAllvLl128UnequalParams{
            .numBlocks = 18,
            .blockSize = 256,
            .base_ints = 16,
            .testName = "18b_256t_960B"},
        // base_ints=64, max=(2*8-1)*64*4 = 3840B
        AllToAllvLl128UnequalParams{
            .numBlocks = 18,
            .blockSize = 512,
            .base_ints = 64,
            .testName = "18b_256t_3840B"},
        // base_ints=256, max=(2*8-1)*256*4 = 15KB
        AllToAllvLl128UnequalParams{
            .numBlocks = 18,
            .blockSize = 512,
            .base_ints = 256,
            .testName = "18b_256t_15KB"},
        // base_ints=512, max=(2*8-1)*512*4 = 30KB
        AllToAllvLl128UnequalParams{
            .numBlocks = 18,
            .blockSize = 512,
            .base_ints = 512,
            .testName = "18b_256t_30KB"}),
    [](const ::testing::TestParamInfo<AllToAllvLl128UnequalParams>& info) {
      return info.param.testName;
    });

// =============================================================================
// Unequal-size with zero-byte peers
// =============================================================================

struct AllToAllvLl128ZeroPeerParams {
  int numBlocks;
  int blockSize;
  size_t base_ints;
  std::string testName;
};

class AllToAllvLl128ZeroPeerTest
    : public AllToAllvLl128TestFixture,
      public ::testing::WithParamInterface<AllToAllvLl128ZeroPeerParams> {};

// Test where some peers have 0 bytes — exercises LL128's nbytes==0 early return
TEST_P(AllToAllvLl128ZeroPeerTest, AllToAllvLl128ZeroPeer) {
  const auto& params = GetParam();
  const int numBlocks = params.numBlocks;
  const int blockSize = params.blockSize;
  const size_t base_ints = params.base_ints;

  XLOGF(
      DBG1,
      "Rank {}: Running {} with zero-byte peers",
      globalRank,
      params.testName);

  // Use formula: (globalRank + rank) * base_ints
  // When globalRank==0 and rank==0, size is 0 (zero-byte self-copy)
  // When globalRank==0 and rank==1, size is base_ints (non-zero peer)
  size_t maxPerPeerInts = (2 * worldSize - 2) * base_ints;
  size_t maxPerPeerBytes = maxPerPeerInts * sizeof(int32_t);

  size_t max_total_ints = worldSize * (2 * worldSize - 2 + 0) / 2 * base_ints;
  size_t max_buffer_size =
      std::max(size_t(16), max_total_ints * sizeof(int32_t));

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = std::max(size_t(2048), max_buffer_size),
      .chunkSize = 512,
      .pipelineDepth = 4,
      .ll128BufferSize =
          ll128_buffer_size(std::max(size_t(16), maxPerPeerBytes)),
  };

  std::unique_ptr<MultiPeerNvlTransport> transport;
  try {
    transport = std::make_unique<MultiPeerNvlTransport>(
        globalRank, worldSize, bootstrap, config);
    transport->exchange();
  } catch (const std::runtime_error& e) {
    XLOGF(ERR, "Rank {}: transport init failed: {}", globalRank, e.what());
    std::abort();
  }

  auto transports_span = transport->getDeviceTransports();

  // Variable sizes: (globalRank + rank) * base_ints — rank 0 to rank 0 = 0
  std::vector<ChunkInfo> h_send_chunk_infos;
  std::vector<ChunkInfo> h_recv_chunk_infos;

  size_t send_offset = 0;
  size_t recv_offset = 0;

  for (int rank = 0; rank < worldSize; rank++) {
    size_t num_ints = (globalRank + rank) * base_ints;
    size_t nbytes = num_ints * sizeof(int32_t);
    h_send_chunk_infos.emplace_back(send_offset, nbytes);
    h_recv_chunk_infos.emplace_back(recv_offset, nbytes);
    send_offset += nbytes;
    recv_offset += nbytes;
  }

  const size_t sendBufferSize = std::max(size_t(16), send_offset);
  const size_t recvBufferSize = std::max(size_t(16), recv_offset);

  DeviceBuffer sendBuffer(sendBufferSize);
  DeviceBuffer recvBuffer(recvBufferSize);

  // Initialize recv buffer with -1
  size_t recvInts = recv_offset / sizeof(int32_t);
  if (recvInts > 0) {
    std::vector<int32_t> h_recv_init(recvInts, -1);
    CUDACHECK_TEST(cudaMemcpy(
        recvBuffer.get(),
        h_recv_init.data(),
        recv_offset,
        cudaMemcpyHostToDevice));
  }

  // Fill send buffer
  size_t sendInts = send_offset / sizeof(int32_t);
  if (sendInts > 0) {
    std::vector<int32_t> h_send_init(sendInts);
    size_t int_offset = 0;
    for (int peer = 0; peer < worldSize; peer++) {
      size_t num_ints = (globalRank + peer) * base_ints;
      for (size_t i = 0; i < num_ints; i++) {
        h_send_init[int_offset + i] =
            globalRank * 1000 + peer * 100 + static_cast<int32_t>(i);
      }
      int_offset += num_ints;
    }
    CUDACHECK_TEST(cudaMemcpy(
        sendBuffer.get(),
        h_send_init.data(),
        send_offset,
        cudaMemcpyHostToDevice));
  }

  DeviceBuffer d_send_chunk_infos(sizeof(ChunkInfo) * worldSize);
  DeviceBuffer d_recv_chunk_infos(sizeof(ChunkInfo) * worldSize);
  CUDACHECK_TEST(cudaMemcpy(
      d_send_chunk_infos.get(),
      h_send_chunk_infos.data(),
      sizeof(ChunkInfo) * worldSize,
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemcpy(
      d_recv_chunk_infos.get(),
      h_recv_chunk_infos.data(),
      sizeof(ChunkInfo) * worldSize,
      cudaMemcpyHostToDevice));

  DeviceSpan<ChunkInfo> send_chunk_infos(
      static_cast<ChunkInfo*>(d_send_chunk_infos.get()), worldSize);
  DeviceSpan<ChunkInfo> recv_chunk_infos(
      static_cast<ChunkInfo*>(d_recv_chunk_infos.get()), worldSize);

  bootstrap->barrierAll();

  test::test_all_to_allv_ll128(
      recvBuffer.get(),
      sendBuffer.get(),
      globalRank,
      worldSize,
      transports_span,
      send_chunk_infos,
      recv_chunk_infos,
      numBlocks,
      blockSize);

  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Verify
  if (recvInts > 0) {
    std::vector<int32_t> h_recv_after(recvInts);
    CUDACHECK_TEST(cudaMemcpy(
        h_recv_after.data(),
        recvBuffer.get(),
        recv_offset,
        cudaMemcpyDeviceToHost));

    int h_errorCount = 0;
    size_t int_offset = 0;

    for (int peer = 0; peer < worldSize; peer++) {
      size_t num_ints = (globalRank + peer) * base_ints;
      for (size_t i = 0; i < num_ints; i++) {
        int32_t expected =
            peer * 1000 + globalRank * 100 + static_cast<int32_t>(i);
        int32_t actual = h_recv_after[int_offset + i];
        if (expected != actual) {
          h_errorCount++;
          if (h_errorCount <= 10) {
            XLOGF(
                ERR,
                "Rank {}: Error at peer {} position {}: expected {}, got {}",
                globalRank,
                peer,
                i,
                expected,
                actual);
          }
        }
      }
      int_offset += num_ints;
    }

    EXPECT_EQ(h_errorCount, 0) << "Rank " << globalRank << " found "
                               << h_errorCount << " verification errors";
  }

  bootstrap->barrierAll();
}

INSTANTIATE_TEST_SUITE_P(
    ZeroPeerConfigs,
    AllToAllvLl128ZeroPeerTest,
    ::testing::Values(
        AllToAllvLl128ZeroPeerParams{
            .numBlocks = 18,
            .blockSize = 512,
            .base_ints = 16,
            .testName = "18b_256t_zero_peers"}),
    [](const ::testing::TestParamInfo<AllToAllvLl128ZeroPeerParams>& info) {
      return info.param.testName;
    });

// =============================================================================
// Pipelined multi-call test
// =============================================================================

TEST_F(AllToAllvLl128TestFixture, PipelinedMultiCall) {
  CUDACHECK_TEST(cudaSetDevice(localRank));

  const size_t numIntsPerRank = 256; // 1KB per peer
  const int numBlocks = 18;
  const int blockSize = 512;
  const int kNumIterations = 5;

  const size_t totalInts = numIntsPerRank * worldSize;
  const size_t bufferSize = totalInts * sizeof(int32_t);
  const size_t perPeerBytes = numIntsPerRank * sizeof(int32_t);

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = std::max(size_t(2048), bufferSize),
      .chunkSize = 512,
      .pipelineDepth = 4,
      .ll128BufferSize = ll128_buffer_size(perPeerBytes),
  };

  std::unique_ptr<MultiPeerNvlTransport> transport;
  try {
    transport = std::make_unique<MultiPeerNvlTransport>(
        globalRank, worldSize, bootstrap, config);
    transport->exchange();
  } catch (const std::runtime_error& e) {
    XLOGF(ERR, "Rank {}: transport init failed: {}", globalRank, e.what());
    std::abort();
  }

  auto transports_span = transport->getDeviceTransports();

  DeviceBuffer sendBuffer(bufferSize);
  DeviceBuffer recvBuffer(bufferSize);

  std::vector<ChunkInfo> h_send_chunk_infos;
  std::vector<ChunkInfo> h_recv_chunk_infos;
  for (int rank = 0; rank < worldSize; rank++) {
    size_t offset = rank * perPeerBytes;
    h_send_chunk_infos.emplace_back(offset, perPeerBytes);
    h_recv_chunk_infos.emplace_back(offset, perPeerBytes);
  }

  DeviceBuffer d_send_chunk_infos(sizeof(ChunkInfo) * worldSize);
  DeviceBuffer d_recv_chunk_infos(sizeof(ChunkInfo) * worldSize);
  CUDACHECK_TEST(cudaMemcpy(
      d_send_chunk_infos.get(),
      h_send_chunk_infos.data(),
      sizeof(ChunkInfo) * worldSize,
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemcpy(
      d_recv_chunk_infos.get(),
      h_recv_chunk_infos.data(),
      sizeof(ChunkInfo) * worldSize,
      cudaMemcpyHostToDevice));

  DeviceSpan<ChunkInfo> send_chunk_infos(
      static_cast<ChunkInfo*>(d_send_chunk_infos.get()), worldSize);
  DeviceSpan<ChunkInfo> recv_chunk_infos(
      static_cast<ChunkInfo*>(d_recv_chunk_infos.get()), worldSize);

  // Run multiple iterations
  for (int iter = 0; iter < kNumIterations; iter++) {
    // Fill send buffer with iteration-dependent data
    std::vector<int32_t> h_send(totalInts);
    for (int peer = 0; peer < worldSize; peer++) {
      for (size_t i = 0; i < numIntsPerRank; i++) {
        h_send[peer * numIntsPerRank + i] = globalRank * 10000 + peer * 1000 +
            iter * 100 + static_cast<int32_t>(i);
      }
    }
    CUDACHECK_TEST(cudaMemcpy(
        sendBuffer.get(), h_send.data(), bufferSize, cudaMemcpyHostToDevice));

    // Clear recv buffer
    test::fillBuffer(reinterpret_cast<int*>(recvBuffer.get()), -1, totalInts);

    bootstrap->barrierAll();

    test::test_all_to_allv_ll128(
        recvBuffer.get(),
        sendBuffer.get(),
        globalRank,
        worldSize,
        transports_span,
        send_chunk_infos,
        recv_chunk_infos,
        numBlocks,
        blockSize);

    CUDACHECK_TEST(cudaDeviceSynchronize());

    // Verify received data
    std::vector<int32_t> h_recv(totalInts);
    CUDACHECK_TEST(cudaMemcpy(
        h_recv.data(), recvBuffer.get(), bufferSize, cudaMemcpyDeviceToHost));

    int h_errorCount = 0;
    for (int peer = 0; peer < worldSize; peer++) {
      for (size_t i = 0; i < numIntsPerRank; i++) {
        int32_t expected = peer * 10000 + globalRank * 1000 + iter * 100 +
            static_cast<int32_t>(i);
        int32_t actual = h_recv[peer * numIntsPerRank + i];
        if (expected != actual) {
          h_errorCount++;
          if (h_errorCount <= 5) {
            XLOGF(
                ERR,
                "Rank {}: Iter {} error at peer {} pos {}: expected {}, got {}",
                globalRank,
                iter,
                peer,
                i,
                expected,
                actual);
          }
        }
      }
    }

    EXPECT_EQ(h_errorCount, 0)
        << "Rank " << globalRank << " iter " << iter << " found "
        << h_errorCount << " verification errors";
  }

  bootstrap->barrierAll();
}

// =============================================================================
// Pipelined multi-call test with chunked LL128 buffer
// Exercises multiple kernel launches on the same transport with a buffer
// smaller than the message. This is the pattern used by the
// Ll128BlockThreadSweep benchmark: same transport, no buffer reset between
// iterations, chunked path.
// =============================================================================

TEST_F(AllToAllvLl128TestFixture, PipelinedMultiCallChunked) {
  CUDACHECK_TEST(cudaSetDevice(localRank));

  const size_t numIntsPerRank = 16384; // 64KB per peer
  const int numBlocks = 18;
  const int blockSize = 512;
  const int kNumIterations = 10;
  const size_t ll128BufferNumPackets = 8; // Heavy chunking

  const size_t totalInts = numIntsPerRank * worldSize;
  const size_t bufferSize = totalInts * sizeof(int32_t);
  const size_t perPeerBytes = numIntsPerRank * sizeof(int32_t);

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = std::max(size_t(2048), bufferSize),
      .chunkSize = 512,
      .pipelineDepth = 4,
      .ll128BufferSize = ll128BufferNumPackets * kLl128PacketSize,
  };

  std::unique_ptr<MultiPeerNvlTransport> transport;
  try {
    transport = std::make_unique<MultiPeerNvlTransport>(
        globalRank, worldSize, bootstrap, config);
    transport->exchange();
  } catch (const std::runtime_error& e) {
    XLOGF(ERR, "Rank {}: transport init failed: {}", globalRank, e.what());
    std::abort();
  }

  auto transports_span = transport->getDeviceTransports();

  DeviceBuffer sendBuffer(bufferSize);
  DeviceBuffer recvBuffer(bufferSize);

  std::vector<ChunkInfo> h_send_chunk_infos;
  std::vector<ChunkInfo> h_recv_chunk_infos;
  for (int rank = 0; rank < worldSize; rank++) {
    size_t offset = rank * perPeerBytes;
    h_send_chunk_infos.emplace_back(offset, perPeerBytes);
    h_recv_chunk_infos.emplace_back(offset, perPeerBytes);
  }

  DeviceBuffer d_send_chunk_infos(sizeof(ChunkInfo) * worldSize);
  DeviceBuffer d_recv_chunk_infos(sizeof(ChunkInfo) * worldSize);
  CUDACHECK_TEST(cudaMemcpy(
      d_send_chunk_infos.get(),
      h_send_chunk_infos.data(),
      sizeof(ChunkInfo) * worldSize,
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemcpy(
      d_recv_chunk_infos.get(),
      h_recv_chunk_infos.data(),
      sizeof(ChunkInfo) * worldSize,
      cudaMemcpyHostToDevice));

  DeviceSpan<ChunkInfo> send_chunk_infos(
      static_cast<ChunkInfo*>(d_send_chunk_infos.get()), worldSize);
  DeviceSpan<ChunkInfo> recv_chunk_infos(
      static_cast<ChunkInfo*>(d_recv_chunk_infos.get()), worldSize);

  // Run multiple iterations WITHOUT resetting the LL128 buffer — same as
  // the benchmark loop. Each iteration reuses the same transport and buffer.
  for (int iter = 0; iter < kNumIterations; iter++) {
    // Fill send buffer with iteration-dependent data
    std::vector<int32_t> h_send(totalInts);
    for (int peer = 0; peer < worldSize; peer++) {
      for (size_t i = 0; i < numIntsPerRank; i++) {
        h_send[peer * numIntsPerRank + i] = globalRank * 10000 + peer * 1000 +
            iter * 100 + static_cast<int32_t>(i);
      }
    }
    CUDACHECK_TEST(cudaMemcpy(
        sendBuffer.get(), h_send.data(), bufferSize, cudaMemcpyHostToDevice));

    // Clear recv buffer
    test::fillBuffer(reinterpret_cast<int*>(recvBuffer.get()), -1, totalInts);

    bootstrap->barrierAll();

    test::test_all_to_allv_ll128(
        recvBuffer.get(),
        sendBuffer.get(),
        globalRank,
        worldSize,
        transports_span,
        send_chunk_infos,
        recv_chunk_infos,
        numBlocks,
        blockSize);

    CUDACHECK_TEST(cudaDeviceSynchronize());

    // Verify received data
    std::vector<int32_t> h_recv(totalInts);
    CUDACHECK_TEST(cudaMemcpy(
        h_recv.data(), recvBuffer.get(), bufferSize, cudaMemcpyDeviceToHost));

    int h_errorCount = 0;
    for (int peer = 0; peer < worldSize; peer++) {
      for (size_t i = 0; i < numIntsPerRank; i++) {
        int32_t expected = peer * 10000 + globalRank * 1000 + iter * 100 +
            static_cast<int32_t>(i);
        int32_t actual = h_recv[peer * numIntsPerRank + i];
        if (expected != actual) {
          h_errorCount++;
          if (h_errorCount <= 5) {
            XLOGF(
                ERR,
                "Rank {}: Iter {} error at peer {} pos {}: expected {}, got {}",
                globalRank,
                iter,
                peer,
                i,
                expected,
                actual);
          }
        }
      }
    }

    EXPECT_EQ(h_errorCount, 0)
        << "Rank " << globalRank << " iter " << iter << " found "
        << h_errorCount << " verification errors (chunked multi-iter)";
  }

  bootstrap->barrierAll();
}

// =============================================================================
// High block count sweep with chunked buffer
// Targeted reproduction of the Ll128BlockThreadSweep benchmark failure.
// Sweeps block counts from 8 to 256 with a fixed message size and chunked
// buffer, verifying data correctness at each configuration.
// =============================================================================

TEST_F(AllToAllvLl128TestFixture, ChunkedBlockCountSweep) {
  CUDACHECK_TEST(cudaSetDevice(localRank));

  const size_t numIntsPerRank = 16384; // 64KB per peer
  const int blockSize = 512;
  const size_t ll128BufferNumPackets = 8; // Heavy chunking

  const size_t totalInts = numIntsPerRank * worldSize;
  const size_t bufferSize = totalInts * sizeof(int32_t);
  const size_t perPeerBytes = numIntsPerRank * sizeof(int32_t);

  const std::vector<int> blockCounts = {8, 16, 32, 64, 128, 256};

  for (int numBlocks : blockCounts) {
    XLOGF(
        INFO,
        "Rank {}: ChunkedBlockCountSweep numBlocks={}",
        globalRank,
        numBlocks);

    MultiPeerNvlTransportConfig config{
        .dataBufferSize = std::max(size_t(2048), bufferSize),
        .chunkSize = 512,
        .pipelineDepth = 4,
        .ll128BufferSize = ll128BufferNumPackets * kLl128PacketSize,
    };

    std::unique_ptr<MultiPeerNvlTransport> transport;
    try {
      transport = std::make_unique<MultiPeerNvlTransport>(
          globalRank, worldSize, bootstrap, config);
      transport->exchange();
    } catch (const std::runtime_error& e) {
      XLOGF(ERR, "Rank {}: transport init failed: {}", globalRank, e.what());
      std::abort();
    }

    auto transports_span = transport->getDeviceTransports();

    DeviceBuffer sendBuffer(bufferSize);
    DeviceBuffer recvBuffer(bufferSize);

    // Initialize send buffer
    std::vector<int32_t> h_send(totalInts);
    for (int peer = 0; peer < worldSize; peer++) {
      for (size_t i = 0; i < numIntsPerRank; i++) {
        h_send[peer * numIntsPerRank + i] =
            globalRank * 1000 + peer * 100 + static_cast<int32_t>(i);
      }
    }
    CUDACHECK_TEST(cudaMemcpy(
        sendBuffer.get(), h_send.data(), bufferSize, cudaMemcpyHostToDevice));
    test::fillBuffer(reinterpret_cast<int*>(recvBuffer.get()), -1, totalInts);

    // Setup ChunkInfo
    std::vector<ChunkInfo> h_send_chunks, h_recv_chunks;
    for (int rank = 0; rank < worldSize; rank++) {
      size_t offset = rank * perPeerBytes;
      h_send_chunks.emplace_back(offset, perPeerBytes);
      h_recv_chunks.emplace_back(offset, perPeerBytes);
    }

    DeviceBuffer d_send_chunks(sizeof(ChunkInfo) * worldSize);
    DeviceBuffer d_recv_chunks(sizeof(ChunkInfo) * worldSize);
    CUDACHECK_TEST(cudaMemcpy(
        d_send_chunks.get(),
        h_send_chunks.data(),
        sizeof(ChunkInfo) * worldSize,
        cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemcpy(
        d_recv_chunks.get(),
        h_recv_chunks.data(),
        sizeof(ChunkInfo) * worldSize,
        cudaMemcpyHostToDevice));

    DeviceSpan<ChunkInfo> send_chunk_infos(
        static_cast<ChunkInfo*>(d_send_chunks.get()), worldSize);
    DeviceSpan<ChunkInfo> recv_chunk_infos(
        static_cast<ChunkInfo*>(d_recv_chunks.get()), worldSize);

    bootstrap->barrierAll();

    test::test_all_to_allv_ll128(
        recvBuffer.get(),
        sendBuffer.get(),
        globalRank,
        worldSize,
        transports_span,
        send_chunk_infos,
        recv_chunk_infos,
        numBlocks,
        blockSize);

    CUDACHECK_TEST(cudaDeviceSynchronize());

    // Verify
    std::vector<int32_t> h_recv(totalInts);
    CUDACHECK_TEST(cudaMemcpy(
        h_recv.data(), recvBuffer.get(), bufferSize, cudaMemcpyDeviceToHost));

    int h_errorCount = 0;
    for (int peer = 0; peer < worldSize; peer++) {
      for (size_t i = 0; i < numIntsPerRank; i++) {
        int32_t expected =
            peer * 1000 + globalRank * 100 + static_cast<int32_t>(i);
        int32_t actual = h_recv[peer * numIntsPerRank + i];
        if (expected != actual) {
          h_errorCount++;
          if (h_errorCount <= 5) {
            XLOGF(
                ERR,
                "Rank {}: numBlocks={} error at peer {} pos {}: expected {}, got {}",
                globalRank,
                numBlocks,
                peer,
                i,
                expected,
                actual);
          }
        }
      }
    }

    EXPECT_EQ(h_errorCount, 0)
        << "Rank " << globalRank << " numBlocks=" << numBlocks << " found "
        << h_errorCount << " verification errors";

    bootstrap->barrierAll();
  }
}

// =============================================================================
// Large message AllToAllV tests — 512KB and 1MB per peer (Gap 4)
// =============================================================================

INSTANTIATE_TEST_SUITE_P(
    LargeMessageConfigs,
    AllToAllvLl128EqualSizeTest,
    ::testing::Values(
        // 512KB per peer (131072 ints)
        AllToAllvLl128EqualParams{
            .numBlocks = 128,
            .blockSize = 512,
            .numIntsPerRank = 131072,
            .testName = "128b_512t_512KB"},
        // 1MB per peer (262144 ints)
        AllToAllvLl128EqualParams{
            .numBlocks = 256,
            .blockSize = 512,
            .numIntsPerRank = 262144,
            .testName = "256b_512t_1MB"}),
    [](const ::testing::TestParamInfo<AllToAllvLl128EqualParams>& info) {
      return info.param.testName;
    });

// =============================================================================
// Pipelined multi-call with varying sizes per iteration (Gap 8)
// =============================================================================

TEST_F(AllToAllvLl128TestFixture, PipelinedVaryingSizes) {
  CUDACHECK_TEST(cudaSetDevice(localRank));

  const std::vector<size_t> intsPerRankPerIter = {
      1024, // 4KB
      4096, // 16KB
      256, // 1KB
      16384, // 64KB
      1024, // 4KB
  };
  const int numBlocks = 18;
  const int blockSize = 512;

  // Use max size for transport config
  size_t maxIntsPerRank =
      *std::max_element(intsPerRankPerIter.begin(), intsPerRankPerIter.end());
  size_t maxPerPeerBytes = maxIntsPerRank * sizeof(int32_t);
  size_t maxTotalInts = maxIntsPerRank * worldSize;
  size_t maxBufferSize = maxTotalInts * sizeof(int32_t);

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = std::max(size_t(2048), maxBufferSize),
      .chunkSize = 512,
      .pipelineDepth = 4,
      .ll128BufferSize = ll128_buffer_size(maxPerPeerBytes),
  };

  std::unique_ptr<MultiPeerNvlTransport> transport;
  try {
    transport = std::make_unique<MultiPeerNvlTransport>(
        globalRank, worldSize, bootstrap, config);
    transport->exchange();
  } catch (const std::runtime_error& e) {
    XLOGF(ERR, "Rank {}: transport init failed: {}", globalRank, e.what());
    std::abort();
  }

  DeviceSpan<Transport> transports_span(
      transport->getTransportsArray(), worldSize);

  DeviceBuffer sendBuffer(maxBufferSize);
  DeviceBuffer recvBuffer(maxBufferSize);

  for (size_t iter = 0; iter < intsPerRankPerIter.size(); iter++) {
    size_t numIntsPerRank = intsPerRankPerIter[iter];
    size_t totalInts = numIntsPerRank * worldSize;
    size_t bufferSize = totalInts * sizeof(int32_t);
    size_t perPeerBytes = numIntsPerRank * sizeof(int32_t);

    // Fill send buffer
    std::vector<int32_t> h_send(totalInts);
    for (int peer = 0; peer < worldSize; peer++) {
      for (size_t i = 0; i < numIntsPerRank; i++) {
        h_send[peer * numIntsPerRank + i] = globalRank * 10000 + peer * 1000 +
            static_cast<int32_t>(iter) * 100 + static_cast<int32_t>(i);
      }
    }
    CUDACHECK_TEST(cudaMemcpy(
        sendBuffer.get(), h_send.data(), bufferSize, cudaMemcpyHostToDevice));

    test::fillBuffer(reinterpret_cast<int*>(recvBuffer.get()), -1, totalInts);

    // Setup ChunkInfo for this iteration's size
    std::vector<ChunkInfo> h_send_chunk_infos;
    std::vector<ChunkInfo> h_recv_chunk_infos;
    for (int rank = 0; rank < worldSize; rank++) {
      size_t offset = rank * perPeerBytes;
      h_send_chunk_infos.emplace_back(offset, perPeerBytes);
      h_recv_chunk_infos.emplace_back(offset, perPeerBytes);
    }

    DeviceBuffer d_send_chunk_infos(sizeof(ChunkInfo) * worldSize);
    DeviceBuffer d_recv_chunk_infos(sizeof(ChunkInfo) * worldSize);
    CUDACHECK_TEST(cudaMemcpy(
        d_send_chunk_infos.get(),
        h_send_chunk_infos.data(),
        sizeof(ChunkInfo) * worldSize,
        cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemcpy(
        d_recv_chunk_infos.get(),
        h_recv_chunk_infos.data(),
        sizeof(ChunkInfo) * worldSize,
        cudaMemcpyHostToDevice));

    DeviceSpan<ChunkInfo> send_chunk_infos(
        static_cast<ChunkInfo*>(d_send_chunk_infos.get()), worldSize);
    DeviceSpan<ChunkInfo> recv_chunk_infos(
        static_cast<ChunkInfo*>(d_recv_chunk_infos.get()), worldSize);

    bootstrap->barrierAll();

    test::test_all_to_allv_ll128(
        recvBuffer.get(),
        sendBuffer.get(),
        globalRank,
        worldSize,
        transports_span,
        send_chunk_infos,
        recv_chunk_infos,
        numBlocks,
        blockSize);

    CUDACHECK_TEST(cudaDeviceSynchronize());

    // Verify
    std::vector<int32_t> h_recv(totalInts);
    CUDACHECK_TEST(cudaMemcpy(
        h_recv.data(), recvBuffer.get(), bufferSize, cudaMemcpyDeviceToHost));

    int h_errorCount = 0;
    for (int peer = 0; peer < worldSize; peer++) {
      for (size_t i = 0; i < numIntsPerRank; i++) {
        int32_t expected = peer * 10000 + globalRank * 1000 +
            static_cast<int32_t>(iter) * 100 + static_cast<int32_t>(i);
        int32_t actual = h_recv[peer * numIntsPerRank + i];
        if (expected != actual) {
          h_errorCount++;
          if (h_errorCount <= 5) {
            XLOGF(
                ERR,
                "Rank {}: Iter {} ({}KB) error at peer {} pos {}: expected {}, got {}",
                globalRank,
                iter,
                numIntsPerRank * 4 / 1024,
                peer,
                i,
                expected,
                actual);
          }
        }
      }
    }

    EXPECT_EQ(h_errorCount, 0)
        << "Rank " << globalRank << " iter " << iter << " ("
        << numIntsPerRank * 4 / 1024 << "KB) found " << h_errorCount
        << " verification errors";
  }

  bootstrap->barrierAll();
}

TEST_F(AllToAllvLl128TestFixture, PipelinedVaryingSizes_Chunked) {
  CUDACHECK_TEST(cudaSetDevice(localRank));

  const std::vector<size_t> intsPerRankPerIter = {
      1024, // 4KB
      4096, // 16KB
      256, // 1KB
      16384, // 64KB
      1024, // 4KB
  };
  const int numBlocks = 18;
  const int blockSize = 512;
  const size_t ll128BufferNumPackets = 8;

  size_t maxIntsPerRank =
      *std::max_element(intsPerRankPerIter.begin(), intsPerRankPerIter.end());
  size_t maxTotalInts = maxIntsPerRank * worldSize;
  size_t maxBufferSize = maxTotalInts * sizeof(int32_t);

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = std::max(size_t(2048), maxBufferSize),
      .chunkSize = 512,
      .pipelineDepth = 4,
      .ll128BufferSize = ll128BufferNumPackets * kLl128PacketSize,
  };

  std::unique_ptr<MultiPeerNvlTransport> transport;
  try {
    transport = std::make_unique<MultiPeerNvlTransport>(
        globalRank, worldSize, bootstrap, config);
    transport->exchange();
  } catch (const std::runtime_error& e) {
    XLOGF(ERR, "Rank {}: transport init failed: {}", globalRank, e.what());
    std::abort();
  }

  DeviceSpan<Transport> transports_span(
      transport->getTransportsArray(), worldSize);

  DeviceBuffer sendBuffer(maxBufferSize);
  DeviceBuffer recvBuffer(maxBufferSize);

  for (size_t iter = 0; iter < intsPerRankPerIter.size(); iter++) {
    size_t numIntsPerRank = intsPerRankPerIter[iter];
    size_t totalInts = numIntsPerRank * worldSize;
    size_t bufferSize = totalInts * sizeof(int32_t);
    size_t perPeerBytes = numIntsPerRank * sizeof(int32_t);

    std::vector<int32_t> h_send(totalInts);
    for (int peer = 0; peer < worldSize; peer++) {
      for (size_t i = 0; i < numIntsPerRank; i++) {
        h_send[peer * numIntsPerRank + i] = globalRank * 10000 + peer * 1000 +
            static_cast<int32_t>(iter) * 100 + static_cast<int32_t>(i);
      }
    }
    CUDACHECK_TEST(cudaMemcpy(
        sendBuffer.get(), h_send.data(), bufferSize, cudaMemcpyHostToDevice));

    test::fillBuffer(reinterpret_cast<int*>(recvBuffer.get()), -1, totalInts);

    std::vector<ChunkInfo> h_send_chunk_infos;
    std::vector<ChunkInfo> h_recv_chunk_infos;
    for (int rank = 0; rank < worldSize; rank++) {
      size_t offset = rank * perPeerBytes;
      h_send_chunk_infos.emplace_back(offset, perPeerBytes);
      h_recv_chunk_infos.emplace_back(offset, perPeerBytes);
    }

    DeviceBuffer d_send_chunk_infos(sizeof(ChunkInfo) * worldSize);
    DeviceBuffer d_recv_chunk_infos(sizeof(ChunkInfo) * worldSize);
    CUDACHECK_TEST(cudaMemcpy(
        d_send_chunk_infos.get(),
        h_send_chunk_infos.data(),
        sizeof(ChunkInfo) * worldSize,
        cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemcpy(
        d_recv_chunk_infos.get(),
        h_recv_chunk_infos.data(),
        sizeof(ChunkInfo) * worldSize,
        cudaMemcpyHostToDevice));

    DeviceSpan<ChunkInfo> send_chunk_infos(
        static_cast<ChunkInfo*>(d_send_chunk_infos.get()), worldSize);
    DeviceSpan<ChunkInfo> recv_chunk_infos(
        static_cast<ChunkInfo*>(d_recv_chunk_infos.get()), worldSize);

    bootstrap->barrierAll();

    test::test_all_to_allv_ll128(
        recvBuffer.get(),
        sendBuffer.get(),
        globalRank,
        worldSize,
        transports_span,
        send_chunk_infos,
        recv_chunk_infos,
        numBlocks,
        blockSize);

    CUDACHECK_TEST(cudaDeviceSynchronize());

    std::vector<int32_t> h_recv(totalInts);
    CUDACHECK_TEST(cudaMemcpy(
        h_recv.data(), recvBuffer.get(), bufferSize, cudaMemcpyDeviceToHost));

    int h_errorCount = 0;
    for (int peer = 0; peer < worldSize; peer++) {
      for (size_t i = 0; i < numIntsPerRank; i++) {
        int32_t expected = peer * 10000 + globalRank * 1000 +
            static_cast<int32_t>(iter) * 100 + static_cast<int32_t>(i);
        int32_t actual = h_recv[peer * numIntsPerRank + i];
        if (expected != actual) {
          h_errorCount++;
          if (h_errorCount <= 5) {
            XLOGF(
                ERR,
                "Rank {}: Iter {} ({}KB chunked) error at peer {} pos {}: expected {}, got {}",
                globalRank,
                iter,
                numIntsPerRank * 4 / 1024,
                peer,
                i,
                expected,
                actual);
          }
        }
      }
    }

    EXPECT_EQ(h_errorCount, 0)
        << "Rank " << globalRank << " iter " << iter << " ("
        << numIntsPerRank * 4 / 1024 << "KB chunked) found " << h_errorCount
        << " verification errors";
  }

  bootstrap->barrierAll();
}

// =============================================================================
// Auto dispatch tests — all_to_allv_auto() (Gap 3)
// =============================================================================

TEST_F(AllToAllvLl128TestFixture, AutoDispatch_1KB_UsesLl128) {
  CUDACHECK_TEST(cudaSetDevice(localRank));

  const size_t numIntsPerRank = 256; // 1KB per peer
  const size_t totalInts = numIntsPerRank * worldSize;
  const size_t bufferSize = totalInts * sizeof(int32_t);
  const size_t perPeerBytes = numIntsPerRank * sizeof(int32_t);

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = std::max(size_t(2048), bufferSize),
      .chunkSize = 512,
      .pipelineDepth = 4,
      .ll128BufferSize = ll128_buffer_size(perPeerBytes),
  };

  std::unique_ptr<MultiPeerNvlTransport> transport;
  try {
    transport = std::make_unique<MultiPeerNvlTransport>(
        globalRank, worldSize, bootstrap, config);
    transport->exchange();
  } catch (const std::runtime_error& e) {
    XLOGF(ERR, "Rank {}: transport init failed: {}", globalRank, e.what());
    std::abort();
  }

  DeviceSpan<Transport> transports_span(
      transport->getTransportsArray(), worldSize);

  DeviceBuffer sendBuffer(bufferSize);
  DeviceBuffer recvBuffer(bufferSize);

  test::fillBuffer(reinterpret_cast<int*>(recvBuffer.get()), -1, totalInts);

  std::vector<int32_t> h_send(totalInts);
  for (int peer = 0; peer < worldSize; peer++) {
    for (size_t i = 0; i < numIntsPerRank; i++) {
      h_send[peer * numIntsPerRank + i] =
          globalRank * 1000 + peer * 100 + static_cast<int32_t>(i);
    }
  }
  CUDACHECK_TEST(cudaMemcpy(
      sendBuffer.get(), h_send.data(), bufferSize, cudaMemcpyHostToDevice));

  std::vector<ChunkInfo> h_send_chunk_infos;
  std::vector<ChunkInfo> h_recv_chunk_infos;
  for (int rank = 0; rank < worldSize; rank++) {
    size_t offset = rank * perPeerBytes;
    h_send_chunk_infos.emplace_back(offset, perPeerBytes);
    h_recv_chunk_infos.emplace_back(offset, perPeerBytes);
  }

  DeviceBuffer d_send_chunk_infos(sizeof(ChunkInfo) * worldSize);
  DeviceBuffer d_recv_chunk_infos(sizeof(ChunkInfo) * worldSize);
  CUDACHECK_TEST(cudaMemcpy(
      d_send_chunk_infos.get(),
      h_send_chunk_infos.data(),
      sizeof(ChunkInfo) * worldSize,
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemcpy(
      d_recv_chunk_infos.get(),
      h_recv_chunk_infos.data(),
      sizeof(ChunkInfo) * worldSize,
      cudaMemcpyHostToDevice));

  DeviceSpan<ChunkInfo> send_chunk_infos(
      static_cast<ChunkInfo*>(d_send_chunk_infos.get()), worldSize);
  DeviceSpan<ChunkInfo> recv_chunk_infos(
      static_cast<ChunkInfo*>(d_recv_chunk_infos.get()), worldSize);

  bootstrap->barrierAll();

  all_to_allv_auto(
      recvBuffer.get(),
      sendBuffer.get(),
      globalRank,
      worldSize,
      transports_span,
      send_chunk_infos,
      recv_chunk_infos,
      perPeerBytes);

  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<int32_t> h_recv(totalInts);
  CUDACHECK_TEST(cudaMemcpy(
      h_recv.data(), recvBuffer.get(), bufferSize, cudaMemcpyDeviceToHost));

  int h_errorCount = 0;
  for (int peer = 0; peer < worldSize; peer++) {
    for (size_t i = 0; i < numIntsPerRank; i++) {
      int32_t expected =
          peer * 1000 + globalRank * 100 + static_cast<int32_t>(i);
      int32_t actual = h_recv[peer * numIntsPerRank + i];
      if (expected != actual) {
        h_errorCount++;
        if (h_errorCount <= 10) {
          XLOGF(
              ERR,
              "Rank {}: AutoDispatch 1KB error at peer {} pos {}: expected {}, got {}",
              globalRank,
              peer,
              i,
              expected,
              actual);
        }
      }
    }
  }

  EXPECT_EQ(h_errorCount, 0) << "Rank " << globalRank << " found "
                             << h_errorCount << " verification errors";
  bootstrap->barrierAll();
}

TEST_F(AllToAllvLl128TestFixture, AutoDispatch_256KB_UsesLl128) {
  CUDACHECK_TEST(cudaSetDevice(localRank));

  const size_t numIntsPerRank = 65536; // 256KB per peer (boundary)
  const size_t totalInts = numIntsPerRank * worldSize;
  const size_t bufferSize = totalInts * sizeof(int32_t);
  const size_t perPeerBytes = numIntsPerRank * sizeof(int32_t);

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = std::max(size_t(2048), bufferSize),
      .chunkSize = 512,
      .pipelineDepth = 4,
      .ll128BufferSize = ll128_buffer_size(perPeerBytes),
  };

  std::unique_ptr<MultiPeerNvlTransport> transport;
  try {
    transport = std::make_unique<MultiPeerNvlTransport>(
        globalRank, worldSize, bootstrap, config);
    transport->exchange();
  } catch (const std::runtime_error& e) {
    XLOGF(ERR, "Rank {}: transport init failed: {}", globalRank, e.what());
    std::abort();
  }

  DeviceSpan<Transport> transports_span(
      transport->getTransportsArray(), worldSize);

  DeviceBuffer sendBuffer(bufferSize);
  DeviceBuffer recvBuffer(bufferSize);

  test::fillBuffer(reinterpret_cast<int*>(recvBuffer.get()), -1, totalInts);

  std::vector<int32_t> h_send(totalInts);
  for (int peer = 0; peer < worldSize; peer++) {
    for (size_t i = 0; i < numIntsPerRank; i++) {
      h_send[peer * numIntsPerRank + i] =
          globalRank * 1000 + peer * 100 + static_cast<int32_t>(i);
    }
  }
  CUDACHECK_TEST(cudaMemcpy(
      sendBuffer.get(), h_send.data(), bufferSize, cudaMemcpyHostToDevice));

  std::vector<ChunkInfo> h_send_chunk_infos;
  std::vector<ChunkInfo> h_recv_chunk_infos;
  for (int rank = 0; rank < worldSize; rank++) {
    size_t offset = rank * perPeerBytes;
    h_send_chunk_infos.emplace_back(offset, perPeerBytes);
    h_recv_chunk_infos.emplace_back(offset, perPeerBytes);
  }

  DeviceBuffer d_send_chunk_infos(sizeof(ChunkInfo) * worldSize);
  DeviceBuffer d_recv_chunk_infos(sizeof(ChunkInfo) * worldSize);
  CUDACHECK_TEST(cudaMemcpy(
      d_send_chunk_infos.get(),
      h_send_chunk_infos.data(),
      sizeof(ChunkInfo) * worldSize,
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemcpy(
      d_recv_chunk_infos.get(),
      h_recv_chunk_infos.data(),
      sizeof(ChunkInfo) * worldSize,
      cudaMemcpyHostToDevice));

  DeviceSpan<ChunkInfo> send_chunk_infos(
      static_cast<ChunkInfo*>(d_send_chunk_infos.get()), worldSize);
  DeviceSpan<ChunkInfo> recv_chunk_infos(
      static_cast<ChunkInfo*>(d_recv_chunk_infos.get()), worldSize);

  bootstrap->barrierAll();

  all_to_allv_auto(
      recvBuffer.get(),
      sendBuffer.get(),
      globalRank,
      worldSize,
      transports_span,
      send_chunk_infos,
      recv_chunk_infos,
      perPeerBytes);

  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<int32_t> h_recv(totalInts);
  CUDACHECK_TEST(cudaMemcpy(
      h_recv.data(), recvBuffer.get(), bufferSize, cudaMemcpyDeviceToHost));

  int h_errorCount = 0;
  for (int peer = 0; peer < worldSize; peer++) {
    for (size_t i = 0; i < numIntsPerRank; i++) {
      int32_t expected =
          peer * 1000 + globalRank * 100 + static_cast<int32_t>(i);
      int32_t actual = h_recv[peer * numIntsPerRank + i];
      if (expected != actual) {
        h_errorCount++;
        if (h_errorCount <= 10) {
          XLOGF(
              ERR,
              "Rank {}: AutoDispatch 256KB error at peer {} pos {}: expected {}, got {}",
              globalRank,
              peer,
              i,
              expected,
              actual);
        }
      }
    }
  }

  EXPECT_EQ(h_errorCount, 0) << "Rank " << globalRank << " found "
                             << h_errorCount << " verification errors";
  bootstrap->barrierAll();
}

TEST_F(AllToAllvLl128TestFixture, AutoDispatch_512KB_UsesSimple) {
  CUDACHECK_TEST(cudaSetDevice(localRank));

  const size_t numIntsPerRank = 131072; // 512KB per peer (above threshold)
  const size_t totalInts = numIntsPerRank * worldSize;
  const size_t bufferSize = totalInts * sizeof(int32_t);
  const size_t perPeerBytes = numIntsPerRank * sizeof(int32_t);

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = std::max(size_t(2048), bufferSize),
      .chunkSize = 512,
      .pipelineDepth = 4,
      .ll128BufferSize = ll128_buffer_size(256 * 1024),
  };

  std::unique_ptr<MultiPeerNvlTransport> transport;
  try {
    transport = std::make_unique<MultiPeerNvlTransport>(
        globalRank, worldSize, bootstrap, config);
    transport->exchange();
  } catch (const std::runtime_error& e) {
    XLOGF(ERR, "Rank {}: transport init failed: {}", globalRank, e.what());
    std::abort();
  }

  DeviceSpan<Transport> transports_span(
      transport->getTransportsArray(), worldSize);

  DeviceBuffer sendBuffer(bufferSize);
  DeviceBuffer recvBuffer(bufferSize);

  test::fillBuffer(reinterpret_cast<int*>(recvBuffer.get()), -1, totalInts);

  std::vector<int32_t> h_send(totalInts);
  for (int peer = 0; peer < worldSize; peer++) {
    for (size_t i = 0; i < numIntsPerRank; i++) {
      h_send[peer * numIntsPerRank + i] =
          globalRank * 1000 + peer * 100 + static_cast<int32_t>(i);
    }
  }
  CUDACHECK_TEST(cudaMemcpy(
      sendBuffer.get(), h_send.data(), bufferSize, cudaMemcpyHostToDevice));

  std::vector<ChunkInfo> h_send_chunk_infos;
  std::vector<ChunkInfo> h_recv_chunk_infos;
  for (int rank = 0; rank < worldSize; rank++) {
    size_t offset = rank * perPeerBytes;
    h_send_chunk_infos.emplace_back(offset, perPeerBytes);
    h_recv_chunk_infos.emplace_back(offset, perPeerBytes);
  }

  DeviceBuffer d_send_chunk_infos(sizeof(ChunkInfo) * worldSize);
  DeviceBuffer d_recv_chunk_infos(sizeof(ChunkInfo) * worldSize);
  CUDACHECK_TEST(cudaMemcpy(
      d_send_chunk_infos.get(),
      h_send_chunk_infos.data(),
      sizeof(ChunkInfo) * worldSize,
      cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemcpy(
      d_recv_chunk_infos.get(),
      h_recv_chunk_infos.data(),
      sizeof(ChunkInfo) * worldSize,
      cudaMemcpyHostToDevice));

  DeviceSpan<ChunkInfo> send_chunk_infos(
      static_cast<ChunkInfo*>(d_send_chunk_infos.get()), worldSize);
  DeviceSpan<ChunkInfo> recv_chunk_infos(
      static_cast<ChunkInfo*>(d_recv_chunk_infos.get()), worldSize);

  bootstrap->barrierAll();

  all_to_allv_auto(
      recvBuffer.get(),
      sendBuffer.get(),
      globalRank,
      worldSize,
      transports_span,
      send_chunk_infos,
      recv_chunk_infos,
      perPeerBytes);

  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<int32_t> h_recv(totalInts);
  CUDACHECK_TEST(cudaMemcpy(
      h_recv.data(), recvBuffer.get(), bufferSize, cudaMemcpyDeviceToHost));

  int h_errorCount = 0;
  for (int peer = 0; peer < worldSize; peer++) {
    for (size_t i = 0; i < numIntsPerRank; i++) {
      int32_t expected =
          peer * 1000 + globalRank * 100 + static_cast<int32_t>(i);
      int32_t actual = h_recv[peer * numIntsPerRank + i];
      if (expected != actual) {
        h_errorCount++;
        if (h_errorCount <= 10) {
          XLOGF(
              ERR,
              "Rank {}: AutoDispatch 512KB error at peer {} pos {}: expected {}, got {}",
              globalRank,
              peer,
              i,
              expected,
              actual);
        }
      }
    }
  }

  EXPECT_EQ(h_errorCount, 0) << "Rank " << globalRank << " found "
                             << h_errorCount << " verification errors";
  bootstrap->barrierAll();
}

} // namespace comms::pipes

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  if (!meta::comms::isTcpEnvironment()) {
    ::testing::AddGlobalTestEnvironment(
        new meta::comms::BenchmarkEnvironment());
  }
  return RUN_ALL_TESTS();
}
