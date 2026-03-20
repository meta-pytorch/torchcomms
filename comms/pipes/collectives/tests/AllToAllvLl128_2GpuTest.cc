// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include "comms/pipes/MultiPeerNvlTransport.h"
#include "comms/pipes/collectives/AllToAllvLl128.cuh"
#include "comms/pipes/ll128/Ll128Packet.cuh"
#include "comms/pipes/tests/AllToAllvLl128Test.cuh"
#include "comms/pipes/tests/Utils.cuh"
#include "comms/testinfra/BenchmarkTestFixture.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

using namespace meta::comms;

namespace comms::pipes {

// =============================================================================
// 2-GPU AllToAllV fixture — exercises minimum rank count (Gap 5)
// =============================================================================

class AllToAllvLl128_2GpuTestFixture : public BenchmarkTestFixture {
 protected:
  void SetUp() override {
    BenchmarkTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }

  void TearDown() override {
    BenchmarkTestFixture::TearDown();
  }
};

TEST_F(AllToAllvLl128_2GpuTestFixture, EqualSize_2GPU_4KB) {
  const size_t numIntsPerRank = 1024; // 4KB per peer
  const int numBlocks = 18;
  const int blockSize = 512;

  XLOGF(
      DBG1,
      "Rank {}: Running EqualSize_2GPU_4KB with worldSize={}",
      globalRank,
      worldSize);

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

TEST_F(AllToAllvLl128_2GpuTestFixture, EqualSize_2GPU_64KB) {
  const size_t numIntsPerRank = 16384; // 64KB per peer
  const int numBlocks = 18;
  const int blockSize = 512;

  XLOGF(
      DBG1,
      "Rank {}: Running EqualSize_2GPU_64KB with worldSize={}",
      globalRank,
      worldSize);

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
