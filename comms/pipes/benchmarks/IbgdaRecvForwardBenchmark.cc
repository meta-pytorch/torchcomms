// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "comms/pipes/MultipeerIbgdaTransport.h"
#include "comms/pipes/benchmarks/IbgdaRecvForward.h"
#include "comms/testinfra/BenchmarkTestFixture.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::BenchmarkEnvironment;
using meta::comms::BenchmarkTestFixture;
using meta::comms::DeviceBuffer;

namespace comms::pipes::benchmark {

using SendRecvConfig = MultipeerIbgdaTransportConfig::SendRecvConfig;

class IbgdaRecvForwardBenchmarkTest : public BenchmarkTestFixture {
 protected:
  void SetUp() override {
    BenchmarkTestFixture::SetUp();
    cudaSetDevice(localRank);
    cudaStreamCreate(&stream_);
  }

  void TearDown() override {
    cudaStreamDestroy(stream_);
    BenchmarkTestFixture::TearDown();
  }

  cudaStream_t stream_{};
};

TEST_F(IbgdaRecvForwardBenchmarkTest, Correctness) {
  if (worldSize != 3) {
    XLOGF(INFO, "Skipping: requires exactly 3 ranks, got {}", worldSize);
    return;
  }

  constexpr std::size_t kDataBytes = 4 * 1024 * 1024;
  constexpr int kNumBlocks = 4;
  constexpr std::size_t kSlotSize = kDataBytes;
  constexpr int kPipelineDepth = 2;

  MultipeerIbgdaTransportConfig transportConfig{
      .cudaDevice = localRank,
      .dataBufferSize = kSlotSize,
      .sendRecv =
          SendRecvConfig{
              .maxGroups = kNumBlocks,
              .pipelineDepth = kPipelineDepth,
          },
  };

  MultipeerIbgdaTransport transport(
      globalRank, worldSize, bootstrap, transportConfig);
  transport.exchange();

  DeviceBuffer sendBuf(kDataBytes);
  DeviceBuffer recvBuf(kDataBytes);

  const uint8_t fillPattern = 0xCA;
  cudaMemset(sendBuf.get(), fillPattern, kDataBytes);
  cudaMemset(recvBuf.get(), 0, kDataBytes);
  cudaDeviceSynchronize();

  const int prevRank = (globalRank + worldSize - 1) % worldSize;
  const int nextRank = (globalRank + 1) % worldSize;

  P2pIbgdaTransportDevice* prevTransport =
      (globalRank != 0) ? transport.getP2pTransportDevice(prevRank) : nullptr;
  P2pIbgdaTransportDevice* nextTransport =
      (globalRank != 2) ? transport.getP2pTransportDevice(nextRank) : nullptr;

  bootstrap->barrierAll();

  launch_ibgda_recv_forward_chain(
      prevTransport,
      nextTransport,
      static_cast<char*>(sendBuf.get()),
      static_cast<char*>(recvBuf.get()),
      kDataBytes,
      kNumBlocks,
      globalRank,
      stream_);

  cudaError_t err = cudaStreamSynchronize(stream_);
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed: " << cudaGetErrorString(err);

  bootstrap->barrierAll();

  if (globalRank != 0) {
    std::vector<uint8_t> hostBuf(kDataBytes);
    cudaMemcpy(
        hostBuf.data(), recvBuf.get(), kDataBytes, cudaMemcpyDeviceToHost);

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
}

TEST_F(IbgdaRecvForwardBenchmarkTest, Bandwidth) {
  if (worldSize != 3) {
    XLOGF(INFO, "Skipping: requires exactly 3 ranks, got {}", worldSize);
    return;
  }

  constexpr int kNumBlocks = 2;
  constexpr std::size_t kSlotSize = 8 * 1024 * 1024;
  constexpr int kPipelineDepth = 2;
  constexpr int kWarmupIters = 5;
  constexpr int kBenchIters = 20;

  std::vector<std::size_t> messageSizes;
  for (std::size_t sz = 1ULL << 20; sz <= 1ULL << 30; sz <<= 1) {
    messageSizes.push_back(sz);
  }

  MultipeerIbgdaTransportConfig transportConfig{
      .cudaDevice = localRank,
      .dataBufferSize = kSlotSize,
      .sendRecv =
          SendRecvConfig{
              .maxGroups = kNumBlocks,
              .pipelineDepth = kPipelineDepth,
          },
  };

  MultipeerIbgdaTransport transport(
      globalRank, worldSize, bootstrap, transportConfig);
  transport.exchange();

  std::size_t maxBytes = messageSizes.back();
  DeviceBuffer sendBuf(maxBytes);
  DeviceBuffer recvBuf(maxBytes);
  cudaMemset(sendBuf.get(), 0xAA, maxBytes);
  cudaMemset(recvBuf.get(), 0, maxBytes);
  cudaDeviceSynchronize();

  const int prevRank = (globalRank + worldSize - 1) % worldSize;
  const int nextRank = (globalRank + 1) % worldSize;

  P2pIbgdaTransportDevice* prevTransport =
      (globalRank != 0) ? transport.getP2pTransportDevice(prevRank) : nullptr;
  P2pIbgdaTransportDevice* nextTransport =
      (globalRank != 2) ? transport.getP2pTransportDevice(nextRank) : nullptr;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  if (globalRank == 0) {
    XLOGF(INFO, "");
    XLOGF(
        INFO,
        "================================================================");
    XLOGF(
        INFO, "  recv_forward vs recv+send (3-rank chain: rank0→rank1→rank2)");
    XLOGF(
        INFO,
        "  numBlocks={}, slotSize={}MB, pipelineDepth={}",
        kNumBlocks,
        kSlotSize / (1024 * 1024),
        kPipelineDepth);
    XLOGF(
        INFO,
        "================================================================");
    XLOGF(
        INFO,
        "{:>10s}  {:>14s}  {:>14s}  {:>10s}",
        "MsgSize",
        "recv_fwd GB/s",
        "recv+snd GB/s",
        "Speedup");
    XLOGF(
        INFO, "--------------------------------------------------------------");
  }

  for (auto nBytes : messageSizes) {
    float recvFwdBw = 0;
    float recvSndBw = 0;

    // --- Benchmark recv_forward ---
    {
      bootstrap->barrierAll();
      for (int i = 0; i < kWarmupIters; ++i) {
        launch_ibgda_recv_forward_chain(
            prevTransport,
            nextTransport,
            static_cast<char*>(sendBuf.get()),
            static_cast<char*>(recvBuf.get()),
            nBytes,
            kNumBlocks,
            globalRank,
            stream_);
        cudaStreamSynchronize(stream_);
        bootstrap->barrierAll();
      }

      bootstrap->barrierAll();
      cudaEventRecord(start, stream_);
      for (int i = 0; i < kBenchIters; ++i) {
        launch_ibgda_recv_forward_chain(
            prevTransport,
            nextTransport,
            static_cast<char*>(sendBuf.get()),
            static_cast<char*>(recvBuf.get()),
            nBytes,
            kNumBlocks,
            globalRank,
            stream_);
      }
      cudaEventRecord(stop, stream_);
      cudaEventSynchronize(stop);
      bootstrap->barrierAll();

      float totalMs = 0;
      cudaEventElapsedTime(&totalMs, start, stop);
      recvFwdBw = (nBytes / 1e9f) / ((totalMs / kBenchIters) / 1000.0f);
    }

    // --- Benchmark recv+send ---
    {
      // Need fresh transport since step state has advanced
      MultipeerIbgdaTransport transport2(
          globalRank, worldSize, bootstrap, transportConfig);
      transport2.exchange();

      P2pIbgdaTransportDevice* prevT2 = (globalRank != 0)
          ? transport2.getP2pTransportDevice(prevRank)
          : nullptr;
      P2pIbgdaTransportDevice* nextT2 = (globalRank != 2)
          ? transport2.getP2pTransportDevice(nextRank)
          : nullptr;

      bootstrap->barrierAll();
      for (int i = 0; i < kWarmupIters; ++i) {
        launch_ibgda_recv_send_chain(
            prevT2,
            nextT2,
            static_cast<char*>(sendBuf.get()),
            static_cast<char*>(recvBuf.get()),
            nBytes,
            kNumBlocks,
            globalRank,
            stream_);
        cudaStreamSynchronize(stream_);
        bootstrap->barrierAll();
      }

      bootstrap->barrierAll();
      cudaEventRecord(start, stream_);
      for (int i = 0; i < kBenchIters; ++i) {
        launch_ibgda_recv_send_chain(
            prevT2,
            nextT2,
            static_cast<char*>(sendBuf.get()),
            static_cast<char*>(recvBuf.get()),
            nBytes,
            kNumBlocks,
            globalRank,
            stream_);
      }
      cudaEventRecord(stop, stream_);
      cudaEventSynchronize(stop);
      bootstrap->barrierAll();

      float totalMs = 0;
      cudaEventElapsedTime(&totalMs, start, stop);
      recvSndBw = (nBytes / 1e9f) / ((totalMs / kBenchIters) / 1000.0f);
    }

    if (globalRank == 0) {
      std::string sizeStr;
      if (nBytes >= 1ULL << 30) {
        sizeStr = fmt::format("{}GB", nBytes >> 30);
      } else {
        sizeStr = fmt::format("{}MB", nBytes >> 20);
      }
      float speedup = (recvSndBw > 0) ? (recvFwdBw / recvSndBw) : 0;
      XLOGF(
          INFO,
          "{:>10s}  {:>14.2f}  {:>14.2f}  {:>9.2f}x",
          sizeStr,
          recvFwdBw,
          recvSndBw,
          speedup);
    }
  }

  if (globalRank == 0) {
    XLOGF(
        INFO,
        "================================================================");
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

} // namespace comms::pipes::benchmark

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  if (!meta::comms::isTcpEnvironment()) {
    ::testing::AddGlobalTestEnvironment(new BenchmarkEnvironment());
  }
  return RUN_ALL_TESTS();
}
