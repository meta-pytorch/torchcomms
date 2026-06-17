// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "comms/prims/benchmarks/IbgdaSendRecv.h"
#include "comms/prims/transport/ibgda/MultipeerIbgdaTransport.h"
#include "comms/testinfra/BenchmarkTestFixture.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::BenchmarkEnvironment;
using meta::comms::BenchmarkTestFixture;
using meta::comms::DeviceBuffer;

#define ASSERT_CUDA_SUCCESS(cmd)                                    \
  do {                                                              \
    cudaError_t ret;                                                \
    ASSERT_EQ(cudaSuccess, ret = (cmd)) << cudaGetErrorString(ret); \
  } while (0)

namespace comms::prims::benchmark {

using SendRecvConfig = MultipeerIbgdaTransportConfig::SendRecvConfig;

namespace {

constexpr std::size_t alignProtocolBytes(std::size_t nbytes) {
  return (nbytes + 15ULL) & ~15ULL;
}

struct SendRecvBenchmarkGeometry {
  int activeBlocks;
  int maxGroups;
  const char* label;
};

constexpr std::array<SendRecvBenchmarkGeometry, 2> kSendRecvBenchmarkGeometries{
    {{1, 2, "activeBlocks=1/maxGroups=2"},
     {2, 2, "activeBlocks=2/maxGroups=2"}}};

std::string formatMessageSize(std::size_t nbytes) {
  if (nbytes >= (1ULL << 30)) {
    return fmt::format("{}GB", nbytes >> 30);
  }
  if (nbytes >= (1ULL << 20)) {
    return fmt::format("{}MB", nbytes >> 20);
  }
  if (nbytes >= (1ULL << 10)) {
    return fmt::format("{}KB", nbytes >> 10);
  }
  return fmt::format("{}B", nbytes);
}

std::vector<std::size_t> makePowerOfTwoMessageSizes(
    std::size_t firstBytes,
    std::size_t lastBytes) {
  std::vector<std::size_t> messageSizes;
  for (std::size_t sz = firstBytes; sz <= lastBytes; sz <<= 1) {
    messageSizes.push_back(sz);
  }
  return messageSizes;
}

std::vector<int64_t> snapshotStepState(
    P2pIbgdaTransportDevice* transport,
    int maxGroups,
    cudaStream_t stream) {
  DeviceBuffer stepBuf(2 * maxGroups * sizeof(int64_t));
  launch_ibgda_snapshot_step_state(
      transport, static_cast<int64_t*>(stepBuf.get()), 2 * maxGroups, stream);

  cudaError_t err = cudaStreamSynchronize(stream);
  EXPECT_EQ(err, cudaSuccess)
      << "Step-state snapshot failed: " << cudaGetErrorString(err);

  std::vector<int64_t> hostSteps(2 * maxGroups, -1);
  err = cudaMemcpy(
      hostSteps.data(),
      stepBuf.get(),
      hostSteps.size() * sizeof(int64_t),
      cudaMemcpyDeviceToHost);
  EXPECT_EQ(err, cudaSuccess)
      << "Step-state copy failed: " << cudaGetErrorString(err);
  return hostSteps;
}

void expectUniformBuffer(
    const DeviceBuffer& buf,
    std::size_t nbytes,
    uint8_t expected,
    int rank) {
  std::vector<uint8_t> hostBuf(nbytes);
  cudaError_t err =
      cudaMemcpy(hostBuf.data(), buf.get(), nbytes, cudaMemcpyDeviceToHost);
  ASSERT_EQ(err, cudaSuccess)
      << "cudaMemcpy failed: " << cudaGetErrorString(err);

  for (std::size_t i = 0; i < nbytes; ++i) {
    ASSERT_EQ(hostBuf[i], expected)
        << "Rank " << rank << ": data mismatch at byte " << i << ", expected "
        << static_cast<int>(expected) << ", got "
        << static_cast<int>(hostBuf[i]);
  }
}

void expectUniformBufferRange(
    const DeviceBuffer& buf,
    std::size_t offset,
    std::size_t nbytes,
    uint8_t expected,
    int rank,
    const char* label) {
  std::vector<uint8_t> hostBuf(nbytes);
  cudaError_t err = cudaMemcpy(
      hostBuf.data(),
      static_cast<const char*>(buf.get()) + offset,
      nbytes,
      cudaMemcpyDeviceToHost);
  ASSERT_EQ(err, cudaSuccess)
      << "cudaMemcpy failed: " << cudaGetErrorString(err);

  for (std::size_t i = 0; i < nbytes; ++i) {
    ASSERT_EQ(hostBuf[i], expected)
        << "Rank " << rank << ": " << label << " data mismatch at byte " << i
        << ", expected " << static_cast<int>(expected) << ", got "
        << static_cast<int>(hostBuf[i]);
  }
}

void expectStepState(
    const std::vector<int64_t>& steps,
    int maxGroups,
    int64_t expected) {
  ASSERT_EQ(steps.size(), 2 * maxGroups);
  for (int i = 0; i < 2 * maxGroups; ++i) {
    EXPECT_EQ(steps[i], expected)
        << "Unexpected stepState[" << i << "], expected " << expected;
  }
}

} // namespace

class IbgdaSendRecvTest : public BenchmarkTestFixture {
 protected:
  void SetUp() override {
    BenchmarkTestFixture::SetUp();
    ASSERT_CUDA_SUCCESS(cudaSetDevice(localRank));
    ASSERT_CUDA_SUCCESS(cudaStreamCreate(&stream_));
  }

  void TearDown() override {
    if (stream_ != nullptr) {
      ASSERT_CUDA_SUCCESS(cudaStreamDestroy(stream_));
    }
    BenchmarkTestFixture::TearDown();
  }

  cudaStream_t stream_{};

  void runBidirectionalBandwidth(
      std::size_t firstMessageBytes,
      const char* benchmarkLabel) {
    if (worldSize != 2) {
      XLOGF(INFO, "Skipping: requires exactly 2 ranks, got {}", worldSize);
      return;
    }

    constexpr std::size_t kSlotSize = 8 * 1024 * 1024; // 8MB
    constexpr int kPipelineDepth = 2;
    constexpr int kWarmupIters = 5;
    constexpr int kBenchIters = 20;

    std::vector<std::size_t> messageSizes =
        makePowerOfTwoMessageSizes(firstMessageBytes, 4ULL << 30);

    int peerRank = (globalRank == 0) ? 1 : 0;
    std::size_t maxBytes = messageSizes.back();

    cudaEvent_t start, stop;
    ASSERT_CUDA_SUCCESS(cudaEventCreate(&start));
    ASSERT_CUDA_SUCCESS(cudaEventCreate(&stop));

    for (const auto& geometry : kSendRecvBenchmarkGeometries) {
      MultipeerIbgdaTransportConfig transportConfig{
          .cudaDevice = localRank,
          .dataBufferSize = kSlotSize,
          .sendRecv =
              SendRecvConfig{
                  .maxGroups = geometry.maxGroups,
                  .pipelineDepth = kPipelineDepth,
              },
      };

      MultipeerIbgdaTransport transport(
          globalRank, worldSize, bootstrap, transportConfig);
      transport.exchange();

      DeviceBuffer sendBuf(maxBytes);
      DeviceBuffer recvBuf(maxBytes);
      ASSERT_CUDA_SUCCESS(cudaMemset(sendBuf.get(), 0xAA, maxBytes));
      ASSERT_CUDA_SUCCESS(cudaMemset(recvBuf.get(), 0, maxBytes));
      ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());
      auto* deviceTransport = transport.getP2pTransportDevice(peerRank);

      if (globalRank == 0) {
        XLOGF(INFO, "");
        XLOGF(
            INFO,
            "================================================================");
        XLOGF(
            INFO,
            "  IBGDA Tile SendRecv Bandwidth {} ({})",
            benchmarkLabel,
            geometry.label);
        XLOGF(
            INFO,
            "  activeBlocks={}, maxGroups={}, slotSize={}MB, pipelineDepth={}, range={}..4GB",
            geometry.activeBlocks,
            geometry.maxGroups,
            kSlotSize / (1024 * 1024),
            kPipelineDepth,
            formatMessageSize(firstMessageBytes));
        XLOGF(
            INFO,
            "================================================================");
        XLOGF(
            INFO,
            "{:>10s}  {:>10s}  {:>12s}  {:>12s}",
            "MsgSize",
            "Sections",
            "BaseMod",
            "BW (GB/s)");
        XLOGF(
            INFO,
            "------------------------------------------------------------");
      }

      for (auto nBytes : messageSizes) {
        const std::size_t baseMod = 0;

        launch_ibgda_reset_send_recv(
            deviceTransport, geometry.maxGroups, stream_);
        ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream_));
        bootstrap->barrierAll();

        // Warmup
        for (int i = 0; i < kWarmupIters; ++i) {
          launch_ibgda_send_recv(
              deviceTransport,
              static_cast<char*>(sendBuf.get()),
              static_cast<char*>(recvBuf.get()),
              nBytes,
              geometry.activeBlocks,
              stream_);
          ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream_));
          bootstrap->barrierAll();
        }
        launch_ibgda_drain_send_recv(
            deviceTransport,
            geometry.activeBlocks,
            nBytes,
            kWarmupIters,
            stream_);
        ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream_));
        launch_ibgda_reset_send_recv(
            deviceTransport, geometry.maxGroups, stream_);
        ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream_));

        bootstrap->barrierAll();

        // Benchmark
        ASSERT_CUDA_SUCCESS(cudaEventRecord(start, stream_));
        for (int i = 0; i < kBenchIters; ++i) {
          launch_ibgda_send_recv(
              deviceTransport,
              static_cast<char*>(sendBuf.get()),
              static_cast<char*>(recvBuf.get()),
              nBytes,
              geometry.activeBlocks,
              stream_);
        }
        launch_ibgda_drain_send_recv(
            deviceTransport,
            geometry.activeBlocks,
            nBytes,
            kBenchIters,
            stream_);
        ASSERT_CUDA_SUCCESS(cudaEventRecord(stop, stream_));
        ASSERT_CUDA_SUCCESS(cudaEventSynchronize(stop));

        bootstrap->barrierAll();

        float totalMs = 0;
        ASSERT_CUDA_SUCCESS(cudaEventElapsedTime(&totalMs, start, stop));
        float avgMs = totalMs / kBenchIters;

        float bwGBs = (2.0f * nBytes / 1e9f) / (avgMs / 1000.0f);
        std::size_t numSections = (nBytes + kSlotSize - 1) / kSlotSize;

        if (globalRank == 0) {
          std::string sizeStr = formatMessageSize(nBytes);
          XLOGF(
              INFO,
              "{:>10s}  {:>10d}  {:>12d}  {:>12.2f}",
              sizeStr,
              numSections,
              baseMod,
              bwGBs);
        }

        launch_ibgda_reset_send_recv(
            deviceTransport, geometry.maxGroups, stream_);
        ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream_));
        bootstrap->barrierAll();
      }

      if (globalRank == 0) {
        XLOGF(
            INFO,
            "================================================================");
      }

      bootstrap->barrierAll();
    }

    ASSERT_CUDA_SUCCESS(cudaEventDestroy(start));
    ASSERT_CUDA_SUCCESS(cudaEventDestroy(stop));
  }
};

TEST_F(IbgdaSendRecvTest, Correctness) {
  if (worldSize != 2) {
    XLOGF(INFO, "Skipping: requires exactly 2 ranks, got {}", worldSize);
    return;
  }

  constexpr std::size_t kDataBytes = 4 * 1024 * 1024; // 4MB
  constexpr int kNumBlocks = 4;
  constexpr std::size_t kSlotSize = kDataBytes; // 1 section = 1 slot
  constexpr int kPipelineDepth = 2;
  constexpr int kMaxBlocks = kNumBlocks;

  int peerRank = (globalRank == 0) ? 1 : 0;

  MultipeerIbgdaTransportConfig transportConfig{
      .cudaDevice = localRank,
      .dataBufferSize = kSlotSize,
      .sendRecv =
          SendRecvConfig{
              .maxGroups = kMaxBlocks,
              .pipelineDepth = kPipelineDepth,
          },
  };

  MultipeerIbgdaTransport transport(
      globalRank, worldSize, bootstrap, transportConfig);
  transport.exchange();

  DeviceBuffer sendBuf(kDataBytes);
  DeviceBuffer recvBuf(kDataBytes);

  uint8_t fillPattern = 0xA0 + globalRank;
  ASSERT_CUDA_SUCCESS(cudaMemset(sendBuf.get(), fillPattern, kDataBytes));
  ASSERT_CUDA_SUCCESS(cudaMemset(recvBuf.get(), 0, kDataBytes));
  ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());

  bootstrap->barrierAll();

  auto* deviceTransport = transport.getP2pTransportDevice(peerRank);

  launch_ibgda_send_recv(
      deviceTransport,
      static_cast<char*>(sendBuf.get()),
      static_cast<char*>(recvBuf.get()),
      kDataBytes,
      kNumBlocks,
      stream_);

  cudaError_t err = cudaStreamSynchronize(stream_);
  ASSERT_EQ(err, cudaSuccess)
      << "Kernel execution failed: " << cudaGetErrorString(err);

  bootstrap->barrierAll();

  uint8_t expectedPattern = 0xA0 + peerRank;
  std::vector<uint8_t> hostBuf(kDataBytes);
  ASSERT_CUDA_SUCCESS(cudaMemcpy(
      hostBuf.data(), recvBuf.get(), kDataBytes, cudaMemcpyDeviceToHost));

  bool correct = true;
  for (std::size_t i = 0; i < kDataBytes; i++) {
    if (hostBuf[i] != expectedPattern) {
      XLOGF(
          ERR,
          "Rank {}: data mismatch at byte {}: expected 0x{:02X}, got 0x{:02X}",
          globalRank,
          i,
          expectedPattern,
          hostBuf[i]);
      correct = false;
      break;
    }
  }
  EXPECT_TRUE(correct) << "Rank " << globalRank
                       << ": tile sendrecv data correctness failed";
  if (correct) {
    XLOGF(
        INFO,
        "Rank {}: tile sendrecv correctness OK ({} bytes)",
        globalRank,
        kDataBytes);
  }
}

TEST_F(IbgdaSendRecvTest, UnalignedPayloadAdvancesAlignedProtocol) {
  if (worldSize != 2) {
    XLOGF(INFO, "Skipping: requires exactly 2 ranks, got {}", worldSize);
    return;
  }

  constexpr std::size_t kDataBytes = 100;
  constexpr int kNumBlocks = 1;
  constexpr std::size_t kSlotSize = 1024;
  constexpr int kPipelineDepth = 2;
  constexpr int kMaxBlocks = kNumBlocks;

  int peerRank = (globalRank == 0) ? 1 : 0;

  MultipeerIbgdaTransportConfig transportConfig{
      .cudaDevice = localRank,
      .dataBufferSize = kSlotSize,
      .sendRecv =
          SendRecvConfig{
              .maxGroups = kMaxBlocks,
              .pipelineDepth = kPipelineDepth,
          },
  };

  MultipeerIbgdaTransport transport(
      globalRank, worldSize, bootstrap, transportConfig);
  transport.exchange();

  DeviceBuffer sendBuf(kDataBytes);
  DeviceBuffer recvBuf(kDataBytes);

  const uint8_t fillPattern = static_cast<uint8_t>(0x70 + globalRank);
  ASSERT_EQ(cudaMemset(sendBuf.get(), fillPattern, kDataBytes), cudaSuccess);
  ASSERT_EQ(cudaMemset(recvBuf.get(), 0, kDataBytes), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  bootstrap->barrierAll();

  auto* deviceTransport = transport.getP2pTransportDevice(peerRank);
  launch_ibgda_send_recv(
      deviceTransport,
      static_cast<char*>(sendBuf.get()),
      static_cast<char*>(recvBuf.get()),
      kDataBytes,
      kNumBlocks,
      stream_);

  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
  bootstrap->barrierAll();

  expectUniformBuffer(
      recvBuf, kDataBytes, static_cast<uint8_t>(0x70 + peerRank), globalRank);
  expectStepState(
      snapshotStepState(deviceTransport, kMaxBlocks, stream_),
      kMaxBlocks,
      alignProtocolBytes(kDataBytes));
}

TEST_F(IbgdaSendRecvTest, StepStatePersistsAcrossKernelLaunches) {
  if (worldSize != 2) {
    XLOGF(INFO, "Skipping: requires exactly 2 ranks, got {}", worldSize);
    return;
  }

  constexpr int kNumBlocks = 4;
  constexpr int kPipelineDepth = 2;
  constexpr std::size_t kSlotSize = 1 * 1024 * 1024;
  constexpr std::size_t kSectionsFirst = 3;
  constexpr std::size_t kSectionsSecond = 2;
  constexpr std::size_t kBytesFirst = kSectionsFirst * kSlotSize;
  constexpr std::size_t kBytesSecond = kSectionsSecond * kSlotSize;
  constexpr std::size_t kBytesPerGroupPerSection = kSlotSize / kNumBlocks;
  constexpr std::size_t kMaxBytes = kBytesFirst;

  int peerRank = (globalRank == 0) ? 1 : 0;

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

  DeviceBuffer sendBuf(kMaxBytes);
  DeviceBuffer recvBuf(kMaxBytes);
  auto* deviceTransport = transport.getP2pTransportDevice(peerRank);

  bootstrap->barrierAll();

  const uint8_t firstPattern = static_cast<uint8_t>(0x40 + globalRank);
  ASSERT_EQ(cudaMemset(sendBuf.get(), firstPattern, kBytesFirst), cudaSuccess);
  ASSERT_EQ(cudaMemset(recvBuf.get(), 0, kBytesFirst), cudaSuccess);

  launch_ibgda_send_recv(
      deviceTransport,
      static_cast<char*>(sendBuf.get()),
      static_cast<char*>(recvBuf.get()),
      kBytesFirst,
      kNumBlocks,
      stream_);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
  bootstrap->barrierAll();

  expectUniformBuffer(
      recvBuf, kBytesFirst, static_cast<uint8_t>(0x40 + peerRank), globalRank);
  expectStepState(
      snapshotStepState(deviceTransport, kNumBlocks, stream_),
      kNumBlocks,
      static_cast<int64_t>(kSectionsFirst * kBytesPerGroupPerSection));

  const uint8_t secondPattern = static_cast<uint8_t>(0x50 + globalRank);
  ASSERT_EQ(
      cudaMemset(sendBuf.get(), secondPattern, kBytesSecond), cudaSuccess);
  ASSERT_EQ(cudaMemset(recvBuf.get(), 0, kBytesSecond), cudaSuccess);

  launch_ibgda_send_recv(
      deviceTransport,
      static_cast<char*>(sendBuf.get()),
      static_cast<char*>(recvBuf.get()),
      kBytesSecond,
      kNumBlocks,
      stream_);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
  bootstrap->barrierAll();

  expectUniformBuffer(
      recvBuf, kBytesSecond, static_cast<uint8_t>(0x50 + peerRank), globalRank);
  expectStepState(
      snapshotStepState(deviceTransport, kNumBlocks, stream_),
      kNumBlocks,
      static_cast<int64_t>(
          (kSectionsFirst + kSectionsSecond) * kBytesPerGroupPerSection));
}

TEST_F(IbgdaSendRecvTest, ChangingMaxSignalBytesWithinKernelUsesByteCursor) {
  if (worldSize != 2) {
    XLOGF(INFO, "Skipping: requires exactly 2 ranks, got {}", worldSize);
    return;
  }

  constexpr int kNumBlocks = 4;
  constexpr int kPipelineDepth = 2;
  constexpr std::size_t kPerBlockSlot = 64 * 1024;
  constexpr std::size_t kSlotSize = kPerBlockSlot * kNumBlocks;
  constexpr std::size_t kFirstBytes = kSlotSize;
  constexpr std::size_t kSecondBytes = (kPerBlockSlot / 2) * kNumBlocks;
  constexpr std::size_t kTotalBytes = kFirstBytes + kSecondBytes;
  constexpr std::size_t kFirstMaxSignalBytes = kPerBlockSlot;
  constexpr std::size_t kSecondMaxSignalBytes = 16 * 1024;

  int peerRank = (globalRank == 0) ? 1 : 0;

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

  DeviceBuffer sendBuf(kTotalBytes);
  DeviceBuffer recvBuf(kTotalBytes);
  auto* deviceTransport = transport.getP2pTransportDevice(peerRank);

  const uint8_t firstPattern = static_cast<uint8_t>(0x60 + globalRank);
  const uint8_t secondPattern = static_cast<uint8_t>(0x70 + globalRank);
  ASSERT_EQ(cudaMemset(sendBuf.get(), firstPattern, kFirstBytes), cudaSuccess);
  ASSERT_EQ(
      cudaMemset(
          static_cast<char*>(sendBuf.get()) + kFirstBytes,
          secondPattern,
          kSecondBytes),
      cudaSuccess);
  ASSERT_EQ(cudaMemset(recvBuf.get(), 0, kTotalBytes), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  bootstrap->barrierAll();

  launch_ibgda_send_recv_two_call(
      deviceTransport,
      static_cast<char*>(sendBuf.get()),
      static_cast<char*>(recvBuf.get()),
      kFirstBytes,
      kSecondBytes,
      kNumBlocks,
      kFirstMaxSignalBytes,
      kSecondMaxSignalBytes,
      stream_);
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
  bootstrap->barrierAll();

  expectUniformBufferRange(
      recvBuf,
      0,
      kFirstBytes,
      static_cast<uint8_t>(0x60 + peerRank),
      globalRank,
      "first");
  expectUniformBufferRange(
      recvBuf,
      kFirstBytes,
      kSecondBytes,
      static_cast<uint8_t>(0x70 + peerRank),
      globalRank,
      "second");
  expectStepState(
      snapshotStepState(deviceTransport, kNumBlocks, stream_),
      kNumBlocks,
      static_cast<int64_t>(kPerBlockSlot + kPerBlockSlot / 2));
}

TEST_F(IbgdaSendRecvTest, StepStatePersistsAcrossCudaGraphReplays) {
  if (worldSize != 2) {
    XLOGF(INFO, "Skipping: requires exactly 2 ranks, got {}", worldSize);
    return;
  }

  constexpr int kNumBlocks = 4;
  constexpr int kPipelineDepth = 2;
  constexpr std::size_t kSlotSize = 1 * 1024 * 1024;
  constexpr std::size_t kSectionsPerReplay = 2;
  constexpr std::size_t kBytesPerReplay = kSectionsPerReplay * kSlotSize;
  constexpr std::size_t kBytesPerGroupPerSection = kSlotSize / kNumBlocks;
  constexpr int kReplays = 3;

  int peerRank = (globalRank == 0) ? 1 : 0;

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

  DeviceBuffer sendBuf(kBytesPerReplay);
  DeviceBuffer recvBuf(kBytesPerReplay);
  auto* deviceTransport = transport.getP2pTransportDevice(peerRank);

  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

  cudaGraph_t graph{};
  cudaGraphExec_t graphExec{};
  ASSERT_EQ(
      cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal),
      cudaSuccess);
  launch_ibgda_send_recv(
      deviceTransport,
      static_cast<char*>(sendBuf.get()),
      static_cast<char*>(recvBuf.get()),
      kBytesPerReplay,
      kNumBlocks,
      stream_);
  ASSERT_EQ(cudaStreamEndCapture(stream_, &graph), cudaSuccess);
  ASSERT_NE(graph, nullptr);
  ASSERT_EQ(
      cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0),
      cudaSuccess);

  bootstrap->barrierAll();

  for (int replay = 0; replay < kReplays; ++replay) {
    const uint8_t sendPattern =
        static_cast<uint8_t>(0x70 + replay * 8 + globalRank);
    ASSERT_EQ(
        cudaMemsetAsync(sendBuf.get(), sendPattern, kBytesPerReplay, stream_),
        cudaSuccess);
    ASSERT_EQ(
        cudaMemsetAsync(recvBuf.get(), 0, kBytesPerReplay, stream_),
        cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);

    bootstrap->barrierAll();
    ASSERT_EQ(cudaGraphLaunch(graphExec, stream_), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
    bootstrap->barrierAll();

    expectUniformBuffer(
        recvBuf,
        kBytesPerReplay,
        static_cast<uint8_t>(0x70 + replay * 8 + peerRank),
        globalRank);
    expectStepState(
        snapshotStepState(deviceTransport, kNumBlocks, stream_),
        kNumBlocks,
        static_cast<int64_t>(
            (replay + 1) * kSectionsPerReplay * kBytesPerGroupPerSection));
  }

  ASSERT_EQ(cudaGraphExecDestroy(graphExec), cudaSuccess);
  ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
}

TEST_F(IbgdaSendRecvTest, BandwidthFrom4B) {
  runBidirectionalBandwidth(4, "(from 4B)");
}

TEST_F(IbgdaSendRecvTest, BandwidthFrom1MB) {
  runBidirectionalBandwidth(1ULL << 20, "(from 1MB)");
}

TEST_F(IbgdaSendRecvTest, UnidirectionalBandwidth) {
  if (worldSize != 2) {
    XLOGF(INFO, "Skipping: requires exactly 2 ranks, got {}", worldSize);
    return;
  }

  constexpr std::size_t kSlotSize = 8 * 1024 * 1024; // 8MB
  constexpr int kPipelineDepth = 2;
  constexpr int kWarmupIters = 5;
  constexpr int kBenchIters = 20;

  std::vector<std::size_t> messageSizes;
  for (std::size_t sz = 4; sz <= 4ULL << 30; sz <<= 1) {
    messageSizes.push_back(sz);
  }

  int peerRank = (globalRank == 0) ? 1 : 0;
  std::size_t maxBytes = messageSizes.back();

  cudaEvent_t start, stop;
  ASSERT_CUDA_SUCCESS(cudaEventCreate(&start));
  ASSERT_CUDA_SUCCESS(cudaEventCreate(&stop));

  for (const auto& geometry : kSendRecvBenchmarkGeometries) {
    MultipeerIbgdaTransportConfig transportConfig{
        .cudaDevice = localRank,
        .dataBufferSize = kSlotSize,
        .sendRecv =
            SendRecvConfig{
                .maxGroups = geometry.maxGroups,
                .pipelineDepth = kPipelineDepth,
            },
    };

    MultipeerIbgdaTransport transport(
        globalRank, worldSize, bootstrap, transportConfig);
    transport.exchange();

    DeviceBuffer sendBuf(maxBytes);
    DeviceBuffer recvBuf(maxBytes);
    ASSERT_CUDA_SUCCESS(cudaMemset(sendBuf.get(), 0xAA, maxBytes));
    ASSERT_CUDA_SUCCESS(cudaMemset(recvBuf.get(), 0, maxBytes));
    ASSERT_CUDA_SUCCESS(cudaDeviceSynchronize());

    auto* deviceTransport = transport.getP2pTransportDevice(peerRank);

    if (globalRank == 0) {
      XLOGF(INFO, "");
      XLOGF(
          INFO,
          "================================================================");
      XLOGF(
          INFO,
          "  IBGDA Tile Unidirectional Bandwidth ({}, rank 0 -> rank 1)",
          geometry.label);
      XLOGF(
          INFO,
          "  activeBlocks={}, maxGroups={}, slotSize={}MB, pipelineDepth={}",
          geometry.activeBlocks,
          geometry.maxGroups,
          kSlotSize / (1024 * 1024),
          kPipelineDepth);
      XLOGF(
          INFO,
          "================================================================");
      XLOGF(
          INFO,
          "{:>10s}  {:>10s}  {:>12s}",
          "MsgSize",
          "Sections",
          "BW (GB/s)");
      XLOGF(INFO, "----------------------------------------------");
    }

    for (auto nBytes : messageSizes) {
      bootstrap->barrierAll();

      // Warmup
      for (int i = 0; i < kWarmupIters; ++i) {
        if (globalRank == 0) {
          launch_ibgda_send(
              deviceTransport,
              static_cast<char*>(sendBuf.get()),
              nBytes,
              geometry.activeBlocks,
              stream_);
        } else {
          launch_ibgda_recv(
              deviceTransport,
              static_cast<char*>(recvBuf.get()),
              nBytes,
              geometry.activeBlocks,
              stream_);
        }
        ASSERT_CUDA_SUCCESS(cudaStreamSynchronize(stream_));
        bootstrap->barrierAll();
      }

      bootstrap->barrierAll();

      // Benchmark
      ASSERT_CUDA_SUCCESS(cudaEventRecord(start, stream_));
      for (int i = 0; i < kBenchIters; ++i) {
        if (globalRank == 0) {
          launch_ibgda_send(
              deviceTransport,
              static_cast<char*>(sendBuf.get()),
              nBytes,
              geometry.activeBlocks,
              stream_);
        } else {
          launch_ibgda_recv(
              deviceTransport,
              static_cast<char*>(recvBuf.get()),
              nBytes,
              geometry.activeBlocks,
              stream_);
        }
      }
      ASSERT_CUDA_SUCCESS(cudaEventRecord(stop, stream_));
      ASSERT_CUDA_SUCCESS(cudaEventSynchronize(stop));

      bootstrap->barrierAll();

      float totalMs = 0;
      ASSERT_CUDA_SUCCESS(cudaEventElapsedTime(&totalMs, start, stop));
      float avgMs = totalMs / kBenchIters;

      // Unidirectional: only one direction
      float bwGBs = (nBytes / 1e9f) / (avgMs / 1000.0f);
      std::size_t numSections = (nBytes + kSlotSize - 1) / kSlotSize;

      if (globalRank == 0) {
        std::string sizeStr = formatMessageSize(nBytes);
        XLOGF(INFO, "{:>10s}  {:>10d}  {:>12.2f}", sizeStr, numSections, bwGBs);
      }
    }

    if (globalRank == 0) {
      XLOGF(
          INFO,
          "================================================================");
    }

    bootstrap->barrierAll();
  }

  ASSERT_CUDA_SUCCESS(cudaEventDestroy(start));
  ASSERT_CUDA_SUCCESS(cudaEventDestroy(stop));
}

} // namespace comms::prims::benchmark

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  if (const char* localRank = std::getenv("LOCAL_RANK")) {
    cudaError_t ret = cudaSetDevice(std::atoi(localRank));
    CHECK_EQ(ret, cudaSuccess) << cudaGetErrorString(ret);
  }
  folly::Init init(&argc, &argv);
  if (!meta::comms::isTcpEnvironment()) {
    ::testing::AddGlobalTestEnvironment(new BenchmarkEnvironment());
  }
  return RUN_ALL_TESTS();
}
