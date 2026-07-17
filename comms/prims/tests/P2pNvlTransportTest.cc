// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include <algorithm>
#include <string>
#include <vector>

#include "comms/prims/benchmarks/TileSendRecv.cuh"
#include "comms/prims/core/TiledBuffer.cuh"
#include "comms/prims/core/TimeoutUtils.h"
#include "comms/prims/tests/P2pNvlTransportTest.cuh"
#include "comms/prims/tests/Utils.cuh"
#include "comms/prims/transport/nvl/MultiPeerNvlTransport.h"
#include "comms/prims/transport/nvl/P2pNvlTransportDevice.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

using namespace meta::comms;

namespace comms::prims::tests {

class P2pNvlTransportTestFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }

  void TearDown() override {
    MpiBaseTestFixture::TearDown();
  }
};

MultiPeerNvlTransportConfig makeNvlConfig(
    std::size_t dataBufferSize,
    std::size_t pipelineDepth,
    int maxNumChannels = 1) {
  const auto channels = static_cast<std::size_t>(maxNumChannels);
  const std::size_t chunkAlign = 16 * std::max<std::size_t>(pipelineDepth, 1);
  const std::size_t perChannelSize =
      std::max<std::size_t>(16, ((dataBufferSize + channels - 1) / channels));
  return MultiPeerNvlTransportConfig{
      .pipelineDepth = pipelineDepth,
      .maxNumChannels = maxNumChannels,
      .perChannelSize =
          ((perChannelSize + chunkAlign - 1) / chunkAlign) * chunkAlign,
  };
}

static std::vector<char> makeTwoCallPatternBuffer(
    size_t firstCallBytes,
    size_t secondCallBytes,
    int numBlocks,
    int rank);

static void expectTwoCallPattern(
    const std::vector<char>& data,
    size_t firstCallBytes,
    size_t secondCallBytes,
    int numBlocks,
    int sourceRank,
    const std::string& label);

TEST_F(P2pNvlTransportTestFixture, PipelineGeometry) {
  if (numRanks != 2) {
    GTEST_SKIP() << "Requires exactly 2 ranks, got " << numRanks;
  }

  constexpr std::size_t perChannelBufferSize = 64 * 1024;
  constexpr std::size_t pipelineDepth = 4;
  constexpr int maxNumChannels = 8;
  constexpr std::size_t pipelineChunk = perChannelBufferSize / pipelineDepth;
  const int peerRank = (globalRank == 0) ? 1 : 0;

  MultiPeerNvlTransportConfig config{
      .pipelineDepth = pipelineDepth,
      .maxNumChannels = maxNumChannels,
      .perChannelSize = perChannelBufferSize,
  };

  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  MultiPeerNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.exchange();
  auto p2p = transport.buildP2pTransportDevice(peerRank);

  EXPECT_EQ(p2p.pipeline_depth(), pipelineDepth);
  EXPECT_EQ(p2p.pipeline_window(), perChannelBufferSize);
  EXPECT_EQ(p2p.pipeline_chunk(), pipelineChunk);
}

TEST_F(P2pNvlTransportTestFixture, IpcMemAccess) {
  // Only test with 2 ranks
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;

  const size_t numElements = 256;
  auto config = makeNvlConfig(sizeof(int) * numElements, 4);

  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  MultiPeerNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.exchange();
  XLOGF(INFO, "Rank {} created transport and exchanged IPC", globalRank);

  // Get host-side copy to access buffer pointers from host
  auto p2p = transport.buildP2pTransportDevice(peerRank);

  auto localAddr =
      static_cast<int*>(static_cast<void*>(p2p.getLocalState().dataBuffer));
  auto remoteAddr =
      static_cast<int*>(static_cast<void*>(p2p.getRemoteState().dataBuffer));
  XLOGF(
      INFO,
      "Rank {}: localAddr: {}, remoteAddr: {}",
      globalRank,
      static_cast<void*>(localAddr),
      static_cast<void*>(remoteAddr));

  // Each rank writes its pattern to local buffer
  // rank0 writes all 0s, rank1 writes all 1s
  int writeValue = globalRank;
  test::fillBuffer(localAddr, writeValue, numElements);
  CUDACHECK_TEST(cudaDeviceSynchronize());
  XLOGF(INFO, "Rank {} filled local buffer with {}", globalRank, writeValue);

  // Barrier to ensure both ranks have written their data
  MPI_Barrier(MPI_COMM_WORLD);
  XLOGF(INFO, "Rank {} passed barrier", globalRank);

  // Now each rank reads from peer buffer and verifies
  // rank0 should read all 1s from rank1
  // rank1 should read all 0s from rank0
  int expectedValue = peerRank;

  // Allocate error counter on device using DeviceBuffer
  DeviceBuffer errorCountBuffer(sizeof(int));
  auto d_errorCount = static_cast<int*>(errorCountBuffer.get());
  CUDACHECK_TEST(cudaMemset(d_errorCount, 0, sizeof(int)));

  test::verifyBuffer(remoteAddr, expectedValue, numElements, d_errorCount);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Copy error count back to host
  int h_errorCount = 0;
  CUDACHECK_TEST(cudaMemcpy(
      &h_errorCount, d_errorCount, sizeof(int), cudaMemcpyDeviceToHost));

  XLOGF(
      INFO,
      "Rank {} verified peer buffer, errors: {}",
      globalRank,
      h_errorCount);

  // Assert no errors
  ASSERT_EQ(h_errorCount, 0)
      << "Rank " << globalRank << " found " << h_errorCount
      << " errors when reading from peer rank " << peerRank;
}

// =============================================================================
// TransportTestHelper - Reduces boilerplate for creating transport objects
// =============================================================================

class TransportTestHelper {
 public:
  TransportTestHelper(
      int globalRank,
      int numRanks,
      int localRank,
      const MultiPeerNvlTransportConfig& config)
      : globalRank_(globalRank),
        numRanks_(numRanks),
        peerRank_((globalRank == 0) ? 1 : 0),
        bootstrap_(std::make_shared<meta::comms::MpiBootstrap>()),
        transport_(
            std::make_unique<MultiPeerNvlTransport>(
                globalRank,
                numRanks,
                bootstrap_,
                config)) {
    CUDACHECK_TEST(cudaSetDevice(localRank));
    transport_->exchange();

    // Build a host copy of P2pNvlTransportDevice for tests that need
    // to access buffer pointers from the host side (e.g., for cudaMemset)
    // Use unique_ptr because P2pNvlTransportDevice has const members and
    // cannot be copy-assigned
    p2pHost_ = std::make_unique<P2pNvlTransportDevice>(
        transport_->buildP2pTransportDevice(peerRank_));

    p2pDevice_ = std::make_unique<DeviceBuffer>(sizeof(P2pNvlTransportDevice));
    CUDACHECK_TEST(cudaMemcpy(
        p2pDevice_->get(),
        p2pHost_.get(),
        sizeof(P2pNvlTransportDevice),
        cudaMemcpyHostToDevice));
  }

  // Returns pointer to preallocated P2pNvlTransportDevice on device
  // This pointer is managed by MultiPeerNvlTransport
  P2pNvlTransportDevice* getDevicePtr() {
    return static_cast<P2pNvlTransportDevice*>(p2pDevice_->get());
  }

  // Returns reference to host copy (for accessing state pointers from host)
  P2pNvlTransportDevice& getHostDevice() {
    return *p2pHost_;
  }

  int peerRank() const {
    return peerRank_;
  }

  int globalRank() const {
    return globalRank_;
  }

 private:
  int globalRank_;
  int numRanks_;
  int peerRank_;
  std::shared_ptr<meta::comms::MpiBootstrap> bootstrap_;
  std::unique_ptr<MultiPeerNvlTransport> transport_;
  std::unique_ptr<P2pNvlTransportDevice> p2pHost_;
  std::unique_ptr<DeviceBuffer> p2pDevice_;
};

// =============================================================================
// Tile sendrecv multi-call correctness test
// =============================================================================

TEST_F(P2pNvlTransportTestFixture, TileSendRecvMultiCall) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping: requires 2 ranks, got {}", numRanks);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;
  const size_t nBytes = 8 * 1024 * 1024; // 8MB
  const int numSendBlocks = 4;
  const int nIters = 5; // call sendrecv 5 times with different data

  auto config = makeNvlConfig(2 * 1024 * 1024, 2, numSendBlocks);

  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  MultiPeerNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.exchange();
  auto p2pHost = transport.buildP2pTransportDevice(peerRank);

  DeviceBuffer sendBuf(nBytes);
  DeviceBuffer recvBuf(nBytes);

  dim3 grid(numSendBlocks * 2);
  dim3 block(256);

  Timeout timeout;

  for (int iter = 0; iter < nIters; iter++) {
    const int pattern = 0x10 + globalRank + iter * 0x20;
    const int peerPattern = 0x10 + peerRank + iter * 0x20;

    CUDACHECK_TEST(cudaMemset(sendBuf.get(), pattern, nBytes));
    CUDACHECK_TEST(cudaMemset(recvBuf.get(), 0, nBytes));

    comms::prims::TiledBuffer<char> sendTiles(
        static_cast<char*>(sendBuf.get()), nBytes, numSendBlocks);
    comms::prims::TiledBuffer<char> recvTiles(
        static_cast<char*>(recvBuf.get()), nBytes, numSendBlocks);
    std::size_t maxSignalBytes = 0;
    void* args[] = {
        &p2pHost, &sendTiles, &recvTiles, &maxSignalBytes, &timeout};

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    CUDACHECK_TEST(cudaLaunchKernel(
        (void*)comms::prims::benchmark::p2pTileSendRecv,
        grid,
        block,
        args,
        0,
        nullptr));
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Verify received data
    std::vector<char> hostBuf(nBytes);
    CUDACHECK_TEST(cudaMemcpy(
        hostBuf.data(), recvBuf.get(), nBytes, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < nBytes; i++) {
      EXPECT_EQ(
          static_cast<unsigned char>(hostBuf[i]),
          static_cast<unsigned char>(peerPattern))
          << "Iter " << iter << ": Mismatch at byte " << i;
      if (static_cast<unsigned char>(hostBuf[i]) !=
          static_cast<unsigned char>(peerPattern)) {
        break;
      }
    }
  }
}

TEST_F(
    P2pNvlTransportTestFixture,
    TileTwoCallSendThenRecvPaddingCreditNoDeadlock) {
  if (numRanks != 2) {
    GTEST_SKIP() << "Skipping: requires 2 ranks, got " << numRanks;
  }

  const int peerRank = (globalRank == 0) ? 1 : 0;
  const int numBlocks = 1;
  const int threadCount = 256;
  const size_t perChannelSize = 1024 * 1024;
  const size_t pipelineDepth = 1;
  const size_t maxSignalBytes = 0;

  auto config = makeNvlConfig(perChannelSize, pipelineDepth, numBlocks);
  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  MultiPeerNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.exchange();
  auto p2pHost = transport.buildP2pTransportDevice(peerRank);

  const size_t firstCallBytes = p2pHost.pipeline_window() / 2;
  const size_t secondCallBytes = p2pHost.pipeline_window();
  const size_t tileBytes = firstCallBytes + secondCallBytes;
  const size_t totalBytes = tileBytes * numBlocks;
  const std::vector<char> hostSend = makeTwoCallPatternBuffer(
      firstCallBytes, secondCallBytes, numBlocks, globalRank);

  DeviceBuffer sendBuf(totalBytes);
  DeviceBuffer recvBuf(totalBytes);
  CUDACHECK_TEST(cudaMemcpy(
      sendBuf.get(), hostSend.data(), totalBytes, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(recvBuf.get(), 0, totalBytes));

  TiledBuffer<char> sendTiles(
      static_cast<char*>(sendBuf.get()), totalBytes, numBlocks);
  TiledBuffer<char> recvTiles(
      static_cast<char*>(recvBuf.get()), totalBytes, numBlocks);

  int device = 0;
  CUDACHECK_TEST(cudaGetDevice(&device));
  Timeout timeout = makeTimeout(5000, device);

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  test::testTileTwoCallSendThenRecv(
      p2pHost,
      sendTiles,
      recvTiles,
      numBlocks,
      firstCallBytes,
      secondCallBytes,
      maxSignalBytes,
      threadCount,
      timeout);
  CUDACHECK_TEST(cudaDeviceSynchronize());
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  std::vector<char> hostRecv(totalBytes);
  CUDACHECK_TEST(cudaMemcpy(
      hostRecv.data(), recvBuf.get(), totalBytes, cudaMemcpyDeviceToHost));
  expectTwoCallPattern(
      hostRecv,
      firstCallBytes,
      secondCallBytes,
      numBlocks,
      peerRank,
      "tile send-then-recv padding credit");
}

TEST_F(P2pNvlTransportTestFixture, TileSendRecvCudaGraphReplay) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping: requires 2 ranks, got {}", numRanks);
    return;
  }

  const int peerRank = (globalRank == 0) ? 1 : 0;
  const size_t nBytes = 2 * 1024 * 1024;
  const int numSendBlocks = 4;
  const size_t maxSignalBytes = 16 * 1024;

  auto config = makeNvlConfig(128 * 1024, 2, numSendBlocks);

  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  MultiPeerNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.exchange();
  auto p2pHost = transport.buildP2pTransportDevice(peerRank);

  DeviceBuffer sendBuf(nBytes);
  DeviceBuffer recvBuf(nBytes);
  comms::prims::TiledBuffer<char> sendTiles(
      static_cast<char*>(sendBuf.get()), nBytes, numSendBlocks);
  comms::prims::TiledBuffer<char> recvTiles(
      static_cast<char*>(recvBuf.get()), nBytes, numSendBlocks);

  std::size_t maxSignalBytesArg = maxSignalBytes;
  Timeout timeout;
  void* args[] = {
      &p2pHost, &sendTiles, &recvTiles, &maxSignalBytesArg, &timeout};

  cudaStream_t stream;
  CUDACHECK_TEST(cudaStreamCreate(&stream));
  cudaGraph_t graph;
  cudaGraphExec_t graphExec;

  CUDACHECK_TEST(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  CUDACHECK_TEST(cudaLaunchKernel(
      (void*)comms::prims::benchmark::p2pTileSendRecv,
      dim3(numSendBlocks * 2),
      dim3(256),
      args,
      0,
      stream));
  CUDACHECK_TEST(cudaStreamEndCapture(stream, &graph));
  CUDACHECK_TEST(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  for (int iter = 0; iter < 3; ++iter) {
    const int pattern = 0x40 + globalRank + iter * 0x20;
    const int peerPattern = 0x40 + peerRank + iter * 0x20;
    CUDACHECK_TEST(cudaMemset(sendBuf.get(), pattern, nBytes));
    CUDACHECK_TEST(cudaMemset(recvBuf.get(), 0, nBytes));
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    CUDACHECK_TEST(cudaGraphLaunch(graphExec, stream));
    CUDACHECK_TEST(cudaStreamSynchronize(stream));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    std::vector<char> hostBuf(nBytes);
    CUDACHECK_TEST(cudaMemcpy(
        hostBuf.data(), recvBuf.get(), nBytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < nBytes; ++i) {
      EXPECT_EQ(
          static_cast<unsigned char>(hostBuf[i]),
          static_cast<unsigned char>(peerPattern))
          << "CudaGraph iter " << iter << ": mismatch at byte " << i;
      if (static_cast<unsigned char>(hostBuf[i]) !=
          static_cast<unsigned char>(peerPattern)) {
        break;
      }
    }
  }

  CUDACHECK_TEST(cudaGraphExecDestroy(graphExec));
  CUDACHECK_TEST(cudaGraphDestroy(graph));
  CUDACHECK_TEST(cudaStreamDestroy(stream));
}

// =============================================================================
// send / recv (per-group) Tests
// =============================================================================

// Helper: run tile sendrecv with given params and verify correctness
static void runTileTest(
    int globalRank,
    int numRanks,
    std::shared_ptr<meta::comms::MpiBootstrap> bootstrap,
    size_t nBytes,
    size_t perChannelSize,
    size_t chunkSize,
    size_t pipelineDepth,
    int numSendBlocks,
    int nIters,
    int threadCount = 256) {
  int peerRank = (globalRank == 0) ? 1 : 0;

  auto config = makeNvlConfig(perChannelSize, pipelineDepth, numSendBlocks);

  MultiPeerNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.exchange();
  auto p2pHost = transport.buildP2pTransportDevice(peerRank);

  DeviceBuffer sendBuf(nBytes);
  DeviceBuffer recvBuf(nBytes);

  dim3 grid(numSendBlocks * 2);
  dim3 block(threadCount);

  Timeout timeout;

  for (int iter = 0; iter < nIters; iter++) {
    const int pattern = 0x10 + globalRank + iter * 0x20;
    const int peerPattern = 0x10 + peerRank + iter * 0x20;

    CUDACHECK_TEST(cudaMemset(sendBuf.get(), pattern, nBytes));
    CUDACHECK_TEST(cudaMemset(recvBuf.get(), 0, nBytes));

    comms::prims::TiledBuffer<char> sendTiles(
        static_cast<char*>(sendBuf.get()), nBytes, numSendBlocks);
    comms::prims::TiledBuffer<char> recvTiles(
        static_cast<char*>(recvBuf.get()), nBytes, numSendBlocks);
    std::size_t maxSignalBytes = 0;
    void* args[] = {
        &p2pHost, &sendTiles, &recvTiles, &maxSignalBytes, &timeout};

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    CUDACHECK_TEST(cudaLaunchKernel(
        (void*)comms::prims::benchmark::p2pTileSendRecv,
        grid,
        block,
        args,
        0,
        nullptr));
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    std::vector<char> hostBuf(nBytes);
    CUDACHECK_TEST(cudaMemcpy(
        hostBuf.data(), recvBuf.get(), nBytes, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < nBytes; i++) {
      EXPECT_EQ(
          static_cast<unsigned char>(hostBuf[i]),
          static_cast<unsigned char>(peerPattern))
          << "Iter " << iter << ": Mismatch at byte " << i
          << " (nBytes=" << nBytes << ", blocks=" << numSendBlocks
          << ", perChannelSize=" << perChannelSize << ", chunk=" << chunkSize
          << ", pd=" << pipelineDepth << ")";
      if (static_cast<unsigned char>(hostBuf[i]) !=
          static_cast<unsigned char>(peerPattern)) {
        return; // stop on first failure
      }
    }
  }
}

static unsigned char multiCallPattern(int rank, int call) {
  return static_cast<unsigned char>(0x30 + rank * 0x20 + call);
}

static size_t alignTileProtocolBytes(size_t nbytes) {
  return (nbytes + 15ULL) & ~15ULL;
}

static uint64_t roundUpToMultiple(uint64_t value, size_t alignment) {
  if (alignment == 0) {
    return value;
  }
  const uint64_t alignment64 = static_cast<uint64_t>(alignment);
  return ((value + alignment64 - 1) / alignment64) * alignment64;
}

static size_t signalAlignment(size_t maxSignalBytes, size_t perBlockSlotSize) {
  const bool usesPartialSlot =
      maxSignalBytes > 0 && maxSignalBytes < perBlockSlotSize;
  size_t alignment =
      usesPartialSlot ? (maxSignalBytes & ~15ULL) : perBlockSlotSize;
  return alignment == 0 ? perBlockSlotSize : alignment;
}

static size_t protocolStepBytes(
    uint64_t baseByte,
    size_t payloadBytes,
    size_t maxSignalBytes,
    size_t perBlockSlotSize) {
  const size_t protocolBytes = alignTileProtocolBytes(payloadBytes);
  const size_t alignment = signalAlignment(maxSignalBytes, perBlockSlotSize);
  const uint64_t payloadEnd = baseByte + protocolBytes;
  return protocolBytes +
      static_cast<size_t>(
             roundUpToMultiple(payloadEnd, alignment) - payloadEnd);
}

static size_t
validPayloadBytes(size_t byteOffset, size_t chunkBytes, size_t payloadBytes) {
  if (byteOffset >= payloadBytes) {
    return 0;
  }
  const size_t remaining = payloadBytes - byteOffset;
  return chunkBytes < remaining ? chunkBytes : remaining;
}

static std::vector<char> makeTwoCallPatternBuffer(
    size_t firstCallBytes,
    size_t secondCallBytes,
    int numBlocks,
    int rank) {
  const size_t tileBytes = firstCallBytes + secondCallBytes;
  std::vector<char> data(tileBytes * numBlocks);

  for (int block = 0; block < numBlocks; ++block) {
    const size_t offset = block * tileBytes;
    std::fill(
        data.begin() + offset,
        data.begin() + offset + firstCallBytes,
        static_cast<char>(multiCallPattern(rank, 0)));
    std::fill(
        data.begin() + offset + firstCallBytes,
        data.begin() + offset + tileBytes,
        static_cast<char>(multiCallPattern(rank, 1)));
  }

  return data;
}

static void expectTwoCallPattern(
    const std::vector<char>& data,
    size_t firstCallBytes,
    size_t secondCallBytes,
    int numBlocks,
    int sourceRank,
    const std::string& label) {
  const size_t callBytes[] = {firstCallBytes, secondCallBytes};
  const size_t tileBytes = firstCallBytes + secondCallBytes;

  for (int block = 0; block < numBlocks; ++block) {
    size_t callOffset = block * tileBytes;
    for (int call = 0; call < 2; ++call) {
      const unsigned char expected = multiCallPattern(sourceRank, call);
      for (size_t i = 0; i < callBytes[call]; ++i) {
        EXPECT_EQ(static_cast<unsigned char>(data[callOffset + i]), expected)
            << label << ": block=" << block << " call=" << call
            << " byte=" << i;
        if (static_cast<unsigned char>(data[callOffset + i]) != expected) {
          return;
        }
      }
      callOffset += callBytes[call];
    }
  }
}

static void expectPersistentTwoCallStagingPattern(
    const std::vector<char>& data,
    size_t slotSize,
    size_t firstCallBytes,
    size_t secondCallBytes,
    size_t maxSignalBytes,
    size_t pipelineDepth,
    int numBlocks,
    int sourceRank,
    const std::string& label) {
  const size_t callBytes[] = {firstCallBytes, secondCallBytes};
  const size_t perBlockSlotSize = (slotSize / numBlocks) & ~15ULL;
  const size_t chunkSize =
      maxSignalBytes > 0 && maxSignalBytes < perBlockSlotSize
      ? (maxSignalBytes & ~15ULL)
      : perBlockSlotSize;
  const size_t effectiveChunk = chunkSize > 0 ? chunkSize : perBlockSlotSize;
  const size_t pipelineBytes = perBlockSlotSize * pipelineDepth;

  for (int block = 0; block < numBlocks; ++block) {
    uint64_t baseByte = 0;
    for (int call = 0; call < 2; ++call) {
      const unsigned char expected = multiCallPattern(sourceRank, call);
      const size_t protocolBytes = alignTileProtocolBytes(callBytes[call]);
      for (size_t dataOff = 0; dataOff < protocolBytes;) {
        const uint64_t streamStart = baseByte + dataOff;
        const size_t pipelineOff =
            static_cast<size_t>(streamStart % pipelineBytes);
        const size_t slot = pipelineOff / perBlockSlotSize;
        const size_t chunkOff = pipelineOff - slot * perBlockSlotSize;
        const size_t slotRemaining = perBlockSlotSize - chunkOff;
        const size_t dataRemaining = protocolBytes - dataOff;
        size_t chunkBytes =
            effectiveChunk < dataRemaining ? effectiveChunk : dataRemaining;
        chunkBytes = chunkBytes < slotRemaining ? chunkBytes : slotRemaining;
        const size_t validBytes =
            validPayloadBytes(dataOff, chunkBytes, callBytes[call]);
        const size_t offset =
            slot * slotSize + block * perBlockSlotSize + chunkOff;

        for (size_t i = 0; i < validBytes; ++i) {
          EXPECT_EQ(static_cast<unsigned char>(data[offset + i]), expected)
              << label << ": block=" << block << " call=" << call
              << " dataOff=" << dataOff << " byte=" << i;
          if (static_cast<unsigned char>(data[offset + i]) != expected) {
            return;
          }
        }
        dataOff += chunkBytes;
      }
      baseByte += protocolStepBytes(
          baseByte, callBytes[call], maxSignalBytes, perBlockSlotSize);
    }
  }
}

// Build a P2pNvlTransportDevice for tile tests that bypass
// MultiPeerNvlTransport. The options must already have channel geometry and
// max_num_channels populated.
static P2pNvlTransportDevice makeLocalTileDevice(
    const P2pNvlTransportOptions& options,
    int numBlocks,
    char* localData,
    char* remoteData,
    NvlChannelState* localChannels,
    NvlChannelState* remoteChannels) {
  P2pNvlTransportOptions opts = options;
  if (opts.max_num_channels == 0) {
    opts.max_num_channels = numBlocks;
  }
  LocalState localState{
      localData,
      DeviceSpan<SignalState>(),
      DeviceSpan<BarrierState>(),
      nullptr,
      nullptr};

  RemoteState remoteState{
      remoteData,
      DeviceSpan<SignalState>(),
      DeviceSpan<BarrierState>(),
      nullptr,
      nullptr};

  return P2pNvlTransportDevice(
      0, 1, opts, localState, remoteState, localChannels, remoteChannels);
}

class LocalTileHarness {
 public:
  LocalTileHarness(const P2pNvlTransportOptions& options, int numBlocks)
      : options_(options),
        numBlocks_(numBlocks),
        stagingBytes_(options.dataBufferSize),
        channelBytes_(sizeof(NvlChannelState) * numBlocks),
        stagingBuf_(stagingBytes_),
        channelBuf_(channelBytes_) {
    zero();
  }

  void zero() {
    CUDACHECK_TEST(cudaMemset(stagingBuf_.get(), 0, stagingBytes_));
    CUDACHECK_TEST(cudaMemset(channelBuf_.get(), 0, channelBytes_));
  }

  P2pNvlTransportDevice device(
      char* localData = nullptr,
      char* remoteData = nullptr) {
    return makeLocalTileDevice(
        options_, numBlocks_, localData, remoteData, channels(), channels());
  }

  P2pNvlTransportDevice stagingDevice() {
    return device(staging(), staging());
  }

  // Same channels for local and peer — single-rank loopback. Equivalent to
  // device(staging(), staging()) but kept distinct for clarity at call sites.
  P2pNvlTransportDevice loopbackDevice() {
    return device(staging(), staging());
  }

  NvlChannelState* channels() {
    return static_cast<NvlChannelState*>(channelBuf_.get());
  }

  char* staging() {
    return static_cast<char*>(stagingBuf_.get());
  }

  DeviceBuffer& stagingBuffer() {
    return stagingBuf_;
  }

  DeviceBuffer& channelBuffer() {
    return channelBuf_;
  }

  size_t stagingBytes() const {
    return stagingBytes_;
  }

 private:
  const P2pNvlTransportOptions options_;
  const int numBlocks_;
  const size_t stagingBytes_;
  const size_t channelBytes_;
  DeviceBuffer stagingBuf_;
  DeviceBuffer channelBuf_;
};

// Test various message sizes with default config
TEST_F(P2pNvlTransportTestFixture, TileSendRecvMessageSizes) {
  if (numRanks != 2) {
    return;
  }
  auto bs = std::make_shared<meta::comms::MpiBootstrap>();

  // Small sizes
  runTileTest(
      globalRank,
      numRanks,
      bs,
      4096,
      8 * 1024 * 1024,
      8 * 1024 * 1024,
      2,
      4,
      1);
  runTileTest(
      globalRank,
      numRanks,
      bs,
      16384,
      8 * 1024 * 1024,
      8 * 1024 * 1024,
      2,
      4,
      1);
  runTileTest(
      globalRank,
      numRanks,
      bs,
      65536,
      8 * 1024 * 1024,
      8 * 1024 * 1024,
      2,
      4,
      1);

  // Medium sizes
  runTileTest(
      globalRank,
      numRanks,
      bs,
      1 * 1024 * 1024,
      8 * 1024 * 1024,
      8 * 1024 * 1024,
      2,
      8,
      1);
  runTileTest(
      globalRank,
      numRanks,
      bs,
      8 * 1024 * 1024,
      8 * 1024 * 1024,
      8 * 1024 * 1024,
      2,
      16,
      1);

  // Large sizes
  runTileTest(
      globalRank,
      numRanks,
      bs,
      64 * 1024 * 1024,
      8 * 1024 * 1024,
      8 * 1024 * 1024,
      2,
      16,
      1);
  runTileTest(
      globalRank,
      numRanks,
      bs,
      256 * 1024 * 1024,
      8 * 1024 * 1024,
      8 * 1024 * 1024,
      2,
      16,
      1);
}

// Test signal granularity (chunkSize < slotSize)
TEST_F(P2pNvlTransportTestFixture, TileSendRecvSignalGranularity) {
  if (numRanks != 2) {
    return;
  }
  auto bs = std::make_shared<meta::comms::MpiBootstrap>();
  const size_t nBytes = 32 * 1024 * 1024; // 32MB

  // Per-slot signaling (chunkSize == slotSize)
  runTileTest(
      globalRank,
      numRanks,
      bs,
      nBytes,
      8 * 1024 * 1024,
      8 * 1024 * 1024,
      2,
      16,
      1);

  // 128KB signal granularity
  runTileTest(
      globalRank, numRanks, bs, nBytes, 8 * 1024 * 1024, 128 * 1024, 2, 16, 1);

  // 512KB signal granularity
  runTileTest(
      globalRank, numRanks, bs, nBytes, 8 * 1024 * 1024, 512 * 1024, 2, 16, 1);

  // 1MB signal granularity
  runTileTest(
      globalRank, numRanks, bs, nBytes, 8 * 1024 * 1024, 1024 * 1024, 2, 16, 1);
}

// Test different block counts
TEST_F(P2pNvlTransportTestFixture, TileSendRecvBlockCounts) {
  if (numRanks != 2) {
    return;
  }
  auto bs = std::make_shared<meta::comms::MpiBootstrap>();
  const size_t nBytes = 16 * 1024 * 1024; // 16MB

  runTileTest(
      globalRank,
      numRanks,
      bs,
      nBytes,
      8 * 1024 * 1024,
      8 * 1024 * 1024,
      2,
      1,
      1);
  runTileTest(
      globalRank,
      numRanks,
      bs,
      nBytes,
      8 * 1024 * 1024,
      8 * 1024 * 1024,
      2,
      2,
      1);
  runTileTest(
      globalRank,
      numRanks,
      bs,
      nBytes,
      8 * 1024 * 1024,
      8 * 1024 * 1024,
      2,
      4,
      1);
  runTileTest(
      globalRank,
      numRanks,
      bs,
      nBytes,
      8 * 1024 * 1024,
      8 * 1024 * 1024,
      2,
      8,
      1);
  runTileTest(
      globalRank,
      numRanks,
      bs,
      nBytes,
      8 * 1024 * 1024,
      8 * 1024 * 1024,
      2,
      16,
      1);
  runTileTest(
      globalRank,
      numRanks,
      bs,
      nBytes,
      8 * 1024 * 1024,
      8 * 1024 * 1024,
      2,
      32,
      1);
}

// Test pipeline depth variations
TEST_F(P2pNvlTransportTestFixture, TileSendRecvPipelineDepth) {
  if (numRanks != 2) {
    return;
  }
  auto bs = std::make_shared<meta::comms::MpiBootstrap>();
  const size_t nBytes = 32 * 1024 * 1024;

  runTileTest(
      globalRank, numRanks, bs, nBytes, 8 * 1024 * 1024, 128 * 1024, 2, 16, 1);
  runTileTest(
      globalRank, numRanks, bs, nBytes, 8 * 1024 * 1024, 128 * 1024, 4, 16, 1);
}

// Test multi-call with persistent step state
TEST_F(P2pNvlTransportTestFixture, TileSendRecvMultiCallPersistentStep) {
  if (numRanks != 2) {
    return;
  }
  auto bs = std::make_shared<meta::comms::MpiBootstrap>();

  // 5 iterations with same size — tests step counter persistence
  runTileTest(
      globalRank,
      numRanks,
      bs,
      8 * 1024 * 1024,
      8 * 1024 * 1024,
      8 * 1024 * 1024,
      2,
      4,
      5);

  // 5 iterations with 128KB signal — more steps per call
  runTileTest(
      globalRank,
      numRanks,
      bs,
      8 * 1024 * 1024,
      8 * 1024 * 1024,
      128 * 1024,
      2,
      4,
      5);
}

TEST_F(
    P2pNvlTransportTestFixture,
    TileSendRecvMultiCallWithoutDrainUsesPersistentCursor) {
  const int numBlocks = 4;
  const int threadCount = 256;
  const size_t maxSignalBytes = 16 * 1024;
  const size_t firstCallBytes = maxSignalBytes;
  const size_t secondCallBytes = maxSignalBytes * 3;
  const size_t tileBytes = firstCallBytes + secondCallBytes;
  const size_t totalBytes = tileBytes * numBlocks;

  P2pNvlTransportOptions options{
      .dataBufferSize = tileBytes * 2 * numBlocks,
      .pipelineDepth = 2,
      .per_channel_buffer = tileBytes * 2,
      .per_channel_slot = tileBytes,
      .max_num_channels = numBlocks,
  };
  LocalTileHarness p2pHarness(options, numBlocks);

  const std::vector<char> hostSend =
      makeTwoCallPatternBuffer(firstCallBytes, secondCallBytes, numBlocks, 0);
  DeviceBuffer sendBuf(totalBytes);
  DeviceBuffer recvBuf(totalBytes);
  CUDACHECK_TEST(cudaMemcpy(
      sendBuf.get(), hostSend.data(), totalBytes, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(recvBuf.get(), 0, totalBytes));

  auto p2pHost = p2pHarness.stagingDevice();

  TiledBuffer<char> sendTiles(
      static_cast<char*>(sendBuf.get()), totalBytes, numBlocks);
  TiledBuffer<char> recvTiles(
      static_cast<char*>(recvBuf.get()), totalBytes, numBlocks);

  test::testPrepareTileTwoCallStaging(
      p2pHost,
      numBlocks,
      firstCallBytes,
      secondCallBytes,
      maxSignalBytes,
      2,
      threadCount);
  CUDACHECK_TEST(cudaDeviceSynchronize());
  test::testTileTwoCallSendOnly(
      p2pHost,
      sendTiles,
      numBlocks,
      firstCallBytes,
      secondCallBytes,
      maxSignalBytes,
      threadCount);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<char> hostStaging(p2pHarness.stagingBytes());
  CUDACHECK_TEST(cudaMemcpy(
      hostStaging.data(),
      p2pHarness.stagingBuffer().get(),
      p2pHarness.stagingBytes(),
      cudaMemcpyDeviceToHost));
  expectPersistentTwoCallStagingPattern(
      hostStaging,
      options.dataBufferSize,
      firstCallBytes,
      secondCallBytes,
      maxSignalBytes,
      options.pipelineDepth,
      numBlocks,
      0,
      "tile send persistent cursor staging");

  CUDACHECK_TEST(cudaMemset(recvBuf.get(), 0, totalBytes));
  test::testPrepareTileTwoCallStaging(
      p2pHost,
      numBlocks,
      firstCallBytes,
      secondCallBytes,
      maxSignalBytes,
      0,
      threadCount);
  CUDACHECK_TEST(cudaDeviceSynchronize());
  test::testTileTwoCallRecvOnly(
      p2pHost,
      recvTiles,
      numBlocks,
      firstCallBytes,
      secondCallBytes,
      maxSignalBytes,
      threadCount);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<char> hostRecv(totalBytes);
  CUDACHECK_TEST(cudaMemcpy(
      hostRecv.data(), recvBuf.get(), totalBytes, cudaMemcpyDeviceToHost));
  expectTwoCallPattern(
      hostRecv,
      firstCallBytes,
      secondCallBytes,
      numBlocks,
      0,
      "tile recv persistent cursor");
}

TEST_F(
    P2pNvlTransportTestFixture,
    TileUnalignedPayloadAdvancesAlignedProtocol) {
  const int numBlocks = 1;
  const int threadCount = 256;
  const size_t perBlockSlotSize = 256;
  const size_t firstCallBytes = 100;
  const size_t secondCallBytes = 100;
  const size_t maxSignalBytes = 64;
  const size_t tileBytes = firstCallBytes + secondCallBytes;
  const size_t totalBytes = tileBytes * numBlocks;
  const size_t firstCallStepBytes =
      protocolStepBytes(0, firstCallBytes, maxSignalBytes, perBlockSlotSize);
  const size_t expectedStep = firstCallStepBytes +
      protocolStepBytes(firstCallStepBytes,
                        secondCallBytes,
                        maxSignalBytes,
                        perBlockSlotSize);

  P2pNvlTransportOptions options{
      .dataBufferSize = perBlockSlotSize * 2 * numBlocks,
      .pipelineDepth = 2,
      .per_channel_buffer = perBlockSlotSize * 2,
      .per_channel_slot = perBlockSlotSize,
      .max_num_channels = numBlocks,
  };

  LocalTileHarness sendHarness(options, numBlocks);
  const std::vector<char> hostSend =
      makeTwoCallPatternBuffer(firstCallBytes, secondCallBytes, numBlocks, 0);
  DeviceBuffer sendBuf(totalBytes);
  CUDACHECK_TEST(cudaMemcpy(
      sendBuf.get(), hostSend.data(), totalBytes, cudaMemcpyHostToDevice));

  auto sendOnlyP2p = sendHarness.stagingDevice();
  TiledBuffer<char> sendTiles(
      static_cast<char*>(sendBuf.get()), totalBytes, numBlocks);

  test::testTileTwoCallSendOnly(
      sendOnlyP2p,
      sendTiles,
      numBlocks,
      firstCallBytes,
      secondCallBytes,
      maxSignalBytes,
      threadCount);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<char> hostStaging(sendHarness.stagingBytes());
  CUDACHECK_TEST(cudaMemcpy(
      hostStaging.data(),
      sendHarness.stagingBuffer().get(),
      sendHarness.stagingBytes(),
      cudaMemcpyDeviceToHost));
  expectPersistentTwoCallStagingPattern(
      hostStaging,
      options.dataBufferSize,
      firstCallBytes,
      secondCallBytes,
      maxSignalBytes,
      options.pipelineDepth,
      numBlocks,
      0,
      "tile send unaligned protocol staging");
  for (size_t i = firstCallBytes; i < firstCallStepBytes; ++i) {
    EXPECT_EQ(static_cast<unsigned char>(hostStaging[i]), 0)
        << "padding byte " << i
        << " between unaligned calls should stay transport-private";
  }

  std::vector<NvlChannelState> sendChannels(numBlocks);
  CUDACHECK_TEST(cudaMemcpy(
      sendChannels.data(),
      sendHarness.channelBuffer().get(),
      sendChannels.size() * sizeof(NvlChannelState),
      cudaMemcpyDeviceToHost));
  EXPECT_EQ(sendChannels[0].send_cursor, static_cast<int64_t>(expectedStep));

  LocalTileHarness loopbackHarness(options, numBlocks);
  DeviceBuffer recvBuf(totalBytes);
  CUDACHECK_TEST(cudaMemset(recvBuf.get(), 0, totalBytes));
  auto loopbackP2p = loopbackHarness.loopbackDevice();
  TiledBuffer<char> recvTiles(
      static_cast<char*>(recvBuf.get()), totalBytes, numBlocks);

  test::testTileTwoCallVariableSignalSendRecv(
      loopbackP2p,
      sendTiles,
      recvTiles,
      numBlocks,
      firstCallBytes,
      secondCallBytes,
      maxSignalBytes,
      maxSignalBytes,
      true,
      threadCount);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<char> hostRecv(totalBytes);
  CUDACHECK_TEST(cudaMemcpy(
      hostRecv.data(), recvBuf.get(), totalBytes, cudaMemcpyDeviceToHost));
  expectTwoCallPattern(
      hostRecv,
      firstCallBytes,
      secondCallBytes,
      numBlocks,
      0,
      "tile sendrecv unaligned protocol");

  std::vector<NvlChannelState> loopbackChannels(numBlocks);
  CUDACHECK_TEST(cudaMemcpy(
      loopbackChannels.data(),
      loopbackHarness.channelBuffer().get(),
      loopbackChannels.size() * sizeof(NvlChannelState),
      cudaMemcpyDeviceToHost));
  EXPECT_EQ(
      loopbackChannels[0].send_cursor, static_cast<int64_t>(expectedStep));
  EXPECT_EQ(
      loopbackChannels[0].recv_cursor, static_cast<int64_t>(expectedStep));
}

TEST_F(
    P2pNvlTransportTestFixture,
    TileForwardUnalignedPayloadAdvancesAlignedProtocol) {
  const int numBlocks = 1;
  const int threadCount = 256;
  const size_t perBlockSlotSize = 256;
  const size_t firstCallBytes = 100;
  const size_t secondCallBytes = 100;
  const size_t maxSignalBytes = 64;
  const size_t tileBytes = firstCallBytes + secondCallBytes;
  const size_t totalBytes = tileBytes * numBlocks;
  const size_t firstCallStepBytes =
      protocolStepBytes(0, firstCallBytes, maxSignalBytes, perBlockSlotSize);
  const size_t expectedStep = firstCallStepBytes +
      protocolStepBytes(firstCallStepBytes,
                        secondCallBytes,
                        maxSignalBytes,
                        perBlockSlotSize);

  P2pNvlTransportOptions options{
      .dataBufferSize = perBlockSlotSize * 2 * numBlocks,
      .pipelineDepth = 2,
      .per_channel_buffer = perBlockSlotSize * 2,
      .per_channel_slot = perBlockSlotSize,
      .max_num_channels = numBlocks,
  };
  LocalTileHarness predHarness(options, numBlocks);
  LocalTileHarness succHarness(options, numBlocks);

  auto pred = predHarness.device(predHarness.staging(), nullptr);
  auto succ = succHarness.device(nullptr, succHarness.staging());

  test::testPrepareTileTwoCallStaging(
      pred,
      numBlocks,
      firstCallBytes,
      secondCallBytes,
      maxSignalBytes,
      0,
      threadCount);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  DeviceBuffer dstBuf(totalBytes);
  CUDACHECK_TEST(cudaMemset(dstBuf.get(), 0, totalBytes));
  TiledBuffer<char> dstTiles(
      static_cast<char*>(dstBuf.get()), totalBytes, numBlocks);

  test::testTileTwoCallForward(
      pred,
      succ,
      dstTiles,
      numBlocks,
      firstCallBytes,
      secondCallBytes,
      maxSignalBytes,
      threadCount);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<char> hostDst(totalBytes);
  CUDACHECK_TEST(cudaMemcpy(
      hostDst.data(), dstBuf.get(), totalBytes, cudaMemcpyDeviceToHost));
  expectTwoCallPattern(
      hostDst,
      firstCallBytes,
      secondCallBytes,
      numBlocks,
      0,
      "tile forward unaligned protocol dst");

  std::vector<char> hostSuccStaging(succHarness.stagingBytes());
  CUDACHECK_TEST(cudaMemcpy(
      hostSuccStaging.data(),
      succHarness.stagingBuffer().get(),
      succHarness.stagingBytes(),
      cudaMemcpyDeviceToHost));
  expectPersistentTwoCallStagingPattern(
      hostSuccStaging,
      options.dataBufferSize,
      firstCallBytes,
      secondCallBytes,
      maxSignalBytes,
      options.pipelineDepth,
      numBlocks,
      0,
      "tile forward unaligned protocol successor staging");
  for (size_t i = firstCallBytes; i < firstCallStepBytes; ++i) {
    EXPECT_EQ(static_cast<unsigned char>(hostSuccStaging[i]), 0)
        << "forward padding byte " << i
        << " between unaligned calls should stay transport-private";
  }

  std::vector<NvlChannelState> predChannels(numBlocks);
  CUDACHECK_TEST(cudaMemcpy(
      predChannels.data(),
      predHarness.channelBuffer().get(),
      predChannels.size() * sizeof(NvlChannelState),
      cudaMemcpyDeviceToHost));
  EXPECT_EQ(predChannels[0].send_cursor, 0);
  EXPECT_EQ(predChannels[0].recv_cursor, static_cast<int64_t>(expectedStep));

  std::vector<NvlChannelState> succChannels(numBlocks);
  CUDACHECK_TEST(cudaMemcpy(
      succChannels.data(),
      succHarness.channelBuffer().get(),
      succChannels.size() * sizeof(NvlChannelState),
      cudaMemcpyDeviceToHost));
  EXPECT_EQ(succChannels[0].send_cursor, static_cast<int64_t>(expectedStep));
  EXPECT_EQ(succChannels[0].recv_cursor, 0);
}

TEST_F(
    P2pNvlTransportTestFixture,
    TileSendRecvChangingMaxSignalBytesWithoutBarrier) {
  const int numBlocks = 4;
  const int threadCount = 256;
  const size_t perBlockSlotSize = 64 * 1024;
  const size_t firstMaxSignalBytes = perBlockSlotSize;
  const size_t secondMaxSignalBytes = 16 * 1024;
  const size_t firstCallBytes = perBlockSlotSize;
  const size_t secondCallBytes = perBlockSlotSize / 2;
  const size_t tileBytes = firstCallBytes + secondCallBytes;
  const size_t totalBytes = tileBytes * numBlocks;

  P2pNvlTransportOptions options{
      .dataBufferSize = perBlockSlotSize * 2 * numBlocks,
      .pipelineDepth = 2,
      .per_channel_buffer = perBlockSlotSize * 2,
      .per_channel_slot = perBlockSlotSize,
      .max_num_channels = numBlocks,
  };
  LocalTileHarness p2pHarness(options, numBlocks);

  const std::vector<char> hostSend =
      makeTwoCallPatternBuffer(firstCallBytes, secondCallBytes, numBlocks, 0);
  DeviceBuffer sendBuf(totalBytes);
  DeviceBuffer recvBuf(totalBytes);

  CUDACHECK_TEST(cudaMemcpy(
      sendBuf.get(), hostSend.data(), totalBytes, cudaMemcpyHostToDevice));
  CUDACHECK_TEST(cudaMemset(recvBuf.get(), 0, totalBytes));

  auto p2pHost = p2pHarness.loopbackDevice();

  TiledBuffer<char> sendTiles(
      static_cast<char*>(sendBuf.get()), totalBytes, numBlocks);
  TiledBuffer<char> recvTiles(
      static_cast<char*>(recvBuf.get()), totalBytes, numBlocks);

  int device = 0;
  CUDACHECK_TEST(cudaGetDevice(&device));
  Timeout timeout = makeTimeout(5000, device);

  test::testTileTwoCallVariableSignalSendRecv(
      p2pHost,
      sendTiles,
      recvTiles,
      numBlocks,
      firstCallBytes,
      secondCallBytes,
      firstMaxSignalBytes,
      secondMaxSignalBytes,
      true,
      threadCount,
      timeout);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<char> hostRecv(totalBytes);
  CUDACHECK_TEST(cudaMemcpy(
      hostRecv.data(), recvBuf.get(), totalBytes, cudaMemcpyDeviceToHost));
  expectTwoCallPattern(
      hostRecv,
      firstCallBytes,
      secondCallBytes,
      numBlocks,
      0,
      "tile sendrecv changing max_signal_bytes");
}

TEST_F(
    P2pNvlTransportTestFixture,
    TileForwardChangingMaxSignalBytesWithoutBarrier) {
  const int numBlocks = 4;
  const int threadCount = 256;
  const size_t perBlockSlotSize = 64 * 1024;
  const size_t firstMaxSignalBytes = perBlockSlotSize;
  const size_t secondMaxSignalBytes = 16 * 1024;
  const size_t firstCallBytes = perBlockSlotSize;
  const size_t secondCallBytes = perBlockSlotSize / 2;
  const size_t tileBytes = firstCallBytes + secondCallBytes;
  const size_t totalBytes = tileBytes * numBlocks;

  P2pNvlTransportOptions options{
      .dataBufferSize = perBlockSlotSize * 2 * numBlocks,
      .pipelineDepth = 2,
      .per_channel_buffer = perBlockSlotSize * 2,
      .per_channel_slot = perBlockSlotSize,
      .max_num_channels = numBlocks,
  };
  LocalTileHarness predHarness(options, numBlocks);
  LocalTileHarness succHarness(options, numBlocks);

  auto pred = predHarness.device(predHarness.staging(), nullptr);
  auto succ = succHarness.device(nullptr, succHarness.staging());

  test::testPrepareTileTwoCallStaging(
      pred,
      numBlocks,
      firstCallBytes,
      secondCallBytes,
      secondMaxSignalBytes,
      0,
      threadCount);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  DeviceBuffer dstBuf(totalBytes);
  CUDACHECK_TEST(cudaMemset(dstBuf.get(), 0, totalBytes));
  TiledBuffer<char> dstTiles(
      static_cast<char*>(dstBuf.get()), totalBytes, numBlocks);

  test::testTileTwoCallVariableSignalForward(
      pred,
      succ,
      dstTiles,
      numBlocks,
      firstCallBytes,
      secondCallBytes,
      firstMaxSignalBytes,
      secondMaxSignalBytes,
      threadCount);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<char> hostDst(totalBytes);
  CUDACHECK_TEST(cudaMemcpy(
      hostDst.data(), dstBuf.get(), totalBytes, cudaMemcpyDeviceToHost));
  expectTwoCallPattern(
      hostDst,
      firstCallBytes,
      secondCallBytes,
      numBlocks,
      0,
      "tile forward changing max_signal_bytes dst");

  std::vector<char> hostSuccStaging(succHarness.stagingBytes());
  CUDACHECK_TEST(cudaMemcpy(
      hostSuccStaging.data(),
      succHarness.stagingBuffer().get(),
      succHarness.stagingBytes(),
      cudaMemcpyDeviceToHost));
  expectPersistentTwoCallStagingPattern(
      hostSuccStaging,
      options.dataBufferSize,
      firstCallBytes,
      secondCallBytes,
      secondMaxSignalBytes,
      options.pipelineDepth,
      numBlocks,
      0,
      "tile forward changing max_signal_bytes successor staging");
}

TEST_F(
    P2pNvlTransportTestFixture,
    TileSendRecvChangingLaunchedBlocksWithSameActiveBlocks) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping: requires 2 ranks, got {}", numRanks);
    return;
  }

  const int peerRank = (globalRank == 0) ? 1 : 0;
  const int activeBlocks = 4;
  const int threadCount = 256;
  const size_t perBlockSlotSize = 64 * 1024;
  const size_t maxSignalBytes = 16 * 1024;
  const size_t nBytes = 512 * 1024;

  MultiPeerNvlTransportConfig config{
      .pipelineDepth = 2,
      .maxNumChannels = activeBlocks,
      .perChannelSize = perBlockSlotSize,
  };

  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  MultiPeerNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.exchange();
  auto p2pHost = transport.buildP2pTransportDevice(peerRank);

  DeviceBuffer sendBuf(nBytes);
  DeviceBuffer recvBuf(nBytes);
  Timeout timeout;

  const std::vector<int> launchedBlocks = {2, 4, 1, 4};
  for (size_t round = 0; round < launchedBlocks.size(); ++round) {
    const int numSendBlocks = launchedBlocks[round];
    const int pattern = 0x50 + globalRank + static_cast<int>(round) * 0x20;
    const int peerPattern = 0x50 + peerRank + static_cast<int>(round) * 0x20;

    CUDACHECK_TEST(cudaMemset(sendBuf.get(), pattern, nBytes));
    CUDACHECK_TEST(cudaMemset(recvBuf.get(), 0, nBytes));

    comms::prims::TiledBuffer<char> sendTiles(
        static_cast<char*>(sendBuf.get()), nBytes, numSendBlocks);
    comms::prims::TiledBuffer<char> recvTiles(
        static_cast<char*>(recvBuf.get()), nBytes, numSendBlocks);
    std::size_t maxSignalBytesArg = maxSignalBytes;
    void* args[] = {
        &p2pHost, &sendTiles, &recvTiles, &maxSignalBytesArg, &timeout};

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    CUDACHECK_TEST(cudaLaunchKernel(
        (void*)comms::prims::benchmark::p2pTileSendRecv,
        dim3(numSendBlocks * 2),
        dim3(threadCount),
        args,
        0,
        nullptr));
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    std::vector<char> hostRecv(nBytes);
    CUDACHECK_TEST(cudaMemcpy(
        hostRecv.data(), recvBuf.get(), nBytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < nBytes; ++i) {
      EXPECT_EQ(
          static_cast<unsigned char>(hostRecv[i]),
          static_cast<unsigned char>(peerPattern))
          << "round=" << round << " launchedBlocks=" << numSendBlocks
          << " byte=" << i;
      if (static_cast<unsigned char>(hostRecv[i]) !=
          static_cast<unsigned char>(peerPattern)) {
        break;
      }
    }
  }
}

TEST_F(P2pNvlTransportTestFixture, TileSendAndForwardWaitForWrappedSubstepAck) {
  // The helper kernels launch one producer block plus one checker block.
  // Fixed-channel bounds use group.total_groups, so the test device needs two
  // channels even though only channel 0 performs the send/forward operation.
  const int numBlocks = 2;
  const int threadCount = 256;
  const size_t maxSignalBytes = 16 * 1024;
  const size_t perBlockSlotSize = 4 * maxSignalBytes;
  const size_t nbytes = 2 * maxSignalBytes;
  const size_t pipelineDepth = 1;
  const size_t pipelineSteps = 4;
  const size_t wrappedStep = pipelineSteps;
  const size_t wrappedByte = wrappedStep * maxSignalBytes;
  const size_t wrappedSubstepOffset = maxSignalBytes;
  const uint64_t initialAckValue = 0;
  const unsigned char sentinel = 0x7e;
  const unsigned char sendPattern = 0x42;
  const unsigned char forwardPattern = 0x55;

  P2pNvlTransportOptions options{
      .dataBufferSize = perBlockSlotSize * pipelineDepth * numBlocks,
      .pipelineDepth = pipelineDepth,
      .per_channel_buffer = perBlockSlotSize * pipelineDepth,
      .per_channel_slot = perBlockSlotSize,
      .max_num_channels = numBlocks,
  };
  const size_t channelBytes = sizeof(NvlChannelState) * numBlocks;

  // Build a host-side channel array with channel-0 pre-populated.
  // `localSlotFreeValue` seeds local_channels_[0].slot_free (replaces the
  // legacy local_signals[numBlocks].signal_ = headValue setup that
  // makeSignals() used to do).
  auto makeChannels = [&](uint64_t sendCursor,
                          uint64_t recvCursor,
                          uint64_t localDataReady,
                          uint64_t localSlotFreeValue) {
    std::vector<NvlChannelState> channels(numBlocks);
    channels[0].send_cursor = static_cast<int64_t>(sendCursor);
    channels[0].recv_cursor = static_cast<int64_t>(recvCursor);
    channels[0].data_ready.signal_ = localDataReady;
    channels[0].slot_free.signal_ = localSlotFreeValue;
    return channels;
  };

  {
    std::vector<char> hostSend(nbytes, static_cast<char>(sendPattern));
    std::vector<char> hostStaging(options.dataBufferSize, sentinel);
    auto hostChannels = makeChannels(wrappedByte, 0, 0, initialAckValue);
    std::vector<NvlChannelState> hostRemoteChannels(numBlocks);

    DeviceBuffer sendBuf(nbytes);
    DeviceBuffer stagingBuf(options.dataBufferSize);
    DeviceBuffer channelBuf(channelBytes);
    DeviceBuffer remoteChannelBuf(channelBytes);
    DeviceBuffer observedBuf(sizeof(int));
    const int initialObserved = -1;

    CUDACHECK_TEST(cudaMemcpy(
        sendBuf.get(), hostSend.data(), nbytes, cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemcpy(
        stagingBuf.get(),
        hostStaging.data(),
        hostStaging.size(),
        cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemcpy(
        channelBuf.get(),
        hostChannels.data(),
        channelBytes,
        cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemcpy(
        remoteChannelBuf.get(),
        hostRemoteChannels.data(),
        channelBytes,
        cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemcpy(
        observedBuf.get(),
        &initialObserved,
        sizeof(initialObserved),
        cudaMemcpyHostToDevice));

    auto p2pHost = makeLocalTileDevice(
        options,
        numBlocks,
        static_cast<char*>(stagingBuf.get()),
        static_cast<char*>(stagingBuf.get()),
        static_cast<NvlChannelState*>(channelBuf.get()),
        static_cast<NvlChannelState*>(remoteChannelBuf.get()));

    test::testTileSendWaitsForWrappedSubstepAck(
        p2pHost,
        static_cast<const char*>(sendBuf.get()),
        nbytes,
        maxSignalBytes,
        sentinel,
        static_cast<int*>(observedBuf.get()),
        threadCount);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    int observed = -1;
    CUDACHECK_TEST(cudaMemcpy(
        &observed,
        observedBuf.get(),
        sizeof(observed),
        cudaMemcpyDeviceToHost));
    EXPECT_EQ(observed, 0)
        << "tile send reused a wrapped nonzero substep before the receiver ACK";

    std::vector<char> finalStaging(options.dataBufferSize);
    CUDACHECK_TEST(cudaMemcpy(
        finalStaging.data(),
        stagingBuf.get(),
        finalStaging.size(),
        cudaMemcpyDeviceToHost));
    EXPECT_EQ(
        static_cast<unsigned char>(finalStaging[wrappedSubstepOffset]),
        sendPattern);
  }

  {
    std::vector<char> hostPredStaging(options.dataBufferSize, 0);
    std::fill(
        hostPredStaging.begin(),
        hostPredStaging.begin() + nbytes,
        static_cast<char>(forwardPattern));
    std::vector<char> hostSuccStaging(options.dataBufferSize, sentinel);

    // pred is the recv side: data_ready[0] = nbytes simulates peer sender
    // having already signaled the full message ready.
    auto hostPredChannels = makeChannels(0, 0, nbytes, 0);
    std::vector<NvlChannelState> hostPredRemoteChannels(numBlocks);

    // succ is the send side: send_cursor[0] pre-wrapped; slot_free[0]
    // pre-seeded at initialAckValue (replaces makeSignals's headValue setup).
    auto hostSuccChannels = makeChannels(wrappedByte, 0, 0, initialAckValue);
    std::vector<NvlChannelState> hostSuccRemoteChannels(numBlocks);

    DeviceBuffer predStagingBuf(options.dataBufferSize);
    DeviceBuffer succStagingBuf(options.dataBufferSize);
    DeviceBuffer dstBuf(nbytes);
    DeviceBuffer predChannelBuf(channelBytes);
    DeviceBuffer predRemoteChannelBuf(channelBytes);
    DeviceBuffer succChannelBuf(channelBytes);
    DeviceBuffer succRemoteChannelBuf(channelBytes);
    DeviceBuffer observedBuf(sizeof(int));
    const int initialObserved = -1;

    CUDACHECK_TEST(cudaMemcpy(
        predStagingBuf.get(),
        hostPredStaging.data(),
        hostPredStaging.size(),
        cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemcpy(
        succStagingBuf.get(),
        hostSuccStaging.data(),
        hostSuccStaging.size(),
        cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemset(dstBuf.get(), 0, nbytes));
    CUDACHECK_TEST(cudaMemcpy(
        predChannelBuf.get(),
        hostPredChannels.data(),
        channelBytes,
        cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemcpy(
        predRemoteChannelBuf.get(),
        hostPredRemoteChannels.data(),
        channelBytes,
        cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemcpy(
        succChannelBuf.get(),
        hostSuccChannels.data(),
        channelBytes,
        cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemcpy(
        succRemoteChannelBuf.get(),
        hostSuccRemoteChannels.data(),
        channelBytes,
        cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaMemcpy(
        observedBuf.get(),
        &initialObserved,
        sizeof(initialObserved),
        cudaMemcpyHostToDevice));

    auto pred = makeLocalTileDevice(
        options,
        numBlocks,
        static_cast<char*>(predStagingBuf.get()),
        nullptr,
        static_cast<NvlChannelState*>(predChannelBuf.get()),
        static_cast<NvlChannelState*>(predRemoteChannelBuf.get()));
    auto succ = makeLocalTileDevice(
        options,
        numBlocks,
        nullptr,
        static_cast<char*>(succStagingBuf.get()),
        static_cast<NvlChannelState*>(succChannelBuf.get()),
        static_cast<NvlChannelState*>(succRemoteChannelBuf.get()));

    test::testTileForwardWaitsForWrappedSubstepAck(
        pred,
        succ,
        static_cast<char*>(dstBuf.get()),
        nbytes,
        maxSignalBytes,
        sentinel,
        static_cast<int*>(observedBuf.get()),
        threadCount);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    int observed = -1;
    CUDACHECK_TEST(cudaMemcpy(
        &observed,
        observedBuf.get(),
        sizeof(observed),
        cudaMemcpyDeviceToHost));
    EXPECT_EQ(observed, 0)
        << "tile forward reused a wrapped nonzero substep before successor ACK";

    std::vector<char> finalSuccStaging(options.dataBufferSize);
    std::vector<char> finalDst(nbytes);
    CUDACHECK_TEST(cudaMemcpy(
        finalSuccStaging.data(),
        succStagingBuf.get(),
        finalSuccStaging.size(),
        cudaMemcpyDeviceToHost));
    CUDACHECK_TEST(cudaMemcpy(
        finalDst.data(),
        dstBuf.get(),
        finalDst.size(),
        cudaMemcpyDeviceToHost));
    EXPECT_EQ(
        static_cast<unsigned char>(finalSuccStaging[wrappedSubstepOffset]),
        forwardPattern);
    EXPECT_EQ(static_cast<unsigned char>(finalDst[0]), forwardPattern);
  }
}

// Test multi-call with different message sizes per call
TEST_F(P2pNvlTransportTestFixture, TileSendRecvMultiCallDifferentSizes) {
  if (numRanks != 2) {
    return;
  }
  int peerRank = (globalRank == 0) ? 1 : 0;

  const int numSendBlocks = 4;
  auto config = makeNvlConfig(2 * 1024 * 1024, 2, numSendBlocks);

  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  MultiPeerNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.exchange();
  auto p2pHost = transport.buildP2pTransportDevice(peerRank);

  // Different sizes for each call
  std::vector<size_t> sizes = {
      2 * 1024 * 1024, // 2MB
      8 * 1024 * 1024, // 8MB
      1 * 1024 * 1024, // 1MB (smaller than first)
      16 * 1024 * 1024, // 16MB
  };

  dim3 grid(numSendBlocks * 2);
  dim3 block(256);
  Timeout timeout;

  for (size_t callIdx = 0; callIdx < sizes.size(); callIdx++) {
    size_t nBytes = sizes[callIdx];
    const int pattern = 0x30 + globalRank + static_cast<int>(callIdx) * 0x10;
    const int peerPattern = 0x30 + peerRank + static_cast<int>(callIdx) * 0x10;

    DeviceBuffer sendBuf(nBytes);
    DeviceBuffer recvBuf(nBytes);
    CUDACHECK_TEST(cudaMemset(sendBuf.get(), pattern, nBytes));
    CUDACHECK_TEST(cudaMemset(recvBuf.get(), 0, nBytes));

    comms::prims::TiledBuffer<char> sendTiles(
        static_cast<char*>(sendBuf.get()), nBytes, numSendBlocks);
    comms::prims::TiledBuffer<char> recvTiles(
        static_cast<char*>(recvBuf.get()), nBytes, numSendBlocks);
    std::size_t maxSignalBytes = 0;
    void* args[] = {
        &p2pHost, &sendTiles, &recvTiles, &maxSignalBytes, &timeout};

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    CUDACHECK_TEST(cudaLaunchKernel(
        (void*)comms::prims::benchmark::p2pTileSendRecv,
        grid,
        block,
        args,
        0,
        nullptr));
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    std::vector<char> hostBuf(nBytes);
    CUDACHECK_TEST(cudaMemcpy(
        hostBuf.data(), recvBuf.get(), nBytes, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < nBytes; i++) {
      EXPECT_EQ(
          static_cast<unsigned char>(hostBuf[i]),
          static_cast<unsigned char>(peerPattern))
          << "Call " << callIdx << " (size=" << nBytes << "): Mismatch at byte "
          << i;
      if (static_cast<unsigned char>(hostBuf[i]) !=
          static_cast<unsigned char>(peerPattern)) {
        break;
      }
    }
  }
}

// Test partial tiles (nbytes not evenly divisible by numBlocks)
TEST_F(P2pNvlTransportTestFixture, TileSendRecvPartialTiles) {
  if (numRanks != 2) {
    return;
  }
  auto bs = std::make_shared<meta::comms::MpiBootstrap>();

  // nBytes not divisible by numBlocks — last block gets fewer bytes
  runTileTest(
      globalRank,
      numRanks,
      bs,
      1000000,
      8 * 1024 * 1024,
      8 * 1024 * 1024,
      2,
      4,
      1);
  runTileTest(
      globalRank,
      numRanks,
      bs,
      3000000,
      8 * 1024 * 1024,
      8 * 1024 * 1024,
      2,
      8,
      1);

  // Odd sizes
  runTileTest(
      globalRank,
      numRanks,
      bs,
      7 * 1024 * 1024 + 12345,
      8 * 1024 * 1024,
      8 * 1024 * 1024,
      2,
      4,
      1);
}

// Test with different staging buffer sizes
TEST_F(P2pNvlTransportTestFixture, TileSendRecvStagingSizes) {
  if (numRanks != 2) {
    return;
  }
  auto bs = std::make_shared<meta::comms::MpiBootstrap>();
  const size_t nBytes = 16 * 1024 * 1024;

  // 4MB staging
  runTileTest(
      globalRank,
      numRanks,
      bs,
      nBytes,
      4 * 1024 * 1024,
      4 * 1024 * 1024,
      2,
      8,
      1);

  // 8MB staging
  runTileTest(
      globalRank,
      numRanks,
      bs,
      nBytes,
      8 * 1024 * 1024,
      8 * 1024 * 1024,
      2,
      8,
      1);

  // 16MB staging
  runTileTest(
      globalRank,
      numRanks,
      bs,
      nBytes,
      16 * 1024 * 1024,
      16 * 1024 * 1024,
      2,
      8,
      1);
}

// =============================================================================
// P2pNvlTransportDevice::put() Tests
// =============================================================================
// Tests for the one-sided put() API that writes directly to peer memory
// via NVLink without using staging buffers.

// Helper to run a write() test with verification
void runPutTest(
    int globalRank,
    P2pNvlTransportDevice* p2p,
    char* localSrc,
    char* remoteDst,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    const std::string& testName) {
  const char testValue = 0x42;
  const uint64_t signal_id = 0;

  if (globalRank == 0) {
    // Rank 0: Initialize source buffer and call write()
    CUDACHECK_TEST(cudaMemset(localSrc, testValue, nbytes));
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Write data to peer's buffer
    test::testPutWithSignal(
        p2p, remoteDst, localSrc, signal_id, nbytes, numBlocks, blockSize);

    CUDACHECK_TEST(cudaDeviceSynchronize());
  } else {
    // Rank 1: Clear destination buffer and verify after write()
    CUDACHECK_TEST(cudaMemset(localSrc, 0, nbytes));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    test::testWait(p2p, CmpOp::CMP_GE, signal_id, nbytes, numBlocks, blockSize);

    CUDACHECK_TEST(cudaDeviceSynchronize());

    // Verify the data was written correctly
    std::vector<char> hostBuffer(nbytes);
    CUDACHECK_TEST(cudaMemcpy(
        hostBuffer.data(), localSrc, nbytes, cudaMemcpyDeviceToHost));

    int errorCount = 0;
    for (size_t i = 0; i < nbytes; i++) {
      if (hostBuffer[i] != testValue) {
        ++errorCount;
        if (errorCount <= 5) {
          XLOGF(
              ERR,
              "{}: Mismatch at index {}: expected 0x{:02x}, got 0x{:02x}",
              testName,
              i,
              static_cast<unsigned char>(testValue),
              static_cast<unsigned char>(hostBuffer[i]));
        }
      }
    }

    ASSERT_EQ(errorCount, 0) << testName << " found " << errorCount
                             << " errors out of " << nbytes << " bytes";
  }
}

// Basic write() test with aligned pointers
TEST_F(P2pNvlTransportTestFixture, PutBasic) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const size_t nbytes = 1024 * 1024; // 1MB
  auto config = makeNvlConfig(nbytes, 1);

  TransportTestHelper helper(globalRank, numRanks, localRank, config);
  auto p2p = helper.getDevicePtr();
  auto& p2pHost = helper.getHostDevice();

  // Get remote destination (peer's local data buffer)
  char* localSrc = p2pHost.getLocalState().dataBuffer;
  char* remoteDst = p2pHost.getRemoteState().dataBuffer;

  runPutTest(globalRank, p2p, localSrc, remoteDst, nbytes, 4, 128, "PutBasic");

  XLOGF(INFO, "Rank {}: PutBasic test completed", globalRank);
}

// Parameterized test for write() with various transfer sizes
struct PutTransferSizeParams {
  size_t nbytes;
  std::string name;
};

class PutTransferSizeTestFixture
    : public MpiBaseTestFixture,
      public ::testing::WithParamInterface<PutTransferSizeParams> {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }
};

TEST_P(PutTransferSizeTestFixture, Put) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const auto& params = GetParam();
  XLOGF(
      INFO,
      "Running write transfer size test: {} (nbytes={})",
      params.name,
      params.nbytes);

  auto config = makeNvlConfig(params.nbytes, 1);

  TransportTestHelper helper(globalRank, numRanks, localRank, config);
  auto p2p = helper.getDevicePtr();
  auto& p2pHost = helper.getHostDevice();

  char* localSrc = p2pHost.getLocalState().dataBuffer;
  char* remoteDst = p2pHost.getRemoteState().dataBuffer;

  runPutTest(
      globalRank, p2p, localSrc, remoteDst, params.nbytes, 4, 128, params.name);

  XLOGF(
      INFO,
      "Rank {}: Put transfer size test '{}' completed",
      globalRank,
      params.name);
}

std::string putTransferSizeParamName(
    const ::testing::TestParamInfo<PutTransferSizeParams>& info) {
  return info.param.name;
}

INSTANTIATE_TEST_SUITE_P(
    PutTransferSizeVariations,
    PutTransferSizeTestFixture,
    ::testing::Values(
        // Small sizes (smaller than vector size of 16 bytes)
        PutTransferSizeParams{.nbytes = 1, .name = "Put_1Byte"},
        PutTransferSizeParams{.nbytes = 7, .name = "Put_7Bytes"},
        PutTransferSizeParams{.nbytes = 15, .name = "Put_15Bytes"},
        // Around vector size boundary
        PutTransferSizeParams{.nbytes = 16, .name = "Put_16Bytes"},
        PutTransferSizeParams{.nbytes = 17, .name = "Put_17Bytes"},
        PutTransferSizeParams{.nbytes = 31, .name = "Put_31Bytes"},
        PutTransferSizeParams{.nbytes = 32, .name = "Put_32Bytes"},
        // Non-aligned sizes
        PutTransferSizeParams{.nbytes = 100, .name = "Put_100Bytes"},
        PutTransferSizeParams{.nbytes = 1000, .name = "Put_1000Bytes"},
        PutTransferSizeParams{.nbytes = 4097, .name = "Put_4097Bytes"},
        // Aligned sizes
        PutTransferSizeParams{.nbytes = 1024, .name = "Put_1KB"},
        PutTransferSizeParams{.nbytes = 64 * 1024, .name = "Put_64KB"},
        PutTransferSizeParams{.nbytes = 256 * 1024, .name = "Put_256KB"},
        PutTransferSizeParams{.nbytes = 1024 * 1024, .name = "Put_1MB"},
        // Large sizes
        PutTransferSizeParams{.nbytes = 4 * 1024 * 1024, .name = "Put_4MB"},
        PutTransferSizeParams{.nbytes = 16 * 1024 * 1024, .name = "Put_16MB"}),
    putTransferSizeParamName);

// Parameterized test for write() with unaligned pointers
struct PutUnalignedParams {
  size_t srcOffset; // Offset from 16-byte alignment for source
  size_t dstOffset; // Offset from 16-byte alignment for destination
  size_t nbytes;
  std::string name;
};

class PutUnalignedTestFixture
    : public MpiBaseTestFixture,
      public ::testing::WithParamInterface<PutUnalignedParams> {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }
};

TEST_P(PutUnalignedTestFixture, Put) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const auto& params = GetParam();
  XLOGF(
      INFO,
      "Running write unaligned test: {} (srcOffset={}, dstOffset={}, nbytes={})",
      params.name,
      params.srcOffset,
      params.dstOffset,
      params.nbytes);

  // Allocate larger staging buffers to accommodate offsets
  const size_t dataBufferSize =
      params.nbytes + std::max(params.srcOffset, params.dstOffset);
  auto config = makeNvlConfig(dataBufferSize, 4);

  TransportTestHelper helper(globalRank, numRanks, localRank, config);
  auto p2p = helper.getDevicePtr();
  auto& p2pHost = helper.getHostDevice();

  // Get remote and local destination with offset applied
  char* localSrc = p2pHost.getLocalState().dataBuffer;
  char* remoteDst = p2pHost.getRemoteState().dataBuffer;
  if (globalRank == 0) {
    localSrc += params.srcOffset;
    remoteDst += params.dstOffset;
  } else {
    localSrc += params.dstOffset;
    remoteDst += params.srcOffset;
  }

  runPutTest(
      globalRank, p2p, localSrc, remoteDst, params.nbytes, 4, 128, params.name);

  XLOGF(
      INFO,
      "Rank {}: Put unaligned test '{}' completed",
      globalRank,
      params.name);
}

std::string putUnalignedParamName(
    const ::testing::TestParamInfo<PutUnalignedParams>& info) {
  return info.param.name;
}

INSTANTIATE_TEST_SUITE_P(
    PutUnalignedVariations,
    PutUnalignedTestFixture,
    ::testing::Values(
        // Same misalignment (can use vectorized copy after aligning)
        PutUnalignedParams{
            .srcOffset = 1,
            .dstOffset = 1,
            .nbytes = 1024,
            .name = "SameMisalign_1"},
        PutUnalignedParams{
            .srcOffset = 7,
            .dstOffset = 7,
            .nbytes = 1024,
            .name = "SameMisalign_7"},
        PutUnalignedParams{
            .srcOffset = 8,
            .dstOffset = 8,
            .nbytes = 1024,
            .name = "SameMisalign_8"},
        PutUnalignedParams{
            .srcOffset = 13,
            .dstOffset = 13,
            .nbytes = 1024,
            .name = "SameMisalign_13"},
        PutUnalignedParams{
            .srcOffset = 15,
            .dstOffset = 15,
            .nbytes = 1024,
            .name = "SameMisalign_15"},
        // Different misalignment (fallback to byte-by-byte)
        PutUnalignedParams{
            .srcOffset = 1,
            .dstOffset = 3,
            .nbytes = 1024,
            .name = "DiffMisalign_1_3"},
        PutUnalignedParams{
            .srcOffset = 0,
            .dstOffset = 7,
            .nbytes = 1024,
            .name = "DiffMisalign_0_7"},
        PutUnalignedParams{
            .srcOffset = 5,
            .dstOffset = 0,
            .nbytes = 1024,
            .name = "DiffMisalign_5_0"},
        PutUnalignedParams{
            .srcOffset = 4,
            .dstOffset = 8,
            .nbytes = 1024,
            .name = "DiffMisalign_4_8"},
        // Larger transfers with misalignment
        PutUnalignedParams{
            .srcOffset = 3,
            .dstOffset = 3,
            .nbytes = 64 * 1024,
            .name = "SameMisalign_3_64KB"},
        PutUnalignedParams{
            .srcOffset = 5,
            .dstOffset = 11,
            .nbytes = 64 * 1024,
            .name = "DiffMisalign_5_11_64KB"},
        // Small transfers with misalignment
        PutUnalignedParams{
            .srcOffset = 7,
            .dstOffset = 7,
            .nbytes = 100,
            .name = "SameMisalign_7_100Bytes"},
        PutUnalignedParams{
            .srcOffset = 1,
            .dstOffset = 9,
            .nbytes = 100,
            .name = "DiffMisalign_1_9_100Bytes"}),
    putUnalignedParamName);

// Regression test for multi-chunk accumulation bug
// Tests that put() correctly handles multiple chunks per thread group.
// The bug caused chunkBytes to accumulate across iterations, leading to
// buffer overflows and data corruption.
TEST_F(P2pNvlTransportTestFixture, PutMultiChunkAccumulationRegression) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  // Parameters chosen to trigger multi-chunk per group:
  // numBlocks=4, blockSize=128 -> total_groups=16
  // nbytes=257 -> numChunks=17, so groups process 2 chunks each
  const size_t nbytes = 257;
  const size_t paddedSize = nbytes + 64; // Extra space to detect overflow
  const char sentinelValue = static_cast<char>(0xDE);

  auto config = makeNvlConfig(paddedSize, 1);

  TransportTestHelper helper(globalRank, numRanks, localRank, config);
  auto* p2p = helper.getDevicePtr(); // device pointer for kernel calls
  auto& p2pHost = helper.getHostDevice(); // host reference for buffer access
  char* localSrc =
      p2pHost.getLocalState().dataBuffer; // dataBuffer is a device ptr
  char* remoteDst = p2pHost.getRemoteState().dataBuffer;

  const uint64_t signal_id = 0;

  if (globalRank == 0) {
    // Fill with sequential pattern [0, 1, 2, ..., nbytes-1]
    std::vector<char> pattern(paddedSize, sentinelValue);
    for (size_t i = 0; i < nbytes; ++i) {
      pattern[i] = static_cast<char>(i % 256);
    }
    CUDACHECK_TEST(cudaMemcpy(
        localSrc, pattern.data(), paddedSize, cudaMemcpyHostToDevice));
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    test::testPutWithSignal(
        p2p, remoteDst, localSrc, signal_id, nbytes, 4, 128);
    CUDACHECK_TEST(cudaDeviceSynchronize());
  } else {
    // Fill destination with sentinel to detect any writes
    CUDACHECK_TEST(cudaMemset(localSrc, sentinelValue, paddedSize));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    test::testWait(p2p, CmpOp::CMP_GE, signal_id, nbytes, 4, 128);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    // Verify sequential pattern
    std::vector<char> result(paddedSize);
    CUDACHECK_TEST(cudaMemcpy(
        result.data(), localSrc, paddedSize, cudaMemcpyDeviceToHost));

    // Check data bytes have correct pattern
    for (size_t i = 0; i < nbytes; ++i) {
      EXPECT_EQ(
          static_cast<unsigned char>(result[i]),
          static_cast<unsigned char>(i % 256))
          << "Data mismatch at byte " << i << " - accumulation bug detected";
    }

    // Check sentinel bytes are untouched (no overflow)
    for (size_t i = nbytes; i < paddedSize; ++i) {
      EXPECT_EQ(
          static_cast<unsigned char>(result[i]),
          static_cast<unsigned char>(sentinelValue))
          << "Buffer overflow detected at byte " << i;
    }
  }
}

// =============================================================================
// LL128 Buffer Wiring Tests
// Verify MultiPeerNvlTransport correctly wires LL128 buffer pointers into
// P2pNvlTransportDevice handles.
// =============================================================================

TEST_F(P2pNvlTransportTestFixture, Ll128BufferWiring_Enabled) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;

  auto config = makeNvlConfig(4096, 2);
  config.ll128BufferSize = ll128_buffer_size(4096);

  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  MultiPeerNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.exchange();

  auto p2p = transport.buildP2pTransportDevice(peerRank);

  ASSERT_NE(p2p.getLocalState().ll128Buffer, nullptr)
      << "Rank " << globalRank
      << ": localState.ll128Buffer should be non-null when ll128BufferSize > 0";
  ASSERT_NE(p2p.getRemoteState().ll128Buffer, nullptr)
      << "Rank " << globalRank
      << ": remoteState.ll128Buffer should be non-null when ll128BufferSize > 0";
  ASSERT_NE(p2p.getLocalState().ll128Buffer, p2p.getRemoteState().ll128Buffer)
      << "Rank " << globalRank
      << ": local and remote ll128Buffer should point to different ranks' buffers";

  XLOGF(INFO, "Rank {}: Ll128BufferWiring_Enabled test completed", globalRank);
}

TEST_F(P2pNvlTransportTestFixture, Ll128BufferWiring_Disabled) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;

  auto config = makeNvlConfig(4096, 2);

  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  MultiPeerNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.exchange();

  auto p2p = transport.buildP2pTransportDevice(peerRank);

  ASSERT_EQ(p2p.getLocalState().ll128Buffer, nullptr)
      << "Rank " << globalRank
      << ": localState.ll128Buffer should be null when ll128BufferSize == 0";
  ASSERT_EQ(p2p.getRemoteState().ll128Buffer, nullptr)
      << "Rank " << globalRank
      << ": remoteState.ll128Buffer should be null when ll128BufferSize == 0";
  ASSERT_EQ(p2p.getLocalState().llBuffer, nullptr)
      << "Rank " << globalRank
      << ": localState.llBuffer should be null when llBufferSize == 0";
  ASSERT_EQ(p2p.getRemoteState().llBuffer, nullptr)
      << "Rank " << globalRank
      << ": remoteState.llBuffer should be null when llBufferSize == 0";

  XLOGF(INFO, "Rank {}: Ll128BufferWiring_Disabled test completed", globalRank);
}

// =============================================================================
// Dynamic block count tests
// =============================================================================
// Verify that changing numBlocks between send/recv rounds works
// correctly with the maxBlocks layout and host-side barrier.

TEST_F(P2pNvlTransportTestFixture, TileSendRecvDynamicBlockCount) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping: requires 2 ranks, got {}", numRanks);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;
  constexpr int maxBlocks = 32;

  MultiPeerNvlTransportConfig config{
      .pipelineDepth = 2,
      .p2pBarrierCount = static_cast<std::size_t>(maxBlocks),
      .maxNumChannels = maxBlocks,
      .perChannelSize = 256 * 1024,
  };

  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  MultiPeerNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.exchange();
  auto p2pHost = transport.buildP2pTransportDevice(peerRank);

  // Sequence of rounds with different block counts and message sizes
  struct Round {
    int numBlocks;
    size_t nBytes;
  };
  std::vector<Round> rounds = {
      {16, 8 * 1024 * 1024}, // 16 blocks, 8MB
      {32, 16 * 1024 * 1024}, // 32 blocks, 16MB (increase blocks)
      {8, 4 * 1024 * 1024}, // 8 blocks, 4MB (decrease blocks)
      {16, 8 * 1024 * 1024}, // 16 blocks again, 8MB
      {32, 32 * 1024 * 1024}, // 32 blocks, 32MB
      {4, 1 * 1024 * 1024}, // 4 blocks, 1MB (small)
      {16, 64 * 1024 * 1024}, // 16 blocks, 64MB (large)
  };

  Timeout timeout;
  int prevBlocks = 0;

  for (size_t roundIdx = 0; roundIdx < rounds.size(); roundIdx++) {
    int numBlocks = rounds[roundIdx].numBlocks;
    size_t nBytes = rounds[roundIdx].nBytes;
    int totalBlocks = numBlocks * 2;

    // Unique pattern per round
    const int pattern = 0x10 + globalRank + static_cast<int>(roundIdx) * 0x20;
    const int peerPattern = 0x10 + peerRank + static_cast<int>(roundIdx) * 0x20;

    DeviceBuffer sendBuf(nBytes);
    DeviceBuffer recvBuf(nBytes);
    CUDACHECK_TEST(cudaMemset(sendBuf.get(), pattern, nBytes));
    CUDACHECK_TEST(cudaMemset(recvBuf.get(), 0, nBytes));

    comms::prims::TiledBuffer<char> sendTiles(
        static_cast<char*>(sendBuf.get()), nBytes, numBlocks);
    comms::prims::TiledBuffer<char> recvTiles(
        static_cast<char*>(recvBuf.get()), nBytes, numBlocks);

    bool needsBarrier = (prevBlocks != 0 && prevBlocks != numBlocks);
    void* args[] = {&p2pHost, &sendTiles, &recvTiles, &needsBarrier, &timeout};

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    CUDACHECK_TEST(cudaLaunchKernel(
        (void*)comms::prims::benchmark::p2pTileSendRecvDynamic,
        dim3(totalBlocks),
        dim3(256),
        args,
        0,
        nullptr));
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Verify received data
    std::vector<char> hostBuf(nBytes);
    CUDACHECK_TEST(cudaMemcpy(
        hostBuf.data(), recvBuf.get(), nBytes, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < nBytes; i++) {
      EXPECT_EQ(
          static_cast<unsigned char>(hostBuf[i]),
          static_cast<unsigned char>(peerPattern))
          << "Round " << roundIdx << " (blocks=" << numBlocks
          << ", size=" << nBytes << "): Mismatch at byte " << i;
      if (static_cast<unsigned char>(hostBuf[i]) !=
          static_cast<unsigned char>(peerPattern)) {
        break;
      }
    }

    prevBlocks = numBlocks;
  }
}

// =============================================================================
// Tile-style forward() tests — 2-rank ring topology
// =============================================================================
// Topology: Rank 0 ──send──▶ Rank 1 ──forward──▶ Rank 0
// - Rank 0 launches p2pTileSendRecv (concurrent send + recv on a single
//   transport, partitioned into 2 * numSendBlocks total blocks).
// - Rank 1 launches p2pTileForward with p2p_pred == p2p_succ == the single
//   transport to rank 0; each of numSendBlocks blocks calls forward(),
//   reading rank 0's incoming data from local staging and dual-writing to
//   rank 1's local output and rank 0's recv staging.
//
// Signal/step slot disjointness:
//   On EACH transport, sender slots are [0, max_groups) and receiver slots
//   are [max_groups, 2 * max_groups). Rank 0's send/recv use disjoint
//   halves; rank 1's forward.recv uses the receiver half on `this` while
//   forward.send uses the sender half on `successor` — with this == succ
//   they still touch disjoint halves.
//
// Verification: Rank 0's recv buffer and Rank 1's local forward output
// should both equal the original send pattern.

// Helper: run a single tile forward round and verify both ranks' data.
// dstOffset shifts rank 1's local forward output buffer to test unaligned
// user-buffer addresses (the staging buffers are always cudaMalloc-aligned,
// so this is the only user-facing pointer we can misalign).
static void runTileForwardTest(
    int globalRank,
    int numRanks,
    const std::shared_ptr<meta::comms::MpiBootstrap>& bootstrap,
    size_t nBytes,
    size_t perChannelSize,
    size_t chunkSize,
    size_t pipelineDepth,
    int numSendBlocks,
    int nIters = 1,
    int threadCount = 256,
    size_t dstOffset = 0) {
  if (numRanks != 2) {
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;

  auto config = makeNvlConfig(perChannelSize, pipelineDepth, numSendBlocks);

  MultiPeerNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.exchange();
  auto p2pHost = transport.buildP2pTransportDevice(peerRank);

  Timeout timeout;

  for (int iter = 0; iter < nIters; iter++) {
    const int pattern = 0x10 + iter * 0x20;

    DeviceBuffer srcBuf(nBytes); // rank 0 source
    DeviceBuffer recvR0Buf(nBytes); // rank 0 recv (forwarded back)
    // Allocate rank 1's forward output with extra padding for dstOffset.
    DeviceBuffer fwdR1Buf(nBytes + dstOffset);

    if (globalRank == 0) {
      CUDACHECK_TEST(cudaMemset(srcBuf.get(), pattern, nBytes));
      CUDACHECK_TEST(cudaMemset(recvR0Buf.get(), 0, nBytes));
    } else {
      CUDACHECK_TEST(cudaMemset(fwdR1Buf.get(), 0, nBytes + dstOffset));
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    if (globalRank == 0) {
      // Rank 0: tile send src + tile recv into recvR0 (single kernel,
      // 2 * numSendBlocks total blocks via role partition).
      comms::prims::TiledBuffer<char> sendTiles(
          static_cast<char*>(srcBuf.get()), nBytes, numSendBlocks);
      comms::prims::TiledBuffer<char> recvTiles(
          static_cast<char*>(recvR0Buf.get()), nBytes, numSendBlocks);
      std::size_t maxSignalBytes = 0;
      void* args[] = {
          &p2pHost, &sendTiles, &recvTiles, &maxSignalBytes, &timeout};

      CUDACHECK_TEST(cudaLaunchKernel(
          (void*)comms::prims::benchmark::p2pTileSendRecv,
          dim3(numSendBlocks * 2),
          dim3(threadCount),
          args,
          0,
          nullptr));
    } else {
      // Rank 1: tile forward through single transport (pred == succ).
      // Apply dstOffset to the user-facing output buffer.
      char* dstPtr = static_cast<char*>(fwdR1Buf.get()) + dstOffset;
      comms::prims::TiledBuffer<char> dstTiles(dstPtr, nBytes, numSendBlocks);
      std::size_t maxSignalBytes = 0;
      void* args[] = {&p2pHost, &p2pHost, &dstTiles, &maxSignalBytes, &timeout};

      CUDACHECK_TEST(cudaLaunchKernel(
          (void*)comms::prims::benchmark::p2pTileForward,
          dim3(numSendBlocks),
          dim3(threadCount),
          args,
          0,
          nullptr));
    }

    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Verify rank 0's recv data matches the original send pattern.
    if (globalRank == 0) {
      std::vector<char> hostBuf(nBytes);
      CUDACHECK_TEST(cudaMemcpy(
          hostBuf.data(), recvR0Buf.get(), nBytes, cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < nBytes; i++) {
        EXPECT_EQ(
            static_cast<unsigned char>(hostBuf[i]),
            static_cast<unsigned char>(pattern))
            << "Iter " << iter << " (rank 0 recv): Mismatch at byte " << i
            << " (nBytes=" << nBytes << ", blocks=" << numSendBlocks
            << ", perChannelSize=" << perChannelSize << ", chunk=" << chunkSize
            << ", pd=" << pipelineDepth << ", dstOffset=" << dstOffset << ")";
        if (static_cast<unsigned char>(hostBuf[i]) !=
            static_cast<unsigned char>(pattern)) {
          return; // stop on first failure
        }
      }
    } else {
      // Read the entire padded buffer so we can also check the prefix bytes
      // (before dstOffset) were not clobbered.
      std::vector<char> hostBuf(nBytes + dstOffset);
      CUDACHECK_TEST(cudaMemcpy(
          hostBuf.data(),
          fwdR1Buf.get(),
          nBytes + dstOffset,
          cudaMemcpyDeviceToHost));
      // Bytes before the offset should still be zero.
      for (size_t i = 0; i < dstOffset; i++) {
        EXPECT_EQ(static_cast<unsigned char>(hostBuf[i]), 0u)
            << "Iter " << iter << " (rank 1 prefix clobbered): byte " << i
            << " (dstOffset=" << dstOffset << ")";
        if (static_cast<unsigned char>(hostBuf[i]) != 0u) {
          return;
        }
      }
      // Payload bytes at [dstOffset, dstOffset + nBytes) should equal pattern.
      for (size_t i = 0; i < nBytes; i++) {
        EXPECT_EQ(
            static_cast<unsigned char>(hostBuf[dstOffset + i]),
            static_cast<unsigned char>(pattern))
            << "Iter " << iter << " (rank 1 forward dst): Mismatch at byte "
            << i << " (nBytes=" << nBytes << ", blocks=" << numSendBlocks
            << ", perChannelSize=" << perChannelSize << ", chunk=" << chunkSize
            << ", pd=" << pipelineDepth << ", dstOffset=" << dstOffset << ")";
        if (static_cast<unsigned char>(hostBuf[dstOffset + i]) !=
            static_cast<unsigned char>(pattern)) {
          return;
        }
      }
    }
  }
}

// Basic single-call test
TEST_F(P2pNvlTransportTestFixture, TileForwardBasic) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping: requires 2 ranks, got {}", numRanks);
    return;
  }
  auto bs = std::make_shared<meta::comms::MpiBootstrap>();

  // 8MB transfer, 8MB slot (single-step), 4 blocks
  runTileForwardTest(
      globalRank,
      numRanks,
      bs,
      8 * 1024 * 1024,
      8 * 1024 * 1024,
      8 * 1024 * 1024,
      2,
      4,
      1);
}

// Various message sizes
TEST_F(P2pNvlTransportTestFixture, TileForwardMessageSizes) {
  if (numRanks != 2) {
    return;
  }
  auto bs = std::make_shared<meta::comms::MpiBootstrap>();

  // Small
  runTileForwardTest(
      globalRank,
      numRanks,
      bs,
      4096,
      8 * 1024 * 1024,
      8 * 1024 * 1024,
      2,
      4,
      1);
  runTileForwardTest(
      globalRank,
      numRanks,
      bs,
      65536,
      8 * 1024 * 1024,
      8 * 1024 * 1024,
      2,
      4,
      1);
  // Medium
  runTileForwardTest(
      globalRank,
      numRanks,
      bs,
      1 * 1024 * 1024,
      8 * 1024 * 1024,
      8 * 1024 * 1024,
      2,
      8,
      1);
  // Large (multi-step)
  runTileForwardTest(
      globalRank,
      numRanks,
      bs,
      64 * 1024 * 1024,
      8 * 1024 * 1024,
      8 * 1024 * 1024,
      2,
      16,
      1);
}

// Signal granularity (chunkSize < slotSize → multiple sub-step signals)
TEST_F(P2pNvlTransportTestFixture, TileForwardSignalGranularity) {
  if (numRanks != 2) {
    return;
  }
  auto bs = std::make_shared<meta::comms::MpiBootstrap>();
  const size_t nBytes = 32 * 1024 * 1024;

  // Per-slot signaling
  runTileForwardTest(
      globalRank,
      numRanks,
      bs,
      nBytes,
      8 * 1024 * 1024,
      8 * 1024 * 1024,
      2,
      16,
      1);
  // Sub-slot signaling
  runTileForwardTest(
      globalRank, numRanks, bs, nBytes, 8 * 1024 * 1024, 128 * 1024, 2, 16, 1);
  runTileForwardTest(
      globalRank, numRanks, bs, nBytes, 8 * 1024 * 1024, 1024 * 1024, 2, 16, 1);
}

// Different block counts
TEST_F(P2pNvlTransportTestFixture, TileForwardBlockCounts) {
  if (numRanks != 2) {
    return;
  }
  auto bs = std::make_shared<meta::comms::MpiBootstrap>();
  const size_t nBytes = 16 * 1024 * 1024;

  for (int blocks : {1, 2, 4, 8, 16, 32}) {
    runTileForwardTest(
        globalRank,
        numRanks,
        bs,
        nBytes,
        8 * 1024 * 1024,
        8 * 1024 * 1024,
        2,
        blocks,
        1);
  }
}

// Pipeline depth variations
TEST_F(P2pNvlTransportTestFixture, TileForwardPipelineDepth) {
  if (numRanks != 2) {
    return;
  }
  auto bs = std::make_shared<meta::comms::MpiBootstrap>();
  const size_t nBytes = 32 * 1024 * 1024;

  runTileForwardTest(
      globalRank, numRanks, bs, nBytes, 8 * 1024 * 1024, 128 * 1024, 2, 16, 1);
  runTileForwardTest(
      globalRank, numRanks, bs, nBytes, 8 * 1024 * 1024, 128 * 1024, 4, 16, 1);
}

// Multi-call: persistent step state across iterations
TEST_F(P2pNvlTransportTestFixture, TileForwardMultiCallPersistentStep) {
  if (numRanks != 2) {
    return;
  }
  auto bs = std::make_shared<meta::comms::MpiBootstrap>();

  runTileForwardTest(
      globalRank,
      numRanks,
      bs,
      8 * 1024 * 1024,
      8 * 1024 * 1024,
      8 * 1024 * 1024,
      2,
      4,
      5);
  // Many sub-step signals per call
  runTileForwardTest(
      globalRank,
      numRanks,
      bs,
      8 * 1024 * 1024,
      8 * 1024 * 1024,
      128 * 1024,
      2,
      4,
      5);
}

TEST_F(P2pNvlTransportTestFixture, TileForwardDesynchronizedStepState) {
  if (numRanks != 2) {
    return;
  }

  constexpr int kNumBlocks = 4;
  constexpr size_t kDataBufferSize = 1024 * 1024;
  constexpr size_t kMaxSignalBytes = 64 * 1024;
  constexpr size_t kPerBlockSlotSize = kDataBufferSize / kNumBlocks;
  constexpr size_t kForwardBytes = kMaxSignalBytes * kNumBlocks;
  constexpr size_t kAdvanceBytes = kForwardBytes * 5;
  constexpr int kThreadCount = 256;
  constexpr unsigned char kAdvancePattern = 0x33;
  constexpr unsigned char kForwardPattern = 0x7a;

  auto bs = std::make_shared<meta::comms::MpiBootstrap>();
  const int peerRank = globalRank == 0 ? 1 : 0;
  auto config = makeNvlConfig(kPerBlockSlotSize, 2, kNumBlocks);
  MultiPeerNvlTransport transport(globalRank, numRanks, bs, config);
  transport.exchange();
  auto p2pHost = transport.buildP2pTransportDevice(peerRank);
  int device = 0;
  CUDACHECK_TEST(cudaGetDevice(&device));
  Timeout timeout = makeTimeout(5000, device);

  if (globalRank == 0) {
    CUDACHECK_TEST(cudaMemset(
        p2pHost.getLocalState().dataBuffer,
        0,
        static_cast<std::size_t>(config.maxNumChannels) *
            config.perChannelSize));
    CUDACHECK_TEST(cudaDeviceSynchronize());
  }

  DeviceBuffer advanceSendBuf(kAdvanceBytes);
  DeviceBuffer advanceRecvBuf(kAdvanceBytes);
  if (globalRank == 1) {
    CUDACHECK_TEST(
        cudaMemset(advanceSendBuf.get(), kAdvancePattern, kAdvanceBytes));
  } else {
    CUDACHECK_TEST(cudaMemset(advanceRecvBuf.get(), 0, kAdvanceBytes));
  }

  // Advance only the rank1->rank0 tile send/recv state by 5 sub-steps. The
  // following forward call then has recvStep == 0 and sendStep == 5 on rank 1.
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  if (globalRank == 1) {
    test::testTileSend(
        p2pHost,
        advanceSendBuf.get(),
        kAdvanceBytes,
        kMaxSignalBytes,
        timeout,
        kNumBlocks,
        kThreadCount);
  } else {
    test::testTileRecv(
        p2pHost,
        advanceRecvBuf.get(),
        kAdvanceBytes,
        kMaxSignalBytes,
        timeout,
        kNumBlocks,
        kThreadCount);
  }
  CUDACHECK_TEST(cudaDeviceSynchronize());
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  if (globalRank == 0) {
    std::vector<char> hostBuf(kAdvanceBytes);
    CUDACHECK_TEST(cudaMemcpy(
        hostBuf.data(),
        advanceRecvBuf.get(),
        kAdvanceBytes,
        cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < kAdvanceBytes; ++i) {
      if (static_cast<unsigned char>(hostBuf[i]) != kAdvancePattern) {
        EXPECT_EQ(static_cast<unsigned char>(hostBuf[i]), kAdvancePattern)
            << "pre-advance mismatch at byte " << i;
        break;
      }
    }
  }

  DeviceBuffer srcBuf(kForwardBytes);
  DeviceBuffer fwdR1Buf(kForwardBytes);
  if (globalRank == 0) {
    CUDACHECK_TEST(cudaMemset(srcBuf.get(), kForwardPattern, kForwardBytes));
  } else {
    CUDACHECK_TEST(cudaMemset(fwdR1Buf.get(), 0, kForwardBytes));
  }

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  if (globalRank == 0) {
    test::testTileSend(
        p2pHost,
        srcBuf.get(),
        kForwardBytes,
        kMaxSignalBytes,
        timeout,
        kNumBlocks,
        kThreadCount);
  } else {
    TiledBuffer<char> dstTiles(
        static_cast<char*>(fwdR1Buf.get()), kForwardBytes, kNumBlocks);
    std::size_t maxSignalBytes = kMaxSignalBytes;
    void* args[] = {&p2pHost, &p2pHost, &dstTiles, &maxSignalBytes, &timeout};

    CUDACHECK_TEST(cudaLaunchKernel(
        (void*)comms::prims::benchmark::p2pTileForward,
        dim3(kNumBlocks),
        dim3(kThreadCount),
        args,
        0,
        nullptr));
  }
  CUDACHECK_TEST(cudaDeviceSynchronize());
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  if (globalRank == 0) {
    std::vector<char> stagingSlot(kDataBufferSize);
    CUDACHECK_TEST(cudaMemcpy(
        stagingSlot.data(),
        static_cast<char*>(p2pHost.getLocalState().dataBuffer) +
            kDataBufferSize,
        kDataBufferSize,
        cudaMemcpyDeviceToHost));
    for (int block = 0; block < kNumBlocks; ++block) {
      const size_t chunkOffset = block * kPerBlockSlotSize + kMaxSignalBytes;
      for (size_t i = 0; i < kMaxSignalBytes; ++i) {
        EXPECT_EQ(
            static_cast<unsigned char>(stagingSlot[chunkOffset + i]),
            kForwardPattern)
            << "forward staging mismatch at block " << block << " byte " << i;
        if (static_cast<unsigned char>(stagingSlot[chunkOffset + i]) !=
            kForwardPattern) {
          return;
        }
      }
    }
  } else {
    std::vector<char> hostBuf(kForwardBytes);
    CUDACHECK_TEST(cudaMemcpy(
        hostBuf.data(), fwdR1Buf.get(), kForwardBytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < kForwardBytes; ++i) {
      EXPECT_EQ(static_cast<unsigned char>(hostBuf[i]), kForwardPattern)
          << "forward mismatch at byte " << i;
      if (static_cast<unsigned char>(hostBuf[i]) != kForwardPattern) {
        return;
      }
    }
  }
}

TEST_F(
    P2pNvlTransportTestFixture,
    TileForwardMultiCallWithoutDrainUsesPersistentCursor) {
  const int numBlocks = 4;
  const int threadCount = 256;
  const size_t maxSignalBytes = 16 * 1024;
  const size_t firstCallBytes = maxSignalBytes;
  const size_t secondCallBytes = maxSignalBytes * 3;
  const size_t tileBytes = firstCallBytes + secondCallBytes;
  const size_t totalBytes = tileBytes * numBlocks;

  P2pNvlTransportOptions options{
      .dataBufferSize = tileBytes * 2 * numBlocks,
      .pipelineDepth = 2,
      .per_channel_buffer = tileBytes * 2,
      .per_channel_slot = tileBytes,
      .max_num_channels = numBlocks,
  };
  const size_t stagingBytes = options.dataBufferSize;

  DeviceBuffer sourceStagingBuf(stagingBytes);
  DeviceBuffer forwardedStagingBuf(stagingBytes);
  DeviceBuffer dstBuf(totalBytes);
  const size_t channelBytes = sizeof(NvlChannelState) * numBlocks;
  DeviceBuffer predChannelBuf(channelBytes);
  DeviceBuffer succChannelBuf(channelBytes);

  CUDACHECK_TEST(cudaMemset(sourceStagingBuf.get(), 0, stagingBytes));
  CUDACHECK_TEST(cudaMemset(forwardedStagingBuf.get(), 0, stagingBytes));
  CUDACHECK_TEST(cudaMemset(dstBuf.get(), 0, totalBytes));
  CUDACHECK_TEST(cudaMemset(predChannelBuf.get(), 0, channelBytes));
  CUDACHECK_TEST(cudaMemset(succChannelBuf.get(), 0, channelBytes));

  auto* predChannels = static_cast<NvlChannelState*>(predChannelBuf.get());
  auto* succChannels = static_cast<NvlChannelState*>(succChannelBuf.get());

  // pred / succ are single-rank loopback devices, so localChannels and
  // remoteChannels both point at the same per-side buffer.
  auto pred = makeLocalTileDevice(
      options,
      numBlocks,
      static_cast<char*>(sourceStagingBuf.get()),
      nullptr,
      predChannels,
      predChannels);
  auto succ = makeLocalTileDevice(
      options,
      numBlocks,
      nullptr,
      static_cast<char*>(forwardedStagingBuf.get()),
      succChannels,
      succChannels);

  test::testPrepareTileTwoCallStaging(
      pred,
      numBlocks,
      firstCallBytes,
      secondCallBytes,
      maxSignalBytes,
      0,
      threadCount);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  TiledBuffer<char> dstTiles(
      static_cast<char*>(dstBuf.get()), totalBytes, numBlocks);
  test::testTileTwoCallForward(
      pred,
      succ,
      dstTiles,
      numBlocks,
      firstCallBytes,
      secondCallBytes,
      maxSignalBytes,
      threadCount);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  std::vector<char> hostStaging(stagingBytes);
  CUDACHECK_TEST(cudaMemcpy(
      hostStaging.data(),
      forwardedStagingBuf.get(),
      stagingBytes,
      cudaMemcpyDeviceToHost));
  expectPersistentTwoCallStagingPattern(
      hostStaging,
      options.dataBufferSize,
      firstCallBytes,
      secondCallBytes,
      maxSignalBytes,
      options.pipelineDepth,
      numBlocks,
      0,
      "tile forward persistent cursor staging");

  std::vector<char> hostDst(totalBytes);
  CUDACHECK_TEST(cudaMemcpy(
      hostDst.data(), dstBuf.get(), totalBytes, cudaMemcpyDeviceToHost));
  expectTwoCallPattern(
      hostDst,
      firstCallBytes,
      secondCallBytes,
      numBlocks,
      0,
      "tile forward persistent cursor dst");
}

// Partial tiles (nbytes not evenly divisible by numBlocks)
TEST_F(P2pNvlTransportTestFixture, TileForwardPartialTiles) {
  if (numRanks != 2) {
    return;
  }
  auto bs = std::make_shared<meta::comms::MpiBootstrap>();

  runTileForwardTest(
      globalRank,
      numRanks,
      bs,
      1000000,
      8 * 1024 * 1024,
      8 * 1024 * 1024,
      2,
      4,
      1);
  runTileForwardTest(
      globalRank,
      numRanks,
      bs,
      7 * 1024 * 1024 + 12345,
      8 * 1024 * 1024,
      8 * 1024 * 1024,
      2,
      4,
      1);
}

// =============================================================================
// Unaligned dstbuff tests for tile forward()
// =============================================================================
// The staging buffers (localState_/remoteState_.dataBuffer) are always
// 256-byte aligned (cudaMalloc), so the only user-facing pointer that can
// be misaligned is `dstbuff` (the local user output buffer on rank 1, where
// the dual-dst memcpy writes its second destination).
//
// These tests offset rank 1's dstbuff by various byte amounts to exercise
// memcpy_vectorized's unaligned path in the forward() implementation.
// They reuse runTileForwardTest with the dstOffset parameter.

// Parameterized test fixture for tile forward unaligned dst
struct ForwardUnalignedParams {
  size_t dstOffset; // bytes added to dst pointer (0 = aligned)
  size_t nbytes;
  std::string name;
};

std::string forwardUnalignedParamName(
    const ::testing::TestParamInfo<ForwardUnalignedParams>& info) {
  return info.param.name;
}

// Tile forward unaligned: reuse runTileForwardTest's dstOffset parameter.
class TileForwardUnalignedTestFixture
    : public MpiBaseTestFixture,
      public ::testing::WithParamInterface<ForwardUnalignedParams> {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }
};

TEST_P(TileForwardUnalignedTestFixture, TileForward) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const auto& params = GetParam();
  XLOGF(
      INFO,
      "Running tile forward unaligned test: {} (nbytes={}, dstOffset={})",
      params.name,
      params.nbytes,
      params.dstOffset);

  auto bs = std::make_shared<meta::comms::MpiBootstrap>();
  // Use a multi-step config so the
  // unaligned dst is exercised across many steps and chunks.
  runTileForwardTest(
      globalRank,
      numRanks,
      bs,
      params.nbytes,
      /*perChannelSize=*/8 * 1024 * 1024,
      /*chunkSize=*/128 * 1024,
      /*pipelineDepth=*/2,
      /*numSendBlocks=*/4,
      /*nIters=*/1,
      /*threadCount=*/256,
      params.dstOffset);
}

INSTANTIATE_TEST_SUITE_P(
    TileForwardUnalignedVariations,
    TileForwardUnalignedTestFixture,
    ::testing::Values(
        // Baseline aligned
        ForwardUnalignedParams{
            .dstOffset = 0,
            .nbytes = 4 * 1024 * 1024,
            .name = "Aligned_4MB"},
        // Various misalignments
        ForwardUnalignedParams{
            .dstOffset = 1,
            .nbytes = 4 * 1024 * 1024,
            .name = "Off1_4MB"},
        ForwardUnalignedParams{
            .dstOffset = 7,
            .nbytes = 4 * 1024 * 1024,
            .name = "Off7_4MB"},
        ForwardUnalignedParams{
            .dstOffset = 8,
            .nbytes = 4 * 1024 * 1024,
            .name = "Off8_4MB"},
        ForwardUnalignedParams{
            .dstOffset = 15,
            .nbytes = 4 * 1024 * 1024,
            .name = "Off15_4MB"},
        // Larger transfer with misalignment
        ForwardUnalignedParams{
            .dstOffset = 3,
            .nbytes = 16 * 1024 * 1024,
            .name = "Off3_16MB"},
        // Unaligned size + unaligned offset
        ForwardUnalignedParams{
            .dstOffset = 5,
            .nbytes = 1000003,
            .name = "Off5_OddSize"}),
    forwardUnalignedParamName);

// =============================================================================
// LL Buffer Wiring Tests
// Verify MultiPeerNvlTransport correctly wires LL buffer pointers into
// P2pNvlTransportDevice handles.
// =============================================================================

TEST_F(P2pNvlTransportTestFixture, LlBufferWiring_Enabled) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;

  auto config = makeNvlConfig(4096, 2);
  config.llBufferSize = ll_buffer_size(4096);

  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  MultiPeerNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.exchange();

  auto p2p = transport.buildP2pTransportDevice(peerRank);

  ASSERT_NE(p2p.getLocalState().llBuffer, nullptr)
      << "Rank " << globalRank
      << ": localState.llBuffer should be non-null when llBufferSize > 0";
  ASSERT_NE(p2p.getRemoteState().llBuffer, nullptr)
      << "Rank " << globalRank
      << ": remoteState.llBuffer should be non-null when llBufferSize > 0";
  ASSERT_NE(p2p.getLocalState().llBuffer, p2p.getRemoteState().llBuffer)
      << "Rank " << globalRank
      << ": local and remote llBuffer should point to different ranks' buffers";

  XLOGF(INFO, "Rank {}: LlBufferWiring_Enabled test completed", globalRank);
}

TEST_F(P2pNvlTransportTestFixture, LlBufferWiring_Disabled) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;

  auto config = makeNvlConfig(4096, 2);

  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  MultiPeerNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.exchange();

  auto p2p = transport.buildP2pTransportDevice(peerRank);

  ASSERT_EQ(p2p.getLocalState().llBuffer, nullptr)
      << "Rank " << globalRank
      << ": localState.llBuffer should be null when llBufferSize == 0";
  ASSERT_EQ(p2p.getRemoteState().llBuffer, nullptr)
      << "Rank " << globalRank
      << ": remoteState.llBuffer should be null when llBufferSize == 0";

  XLOGF(INFO, "Rank {}: LlBufferWiring_Disabled test completed", globalRank);
}

} // namespace comms::prims::tests

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  auto mpi_env = std::make_unique<MPIEnvironmentBase>();
  ::testing::AddGlobalTestEnvironment(mpi_env.get());
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
