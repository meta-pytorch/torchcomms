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

// Parameters for transfer size tests: (nbytes, dataBufferSize, chunkSize, name,
// useDualStateBuffer, useCudaGraph)
struct TransferSizeParams {
  size_t nbytes;
  size_t dataBufferSize;
  size_t chunkSize;
  bool useDualStateBuffer;
  bool useCudaGraph;
  std::string name;
};

// Parameters for group type tests: (groupType, numBlocks, blockSize,
// blocksPerGroup, useDualStateBuffer, useCudaGraph, name)
struct GroupTypeParams {
  test::GroupType groupType;
  int numBlocks;
  int blockSize;
  int blocksPerGroup;
  bool useDualStateBuffer;
  bool useCudaGraph;
  std::string name;
};

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

TEST_F(P2pNvlTransportTestFixture, IpcMemAccess) {
  // Only test with 2 ranks
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;

  const size_t numElements = 256;
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = sizeof(int) * numElements,
      .chunkSize = 256, // 256 bytes
      .pipelineDepth = 4,
  };

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

// Helper to verify received data with early exit on first mismatch
void verifyReceivedData(
    const int* dst_d,
    size_t nbytes,
    int expectedValue,
    const std::string& context = "") {
  const size_t numInts = nbytes / sizeof(int);
  std::vector<int> hostBuffer(numInts);
  CUDACHECK_TEST(
      cudaMemcpy(hostBuffer.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < numInts; i++) {
    EXPECT_EQ(hostBuffer[i], expectedValue)
        << context << "Mismatch at index " << i << ": expected "
        << expectedValue << ", got " << hostBuffer[i];
    if (hostBuffer[i] != expectedValue) {
      break;
    }
  }
}
// Helper to run a single send/recv iteration with verification
void runSendRecvIteration(
    int globalRank,
    P2pNvlTransportDevice* p2p,
    int* src_d,
    int* dst_d,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    int iter,
    test::GroupType groupType = test::GroupType::WARP) {
  const size_t numInts = nbytes / sizeof(int);
  const int testValue = 42 + iter;

  if (globalRank == 0) {
    test::fillBuffer(src_d, testValue, numInts);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    test::testSend(p2p, src_d, nbytes, numBlocks, blockSize, groupType, 1);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  } else {
    CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    test::testRecv(p2p, dst_d, nbytes, numBlocks, blockSize, groupType, 1);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    std::vector<int> hostBuffer(numInts);
    CUDACHECK_TEST(
        cudaMemcpy(hostBuffer.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < numInts; i++) {
      EXPECT_EQ(hostBuffer[i], testValue)
          << "Iter " << iter << ": Mismatch at index " << i << ": expected "
          << testValue << ", got " << hostBuffer[i];
      if (hostBuffer[i] != testValue) {
        break;
      }
    }
  }
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
// runBasicSendRecvTest - Common test pattern for send/recv verification
// =============================================================================

void runBasicSendRecvTest(
    TransportTestHelper& helper,
    size_t nbytes,
    int numBlocks,
    int blockSize,
    int nIter = 1,
    test::GroupType groupType = test::GroupType::WARP,
    bool useCudaGraph = false) {
  auto p2p = helper.getDevicePtr();

  DeviceBuffer srcBuffer(nbytes);
  DeviceBuffer dstBuffer(nbytes);

  auto src_d = static_cast<int*>(srcBuffer.get());
  auto dst_d = static_cast<int*>(dstBuffer.get());

  if (!useCudaGraph) {
    // Direct kernel launch mode
    for (int iter = 0; iter < nIter; iter++) {
      runSendRecvIteration(
          helper.globalRank(),
          p2p,
          src_d,
          dst_d,
          nbytes,
          numBlocks,
          blockSize,
          iter,
          groupType);
    }
  } else {
    // CUDA graph mode: capture send/recv into graphs, then replay
    // Enforce minimum 3 iterations for graph replay testing
    const int graphIter = std::max(nIter, 3);

    cudaStream_t stream;
    CUDACHECK_TEST(cudaStreamCreate(&stream));

    // Capture the send or recv kernel into a graph
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    CUDACHECK_TEST(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    if (helper.globalRank() == 0) {
      test::testSend(
          p2p, src_d, nbytes, numBlocks, blockSize, groupType, 1, stream);
    } else {
      test::testRecv(
          p2p, dst_d, nbytes, numBlocks, blockSize, groupType, 1, stream);
    }
    CUDACHECK_TEST(cudaStreamEndCapture(stream, &graph));
    CUDACHECK_TEST(
        cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    const size_t numInts = nbytes / sizeof(int);
    std::vector<int> hostBuffer(numInts);

    // Replay the graph multiple times with different data patterns
    for (int iter = 0; iter < graphIter; iter++) {
      const int testValue = 42 + iter;

      if (helper.globalRank() == 0) {
        // Sender: fill source buffer with test value
        test::fillBuffer(src_d, testValue, numInts);
        CUDACHECK_TEST(cudaDeviceSynchronize());
      } else {
        // Receiver: clear destination buffer
        CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));
      }

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      // Launch the graph
      CUDACHECK_TEST(cudaGraphLaunch(graphExec, stream));
      CUDACHECK_TEST(cudaStreamSynchronize(stream));

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      // Receiver verifies the data
      if (helper.globalRank() == 1) {
        CUDACHECK_TEST(cudaMemcpy(
            hostBuffer.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < numInts; i++) {
          EXPECT_EQ(hostBuffer[i], testValue)
              << "CudaGraph iter " << iter << ": Mismatch at index " << i
              << ": expected " << testValue << ", got " << hostBuffer[i];
          if (hostBuffer[i] != testValue) {
            break;
          }
        }
      }
    }

    // Cleanup
    CUDACHECK_TEST(cudaGraphExecDestroy(graphExec));
    CUDACHECK_TEST(cudaGraphDestroy(graph));
    CUDACHECK_TEST(cudaStreamDestroy(stream));
  }
}

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

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = 8 * 1024 * 1024, // 8MB slot
      .chunkSize = 8 * 1024 * 1024,
      .pipelineDepth = 2,
  };

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

TEST_F(P2pNvlTransportTestFixture, TileSendRecvCudaGraphReplay) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping: requires 2 ranks, got {}", numRanks);
    return;
  }

  const int peerRank = (globalRank == 0) ? 1 : 0;
  const size_t nBytes = 2 * 1024 * 1024;
  const int numSendBlocks = 4;
  const size_t maxSignalBytes = 16 * 1024;

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = 512 * 1024,
      .chunkSize = 512 * 1024,
      .pipelineDepth = 2,
  };

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
    size_t dataBufferSize,
    size_t chunkSize,
    size_t pipelineDepth,
    int numSendBlocks,
    int nIters,
    int threadCount = 256) {
  int peerRank = (globalRank == 0) ? 1 : 0;

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = chunkSize,
      .pipelineDepth = pipelineDepth,
  };

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
          << ", slot=" << dataBufferSize << ", chunk=" << chunkSize
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
      baseByte += protocolBytes;
    }
  }
}

// Build a P2pNvlTransportDevice for tile tests that bypass
// MultiPeerNvlTransport. The options must already have per_channel_slot and
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
      DeviceSpan<ChunkState>(),
      DeviceSpan<ChunkState>(),
      DeviceSpan<SignalState>(),
      DeviceSpan<BarrierState>(),
      nullptr};

  RemoteState remoteState{
      remoteData,
      DeviceSpan<ChunkState>(),
      DeviceSpan<ChunkState>(),
      DeviceSpan<SignalState>(),
      DeviceSpan<BarrierState>(),
      nullptr};

  return P2pNvlTransportDevice(
      0, 1, opts, localState, remoteState, localChannels, remoteChannels);
}

class LocalTileHarness {
 public:
  LocalTileHarness(const P2pNvlTransportOptions& options, int numBlocks)
      : options_(options),
        numBlocks_(numBlocks),
        stagingBytes_(options.dataBufferSize * options.pipelineDepth),
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
      .dataBufferSize = tileBytes * numBlocks,
      .chunkSize = maxSignalBytes,
      .pipelineDepth = 2,
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
  const size_t expectedStep = alignTileProtocolBytes(firstCallBytes) +
      alignTileProtocolBytes(secondCallBytes);

  P2pNvlTransportOptions options{
      .dataBufferSize = perBlockSlotSize * numBlocks,
      .chunkSize = maxSignalBytes,
      .pipelineDepth = 2,
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
  for (size_t i = firstCallBytes; i < alignTileProtocolBytes(firstCallBytes);
       ++i) {
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
  const size_t expectedStep = alignTileProtocolBytes(firstCallBytes) +
      alignTileProtocolBytes(secondCallBytes);

  P2pNvlTransportOptions options{
      .dataBufferSize = perBlockSlotSize * numBlocks,
      .chunkSize = maxSignalBytes,
      .pipelineDepth = 2,
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
  for (size_t i = firstCallBytes; i < alignTileProtocolBytes(firstCallBytes);
       ++i) {
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
      .dataBufferSize = perBlockSlotSize * numBlocks,
      .chunkSize = perBlockSlotSize,
      .pipelineDepth = 2,
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
      .dataBufferSize = perBlockSlotSize * numBlocks,
      .chunkSize = perBlockSlotSize,
      .pipelineDepth = 2,
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
      .dataBufferSize = perBlockSlotSize * activeBlocks,
      .chunkSize = perBlockSlotSize,
      .pipelineDepth = 2,
      .max_num_channels = activeBlocks,
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
      .dataBufferSize = perBlockSlotSize * numBlocks,
      .chunkSize = maxSignalBytes,
      .pipelineDepth = pipelineDepth,
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
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = 8 * 1024 * 1024,
      .chunkSize = 8 * 1024 * 1024,
      .pipelineDepth = 2,
  };

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
// Parameterized Test Fixture for Transfer Size Variations
// =============================================================================

class TransferSizeTestFixture
    : public MpiBaseTestFixture,
      public ::testing::WithParamInterface<TransferSizeParams> {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }
};

TEST_P(TransferSizeTestFixture, SendRecv) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const auto& params = GetParam();
  XLOGF(
      INFO,
      "Running transfer size test: {} (nbytes={}, bufferSize={}, chunkSize={}, dualState={}, cudaGraph={})",
      params.name,
      params.nbytes,
      params.dataBufferSize,
      params.chunkSize,
      params.useDualStateBuffer,
      params.useCudaGraph);

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = params.dataBufferSize,
      .chunkSize = params.chunkSize,
      .pipelineDepth = 4,
      .useDualStateBuffer = params.useDualStateBuffer,
  };

  TransportTestHelper helper(globalRank, numRanks, localRank, config);
  runBasicSendRecvTest(
      helper,
      params.nbytes,
      2,
      64,
      1,
      test::GroupType::WARP,
      params.useCudaGraph);

  XLOGF(
      INFO,
      "Rank {}: Transfer size test '{}' completed",
      globalRank,
      params.name);
}

std::string transferSizeParamName(
    const ::testing::TestParamInfo<TransferSizeParams>& info) {
  return info.param.name + (info.param.useCudaGraph ? "_CudaGraph" : "");
}

INSTANTIATE_TEST_SUITE_P(
    TransferSizeVariations,
    TransferSizeTestFixture,
    ::testing::Values(
        // ===== SINGLE STATE BUFFER MODE (default) =====
        // Small transfer: nbytes < chunkSize
        TransferSizeParams{
            .nbytes = 512,
            .dataBufferSize = 4096,
            .chunkSize = 1024,
            .useDualStateBuffer = false,
            .useCudaGraph = false,
            .name = "SmallTransfer_LessThanChunk_SingleState"},
        TransferSizeParams{
            .nbytes = 512,
            .dataBufferSize = 4096,
            .chunkSize = 1024,
            .useDualStateBuffer = false,
            .useCudaGraph = true,
            .name = "SmallTransfer_LessThanChunk_SingleState"},
        // Single chunk: nbytes == chunkSize
        TransferSizeParams{
            .nbytes = 1024,
            .dataBufferSize = 4096,
            .chunkSize = 1024,
            .useDualStateBuffer = false,
            .useCudaGraph = false,
            .name = "SingleChunk_ExactMatch_SingleState"},
        TransferSizeParams{
            .nbytes = 1024,
            .dataBufferSize = 4096,
            .chunkSize = 1024,
            .useDualStateBuffer = false,
            .useCudaGraph = true,
            .name = "SingleChunk_ExactMatch_SingleState"},
        // Transfer not aligned to chunk size
        TransferSizeParams{
            .nbytes = 1000,
            .dataBufferSize = 4096,
            .chunkSize = 256,
            .useDualStateBuffer = false,
            .useCudaGraph = false,
            .name = "UnalignedToChunk_SingleState"},
        TransferSizeParams{
            .nbytes = 1000,
            .dataBufferSize = 4096,
            .chunkSize = 256,
            .useDualStateBuffer = false,
            .useCudaGraph = true,
            .name = "UnalignedToChunk_SingleState"},
        // Transfer not aligned to vector size (16 bytes)
        TransferSizeParams{
            .nbytes = 1000,
            .dataBufferSize = 4096,
            .chunkSize = 256,
            .useDualStateBuffer = false,
            .useCudaGraph = false,
            .name = "NonVectorAligned_1000bytes_SingleState"},
        TransferSizeParams{
            .nbytes = 1000,
            .dataBufferSize = 4096,
            .chunkSize = 256,
            .useDualStateBuffer = false,
            .useCudaGraph = true,
            .name = "NonVectorAligned_1000bytes_SingleState"},
        // Another non-vector-aligned size
        TransferSizeParams{
            .nbytes = 100,
            .dataBufferSize = 1024,
            .chunkSize = 64,
            .useDualStateBuffer = false,
            .useCudaGraph = false,
            .name = "NonVectorAligned_100bytes_SingleState"},
        TransferSizeParams{
            .nbytes = 100,
            .dataBufferSize = 1024,
            .chunkSize = 64,
            .useDualStateBuffer = false,
            .useCudaGraph = true,
            .name = "NonVectorAligned_100bytes_SingleState"},
        // Transfer exactly equals buffer size (single step)
        TransferSizeParams{
            .nbytes = 4096,
            .dataBufferSize = 4096,
            .chunkSize = 256,
            .useDualStateBuffer = false,
            .useCudaGraph = false,
            .name = "ExactBufferSize_SingleState"},
        TransferSizeParams{
            .nbytes = 4096,
            .dataBufferSize = 4096,
            .chunkSize = 256,
            .useDualStateBuffer = false,
            .useCudaGraph = true,
            .name = "ExactBufferSize_SingleState"},
        // Multiple steps: transfer > buffer size
        TransferSizeParams{
            .nbytes = 16 * 1024,
            .dataBufferSize = 4096,
            .chunkSize = 256,
            .useDualStateBuffer = false,
            .useCudaGraph = false,
            .name = "MultipleSteps_4x_SingleState"},
        TransferSizeParams{
            .nbytes = 16 * 1024,
            .dataBufferSize = 4096,
            .chunkSize = 256,
            .useDualStateBuffer = false,
            .useCudaGraph = true,
            .name = "MultipleSteps_4x_SingleState"},
        // Large transfer with multiple steps
        TransferSizeParams{
            .nbytes = 4 * 1024 * 1024,
            .dataBufferSize = 1024 * 1024,
            .chunkSize = 4096,
            .useDualStateBuffer = false,
            .useCudaGraph = false,
            .name = "LargeMultiStep_4MB_SingleState"},
        TransferSizeParams{
            .nbytes = 4 * 1024 * 1024,
            .dataBufferSize = 1024 * 1024,
            .chunkSize = 4096,
            .useDualStateBuffer = false,
            .useCudaGraph = true,
            .name = "LargeMultiStep_4MB_SingleState"},
        // Very large transfer (64MB with 8MB buffer = 8 steps)
        TransferSizeParams{
            .nbytes = 64 * 1024 * 1024,
            .dataBufferSize = 8 * 1024 * 1024,
            .chunkSize = 1024,
            .useDualStateBuffer = false,
            .useCudaGraph = false,
            .name = "VeryLargeMultiStep_64MB_SingleState"},
        TransferSizeParams{
            .nbytes = 64 * 1024 * 1024,
            .dataBufferSize = 8 * 1024 * 1024,
            .chunkSize = 1024,
            .useDualStateBuffer = false,
            .useCudaGraph = true,
            .name = "VeryLargeMultiStep_64MB_SingleState"},
        // Edge case: stepBytes exactly divisible by chunkSize (no partial
        // chunk) Tests that we don't process any 0-byte chunks
        TransferSizeParams{
            .nbytes = 4096,
            .dataBufferSize = 4096,
            .chunkSize = 1024,
            .useDualStateBuffer = false,
            .useCudaGraph = false,
            .name = "ExactChunkBoundary_4Chunks_SingleState"},
        TransferSizeParams{
            .nbytes = 4096,
            .dataBufferSize = 4096,
            .chunkSize = 1024,
            .useDualStateBuffer = false,
            .useCudaGraph = true,
            .name = "ExactChunkBoundary_4Chunks_SingleState"},
        // Edge case: last chunk has minimal bytes (1 byte remainder)
        // stepBytes=4097, chunkSize=1024 → 5 chunks, last chunk = 1 byte
        TransferSizeParams{
            .nbytes = 4097,
            .dataBufferSize = 8192,
            .chunkSize = 1024,
            .useDualStateBuffer = false,
            .useCudaGraph = false,
            .name = "MinimalLastChunk_1Byte_SingleState"},
        TransferSizeParams{
            .nbytes = 4097,
            .dataBufferSize = 8192,
            .chunkSize = 1024,
            .useDualStateBuffer = false,
            .useCudaGraph = true,
            .name = "MinimalLastChunk_1Byte_SingleState"},
        // Edge case: multiple steps where each step ends exactly on chunk
        // boundary 8KB transfer, 4KB buffer, 1KB chunks → 2 steps × 4 chunks
        // each
        TransferSizeParams{
            .nbytes = 8 * 1024,
            .dataBufferSize = 4 * 1024,
            .chunkSize = 1024,
            .useDualStateBuffer = false,
            .useCudaGraph = false,
            .name = "MultiStep_ExactChunkBoundaries_SingleState"},
        TransferSizeParams{
            .nbytes = 8 * 1024,
            .dataBufferSize = 4 * 1024,
            .chunkSize = 1024,
            .useDualStateBuffer = false,
            .useCudaGraph = true,
            .name = "MultiStep_ExactChunkBoundaries_SingleState"},
        // Edge case: chunkSize larger than stepBytes
        // Forces single chunk per step with partial fill
        TransferSizeParams{
            .nbytes = 2048,
            .dataBufferSize = 1024,
            .chunkSize = 2048,
            .useDualStateBuffer = false,
            .useCudaGraph = false,
            .name = "ChunkLargerThanStep_SingleState"},
        TransferSizeParams{
            .nbytes = 2048,
            .dataBufferSize = 1024,
            .chunkSize = 2048,
            .useDualStateBuffer = false,
            .useCudaGraph = true,
            .name = "ChunkLargerThanStep_SingleState"},

        // ===== DUAL STATE BUFFER MODE =====
        // Small transfer: nbytes < chunkSize
        TransferSizeParams{
            .nbytes = 512,
            .dataBufferSize = 4096,
            .chunkSize = 1024,
            .useDualStateBuffer = true,
            .useCudaGraph = false,
            .name = "SmallTransfer_LessThanChunk_DualState"},
        TransferSizeParams{
            .nbytes = 512,
            .dataBufferSize = 4096,
            .chunkSize = 1024,
            .useDualStateBuffer = true,
            .useCudaGraph = true,
            .name = "SmallTransfer_LessThanChunk_DualState"},
        // Single chunk: nbytes == chunkSize
        TransferSizeParams{
            .nbytes = 1024,
            .dataBufferSize = 4096,
            .chunkSize = 1024,
            .useDualStateBuffer = true,
            .useCudaGraph = false,
            .name = "SingleChunk_ExactMatch_DualState"},
        TransferSizeParams{
            .nbytes = 1024,
            .dataBufferSize = 4096,
            .chunkSize = 1024,
            .useDualStateBuffer = true,
            .useCudaGraph = true,
            .name = "SingleChunk_ExactMatch_DualState"},
        // Transfer not aligned to chunk size
        TransferSizeParams{
            .nbytes = 1000,
            .dataBufferSize = 4096,
            .chunkSize = 256,
            .useDualStateBuffer = true,
            .useCudaGraph = false,
            .name = "UnalignedToChunk_DualState"},
        TransferSizeParams{
            .nbytes = 1000,
            .dataBufferSize = 4096,
            .chunkSize = 256,
            .useDualStateBuffer = true,
            .useCudaGraph = true,
            .name = "UnalignedToChunk_DualState"},
        // Multiple steps: transfer > buffer size
        TransferSizeParams{
            .nbytes = 16 * 1024,
            .dataBufferSize = 4096,
            .chunkSize = 256,
            .useDualStateBuffer = true,
            .useCudaGraph = false,
            .name = "MultipleSteps_4x_DualState"},
        TransferSizeParams{
            .nbytes = 16 * 1024,
            .dataBufferSize = 4096,
            .chunkSize = 256,
            .useDualStateBuffer = true,
            .useCudaGraph = true,
            .name = "MultipleSteps_4x_DualState"},
        // Large transfer with multiple steps
        TransferSizeParams{
            .nbytes = 4 * 1024 * 1024,
            .dataBufferSize = 1024 * 1024,
            .chunkSize = 4096,
            .useDualStateBuffer = true,
            .useCudaGraph = false,
            .name = "LargeMultiStep_4MB_DualState"},
        TransferSizeParams{
            .nbytes = 4 * 1024 * 1024,
            .dataBufferSize = 1024 * 1024,
            .chunkSize = 4096,
            .useDualStateBuffer = true,
            .useCudaGraph = true,
            .name = "LargeMultiStep_4MB_DualState"},
        // Very large transfer (64MB with 8MB buffer = 8 steps)
        TransferSizeParams{
            .nbytes = 64 * 1024 * 1024,
            .dataBufferSize = 8 * 1024 * 1024,
            .chunkSize = 1024,
            .useDualStateBuffer = true,
            .useCudaGraph = false,
            .name = "VeryLargeMultiStep_64MB_DualState"},
        TransferSizeParams{
            .nbytes = 64 * 1024 * 1024,
            .dataBufferSize = 8 * 1024 * 1024,
            .chunkSize = 1024,
            .useDualStateBuffer = true,
            .useCudaGraph = true,
            .name = "VeryLargeMultiStep_64MB_DualState"}),
    transferSizeParamName);

// =============================================================================
// Parameterized Test Fixture for Group Type Variations
// =============================================================================

class GroupTypeTestFixture
    : public MpiBaseTestFixture,
      public ::testing::WithParamInterface<GroupTypeParams> {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }
};

TEST_P(GroupTypeTestFixture, SendRecv) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const auto& params = GetParam();
  XLOGF(
      INFO,
      "Running group type test: {} (numBlocks={}, blockSize={}, dualState={}, cudaGraph={})",
      params.name,
      params.numBlocks,
      params.blockSize,
      params.useDualStateBuffer,
      params.useCudaGraph);

  const size_t dataBufferSize = 1024 * 1024; // 1MB staging buffer
  const size_t nbytes = 4 * 1024 * 1024; // 4MB total transfer
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 1024,
      .pipelineDepth = 4,
      .useDualStateBuffer = params.useDualStateBuffer,
  };

  TransportTestHelper helper(globalRank, numRanks, localRank, config);
  runBasicSendRecvTest(
      helper,
      nbytes,
      params.numBlocks,
      params.blockSize,
      1,
      params.groupType,
      params.useCudaGraph);

  XLOGF(
      INFO, "Rank {}: Group type test '{}' completed", globalRank, params.name);
}

std::string groupTypeParamName(
    const ::testing::TestParamInfo<GroupTypeParams>& info) {
  return info.param.name + (info.param.useCudaGraph ? "_CudaGraph" : "");
}

INSTANTIATE_TEST_SUITE_P(
    GroupTypeVariations,
    GroupTypeTestFixture,
    ::testing::Values(
        // ===== SINGLE STATE BUFFER MODE (default) =====
        // Warp-based groups (32 threads per group)
        GroupTypeParams{
            .groupType = test::GroupType::WARP,
            .numBlocks = 4,
            .blockSize = 128,
            .blocksPerGroup = 1,
            .useDualStateBuffer = false,
            .useCudaGraph = false,
            .name = "Warp_4Blocks_128Threads_SingleState"},
        GroupTypeParams{
            .groupType = test::GroupType::WARP,
            .numBlocks = 4,
            .blockSize = 128,
            .blocksPerGroup = 1,
            .useDualStateBuffer = false,
            .useCudaGraph = true,
            .name = "Warp_4Blocks_128Threads_SingleState"},
        GroupTypeParams{
            .groupType = test::GroupType::WARP,
            .numBlocks = 8,
            .blockSize = 256,
            .blocksPerGroup = 1,
            .useDualStateBuffer = false,
            .useCudaGraph = false,
            .name = "Warp_8Blocks_256Threads_SingleState"},
        GroupTypeParams{
            .groupType = test::GroupType::WARP,
            .numBlocks = 8,
            .blockSize = 256,
            .blocksPerGroup = 1,
            .useDualStateBuffer = false,
            .useCudaGraph = true,
            .name = "Warp_8Blocks_256Threads_SingleState"},
        // Block-based groups (all threads in block form one group)
        GroupTypeParams{
            .groupType = test::GroupType::BLOCK,
            .numBlocks = 4,
            .blockSize = 128,
            .blocksPerGroup = 1,
            .useDualStateBuffer = false,
            .useCudaGraph = false,
            .name = "Block_4Groups_128Threads_SingleState"},
        GroupTypeParams{
            .groupType = test::GroupType::BLOCK,
            .numBlocks = 4,
            .blockSize = 128,
            .blocksPerGroup = 1,
            .useDualStateBuffer = false,
            .useCudaGraph = true,
            .name = "Block_4Groups_128Threads_SingleState"},
        GroupTypeParams{
            .groupType = test::GroupType::BLOCK,
            .numBlocks = 8,
            .blockSize = 256,
            .blocksPerGroup = 1,
            .useDualStateBuffer = false,
            .useCudaGraph = false,
            .name = "Block_8Groups_256Threads_SingleState"},
        GroupTypeParams{
            .groupType = test::GroupType::BLOCK,
            .numBlocks = 8,
            .blockSize = 256,
            .blocksPerGroup = 1,
            .useDualStateBuffer = false,
            .useCudaGraph = true,
            .name = "Block_8Groups_256Threads_SingleState"},
        GroupTypeParams{
            .groupType = test::GroupType::BLOCK,
            .numBlocks = 2,
            .blockSize = 512,
            .blocksPerGroup = 1,
            .useDualStateBuffer = false,
            .useCudaGraph = false,
            .name = "Block_2Groups_512Threads_SingleState"},
        GroupTypeParams{
            .groupType = test::GroupType::BLOCK,
            .numBlocks = 2,
            .blockSize = 512,
            .blocksPerGroup = 1,
            .useDualStateBuffer = false,
            .useCudaGraph = true,
            .name = "Block_2Groups_512Threads_SingleState"},

        // ===== DUAL STATE BUFFER MODE =====
        // Warp-based groups (32 threads per group)
        GroupTypeParams{
            .groupType = test::GroupType::WARP,
            .numBlocks = 4,
            .blockSize = 128,
            .blocksPerGroup = 1,
            .useDualStateBuffer = true,
            .useCudaGraph = false,
            .name = "Warp_4Blocks_128Threads_DualState"},
        GroupTypeParams{
            .groupType = test::GroupType::WARP,
            .numBlocks = 8,
            .blockSize = 256,
            .blocksPerGroup = 1,
            .useDualStateBuffer = true,
            .useCudaGraph = false,
            .name = "Warp_8Blocks_256Threads_DualState"},
        // Block-based groups (all threads in block form one group)
        GroupTypeParams{
            .groupType = test::GroupType::BLOCK,
            .numBlocks = 4,
            .blockSize = 128,
            .blocksPerGroup = 1,
            .useDualStateBuffer = true,
            .useCudaGraph = false,
            .name = "Block_4Groups_128Threads_DualState"}),
    groupTypeParamName);

// =============================================================================
// Bidirectional Send/Recv Test
// =============================================================================

TEST_F(P2pNvlTransportTestFixture, BidirectionalSendRecv) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const size_t dataBufferSize = 1024 * 1024; // 1MB staging buffer
  const size_t nbytes = 4 * 1024 * 1024; // 4MB total transfer
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 1024,
      .pipelineDepth = 4,
  };

  TransportTestHelper helper(globalRank, numRanks, localRank, config);
  auto p2p = helper.getDevicePtr();

  const size_t numInts = nbytes / sizeof(int);

  // Each rank has both send and receive buffers
  DeviceBuffer sendBuffer(nbytes);
  DeviceBuffer recvBuffer(nbytes);

  auto send_d = static_cast<int*>(sendBuffer.get());
  auto recv_d = static_cast<int*>(recvBuffer.get());

  const int numBlocks = 4;
  const int blockSize = 128;

  // Each rank uses a different test value
  const int sendValue = 100 + globalRank;
  const int expectedRecvValue = 100 + helper.peerRank();

  // Fill send buffer with this rank's value
  test::fillBuffer(send_d, sendValue, numInts);
  CUDACHECK_TEST(cudaMemset(recv_d, 0, nbytes));
  CUDACHECK_TEST(cudaDeviceSynchronize());

  XLOGF(
      INFO,
      "Rank {}: filled send buffer with {}, expecting to receive {}",
      globalRank,
      sendValue,
      expectedRecvValue);

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Both ranks send and receive simultaneously
  // Rank 0 sends first, then receives
  // Rank 1 receives first, then sends
  // This tests that the state buffers are managed correctly for bidirectional
  if (globalRank == 0) {
    test::testSend(p2p, send_d, nbytes, numBlocks, blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    test::testRecv(p2p, recv_d, nbytes, numBlocks, blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  } else {
    test::testRecv(p2p, recv_d, nbytes, numBlocks, blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    test::testSend(p2p, send_d, nbytes, numBlocks, blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  }

  // Verify received data
  std::vector<int> hostBuffer(numInts);
  CUDACHECK_TEST(
      cudaMemcpy(hostBuffer.data(), recv_d, nbytes, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < numInts; i++) {
    EXPECT_EQ(hostBuffer[i], expectedRecvValue)
        << "Rank " << globalRank << ": Mismatch at index " << i << ": expected "
        << expectedRecvValue << ", got " << hostBuffer[i];
    if (hostBuffer[i] != expectedRecvValue) {
      break;
    }
  }

  XLOGF(INFO, "Rank {}: Bidirectional test completed", globalRank);
}

// =============================================================================
// Stress Test with Many Iterations
// =============================================================================

TEST_F(P2pNvlTransportTestFixture, SendRecvStress) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const size_t dataBufferSize = 512 * 1024; // 512KB staging buffer
  const size_t nbytes = 2 * 1024 * 1024; // 2MB total transfer
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 512,
      .pipelineDepth = 4,
  };

  TransportTestHelper helper(globalRank, numRanks, localRank, config);
  const int nIter = 100;

  XLOGF(
      INFO,
      "Rank {}: Starting stress test with {} iterations",
      globalRank,
      nIter);

  runBasicSendRecvTest(helper, nbytes, 4, 128, nIter);

  XLOGF(
      INFO,
      "Rank {}: Stress test completed ({} iterations)",
      globalRank,
      nIter);
}

// =============================================================================
// Zero-Byte Transfer Test
// =============================================================================

TEST_F(P2pNvlTransportTestFixture, SendRecvZeroBytes) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const size_t dataBufferSize = 4096;
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 256,
      .pipelineDepth = 4,
  };

  TransportTestHelper helper(globalRank, numRanks, localRank, config);
  auto p2p = helper.getDevicePtr();

  // Allocate small buffers for the zero-byte transfer test
  const size_t bufferSize = 64;
  const size_t numInts = bufferSize / sizeof(int);
  DeviceBuffer srcBuffer(bufferSize);
  DeviceBuffer dstBuffer(bufferSize);

  auto src_d = static_cast<int*>(srcBuffer.get());
  auto dst_d = static_cast<int*>(dstBuffer.get());

  const int numBlocks = 1;
  const int blockSize = 32;
  const size_t nbytes = 0; // Zero-byte transfer

  // Initialize destination buffer with a known pattern to verify it remains
  // unchanged
  const int initialValue = 999;
  test::fillBuffer(dst_d, initialValue, numInts);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  if (globalRank == 0) {
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    test::testSend(p2p, src_d, nbytes, numBlocks, blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  } else {
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    test::testRecv(p2p, dst_d, nbytes, numBlocks, blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Verify that the destination buffer was NOT modified (since zero bytes
    // transferred)
    std::vector<int> hostBuffer(numInts);
    CUDACHECK_TEST(cudaMemcpy(
        hostBuffer.data(), dst_d, bufferSize, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < numInts; i++) {
      EXPECT_EQ(hostBuffer[i], initialValue)
          << "Zero-byte transfer modified buffer at index " << i
          << ": expected " << initialValue << ", got " << hostBuffer[i];
      if (hostBuffer[i] != initialValue) {
        break;
      }
    }
  }

  XLOGF(INFO, "Rank {}: Zero-byte transfer test completed", globalRank);
}

// =============================================================================
// Multiple Sends in Single Kernel Test
// =============================================================================

TEST_F(P2pNvlTransportTestFixture, MultiSendInKernel) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const size_t dataBufferSize = 512 * 1024; // 512KB staging buffer
  const size_t nbytesPerSend = 256 * 1024; // 256KB per send
  const int numSends = 16;
  const size_t totalBytes = nbytesPerSend * numSends;

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 1024,
      .pipelineDepth = 4,
  };

  TransportTestHelper helper(globalRank, numRanks, localRank, config);
  auto p2p = helper.getDevicePtr();

  DeviceBuffer srcBuffer(totalBytes);
  DeviceBuffer dstBuffer(totalBytes);

  auto src_d = static_cast<int*>(srcBuffer.get());
  auto dst_d = static_cast<int*>(dstBuffer.get());

  const int numBlocks = 4;
  const int blockSize = 128;

  // Fill source buffer with different values for each segment
  const size_t intsPerSend = nbytesPerSend / sizeof(int);
  for (int i = 0; i < numSends; i++) {
    test::fillBuffer(src_d + i * intsPerSend, 100 + i, intsPerSend);
  }
  CUDACHECK_TEST(cudaDeviceSynchronize());

  if (helper.globalRank() == 0) {
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    // Single kernel launch that does multiple sends
    test::testMultiSend(
        p2p, src_d, nbytesPerSend, numSends, numBlocks, blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    std::cout << "Rank 0: MultiSendInKernel test completed" << std::endl;
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  } else {
    CUDACHECK_TEST(cudaMemset(dst_d, 0, totalBytes));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    // Single kernel launch that does multiple recvs
    test::testMultiRecv(
        p2p, dst_d, nbytesPerSend, numSends, numBlocks, blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    std::cout << "Rank 1: MultiRecvKernel test completed" << std::endl;
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Verify each segment
    std::vector<int> hostBuffer(intsPerSend);
    for (int i = 0; i < numSends; i++) {
      CUDACHECK_TEST(cudaMemcpy(
          hostBuffer.data(),
          dst_d + i * intsPerSend,
          nbytesPerSend,
          cudaMemcpyDeviceToHost));

      const int expectedValue = 100 + i;
      for (size_t j = 0; j < intsPerSend; j++) {
        EXPECT_EQ(hostBuffer[j], expectedValue)
            << "Segment " << i << ", index " << j << ": expected "
            << expectedValue << ", got " << hostBuffer[j];
        if (hostBuffer[j] != expectedValue) {
          break;
        }
      }
    }
  }

  XLOGF(INFO, "Rank {}: MultiSendInKernel test completed", helper.globalRank());
}

// =============================================================================
// Multiple Recvs in Single Kernel Test
// =============================================================================

TEST_F(P2pNvlTransportTestFixture, MultiRecvInKernel) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const size_t dataBufferSize = 512 * 1024; // 512KB staging buffer
  const size_t nbytesPerRecv = 128 * 1024; // 128KB per recv
  const int numRecvs = 8;
  const size_t totalBytes = nbytesPerRecv * numRecvs;

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 512,
      .pipelineDepth = 4,
  };

  TransportTestHelper helper(globalRank, numRanks, localRank, config);
  auto p2p = helper.getDevicePtr();

  DeviceBuffer srcBuffer(totalBytes);
  DeviceBuffer dstBuffer(totalBytes);

  auto src_d = static_cast<int*>(srcBuffer.get());
  auto dst_d = static_cast<int*>(dstBuffer.get());

  const int numBlocks = 2;
  const int blockSize = 64;

  // Fill source buffer with unique pattern
  const size_t intsPerRecv = nbytesPerRecv / sizeof(int);
  for (int i = 0; i < numRecvs; i++) {
    test::fillBuffer(src_d + i * intsPerRecv, 200 + i, intsPerRecv);
  }
  CUDACHECK_TEST(cudaDeviceSynchronize());

  if (helper.globalRank() == 0) {
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    test::testMultiSend(
        p2p, src_d, nbytesPerRecv, numRecvs, numBlocks, blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  } else {
    CUDACHECK_TEST(cudaMemset(dst_d, 0, totalBytes));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    test::testMultiRecv(
        p2p, dst_d, nbytesPerRecv, numRecvs, numBlocks, blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Verify each segment
    std::vector<int> hostBuffer(intsPerRecv);
    for (int i = 0; i < numRecvs; i++) {
      CUDACHECK_TEST(cudaMemcpy(
          hostBuffer.data(),
          dst_d + i * intsPerRecv,
          nbytesPerRecv,
          cudaMemcpyDeviceToHost));

      const int expectedValue = 200 + i;
      for (size_t j = 0; j < intsPerRecv; j++) {
        EXPECT_EQ(hostBuffer[j], expectedValue)
            << "Segment " << i << ", index " << j << ": expected "
            << expectedValue << ", got " << hostBuffer[j];
        if (hostBuffer[j] != expectedValue) {
          break;
        }
      }
    }
  }

  XLOGF(INFO, "Rank {}: MultiRecvInKernel test completed", helper.globalRank());
}

// =============================================================================
// Simultaneous Send+Recv in Single Kernel Test
// =============================================================================

TEST_F(P2pNvlTransportTestFixture, SimultaneousSendRecvInKernel) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const size_t dataBufferSize = 1024 * 1024; // 1MB staging buffer
  const size_t nbytes = 2 * 1024 * 1024; // 2MB transfer each direction

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 1024,
      .pipelineDepth = 4,
  };

  TransportTestHelper helper(globalRank, numRanks, localRank, config);
  auto p2p = helper.getDevicePtr();

  const size_t numInts = nbytes / sizeof(int);

  // Each rank has send and receive buffers
  DeviceBuffer sendBuffer(nbytes);
  DeviceBuffer recvBuffer(nbytes);

  auto send_d = static_cast<int*>(sendBuffer.get());
  auto recv_d = static_cast<int*>(recvBuffer.get());

  const int numBlocks = 4;
  const int blockSize = 128;

  // Each rank uses unique values
  const int sendValue = 300 + globalRank;
  const int expectedRecvValue = 300 + helper.peerRank();

  test::fillBuffer(send_d, sendValue, numInts);
  CUDACHECK_TEST(cudaMemset(recv_d, 0, nbytes));
  CUDACHECK_TEST(cudaDeviceSynchronize());

  XLOGF(
      INFO,
      "Rank {}: Simulatenous test - sending {}, expecting {}",
      globalRank,
      sendValue,
      expectedRecvValue);

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Both ranks do send+recv in a single kernel, but in opposite order
  // to avoid deadlock: rank 0 sends then recvs, rank 1 recvs then sends
  if (helper.globalRank() == 0) {
    test::testSendRecv(p2p, send_d, recv_d, nbytes, numBlocks, blockSize);
  } else {
    test::testRecvSend(p2p, recv_d, send_d, nbytes, numBlocks, blockSize);
  }
  CUDACHECK_TEST(cudaDeviceSynchronize());

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Verify received data
  std::vector<int> hostBuffer(numInts);
  CUDACHECK_TEST(
      cudaMemcpy(hostBuffer.data(), recv_d, nbytes, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < numInts; i++) {
    EXPECT_EQ(hostBuffer[i], expectedRecvValue)
        << "Rank " << globalRank << ": Mismatch at index " << i << ": expected "
        << expectedRecvValue << ", got " << hostBuffer[i];
    if (hostBuffer[i] != expectedRecvValue) {
      break;
    }
  }

  XLOGF(
      INFO,
      "Rank {}: SimultaneousSendRecvInKernel test completed",
      helper.globalRank());
}

// =============================================================================
// Parameterized Test Fixture for Weighted Partition Send/Recv
// =============================================================================
// Tests unequal send/recv partitioning with weighted splits

struct WeightedPartitionParams {
  uint32_t sendWeight;
  uint32_t recvWeight;
  std::string name;
};

class WeightedPartitionTestFixture
    : public MpiBaseTestFixture,
      public ::testing::WithParamInterface<WeightedPartitionParams> {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }
};

TEST_P(WeightedPartitionTestFixture, SendRecv) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const auto& params = GetParam();
  const size_t nbytes = 2 * 1024 * 1024; // 2MB
  const int numBlocks = 4;
  const int blockSize = 128;

  const size_t dataBufferSize = 1024 * 1024; // 1MB staging buffer
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 1024,
      .pipelineDepth = 4,
  };

  TransportTestHelper helper(globalRank, numRanks, localRank, config);
  auto p2p = helper.getDevicePtr();

  const size_t numInts = nbytes / sizeof(int);

  DeviceBuffer sendBuffer(nbytes);
  DeviceBuffer recvBuffer(nbytes);

  auto send_d = static_cast<int*>(sendBuffer.get());
  auto recv_d = static_cast<int*>(recvBuffer.get());

  const int sendValue = 400 + globalRank;
  const int expectedRecvValue = 400 + helper.peerRank();

  test::fillBuffer(send_d, sendValue, numInts);
  CUDACHECK_TEST(cudaMemset(recv_d, 0, nbytes));
  CUDACHECK_TEST(cudaDeviceSynchronize());

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Rank 0 sends then recvs, rank 1 recvs then sends
  if (helper.globalRank() == 0) {
    test::testWeightedSendRecv(
        p2p,
        send_d,
        recv_d,
        nbytes,
        numBlocks,
        blockSize,
        params.sendWeight,
        params.recvWeight);
  } else {
    test::testWeightedRecvSend(
        p2p,
        recv_d,
        send_d,
        nbytes,
        numBlocks,
        blockSize,
        params.sendWeight,
        params.recvWeight);
  }
  CUDACHECK_TEST(cudaDeviceSynchronize());

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // Verify received data
  std::vector<int> hostBuffer(numInts);
  CUDACHECK_TEST(
      cudaMemcpy(hostBuffer.data(), recv_d, nbytes, cudaMemcpyDeviceToHost));

  for (size_t i = 0; i < numInts; i++) {
    EXPECT_EQ(hostBuffer[i], expectedRecvValue)
        << "Rank " << globalRank << ": Mismatch at index " << i << ": expected "
        << expectedRecvValue << ", got " << hostBuffer[i];
    if (hostBuffer[i] != expectedRecvValue) {
      break;
    }
  }

  XLOGF(
      INFO,
      "Rank {}: Weighted partition test '{}' completed",
      globalRank,
      params.name);
}

std::string weightedPartitionParamName(
    const ::testing::TestParamInfo<WeightedPartitionParams>& info) {
  return info.param.name;
}

INSTANTIATE_TEST_SUITE_P(
    WeightedPartitionVariations,
    WeightedPartitionTestFixture,
    ::testing::Values(
        WeightedPartitionParams{
            .sendWeight = 3,
            .recvWeight = 1,
            .name = "Send3_Recv1"},
        WeightedPartitionParams{
            .sendWeight = 1,
            .recvWeight = 3,
            .name = "Send1_Recv3"},
        // Extreme case: 1:99 split - tests that at least 1 warp is assigned to
        // send
        WeightedPartitionParams{
            .sendWeight = 1,
            .recvWeight = 99,
            .name = "Send1_Recv99"}),
    weightedPartitionParamName);

// =============================================================================
// Parameterized Test Fixture for Pipeline Depth Variation
// =============================================================================
// Test different pipelineDepth values to verify pipelining works correctly:
// - pipelineDepth = 1 (no pipelining, sequential)
// - pipelineDepth = 2 (minimal pipelining)
// - pipelineDepth = 4 (default)
// - pipelineDepth = 8 (deep pipelining)

struct PipelineDepthParams {
  size_t pipelineDepth;
  size_t nbytes;
  size_t dataBufferSize;
  size_t chunkSize;
  bool useCudaGraph;
  std::string name;
};

class PipelineDepthTestFixture
    : public MpiBaseTestFixture,
      public ::testing::WithParamInterface<PipelineDepthParams> {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }
};

TEST_P(PipelineDepthTestFixture, SendRecv) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const auto& params = GetParam();
  XLOGF(
      INFO,
      "Running pipeline depth test: {} (pipelineDepth={}, nbytes={}, bufferSize={}, chunkSize={}, cudaGraph={})",
      params.name,
      params.pipelineDepth,
      params.nbytes,
      params.dataBufferSize,
      params.chunkSize,
      params.useCudaGraph);

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = params.dataBufferSize,
      .chunkSize = params.chunkSize,
      .pipelineDepth = params.pipelineDepth,
  };

  // Calculate expected number of steps to verify pipelining
  const size_t totalSteps =
      (params.nbytes + params.dataBufferSize - 1) / params.dataBufferSize;
  XLOGF(
      INFO,
      "Rank {}: Transfer will use {} steps with pipeline depth {}",
      globalRank,
      totalSteps,
      params.pipelineDepth);

  TransportTestHelper helper(globalRank, numRanks, localRank, config);
  runBasicSendRecvTest(
      helper,
      params.nbytes,
      4,
      128,
      1,
      test::GroupType::WARP,
      params.useCudaGraph);

  XLOGF(
      INFO,
      "Rank {}: Pipeline depth test '{}' completed",
      globalRank,
      params.name);
}

std::string pipelineDepthParamName(
    const ::testing::TestParamInfo<PipelineDepthParams>& info) {
  return info.param.name + (info.param.useCudaGraph ? "_CudaGraph" : "");
}

INSTANTIATE_TEST_SUITE_P(
    PipelineDepthVariations,
    PipelineDepthTestFixture,
    ::testing::Values(
        // pipelineDepth=1: No pipelining, sequential execution
        // 4MB transfer with 512KB buffer = 8 steps, all executed sequentially
        PipelineDepthParams{
            .pipelineDepth = 1,
            .nbytes = 4 * 1024 * 1024,
            .dataBufferSize = 512 * 1024,
            .chunkSize = 1024,
            .useCudaGraph = false,
            .name = "Depth1_Sequential_8Steps"},
        PipelineDepthParams{
            .pipelineDepth = 1,
            .nbytes = 4 * 1024 * 1024,
            .dataBufferSize = 512 * 1024,
            .chunkSize = 1024,
            .useCudaGraph = true,
            .name = "Depth1_Sequential_8Steps"},
        // pipelineDepth=2: Minimal pipelining
        // 4MB transfer with 512KB buffer = 8 steps, using 2 slots
        PipelineDepthParams{
            .pipelineDepth = 2,
            .nbytes = 4 * 1024 * 1024,
            .dataBufferSize = 512 * 1024,
            .chunkSize = 1024,
            .useCudaGraph = false,
            .name = "Depth2_MinimalPipeline_8Steps"},
        PipelineDepthParams{
            .pipelineDepth = 2,
            .nbytes = 4 * 1024 * 1024,
            .dataBufferSize = 512 * 1024,
            .chunkSize = 1024,
            .useCudaGraph = true,
            .name = "Depth2_MinimalPipeline_8Steps"},
        // pipelineDepth=4: Default pipelining
        // 4MB transfer with 512KB buffer = 8 steps, using 4 slots
        PipelineDepthParams{
            .pipelineDepth = 4,
            .nbytes = 4 * 1024 * 1024,
            .dataBufferSize = 512 * 1024,
            .chunkSize = 1024,
            .useCudaGraph = false,
            .name = "Depth4_DefaultPipeline_8Steps"},
        PipelineDepthParams{
            .pipelineDepth = 4,
            .nbytes = 4 * 1024 * 1024,
            .dataBufferSize = 512 * 1024,
            .chunkSize = 1024,
            .useCudaGraph = true,
            .name = "Depth4_DefaultPipeline_8Steps"},
        // pipelineDepth=8: Deep pipelining
        // 4MB transfer with 512KB buffer = 8 steps, using all 8 slots
        PipelineDepthParams{
            .pipelineDepth = 8,
            .nbytes = 4 * 1024 * 1024,
            .dataBufferSize = 512 * 1024,
            .chunkSize = 1024,
            .useCudaGraph = false,
            .name = "Depth8_DeepPipeline_8Steps"},
        PipelineDepthParams{
            .pipelineDepth = 8,
            .nbytes = 4 * 1024 * 1024,
            .dataBufferSize = 512 * 1024,
            .chunkSize = 1024,
            .useCudaGraph = true,
            .name = "Depth8_DeepPipeline_8Steps"},
        // pipelineDepth=8 with more steps than depth
        // 8MB transfer with 512KB buffer = 16 steps, using 8 slots (slot reuse)
        PipelineDepthParams{
            .pipelineDepth = 8,
            .nbytes = 8 * 1024 * 1024,
            .dataBufferSize = 512 * 1024,
            .chunkSize = 1024,
            .useCudaGraph = false,
            .name = "Depth8_SlotReuse_16Steps"},
        PipelineDepthParams{
            .pipelineDepth = 8,
            .nbytes = 8 * 1024 * 1024,
            .dataBufferSize = 512 * 1024,
            .chunkSize = 1024,
            .useCudaGraph = true,
            .name = "Depth8_SlotReuse_16Steps"}),
    pipelineDepthParamName);

// =============================================================================
// Parameterized Test Fixture for Pipeline Slot Reuse
// =============================================================================
// Tests that pipeline slots are correctly reused when totalSteps >
// pipelineDepth:
// - Verifies stepId % pipelineDepth indexing works correctly
// - Verifies state buffer is properly reset when slots are reused
// - Ensures data integrity across multiple reuses of the same slot

struct PipelineSaturationParams {
  size_t pipelineDepth;
  size_t totalSteps;
  size_t chunkSize;
  bool useCudaGraph;
  std::string name;
};

class PipelineSaturationTestFixture
    : public MpiBaseTestFixture,
      public ::testing::WithParamInterface<PipelineSaturationParams> {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }
};

TEST_P(PipelineSaturationTestFixture, SendRecv) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const auto& params = GetParam();
  // Calculate buffer size and total bytes to achieve desired number of steps
  const size_t dataBufferSize = 256 * 1024; // 256KB per step
  const size_t nbytes = params.totalSteps * dataBufferSize;

  XLOGF(
      INFO,
      "Running pipeline saturation test: {} (pipelineDepth={}, steps={}, nbytes={}, cudaGraph={})",
      params.name,
      params.pipelineDepth,
      params.totalSteps,
      nbytes,
      params.useCudaGraph);

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = params.chunkSize,
      .pipelineDepth = params.pipelineDepth,
  };

  TransportTestHelper helper(globalRank, numRanks, localRank, config);
  runBasicSendRecvTest(
      helper, nbytes, 4, 128, 1, test::GroupType::WARP, params.useCudaGraph);

  XLOGF(
      INFO,
      "Rank {}: Pipeline saturation test '{}' completed",
      globalRank,
      params.name);
}

std::string pipelineSaturationParamName(
    const ::testing::TestParamInfo<PipelineSaturationParams>& info) {
  return info.param.name + (info.param.useCudaGraph ? "_CudaGraph" : "");
}

INSTANTIATE_TEST_SUITE_P(
    PipelineSlotReuseVariations,
    PipelineSaturationTestFixture,
    ::testing::Values(
        // pipelineDepth=2, 10 steps: each slot used 5 times (steps 0,2,4,6,8
        // and 1,3,5,7,9)
        PipelineSaturationParams{
            .pipelineDepth = 2,
            .totalSteps = 10,
            .chunkSize = 1024,
            .useCudaGraph = false,
            .name = "Depth2_10Steps_5xReuse"},
        PipelineSaturationParams{
            .pipelineDepth = 2,
            .totalSteps = 10,
            .chunkSize = 1024,
            .useCudaGraph = true,
            .name = "Depth2_10Steps_5xReuse"},
        // pipelineDepth=2, 16 steps: each slot used 8 times
        PipelineSaturationParams{
            .pipelineDepth = 2,
            .totalSteps = 16,
            .chunkSize = 1024,
            .useCudaGraph = false,
            .name = "Depth2_16Steps_8xReuse"},
        PipelineSaturationParams{
            .pipelineDepth = 2,
            .totalSteps = 16,
            .chunkSize = 1024,
            .useCudaGraph = true,
            .name = "Depth2_16Steps_8xReuse"},
        // pipelineDepth=3, 12 steps: each slot used 4 times
        PipelineSaturationParams{
            .pipelineDepth = 3,
            .totalSteps = 12,
            .chunkSize = 1024,
            .useCudaGraph = false,
            .name = "Depth3_12Steps_4xReuse"},
        PipelineSaturationParams{
            .pipelineDepth = 3,
            .totalSteps = 12,
            .chunkSize = 1024,
            .useCudaGraph = true,
            .name = "Depth3_12Steps_4xReuse"},
        // pipelineDepth=4, 20 steps: each slot used 5 times
        PipelineSaturationParams{
            .pipelineDepth = 4,
            .totalSteps = 20,
            .chunkSize = 512,
            .useCudaGraph = false,
            .name = "Depth4_20Steps_5xReuse"},
        PipelineSaturationParams{
            .pipelineDepth = 4,
            .totalSteps = 20,
            .chunkSize = 512,
            .useCudaGraph = true,
            .name = "Depth4_20Steps_5xReuse"}),
    pipelineSaturationParamName);

// =============================================================================
// Parameterized Test Fixture for Chunk Count Edge Cases
// =============================================================================
// Test edge cases in chunk distribution:
// - numChunks < numWarps (some warps have no work)
// - numChunks = 1 (single chunk)
// - numChunks = numWarps (exactly 1 chunk per warp)
// - Very small transfer (< chunkSize)

struct ChunkCountEdgeCaseParams {
  size_t nbytes;
  size_t dataBufferSize;
  size_t chunkSize;
  int numBlocks;
  int blockSize;
  bool useCudaGraph;
  std::string name;
};

class ChunkCountEdgeCaseTestFixture
    : public MpiBaseTestFixture,
      public ::testing::WithParamInterface<ChunkCountEdgeCaseParams> {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }
};

TEST_P(ChunkCountEdgeCaseTestFixture, SendRecv) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const auto& params = GetParam();

  // Calculate chunk distribution info
  const size_t chunksPerStep =
      (params.dataBufferSize + params.chunkSize - 1) / params.chunkSize;
  const int numWarps =
      (params.numBlocks * params.blockSize + 31) / 32; // Approximate

  XLOGF(
      INFO,
      "Running chunk edge case test: {} (nbytes={}, chunkSize={}, ~{} chunks, ~{} warps, cudaGraph={})",
      params.name,
      params.nbytes,
      params.chunkSize,
      chunksPerStep,
      numWarps,
      params.useCudaGraph);

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = params.dataBufferSize,
      .chunkSize = params.chunkSize,
      .pipelineDepth = 4,
  };

  TransportTestHelper helper(globalRank, numRanks, localRank, config);
  runBasicSendRecvTest(
      helper,
      params.nbytes,
      params.numBlocks,
      params.blockSize,
      1,
      test::GroupType::WARP,
      params.useCudaGraph);

  XLOGF(
      INFO,
      "Rank {}: Chunk edge case test '{}' completed",
      globalRank,
      params.name);
}

std::string chunkCountEdgeCaseParamName(
    const ::testing::TestParamInfo<ChunkCountEdgeCaseParams>& info) {
  return info.param.name + (info.param.useCudaGraph ? "_CudaGraph" : "");
}

INSTANTIATE_TEST_SUITE_P(
    ChunkCountEdgeCases,
    ChunkCountEdgeCaseTestFixture,
    ::testing::Values(
        // numChunks < numWarps: 4 chunks with 64 warps (8 blocks × 256 threads)
        // Many warps will have no work
        ChunkCountEdgeCaseParams{
            .nbytes = 4 * 1024, // 4KB
            .dataBufferSize = 4 * 1024,
            .chunkSize = 1024, // 4 chunks
            .numBlocks = 8,
            .blockSize = 256, // 64 warps
            .useCudaGraph = false,
            .name = "FewChunks_4Chunks_64Warps"},
        ChunkCountEdgeCaseParams{
            .nbytes = 4 * 1024, // 4KB
            .dataBufferSize = 4 * 1024,
            .chunkSize = 1024, // 4 chunks
            .numBlocks = 8,
            .blockSize = 256, // 64 warps
            .useCudaGraph = true,
            .name = "FewChunks_4Chunks_64Warps"},
        // numChunks = 1: Single chunk transfer
        ChunkCountEdgeCaseParams{
            .nbytes = 512, // 512 bytes
            .dataBufferSize = 1024,
            .chunkSize = 1024, // 1 chunk (transfer < chunkSize)
            .numBlocks = 4,
            .blockSize = 128,
            .useCudaGraph = false,
            .name = "SingleChunk_512Bytes"},
        ChunkCountEdgeCaseParams{
            .nbytes = 512, // 512 bytes
            .dataBufferSize = 1024,
            .chunkSize = 1024, // 1 chunk (transfer < chunkSize)
            .numBlocks = 4,
            .blockSize = 128,
            .useCudaGraph = true,
            .name = "SingleChunk_512Bytes"},
        // numChunks = 1 with larger chunk
        ChunkCountEdgeCaseParams{
            .nbytes = 4 * 1024, // 4KB
            .dataBufferSize = 4 * 1024,
            .chunkSize = 4 * 1024, // 1 chunk
            .numBlocks = 4,
            .blockSize = 128,
            .useCudaGraph = false,
            .name = "SingleChunk_4KB"},
        ChunkCountEdgeCaseParams{
            .nbytes = 4 * 1024, // 4KB
            .dataBufferSize = 4 * 1024,
            .chunkSize = 4 * 1024, // 1 chunk
            .numBlocks = 4,
            .blockSize = 128,
            .useCudaGraph = true,
            .name = "SingleChunk_4KB"},
        // numChunks = numWarps: Exactly 1 chunk per warp
        // 16 chunks with 16 warps (4 blocks × 128 threads = 16 warps)
        ChunkCountEdgeCaseParams{
            .nbytes = 16 * 1024, // 16KB
            .dataBufferSize = 16 * 1024,
            .chunkSize = 1024, // 16 chunks
            .numBlocks = 4,
            .blockSize = 128, // 16 warps
            .useCudaGraph = false,
            .name = "ExactMatch_16Chunks_16Warps"},
        ChunkCountEdgeCaseParams{
            .nbytes = 16 * 1024, // 16KB
            .dataBufferSize = 16 * 1024,
            .chunkSize = 1024, // 16 chunks
            .numBlocks = 4,
            .blockSize = 128, // 16 warps
            .useCudaGraph = true,
            .name = "ExactMatch_16Chunks_16Warps"},
        // Very small transfer (< chunkSize)
        ChunkCountEdgeCaseParams{
            .nbytes = 64, // 64 bytes (much smaller than chunk)
            .dataBufferSize = 1024,
            .chunkSize = 256,
            .numBlocks = 2,
            .blockSize = 64,
            .useCudaGraph = false,
            .name = "VerySmall_64Bytes"},
        ChunkCountEdgeCaseParams{
            .nbytes = 64, // 64 bytes (much smaller than chunk)
            .dataBufferSize = 1024,
            .chunkSize = 256,
            .numBlocks = 2,
            .blockSize = 64,
            .useCudaGraph = true,
            .name = "VerySmall_64Bytes"},
        // Another very small transfer
        ChunkCountEdgeCaseParams{
            .nbytes = 128, // 128 bytes
            .dataBufferSize = 1024,
            .chunkSize = 512,
            .numBlocks = 2,
            .blockSize = 64,
            .useCudaGraph = false,
            .name = "VerySmall_128Bytes"},
        ChunkCountEdgeCaseParams{
            .nbytes = 128, // 128 bytes
            .dataBufferSize = 1024,
            .chunkSize = 512,
            .numBlocks = 2,
            .blockSize = 64,
            .useCudaGraph = true,
            .name = "VerySmall_128Bytes"},
        // Edge case: nbytes not aligned to chunk or vector size
        ChunkCountEdgeCaseParams{
            .nbytes = 100, // Non-aligned size
            .dataBufferSize = 1024,
            .chunkSize = 64,
            .numBlocks = 2,
            .blockSize = 64,
            .useCudaGraph = false,
            .name = "NonAligned_100Bytes"},
        ChunkCountEdgeCaseParams{
            .nbytes = 100, // Non-aligned size
            .dataBufferSize = 1024,
            .chunkSize = 64,
            .numBlocks = 2,
            .blockSize = 64,
            .useCudaGraph = true,
            .name = "NonAligned_100Bytes"}),
    chunkCountEdgeCaseParamName);

// =============================================================================
// Parameterized Test Fixture for Large Transfers (Stress Test)
// =============================================================================
// Stress test with large transfers:
// - 64MB, 128MB, 256MB transfers
// - Exercises full pipeline depth, many steps, many chunks

struct LargeTransferParams {
  size_t nbytes;
  size_t dataBufferSize;
  size_t chunkSize;
  size_t pipelineDepth;
  bool useCudaGraph;
  std::string name;
};

class LargeTransferTestFixture
    : public MpiBaseTestFixture,
      public ::testing::WithParamInterface<LargeTransferParams> {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }
};

TEST_P(LargeTransferTestFixture, SendRecv) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const auto& params = GetParam();

  // Calculate transfer statistics
  const size_t totalSteps =
      (params.nbytes + params.dataBufferSize - 1) / params.dataBufferSize;
  const size_t chunksPerStep =
      (params.dataBufferSize + params.chunkSize - 1) / params.chunkSize;

  XLOGF(
      INFO,
      "Running large transfer test: {} (nbytes={}MB, {} steps, {} chunks/step, cudaGraph={})",
      params.name,
      params.nbytes / (1024 * 1024),
      totalSteps,
      chunksPerStep,
      params.useCudaGraph);

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = params.dataBufferSize,
      .chunkSize = params.chunkSize,
      .pipelineDepth = params.pipelineDepth,
  };

  TransportTestHelper helper(globalRank, numRanks, localRank, config);
  runBasicSendRecvTest(
      helper,
      params.nbytes,
      8,
      256,
      1,
      test::GroupType::WARP,
      params.useCudaGraph);

  XLOGF(
      INFO,
      "Rank {}: Large transfer test '{}' completed",
      globalRank,
      params.name);
}

std::string largeTransferParamName(
    const ::testing::TestParamInfo<LargeTransferParams>& info) {
  return info.param.name + (info.param.useCudaGraph ? "_CudaGraph" : "");
}

INSTANTIATE_TEST_SUITE_P(
    LargeTransferVariations,
    LargeTransferTestFixture,
    ::testing::Values(
        // 64MB transfer with 8MB buffer = 8 steps
        LargeTransferParams{
            .nbytes = 64 * 1024 * 1024,
            .dataBufferSize = 8 * 1024 * 1024,
            .chunkSize = 4 * 1024,
            .pipelineDepth = 4,
            .useCudaGraph = false,
            .name = "Large_64MB_8MBBuffer"},
        LargeTransferParams{
            .nbytes = 64 * 1024 * 1024,
            .dataBufferSize = 8 * 1024 * 1024,
            .chunkSize = 4 * 1024,
            .pipelineDepth = 4,
            .useCudaGraph = true,
            .name = "Large_64MB_8MBBuffer"},
        // 128MB transfer with 8MB buffer = 16 steps
        LargeTransferParams{
            .nbytes = 128 * 1024 * 1024,
            .dataBufferSize = 8 * 1024 * 1024,
            .chunkSize = 4 * 1024,
            .pipelineDepth = 4,
            .useCudaGraph = false,
            .name = "Large_128MB_8MBBuffer"},
        LargeTransferParams{
            .nbytes = 128 * 1024 * 1024,
            .dataBufferSize = 8 * 1024 * 1024,
            .chunkSize = 4 * 1024,
            .pipelineDepth = 4,
            .useCudaGraph = true,
            .name = "Large_128MB_8MBBuffer"},
        // 256MB transfer with 8MB buffer = 32 steps
        LargeTransferParams{
            .nbytes = 256 * 1024 * 1024,
            .dataBufferSize = 8 * 1024 * 1024,
            .chunkSize = 4 * 1024,
            .pipelineDepth = 4,
            .useCudaGraph = false,
            .name = "Large_256MB_8MBBuffer"},
        LargeTransferParams{
            .nbytes = 256 * 1024 * 1024,
            .dataBufferSize = 8 * 1024 * 1024,
            .chunkSize = 4 * 1024,
            .pipelineDepth = 4,
            .useCudaGraph = true,
            .name = "Large_256MB_8MBBuffer"},
        // 64MB transfer with smaller buffer = more steps
        LargeTransferParams{
            .nbytes = 64 * 1024 * 1024,
            .dataBufferSize = 4 * 1024 * 1024,
            .chunkSize = 2 * 1024,
            .pipelineDepth = 8,
            .useCudaGraph = false,
            .name = "Large_64MB_4MBBuffer_DeepPipeline"},
        LargeTransferParams{
            .nbytes = 64 * 1024 * 1024,
            .dataBufferSize = 4 * 1024 * 1024,
            .chunkSize = 2 * 1024,
            .pipelineDepth = 8,
            .useCudaGraph = true,
            .name = "Large_64MB_4MBBuffer_DeepPipeline"},
        // 128MB transfer with deep pipeline
        LargeTransferParams{
            .nbytes = 128 * 1024 * 1024,
            .dataBufferSize = 4 * 1024 * 1024,
            .chunkSize = 2 * 1024,
            .pipelineDepth = 8,
            .useCudaGraph = false,
            .name = "Large_128MB_4MBBuffer_DeepPipeline"},
        LargeTransferParams{
            .nbytes = 128 * 1024 * 1024,
            .dataBufferSize = 4 * 1024 * 1024,
            .chunkSize = 2 * 1024,
            .pipelineDepth = 8,
            .useCudaGraph = true,
            .name = "Large_128MB_4MBBuffer_DeepPipeline"}),
    largeTransferParamName);

// =============================================================================
// Parameterized Test Fixture for Asymmetric Group Configurations
// =============================================================================
// Tests that sender and receiver can use different thread group configurations
// This validates that the protocol works across asymmetric kernel launches.

struct AsymmetricGroupParams {
  test::GroupType senderGroupType;
  int senderNumBlocks;
  int senderBlockSize;
  test::GroupType receiverGroupType;
  int receiverNumBlocks;
  int receiverBlockSize;
  bool useCudaGraph;
  std::string name;
};

class AsymmetricGroupTestFixture
    : public MpiBaseTestFixture,
      public ::testing::WithParamInterface<AsymmetricGroupParams> {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }
};

TEST_P(AsymmetricGroupTestFixture, SendRecv) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const auto& params = GetParam();
  XLOGF(
      INFO,
      "Running asymmetric group test: {} (sender: {} {}x{}, receiver: {} {}x{}, cudaGraph={})",
      params.name,
      params.senderGroupType == test::GroupType::WARP ? "WARP" : "BLOCK",
      params.senderNumBlocks,
      params.senderBlockSize,
      params.receiverGroupType == test::GroupType::WARP ? "WARP" : "BLOCK",
      params.receiverNumBlocks,
      params.receiverBlockSize,
      params.useCudaGraph);

  const size_t dataBufferSize = 1024 * 1024; // 1MB staging buffer
  const size_t nbytes = 4 * 1024 * 1024; // 4MB total transfer
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 1024,
      .pipelineDepth = 4,
  };

  TransportTestHelper helper(globalRank, numRanks, localRank, config);
  auto p2p = helper.getDevicePtr();

  const size_t numInts = nbytes / sizeof(int);

  DeviceBuffer srcBuffer(nbytes);
  DeviceBuffer dstBuffer(nbytes);

  auto src_d = static_cast<int*>(srcBuffer.get());
  auto dst_d = static_cast<int*>(dstBuffer.get());

  if (!params.useCudaGraph) {
    // Direct kernel launch mode
    if (globalRank == 0) {
      // Sender
      test::fillBuffer(src_d, 42, numInts);
      CUDACHECK_TEST(cudaDeviceSynchronize());
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
      test::testSend(
          p2p,
          src_d,
          nbytes,
          params.senderNumBlocks,
          params.senderBlockSize,
          params.senderGroupType);
      CUDACHECK_TEST(cudaDeviceSynchronize());
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    } else {
      // Receiver
      CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
      test::testRecv(
          p2p,
          dst_d,
          nbytes,
          params.receiverNumBlocks,
          params.receiverBlockSize,
          params.receiverGroupType);
      CUDACHECK_TEST(cudaDeviceSynchronize());
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      // Verify received data
      std::vector<int> hostBuffer(numInts);
      CUDACHECK_TEST(
          cudaMemcpy(hostBuffer.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

      for (size_t i = 0; i < numInts; i++) {
        EXPECT_EQ(hostBuffer[i], 42)
            << "Rank " << globalRank << ": Mismatch at index " << i
            << ": expected 42, got " << hostBuffer[i];
        if (hostBuffer[i] != 42) {
          break;
        }
      }
    }
  } else {
    // CUDA graph mode
    cudaStream_t stream;
    CUDACHECK_TEST(cudaStreamCreate(&stream));

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    // Capture the send or recv kernel into a graph
    CUDACHECK_TEST(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    if (globalRank == 0) {
      test::testSend(
          p2p,
          src_d,
          nbytes,
          params.senderNumBlocks,
          params.senderBlockSize,
          params.senderGroupType,
          1,
          stream);
    } else {
      test::testRecv(
          p2p,
          dst_d,
          nbytes,
          params.receiverNumBlocks,
          params.receiverBlockSize,
          params.receiverGroupType,
          1,
          stream);
    }
    CUDACHECK_TEST(cudaStreamEndCapture(stream, &graph));
    CUDACHECK_TEST(
        cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    std::vector<int> hostBuffer(numInts);
    const int graphIter = 3;

    // Replay graph 3 times with different data patterns
    for (int iter = 0; iter < graphIter; iter++) {
      const int testValue = 42 + iter;

      if (globalRank == 0) {
        test::fillBuffer(src_d, testValue, numInts);
        CUDACHECK_TEST(cudaDeviceSynchronize());
      } else {
        CUDACHECK_TEST(cudaMemset(dst_d, 0, nbytes));
      }

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      CUDACHECK_TEST(cudaGraphLaunch(graphExec, stream));
      CUDACHECK_TEST(cudaStreamSynchronize(stream));

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      if (globalRank == 1) {
        CUDACHECK_TEST(cudaMemcpy(
            hostBuffer.data(), dst_d, nbytes, cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < numInts; i++) {
          EXPECT_EQ(hostBuffer[i], testValue)
              << "CudaGraph iter " << iter << ": Mismatch at index " << i
              << ": expected " << testValue << ", got " << hostBuffer[i];
          if (hostBuffer[i] != testValue) {
            break;
          }
        }
      }
    }

    CUDACHECK_TEST(cudaGraphExecDestroy(graphExec));
    CUDACHECK_TEST(cudaGraphDestroy(graph));
    CUDACHECK_TEST(cudaStreamDestroy(stream));
  }

  XLOGF(
      INFO,
      "Rank {}: Asymmetric group test '{}' completed",
      globalRank,
      params.name);
}

std::string asymmetricGroupParamName(
    const ::testing::TestParamInfo<AsymmetricGroupParams>& info) {
  return info.param.name + (info.param.useCudaGraph ? "_CudaGraph" : "");
}

INSTANTIATE_TEST_SUITE_P(
    AsymmetricGroupVariations,
    AsymmetricGroupTestFixture,
    ::testing::Values(
        // Sender uses WARP groups, receiver uses BLOCK groups
        AsymmetricGroupParams{
            .senderGroupType = test::GroupType::WARP,
            .senderNumBlocks = 4,
            .senderBlockSize = 128,
            .receiverGroupType = test::GroupType::BLOCK,
            .receiverNumBlocks = 4,
            .receiverBlockSize = 128,
            .useCudaGraph = false,
            .name = "SenderWarp_ReceiverBlock"},
        AsymmetricGroupParams{
            .senderGroupType = test::GroupType::WARP,
            .senderNumBlocks = 4,
            .senderBlockSize = 128,
            .receiverGroupType = test::GroupType::BLOCK,
            .receiverNumBlocks = 4,
            .receiverBlockSize = 128,
            .useCudaGraph = true,
            .name = "SenderWarp_ReceiverBlock"},
        // Sender uses BLOCK groups, receiver uses WARP groups
        AsymmetricGroupParams{
            .senderGroupType = test::GroupType::BLOCK,
            .senderNumBlocks = 4,
            .senderBlockSize = 128,
            .receiverGroupType = test::GroupType::WARP,
            .receiverNumBlocks = 4,
            .receiverBlockSize = 128,
            .useCudaGraph = false,
            .name = "SenderBlock_ReceiverWarp"},
        AsymmetricGroupParams{
            .senderGroupType = test::GroupType::BLOCK,
            .senderNumBlocks = 4,
            .senderBlockSize = 128,
            .receiverGroupType = test::GroupType::WARP,
            .receiverNumBlocks = 4,
            .receiverBlockSize = 128,
            .useCudaGraph = true,
            .name = "SenderBlock_ReceiverWarp"},
        // Different block configurations
        AsymmetricGroupParams{
            .senderGroupType = test::GroupType::WARP,
            .senderNumBlocks = 8,
            .senderBlockSize = 256,
            .receiverGroupType = test::GroupType::BLOCK,
            .receiverNumBlocks = 2,
            .receiverBlockSize = 512,
            .useCudaGraph = false,
            .name = "SenderWarp8x256_ReceiverBlock2x512"},
        AsymmetricGroupParams{
            .senderGroupType = test::GroupType::WARP,
            .senderNumBlocks = 8,
            .senderBlockSize = 256,
            .receiverGroupType = test::GroupType::BLOCK,
            .receiverNumBlocks = 2,
            .receiverBlockSize = 512,
            .useCudaGraph = true,
            .name = "SenderWarp8x256_ReceiverBlock2x512"},
        // Same group type but different configurations
        AsymmetricGroupParams{
            .senderGroupType = test::GroupType::WARP,
            .senderNumBlocks = 2,
            .senderBlockSize = 64,
            .receiverGroupType = test::GroupType::WARP,
            .receiverNumBlocks = 8,
            .receiverBlockSize = 256,
            .useCudaGraph = false,
            .name = "SenderWarp2x64_ReceiverWarp8x256"},
        AsymmetricGroupParams{
            .senderGroupType = test::GroupType::WARP,
            .senderNumBlocks = 2,
            .senderBlockSize = 64,
            .receiverGroupType = test::GroupType::WARP,
            .receiverNumBlocks = 8,
            .receiverBlockSize = 256,
            .useCudaGraph = true,
            .name = "SenderWarp2x64_ReceiverWarp8x256"}),
    asymmetricGroupParamName);

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
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = nbytes,
      .chunkSize = 1,
      .pipelineDepth = 1,
  };

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

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = params.nbytes,
      .chunkSize = 1,
      .pipelineDepth = 1,
  };

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
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = 1024,
      .pipelineDepth = 4,
  };

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

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = paddedSize,
      .chunkSize = 1,
      .pipelineDepth = 1,
  };

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

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = 4096,
      .chunkSize = 256,
      .pipelineDepth = 2,
      .ll128BufferSize = ll128_buffer_size(4096),
  };

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

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = 4096,
      .chunkSize = 256,
      .pipelineDepth = 2,
  };

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
      .dataBufferSize = 8 * 1024 * 1024, // 8MB slot
      .chunkSize = 8 * 1024 * 1024,
      .pipelineDepth = 2,
      .p2pBarrierCount = static_cast<std::size_t>(maxBlocks),
      .max_num_channels = maxBlocks,
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
// forward_group() Tests - 2-rank ring topology
// =============================================================================
// Topology: Rank 0 → Rank 1 → Rank 0
// - Rank 0: send_group to rank 1, recv_group from rank 1 (concurrent streams)
// - Rank 1: forward_group with pred==succ==same device (reads from localState_
//   staging where rank 0 wrote, writes to remoteState_ staging for rank 0's
//   recv)
// Uses a single transport between ranks — send uses remoteState_, recv uses
// localState_, so both directions coexist without conflict.

void runForwardGroupTest(
    int globalRank,
    int numRanks,
    int localRank,
    size_t nbytes,
    size_t dataBufferSize,
    size_t chunkSize,
    size_t pipelineDepth,
    bool useDualStateBuffer,
    int numBlocks = 4,
    int blockSize = 128,
    size_t dstOffset = 0) {
  if (numRanks != 2) {
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;

  // Single transport between rank 0 and rank 1 — supports bidirectional
  // communication. send_group uses remoteState_, recv_group uses localState_,
  // so both directions can coexist on the same device without conflict.
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = chunkSize,
      .pipelineDepth = pipelineDepth,
      .useDualStateBuffer = useDualStateBuffer,
  };

  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  MultiPeerNvlTransport transport(globalRank, numRanks, bootstrap, config);
  transport.exchange();

  // Build host-side device handle
  auto p2pHost = transport.buildP2pTransportDevice(peerRank);

  // Copy to device memory
  DeviceBuffer devP2p(sizeof(P2pNvlTransportDevice));
  CUDACHECK_TEST(cudaMemcpy(
      devP2p.get(),
      &p2pHost,
      sizeof(P2pNvlTransportDevice),
      cudaMemcpyHostToDevice));

  auto* p2pDev = static_cast<P2pNvlTransportDevice*>(devP2p.get());

  const size_t numInts = nbytes / sizeof(int);
  const int testValue = 77;

  DeviceBuffer srcBuffer(nbytes);
  DeviceBuffer dstRank0(nbytes); // rank 0's recv buffer
  // Rank 1's local forward output, padded to support unaligned dstOffset.
  DeviceBuffer dstRank1(nbytes + dstOffset);

  auto* src_d = static_cast<int*>(srcBuffer.get());
  auto* recvR0_d = static_cast<int*>(dstRank0.get());

  if (globalRank == 0) {
    // Fill source buffer
    test::fillBuffer(src_d, testValue, numInts);
    CUDACHECK_TEST(cudaMemset(recvR0_d, 0, nbytes));
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Launch send and recv concurrently on different streams to avoid
    // pipeline deadlock when totalSteps > pipelineDepth
    cudaStream_t sendStream, recvStream;
    CUDACHECK_TEST(cudaStreamCreate(&sendStream));
    CUDACHECK_TEST(cudaStreamCreate(&recvStream));

    // Send to rank 1 (writes to remoteState_.dataBuffer)
    test::testSend(
        p2pDev,
        src_d,
        nbytes,
        numBlocks,
        blockSize,
        test::GroupType::WARP,
        1,
        sendStream);
    // Recv from rank 1 (reads from localState_.dataBuffer — no conflict)
    test::testRecv(
        p2pDev,
        recvR0_d,
        nbytes,
        numBlocks,
        blockSize,
        test::GroupType::WARP,
        1,
        recvStream);

    CUDACHECK_TEST(cudaStreamSynchronize(sendStream));
    CUDACHECK_TEST(cudaStreamSynchronize(recvStream));

    CUDACHECK_TEST(cudaStreamDestroy(sendStream));
    CUDACHECK_TEST(cudaStreamDestroy(recvStream));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Verify rank 0 received correct data
    std::vector<int> hostBuffer(numInts);
    CUDACHECK_TEST(cudaMemcpy(
        hostBuffer.data(), recvR0_d, nbytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < numInts; i++) {
      EXPECT_EQ(hostBuffer[i], testValue)
          << "Rank 0 recv: Mismatch at index " << i
          << " (dstOffset=" << dstOffset << ")";
      if (hostBuffer[i] != testValue) {
        break;
      }
    }
  } else {
    // Rank 1: forward (pred == succ == same device, reads localState_ staging
    // and writes to remoteState_ staging). Zero entire padded buffer so we
    // can verify both prefix safety and payload correctness.
    CUDACHECK_TEST(cudaMemset(dstRank1.get(), 0, nbytes + dstOffset));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Apply byte-level dstOffset to the user-facing dst pointer.
    char* dstPtr = static_cast<char*>(dstRank1.get()) + dstOffset;
    test::testForward(p2pDev, p2pDev, dstPtr, nbytes, numBlocks, blockSize);
    CUDACHECK_TEST(cudaDeviceSynchronize());

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Read entire padded buffer to verify both prefix safety and payload.
    std::vector<unsigned char> hostBuf(nbytes + dstOffset);
    CUDACHECK_TEST(cudaMemcpy(
        hostBuf.data(),
        dstRank1.get(),
        nbytes + dstOffset,
        cudaMemcpyDeviceToHost));

    // Prefix bytes [0, dstOffset) must be untouched (still 0).
    for (size_t i = 0; i < dstOffset; i++) {
      EXPECT_EQ(hostBuf[i], 0u) << "Rank 1 prefix clobbered at byte " << i
                                << " (dstOffset=" << dstOffset << ")";
      if (hostBuf[i] != 0u) {
        return;
      }
    }
    // Payload bytes [dstOffset, dstOffset + nbytes) must match the byte
    // pattern of testValue (a repeated int). Build expected bytes from the
    // same int fill pattern so byte-level offsets work for any dstOffset.
    std::vector<int> expectedInts(numInts, testValue);
    const auto* expectedBytes =
        reinterpret_cast<const unsigned char*>(expectedInts.data());
    for (size_t i = 0; i < nbytes; i++) {
      EXPECT_EQ(hostBuf[dstOffset + i], expectedBytes[i])
          << "Rank 1 forward dst: Mismatch at byte " << i
          << " (nbytes=" << nbytes << ", dstOffset=" << dstOffset << ")";
      if (hostBuf[dstOffset + i] != expectedBytes[i]) {
        return;
      }
    }
  }
}

TEST_F(P2pNvlTransportTestFixture, ForwardGroupSingleState) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  CUDACHECK_TEST(cudaSetDevice(localRank));

  // Basic test: 4MB transfer, 1MB staging, 1KB chunks
  runForwardGroupTest(
      globalRank,
      numRanks,
      localRank,
      4 * 1024 * 1024, // nbytes
      1024 * 1024, // dataBufferSize
      1024, // chunkSize
      4, // pipelineDepth
      false); // useDualStateBuffer

  XLOGF(INFO, "Rank {}: ForwardGroupSingleState completed", globalRank);
}

TEST_F(P2pNvlTransportTestFixture, ForwardGroupDualState) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  CUDACHECK_TEST(cudaSetDevice(localRank));

  // Basic test: 4MB transfer, 1MB staging, 1KB chunks
  runForwardGroupTest(
      globalRank,
      numRanks,
      localRank,
      4 * 1024 * 1024, // nbytes
      1024 * 1024, // dataBufferSize
      1024, // chunkSize
      4, // pipelineDepth
      true); // useDualStateBuffer

  XLOGF(INFO, "Rank {}: ForwardGroupDualState completed", globalRank);
}

// Parameterized forward_group test with various sizes
struct ForwardGroupParams {
  size_t nbytes;
  size_t dataBufferSize;
  size_t chunkSize;
  size_t pipelineDepth;
  bool useDualStateBuffer;
  std::string name;
};

class ForwardGroupTestFixture
    : public MpiBaseTestFixture,
      public ::testing::WithParamInterface<ForwardGroupParams> {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }
};

TEST_P(ForwardGroupTestFixture, ForwardGroup) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const auto& params = GetParam();
  XLOGF(
      INFO,
      "Running forward_group test: {} (nbytes={}, dualState={})",
      params.name,
      params.nbytes,
      params.useDualStateBuffer);

  runForwardGroupTest(
      globalRank,
      numRanks,
      localRank,
      params.nbytes,
      params.dataBufferSize,
      params.chunkSize,
      params.pipelineDepth,
      params.useDualStateBuffer);

  XLOGF(
      INFO,
      "Rank {}: Forward group test '{}' completed",
      globalRank,
      params.name);
}

std::string forwardGroupParamName(
    const ::testing::TestParamInfo<ForwardGroupParams>& info) {
  return info.param.name;
}

INSTANTIATE_TEST_SUITE_P(
    ForwardGroupVariations,
    ForwardGroupTestFixture,
    ::testing::Values(
        // Single state buffer mode
        ForwardGroupParams{
            .nbytes = 512,
            .dataBufferSize = 4096,
            .chunkSize = 1024,
            .pipelineDepth = 4,
            .useDualStateBuffer = false,
            .name = "Small_512B_SingleState"},
        ForwardGroupParams{
            .nbytes = 16 * 1024,
            .dataBufferSize = 4096,
            .chunkSize = 256,
            .pipelineDepth = 4,
            .useDualStateBuffer = false,
            .name = "MultiStep_16KB_SingleState"},
        ForwardGroupParams{
            .nbytes = 4 * 1024 * 1024,
            .dataBufferSize = 1024 * 1024,
            .chunkSize = 4096,
            .pipelineDepth = 4,
            .useDualStateBuffer = false,
            .name = "Large_4MB_SingleState"},
        ForwardGroupParams{
            .nbytes = 64 * 1024 * 1024,
            .dataBufferSize = 8 * 1024 * 1024,
            .chunkSize = 512 * 1024,
            .pipelineDepth = 2,
            .useDualStateBuffer = false,
            .name = "VeryLarge_64MB_SingleState"},
        // Dual state buffer mode
        ForwardGroupParams{
            .nbytes = 512,
            .dataBufferSize = 4096,
            .chunkSize = 1024,
            .pipelineDepth = 4,
            .useDualStateBuffer = true,
            .name = "Small_512B_DualState"},
        ForwardGroupParams{
            .nbytes = 16 * 1024,
            .dataBufferSize = 4096,
            .chunkSize = 256,
            .pipelineDepth = 4,
            .useDualStateBuffer = true,
            .name = "MultiStep_16KB_DualState"},
        ForwardGroupParams{
            .nbytes = 4 * 1024 * 1024,
            .dataBufferSize = 1024 * 1024,
            .chunkSize = 4096,
            .pipelineDepth = 4,
            .useDualStateBuffer = true,
            .name = "Large_4MB_DualState"},
        ForwardGroupParams{
            .nbytes = 64 * 1024 * 1024,
            .dataBufferSize = 8 * 1024 * 1024,
            .chunkSize = 512 * 1024,
            .pipelineDepth = 2,
            .useDualStateBuffer = true,
            .name = "VeryLarge_64MB_DualState"}),
    forwardGroupParamName);

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
    size_t dataBufferSize,
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

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = dataBufferSize,
      .chunkSize = chunkSize,
      .pipelineDepth = pipelineDepth,
  };

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
            << ", slot=" << dataBufferSize << ", chunk=" << chunkSize
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
            << ", slot=" << dataBufferSize << ", chunk=" << chunkSize
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
  MultiPeerNvlTransportConfig config{
      .dataBufferSize = kDataBufferSize,
      .chunkSize = kDataBufferSize,
      .pipelineDepth = 2,
  };
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
        config.pipelineDepth * config.dataBufferSize));
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
      .dataBufferSize = tileBytes * numBlocks,
      .chunkSize = maxSignalBytes,
      .pipelineDepth = 2,
      .per_channel_slot = tileBytes,
      .max_num_channels = numBlocks,
  };
  const size_t stagingBytes = options.dataBufferSize * options.pipelineDepth;

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
// Unaligned dstbuff tests for forward_group() and tile forward()
// =============================================================================
// The staging buffers (localState_/remoteState_.dataBuffer) are always
// 256-byte aligned (cudaMalloc), so the only user-facing pointer that can
// be misaligned is `dstbuff` (the local user output buffer on rank 1, where
// the dual-dst memcpy writes its second destination).
//
// These tests offset rank 1's dstbuff by various byte amounts to exercise
// memcpy_vectorized's unaligned path in the forward() implementation.
// They reuse runForwardGroupTest / runTileForwardTest with the dstOffset
// parameter — no duplicate helpers.

// Parameterized test fixture for forward_group / tile forward unaligned dst
struct ForwardUnalignedParams {
  size_t dstOffset; // bytes added to dst pointer (0 = aligned)
  size_t nbytes;
  bool useDualStateBuffer;
  std::string name;
};

class ForwardGroupUnalignedTestFixture
    : public MpiBaseTestFixture,
      public ::testing::WithParamInterface<ForwardUnalignedParams> {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }
};

TEST_P(ForwardGroupUnalignedTestFixture, ForwardGroup) {
  if (numRanks != 2) {
    XLOGF(WARNING, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  const auto& params = GetParam();
  XLOGF(
      INFO,
      "Running forward_group unaligned test: {} (nbytes={}, dstOffset={}, dualState={})",
      params.name,
      params.nbytes,
      params.dstOffset,
      params.useDualStateBuffer);

  // Use a smallish staging buffer + chunk so multi-step + sub-step paths are
  // exercised even at modest message sizes.
  runForwardGroupTest(
      globalRank,
      numRanks,
      localRank,
      params.nbytes,
      /*dataBufferSize=*/256 * 1024,
      /*chunkSize=*/4096,
      /*pipelineDepth=*/4,
      params.useDualStateBuffer,
      /*numBlocks=*/4,
      /*blockSize=*/128,
      params.dstOffset);
}

std::string forwardUnalignedParamName(
    const ::testing::TestParamInfo<ForwardUnalignedParams>& info) {
  return info.param.name;
}

INSTANTIATE_TEST_SUITE_P(
    ForwardGroupUnalignedVariations,
    ForwardGroupUnalignedTestFixture,
    ::testing::Values(
        // Baseline aligned (offset 0) for sanity
        ForwardUnalignedParams{
            .dstOffset = 0,
            .nbytes = 64 * 1024,
            .useDualStateBuffer = false,
            .name = "Aligned_64KB_SingleState"},
        // Various small misalignments (single state)
        ForwardUnalignedParams{
            .dstOffset = 1,
            .nbytes = 64 * 1024,
            .useDualStateBuffer = false,
            .name = "Off1_64KB_SingleState"},
        ForwardUnalignedParams{
            .dstOffset = 7,
            .nbytes = 64 * 1024,
            .useDualStateBuffer = false,
            .name = "Off7_64KB_SingleState"},
        ForwardUnalignedParams{
            .dstOffset = 8,
            .nbytes = 64 * 1024,
            .useDualStateBuffer = false,
            .name = "Off8_64KB_SingleState"},
        ForwardUnalignedParams{
            .dstOffset = 15,
            .nbytes = 64 * 1024,
            .useDualStateBuffer = false,
            .name = "Off15_64KB_SingleState"},
        // Larger transfer with misalignment (multi-step pipeline)
        ForwardUnalignedParams{
            .dstOffset = 3,
            .nbytes = 4 * 1024 * 1024,
            .useDualStateBuffer = false,
            .name = "Off3_4MB_SingleState"},
        ForwardUnalignedParams{
            .dstOffset = 13,
            .nbytes = 4 * 1024 * 1024,
            .useDualStateBuffer = false,
            .name = "Off13_4MB_SingleState"},
        // Unaligned size + unaligned offset
        ForwardUnalignedParams{
            .dstOffset = 5,
            .nbytes = 1000003,
            .useDualStateBuffer = false,
            .name = "Off5_OddSize_SingleState"},
        // Dual state mode with misalignment
        ForwardUnalignedParams{
            .dstOffset = 1,
            .nbytes = 64 * 1024,
            .useDualStateBuffer = true,
            .name = "Off1_64KB_DualState"},
        ForwardUnalignedParams{
            .dstOffset = 7,
            .nbytes = 64 * 1024,
            .useDualStateBuffer = true,
            .name = "Off7_64KB_DualState"},
        ForwardUnalignedParams{
            .dstOffset = 13,
            .nbytes = 4 * 1024 * 1024,
            .useDualStateBuffer = true,
            .name = "Off13_4MB_DualState"}),
    forwardUnalignedParamName);

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
  // Tile API doesn't use useDualStateBuffer; use a multi-step config so the
  // unaligned dst is exercised across many steps and chunks.
  runTileForwardTest(
      globalRank,
      numRanks,
      bs,
      params.nbytes,
      /*dataBufferSize=*/8 * 1024 * 1024,
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
            .useDualStateBuffer = false,
            .name = "Aligned_4MB"},
        // Various misalignments
        ForwardUnalignedParams{
            .dstOffset = 1,
            .nbytes = 4 * 1024 * 1024,
            .useDualStateBuffer = false,
            .name = "Off1_4MB"},
        ForwardUnalignedParams{
            .dstOffset = 7,
            .nbytes = 4 * 1024 * 1024,
            .useDualStateBuffer = false,
            .name = "Off7_4MB"},
        ForwardUnalignedParams{
            .dstOffset = 8,
            .nbytes = 4 * 1024 * 1024,
            .useDualStateBuffer = false,
            .name = "Off8_4MB"},
        ForwardUnalignedParams{
            .dstOffset = 15,
            .nbytes = 4 * 1024 * 1024,
            .useDualStateBuffer = false,
            .name = "Off15_4MB"},
        // Larger transfer with misalignment
        ForwardUnalignedParams{
            .dstOffset = 3,
            .nbytes = 16 * 1024 * 1024,
            .useDualStateBuffer = false,
            .name = "Off3_16MB"},
        // Unaligned size + unaligned offset
        ForwardUnalignedParams{
            .dstOffset = 5,
            .nbytes = 1000003,
            .useDualStateBuffer = false,
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

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = 4096,
      .chunkSize = 256,
      .pipelineDepth = 2,
      .llBufferSize = ll_buffer_size(4096),
  };

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

  MultiPeerNvlTransportConfig config{
      .dataBufferSize = 4096,
      .chunkSize = 256,
      .pipelineDepth = 2,
  };

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
