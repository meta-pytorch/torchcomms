// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>
#include <folly/init/Init.h>
#include <folly/io/async/ScopedEventBaseThread.h>
#include <folly/logging/Init.h>
#include <gtest/gtest.h>
#include <vector>

#include "comms/ctran/ibverbx/Ibverbx.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/torchcomms/transport/StagedRdmaTransport.h"
#include "comms/utils/cvars/nccl_cvars.h"

// NOLINTNEXTLINE(google-build-using-namespace)
using namespace torch::comms;
using namespace meta::comms;

FOLLY_INIT_LOGGING_CONFIG(
    ".=WARNING"
    ";default:async=true,sync_level=WARNING");

// --- Fill mode for transfer helpers ---

namespace {

enum class FillMode { CONSTANT, POSITIONAL };

// Fill buffer with position-dependent pattern: buf[i] = (uint8_t)(i % 251).
// Prime modulus 251 prevents alignment with chunk boundaries.
void fillPositionalPattern(std::vector<uint8_t>& buf) {
  for (size_t i = 0; i < buf.size(); ++i) {
    buf[i] = static_cast<uint8_t>(i % 251);
  }
}

// Verify positional pattern. Returns (valid, firstMismatchIdx).
std::pair<bool, size_t> verifyPositionalPattern(
    const std::vector<uint8_t>& buf) {
  for (size_t i = 0; i < buf.size(); ++i) {
    if (buf[i] != static_cast<uint8_t>(i % 251)) {
      return {false, i};
    }
  }
  return {true, 0};
}

} // namespace

// --- Construction tests (no MPI needed, run independently on each rank) ---

TEST(StagedRdmaTransportTest, ConstructAndDestroy) {
  StagedTransferConfig config;
  config.stagingBufSize = 2 * 1024 * 1024;
  StagedRdmaServerTransport server(0, nullptr, config);
  StagedRdmaClientTransport client(0, nullptr, config);
  EXPECT_EQ(server.stagingBufSize(), config.stagingBufSize);
  EXPECT_EQ(client.stagingBufSize(), config.stagingBufSize);
}

TEST(StagedRdmaTransportTest, ConstructWithConfig) {
  StagedTransferConfig config;
  config.stagingBufSize = 1024 * 1024;
  config.chunkTimeout = std::chrono::milliseconds{5000};

  StagedRdmaServerTransport server(0, nullptr, config);
  StagedRdmaClientTransport client(0, nullptr, config);
  EXPECT_EQ(server.stagingBufSize(), config.stagingBufSize);
  EXPECT_EQ(client.stagingBufSize(), config.stagingBufSize);
}

TEST(StagedRdmaTransportTest, ConstructWithEventBase) {
  StagedTransferConfig config;
  config.stagingBufSize = 8 * 1024 * 1024;
  folly::ScopedEventBaseThread evbThread("test-evb");
  StagedRdmaServerTransport server(0, evbThread.getEventBase(), config);
  StagedRdmaClientTransport client(0, evbThread.getEventBase(), config);
  EXPECT_EQ(server.stagingBufSize(), config.stagingBufSize);
  EXPECT_EQ(client.stagingBufSize(), config.stagingBufSize);
}

// --- Distributed test fixture (MPI-based, 2 ranks) ---

class StagedRdmaTransportDistributedTest : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    if (numRanks < 2) {
      GTEST_SKIP() << "Need at least 2 ranks";
    }
    ncclCvarInit();
    ASSERT_TRUE(ibverbx::ibvInit());
  }

  // Exchange connection info between rank 0 (server) and rank 1 (client).
  // Returns the peer's connection info string.
  std::string exchangeConnInfo(const std::string& localConnInfo) {
    // Exchange lengths first
    int localLen = static_cast<int>(localConnInfo.size());
    int peerLen = 0;
    int peerRank = 1 - globalRank;
    MPI_CHECK(MPI_Sendrecv(
        &localLen,
        1,
        MPI_INT,
        peerRank,
        0,
        &peerLen,
        1,
        MPI_INT,
        peerRank,
        0,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE));

    // Exchange actual strings
    std::string peerConnInfo(peerLen, '\0');
    MPI_CHECK(MPI_Sendrecv(
        localConnInfo.data(),
        localLen,
        MPI_CHAR,
        peerRank,
        1,
        peerConnInfo.data(),
        peerLen,
        MPI_CHAR,
        peerRank,
        1,
        MPI_COMM_WORLD,
        MPI_STATUS_IGNORE));

    return peerConnInfo;
  }

  // Broadcast a size_t from rank 0 to all ranks.
  size_t broadcastSize(size_t value) {
    MPI_CHECK(MPI_Bcast(&value, sizeof(size_t), MPI_BYTE, 0, MPI_COMM_WORLD));
    return value;
  }

  // Run a contiguous transfer: server fills + sends, client recvs + verifies.
  // Both ranks must call this. Internally branches on globalRank.
  void runTransfer(
      size_t totalBytes,
      uint8_t fillByte,
      StagedTransferConfig config = {},
      FillMode fillMode = FillMode::CONSTANT) {
    int cudaDev = localRank;
    ASSERT_EQ(cudaSetDevice(cudaDev), cudaSuccess);

    folly::ScopedEventBaseThread evbThread("RDMA-Worker");

    if (globalRank == 0) {
      // --- Server ---
      StagedRdmaServerTransport transport(
          cudaDev, evbThread.getEventBase(), config);
      auto connInfo = transport.setupLocalTransport();
      auto peerConnInfo = exchangeConnInfo(connInfo);
      transport.connectRemoteTransport(peerConnInfo);

      broadcastSize(totalBytes);

      // Allocate source GPU buffer and fill with pattern
      void* srcGpu = nullptr;
      ASSERT_EQ(cudaMalloc(&srcGpu, totalBytes), cudaSuccess);
      std::vector<uint8_t> hostBuf(totalBytes, fillByte);
      if (fillMode == FillMode::POSITIONAL) {
        fillPositionalPattern(hostBuf);
      }
      ASSERT_EQ(
          cudaMemcpy(
              srcGpu, hostBuf.data(), totalBytes, cudaMemcpyHostToDevice),
          cudaSuccess);

      ScatterGatherDescriptor sgDesc;
      sgDesc.entries.push_back({srcGpu, totalBytes});
      auto result = transport.send(sgDesc).get();
      EXPECT_EQ(result, commSuccess);

      EXPECT_EQ(cudaFree(srcGpu), cudaSuccess);
    } else {
      // --- Client ---
      StagedRdmaClientTransport transport(
          cudaDev, evbThread.getEventBase(), config);
      auto connInfo = transport.setupLocalTransport();
      auto peerConnInfo = exchangeConnInfo(connInfo);
      transport.connectRemoteTransport(peerConnInfo);

      auto recvBytes = broadcastSize(0);

      // Allocate destination GPU buffer (zeroed)
      void* dstGpu = nullptr;
      ASSERT_EQ(cudaMalloc(&dstGpu, recvBytes), cudaSuccess);
      ASSERT_EQ(cudaMemset(dstGpu, 0, recvBytes), cudaSuccess);

      ScatterGatherDescriptor sgDesc;
      sgDesc.entries.push_back({dstGpu, recvBytes});
      auto result = transport.recv(sgDesc).get();
      EXPECT_EQ(result, commSuccess);

      if (result == commSuccess) {
        std::vector<uint8_t> hostBuf(recvBytes);
        ASSERT_EQ(
            cudaMemcpy(
                hostBuf.data(), dstGpu, recvBytes, cudaMemcpyDeviceToHost),
            cudaSuccess);

        if (fillMode == FillMode::POSITIONAL) {
          auto [valid, idx] = verifyPositionalPattern(hostBuf);
          EXPECT_TRUE(valid)
              << "Positional mismatch at byte " << idx << ": expected 0x"
              << std::hex << static_cast<int>(idx % 251) << " got 0x"
              << static_cast<int>(hostBuf[idx]);
        } else {
          for (size_t i = 0; i < recvBytes; ++i) {
            if (hostBuf[i] != fillByte) {
              ADD_FAILURE() << "Data mismatch at byte " << i << ": expected 0x"
                            << std::hex << static_cast<int>(fillByte)
                            << " got 0x" << static_cast<int>(hostBuf[i]);
              break;
            }
          }
        }
      }

      EXPECT_EQ(cudaFree(dstGpu), cudaSuccess);
    }

    MPI_Barrier(MPI_COMM_WORLD);
  }
};

// --- Connection setup test ---

TEST_F(StagedRdmaTransportDistributedTest, SetupAndConnect) {
  int cudaDev = localRank;
  ASSERT_EQ(cudaSetDevice(cudaDev), cudaSuccess);

  if (globalRank == 0) {
    StagedRdmaServerTransport transport(cudaDev);
    auto connInfo = transport.setupLocalTransport();
    EXPECT_FALSE(connInfo.empty());
    auto peerConnInfo = exchangeConnInfo(connInfo);
    transport.connectRemoteTransport(peerConnInfo);
  } else {
    StagedRdmaClientTransport transport(cudaDev);
    auto connInfo = transport.setupLocalTransport();
    EXPECT_FALSE(connInfo.empty());
    auto peerConnInfo = exchangeConnInfo(connInfo);
    transport.connectRemoteTransport(peerConnInfo);
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

// --- Data transfer tests ---

TEST_F(StagedRdmaTransportDistributedTest, SingleChunkTransfer) {
  // Transfer smaller than staging buffer → single chunk
  runTransfer(8192, 0xAB);
}

TEST_F(StagedRdmaTransportDistributedTest, MultiChunkTransfer) {
  // 256KB total with 64KB staging = 4 chunks
  StagedTransferConfig config;
  config.stagingBufSize = 64 * 1024;
  runTransfer(256 * 1024, 0xCD, config);
}

TEST_F(StagedRdmaTransportDistributedTest, LastChunkSmaller) {
  // 100KB total with 64KB staging = 2 chunks (64KB + 36KB)
  StagedTransferConfig config;
  config.stagingBufSize = 64 * 1024;
  runTransfer(100 * 1024, 0xEF, config);
}

TEST_F(StagedRdmaTransportDistributedTest, ExactlyOneStagingBuffer) {
  // Transfer exactly equal to staging buffer size → single chunk, no remainder
  StagedTransferConfig config;
  config.stagingBufSize = 64 * 1024;
  runTransfer(64 * 1024, 0x42, config);
}

TEST_F(StagedRdmaTransportDistributedTest, LargeTransfer) {
  // 4MB transfer with default 64MB staging = single chunk but large data
  runTransfer(4 * 1024 * 1024, 0x77);
}

TEST_F(StagedRdmaTransportDistributedTest, UniquePatternIntegrity) {
  // Positional pattern catches offset bugs that constant fill cannot.
  // 200KB with 64KB staging = 4 chunks (non-aligned last chunk).
  StagedTransferConfig config;
  config.stagingBufSize = 64 * 1024;
  runTransfer(200 * 1024, 0, config, FillMode::POSITIONAL);
}

TEST_F(StagedRdmaTransportDistributedTest, ManyChunksTransfer) {
  // 2MB total with 32KB staging = 64 chunks.
  StagedTransferConfig config;
  config.stagingBufSize = 32 * 1024;
  runTransfer(2 * 1024 * 1024, 0, config, FillMode::POSITIONAL);
}

// --- Sequential transfers (transport reuse) ---

TEST_F(StagedRdmaTransportDistributedTest, SequentialTransfers) {
  // Two transfers on the same transport pair with different fill bytes.
  // Tests the always-auto-replenish recv WR fix.
  // Both ranks construct identical transfer specs; no broadcastSize needed.
  StagedTransferConfig config;
  config.stagingBufSize = 32 * 1024;

  struct TransferSpec {
    size_t totalBytes;
    uint8_t fillByte;
  };
  std::vector<TransferSpec> transfers = {
      {64 * 1024, 0xAA},
      {64 * 1024, 0xBB},
  };

  int cudaDev = localRank;
  ASSERT_EQ(cudaSetDevice(cudaDev), cudaSuccess);

  folly::ScopedEventBaseThread evbThread("RDMA-Worker");

  if (globalRank == 0) {
    // --- Server ---
    StagedRdmaServerTransport transport(
        cudaDev, evbThread.getEventBase(), config);
    auto connInfo = transport.setupLocalTransport();
    auto peerConnInfo = exchangeConnInfo(connInfo);
    transport.connectRemoteTransport(peerConnInfo);

    for (size_t t = 0; t < transfers.size(); ++t) {
      auto& spec = transfers[t];

      void* srcGpu = nullptr;
      ASSERT_EQ(cudaMalloc(&srcGpu, spec.totalBytes), cudaSuccess);
      std::vector<uint8_t> hostBuf(spec.totalBytes, spec.fillByte);
      ASSERT_EQ(
          cudaMemcpy(
              srcGpu, hostBuf.data(), spec.totalBytes, cudaMemcpyHostToDevice),
          cudaSuccess);

      ScatterGatherDescriptor sgDesc;
      sgDesc.entries.push_back({srcGpu, spec.totalBytes});
      auto result = transport.send(sgDesc).get();
      EXPECT_EQ(result, commSuccess) << "Transfer " << t << " failed";

      EXPECT_EQ(cudaFree(srcGpu), cudaSuccess);
    }
  } else {
    // --- Client ---
    StagedRdmaClientTransport transport(
        cudaDev, evbThread.getEventBase(), config);
    auto connInfo = transport.setupLocalTransport();
    auto peerConnInfo = exchangeConnInfo(connInfo);
    transport.connectRemoteTransport(peerConnInfo);

    for (size_t t = 0; t < transfers.size(); ++t) {
      auto& spec = transfers[t];

      void* dstGpu = nullptr;
      ASSERT_EQ(cudaMalloc(&dstGpu, spec.totalBytes), cudaSuccess);
      ASSERT_EQ(cudaMemset(dstGpu, 0, spec.totalBytes), cudaSuccess);

      ScatterGatherDescriptor sgDesc;
      sgDesc.entries.push_back({dstGpu, spec.totalBytes});
      auto result = transport.recv(sgDesc).get();
      EXPECT_EQ(result, commSuccess) << "Transfer " << t << " failed";

      if (result == commSuccess) {
        std::vector<uint8_t> hostBuf(spec.totalBytes);
        ASSERT_EQ(
            cudaMemcpy(
                hostBuf.data(),
                dstGpu,
                spec.totalBytes,
                cudaMemcpyDeviceToHost),
            cudaSuccess);
        for (size_t i = 0; i < spec.totalBytes; ++i) {
          if (hostBuf[i] != spec.fillByte) {
            ADD_FAILURE() << "Transfer " << t << ": mismatch at byte " << i
                          << ": expected 0x" << std::hex
                          << static_cast<int>(spec.fillByte) << " got 0x"
                          << static_cast<int>(hostBuf[i]);
            break;
          }
        }
      }

      EXPECT_EQ(cudaFree(dstGpu), cudaSuccess);
    }
  }
}

// --- Parameterized buffer size tests ---

class StagedRdmaTransportBufferSizeTest
    : public StagedRdmaTransportDistributedTest,
      public ::testing::WithParamInterface<size_t> {};

TEST_P(StagedRdmaTransportBufferSizeTest, TransferWithSize) {
  const size_t totalBytes = GetParam();
  StagedTransferConfig config;
  config.stagingBufSize = 64 * 1024;
  runTransfer(totalBytes, 0, config, FillMode::POSITIONAL);
}

INSTANTIATE_TEST_SUITE_P(
    BufferSizes,
    StagedRdmaTransportBufferSizeTest,
    ::testing::Values(
        1, // 1 byte — minimal transfer
        1023, // sub-page non-power-of-2
        1024, // 1KB — sub-page
        4096, // page boundary
        4097, // page boundary + 1
        65535, // staging - 1
        65536, // exactly staging
        65537, // staging + 1 (forces 2 chunks)
        131072, // exactly 2 chunks
        1048576), // 16 chunks (1MB)
    [](const ::testing::TestParamInfo<size_t>& info) {
      return "Size_" + std::to_string(info.param);
    });

// --- Scatter/Gather transfer tests ---

// Server sends contiguous data, client scatters to non-contiguous GPU regions.
TEST_F(StagedRdmaTransportDistributedTest, ScatterRecvTransfer) {
  const size_t entrySize = 4096;
  const size_t numEntries = 4;
  const size_t totalBytes = entrySize * numEntries;

  int cudaDev = localRank;
  ASSERT_EQ(cudaSetDevice(cudaDev), cudaSuccess);

  folly::ScopedEventBaseThread evbThread("RDMA-Worker");

  if (globalRank == 0) {
    StagedRdmaServerTransport transport(cudaDev, evbThread.getEventBase());
    auto connInfo = transport.setupLocalTransport();
    auto peerConnInfo = exchangeConnInfo(connInfo);
    transport.connectRemoteTransport(peerConnInfo);

    broadcastSize(totalBytes);

    // Contiguous source with positional pattern
    void* srcGpu = nullptr;
    ASSERT_EQ(cudaMalloc(&srcGpu, totalBytes), cudaSuccess);
    std::vector<uint8_t> hostBuf(totalBytes);
    fillPositionalPattern(hostBuf);
    ASSERT_EQ(
        cudaMemcpy(srcGpu, hostBuf.data(), totalBytes, cudaMemcpyHostToDevice),
        cudaSuccess);

    ScatterGatherDescriptor putDesc;
    putDesc.entries.push_back({srcGpu, totalBytes});
    auto result = transport.send(putDesc).get();
    EXPECT_EQ(result, commSuccess);

    EXPECT_EQ(cudaFree(srcGpu), cudaSuccess);
  } else {
    StagedRdmaClientTransport transport(cudaDev, evbThread.getEventBase());
    auto connInfo = transport.setupLocalTransport();
    auto peerConnInfo = exchangeConnInfo(connInfo);
    transport.connectRemoteTransport(peerConnInfo);

    broadcastSize(0);

    // Allocate non-contiguous destination: 4 separate GPU buffers
    std::vector<void*> dstPtrs(numEntries);
    for (size_t i = 0; i < numEntries; ++i) {
      ASSERT_EQ(cudaMalloc(&dstPtrs[i], entrySize), cudaSuccess);
      ASSERT_EQ(cudaMemset(dstPtrs[i], 0, entrySize), cudaSuccess);
    }

    ScatterGatherDescriptor sgDesc;
    for (size_t i = 0; i < numEntries; ++i) {
      sgDesc.entries.push_back({dstPtrs[i], entrySize});
    }

    auto result = transport.recv(sgDesc).get();
    EXPECT_EQ(result, commSuccess);

    if (result == commSuccess) {
      for (size_t i = 0; i < numEntries; ++i) {
        std::vector<uint8_t> hostBuf(entrySize);
        ASSERT_EQ(
            cudaMemcpy(
                hostBuf.data(), dstPtrs[i], entrySize, cudaMemcpyDeviceToHost),
            cudaSuccess);
        size_t globalOffset = i * entrySize;
        for (size_t j = 0; j < entrySize; ++j) {
          uint8_t expected = static_cast<uint8_t>((globalOffset + j) % 251);
          EXPECT_EQ(hostBuf[j], expected)
              << "Mismatch in entry " << i << " at byte " << j
              << " (global offset " << globalOffset + j << ")";
          if (hostBuf[j] != expected) {
            break;
          }
        }
      }
    }

    for (auto* ptr : dstPtrs) {
      EXPECT_EQ(cudaFree(ptr), cudaSuccess);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

// Server gathers from non-contiguous GPU regions, client receives contiguously.
TEST_F(StagedRdmaTransportDistributedTest, GatherPutTransfer) {
  const size_t entrySize = 4096;
  const size_t numEntries = 4;
  const size_t totalBytes = entrySize * numEntries;

  int cudaDev = localRank;
  ASSERT_EQ(cudaSetDevice(cudaDev), cudaSuccess);

  folly::ScopedEventBaseThread evbThread("RDMA-Worker");

  if (globalRank == 0) {
    StagedRdmaServerTransport transport(cudaDev, evbThread.getEventBase());
    auto connInfo = transport.setupLocalTransport();
    auto peerConnInfo = exchangeConnInfo(connInfo);
    transport.connectRemoteTransport(peerConnInfo);
    broadcastSize(totalBytes);

    // Non-contiguous source with positional pattern
    std::vector<void*> srcPtrs(numEntries);
    for (size_t i = 0; i < numEntries; ++i) {
      ASSERT_EQ(cudaMalloc(&srcPtrs[i], entrySize), cudaSuccess);
      std::vector<uint8_t> hostBuf(entrySize);
      size_t globalOffset = i * entrySize;
      for (size_t j = 0; j < entrySize; ++j) {
        hostBuf[j] = static_cast<uint8_t>((globalOffset + j) % 251);
      }
      ASSERT_EQ(
          cudaMemcpy(
              srcPtrs[i], hostBuf.data(), entrySize, cudaMemcpyHostToDevice),
          cudaSuccess);
    }

    ScatterGatherDescriptor sgDesc;
    for (size_t i = 0; i < numEntries; ++i) {
      sgDesc.entries.push_back({srcPtrs[i], entrySize});
    }

    auto result = transport.send(sgDesc).get();
    EXPECT_EQ(result, commSuccess);

    for (auto* ptr : srcPtrs) {
      EXPECT_EQ(cudaFree(ptr), cudaSuccess);
    }
  } else {
    StagedRdmaClientTransport transport(cudaDev, evbThread.getEventBase());
    auto connInfo = transport.setupLocalTransport();
    auto peerConnInfo = exchangeConnInfo(connInfo);
    transport.connectRemoteTransport(peerConnInfo);

    auto recvBytes = broadcastSize(0);

    // Contiguous destination
    void* dstGpu = nullptr;
    ASSERT_EQ(cudaMalloc(&dstGpu, recvBytes), cudaSuccess);
    ASSERT_EQ(cudaMemset(dstGpu, 0, recvBytes), cudaSuccess);

    ScatterGatherDescriptor sgDesc;
    sgDesc.entries.push_back({dstGpu, recvBytes});
    auto result = transport.recv(sgDesc).get();
    EXPECT_EQ(result, commSuccess);

    if (result == commSuccess) {
      std::vector<uint8_t> hostBuf(recvBytes);
      ASSERT_EQ(
          cudaMemcpy(hostBuf.data(), dstGpu, recvBytes, cudaMemcpyDeviceToHost),
          cudaSuccess);
      auto [valid, idx] = verifyPositionalPattern(hostBuf);
      EXPECT_TRUE(valid) << "Positional mismatch at byte " << idx
                         << ": expected 0x" << std::hex
                         << static_cast<int>(idx % 251) << " got 0x"
                         << static_cast<int>(hostBuf[idx]);
    }

    EXPECT_EQ(cudaFree(dstGpu), cudaSuccess);
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

// Both sides use scatter/gather: server gathers, client scatters.
TEST_F(StagedRdmaTransportDistributedTest, GatherPutScatterRecv) {
  const size_t entrySize = 4096;
  const size_t numEntries = 4;
  const size_t totalBytes = entrySize * numEntries;

  int cudaDev = localRank;
  ASSERT_EQ(cudaSetDevice(cudaDev), cudaSuccess);

  folly::ScopedEventBaseThread evbThread("RDMA-Worker");

  if (globalRank == 0) {
    StagedRdmaServerTransport transport(cudaDev, evbThread.getEventBase());
    auto connInfo = transport.setupLocalTransport();
    auto peerConnInfo = exchangeConnInfo(connInfo);
    transport.connectRemoteTransport(peerConnInfo);
    broadcastSize(totalBytes);

    // Non-contiguous source
    std::vector<void*> srcPtrs(numEntries);
    for (size_t i = 0; i < numEntries; ++i) {
      ASSERT_EQ(cudaMalloc(&srcPtrs[i], entrySize), cudaSuccess);
      std::vector<uint8_t> hostBuf(entrySize);
      size_t globalOffset = i * entrySize;
      for (size_t j = 0; j < entrySize; ++j) {
        hostBuf[j] = static_cast<uint8_t>((globalOffset + j) % 251);
      }
      ASSERT_EQ(
          cudaMemcpy(
              srcPtrs[i], hostBuf.data(), entrySize, cudaMemcpyHostToDevice),
          cudaSuccess);
    }

    ScatterGatherDescriptor sgDesc;
    for (size_t i = 0; i < numEntries; ++i) {
      sgDesc.entries.push_back({srcPtrs[i], entrySize});
    }

    auto result = transport.send(sgDesc).get();
    EXPECT_EQ(result, commSuccess);

    for (auto* ptr : srcPtrs) {
      EXPECT_EQ(cudaFree(ptr), cudaSuccess);
    }
  } else {
    StagedRdmaClientTransport transport(cudaDev, evbThread.getEventBase());
    auto connInfo = transport.setupLocalTransport();
    auto peerConnInfo = exchangeConnInfo(connInfo);
    transport.connectRemoteTransport(peerConnInfo);

    broadcastSize(0);

    // Non-contiguous destination
    std::vector<void*> dstPtrs(numEntries);
    for (size_t i = 0; i < numEntries; ++i) {
      ASSERT_EQ(cudaMalloc(&dstPtrs[i], entrySize), cudaSuccess);
      ASSERT_EQ(cudaMemset(dstPtrs[i], 0, entrySize), cudaSuccess);
    }

    ScatterGatherDescriptor sgDesc;
    for (size_t i = 0; i < numEntries; ++i) {
      sgDesc.entries.push_back({dstPtrs[i], entrySize});
    }

    auto result = transport.recv(sgDesc).get();
    EXPECT_EQ(result, commSuccess);

    if (result == commSuccess) {
      for (size_t i = 0; i < numEntries; ++i) {
        std::vector<uint8_t> hostBuf(entrySize);
        ASSERT_EQ(
            cudaMemcpy(
                hostBuf.data(), dstPtrs[i], entrySize, cudaMemcpyDeviceToHost),
            cudaSuccess);
        size_t globalOffset = i * entrySize;
        for (size_t j = 0; j < entrySize; ++j) {
          uint8_t expected = static_cast<uint8_t>((globalOffset + j) % 251);
          EXPECT_EQ(hostBuf[j], expected)
              << "Mismatch in entry " << i << " at byte " << j;
          if (hostBuf[j] != expected) {
            break;
          }
        }
      }
    }

    for (auto* ptr : dstPtrs) {
      EXPECT_EQ(cudaFree(ptr), cudaSuccess);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

// Scatter recv with multi-chunk transfer. 256KB total, 64KB staging = 4 chunks,
// scattered across 8 non-contiguous 32KB destination buffers.
TEST_F(StagedRdmaTransportDistributedTest, ScatterRecvMultiChunk) {
  const size_t entrySize = 32 * 1024;
  const size_t numEntries = 8;
  const size_t totalBytes = entrySize * numEntries;
  StagedTransferConfig config;
  config.stagingBufSize = 64 * 1024;

  int cudaDev = localRank;
  ASSERT_EQ(cudaSetDevice(cudaDev), cudaSuccess);

  folly::ScopedEventBaseThread evbThread("RDMA-Worker");

  if (globalRank == 0) {
    StagedRdmaServerTransport transport(
        cudaDev, evbThread.getEventBase(), config);
    auto connInfo = transport.setupLocalTransport();
    auto peerConnInfo = exchangeConnInfo(connInfo);
    transport.connectRemoteTransport(peerConnInfo);

    broadcastSize(totalBytes);

    // Contiguous source with positional pattern
    void* srcGpu = nullptr;
    ASSERT_EQ(cudaMalloc(&srcGpu, totalBytes), cudaSuccess);
    std::vector<uint8_t> hostBuf(totalBytes);
    fillPositionalPattern(hostBuf);
    ASSERT_EQ(
        cudaMemcpy(srcGpu, hostBuf.data(), totalBytes, cudaMemcpyHostToDevice),
        cudaSuccess);

    ScatterGatherDescriptor putDesc;
    putDesc.entries.push_back({srcGpu, totalBytes});
    auto result = transport.send(putDesc).get();
    EXPECT_EQ(result, commSuccess);

    EXPECT_EQ(cudaFree(srcGpu), cudaSuccess);
  } else {
    StagedRdmaClientTransport transport(
        cudaDev, evbThread.getEventBase(), config);
    auto connInfo = transport.setupLocalTransport();
    auto peerConnInfo = exchangeConnInfo(connInfo);
    transport.connectRemoteTransport(peerConnInfo);

    broadcastSize(0);

    // 8 non-contiguous 32KB destination buffers
    std::vector<void*> dstPtrs(numEntries);
    for (size_t i = 0; i < numEntries; ++i) {
      ASSERT_EQ(cudaMalloc(&dstPtrs[i], entrySize), cudaSuccess);
      ASSERT_EQ(cudaMemset(dstPtrs[i], 0, entrySize), cudaSuccess);
    }

    ScatterGatherDescriptor sgDesc;
    for (size_t i = 0; i < numEntries; ++i) {
      sgDesc.entries.push_back({dstPtrs[i], entrySize});
    }

    auto result = transport.recv(sgDesc).get();
    EXPECT_EQ(result, commSuccess);

    if (result == commSuccess) {
      for (size_t i = 0; i < numEntries; ++i) {
        std::vector<uint8_t> hostBuf(entrySize);
        ASSERT_EQ(
            cudaMemcpy(
                hostBuf.data(), dstPtrs[i], entrySize, cudaMemcpyDeviceToHost),
            cudaSuccess);
        size_t globalOffset = i * entrySize;
        for (size_t j = 0; j < entrySize; ++j) {
          uint8_t expected = static_cast<uint8_t>((globalOffset + j) % 251);
          EXPECT_EQ(hostBuf[j], expected)
              << "Mismatch in entry " << i << " at byte " << j
              << " (global offset " << globalOffset + j << ")";
          if (hostBuf[j] != expected) {
            break;
          }
        }
      }
    }

    for (auto* ptr : dstPtrs) {
      EXPECT_EQ(cudaFree(ptr), cudaSuccess);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

// Scatter recv with uneven entry sizes. 3 entries of 40KB, 24KB, 64KB = 128KB
// total, with 48KB staging = 3 chunks.
TEST_F(StagedRdmaTransportDistributedTest, ScatterRecvUnevenEntries) {
  const std::vector<size_t> entrySizes = {
      40 * 1024,
      24 * 1024,
      64 * 1024,
  };
  size_t totalBytes = 0;
  for (auto s : entrySizes) {
    totalBytes += s;
  }
  StagedTransferConfig config;
  config.stagingBufSize = 48 * 1024;

  int cudaDev = localRank;
  ASSERT_EQ(cudaSetDevice(cudaDev), cudaSuccess);

  folly::ScopedEventBaseThread evbThread("RDMA-Worker");

  if (globalRank == 0) {
    StagedRdmaServerTransport transport(
        cudaDev, evbThread.getEventBase(), config);
    auto connInfo = transport.setupLocalTransport();
    auto peerConnInfo = exchangeConnInfo(connInfo);
    transport.connectRemoteTransport(peerConnInfo);

    broadcastSize(totalBytes);

    void* srcGpu = nullptr;
    ASSERT_EQ(cudaMalloc(&srcGpu, totalBytes), cudaSuccess);
    std::vector<uint8_t> hostBuf(totalBytes);
    fillPositionalPattern(hostBuf);
    ASSERT_EQ(
        cudaMemcpy(srcGpu, hostBuf.data(), totalBytes, cudaMemcpyHostToDevice),
        cudaSuccess);

    ScatterGatherDescriptor putDesc;
    putDesc.entries.push_back({srcGpu, totalBytes});
    auto result = transport.send(putDesc).get();
    EXPECT_EQ(result, commSuccess);

    EXPECT_EQ(cudaFree(srcGpu), cudaSuccess);
  } else {
    StagedRdmaClientTransport transport(
        cudaDev, evbThread.getEventBase(), config);
    auto connInfo = transport.setupLocalTransport();
    auto peerConnInfo = exchangeConnInfo(connInfo);
    transport.connectRemoteTransport(peerConnInfo);

    broadcastSize(0);

    // Allocate separate GPU buffers with varying sizes
    std::vector<void*> dstPtrs(entrySizes.size());
    for (size_t i = 0; i < entrySizes.size(); ++i) {
      ASSERT_EQ(cudaMalloc(&dstPtrs[i], entrySizes[i]), cudaSuccess);
      ASSERT_EQ(cudaMemset(dstPtrs[i], 0, entrySizes[i]), cudaSuccess);
    }

    ScatterGatherDescriptor sgDesc;
    for (size_t i = 0; i < entrySizes.size(); ++i) {
      sgDesc.entries.push_back({dstPtrs[i], entrySizes[i]});
    }

    auto result = transport.recv(sgDesc).get();
    EXPECT_EQ(result, commSuccess);

    if (result == commSuccess) {
      size_t globalOffset = 0;
      for (size_t i = 0; i < entrySizes.size(); ++i) {
        std::vector<uint8_t> hostBuf(entrySizes[i]);
        ASSERT_EQ(
            cudaMemcpy(
                hostBuf.data(),
                dstPtrs[i],
                entrySizes[i],
                cudaMemcpyDeviceToHost),
            cudaSuccess);
        for (size_t j = 0; j < entrySizes[i]; ++j) {
          uint8_t expected = static_cast<uint8_t>((globalOffset + j) % 251);
          EXPECT_EQ(hostBuf[j], expected)
              << "Mismatch in entry " << i << " at byte " << j;
          if (hostBuf[j] != expected) {
            break;
          }
        }
        globalOffset += entrySizes[i];
      }
    }

    for (auto* ptr : dstPtrs) {
      EXPECT_EQ(cudaFree(ptr), cudaSuccess);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

// --- main ---

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
