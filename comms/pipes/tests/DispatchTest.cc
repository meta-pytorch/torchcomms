// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include <numeric>
#include <vector>

#include "comms/pipes/MultiPeerNvlTransport.h"
#include "comms/pipes/tests/DispatchTestKernels.cuh"
#include "comms/pipes/tests/Utils.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

using namespace meta::comms; // NOLINT(google-build-using-namespace)

namespace comms::pipes::tests {

// Constants for chunk sizes
constexpr size_t kSmallChunkSize = 16 * 1024; // 16KB
constexpr size_t kLargeChunkSize = 1024 * 1024; // 1MB

// Test parameters for parameterized tests
struct DispatchTestParams {
  std::string name;
  std::vector<size_t> countsPerRank; // How many chunks go to each rank
  std::vector<size_t> chunkSizes; // Size of each chunk
};

// Test configuration derived from params
struct DispatchTestConfig {
  std::vector<size_t> chunkSizes;
  std::vector<size_t> chunkIndices;
  std::vector<size_t> chunkIndicesCountPerRank;
};

// Helper struct to hold device buffers for dispatch
struct DispatchDeviceBuffers {
  std::unique_ptr<DeviceBuffer> transportsDevice;
  std::unique_ptr<DeviceBuffer> sendBuffer;
  std::vector<std::unique_ptr<DeviceBuffer>> recvBuffers;
  std::unique_ptr<DeviceBuffer> recvBufferPtrsDevice;
  std::unique_ptr<DeviceBuffer> chunkSizesDevice;
  std::unique_ptr<DeviceBuffer> chunkIndicesDevice;
  std::unique_ptr<DeviceBuffer> chunkIndicesCountPerRankDevice;
  std::unique_ptr<DeviceBuffer> outputChunkSizesPerRankDevice;
  std::vector<void*> recvBufferPtrsHost;
  std::vector<uint8_t> sendData;
  size_t totalBufferSize;
};

// Helper to create uniform chunk sizes
inline std::vector<size_t> uniformSizes(size_t numChunks, size_t size) {
  return std::vector<size_t>(numChunks, size);
}

// Helper to create alternating chunk sizes
inline std::vector<size_t>
alternatingSizes(size_t numChunks, size_t size1, size_t size2) {
  std::vector<size_t> sizes(numChunks);
  for (size_t i = 0; i < numChunks; i++) {
    sizes[i] = (i % 2 == 0) ? size1 : size2;
  }
  return sizes;
}

// Helper to create ascending chunk sizes (8KB, 16KB, 32KB, 64KB, repeating)
inline std::vector<size_t> ascendingSizes(size_t numChunks) {
  std::vector<size_t> pattern = {
      8 * 1024, 16 * 1024, 32 * 1024, 64 * 1024}; // 8KB, 16KB, 32KB, 64KB
  std::vector<size_t> sizes(numChunks);
  for (size_t i = 0; i < numChunks; i++) {
    sizes[i] = pattern[i % pattern.size()];
  }
  return sizes;
}

// Helper to create variable chunk sizes for VarBoth test
inline std::vector<size_t> variableSizes(size_t numChunks) {
  std::vector<size_t> pattern = {
      16 * 1024, 32 * 1024, 64 * 1024, 32 * 1024}; // 16KB, 32KB, 64KB, 32KB
  std::vector<size_t> sizes(numChunks);
  for (size_t i = 0; i < numChunks; i++) {
    sizes[i] = pattern[i % pattern.size()];
  }
  return sizes;
}

class DispatchTestFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
  }

  void TearDown() override {
    MpiBaseTestFixture::TearDown();
  }

  // Setup transport array on device
  void setupTransports(
      MultiPeerNvlTransport& transport,
      DispatchDeviceBuffers& buffers) {
    std::size_t transportsSize = numRanks * sizeof(Transport);
    std::vector<char> transportsHostBuffer(transportsSize);

    for (int rank = 0; rank < numRanks; rank++) {
      Transport* slot = reinterpret_cast<Transport*>(
          transportsHostBuffer.data() + rank * sizeof(Transport));
      if (rank == globalRank) {
        new (slot) Transport(P2pSelfTransportDevice());
      } else {
        new (slot) Transport(transport.getP2pTransportDevice(rank));
      }
    }

    buffers.transportsDevice = std::make_unique<DeviceBuffer>(transportsSize);
    CUDACHECK_TEST(cudaMemcpy(
        buffers.transportsDevice->get(),
        transportsHostBuffer.data(),
        transportsSize,
        cudaMemcpyHostToDevice));

    // Destroy host Transport objects
    for (int rank = 0; rank < numRanks; rank++) {
      Transport* slot = reinterpret_cast<Transport*>(
          transportsHostBuffer.data() + rank * sizeof(Transport));
      slot->~Transport();
    }
  }

  // Setup send buffer with pattern data
  void setupSendBuffer(
      const DispatchTestConfig& config,
      DispatchDeviceBuffers& buffers) {
    size_t numChunks = config.chunkSizes.size();
    buffers.totalBufferSize = std::accumulate(
        config.chunkSizes.begin(), config.chunkSizes.end(), size_t{0});

    buffers.sendBuffer =
        std::make_unique<DeviceBuffer>(buffers.totalBufferSize);
    buffers.sendData.resize(buffers.totalBufferSize);

    size_t offset = 0;
    for (size_t i = 0; i < numChunks; i++) {
      // Fill each chunk with pattern: (rank * numChunks + chunk_index) mod 256
      uint8_t pattern =
          static_cast<uint8_t>((globalRank * numChunks + i) % 256);
      std::fill(
          buffers.sendData.begin() + offset,
          buffers.sendData.begin() + offset + config.chunkSizes[i],
          pattern);
      offset += config.chunkSizes[i];
    }

    CUDACHECK_TEST(cudaMemcpy(
        buffers.sendBuffer->get(),
        buffers.sendData.data(),
        buffers.totalBufferSize,
        cudaMemcpyHostToDevice));
  }

  // Setup receive buffers
  void setupRecvBuffers(DispatchDeviceBuffers& buffers) {
    buffers.recvBuffers.clear();
    buffers.recvBufferPtrsHost.resize(numRanks);

    for (int r = 0; r < numRanks; r++) {
      buffers.recvBuffers.push_back(
          std::make_unique<DeviceBuffer>(buffers.totalBufferSize));
      buffers.recvBufferPtrsHost[r] = buffers.recvBuffers[r]->get();
      CUDACHECK_TEST(cudaMemset(
          buffers.recvBuffers[r]->get(), 0xFF, buffers.totalBufferSize));
    }

    buffers.recvBufferPtrsDevice =
        std::make_unique<DeviceBuffer>(numRanks * sizeof(void*));
    CUDACHECK_TEST(cudaMemcpy(
        buffers.recvBufferPtrsDevice->get(),
        buffers.recvBufferPtrsHost.data(),
        numRanks * sizeof(void*),
        cudaMemcpyHostToDevice));
  }

  // Setup chunk metadata on device
  void setupChunkMetadata(
      const DispatchTestConfig& config,
      DispatchDeviceBuffers& buffers) {
    size_t numChunks = config.chunkSizes.size();

    // Chunk sizes
    buffers.chunkSizesDevice =
        std::make_unique<DeviceBuffer>(numChunks * sizeof(size_t));
    CUDACHECK_TEST(cudaMemcpy(
        buffers.chunkSizesDevice->get(),
        config.chunkSizes.data(),
        numChunks * sizeof(size_t),
        cudaMemcpyHostToDevice));

    // Chunk indices
    buffers.chunkIndicesDevice = std::make_unique<DeviceBuffer>(
        config.chunkIndices.size() * sizeof(size_t));
    CUDACHECK_TEST(cudaMemcpy(
        buffers.chunkIndicesDevice->get(),
        config.chunkIndices.data(),
        config.chunkIndices.size() * sizeof(size_t),
        cudaMemcpyHostToDevice));

    // Chunk indices count per rank
    buffers.chunkIndicesCountPerRankDevice =
        std::make_unique<DeviceBuffer>(numRanks * sizeof(size_t));
    CUDACHECK_TEST(cudaMemcpy(
        buffers.chunkIndicesCountPerRankDevice->get(),
        config.chunkIndicesCountPerRank.data(),
        numRanks * sizeof(size_t),
        cudaMemcpyHostToDevice));

    // Output chunk sizes per rank
    buffers.outputChunkSizesPerRankDevice =
        std::make_unique<DeviceBuffer>(numRanks * numChunks * sizeof(size_t));
    CUDACHECK_TEST(cudaMemset(
        buffers.outputChunkSizesPerRankDevice->get(),
        0,
        numRanks * numChunks * sizeof(size_t)));
  }

  // Run dispatch
  void runDispatch(
      const DispatchTestConfig& config,
      DispatchDeviceBuffers& buffers) {
    size_t numChunks = config.chunkSizes.size();

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    XLOGF(INFO, "Rank {}: Calling dispatch", globalRank);

    test::testDispatch(
        // Outputs
        DeviceSpan<void* const>(
            static_cast<void* const*>(buffers.recvBufferPtrsDevice->get()),
            numRanks),
        DeviceSpan<size_t>(
            static_cast<size_t*>(buffers.outputChunkSizesPerRankDevice->get()),
            numRanks * numChunks),
        // Inputs
        DeviceSpan<Transport>(
            static_cast<Transport*>(buffers.transportsDevice->get()), numRanks),
        globalRank,
        buffers.sendBuffer->get(),
        DeviceSpan<const size_t>(
            static_cast<const size_t*>(buffers.chunkSizesDevice->get()),
            numChunks),
        static_cast<const size_t*>(buffers.chunkIndicesDevice->get()),
        DeviceSpan<const size_t>(
            static_cast<const size_t*>(
                buffers.chunkIndicesCountPerRankDevice->get()),
            numRanks),
        nullptr, // default stream
        4, // num_blocks
        256); // num_threads

    CUDACHECK_TEST(cudaDeviceSynchronize());
    XLOGF(INFO, "Rank {}: Dispatch completed", globalRank);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  }

  // Verify received data from all peers
  void verifyReceivedData(
      const DispatchTestConfig& config,
      DispatchDeviceBuffers& buffers) {
    size_t numChunks = config.chunkSizes.size();

    for (int peerRank = 0; peerRank < numRanks; peerRank++) {
      std::vector<uint8_t> recvData(buffers.totalBufferSize);
      CUDACHECK_TEST(cudaMemcpy(
          recvData.data(),
          buffers.recvBufferPtrsHost[peerRank],
          buffers.totalBufferSize,
          cudaMemcpyDeviceToHost));

      // Calculate which chunks this peer sent to us (globalRank)
      size_t indicesOffset = 0;
      for (int r = 0; r < globalRank; r++) {
        indicesOffset += config.chunkIndicesCountPerRank[r];
      }
      size_t chunkCount = config.chunkIndicesCountPerRank[globalRank];

      for (size_t i = 0; i < chunkCount; i++) {
        size_t chunkIdx = config.chunkIndices[indicesOffset + i];
        size_t chunkSize = config.chunkSizes[chunkIdx];

        // Calculate offset in buffer for this chunk
        size_t chunkOffset = 0;
        for (size_t j = 0; j < chunkIdx; j++) {
          chunkOffset += config.chunkSizes[j];
        }

        // Expected pattern from peer
        uint8_t expectedValue =
            static_cast<uint8_t>((peerRank * numChunks + chunkIdx) % 256);

        for (size_t j = 0; j < chunkSize; j++) {
          EXPECT_EQ(recvData[chunkOffset + j], expectedValue)
              << "Rank " << globalRank << ": Mismatch at byte " << j
              << " of chunk " << chunkIdx << " from "
              << (peerRank == globalRank ? "self" : "peer ") << peerRank
              << ": expected " << static_cast<int>(expectedValue) << ", got "
              << static_cast<int>(recvData[chunkOffset + j]);
          if (recvData[chunkOffset + j] != expectedValue) {
            break; // Stop at first mismatch for this chunk
          }
        }
      }
    }
  }

  // Verify output chunk sizes (all-gather of input_chunk_sizes from each peer)
  void verifyOutputChunkSizes(
      const DispatchTestConfig& config,
      DispatchDeviceBuffers& buffers) {
    size_t numChunks = config.chunkSizes.size();

    std::vector<size_t> outputChunkSizes(numRanks * numChunks);
    CUDACHECK_TEST(cudaMemcpy(
        outputChunkSizes.data(),
        buffers.outputChunkSizesPerRankDevice->get(),
        numRanks * numChunks * sizeof(size_t),
        cudaMemcpyDeviceToHost));

    // All ranks use the same config, so all peers should have the same sizes
    for (int peerRank = 0; peerRank < numRanks; peerRank++) {
      size_t* peerChunkSizes = &outputChunkSizes[peerRank * numChunks];

      for (size_t i = 0; i < numChunks; i++) {
        EXPECT_EQ(peerChunkSizes[i], config.chunkSizes[i])
            << "Rank " << globalRank << ": Output chunk size mismatch for "
            << (peerRank == globalRank ? "self" : "peer ") << peerRank
            << " chunk " << i << ": expected " << config.chunkSizes[i]
            << ", got " << peerChunkSizes[i];
      }
    }
  }

  // Convert params to config
  DispatchTestConfig makeConfigFromParams(const DispatchTestParams& params) {
    DispatchTestConfig config;

    // Copy chunk sizes directly
    config.chunkSizes = params.chunkSizes;

    // Sequential indices: [0, 1, 2, ..., numChunks-1]
    size_t numChunks = params.chunkSizes.size();
    config.chunkIndices.resize(numChunks);
    std::iota(config.chunkIndices.begin(), config.chunkIndices.end(), 0);

    // Copy counts per rank
    config.chunkIndicesCountPerRank = params.countsPerRank;

    return config;
  }

  // Run a complete dispatch test with given config
  void runDispatchTest(const DispatchTestConfig& config) {
    if (numRanks != 8) {
      XLOGF(
          WARNING, "Skipping test: requires exactly 8 ranks, got {}", numRanks);
      return;
    }

    // Calculate buffer size needed for transport
    size_t totalBufferSize = std::accumulate(
        config.chunkSizes.begin(), config.chunkSizes.end(), size_t{0});
    size_t maxChunkSize =
        *std::max_element(config.chunkSizes.begin(), config.chunkSizes.end());

    // Transport configuration
    MultiPeerNvlTransportConfig transportConfig{
        .dataBufferSize = totalBufferSize + 4096,
        .chunkSize = std::min(maxChunkSize, size_t{65536}),
        .pipelineDepth = 4,
    };

    auto bootstrap = std::make_shared<MpiBootstrap>();
    MultiPeerNvlTransport transport(
        globalRank, numRanks, bootstrap, transportConfig);
    transport.exchange();
    XLOGF(INFO, "Rank {}: Transport initialized", globalRank);

    DispatchDeviceBuffers buffers;
    setupTransports(transport, buffers);
    setupSendBuffer(config, buffers);
    setupRecvBuffers(buffers);
    setupChunkMetadata(config, buffers);

    runDispatch(config, buffers);

    verifyReceivedData(config, buffers);
    verifyOutputChunkSizes(config, buffers);

    XLOGF(INFO, "Rank {}: Test completed successfully", globalRank);
  }
};

// Parameterized test class
class DispatchParamTest
    : public DispatchTestFixture,
      public ::testing::WithParamInterface<DispatchTestParams> {};

// The parameterized test
TEST_P(DispatchParamTest, Run) {
  auto params = GetParam();
  XLOGF(INFO, "Rank {}: Starting test '{}'", globalRank, params.name);
  auto config = makeConfigFromParams(params);
  runDispatchTest(config);
}

// Custom naming function for better test output
std::string DispatchTestName(
    const ::testing::TestParamInfo<DispatchTestParams>& info) {
  return info.param.name;
}

// Balanced test cases: uniform counts, uniform/mixed sizes
INSTANTIATE_TEST_SUITE_P(
    Balanced,
    DispatchParamTest,
    ::testing::Values(
        // Each rank sends 1 small chunk (16KB) to each peer
        DispatchTestParams{
            "Uniform_1x16K",
            {1, 1, 1, 1, 1, 1, 1, 1},
            uniformSizes(8, kSmallChunkSize)},
        // Each rank sends 2 small chunks (16KB each) to each peer
        DispatchTestParams{
            "Uniform_2x16K",
            {2, 2, 2, 2, 2, 2, 2, 2},
            uniformSizes(16, kSmallChunkSize)},
        // Each rank sends 2 large chunks (1MB each) to each peer
        DispatchTestParams{
            "Uniform_2x1M",
            {2, 2, 2, 2, 2, 2, 2, 2},
            uniformSizes(16, kLargeChunkSize)},
        // Each rank sends 2 chunks with alternating sizes (16KB, 1MB) to each
        // peer
        DispatchTestParams{
            "Mixed_2xAlt",
            {2, 2, 2, 2, 2, 2, 2, 2},
            alternatingSizes(16, kSmallChunkSize, kLargeChunkSize)}),
    DispatchTestName);

// Imbalanced test cases: variable counts and/or variable sizes
INSTANTIATE_TEST_SUITE_P(
    Imbalanced,
    DispatchParamTest,
    ::testing::Values(
        // Alternating chunk counts per peer: 1,3,1,3... (some peers get more)
        DispatchTestParams{
            "VarCounts",
            {1, 3, 1, 3, 1, 3, 1, 3},
            uniformSizes(16, kSmallChunkSize)},
        // Uniform counts but ascending chunk sizes: 8KB, 16KB, 32KB, 64KB
        DispatchTestParams{
            "VarSizes",
            {2, 2, 2, 2, 2, 2, 2, 2},
            ascendingSizes(16)},
        // Both variable counts (1,2,3,2,1,2,3,2) and variable sizes
        DispatchTestParams{
            "VarBoth",
            {1, 2, 3, 2, 1, 2, 3, 2},
            variableSizes(16)}),
    DispatchTestName);

} // namespace comms::pipes::tests

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
