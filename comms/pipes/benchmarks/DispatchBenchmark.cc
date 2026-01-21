// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>
#include <gtest/gtest.h>

#include <iomanip>
#include <numeric>
#include <sstream>
#include <vector>

#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/MultiPeerNvlTransport.h"
#include "comms/pipes/P2pSelfTransportDevice.cuh"
#include "comms/pipes/Transport.cuh"
#include "comms/pipes/collectives/dispatch.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::CudaEvent;
using meta::comms::DeviceBuffer;
using meta::comms::MpiBaseTestFixture;
using meta::comms::MPIEnvironmentBase;

namespace comms::pipes::benchmark {

// CUDA error checking macro for void functions
#define CUDA_CHECK_VOID(call)        \
  do {                               \
    cudaError_t err = call;          \
    if (err != cudaSuccess) {        \
      XLOGF(                         \
          ERR,                       \
          "CUDA error at {}:{}: {}", \
          __FILE__,                  \
          __LINE__,                  \
          cudaGetErrorString(err));  \
      return;                        \
    }                                \
  } while (0)

// CUDA error checking macro for float-returning functions
#define CUDA_CHECK(call)             \
  do {                               \
    cudaError_t err = call;          \
    if (err != cudaSuccess) {        \
      XLOGF(                         \
          ERR,                       \
          "CUDA error at {}:{}: {}", \
          __FILE__,                  \
          __LINE__,                  \
          cudaGetErrorString(err));  \
      return 0.0f;                   \
    }                                \
  } while (0)

namespace {

struct DispatchBenchmarkConfig {
  std::size_t perPeerBytes; // Data size per peer
  int chunksPerPeer; // Number of chunks per peer (1 or 2)
  int numBlocks;
  int numThreads;
};

struct DispatchBenchmarkResult {
  std::size_t perPeerBytes;
  int chunksPerPeer;
  std::size_t chunkSize;
  std::size_t totalBytes;
  int numBlocks;
  int numThreads;
  float latencyUs;
  float bandwidthGBps;
};

class DispatchBenchmarkFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDA_CHECK_VOID(cudaSetDevice(localRank));
  }

  void TearDown() override {
    MpiBaseTestFixture::TearDown();
  }

  float runDispatchBenchmark(
      const DispatchBenchmarkConfig& config,
      float& latencyUs) {
    const int nranks = numRanks;
    const std::size_t perPeerBytes = config.perPeerBytes;
    const int chunksPerPeer = config.chunksPerPeer;
    const std::size_t chunkSize = perPeerBytes / chunksPerPeer;
    const int totalChunks = nranks * chunksPerPeer;
    const std::size_t totalBytes = perPeerBytes * nranks;

    // Allocate send buffer
    DeviceBuffer sendBuffer(totalBytes);
    std::vector<uint8_t> h_send(totalBytes, 0xAB);
    CUDA_CHECK(cudaMemcpy(
        sendBuffer.get(), h_send.data(), totalBytes, cudaMemcpyHostToDevice));

    // Allocate receive buffers (one per rank)
    std::vector<std::unique_ptr<DeviceBuffer>> recvBuffers;
    std::vector<void*> recvBufferPtrsHost(nranks);
    for (int r = 0; r < nranks; r++) {
      recvBuffers.push_back(std::make_unique<DeviceBuffer>(totalBytes));
      recvBufferPtrsHost[r] = recvBuffers[r]->get();
      CUDA_CHECK(cudaMemset(recvBuffers[r]->get(), 0, totalBytes));
    }

    DeviceBuffer recvBufferPtrsDevice(nranks * sizeof(void*));
    CUDA_CHECK(cudaMemcpy(
        recvBufferPtrsDevice.get(),
        recvBufferPtrsHost.data(),
        nranks * sizeof(void*),
        cudaMemcpyHostToDevice));

    // Setup chunk sizes (all equal)
    std::vector<std::size_t> chunkSizes(totalChunks, chunkSize);
    DeviceBuffer chunkSizesDevice(totalChunks * sizeof(std::size_t));
    CUDA_CHECK(cudaMemcpy(
        chunkSizesDevice.get(),
        chunkSizes.data(),
        totalChunks * sizeof(std::size_t),
        cudaMemcpyHostToDevice));

    // Setup chunk indices: sequential [0, 1, 2, ..., totalChunks-1]
    std::vector<std::size_t> chunkIndices(totalChunks);
    std::iota(chunkIndices.begin(), chunkIndices.end(), 0);
    DeviceBuffer chunkIndicesDevice(totalChunks * sizeof(std::size_t));
    CUDA_CHECK(cudaMemcpy(
        chunkIndicesDevice.get(),
        chunkIndices.data(),
        totalChunks * sizeof(std::size_t),
        cudaMemcpyHostToDevice));

    // Setup chunk indices count per rank (equal distribution)
    std::vector<std::size_t> chunkIndicesCountPerRank(nranks, chunksPerPeer);
    DeviceBuffer chunkIndicesCountPerRankDevice(nranks * sizeof(std::size_t));
    CUDA_CHECK(cudaMemcpy(
        chunkIndicesCountPerRankDevice.get(),
        chunkIndicesCountPerRank.data(),
        nranks * sizeof(std::size_t),
        cudaMemcpyHostToDevice));

    // Setup output chunk sizes per rank
    DeviceBuffer outputChunkSizesPerRankDevice(
        nranks * totalChunks * sizeof(std::size_t));
    CUDA_CHECK(cudaMemset(
        outputChunkSizesPerRankDevice.get(),
        0,
        nranks * totalChunks * sizeof(std::size_t)));

    // Setup transport - use larger buffer for bigger messages
    std::size_t dataBufferSize =
        std::max(totalBytes + 4096, std::size_t{8 * 1024 * 1024});
    MultiPeerNvlTransportConfig transportConfig{
        .dataBufferSize = dataBufferSize,
        .chunkSize = std::min(chunkSize, std::size_t{512 * 1024}),
        .pipelineDepth = 4,
    };

    auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
    MultiPeerNvlTransport transport(
        globalRank, nranks, bootstrap, transportConfig);
    transport.exchange();

    // Create transport array on device
    std::size_t transportsSize = nranks * sizeof(Transport);
    std::vector<char> transportsHostBuffer(transportsSize);
    for (int rank = 0; rank < nranks; rank++) {
      Transport* slot = reinterpret_cast<Transport*>(
          transportsHostBuffer.data() + rank * sizeof(Transport));
      if (rank == globalRank) {
        new (slot) Transport(P2pSelfTransportDevice());
      } else {
        new (slot) Transport(transport.getP2pTransportDevice(rank));
      }
    }

    DeviceBuffer transportsDevice(transportsSize);
    CUDA_CHECK(cudaMemcpy(
        transportsDevice.get(),
        transportsHostBuffer.data(),
        transportsSize,
        cudaMemcpyHostToDevice));

    // Destroy host Transport objects
    for (int rank = 0; rank < nranks; rank++) {
      Transport* slot = reinterpret_cast<Transport*>(
          transportsHostBuffer.data() + rank * sizeof(Transport));
      slot->~Transport();
    }

    // Benchmark timing
    CudaEvent start, stop;
    const int nIter = 20;
    const int nIterWarmup = 5;

    // Warmup
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    for (int i = 0; i < nIterWarmup; i++) {
      collectives::dispatch(
          DeviceSpan<void* const>(
              static_cast<void* const*>(recvBufferPtrsDevice.get()), nranks),
          DeviceSpan<std::size_t>(
              static_cast<std::size_t*>(outputChunkSizesPerRankDevice.get()),
              nranks * totalChunks),
          DeviceSpan<Transport>(
              static_cast<Transport*>(transportsDevice.get()), nranks),
          globalRank,
          sendBuffer.get(),
          DeviceSpan<const std::size_t>(
              static_cast<const std::size_t*>(chunkSizesDevice.get()),
              totalChunks),
          static_cast<const std::size_t*>(chunkIndicesDevice.get()),
          DeviceSpan<const std::size_t>(
              static_cast<const std::size_t*>(
                  chunkIndicesCountPerRankDevice.get()),
              nranks),
          nullptr,
          config.numBlocks,
          config.numThreads);
      CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Timed iterations
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    std::vector<float> latencies(nIter);
    for (int i = 0; i < nIter; i++) {
      CUDA_CHECK(cudaEventRecord(start.get()));
      collectives::dispatch(
          DeviceSpan<void* const>(
              static_cast<void* const*>(recvBufferPtrsDevice.get()), nranks),
          DeviceSpan<std::size_t>(
              static_cast<std::size_t*>(outputChunkSizesPerRankDevice.get()),
              nranks * totalChunks),
          DeviceSpan<Transport>(
              static_cast<Transport*>(transportsDevice.get()), nranks),
          globalRank,
          sendBuffer.get(),
          DeviceSpan<const std::size_t>(
              static_cast<const std::size_t*>(chunkSizesDevice.get()),
              totalChunks),
          static_cast<const std::size_t*>(chunkIndicesDevice.get()),
          DeviceSpan<const std::size_t>(
              static_cast<const std::size_t*>(
                  chunkIndicesCountPerRankDevice.get()),
              nranks),
          nullptr,
          config.numBlocks,
          config.numThreads);
      CUDA_CHECK(cudaEventRecord(stop.get()));
      CUDA_CHECK(cudaEventSynchronize(stop.get()));
      float ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, start.get(), stop.get()));
      latencies[i] = ms * 1000.0f; // Convert to microseconds
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Compute average latency
    float totalLatency = 0.0f;
    for (float lat : latencies) {
      totalLatency += lat;
    }
    latencyUs = totalLatency / nIter;

    // Compute bandwidth: total bytes / latency
    // BW (GB/s) = bytes / (latency_us * 1e-6) / 1e9 = bytes / (latency_us *
    // 1e3)
    float bandwidthGBps = static_cast<float>(totalBytes) / (latencyUs * 1e3);

    return bandwidthGBps;
  }

  void printResultsTable(const std::vector<DispatchBenchmarkResult>& results) {
    if (globalRank != 0) {
      return;
    }

    auto formatBytes = [](std::size_t bytes) -> std::string {
      std::stringstream ss;
      if (bytes < 1024) {
        ss << bytes << " B";
      } else if (bytes < 1024 * 1024) {
        ss << (bytes / 1024) << " KB";
      } else if (bytes < 1024ULL * 1024 * 1024) {
        ss << (bytes / (1024 * 1024)) << " MB";
      } else {
        ss << (bytes / (1024ULL * 1024 * 1024)) << " GB";
      }
      return ss.str();
    };

    std::stringstream ss;
    ss << "\n";
    ss << "=== Dispatch Benchmark (" << numRanks
       << " ranks, 256 threads) ===\n";
    ss << "Warmup: 5 iterations, Timed: 20 iterations\n\n";

    ss << "| " << std::right << std::setw(10) << "Per-Peer"
       << " | " << std::setw(6) << "Chunks"
       << " | " << std::setw(10) << "Chunk Size"
       << " | " << std::setw(10) << "Total Data"
       << " | " << std::setw(6) << "Blocks"
       << " | " << std::setw(12) << "Latency (μs)"
       << " | " << std::setw(10) << "BW (GB/s)"
       << " |\n";

    ss << "|" << std::string(12, '-') << "|" << std::string(8, '-') << "|"
       << std::string(12, '-') << "|" << std::string(12, '-') << "|"
       << std::string(8, '-') << "|" << std::string(14, '-') << "|"
       << std::string(12, '-') << "|\n";

    for (const auto& r : results) {
      ss << "| " << std::right << std::setw(10) << formatBytes(r.perPeerBytes)
         << " | " << std::setw(6) << r.chunksPerPeer << " | " << std::setw(10)
         << formatBytes(r.chunkSize) << " | " << std::setw(10)
         << formatBytes(r.totalBytes) << " | " << std::setw(6) << r.numBlocks
         << " | " << std::setw(12) << std::fixed << std::setprecision(2)
         << r.latencyUs << " | " << std::setw(10) << std::fixed
         << std::setprecision(2) << r.bandwidthGBps << " |\n";
    }

    ss << "\n";
    XLOG(INFO) << ss.str();
  }
};

TEST_F(DispatchBenchmarkFixture, Benchmark) {
  if (numRanks != 8) {
    XLOGF(WARNING, "Skipping: requires exactly 8 ranks, got {}", numRanks);
    return;
  }

  std::vector<DispatchBenchmarkResult> results;

  // Per-peer data sizes: 8KB, 16KB, 1MB
  std::vector<std::size_t> perPeerSizes = {
      8 * 1024, // 8KB
      // 16 * 1024, // 16KB
      // 1024 * 1024 // 1MB
  };
  std::vector<int> chunksPerPeerOptions = {1};
  std::vector<int> blockOptions = {8, 16, 32, 64, 128};

  for (std::size_t perPeerBytes : perPeerSizes) {
    for (int chunksPerPeer : chunksPerPeerOptions) {
      for (int numBlocks : blockOptions) {
        DispatchBenchmarkConfig config{
            .perPeerBytes = perPeerBytes,
            .chunksPerPeer = chunksPerPeer,
            .numBlocks = numBlocks,
            .numThreads = 256,
        };

        float latencyUs = 0.0f;
        float bandwidthGBps = runDispatchBenchmark(config, latencyUs);

        if (globalRank == 0) {
          std::size_t chunkSize = perPeerBytes / chunksPerPeer;
          std::size_t totalBytes = perPeerBytes * numRanks;

          DispatchBenchmarkResult result{
              .perPeerBytes = perPeerBytes,
              .chunksPerPeer = chunksPerPeer,
              .chunkSize = chunkSize,
              .totalBytes = totalBytes,
              .numBlocks = numBlocks,
              .numThreads = 256,
              .latencyUs = latencyUs,
              .bandwidthGBps = bandwidthGBps,
          };
          results.push_back(result);
        }

        MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
      }
    }
  }

  printResultsTable(results);
}

} // namespace

} // namespace comms::pipes::benchmark

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new meta::comms::MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
