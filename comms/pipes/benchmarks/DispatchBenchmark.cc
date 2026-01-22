// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>
#include <gtest/gtest.h>

#include <iomanip>
#include <iostream>
#include <map>
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

using comms::pipes::collectives::ShardingMode;
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
  ShardingMode mode; // Sharding mode (VERTICAL or HORIZONTAL)
};

struct DispatchBenchmarkResult {
  std::size_t perPeerBytes;
  int chunksPerPeer;
  std::size_t chunkSize;
  std::size_t totalBytes;
  int numBlocks;
  int numThreads;
  ShardingMode mode;
  float latencyUs;
  float bandwidthGBps;
};

// Config for imbalanced benchmark: rank 0 sends largeBytes to rank 1,
// all other sends are smallBytes
struct ImbalancedBenchmarkConfig {
  std::size_t largeBytes; // Size of large transfer (rank 0 -> rank 1)
  std::size_t smallBytes; // Size of small transfers (all others)
  int numBlocks;
  int numThreads;
  ShardingMode mode;
};

struct ImbalancedBenchmarkResult {
  std::size_t largeBytes;
  std::size_t smallBytes;
  int numBlocks;
  ShardingMode mode;
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
          config.numThreads,
          config.mode);
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
          config.numThreads,
          config.mode);
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

  // Imbalanced benchmark: rank 0 sends largeBytes to rank 1,
  // all other transfers are smallBytes
  float runImbalancedDispatchBenchmark(
      const ImbalancedBenchmarkConfig& config,
      float& latencyUs) {
    const int nranks = numRanks;
    const std::size_t largeBytes = config.largeBytes;
    const std::size_t smallBytes = config.smallBytes;

    // Each rank has nranks chunks (one per destination)
    // For rank 0: chunk[1] = largeBytes, others = smallBytes
    // For other ranks: all chunks = smallBytes
    std::vector<std::size_t> chunkSizes(nranks);
    std::size_t totalBytes = 0;
    for (int r = 0; r < nranks; r++) {
      if (globalRank == 0 && r == 1) {
        chunkSizes[r] = largeBytes;
      } else {
        chunkSizes[r] = smallBytes;
      }
      totalBytes += chunkSizes[r];
    }

    // Allocate send buffer
    DeviceBuffer sendBuffer(totalBytes);
    std::vector<uint8_t> h_send(totalBytes, 0xAB);
    CUDA_CHECK(cudaMemcpy(
        sendBuffer.get(), h_send.data(), totalBytes, cudaMemcpyHostToDevice));

    // Allocate receive buffers (one per rank)
    // Each recv buffer needs to hold the max possible size from any sender
    std::size_t maxRecvSize = std::max(largeBytes, smallBytes) * nranks;
    std::vector<std::unique_ptr<DeviceBuffer>> recvBuffers;
    std::vector<void*> recvBufferPtrsHost(nranks);
    for (int r = 0; r < nranks; r++) {
      recvBuffers.push_back(std::make_unique<DeviceBuffer>(maxRecvSize));
      recvBufferPtrsHost[r] = recvBuffers[r]->get();
      CUDA_CHECK(cudaMemset(recvBuffers[r]->get(), 0, maxRecvSize));
    }

    DeviceBuffer recvBufferPtrsDevice(nranks * sizeof(void*));
    CUDA_CHECK(cudaMemcpy(
        recvBufferPtrsDevice.get(),
        recvBufferPtrsHost.data(),
        nranks * sizeof(void*),
        cudaMemcpyHostToDevice));

    // Setup chunk sizes on device
    DeviceBuffer chunkSizesDevice(nranks * sizeof(std::size_t));
    CUDA_CHECK(cudaMemcpy(
        chunkSizesDevice.get(),
        chunkSizes.data(),
        nranks * sizeof(std::size_t),
        cudaMemcpyHostToDevice));

    // Setup chunk indices: sequential [0, 1, 2, ..., nranks-1]
    std::vector<std::size_t> chunkIndices(nranks);
    std::iota(chunkIndices.begin(), chunkIndices.end(), 0);
    DeviceBuffer chunkIndicesDevice(nranks * sizeof(std::size_t));
    CUDA_CHECK(cudaMemcpy(
        chunkIndicesDevice.get(),
        chunkIndices.data(),
        nranks * sizeof(std::size_t),
        cudaMemcpyHostToDevice));

    // Setup chunk indices count per rank (1 chunk per rank)
    std::vector<std::size_t> chunkIndicesCountPerRank(nranks, 1);
    DeviceBuffer chunkIndicesCountPerRankDevice(nranks * sizeof(std::size_t));
    CUDA_CHECK(cudaMemcpy(
        chunkIndicesCountPerRankDevice.get(),
        chunkIndicesCountPerRank.data(),
        nranks * sizeof(std::size_t),
        cudaMemcpyHostToDevice));

    // Setup output chunk sizes per rank
    DeviceBuffer outputChunkSizesPerRankDevice(
        nranks * nranks * sizeof(std::size_t));
    CUDA_CHECK(cudaMemset(
        outputChunkSizesPerRankDevice.get(),
        0,
        nranks * nranks * sizeof(std::size_t)));

    // Setup transport - use larger buffer for the 1MB transfer
    std::size_t dataBufferSize =
        std::max(largeBytes + 4096, std::size_t{8 * 1024 * 1024});
    MultiPeerNvlTransportConfig transportConfig{
        .dataBufferSize = dataBufferSize,
        .chunkSize = std::min(largeBytes, std::size_t{512 * 1024}),
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
              nranks * nranks),
          DeviceSpan<Transport>(
              static_cast<Transport*>(transportsDevice.get()), nranks),
          globalRank,
          sendBuffer.get(),
          DeviceSpan<const std::size_t>(
              static_cast<const std::size_t*>(chunkSizesDevice.get()), nranks),
          static_cast<const std::size_t*>(chunkIndicesDevice.get()),
          DeviceSpan<const std::size_t>(
              static_cast<const std::size_t*>(
                  chunkIndicesCountPerRankDevice.get()),
              nranks),
          nullptr,
          config.numBlocks,
          config.numThreads,
          config.mode);
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
              nranks * nranks),
          DeviceSpan<Transport>(
              static_cast<Transport*>(transportsDevice.get()), nranks),
          globalRank,
          sendBuffer.get(),
          DeviceSpan<const std::size_t>(
              static_cast<const std::size_t*>(chunkSizesDevice.get()), nranks),
          static_cast<const std::size_t*>(chunkIndicesDevice.get()),
          DeviceSpan<const std::size_t>(
              static_cast<const std::size_t*>(
                  chunkIndicesCountPerRankDevice.get()),
              nranks),
          nullptr,
          config.numBlocks,
          config.numThreads,
          config.mode);
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

    // Compute bandwidth based on max transfer (the 1MB one dominates)
    // For rank 0/1, this is the large transfer; for others, it's small
    std::size_t maxTransfer = (globalRank == 0 || globalRank == 1)
        ? largeBytes
        : smallBytes * (nranks - 1);
    float bandwidthGBps = static_cast<float>(maxTransfer) / (latencyUs * 1e3);

    return bandwidthGBps;
  }

  // Helper function to print tables with proper synchronization
  void printTable(const std::string& table) {
    // Synchronize all ranks and flush CUDA before printing
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    cudaDeviceSynchronize();

    if (globalRank == 0) {
      std::cout << table << std::flush;
    }
  }

  // Helper function to format bytes
  std::string formatBytes(std::size_t bytes) {
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
  }

  // Helper function to format mode
  std::string formatMode(ShardingMode mode) {
    return mode == ShardingMode::HORIZONTAL ? "H" : "V";
  }

  void printResultsTable(const std::vector<DispatchBenchmarkResult>& results) {
    std::stringstream ss;
    ss << "\n";
    ss << "╔════════════════════════════════════════════════════════════════════════════════════════╗\n";
    ss << "║                     TABLE 1: BALANCED WORKLOAD BENCHMARK                               ║\n";
    ss << "╠════════════════════════════════════════════════════════════════════════════════════════╣\n";
    ss << "║  " << numRanks
       << " ranks, 256 threads | Warmup: 5 iter, Timed: 20 iter                                ║\n";
    ss << "║  All ranks send equal-sized data to all peers                                          ║\n";
    ss << "║  Mode: H=Horizontal (parallel peers), V=Vertical (sequential peers)                    ║\n";
    ss << "╚════════════════════════════════════════════════════════════════════════════════════════╝\n\n";

    ss << "| " << std::right << std::setw(4) << "Mode"
       << " | " << std::setw(10) << "Per-Peer"
       << " | " << std::setw(6) << "Chunks"
       << " | " << std::setw(10) << "Chunk Size"
       << " | " << std::setw(10) << "Total Data"
       << " | " << std::setw(6) << "Blocks"
       << " | " << std::setw(12) << "Latency (μs)"
       << " | " << std::setw(10) << "BW (GB/s)"
       << " | " << std::setw(6) << "Winner"
       << " |\n";

    ss << "|" << std::string(6, '-') << "|" << std::string(12, '-') << "|"
       << std::string(8, '-') << "|" << std::string(12, '-') << "|"
       << std::string(12, '-') << "|" << std::string(8, '-') << "|"
       << std::string(14, '-') << "|" << std::string(12, '-') << "|"
       << std::string(8, '-') << "|\n";

    // Group results by per-peer size to determine winner
    std::map<std::size_t, std::pair<float, float>>
        latencyBySize; // size -> (H latency, V latency)
    for (const auto& r : results) {
      if (r.mode == ShardingMode::HORIZONTAL) {
        latencyBySize[r.perPeerBytes].first = r.latencyUs;
      } else {
        latencyBySize[r.perPeerBytes].second = r.latencyUs;
      }
    }

    for (const auto& r : results) {
      auto& latencies = latencyBySize[r.perPeerBytes];
      std::string winner = "";
      if (r.mode == ShardingMode::HORIZONTAL &&
          latencies.first < latencies.second) {
        winner = "<-- H";
      } else if (
          r.mode == ShardingMode::VERTICAL &&
          latencies.second < latencies.first) {
        winner = "<-- V";
      }

      ss << "| " << std::right << std::setw(4) << formatMode(r.mode) << " | "
         << std::setw(10) << formatBytes(r.perPeerBytes) << " | "
         << std::setw(6) << r.chunksPerPeer << " | " << std::setw(10)
         << formatBytes(r.chunkSize) << " | " << std::setw(10)
         << formatBytes(r.totalBytes) << " | " << std::setw(6) << r.numBlocks
         << " | " << std::setw(12) << std::fixed << std::setprecision(2)
         << r.latencyUs << " | " << std::setw(10) << std::fixed
         << std::setprecision(2) << r.bandwidthGBps << " | " << std::setw(6)
         << winner << " |\n";
    }

    ss << "\n";
    printTable(ss.str());
  }

  void printImbalancedResultsTable(
      const std::vector<ImbalancedBenchmarkResult>& results) {
    std::stringstream ss;
    ss << "\n";
    ss << "╔════════════════════════════════════════════════════════════════════════════════════════╗\n";
    ss << "║                     TABLE 2: IMBALANCED WORKLOAD BENCHMARK                             ║\n";
    ss << "╠════════════════════════════════════════════════════════════════════════════════════════╣\n";
    ss << "║  " << numRanks
       << " ranks, 256 threads | Warmup: 5 iter, Timed: 20 iter                                ║\n";
    ss << "║  Rank 0 sends LARGE to Rank 1, all other transfers are SMALL                           ║\n";
    ss << "║  Mode: H=Horizontal (parallel peers), V=Vertical (sequential peers)                    ║\n";
    ss << "╚════════════════════════════════════════════════════════════════════════════════════════╝\n\n";

    ss << "| " << std::right << std::setw(4) << "Mode"
       << " | " << std::setw(10) << "Large"
       << " | " << std::setw(10) << "Small"
       << " | " << std::setw(6) << "Blocks"
       << " | " << std::setw(12) << "Latency (μs)"
       << " | " << std::setw(10) << "BW (GB/s)"
       << " | " << std::setw(6) << "Winner"
       << " |\n";

    ss << "|" << std::string(6, '-') << "|" << std::string(12, '-') << "|"
       << std::string(12, '-') << "|" << std::string(8, '-') << "|"
       << std::string(14, '-') << "|" << std::string(12, '-') << "|"
       << std::string(8, '-') << "|\n";

    // Group results by large size to determine winner
    std::map<std::size_t, std::pair<float, float>>
        latencyBySize; // size -> (H latency, V latency)
    for (const auto& r : results) {
      if (r.mode == ShardingMode::HORIZONTAL) {
        latencyBySize[r.largeBytes].first = r.latencyUs;
      } else {
        latencyBySize[r.largeBytes].second = r.latencyUs;
      }
    }

    for (const auto& r : results) {
      auto& latencies = latencyBySize[r.largeBytes];
      std::string winner = "";
      if (r.mode == ShardingMode::HORIZONTAL &&
          latencies.first < latencies.second) {
        winner = "<-- H";
      } else if (
          r.mode == ShardingMode::VERTICAL &&
          latencies.second < latencies.first) {
        winner = "<-- V";
      }

      ss << "| " << std::right << std::setw(4) << formatMode(r.mode) << " | "
         << std::setw(10) << formatBytes(r.largeBytes) << " | " << std::setw(10)
         << formatBytes(r.smallBytes) << " | " << std::setw(6) << r.numBlocks
         << " | " << std::setw(12) << std::fixed << std::setprecision(2)
         << r.latencyUs << " | " << std::setw(10) << std::fixed
         << std::setprecision(2) << r.bandwidthGBps << " | " << std::setw(6)
         << winner << " |\n";
    }

    ss << "\n";
    printTable(ss.str());
  }
};

TEST_F(DispatchBenchmarkFixture, Benchmark) {
  if (numRanks != 8) {
    XLOGF(WARNING, "Skipping: requires exactly 8 ranks, got {}", numRanks);
    return;
  }

  std::vector<DispatchBenchmarkResult> results;

  // Per-peer data sizes: 8KB, 64KB, 1MB
  std::vector<std::size_t> perPeerSizes = {
      8 * 1024, // 8KB
      64 * 1024, // 64KB
      1024 * 1024, // 1MB
  };
  std::vector<int> chunksPerPeerOptions = {1};
  std::vector<int> blockOptions = {8, 16, 32};
  std::vector<ShardingMode> modeOptions = {
      ShardingMode::HORIZONTAL,
      ShardingMode::VERTICAL,
  };

  for (ShardingMode mode : modeOptions) {
    for (std::size_t perPeerBytes : perPeerSizes) {
      for (int chunksPerPeer : chunksPerPeerOptions) {
        for (int numBlocks : blockOptions) {
          DispatchBenchmarkConfig config{
              .perPeerBytes = perPeerBytes,
              .chunksPerPeer = chunksPerPeer,
              .numBlocks = numBlocks,
              .numThreads = 256,
              .mode = mode,
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
                .mode = mode,
                .latencyUs = latencyUs,
                .bandwidthGBps = bandwidthGBps,
            };
            results.push_back(result);
          }

          MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
        }
      }
    }
  }

  printResultsTable(results);
}

TEST_F(DispatchBenchmarkFixture, ImbalancedBenchmark) {
  if (numRanks != 8) {
    XLOGF(WARNING, "Skipping: requires exactly 8 ranks, got {}", numRanks);
    return;
  }

  std::vector<ImbalancedBenchmarkResult> results;

  // Imbalanced case: rank 0 sends 32MB to rank 1, all others send/recv 8KB
  // Sweep block counts to show effect of parallelism
  std::size_t largeBytes = 32 * 1024 * 1024; // 32MB
  std::size_t smallBytes = 8 * 1024; // 8KB

  std::vector<int> blockOptions = {8, 16, 32};
  std::vector<ShardingMode> modeOptions = {
      ShardingMode::HORIZONTAL,
      ShardingMode::VERTICAL,
  };

  for (ShardingMode mode : modeOptions) {
    for (int numBlocks : blockOptions) {
      ImbalancedBenchmarkConfig config{
          .largeBytes = largeBytes,
          .smallBytes = smallBytes,
          .numBlocks = numBlocks,
          .numThreads = 256,
          .mode = mode,
      };

      float latencyUs = 0.0f;
      float bandwidthGBps = runImbalancedDispatchBenchmark(config, latencyUs);

      if (globalRank == 0) {
        ImbalancedBenchmarkResult result{
            .largeBytes = largeBytes,
            .smallBytes = smallBytes,
            .numBlocks = numBlocks,
            .mode = mode,
            .latencyUs = latencyUs,
            .bandwidthGBps = bandwidthGBps,
        };
        results.push_back(result);
      }

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    }
  }

  printImbalancedResultsTable(results);
}

} // namespace

} // namespace comms::pipes::benchmark

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new meta::comms::MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
