// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>
#include <nccl.h>

#include "comms/common/CudaWrap.h"
#include "comms/pipes/MultiPeerNvlTransport.h"
#include "comms/pipes/benchmarks/BenchmarkMacros.h"
#include "comms/pipes/collectives/benchmarks/CollectiveBenchmark.cuh"
#include "comms/testinfra/BenchmarkTestFixture.h"
#include "comms/utils/CudaRAII.h"

#include <iomanip>
#include <sstream>
#include <vector>

using meta::comms::CudaEvent;
using meta::comms::DeviceBuffer;

namespace comms::pipes::benchmark {

namespace {
/**
 * Test configuration for AllGather benchmark.
 */
struct AllGatherBenchmarkConfig {
  std::size_t sendcount; // Message size per rank
  int numBlocks;
  int numThreads;
  std::size_t pipelineDepth = 4;
  std::size_t chunkSize = 512 * 1024; // 512KB default
  std::size_t dataBufferSize = 2048; // Data buffer size for P2P transport
  bool spreadClusterLaunch = false; // Use spread cluster kernel launch
  std::string name;
};

/**
 * Result struct for collecting benchmark data.
 */
struct AllGatherBenchmarkResult {
  std::string testName;
  std::size_t sendcount{}; // Per-rank message size
  std::size_t totalBytes{}; // Total across all ranks
  std::size_t pipelineDepth{};
  std::size_t chunkSize{};
  float ncclBandwidth{}; // GB/s (ncclAllGather)
  float allgatherBandwidth{}; // GB/s (Pipes)
  float ncclLatency{}; // microseconds
  float allgatherLatency{}; // microseconds
  float speedupVsNccl{}; // Pipes / ncclAllGather
};

class AllGatherBenchmarkFixture : public meta::comms::BenchmarkTestFixture {
 protected:
  void SetUp() override {
    // Initialize bootstrap and rank variables from base class
    BenchmarkTestFixture::SetUp();
    // Use localRank for cudaSetDevice since each node has its own set of GPUs
    CUDA_CHECK_VOID(cudaSetDevice(localRank));

    // Initialize NCCL with default channel settings
    NCCL_CHECK_VOID(
        ncclCommInitRank(&ncclComm_, worldSize, getNCCLId(), globalRank));
    CUDA_CHECK_VOID(cudaStreamCreate(&stream_));
  }

  void TearDown() override {
    NCCL_CHECK_VOID(ncclCommDestroy(ncclComm_));
    CUDA_CHECK_VOID(cudaStreamDestroy(stream_));
    BenchmarkTestFixture::TearDown();
  }

  ncclUniqueId getNCCLId() {
    ncclUniqueId id;
    if (globalRank == 0) {
      ncclResult_t res = ncclGetUniqueId(&id);
      if (res != ncclSuccess) {
        XLOGF(ERR, "ncclGetUniqueId failed: {}", ncclGetErrorString(res));
        std::abort();
      }
    }
    // Broadcast NCCL ID using bootstrap allGather
    std::vector<ncclUniqueId> allIds(worldSize);
    allIds[globalRank] = id;
    auto result =
        bootstrap
            ->allGather(
                allIds.data(), sizeof(ncclUniqueId), globalRank, worldSize)
            .get();
    if (result != 0) {
      XLOG(ERR) << "Bootstrap allGather for NCCL ID failed";
      std::abort();
    }
    id = allIds[0]; // Take rank 0's ID
    return id;
  }

  /**
   * Run NCCL AllGather benchmark using ncclAllGather API.
   * Returns bandwidth in GB/s and sets latency in microseconds.
   */
  float runNcclAllGatherBenchmark(
      const AllGatherBenchmarkConfig& config,
      float& latencyUs) {
    XLOGF(
        DBG1,
        "Rank {}: Running NCCL AllGather benchmark: {}",
        globalRank,
        config.name);

    const int nranks = worldSize;
    const std::size_t sendcount = config.sendcount;
    const std::size_t recvcount = sendcount * nranks;

    // Allocate send and recv buffers
    DeviceBuffer sendBuffer(sendcount);
    DeviceBuffer recvBuffer(recvcount);

    // Initialize send buffer
    std::vector<char> h_send(sendcount);
    for (std::size_t i = 0; i < sendcount; i++) {
      h_send[i] = static_cast<char>(globalRank * 10 + (i % 256));
    }
    CUDA_CHECK(cudaMemcpy(
        sendBuffer.get(), h_send.data(), sendcount, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(recvBuffer.get(), 0, recvcount));

    CudaEvent start, stop;
    const int nIter = 100;
    const int nIterWarmup = 5;

    // Warmup
    bootstrap->barrierAll();
    for (int i = 0; i < nIterWarmup; i++) {
      NCCL_CHECK(ncclAllGather(
          sendBuffer.get(),
          recvBuffer.get(),
          sendcount,
          ncclChar,
          ncclComm_,
          stream_));
    }

    // Benchmark
    CUDA_CHECK(cudaEventRecord(start.get(), stream_));
    for (int i = 0; i < nIter; i++) {
      NCCL_CHECK(ncclAllGather(
          sendBuffer.get(),
          recvBuffer.get(),
          sendcount,
          ncclChar,
          ncclComm_,
          stream_));
    }
    CUDA_CHECK(cudaEventRecord(stop.get(), stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    float totalTime_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime_ms, start.get(), stop.get()));
    float avgTime_ms = totalTime_ms / nIter;
    latencyUs = avgTime_ms * 1000.0f;

    // Algorithm bandwidth: total data moved / time
    // Each rank sends sendcount bytes and receives (nranks - 1) * sendcount
    // Total data moved = nranks * sendcount (output size)
    std::size_t totalDataMoved = recvcount;
    float bandwidth_GBps = (totalDataMoved / (1000.0f * 1000.0f * 1000.0f)) /
        (avgTime_ms / 1000.0f);

    bootstrap->barrierAll();

    return bandwidth_GBps;
  }

  /**
   * Run AllGather benchmark.
   * Returns bandwidth in GB/s and sets latency in microseconds.
   */
  float runAllGatherBenchmark(
      const AllGatherBenchmarkConfig& config,
      float& latencyUs) {
    XLOGF(
        DBG1,
        "Rank {}: Running AllGather benchmark: {}",
        globalRank,
        config.name);

    const int nranks = worldSize;
    const std::size_t sendcount = config.sendcount;
    const std::size_t recvcount = sendcount * nranks;

    // Allocate send and recv buffers
    DeviceBuffer sendBuffer(sendcount);
    DeviceBuffer recvBuffer(recvcount);

    // Initialize send buffer
    std::vector<char> h_send(sendcount);
    for (std::size_t i = 0; i < sendcount; i++) {
      h_send[i] = static_cast<char>(globalRank * 10 + (i % 256));
    }
    CUDA_CHECK(cudaMemcpy(
        sendBuffer.get(), h_send.data(), sendcount, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(recvBuffer.get(), 0, recvcount));

    // Setup P2P NVL transport
    MultiPeerNvlTransportConfig nvlConfig{
        .dataBufferSize = config.dataBufferSize,
        .chunkSize = config.chunkSize,
        .pipelineDepth = config.pipelineDepth,
    };

    // Create transport with bootstrap and exchange IPC handles
    MultiPeerNvlTransport transport(globalRank, nranks, bootstrap, nvlConfig);
    transport.exchange();

    // Create transport array: self for my rank, P2P for others
    P2pSelfTransportDevice selfTransport;
    std::vector<Transport> h_transports;
    h_transports.reserve(nranks);

    for (int rank = 0; rank < nranks; rank++) {
      if (rank == globalRank) {
        h_transports.emplace_back(selfTransport);
      } else {
        h_transports.emplace_back(transport.getP2pTransportDevice(rank));
      }
    }

    // Copy transports to device
    DeviceBuffer d_transports(sizeof(Transport) * nranks);
    CUDA_CHECK(cudaMemcpy(
        d_transports.get(),
        h_transports.data(),
        sizeof(Transport) * nranks,
        cudaMemcpyHostToDevice));

    // Create device span
    DeviceSpan<Transport> transports_span(
        static_cast<Transport*>(d_transports.get()), nranks);

    // Prepare kernel launch parameters
    dim3 gridDim(config.numBlocks);
    dim3 blockDim(config.numThreads);

    // Get device pointers from DeviceBuffer objects
    void* recvBuff_d = recvBuffer.get();
    void* sendBuff_d = sendBuffer.get();

    // Create timeout (default = no timeout)
    Timeout timeout;

    // Need non-const copy for kernel args
    std::size_t sendcount_arg = sendcount;
    int rank_arg = globalRank;

    void* args[] = {
        &recvBuff_d,
        &sendBuff_d,
        &sendcount_arg,
        &rank_arg,
        &transports_span,
        &timeout};

    CudaEvent start, stop;
    const int nIter = 100;
    const int nIterWarmup = 5;

    // Use pointer to cluster dimension for clustered launch
    dim3 defaultClusterDim(comms::common::kDefaultClusterSize, 1, 1);
    std::optional<dim3> clusterDimOpt = config.spreadClusterLaunch
        ? std::optional{defaultClusterDim}
        : std::nullopt;

    // Warmup
    bootstrap->barrierAll();

    for (int i = 0; i < nIterWarmup; i++) {
      CUDA_CHECK(
          comms::common::launchKernel(
              (void*)all_gather_kernel,
              gridDim,
              blockDim,
              args,
              nullptr,
              clusterDimOpt));
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    bootstrap->barrierAll();

    // Benchmark
    CUDA_CHECK(cudaEventRecord(start.get()));
    for (int i = 0; i < nIter; i++) {
      CUDA_CHECK(
          comms::common::launchKernel(
              (void*)all_gather_kernel,
              gridDim,
              blockDim,
              args,
              nullptr,
              clusterDimOpt));
    }
    CUDA_CHECK(cudaEventRecord(stop.get()));
    CUDA_CHECK(cudaDeviceSynchronize());

    float totalTime_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime_ms, start.get(), stop.get()));
    float avgTime_ms = totalTime_ms / nIter;
    latencyUs = avgTime_ms * 1000.0f;

    // Algorithm bandwidth: total data moved / time
    std::size_t totalDataMoved = recvcount;
    float bandwidth_GBps = (totalDataMoved / (1000.0f * 1000.0f * 1000.0f)) /
        (avgTime_ms / 1000.0f);

    bootstrap->barrierAll();

    return bandwidth_GBps;
  }

  void printResultsTable(const std::vector<AllGatherBenchmarkResult>& results) {
    if (globalRank != 0) {
      return; // Only rank 0 prints
    }

    std::stringstream ss;
    ss << "\n";
    ss << "================================================================================\n";
    ss << "              NCCL AllGather vs Pipes AllGather Benchmark Results\n";
    ss << "================================================================================\n";
    ss << std::left << std::setw(12) << "Test" << std::right << std::setw(10)
       << "Size" << std::right << std::setw(4) << "PD" << std::right
       << std::setw(8) << "Chunk" << std::right << std::setw(10) << "NCCL"
       << std::right << std::setw(10) << "Pipes" << std::right << std::setw(10)
       << "Speedup" << std::right << std::setw(12) << "NCCL Lat" << std::right
       << std::setw(12) << "Pipes Lat\n";
    ss << std::left << std::setw(12) << "" << std::right << std::setw(10) << ""
       << std::right << std::setw(4) << "" << std::right << std::setw(8) << ""
       << std::right << std::setw(10) << "(GB/s)" << std::right << std::setw(10)
       << "(GB/s)" << std::right << std::setw(10) << "" << std::right
       << std::setw(12) << "(us)" << std::right << std::setw(12) << "(us)\n";
    ss << "--------------------------------------------------------------------------------\n";

    auto formatBytes = [](std::size_t bytes) -> std::string {
      if (bytes < 1024) {
        return std::to_string(bytes) + "B";
      }
      if (bytes < 1024 * 1024) {
        return std::to_string(bytes / 1024) + "KB";
      }
      if (bytes < 1024 * 1024 * 1024) {
        return std::to_string(bytes / (1024 * 1024)) + "MB";
      }
      return std::to_string(bytes / (1024 * 1024 * 1024)) + "GB";
    };

    for (const auto& r : results) {
      ss << std::left << std::setw(12) << r.testName << std::right
         << std::setw(10) << formatBytes(r.sendcount) << std::right
         << std::setw(4) << r.pipelineDepth << std::right << std::setw(8)
         << formatBytes(r.chunkSize) << std::right << std::setw(10)
         << std::fixed << std::setprecision(2) << r.ncclBandwidth << std::right
         << std::setw(10) << std::fixed << std::setprecision(2)
         << r.allgatherBandwidth << std::right << std::setw(9) << std::fixed
         << std::setprecision(2) << r.speedupVsNccl << "x" << std::right
         << std::setw(12) << std::fixed << std::setprecision(1) << r.ncclLatency
         << std::right << std::setw(12) << std::fixed << std::setprecision(1)
         << r.allgatherLatency << "\n";
    }

    ss << "================================================================================\n";
    ss << "Sendcount: Message size per rank, " << worldSize << " ranks\n";
    ss << "PD = Pipeline Depth, Chunk = Chunk Size\n";
    ss << "NCCL = ncclAllGather (Ring algorithm)\n";
    ss << "Pipes = Pipes AllGather (direct P2P)\n";
    ss << "Speedup = Pipes BW / NCCL BW (>1 means Pipes is faster)\n";
    ss << "================================================================================\n";
    ss << "\n";

    XLOG(INFO) << ss.str();
  }

  ncclComm_t ncclComm_{};
  cudaStream_t stream_{};
};

TEST_F(AllGatherBenchmarkFixture, OptimalConfigs) {
  if (globalRank == 0) {
    XLOG(INFO)
        << "\n=== OPTIMAL AllGather vs NCCL Comparison (All Message Sizes) ===\n";
  }

  std::vector<AllGatherBenchmarkConfig> configs;
  std::size_t kDataBufferSize = 8 * 1024 * 1024; // 8MB

  // === Configuration Notes ===
  // Block counts are matched to NCCL channel counts for fair comparison.
  // Chunk sizes optimized per message size range:
  // - Small (256KB-1MB): 32KB chunks for lower overhead
  // - Medium (2MB-16MB): 64KB chunks for balanced pipelining
  // - Large (32MB-128MB): 128KB-512KB chunks
  // - Very large (256MB-1GB): 256KB-1MB chunks for bandwidth efficiency
  //
  // === Adaptive Pipeline Depth ===
  // Based on performance analysis (see
  // ~/docs/Pipes_AllGather_Performance_Analysis.md):
  // - Small messages (< 1MB): pipelineDepth=4 (latency-bound, benefits from
  // overlapping)
  // - Large messages (≥ 1MB): pipelineDepth=2 (bandwidth-bound, less overhead
  // is better) Deeper pipelining adds memory footprint and ChunkState
  // synchronization overhead that hurts bandwidth-bound transfers.

  // 256KB with 8 blocks, chunkSize = 32KB
  // Small message: use pipelineDepth=4 for latency hiding
  configs.push_back({
      .sendcount = 256 * 1024,
      .numBlocks = 8,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 64 * 1024,
      .dataBufferSize = kDataBufferSize,
      .spreadClusterLaunch = true,
      .name = "256K_8B",
  });

  // 512KB with 16 blocks, chunkSize = 32KB
  // Small message: use pipelineDepth=4 for latency hiding
  configs.push_back({
      .sendcount = 512 * 1024,
      .numBlocks = 16,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 64 * 1024,
      .dataBufferSize = kDataBufferSize,
      .spreadClusterLaunch = true,
      .name = "512K_16B",
  });

  // 1MB with 16 blocks, chunkSize = 32KB
  // Transition point: use pipelineDepth=2 (bandwidth starts to dominate)
  configs.push_back({
      .sendcount = 1 * 1024 * 1024,
      .numBlocks = 16,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 64 * 1024,
      .dataBufferSize = kDataBufferSize,
      .spreadClusterLaunch = true,
      .name = "1M_16B",
  });

  // 2MB with 16 blocks, chunkSize = 64KB
  configs.push_back({
      .sendcount = 2 * 1024 * 1024,
      .numBlocks = 16,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 128 * 1024,
      .dataBufferSize = kDataBufferSize,
      .spreadClusterLaunch = true,
      .name = "2M_16B",
  });

  // 4MB with 16 blocks, chunkSize = 64KB
  configs.push_back({
      .sendcount = 4 * 1024 * 1024,
      .numBlocks = 16,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 128 * 1024,
      .dataBufferSize = kDataBufferSize,
      .spreadClusterLaunch = true,
      .name = "4M_16B",
  });

  // 8MB with 16 blocks, chunkSize = 64KB
  configs.push_back({
      .sendcount = 8 * 1024 * 1024,
      .numBlocks = 16,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 128 * 1024,
      .dataBufferSize = kDataBufferSize,
      .spreadClusterLaunch = true,
      .name = "8M_16B",
  });

  // 16MB with 16 blocks, chunkSize = 64KB
  configs.push_back({
      .sendcount = 16 * 1024 * 1024,
      .numBlocks = 16,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 128 * 1024,
      .dataBufferSize = kDataBufferSize,
      .spreadClusterLaunch = true,
      .name = "16M_16B",
  });

  // 32MB with 16 blocks, chunkSize = 128KB
  configs.push_back({
      .sendcount = 32 * 1024 * 1024,
      .numBlocks = 16,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 128 * 1024,
      .dataBufferSize = kDataBufferSize,
      .spreadClusterLaunch = true,
      .name = "32M_16B",
  });

  // 64MB with 16 blocks, chunkSize = 256KB
  configs.push_back({
      .sendcount = 64 * 1024 * 1024,
      .numBlocks = 16,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 256 * 1024,
      .dataBufferSize = kDataBufferSize,
      .spreadClusterLaunch = true,
      .name = "64M_16B",
  });

  // 128MB with 16 blocks, chunkSize = 512KB
  configs.push_back({
      .sendcount = 128 * 1024 * 1024,
      .numBlocks = 16,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 256 * 1024,
      .dataBufferSize = kDataBufferSize,
      .spreadClusterLaunch = true,
      .name = "128M_16B",
  });

  // 256MB with 32 blocks, chunkSize = 256KB
  configs.push_back({
      .sendcount = 256 * 1024 * 1024,
      .numBlocks = 32,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 256 * 1024,
      .dataBufferSize = kDataBufferSize,
      .spreadClusterLaunch = true,
      .name = "256M_32B",
  });

  // 512MB with 32 blocks, chunkSize = 512KB
  configs.push_back({
      .sendcount = 512 * 1024 * 1024,
      .numBlocks = 32,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 256 * 1024,
      .dataBufferSize = kDataBufferSize,
      .spreadClusterLaunch = true,
      .name = "512M_32B",
  });

  // 1GB with 32 blocks, chunkSize = 1MB
  configs.push_back({
      .sendcount = 1024 * 1024 * 1024,
      .numBlocks = 32,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 256 * 1024,
      .dataBufferSize = kDataBufferSize,
      .spreadClusterLaunch = true,
      .name = "1G_32B",
  });

  std::vector<AllGatherBenchmarkResult> results;

  for (const auto& config : configs) {
    float ncclLatencyUs = 0.0f;
    float ncclBandwidth = runNcclAllGatherBenchmark(config, ncclLatencyUs);

    float allgatherLatencyUs = 0.0f;
    float allgatherBandwidth =
        runAllGatherBenchmark(config, allgatherLatencyUs);

    if (globalRank == 0) {
      AllGatherBenchmarkResult result;
      result.testName = config.name;
      result.sendcount = config.sendcount;
      result.totalBytes = config.sendcount * worldSize;
      result.pipelineDepth = config.pipelineDepth;
      result.chunkSize = config.chunkSize;
      result.ncclBandwidth = ncclBandwidth;
      result.allgatherBandwidth = allgatherBandwidth;
      result.ncclLatency = ncclLatencyUs;
      result.allgatherLatency = allgatherLatencyUs;
      result.speedupVsNccl = allgatherBandwidth / ncclBandwidth;
      results.push_back(result);
    }

    bootstrap->barrierAll();
  }

  printResultsTable(results);
}

} // namespace

} // namespace comms::pipes::benchmark

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);

  // Set up distributed environment
  ::testing::AddGlobalTestEnvironment(new meta::comms::BenchmarkEnvironment());

  return RUN_ALL_TESTS();
}
