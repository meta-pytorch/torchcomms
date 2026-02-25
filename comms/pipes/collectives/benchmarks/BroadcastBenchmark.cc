// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gflags/gflags.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>
#include <nccl.h>

#include "comms/common/CudaWrap.h"
#include "comms/pipes/MultiPeerNvlTransport.h"
#include "comms/pipes/benchmarks/BenchmarkMacros.h"
#include "comms/pipes/collectives/benchmarks/CollectiveBenchmark.cuh"
#include "comms/pipes/collectives/benchmarks/CollectiveBenchmarkUtils.h"
#include "comms/utils/CudaRAII.h"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <vector>

DEFINE_bool(
    verify_correctness,
    false,
    "Enable data correctness verification after broadcast operations");

using meta::comms::CudaEvent;
using meta::comms::DeviceBuffer;

namespace comms::pipes::benchmark {

namespace {

/**
 * Test configuration for Broadcast benchmark.
 */
struct BroadcastBenchmarkConfig {
  std::size_t nbytes; // Total message size (single buffer)
  int numBlocks;
  int numThreads;
  std::size_t pipelineDepth = 2;
  std::size_t chunkSize = 128 * 1024; // 128KB default
  std::size_t dataBufferSize = 8 * 1024 * 1024; // 8MB default
  bool spreadClusterLaunch = false;
  int rootRank = 0; // Broadcast root rank
  std::string name;
};

/**
 * Result struct for collecting benchmark data.
 */
struct BroadcastBenchmarkResult {
  std::string testName;
  std::size_t nbytes{}; // Total message size
  std::size_t pipelineDepth{};
  std::size_t chunkSize{};
  float ncclBandwidth{}; // GB/s (ncclBroadcast)
  float pipesBandwidth{}; // GB/s (Pipes Broadcast)
  float ncclLatency{}; // microseconds
  float pipesLatency{}; // microseconds
  float speedupVsNccl{}; // Pipes / NCCL (>1 means Pipes is faster)
  bool ncclVerified{}; // Data correctness verified for NCCL
  bool pipesVerified{}; // Data correctness verified for Pipes
};

class BroadcastBenchmarkFixture : public NcclBenchmarkFixture {
 protected:
  /**
   * Generate expected data pattern for broadcast verification.
   * Uses i % 256 pattern (same as root rank initialization).
   */
  std::vector<char> generate_expected_data(std::size_t size) {
    std::vector<char> data(size);
    for (std::size_t i = 0; i < size; i++) {
      data[i] = static_cast<char>(i % 256);
    }
    return data;
  }

  /**
   * Verify broadcast buffer matches expected data pattern.
   * Returns true if verification passes, false otherwise.
   */
  bool verify_broadcast_data(
      void* buffer_d,
      std::size_t size,
      const std::string& test_name,
      const std::string& impl) {
    if (size == 0) {
      return true; // Nothing to verify
    }
    std::vector<char> h_result(size);
    CUDA_CHECK_BOOL(
        cudaMemcpy(h_result.data(), buffer_d, size, cudaMemcpyDeviceToHost));

    auto expected = generate_expected_data(size);
    if (h_result.size() != expected.size()) {
      XLOGF(
          ERR,
          "Rank {}: {} {} verification FAILED: expected size {} != actual size {}",
          globalRank,
          impl,
          test_name,
          expected.size(),
          h_result.size());
      return false;
    }

    std::size_t mismatch_count = 0;
    std::size_t first_mismatch = 0;
    char expected_val = 0;
    char actual_val = 0;

    for (std::size_t i = 0; i < size; i++) {
      if (h_result[i] != expected[i]) {
        if (mismatch_count == 0) {
          first_mismatch = i;
          expected_val = expected[i];
          actual_val = h_result[i];
        }
        mismatch_count++;
      }
    }

    if (mismatch_count > 0) {
      XLOGF(
          ERR,
          "Rank {}: {} {} verification FAILED: {} mismatches out of {} bytes. "
          "First mismatch at byte {}: expected {}, got {}",
          globalRank,
          impl,
          test_name,
          mismatch_count,
          size,
          first_mismatch,
          static_cast<int>(expected_val),
          static_cast<int>(actual_val));
      return false;
    }

    XLOGF(
        DBG1,
        "Rank {}: {} {} verification PASSED ({} bytes)",
        globalRank,
        impl,
        test_name,
        size);
    return true;
  }

  /**
   * Run NCCL Broadcast benchmark using ncclBroadcast API.
   * Returns bandwidth in GB/s and sets latency in microseconds.
   * If FLAGS_verify_correctness is set, verifies data and sets verified.
   */
  float runNcclBroadcastBenchmark(
      const BroadcastBenchmarkConfig& config,
      float& latencyUs,
      bool& verified) {
    verified = false;
    XLOGF(
        DBG1,
        "Rank {}: Running NCCL Broadcast benchmark: {}",
        globalRank,
        config.name);

    const std::size_t nbytes = config.nbytes;

    // Single buffer for in-place broadcast
    DeviceBuffer buffer(nbytes);

    // Root initializes buffer with pattern data
    if (globalRank == config.rootRank) {
      std::vector<char> h_data(nbytes);
      for (std::size_t i = 0; i < nbytes; i++) {
        h_data[i] = static_cast<char>(i % 256);
      }
      CUDA_CHECK(cudaMemcpy(
          buffer.get(), h_data.data(), nbytes, cudaMemcpyHostToDevice));
    } else {
      CUDA_CHECK(cudaMemset(buffer.get(), 0, nbytes));
    }

    CudaEvent start, stop;
    const int nIter = 100;
    const int nIterWarmup = 5;

    // Warmup
    bootstrap->barrierAll();
    for (int i = 0; i < nIterWarmup; i++) {
      NCCL_CHECK(ncclBroadcast(
          buffer.get(),
          buffer.get(),
          nbytes,
          ncclChar,
          config.rootRank,
          ncclComm_,
          stream_));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    bootstrap->barrierAll();

    // Benchmark
    CUDA_CHECK(cudaEventRecord(start.get(), stream_));
    for (int i = 0; i < nIter; i++) {
      NCCL_CHECK(ncclBroadcast(
          buffer.get(),
          buffer.get(),
          nbytes,
          ncclChar,
          config.rootRank,
          ncclComm_,
          stream_));
    }
    CUDA_CHECK(cudaEventRecord(stop.get(), stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    float totalTime_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime_ms, start.get(), stop.get()));
    float avgTime_ms = totalTime_ms / nIter;
    latencyUs = avgTime_ms * 1000.0f;

    // Bandwidth: total data broadcast / time
    float bandwidth_GBps =
        (nbytes / (1000.0f * 1000.0f * 1000.0f)) / (avgTime_ms / 1000.0f);

    bootstrap->barrierAll();

    if (FLAGS_verify_correctness) {
      verified =
          verify_broadcast_data(buffer.get(), nbytes, config.name, "NCCL");
    }

    return bandwidth_GBps;
  }

  /**
   * Run Pipes Broadcast benchmark.
   * Returns bandwidth in GB/s and sets latency in microseconds.
   * If FLAGS_verify_correctness is set, verifies data and sets verified.
   */
  float runPipesBroadcastBenchmark(
      const BroadcastBenchmarkConfig& config,
      float& latencyUs,
      bool& verified) {
    verified = false;
    XLOGF(
        DBG1,
        "Rank {}: Running Pipes Broadcast benchmark: {}",
        globalRank,
        config.name);

    const int nranks = worldSize;
    const std::size_t nbytes = config.nbytes;

    // Single buffer for in-place broadcast
    DeviceBuffer buffer(nbytes);

    // Root initializes buffer with pattern data
    if (globalRank == config.rootRank) {
      std::vector<char> h_data(nbytes);
      for (std::size_t i = 0; i < nbytes; i++) {
        h_data[i] = static_cast<char>(i % 256);
      }
      CUDA_CHECK(cudaMemcpy(
          buffer.get(), h_data.data(), nbytes, cudaMemcpyHostToDevice));
    } else {
      CUDA_CHECK(cudaMemset(buffer.get(), 0, nbytes));
    }

    // Setup P2P NVL transport
    MultiPeerNvlTransportConfig nvlConfig{
        .dataBufferSize = config.dataBufferSize,
        .chunkSize = config.chunkSize,
        .pipelineDepth = config.pipelineDepth,
    };

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

    // Get device pointer from DeviceBuffer
    void* buff_d = buffer.get();

    // Need non-const copies for kernel args
    int rank_arg = globalRank;
    int root_rank_arg = config.rootRank;
    std::size_t nbytes_arg = nbytes;

    void* args[] = {
        &buff_d, &rank_arg, &root_rank_arg, &transports_span, &nbytes_arg};

    CudaEvent start, stop;
    const int nIter = 100;
    const int nIterWarmup = 5;

    // Use pointer to cluster dimension for clustered launch
    dim3 defaultClusterDim(comms::common::kDefaultClusterSize, 1, 1);
    std::optional<dim3> clusterDimOpt = config.spreadClusterLaunch
        ? std::optional{defaultClusterDim}
        : std::nullopt;

    void* kernelFunc = (void*)broadcast_kernel;

    // Warmup
    bootstrap->barrierAll();

    for (int i = 0; i < nIterWarmup; i++) {
      CUDA_CHECK(
          comms::common::launchKernel(
              kernelFunc, gridDim, blockDim, args, nullptr, clusterDimOpt));
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    bootstrap->barrierAll();

    // Benchmark
    CUDA_CHECK(cudaEventRecord(start.get()));
    for (int i = 0; i < nIter; i++) {
      CUDA_CHECK(
          comms::common::launchKernel(
              kernelFunc, gridDim, blockDim, args, nullptr, clusterDimOpt));
    }
    CUDA_CHECK(cudaEventRecord(stop.get()));
    CUDA_CHECK(cudaDeviceSynchronize());

    float totalTime_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime_ms, start.get(), stop.get()));
    float avgTime_ms = totalTime_ms / nIter;
    latencyUs = avgTime_ms * 1000.0f;

    // Bandwidth: total data broadcast / time
    float bandwidth_GBps =
        (nbytes / (1000.0f * 1000.0f * 1000.0f)) / (avgTime_ms / 1000.0f);

    bootstrap->barrierAll();

    if (FLAGS_verify_correctness) {
      verified =
          verify_broadcast_data(buffer.get(), nbytes, config.name, "Pipes");
    }

    return bandwidth_GBps;
  }

  void printResultsTable(const std::vector<BroadcastBenchmarkResult>& results) {
    if (globalRank != 0) {
      return; // Only rank 0 prints
    }

    std::stringstream ss;
    ss << "\n";
    ss << "========================================================================================\n";
    ss << "     NCCL Broadcast vs Pipes Broadcast Benchmark Results\n";
    ss << "========================================================================================\n";
    ss << std::left << std::setw(12) << "Test" << std::right << std::setw(10)
       << "MsgSize" << std::right << std::setw(4) << "PD" << std::right
       << std::setw(8) << "Chunk" << std::right << std::setw(10) << "NCCL"
       << std::right << std::setw(10) << "Pipes" << std::right << std::setw(10)
       << "Speedup" << std::right << std::setw(12) << "NCCL Lat" << std::right
       << std::setw(12) << "Pipes Lat\n";
    ss << std::left << std::setw(12) << "" << std::right << std::setw(10) << ""
       << std::right << std::setw(4) << "" << std::right << std::setw(8) << ""
       << std::right << std::setw(10) << "(GB/s)" << std::right << std::setw(10)
       << "(GB/s)" << std::right << std::setw(10) << "" << std::right
       << std::setw(12) << "(us)" << std::right << std::setw(12) << "(us)\n";
    ss << "----------------------------------------------------------------------------------------\n";

    for (const auto& r : results) {
      ss << std::left << std::setw(12) << r.testName << std::right
         << std::setw(10) << format_bytes(r.nbytes) << std::right
         << std::setw(4) << r.pipelineDepth << std::right << std::setw(8)
         << format_bytes(r.chunkSize) << std::right << std::setw(10)
         << std::fixed << std::setprecision(2) << r.ncclBandwidth << std::right
         << std::setw(10) << std::fixed << std::setprecision(2)
         << r.pipesBandwidth << std::right << std::setw(9) << std::fixed
         << std::setprecision(2) << r.speedupVsNccl << "x" << std::right
         << std::setw(12) << std::fixed << std::setprecision(1) << r.ncclLatency
         << std::right << std::setw(12) << std::fixed << std::setprecision(1)
         << r.pipesLatency << "\n";
    }

    ss << "========================================================================================\n";
    ss << "Size: Total broadcast message size, " << worldSize << " ranks\n";
    ss << "PD = Pipeline Depth, Chunk = Chunk Size\n";
    ss << "NCCL = ncclBroadcast\n";
    ss << "Pipes = Pipes Broadcast\n";
    ss << "Speedup = Pipes BW / NCCL BW (>1 means Pipes is faster)\n";

    if (FLAGS_verify_correctness) {
      bool allPassed =
          std::all_of(results.begin(), results.end(), [](const auto& r) {
            return r.ncclVerified && r.pipesVerified;
          });
      ss << "Verification: " << (allPassed ? "ALL PASSED" : "SOME FAILED")
         << "\n";
    }

    ss << "========================================================================================\n";
    ss << "\n";

    XLOG(INFO) << ss.str();
  }
};

TEST_F(BroadcastBenchmarkFixture, OptimalConfigs) {
  if (globalRank == 0) {
    XLOG(INFO)
        << "\n=== OPTIMAL Broadcast vs NCCL Comparison (Large Message Sizes) ===\n";
    if (FLAGS_verify_correctness) {
      XLOG(INFO) << "Data correctness verification ENABLED\n";
    }
  }

  std::vector<BroadcastBenchmarkConfig> configs;
  std::size_t kDataBufferSize = 8 * 1024 * 1024; // 8MB

  // === Configuration Notes ===
  // Pipes Broadcast targets large messages where the pipelined
  // dual-destination fused copy provides bandwidth benefits.
  // Starting at 4MB since smaller messages are better served by
  // FlatTree or Ring topologies.

  // 4MB with 16 blocks, chunkSize = 16KB (256 warps, 256 chunks/step = 100%)
  // 16KB chunks fix warp utilization vs 32KB (100% vs 50%), improving
  // performance from ~0.70x to ~0.89x NCCL.
  configs.push_back({
      .nbytes = 4 * 1024 * 1024,
      .numBlocks = 16,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 16 * 1024,
      .dataBufferSize = 4 * 1024 * 1024,
      .spreadClusterLaunch = true,
      .rootRank = 0,
      .name = "4M_16B",
  });

  // 8MB with 16 blocks, chunkSize = 32KB (256 warps, 256 chunks/step = 100%)
  configs.push_back({
      .nbytes = 8 * 1024 * 1024,
      .numBlocks = 16,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 32 * 1024,
      .dataBufferSize = kDataBufferSize,
      .spreadClusterLaunch = true,
      .rootRank = 0,
      .name = "8M_16B",
  });

  // 16MB with 16 blocks, chunkSize = 32KB (256 warps, 256 chunks/step = 100%)
  configs.push_back({
      .nbytes = 16 * 1024 * 1024,
      .numBlocks = 16,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 32 * 1024,
      .dataBufferSize = kDataBufferSize,
      .spreadClusterLaunch = true,
      .rootRank = 0,
      .name = "16M_16B",
  });

  // 32MB with 16 blocks, chunkSize = 32KB (256 warps, 256 chunks/step = 100%)
  configs.push_back({
      .nbytes = 32 * 1024 * 1024,
      .numBlocks = 16,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 32 * 1024,
      .dataBufferSize = kDataBufferSize,
      .spreadClusterLaunch = true,
      .rootRank = 0,
      .name = "32M_16B",
  });

  // 64MB with 16 blocks, chunkSize = 32KB (256 warps, 256 chunks/step = 100%)
  configs.push_back({
      .nbytes = 64 * 1024 * 1024,
      .numBlocks = 16,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 32 * 1024,
      .dataBufferSize = kDataBufferSize,
      .spreadClusterLaunch = true,
      .rootRank = 0,
      .name = "64M_16B",
  });

  // 128MB with 16 blocks, chunkSize = 32KB (256 warps, 256 chunks/step = 100%)
  configs.push_back({
      .nbytes = 128 * 1024 * 1024,
      .numBlocks = 16,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 32 * 1024,
      .dataBufferSize = kDataBufferSize,
      .spreadClusterLaunch = true,
      .rootRank = 0,
      .name = "128M_16B",
  });

  // 256MB with 32 blocks, chunkSize = 16KB (512 warps, 512 chunks/step = 100%)
  configs.push_back({
      .nbytes = 256 * 1024 * 1024,
      .numBlocks = 32,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 16 * 1024,
      .dataBufferSize = kDataBufferSize,
      .spreadClusterLaunch = true,
      .rootRank = 0,
      .name = "256M_32B",
  });

  // 512MB with 32 blocks, chunkSize = 16KB (512 warps, 512 chunks/step = 100%)
  configs.push_back({
      .nbytes = 512 * 1024 * 1024,
      .numBlocks = 32,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 16 * 1024,
      .dataBufferSize = kDataBufferSize,
      .spreadClusterLaunch = true,
      .rootRank = 0,
      .name = "512M_32B",
  });

  // 1GB with 32 blocks, chunkSize = 16KB (512 warps, 512 chunks/step = 100%)
  configs.push_back({
      .nbytes = 1024 * 1024 * 1024,
      .numBlocks = 32,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 16 * 1024,
      .dataBufferSize = kDataBufferSize,
      .spreadClusterLaunch = true,
      .rootRank = 0,
      .name = "1G_32B",
  });

  // === 64-block with PD=2 (comparison baseline) ===

  // 64MB, 64 blocks, 16KB, 16MB buffer
  configs.push_back({
      .nbytes = 64 * 1024 * 1024,
      .numBlocks = 64,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 16 * 1024,
      .dataBufferSize = 16 * 1024 * 1024,
      .spreadClusterLaunch = true,
      .rootRank = 0,
      .name = "64M_64B",
  });

  // 128MB, 64 blocks, 16KB, 16MB buffer
  configs.push_back({
      .nbytes = 128 * 1024 * 1024,
      .numBlocks = 64,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 16 * 1024,
      .dataBufferSize = 16 * 1024 * 1024,
      .spreadClusterLaunch = true,
      .rootRank = 0,
      .name = "128M_64B",
  });

  // 256MB, 64 blocks, 16KB, 16MB buffer (1024 warps, 1024 chunks = 1/warp)
  configs.push_back({
      .nbytes = 256 * 1024 * 1024,
      .numBlocks = 64,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 16 * 1024,
      .dataBufferSize = 16 * 1024 * 1024,
      .spreadClusterLaunch = true,
      .rootRank = 0,
      .name = "256M_64B",
  });

  // 512MB, 64 blocks, 16KB, 16MB buffer
  configs.push_back({
      .nbytes = 512 * 1024 * 1024,
      .numBlocks = 64,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 16 * 1024,
      .dataBufferSize = 16 * 1024 * 1024,
      .spreadClusterLaunch = true,
      .rootRank = 0,
      .name = "512M_64B",
  });

  // 1GB, 64 blocks, 16KB, 16MB buffer (1024 warps, 1024 chunks = 1/warp)
  configs.push_back({
      .nbytes = 1024 * 1024 * 1024,
      .numBlocks = 64,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 16 * 1024,
      .dataBufferSize = 16 * 1024 * 1024,
      .spreadClusterLaunch = true,
      .rootRank = 0,
      .name = "1G_64B",
  });

  // === 64-block with pipeline depth 4, 32KB chunks (recommended config) ===
  // 512 chunks/step, 1024 warps (2 warps per 32KB chunk).
  // Memory per rank: 7 × 4 × 16MB = 448MB

  // 64MB, 64 blocks, PD=4
  configs.push_back({
      .nbytes = 64 * 1024 * 1024,
      .numBlocks = 64,
      .numThreads = 512,
      .pipelineDepth = 4,
      .chunkSize = 32 * 1024,
      .dataBufferSize = 16 * 1024 * 1024,
      .spreadClusterLaunch = true,
      .rootRank = 0,
      .name = "64M_64B_PD4",
  });

  // 128MB, 64 blocks, PD=4
  configs.push_back({
      .nbytes = 128 * 1024 * 1024,
      .numBlocks = 64,
      .numThreads = 512,
      .pipelineDepth = 4,
      .chunkSize = 32 * 1024,
      .dataBufferSize = 16 * 1024 * 1024,
      .spreadClusterLaunch = true,
      .rootRank = 0,
      .name = "128M_64B_PD4",
  });

  // 256MB, 64 blocks, PD=4
  configs.push_back({
      .nbytes = 256 * 1024 * 1024,
      .numBlocks = 64,
      .numThreads = 512,
      .pipelineDepth = 4,
      .chunkSize = 32 * 1024,
      .dataBufferSize = 16 * 1024 * 1024,
      .spreadClusterLaunch = true,
      .rootRank = 0,
      .name = "256M_64B_PD4",
  });

  // 512MB, 64 blocks, PD=4
  configs.push_back({
      .nbytes = 512 * 1024 * 1024,
      .numBlocks = 64,
      .numThreads = 512,
      .pipelineDepth = 4,
      .chunkSize = 32 * 1024,
      .dataBufferSize = 16 * 1024 * 1024,
      .spreadClusterLaunch = true,
      .rootRank = 0,
      .name = "512M_64B_PD4",
  });

  // 1GB, 64 blocks, PD=4
  configs.push_back({
      .nbytes = 1024 * 1024 * 1024,
      .numBlocks = 64,
      .numThreads = 512,
      .pipelineDepth = 4,
      .chunkSize = 32 * 1024,
      .dataBufferSize = 16 * 1024 * 1024,
      .spreadClusterLaunch = true,
      .rootRank = 0,
      .name = "1G_64B_PD4",
  });

  // === Intermediate chunk sizes (64KB, 128KB, 256KB) ===
  // Maps the performance cliff between 32KB and 512KB chunks.
  // With 16MB buffer: 64KB → 256 chunks, 128KB → 128 chunks, 256KB → 64 chunks.
  // Predictions based on MLP analysis:
  //   64KB:  256 active warps → should approach 32KB performance
  //   128KB: 128 active warps → ~4x of 512KB throughput (~200+ GB/s)
  //   256KB: 64 active warps  → ~2x of 512KB throughput (~100+ GB/s)

  for (auto chunkSize : {64UL * 1024, 128UL * 1024, 256UL * 1024}) {
    std::string chunkLabel = (chunkSize == 64 * 1024) ? "64K"
        : (chunkSize == 128 * 1024)                   ? "128K"
                                                      : "256K";
    for (auto nbytes :
         {64UL * 1024 * 1024, 256UL * 1024 * 1024, 1024UL * 1024 * 1024}) {
      std::string sizeLabel = (nbytes == 64 * 1024 * 1024) ? "64M"
          : (nbytes == 256 * 1024 * 1024)                  ? "256M"
                                                           : "1G";
      configs.push_back({
          .nbytes = nbytes,
          .numBlocks = 64,
          .numThreads = 512,
          .pipelineDepth = 4,
          .chunkSize = chunkSize,
          .dataBufferSize = 16 * 1024 * 1024,
          .spreadClusterLaunch = true,
          .rootRank = 0,
          .name = sizeLabel + "_64B_" + chunkLabel,
      });
    }
  }

  // === Isolated pipeline depth comparison ===
  // Tests PD=2, PD=3, PD=4 with identical 32KB chunks, 64 blocks, 16MB buffer.
  // Eliminates the chunk-size confound in the existing PD=2 (16KB) vs PD=4
  // (32KB) comparison. PD=3 uses 336MB staging (7 × 3 × 16MB) vs PD=4's
  // 448MB (7 × 4 × 16MB), so if it captures most of the gain, it offers
  // a lower memory cost.

  for (auto pd : {2UL, 3UL, 4UL}) {
    std::string pdLabel = "PD" + std::to_string(pd);
    for (auto nbytes :
         {64UL * 1024 * 1024,
          256UL * 1024 * 1024,
          512UL * 1024 * 1024,
          1024UL * 1024 * 1024}) {
      std::string sizeLabel = (nbytes == 64 * 1024 * 1024) ? "64M"
          : (nbytes == 256 * 1024 * 1024)                  ? "256M"
          : (nbytes == 512 * 1024 * 1024)                  ? "512M"
                                                           : "1G";
      configs.push_back({
          .nbytes = nbytes,
          .numBlocks = 64,
          .numThreads = 512,
          .pipelineDepth = pd,
          .chunkSize = 32 * 1024,
          .dataBufferSize = 16 * 1024 * 1024,
          .spreadClusterLaunch = true,
          .rootRank = 0,
          .name = sizeLabel + "_iso_" + pdLabel,
      });
    }
  }

  // === Small-message regime: 8KB chunks ===
  // 4MB/8KB = 512 chunks for 256 warps (2 chunks per warp).
  // Tests whether finer granularity improves small-message performance
  // beyond the 0.88x achieved with 16KB chunks (256 chunks = 1:1).

  configs.push_back({
      .nbytes = 4 * 1024 * 1024,
      .numBlocks = 16,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 8 * 1024,
      .dataBufferSize = 4 * 1024 * 1024,
      .spreadClusterLaunch = true,
      .rootRank = 0,
      .name = "4M_16B_8K",
  });

  // === 512KB chunk exploration ===
  // AllGather/AllToAllv use 64-256KB chunks; Dispatch caps at 512KB.
  // Broadcast has only been tested with 16-32KB. Larger chunks reduce
  // per-byte signaling overhead (fewer ChunkState wait/signal ops).
  // With 512KB chunks and 16MB buffer: 32 chunks/step. Even with
  // few active warps, 32 concurrent 512KB NVLink copies should
  // saturate bandwidth. Pipeline fill cost (7 hops × ~10µs/hop ≈ 70µs)
  // is negligible for large messages.

  // 512KB chunks, 16MB buffer, PD=4, 8 blocks (128 warps, 32 chunks = 25%)
  // Minimal block count — 32 chunks need only 32 warps.
  // Memory: 7 × 4 × 16MB = 448MB
  for (auto nbytes :
       {64UL * 1024 * 1024, 256UL * 1024 * 1024, 1024UL * 1024 * 1024}) {
    std::string label = (nbytes == 64 * 1024 * 1024) ? "64M"
        : (nbytes == 256 * 1024 * 1024)              ? "256M"
                                                     : "1G";
    configs.push_back({
        .nbytes = nbytes,
        .numBlocks = 8,
        .numThreads = 512,
        .pipelineDepth = 4,
        .chunkSize = 512 * 1024,
        .dataBufferSize = 16 * 1024 * 1024,
        .spreadClusterLaunch = true,
        .rootRank = 0,
        .name = label + "_8B_512K",
    });
  }

  // 512KB chunks, 16MB buffer, PD=4, 16 blocks (256 warps, 32 chunks = 12.5%)
  // Memory: 7 × 4 × 16MB = 448MB
  for (auto nbytes :
       {64UL * 1024 * 1024, 256UL * 1024 * 1024, 1024UL * 1024 * 1024}) {
    std::string label = (nbytes == 64 * 1024 * 1024) ? "64M"
        : (nbytes == 256 * 1024 * 1024)              ? "256M"
                                                     : "1G";
    configs.push_back({
        .nbytes = nbytes,
        .numBlocks = 16,
        .numThreads = 512,
        .pipelineDepth = 4,
        .chunkSize = 512 * 1024,
        .dataBufferSize = 16 * 1024 * 1024,
        .spreadClusterLaunch = true,
        .rootRank = 0,
        .name = label + "_16B_512K",
    });
  }

  // 512KB chunks, 32MB buffer, PD=4, 16 blocks (256 warps, 64 chunks = 25%)
  // Larger buffer gives more chunks per step.
  // Memory: 7 × 4 × 32MB = 896MB
  for (auto nbytes :
       {64UL * 1024 * 1024, 256UL * 1024 * 1024, 1024UL * 1024 * 1024}) {
    std::string label = (nbytes == 64 * 1024 * 1024) ? "64M"
        : (nbytes == 256 * 1024 * 1024)              ? "256M"
                                                     : "1G";
    configs.push_back({
        .nbytes = nbytes,
        .numBlocks = 16,
        .numThreads = 512,
        .pipelineDepth = 4,
        .chunkSize = 512 * 1024,
        .dataBufferSize = 32 * 1024 * 1024,
        .spreadClusterLaunch = true,
        .rootRank = 0,
        .name = label + "_16B_512K_32M",
    });
  }

  std::vector<BroadcastBenchmarkResult> results;

  for (const auto& config : configs) {
    float ncclLatencyUs = 0.0f;
    bool ncclVerified = false;
    float ncclBandwidth =
        runNcclBroadcastBenchmark(config, ncclLatencyUs, ncclVerified);

    float pipesLatencyUs = 0.0f;
    bool pipesVerified = false;
    float pipesBandwidth =
        runPipesBroadcastBenchmark(config, pipesLatencyUs, pipesVerified);

    if (globalRank == 0) {
      BroadcastBenchmarkResult result;
      result.testName = config.name;
      result.nbytes = config.nbytes;
      result.pipelineDepth = config.pipelineDepth;
      result.chunkSize = config.chunkSize;
      result.ncclBandwidth = ncclBandwidth;
      result.pipesBandwidth = pipesBandwidth;
      result.ncclLatency = ncclLatencyUs;
      result.pipesLatency = pipesLatencyUs;
      result.speedupVsNccl = pipesBandwidth / ncclBandwidth;
      result.ncclVerified = ncclVerified;
      result.pipesVerified = pipesVerified;
      results.push_back(result);

      if (FLAGS_verify_correctness) {
        EXPECT_TRUE(ncclVerified)
            << "NCCL verification failed for " << config.name;
        EXPECT_TRUE(pipesVerified)
            << "Pipes verification failed for " << config.name;
      }
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
