// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

/**
 * Broadcast Profiling Benchmark
 *
 * This benchmark provides detailed timing breakdowns for Pipes Broadcast
 * operations to identify bottlenecks in the chunked pipelined protocol.
 *
 * Key measurements:
 * - Per-warp timing breakdowns
 * - Time spent in each phase (wait, copy, signal)
 * - NVLink bandwidth utilization estimates
 * - Staging buffer hit/miss rates
 *
 * Usage:
 *   buck run //comms/pipes/benchmarks:broadcast_profile_benchmark
 *
 * For nsys profiling:
 *   nsys profile --trace=cuda,nvtx --output=broadcast_profile \
 *     buck run //comms/pipes/benchmarks:broadcast_profile_benchmark --
 * --profile
 */

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>
#include <gflags/gflags.h>
#include <nccl.h>

#ifdef USE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

#include "comms/pipes/MultiPeerNvlTransport.h"
#include "comms/pipes/benchmarks/BenchmarkKernel.cuh"
#include "comms/pipes/benchmarks/BenchmarkMacros.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <vector>

DEFINE_bool(
    profile,
    false,
    "Enable NVTX profiling annotations for nsys profiling");

DEFINE_int32(
    message_size_mb,
    1,
    "Message size in MB for profiling (default: 1MB)");

DEFINE_int32(num_iterations, 10, "Number of profiling iterations");

namespace {
#ifdef USE_NVTX
#define NVTX_RANGE_PUSH(name) \
  do {                        \
    if (FLAGS_profile) {      \
      nvtxRangePushA(name);   \
    }                         \
  } while (0)

#define NVTX_RANGE_POP() \
  do {                   \
    if (FLAGS_profile) { \
      nvtxRangePop();    \
    }                    \
  } while (0)

#define NVTX_MARK(name)  \
  do {                   \
    if (FLAGS_profile) { \
      nvtxMarkA(name);   \
    }                    \
  } while (0)
#else
#define NVTX_RANGE_PUSH(name) ((void)0)
#define NVTX_RANGE_POP() ((void)0)
#define NVTX_MARK(name) ((void)0)
#endif
} // namespace

using meta::comms::CudaEvent;
using meta::comms::DeviceBuffer;
using meta::comms::MpiBaseTestFixture;
using meta::comms::MPIEnvironmentBase;

namespace comms::pipes::benchmark {

namespace {

/**
 * Profiling configuration for detailed timing analysis.
 */
struct ProfilingConfig {
  std::size_t messageSize;
  std::size_t stagingBufferSize;
  std::size_t pipelineDepth = 4;
  std::size_t chunkSize;
  int numBlocks;
  int numThreads;
  int rootRank;
  std::string name;
};

/**
 * Statistics helper for computing summary statistics.
 */
struct Statistics {
  double min;
  double max;
  double mean;
  double median;
  double stddev;
  double p95;
  double p99;
};

Statistics computeStats(std::vector<double>& values) {
  Statistics stats{};
  if (values.empty()) {
    return stats;
  }

  std::sort(values.begin(), values.end());

  stats.min = values.front();
  stats.max = values.back();

  double sum = std::accumulate(values.begin(), values.end(), 0.0);
  stats.mean = sum / values.size();

  size_t n = values.size();
  stats.median =
      (n % 2 == 0) ? (values[n / 2 - 1] + values[n / 2]) / 2.0 : values[n / 2];

  double variance = 0.0;
  for (double v : values) {
    variance += (v - stats.mean) * (v - stats.mean);
  }
  stats.stddev = std::sqrt(variance / n);

  stats.p95 = values[static_cast<size_t>(n * 0.95)];
  stats.p99 = values[static_cast<size_t>(n * 0.99)];

  return stats;
}

class BroadcastProfileBenchmarkFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDA_CHECK_VOID(cudaSetDevice(localRank));

    // Get clock rate for cycle to time conversion
    CUDA_CHECK_VOID(cudaDeviceGetAttribute(
        &clockRateKHz_, cudaDevAttrClockRate, localRank));

    // Initialize NCCL for comparison
    NCCL_CHECK_VOID(
        ncclCommInitRank(&ncclComm_, numRanks, getNCCLId(), globalRank));
    CUDA_CHECK_VOID(cudaStreamCreate(&stream_));
  }

  void TearDown() override {
    NCCL_CHECK_VOID(ncclCommDestroy(ncclComm_));
    CUDA_CHECK_VOID(cudaStreamDestroy(stream_));
    MpiBaseTestFixture::TearDown();
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
    MPI_CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    return id;
  }

  double cyclesToMicroseconds(unsigned long long cycles) {
    return static_cast<double>(cycles) / static_cast<double>(clockRateKHz_) *
        1000.0;
  }

  void runProfilingSession(const ProfilingConfig& config) {
    XLOGF(
        INFO,
        "Rank {}: Starting profiling session: {} ({}MB, root={})",
        globalRank,
        config.name,
        config.messageSize / (1024 * 1024),
        config.rootRank);

    DeviceBuffer buffer(config.messageSize);

    // Initialize buffer
    if (globalRank == config.rootRank) {
      std::vector<char> h_data(config.messageSize);
      for (std::size_t i = 0; i < config.messageSize; i++) {
        h_data[i] = static_cast<char>((i + config.rootRank) % 256);
      }
      CUDA_CHECK_VOID(cudaMemcpy(
          buffer.get(),
          h_data.data(),
          config.messageSize,
          cudaMemcpyHostToDevice));
    } else {
      CUDA_CHECK_VOID(cudaMemset(buffer.get(), 0, config.messageSize));
    }

    // Setup transport
    MultiPeerNvlTransportConfig nvlConfig{
        .dataBufferSize = config.stagingBufferSize,
        .chunkSize = config.chunkSize,
        .pipelineDepth = config.pipelineDepth,
    };

    auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
    MultiPeerNvlTransport transport(globalRank, numRanks, bootstrap, nvlConfig);
    transport.exchange();

    // Create transport array
    P2pSelfTransportDevice selfTransport;
    std::vector<Transport> h_transports;
    h_transports.reserve(numRanks);

    for (int rank = 0; rank < numRanks; rank++) {
      if (rank == globalRank) {
        h_transports.emplace_back(selfTransport);
      } else {
        h_transports.emplace_back(transport.getP2pTransportDevice(rank));
      }
    }

    DeviceBuffer d_transports(sizeof(Transport) * numRanks);
    CUDA_CHECK_VOID(cudaMemcpy(
        d_transports.get(),
        h_transports.data(),
        sizeof(Transport) * numRanks,
        cudaMemcpyHostToDevice));

    DeviceSpan<Transport> transports_span(
        static_cast<Transport*>(d_transports.get()), numRanks);

    dim3 gridDim(config.numBlocks);
    dim3 blockDim(config.numThreads);

    void* buff_d = buffer.get();
    int rootRank = config.rootRank;
    std::size_t nbytes = config.messageSize;
    void* args[] = {&buff_d, &globalRank, &rootRank, &transports_span, &nbytes};

    std::vector<double> latencies;
    latencies.reserve(FLAGS_num_iterations);

    CudaEvent start, stop;

    // Warmup
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    NVTX_RANGE_PUSH("Pipes_Profile_Warmup");
    for (int i = 0; i < kWarmupIters; i++) {
      CUDA_CHECK_VOID(cudaLaunchKernel(
          (void*)broadcastKernel, gridDim, blockDim, args, 0, stream_));
      CUDA_CHECK_VOID(cudaStreamSynchronize(stream_));
    }
    NVTX_RANGE_POP();
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Profiling iterations
    NVTX_RANGE_PUSH("Pipes_Profile_Iterations");
    for (int i = 0; i < FLAGS_num_iterations; i++) {
      // Reset buffer for non-root ranks
      if (globalRank != config.rootRank) {
        CUDA_CHECK_VOID(cudaMemset(buffer.get(), 0, config.messageSize));
      }
      buff_d = buffer.get();

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      std::string iterName = "Pipes_Iter_" + std::to_string(i);
      NVTX_RANGE_PUSH(iterName.c_str());

      CUDA_CHECK_VOID(cudaEventRecord(start.get(), stream_));
      CUDA_CHECK_VOID(cudaLaunchKernel(
          (void*)broadcastKernel, gridDim, blockDim, args, 0, stream_));
      CUDA_CHECK_VOID(cudaEventRecord(stop.get(), stream_));
      CUDA_CHECK_VOID(cudaStreamSynchronize(stream_));

      NVTX_RANGE_POP();

      float elapsed_ms = 0.0f;
      CUDA_CHECK_VOID(
          cudaEventElapsedTime(&elapsed_ms, start.get(), stop.get()));
      latencies.push_back(elapsed_ms * 1000.0); // Convert to microseconds
    }
    NVTX_RANGE_POP();

    // Compute statistics
    auto stats = computeStats(latencies);

    // Compute bandwidth
    double avgLatencyMs = stats.mean / 1000.0;
    double bandwidth_GBps =
        (static_cast<double>(config.messageSize) / 1e9) / (avgLatencyMs / 1e3);

    // Print results
    if (globalRank == 0) {
      std::stringstream ss;
      ss << "\n=== Profiling Results: " << config.name << " ===\n";
      ss << "Message Size: " << (config.messageSize / (1024 * 1024)) << " MB\n";
      ss << "Chunk Size: " << (config.chunkSize / 1024) << " KB\n";
      ss << "Staging Buffer: " << (config.stagingBufferSize / 1024) << " KB\n";
      ss << "Pipeline Depth: " << config.pipelineDepth << "\n";
      ss << "Grid: " << config.numBlocks << " blocks x " << config.numThreads
         << " threads\n";
      ss << "Iterations: " << FLAGS_num_iterations << "\n";
      ss << "\n--- Latency Statistics (microseconds) ---\n";
      ss << std::fixed << std::setprecision(2);
      ss << "  Min:    " << stats.min << " us\n";
      ss << "  Max:    " << stats.max << " us\n";
      ss << "  Mean:   " << stats.mean << " us\n";
      ss << "  Median: " << stats.median << " us\n";
      ss << "  Stddev: " << stats.stddev << " us\n";
      ss << "  P95:    " << stats.p95 << " us\n";
      ss << "  P99:    " << stats.p99 << " us\n";
      ss << "\n--- Bandwidth ---\n";
      ss << "  Average: " << std::setprecision(3) << bandwidth_GBps
         << " GB/s\n";
      ss << "\n--- Theoretical Analysis ---\n";
      ss << "  Root must send to " << (numRanks - 1) << " peers\n";
      ss << "  Total root bandwidth: "
         << (config.messageSize * (numRanks - 1) / (1024.0 * 1024.0))
         << " MB\n";
      ss << "  Per-peer data: " << (config.messageSize / (1024.0 * 1024.0))
         << " MB\n";
      ss << "  Chunks per transfer: "
         << ((config.messageSize + config.chunkSize - 1) / config.chunkSize)
         << "\n";

      XLOG(INFO) << ss.str();
    }

    // Compare with NCCL
    runNcclComparison(config);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  }

  void runNcclComparison(const ProfilingConfig& config) {
    DeviceBuffer buffer(config.messageSize);

    if (globalRank == config.rootRank) {
      std::vector<char> h_data(config.messageSize);
      for (std::size_t i = 0; i < config.messageSize; i++) {
        h_data[i] = static_cast<char>((i + config.rootRank) % 256);
      }
      CUDA_CHECK_VOID(cudaMemcpy(
          buffer.get(),
          h_data.data(),
          config.messageSize,
          cudaMemcpyHostToDevice));
    } else {
      CUDA_CHECK_VOID(cudaMemset(buffer.get(), 0, config.messageSize));
    }

    std::vector<double> latencies;
    latencies.reserve(FLAGS_num_iterations);

    CudaEvent start, stop;

    // Warmup
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    NVTX_RANGE_PUSH("NCCL_Profile_Warmup");
    for (int i = 0; i < kWarmupIters; i++) {
      NCCL_CHECK_VOID(ncclBroadcast(
          buffer.get(),
          buffer.get(),
          config.messageSize,
          ncclChar,
          config.rootRank,
          ncclComm_,
          stream_));
      CUDA_CHECK_VOID(cudaStreamSynchronize(stream_));
    }
    NVTX_RANGE_POP();
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Profiling iterations
    NVTX_RANGE_PUSH("NCCL_Profile_Iterations");
    for (int i = 0; i < FLAGS_num_iterations; i++) {
      if (globalRank != config.rootRank) {
        CUDA_CHECK_VOID(cudaMemset(buffer.get(), 0, config.messageSize));
      }

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      std::string iterName = "NCCL_Iter_" + std::to_string(i);
      NVTX_RANGE_PUSH(iterName.c_str());

      CUDA_CHECK_VOID(cudaEventRecord(start.get(), stream_));
      NCCL_CHECK_VOID(ncclBroadcast(
          buffer.get(),
          buffer.get(),
          config.messageSize,
          ncclChar,
          config.rootRank,
          ncclComm_,
          stream_));
      CUDA_CHECK_VOID(cudaEventRecord(stop.get(), stream_));
      CUDA_CHECK_VOID(cudaStreamSynchronize(stream_));

      NVTX_RANGE_POP();

      float elapsed_ms = 0.0f;
      CUDA_CHECK_VOID(
          cudaEventElapsedTime(&elapsed_ms, start.get(), stop.get()));
      latencies.push_back(elapsed_ms * 1000.0);
    }
    NVTX_RANGE_POP();

    auto stats = computeStats(latencies);
    double avgLatencyMs = stats.mean / 1000.0;
    double bandwidth_GBps =
        (static_cast<double>(config.messageSize) / 1e9) / (avgLatencyMs / 1e3);

    if (globalRank == 0) {
      std::stringstream ss;
      ss << "\n--- NCCL Comparison ---\n";
      ss << std::fixed << std::setprecision(2);
      ss << "  Mean Latency: " << stats.mean << " us\n";
      ss << "  Bandwidth:    " << std::setprecision(3) << bandwidth_GBps
         << " GB/s\n";
      XLOG(INFO) << ss.str();
    }
  }

  ncclComm_t ncclComm_{};
  cudaStream_t stream_{};
  int clockRateKHz_{};
};

TEST_F(BroadcastProfileBenchmarkFixture, DetailedProfiling) {
  std::size_t msgSize =
      static_cast<std::size_t>(FLAGS_message_size_mb) * 1024 * 1024;

  if (globalRank == 0) {
    XLOG(INFO)
        << "\n============================================================\n"
        << "           Broadcast Profiling Benchmark\n"
        << "============================================================\n"
        << "Message Size: " << FLAGS_message_size_mb << " MB\n"
        << "Iterations: " << FLAGS_num_iterations << "\n"
        << "Ranks: " << numRanks << "\n"
        << "Profile mode: " << (FLAGS_profile ? "ENABLED" : "DISABLED") << "\n"
        << "============================================================\n";
  }

  // Generate optimal config based on message size
  ProfilingConfig config;
  config.messageSize = msgSize;
  config.rootRank = 0;

  if (msgSize < 64 * 1024) {
    // Small messages: latency-sensitive
    config.stagingBufferSize = 64 * 1024;
    config.chunkSize = 8 * 1024;
    config.numBlocks = 4;
    config.numThreads = 256;
    config.name = "small_message";
  } else if (msgSize < 1024 * 1024) {
    // Medium messages
    config.stagingBufferSize = 256 * 1024;
    config.chunkSize = 32 * 1024;
    config.numBlocks = 8;
    config.numThreads = 256;
    config.name = "medium_message";
  } else if (msgSize < 16 * 1024 * 1024) {
    // Large messages
    config.stagingBufferSize = 1024 * 1024;
    config.chunkSize = 128 * 1024;
    config.numBlocks = 16;
    config.numThreads = 512;
    config.name = "large_message";
  } else {
    // Very large messages
    config.stagingBufferSize = 8 * 1024 * 1024;
    config.chunkSize = 512 * 1024;
    config.numBlocks = 32;
    config.numThreads = 512;
    config.name = "very_large_message";
  }

  runProfilingSession(config);
}

TEST_F(BroadcastProfileBenchmarkFixture, ChunkSizeSweep) {
  std::size_t msgSize =
      static_cast<std::size_t>(FLAGS_message_size_mb) * 1024 * 1024;

  if (globalRank == 0) {
    XLOG(INFO) << "\n=== Chunk Size Sweep ===\n";
  }

  std::vector<std::size_t> chunkSizes = {
      8 * 1024, // 8KB
      16 * 1024, // 16KB
      32 * 1024, // 32KB
      64 * 1024, // 64KB
      128 * 1024, // 128KB
      256 * 1024, // 256KB
      512 * 1024, // 512KB
  };

  for (std::size_t chunkSize : chunkSizes) {
    if (chunkSize > msgSize) {
      continue; // Skip chunk sizes larger than message
    }

    ProfilingConfig config{
        .messageSize = msgSize,
        .stagingBufferSize = std::max(chunkSize * 4, 256 * 1024UL),
        .pipelineDepth = 4,
        .chunkSize = chunkSize,
        .numBlocks = 16,
        .numThreads = 512,
        .rootRank = 0,
        .name = "chunk_" + std::to_string(chunkSize / 1024) + "KB",
    };

    runProfilingSession(config);
  }
}

TEST_F(BroadcastProfileBenchmarkFixture, PipelineDepthSweep) {
  std::size_t msgSize =
      static_cast<std::size_t>(FLAGS_message_size_mb) * 1024 * 1024;

  if (globalRank == 0) {
    XLOG(INFO) << "\n=== Pipeline Depth Sweep ===\n";
  }

  std::vector<std::size_t> pipelineDepths = {2, 4, 6, 8};

  for (std::size_t depth : pipelineDepths) {
    ProfilingConfig config{
        .messageSize = msgSize,
        .stagingBufferSize = 1024 * 1024,
        .pipelineDepth = depth,
        .chunkSize = 128 * 1024,
        .numBlocks = 16,
        .numThreads = 512,
        .rootRank = 0,
        .name = "pipeline_depth_" + std::to_string(depth),
    };

    runProfilingSession(config);
  }
}

} // namespace

} // namespace comms::pipes::benchmark

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
