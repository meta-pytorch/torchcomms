// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <vector>

#include "comms/common/DeviceConstants.cuh"
#include "comms/pipes/MultiPeerNvlTransport.h"
#include "comms/pipes/benchmarks/MultiPeerBenchmark.cuh"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

// Command-line configurable parameters
DEFINE_int32(benchmark_iters, 50, "Number of benchmark iterations");
DEFINE_int32(steps_per_iter, 100, "Number of steps per kernel launch");
DEFINE_int32(warmup_iters, 10, "Number of warmup iterations");

using meta::comms::CudaEvent;
using meta::comms::MpiBaseTestFixture;
using meta::comms::MPIEnvironmentBase;

namespace comms::pipes::benchmark {

constexpr int kDefaultThreads = 128; // Thread configuration

// =============================================================================
// Runtime Configuration Helpers
// =============================================================================

// Validate flag values are positive
inline int getWarmupIters() {
  CHECK_GT(FLAGS_warmup_iters, 0) << "warmup_iters must be positive";
  return FLAGS_warmup_iters;
}

inline int getBenchmarkIters() {
  CHECK_GT(FLAGS_benchmark_iters, 0) << "benchmark_iters must be positive";
  return FLAGS_benchmark_iters;
}

inline int getStepsPerIter() {
  CHECK_GT(FLAGS_steps_per_iter, 0) << "steps_per_iter must be positive";
  return FLAGS_steps_per_iter;
}

// =============================================================================
// Configuration and Results
// =============================================================================

struct MultiPeerBenchConfig {
  int numBlocks{1};
  bool useBlockGroups{false};
  std::string name;

  // Calculate number of thread groups (and required slots)
  int numGroups() const {
    return useBlockGroups
        ? numBlocks
        : numBlocks * (kDefaultThreads / comms::device::kWarpSize);
  }

  // Get scope string for display (returns const char* for efficiency)
  const char* scopeString() const {
    return useBlockGroups ? "block" : "warp";
  }
};

struct MultiPeerBenchResult {
  std::string configName;
  int nRanks{};
  std::string groupScope;
  float latencyUs{};
};

// =============================================================================
// Configuration Helpers
// =============================================================================

// Single source of truth for all configurations
std::vector<MultiPeerBenchConfig> getStandardConfigs(bool reduced = false) {
  // Full configuration list
  static const std::vector<MultiPeerBenchConfig> kAllConfigs = {
      {.numBlocks = 1, .useBlockGroups = false, .name = "1b_warp"},
      {.numBlocks = 4, .useBlockGroups = false, .name = "4b_warp"},
      {.numBlocks = 16, .useBlockGroups = false, .name = "16b_warp"},
      {.numBlocks = 1, .useBlockGroups = true, .name = "1b_block"},
      {.numBlocks = 4, .useBlockGroups = true, .name = "4b_block"},
      {.numBlocks = 16, .useBlockGroups = true, .name = "16b_block"},
  };

  if (!reduced) {
    return kAllConfigs;
  }

  // Reduced set for expensive operations (SignalAll) - filter from full list
  std::vector<MultiPeerBenchConfig> filtered;
  filtered.reserve(4);
  for (const auto& c : kAllConfigs) {
    if (c.numBlocks <= 4) {
      filtered.push_back(c);
    }
  }
  return filtered;
}

// Calculate max slots with validation for empty config list
inline int getMaxSlots(const std::vector<MultiPeerBenchConfig>& configs) {
  CHECK(!configs.empty()) << "Config list cannot be empty";
  int maxSlots = 0;
  for (const auto& c : configs) {
    maxSlots = std::max(maxSlots, c.numGroups());
  }
  return maxSlots;
}

// =============================================================================
// Benchmark Fixture
// =============================================================================

class MultiPeerBenchmarkFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    ASSERT_EQ(cudaSetDevice(localRank), cudaSuccess);
    ASSERT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
  }

  void TearDown() override {
    ASSERT_EQ(cudaStreamDestroy(stream_), cudaSuccess);
    MpiBaseTestFixture::TearDown();
  }

  // -------------------------------------------------------------------------
  // Timing Result Structure
  // -------------------------------------------------------------------------

  struct TimingResult {
    float avgLatencyUs{0.0f};
    bool success{true};
  };

  // -------------------------------------------------------------------------
  // Benchmark Execution Helper
  // -------------------------------------------------------------------------

  // Note: Lambda returns cudaError_t to avoid ASSERT inside lambda issues
  template <typename LaunchFn>
  TimingResult runBenchmark(
      LaunchFn launchKernel,
      int stepsPerIter,
      bool useMpiBarrier = true) {
    const int warmupIters = getWarmupIters();
    const int benchIters = getBenchmarkIters();

    TimingResult result;

    // --- Warmup Phase ---
    if (useMpiBarrier) {
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    for (int i = 0; i < warmupIters; ++i) {
      cudaError_t err = launchKernel();
      if (err != cudaSuccess) {
        XLOGF(
            ERR,
            "Kernel launch failed during warmup: {}",
            cudaGetErrorString(err));
        result.success = false;
        return result;
      }
      err = cudaStreamSynchronize(stream_);
      if (err != cudaSuccess) {
        XLOGF(
            ERR,
            "Stream sync failed during warmup: {}",
            cudaGetErrorString(err));
        result.success = false;
        return result;
      }
    }

    // Verify no CUDA errors accumulated after warmup
    cudaError_t lastErr = cudaGetLastError();
    if (lastErr != cudaSuccess) {
      XLOGF(
          ERR,
          "CUDA error detected after warmup: {}",
          cudaGetErrorString(lastErr));
      result.success = false;
      return result;
    }

    if (useMpiBarrier) {
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    // --- Batch Timing for Average ---
    CudaEvent batchStart, batchStop;

    cudaError_t err = cudaEventRecord(batchStart.get(), stream_);
    if (err != cudaSuccess) {
      result.success = false;
      return result;
    }

    for (int i = 0; i < benchIters; ++i) {
      err = launchKernel();
      if (err != cudaSuccess) {
        result.success = false;
        return result;
      }
    }

    err = cudaEventRecord(batchStop.get(), stream_);
    if (err != cudaSuccess) {
      result.success = false;
      return result;
    }

    err = cudaStreamSynchronize(stream_);
    if (err != cudaSuccess) {
      result.success = false;
      return result;
    }

    float totalTimeMs = 0.0f;
    err = cudaEventElapsedTime(&totalTimeMs, batchStart.get(), batchStop.get());
    if (err != cudaSuccess) {
      result.success = false;
      return result;
    }

    // Calculate average per-step latency
    int totalSteps = benchIters * stepsPerIter;
    result.avgLatencyUs = (totalTimeMs / totalSteps) * 1000.0f;

    if (useMpiBarrier) {
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    return result;
  }

  // -------------------------------------------------------------------------
  // Unified Table Printing
  // -------------------------------------------------------------------------

  void printResultsTable(
      const std::string& title,
      const std::string& latencyLabel,
      const std::vector<MultiPeerBenchResult>& results,
      bool showRanks = true) {
    if (globalRank != 0) {
      return;
    }

    std::stringstream ss;
    ss << "\n";
    ss << "========================================================================\n";
    ss << "     " << title << "\n";
    ss << "========================================================================\n";

    // Header
    ss << std::left << std::setw(12) << "Config";
    if (showRanks) {
      ss << std::right << std::setw(8) << "nRanks";
    }
    ss << std::right << std::setw(10) << "Scope" << std::right << std::setw(16)
       << "Latency (us)\n";
    ss << "------------------------------------------------------------------------\n";

    // Rows
    for (const auto& r : results) {
      ss << std::left << std::setw(12) << r.configName;
      if (showRanks) {
        ss << std::right << std::setw(8) << r.nRanks;
      }
      ss << std::right << std::setw(10) << r.groupScope << std::right
         << std::setw(16) << std::fixed << std::setprecision(3) << r.latencyUs
         << "\n";
    }

    ss << "========================================================================\n";
    ss << latencyLabel << "\n";
    ss << "Each measurement: " << getBenchmarkIters() << " iterations x "
       << getStepsPerIter() << " steps/iter\n";
    ss << "========================================================================\n\n";

    XLOG(INFO) << ss.str();
  }

  cudaStream_t stream_{};
};

// =============================================================================
// Benchmark Test Cases
// =============================================================================

// =============================================================================
// Test: Signal-All Latency (N-way signaling)
// =============================================================================

TEST_F(MultiPeerBenchmarkFixture, SignalAllLatency) {
  if (numRanks < 2) {
    GTEST_SKIP() << "Requires >= 2 ranks, got " << numRanks;
  }

  // Use reduced config set for expensive operations
  auto configs = getStandardConfigs(/*reduced=*/true);
  std::vector<MultiPeerBenchResult> results;

  int maxSlots = getMaxSlots(configs);
  MultiPeerNvlTransportConfig transportConfig{
      .dataBufferSize = 1,
      .chunkSize = 1,
      .pipelineDepth = 1,
      .signalCount = static_cast<std::size_t>(maxSlots),
  };

  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  MultiPeerNvlTransport transport(
      globalRank, numRanks, bootstrap, transportConfig);
  transport.exchange();
  auto device = transport.getMultiPeerDeviceTransport();

  for (const auto& config : configs) {
    void* kernelFunc = config.useBlockGroups
        ? (void*)multiPeerSignalAllKernel<SyncScope::BLOCK>
        : (void*)multiPeerSignalAllKernel<SyncScope::WARP>;

    int nSteps = getStepsPerIter();
    dim3 grid(config.numBlocks);
    dim3 block(kDefaultThreads);
    void* args[] = {&device, &nSteps};

    auto timing = runBenchmark(
        [&]() -> cudaError_t {
          return cudaLaunchKernel(kernelFunc, grid, block, args, 0, stream_);
        },
        nSteps,
        /*useMpiBarrier=*/true);

    ASSERT_TRUE(timing.success)
        << "Benchmark failed for config: " << config.name;

    results.push_back({
        .configName = config.name,
        .nRanks = numRanks,
        .groupScope = config.scopeString(),
        .latencyUs = timing.avgLatencyUs,
    });
  }

  printResultsTable(
      "Signal-All Benchmark",
      "Latency = Average time per signalAll/waitAll cycle",
      results);
}

// =============================================================================
// Test: Signal Ping-Pong Latency (2-way signaling)
// =============================================================================

TEST_F(MultiPeerBenchmarkFixture, SignalPingPongLatency) {
  if (numRanks != 2) {
    GTEST_SKIP() << "Requires exactly 2 ranks, got " << numRanks;
  }

  int peerIndex = 0; // In 2-rank setup, each rank has exactly one peer
  auto configs = getStandardConfigs();
  std::vector<MultiPeerBenchResult> results;

  int maxSlots = getMaxSlots(configs);
  MultiPeerNvlTransportConfig transportConfig{
      .dataBufferSize = 1,
      .chunkSize = 1,
      .pipelineDepth = 1,
      .signalCount = static_cast<std::size_t>(maxSlots),
  };

  auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
  MultiPeerNvlTransport transport(
      globalRank, numRanks, bootstrap, transportConfig);
  transport.exchange();
  auto device = transport.getMultiPeerDeviceTransport();

  for (const auto& config : configs) {
    void* kernelFunc = config.useBlockGroups
        ? (void*)multiPeerSignalPingPongKernel<SyncScope::BLOCK>
        : (void*)multiPeerSignalPingPongKernel<SyncScope::WARP>;

    int nSteps = getStepsPerIter();
    dim3 grid(config.numBlocks);
    dim3 block(kDefaultThreads);
    void* args[] = {&device, &peerIndex, &nSteps};

    auto timing = runBenchmark(
        [&]() -> cudaError_t {
          return cudaLaunchKernel(kernelFunc, grid, block, args, 0, stream_);
        },
        nSteps,
        /*useMpiBarrier=*/true);

    ASSERT_TRUE(timing.success)
        << "Benchmark failed for config: " << config.name;

    results.push_back({
        .configName = config.name,
        .nRanks = 2,
        .groupScope = config.scopeString(),
        .latencyUs = timing.avgLatencyUs,
    });
  }

  printResultsTable(
      "Signal Ping-Pong Benchmark (Half-Duplex)",
      "Latency = Average time per signal/wait pair",
      results);
}

} // namespace comms::pipes::benchmark

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
