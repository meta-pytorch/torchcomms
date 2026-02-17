// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>
#include <gflags/gflags.h>
#include <nccl.h>
#include <type_traits>

#ifdef USE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

#include "comms/common/CudaWrap.h"
#include "comms/pipes/MultiPeerNvlTransport.h"
#include "comms/pipes/benchmarks/BenchmarkKernel.cuh"
#include "comms/pipes/benchmarks/BenchmarkMacros.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

#include <iomanip>
#include <sstream>
#include <vector>

// Command-line flag for enabling data correctness verification
DEFINE_bool(
    verify_correctness,
    false,
    "Enable data correctness verification after broadcast operations");

// Command-line flag for enabling NVTX profiling annotations
DEFINE_bool(
    profile,
    false,
    "Enable NVTX profiling annotations for nsys profiling");

// Command-line flag for selecting which benchmarks to run
// Valid values: "all", "optimal", "tuning", "algorithm"
// Can also be comma-separated: "optimal,algorithm"
DEFINE_string(
    benchmark,
    "all",
    "Which benchmark(s) to run: all, clustered, rootsweep, extended, gridconfig, optimal, tuning, algorithm (comma-separated)");

namespace {
// NVTX helper macros for profiling
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

// ============================================================================
// FILE ORGANIZATION
// ============================================================================
// This file is organized into the following sections:
//
//   1. CONFIGURATION STRUCTS
//      - BroadcastBenchmarkConfig, GridConfig, BroadcastBenchmarkResult
//
//   2. TEST FIXTURE CLASS
//      - BroadcastBenchmarkFixture with SetUp/TearDown
//      - Data verification helpers
//      - NCCL benchmark runner
//      - Pipes transport setup helpers
//      - Pipes benchmark runners (standard and clustered)
//      - Result formatting utilities
//
//   3. TEST CASES
//      - ClusteredLaunchComparison: Standard vs clustered kernel launch
//      - RootRankSweep: Performance with different root ranks
//      - ExtendedMessageSizes: 64B to 256MB message sweep
//      - GridConfigSweep: Block/thread configuration optimization
//
//   4. MAIN FUNCTION
//      - CLI argument parsing and test filtering
// ============================================================================

namespace {

// ============================================================================
// SECTION 1: CONFIGURATION STRUCTS
// ============================================================================

/**
 * Test configuration for Broadcast benchmark.
 */
struct BroadcastBenchmarkConfig {
  std::size_t messageSize;
  std::size_t stagingBufferSize;
  std::size_t pipelineDepth = 4;
  std::size_t chunkSize = 32 * 1024; // 32KB default (profiling shows optimal)
  int numBlocks;
  int numThreads;
  int rootRank; // Which rank is the broadcast source (use -1 for dynamic)
  std::string name;
};

/**
 * Grid configuration for GPU kernel launches.
 * Used for sweeping block/thread configurations in benchmarks.
 */
struct GridConfig {
  int numBlocks;
  int numThreads;
  std::string name; // Display name, e.g., "4x128", "32x512"
};

/**
 * Result struct for collecting benchmark data.
 */
struct BroadcastBenchmarkResult {
  std::string testName;
  std::size_t messageSize{};
  int rootRank{};
  int nRanks{};
  float ncclBandwidth{}; // GB/s
  float pipesBandwidth{}; // GB/s
  float ncclLatency{}; // microseconds
  float pipesLatency{}; // microseconds
  float speedup{}; // Pipes / NCCL
  bool ncclVerified{}; // Data correctness verified for NCCL
  bool pipesVerified{}; // Data correctness verified for Pipes
};

/**
 * Broadcast algorithm types for parameterized testing.
 * New algorithms should be added here as they are implemented.
 */
enum class BroadcastAlgorithm {
  FlatTree,
  BinomialTree,
  Ring,
  // Adaptive - added in D91729719
};

inline std::string algorithmName(BroadcastAlgorithm algo) {
  switch (algo) {
    case BroadcastAlgorithm::FlatTree:
      return "Flat-Tree";
    case BroadcastAlgorithm::BinomialTree:
      return "Binomial";
    case BroadcastAlgorithm::Ring:
      return "Ring";
  }
  return "Unknown";
}

/**
 * All broadcast algorithms for parameterized testing.
 * Add new algorithms here as they are implemented.
 */
const std::vector<BroadcastAlgorithm> kAllAlgorithms = {
    BroadcastAlgorithm::FlatTree,
    BroadcastAlgorithm::BinomialTree,
    BroadcastAlgorithm::Ring,
};

// ============================================================================
// SECTION 2: TEST FIXTURE CLASS
// ============================================================================

class BroadcastBenchmarkFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    // Use localRank for cudaSetDevice since each node has its own set of GPUs
    CUDA_CHECK_VOID(cudaSetDevice(localRank));

    // Initialize NCCL
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

  // --------------------------------------------------------------------------
  // Data Generation and Verification Helpers
  // --------------------------------------------------------------------------

  /**
   * Generate expected data pattern for a given root rank.
   * The pattern is (byte_index + rootRank) % 256.
   */
  std::vector<char> generateExpectedData(std::size_t size, int rootRank) {
    std::vector<char> data(size);
    for (std::size_t i = 0; i < size; i++) {
      data[i] = static_cast<char>((i + rootRank) % 256);
    }
    return data;
  }

  /**
   * Verify that buffer contains the expected broadcast data.
   * Returns true if verification passes, false otherwise.
   */
  bool verifyBroadcastData(
      void* buffer_d,
      std::size_t size,
      int rootRank,
      const std::string& testName,
      const std::string& impl) {
    std::vector<char> h_result(size);
    CUDA_CHECK_BOOL(
        cudaMemcpy(h_result.data(), buffer_d, size, cudaMemcpyDeviceToHost));

    auto expected = generateExpectedData(size, rootRank);

    std::size_t mismatchCount = 0;
    std::size_t firstMismatch = 0;
    char expectedVal = 0, actualVal = 0;

    for (std::size_t i = 0; i < size; i++) {
      if (h_result[i] != expected[i]) {
        if (mismatchCount == 0) {
          firstMismatch = i;
          expectedVal = expected[i];
          actualVal = h_result[i];
        }
        mismatchCount++;
      }
    }

    if (mismatchCount > 0) {
      XLOGF(
          ERR,
          "Rank {}: {} {} verification FAILED: {} mismatches out of {} bytes. "
          "First mismatch at byte {}: expected {}, got {}",
          globalRank,
          impl,
          testName,
          mismatchCount,
          size,
          firstMismatch,
          static_cast<int>(expectedVal),
          static_cast<int>(actualVal));
      return false;
    }

    XLOGF(
        DBG1,
        "Rank {}: {} {} verification PASSED ({} bytes)",
        globalRank,
        impl,
        testName,
        size);
    return true;
  }

  // --------------------------------------------------------------------------
  // NCCL Baseline Benchmark
  // --------------------------------------------------------------------------

  /**
   * Run NCCL Broadcast benchmark.
   * Returns bandwidth in GB/s and sets latency in microseconds.
   */
  float runNcclBroadcastBenchmark(
      const BroadcastBenchmarkConfig& config,
      int rootRank,
      float& latencyUs,
      bool& verified) {
    XLOGF(
        DBG1,
        "Rank {}: Running NCCL Broadcast benchmark: {} (root={})",
        globalRank,
        config.name,
        rootRank);

    verified = false;
    DeviceBuffer buffer(config.messageSize);

    // Initialize buffer: root has data, others have zeros
    if (globalRank == rootRank) {
      auto h_data = generateExpectedData(config.messageSize, rootRank);
      CUDA_CHECK(cudaMemcpy(
          buffer.get(),
          h_data.data(),
          config.messageSize,
          cudaMemcpyHostToDevice));
    } else {
      CUDA_CHECK(cudaMemset(buffer.get(), 0, config.messageSize));
    }

    CudaEvent start, stop;

    // Warmup
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    NVTX_RANGE_PUSH("NCCL_Broadcast_Warmup");
    for (int i = 0; i < kWarmupIters; i++) {
      NCCL_CHECK(ncclBroadcast(
          buffer.get(),
          buffer.get(),
          config.messageSize,
          ncclChar,
          rootRank,
          ncclComm_,
          stream_));
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    NVTX_RANGE_POP();
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Benchmark
    NVTX_RANGE_PUSH("NCCL_Broadcast_Benchmark");
    CUDA_CHECK(cudaEventRecord(start.get(), stream_));
    for (int i = 0; i < kBenchmarkIters; i++) {
      NVTX_RANGE_PUSH("NCCL_Broadcast_Iter");
      NCCL_CHECK(ncclBroadcast(
          buffer.get(),
          buffer.get(),
          config.messageSize,
          ncclChar,
          rootRank,
          ncclComm_,
          stream_));
      NVTX_RANGE_POP();
    }
    CUDA_CHECK(cudaEventRecord(stop.get(), stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    NVTX_RANGE_POP();

    float totalTime_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime_ms, start.get(), stop.get()));
    float avgTime_ms = totalTime_ms / kBenchmarkIters;
    latencyUs = avgTime_ms * 1000.0f;

    // Broadcast bandwidth: data transferred / time
    // (root sends to all other ranks, so effective data = messageSize)
    // Use double for intermediate calculations to avoid precision loss with
    // small messages
    float bandwidth_GBps = static_cast<float>(
        (static_cast<double>(config.messageSize) / 1e9) /
        (static_cast<double>(avgTime_ms) / 1000.0));

    // Verify data correctness if enabled
    if (FLAGS_verify_correctness) {
      // Re-initialize and run one more broadcast for verification
      if (globalRank == rootRank) {
        auto h_data = generateExpectedData(config.messageSize, rootRank);
        CUDA_CHECK(cudaMemcpy(
            buffer.get(),
            h_data.data(),
            config.messageSize,
            cudaMemcpyHostToDevice));
      } else {
        CUDA_CHECK(cudaMemset(buffer.get(), 0, config.messageSize));
      }

      NCCL_CHECK(ncclBroadcast(
          buffer.get(),
          buffer.get(),
          config.messageSize,
          ncclChar,
          rootRank,
          ncclComm_,
          stream_));
      CUDA_CHECK(cudaStreamSynchronize(stream_));

      verified = verifyBroadcastData(
          buffer.get(), config.messageSize, rootRank, config.name, "NCCL");
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    return bandwidth_GBps;
  }

  // --------------------------------------------------------------------------
  // Pipes Transport Setup Helpers
  // --------------------------------------------------------------------------

  /**
   * Initialize broadcast buffer: root rank gets expected data, others get
   * zeros. Returns 0.0f on failure (for CUDA_CHECK compatibility).
   */
  float initializeBroadcastBuffer(
      DeviceBuffer& buffer,
      std::size_t messageSize,
      int rootRank) {
    if (globalRank == rootRank) {
      auto h_data = generateExpectedData(messageSize, rootRank);
      CUDA_CHECK(cudaMemcpy(
          buffer.get(), h_data.data(), messageSize, cudaMemcpyHostToDevice));
    } else {
      CUDA_CHECK(cudaMemset(buffer.get(), 0, messageSize));
    }
    return 1.0f; // Success
  }

  /**
   * Result struct for transport setup.
   */
  struct TransportSetupResult {
    std::unique_ptr<MultiPeerNvlTransport> transport;
    DeviceBuffer d_transports;
    DeviceSpan<Transport> transports_span;
  };

  /**
   * Setup P2P NVL transports for Pipes benchmark.
   * Returns a struct containing the transport objects and device span.
   */
  TransportSetupResult setupPipesTransports(
      const BroadcastBenchmarkConfig& config) {
    MultiPeerNvlTransportConfig nvlConfig{
        .dataBufferSize = config.stagingBufferSize,
        .chunkSize = config.chunkSize,
        .pipelineDepth = config.pipelineDepth,
    };

    auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
    auto transport = std::make_unique<MultiPeerNvlTransport>(
        globalRank, numRanks, bootstrap, nvlConfig);
    transport->exchange();

    // Create transport array: self for my rank, P2P for others
    P2pSelfTransportDevice selfTransport;
    std::vector<Transport> h_transports;
    h_transports.reserve(numRanks);

    for (int rank = 0; rank < numRanks; rank++) {
      if (rank == globalRank) {
        h_transports.emplace_back(selfTransport);
      } else {
        h_transports.emplace_back(transport->getP2pTransportDevice(rank));
      }
    }

    // Copy transports to device
    DeviceBuffer d_transports(sizeof(Transport) * numRanks);
    cudaMemcpy(
        d_transports.get(),
        h_transports.data(),
        sizeof(Transport) * numRanks,
        cudaMemcpyHostToDevice);

    DeviceSpan<Transport> transports_span(
        static_cast<Transport*>(d_transports.get()), numRanks);

    return TransportSetupResult{
        std::move(transport),
        std::move(d_transports),
        transports_span,
    };
  }

  /**
   * Calculate bandwidth and latency from timing data.
   */
  struct BenchmarkTiming {
    float bandwidthGBps;
    float latencyUs;
  };

  BenchmarkTiming calculateBenchmarkTiming(
      float totalTimeMs,
      std::size_t messageSize) {
    float avgTimeMs = totalTimeMs / kBenchmarkIters;
    float latencyUs = avgTimeMs * 1000.0f;

    // Use double for intermediate calculations to avoid precision loss with
    // small messages
    float bandwidthGBps = static_cast<float>(
        (static_cast<double>(messageSize) / 1e9) /
        (static_cast<double>(avgTimeMs) / 1000.0));

    return BenchmarkTiming{bandwidthGBps, latencyUs};
  }

  /**
   * Generic Pipes broadcast benchmark runner.
   *
   * Runs the specified broadcast kernel through warmup, benchmark, and optional
   * verification phases. This eliminates code duplication across algorithm-
   * specific benchmark functions.
   *
   * @param kernelFunc Pointer to the broadcast kernel function
   * @param algorithmName Name of the algorithm for logging/verification
   * @param config Benchmark configuration
   * @param rootRank Root rank for the broadcast
   * @param latencyUs Output: average latency in microseconds
   * @param verified Output: whether verification passed (if enabled)
   * @return Bandwidth in GB/s
   */
  template <typename KernelFunc>
  float runGenericPipesBroadcast(
      KernelFunc kernelFunc,
      const std::string& algorithmName,
      const BroadcastBenchmarkConfig& config,
      int rootRank,
      float& latencyUs,
      bool& verified) {
    XLOGF(
        DBG1,
        "Rank {}: Running Pipes {} broadcast: {} (root={})",
        globalRank,
        algorithmName,
        config.name,
        rootRank);

    verified = false;
    DeviceBuffer buffer(config.messageSize);

    // Initialize buffer using helper
    if (initializeBroadcastBuffer(buffer, config.messageSize, rootRank) ==
        0.0f) {
      return 0.0f;
    }

    // Setup transports using helper
    auto transportSetup = setupPipesTransports(config);

    // Prepare kernel launch parameters
    dim3 gridDim(config.numBlocks);
    dim3 blockDim(config.numThreads);

    void* buff_d = buffer.get();
    std::size_t nbytes = config.messageSize;
    void* args[] = {
        &buff_d,
        &globalRank,
        &rootRank,
        &transportSetup.transports_span,
        &nbytes};

    CudaEvent start, stop;

    // Warmup
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    std::string warmupRange = "Pipes_" + algorithmName + "_Warmup";
    NVTX_RANGE_PUSH(warmupRange.c_str());
    for (int i = 0; i < kWarmupIters; i++) {
      CUDA_CHECK(cudaLaunchKernel(
          (void*)kernelFunc, gridDim, blockDim, args, 0, stream_));
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    NVTX_RANGE_POP();
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Benchmark
    std::string benchmarkRange = "Pipes_" + algorithmName + "_Benchmark";
    NVTX_RANGE_PUSH(benchmarkRange.c_str());
    CUDA_CHECK(cudaEventRecord(start.get(), stream_));
    for (int i = 0; i < kBenchmarkIters; i++) {
      std::string iterRange = "Pipes_" + algorithmName + "_Iter";
      NVTX_RANGE_PUSH(iterRange.c_str());
      CUDA_CHECK(cudaLaunchKernel(
          (void*)kernelFunc, gridDim, blockDim, args, 0, stream_));
      NVTX_RANGE_POP();
    }
    CUDA_CHECK(cudaEventRecord(stop.get(), stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    NVTX_RANGE_POP();

    float totalTime_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime_ms, start.get(), stop.get()));

    // Calculate timing using helper
    auto timing = calculateBenchmarkTiming(totalTime_ms, config.messageSize);
    latencyUs = timing.latencyUs;

    // Verify data correctness if enabled
    if (FLAGS_verify_correctness) {
      // Re-initialize buffer for verification
      if (initializeBroadcastBuffer(buffer, config.messageSize, rootRank) ==
          0.0f) {
        return 0.0f;
      }

      buff_d = buffer.get();
      CUDA_CHECK(cudaLaunchKernel(
          (void*)kernelFunc, gridDim, blockDim, args, 0, stream_));
      CUDA_CHECK(cudaStreamSynchronize(stream_));

      verified = verifyBroadcastData(
          buffer.get(),
          config.messageSize,
          rootRank,
          config.name,
          algorithmName);
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    return timing.bandwidthGBps;
  }

  // --------------------------------------------------------------------------
  // Pipes Benchmark Runners
  // --------------------------------------------------------------------------

  /**
   * Run Pipes Broadcast benchmark with flat-tree algorithm.
   * Returns bandwidth in GB/s and sets latency in microseconds.
   */
  float runPipesBroadcastFlatBenchmark(
      const BroadcastBenchmarkConfig& config,
      int rootRank,
      float& latencyUs,
      bool& verified) {
    return runGenericPipesBroadcast(
        broadcastFlatKernel,
        "Flat-Tree",
        config,
        rootRank,
        latencyUs,
        verified);
  }

  /**
   * Run Pipes Broadcast benchmark with clustered kernel launch.
   * Uses cudaClusterSchedulingPolicySpread to spread clusters across GPCs.
   * Returns bandwidth in GB/s and sets latency in microseconds.
   */
  float runPipesClusteredBroadcastBenchmark(
      const BroadcastBenchmarkConfig& config,
      int rootRank,
      float& latencyUs,
      bool& verified,
      int clusterSize = kDefaultClusterSize) {
    XLOGF(
        DBG1,
        "Rank {}: Running Pipes Clustered broadcast: {} (root={}, cluster={})",
        globalRank,
        config.name,
        rootRank,
        clusterSize);

    verified = false;
    DeviceBuffer buffer(config.messageSize);

    // Initialize buffer using helper
    if (initializeBroadcastBuffer(buffer, config.messageSize, rootRank) ==
        0.0f) {
      return 0.0f;
    }

    // Setup transports using helper
    auto transportSetup = setupPipesTransports(config);

    // Prepare kernel launch parameters
    dim3 gridDim(config.numBlocks);
    dim3 blockDim(config.numThreads);
    dim3 clusterDim(clusterSize, 1, 1);

    void* buff_d = buffer.get();
    std::size_t nbytes = config.messageSize;
    void* args[] = {
        &buff_d,
        &globalRank,
        &rootRank,
        &transportSetup.transports_span,
        &nbytes};

    CudaEvent start, stop;

    // Warmup with clustered launch
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    NVTX_RANGE_PUSH("Pipes_Clustered_Warmup");
    for (int i = 0; i < kWarmupIters; i++) {
      CUDA_CHECK(
          comms::common::launchKernel(
              (void*)broadcastFlatKernel,
              gridDim,
              blockDim,
              args,
              stream_,
              clusterDim));
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    NVTX_RANGE_POP();
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Benchmark with clustered launch
    NVTX_RANGE_PUSH("Pipes_Clustered_Benchmark");
    CUDA_CHECK(cudaEventRecord(start.get(), stream_));
    for (int i = 0; i < kBenchmarkIters; i++) {
      NVTX_RANGE_PUSH("Pipes_Clustered_Iter");
      CUDA_CHECK(
          comms::common::launchKernel(
              (void*)broadcastFlatKernel,
              gridDim,
              blockDim,
              args,
              stream_,
              clusterDim));
      NVTX_RANGE_POP();
    }
    CUDA_CHECK(cudaEventRecord(stop.get(), stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    NVTX_RANGE_POP();

    float totalTime_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime_ms, start.get(), stop.get()));

    // Calculate timing using helper
    auto timing = calculateBenchmarkTiming(totalTime_ms, config.messageSize);
    latencyUs = timing.latencyUs;

    // Verify data correctness if enabled
    if (FLAGS_verify_correctness) {
      // Re-initialize buffer for verification
      if (initializeBroadcastBuffer(buffer, config.messageSize, rootRank) ==
          0.0f) {
        return 0.0f;
      }

      buff_d = buffer.get();
      CUDA_CHECK(
          comms::common::launchKernel(
              (void*)broadcastFlatKernel,
              gridDim,
              blockDim,
              args,
              stream_,
              clusterDim));
      CUDA_CHECK(cudaStreamSynchronize(stream_));

      verified = verifyBroadcastData(
          buffer.get(), config.messageSize, rootRank, config.name, "Clustered");
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    return timing.bandwidthGBps;
  }

  // --------------------------------------------------------------------------
  // Unified Algorithm Runner
  // --------------------------------------------------------------------------

  /**
   * Unified runner that dispatches to the appropriate algorithm.
   * This allows parameterized testing across all algorithms.
   * Add new algorithm cases here as they are implemented.
   */
  float runPipesBroadcast(
      BroadcastAlgorithm algo,
      const BroadcastBenchmarkConfig& config,
      int rootRank,
      float& latencyUs,
      bool& verified) {
    switch (algo) {
      case BroadcastAlgorithm::FlatTree:
        return runPipesBroadcastFlatBenchmark(
            config, rootRank, latencyUs, verified);
      case BroadcastAlgorithm::BinomialTree:
        return runPipesBinomialTreeBenchmark(
            config, rootRank, latencyUs, verified);
      case BroadcastAlgorithm::Ring:
        return runPipesRingBenchmark(config, rootRank, latencyUs, verified);
        // case BroadcastAlgorithm::Adaptive - added in D91729719
    }
    return 0.0f;
  }

  /**
   * Run Pipes Broadcast benchmark with binomial tree algorithm.
   * Returns bandwidth in GB/s and sets latency in microseconds.
   */
  float runPipesBinomialTreeBenchmark(
      const BroadcastBenchmarkConfig& config,
      int rootRank,
      float& latencyUs,
      bool& verified) {
    XLOGF(
        DBG1,
        "Rank {}: Running Pipes Binomial Tree broadcast: {} (root={})",
        globalRank,
        config.name,
        rootRank);

    verified = false;
    DeviceBuffer buffer(config.messageSize);

    // Initialize buffer: root has data, others have zeros
    if (globalRank == rootRank) {
      auto h_data = generateExpectedData(config.messageSize, rootRank);
      CUDA_CHECK(cudaMemcpy(
          buffer.get(),
          h_data.data(),
          config.messageSize,
          cudaMemcpyHostToDevice));
    } else {
      CUDA_CHECK(cudaMemset(buffer.get(), 0, config.messageSize));
    }

    // Setup P2P NVL transport
    MultiPeerNvlTransportConfig nvlConfig{
        .dataBufferSize = config.stagingBufferSize,
        .chunkSize = config.chunkSize,
        .pipelineDepth = config.pipelineDepth,
    };

    auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
    MultiPeerNvlTransport transport(globalRank, numRanks, bootstrap, nvlConfig);
    transport.exchange();

    // Create transport array: self for my rank, P2P for others
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

    // Copy transports to device
    DeviceBuffer d_transports(sizeof(Transport) * numRanks);
    CUDA_CHECK(cudaMemcpy(
        d_transports.get(),
        h_transports.data(),
        sizeof(Transport) * numRanks,
        cudaMemcpyHostToDevice));

    DeviceSpan<Transport> transports_span(
        static_cast<Transport*>(d_transports.get()), numRanks);

    // Prepare kernel launch parameters
    dim3 gridDim(config.numBlocks);
    dim3 blockDim(config.numThreads);

    void* buff_d = buffer.get();
    std::size_t nbytes = config.messageSize;
    void* args[] = {&buff_d, &globalRank, &rootRank, &transports_span, &nbytes};

    CudaEvent start, stop;

    // Warmup
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    NVTX_RANGE_PUSH("Pipes_BinomialTree_Warmup");
    for (int i = 0; i < kWarmupIters; i++) {
      CUDA_CHECK(cudaLaunchKernel(
          (void*)broadcastBinomialTreeKernel,
          gridDim,
          blockDim,
          args,
          0,
          stream_));
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    NVTX_RANGE_POP();
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Benchmark
    NVTX_RANGE_PUSH("Pipes_BinomialTree_Benchmark");
    CUDA_CHECK(cudaEventRecord(start.get(), stream_));
    for (int i = 0; i < kBenchmarkIters; i++) {
      NVTX_RANGE_PUSH("Pipes_BinomialTree_Iter");
      CUDA_CHECK(cudaLaunchKernel(
          (void*)broadcastBinomialTreeKernel,
          gridDim,
          blockDim,
          args,
          0,
          stream_));
      NVTX_RANGE_POP();
    }
    CUDA_CHECK(cudaEventRecord(stop.get(), stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    NVTX_RANGE_POP();

    float totalTime_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime_ms, start.get(), stop.get()));
    float avgTime_ms = totalTime_ms / kBenchmarkIters;
    latencyUs = avgTime_ms * 1000.0f;

    // Broadcast bandwidth: data transferred / time
    float bandwidth_GBps = (config.messageSize / 1e9f) / (avgTime_ms / 1000.0f);

    // Verify data correctness if enabled
    if (FLAGS_verify_correctness) {
      // Re-initialize and run one more broadcast for verification
      if (globalRank == rootRank) {
        auto h_data = generateExpectedData(config.messageSize, rootRank);
        cudaMemcpy(
            buffer.get(),
            h_data.data(),
            config.messageSize,
            cudaMemcpyHostToDevice);
      } else {
        cudaMemset(buffer.get(), 0, config.messageSize);
      }

      buff_d = buffer.get();
      cudaLaunchKernel(
          (void*)broadcastBinomialTreeKernel,
          gridDim,
          blockDim,
          args,
          0,
          stream_);
      cudaStreamSynchronize(stream_);

      verified = verifyBroadcastData(
          buffer.get(),
          config.messageSize,
          rootRank,
          config.name,
          "BinomialTree");
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    return bandwidth_GBps;
  }

  /**
   * Run Pipes Broadcast benchmark with ring algorithm.
   * Returns bandwidth in GB/s and sets latency in microseconds.
   */
  float runPipesRingBenchmark(
      const BroadcastBenchmarkConfig& config,
      int rootRank,
      float& latencyUs,
      bool& verified) {
    XLOGF(
        DBG1,
        "Rank {}: Running Pipes Ring broadcast: {} (root={})",
        globalRank,
        config.name,
        rootRank);

    verified = false;
    DeviceBuffer buffer(config.messageSize);

    // Initialize buffer: root has data, others have zeros
    if (globalRank == rootRank) {
      auto h_data = generateExpectedData(config.messageSize, rootRank);
      CUDA_CHECK(cudaMemcpy(
          buffer.get(),
          h_data.data(),
          config.messageSize,
          cudaMemcpyHostToDevice));
    } else {
      CUDA_CHECK(cudaMemset(buffer.get(), 0, config.messageSize));
    }

    // Setup P2P NVL transport
    MultiPeerNvlTransportConfig nvlConfig{
        .dataBufferSize = config.stagingBufferSize,
        .chunkSize = config.chunkSize,
        .pipelineDepth = config.pipelineDepth,
    };

    auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
    MultiPeerNvlTransport transport(globalRank, numRanks, bootstrap, nvlConfig);
    transport.exchange();

    // Create transport array: self for my rank, P2P for others
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

    // Copy transports to device
    DeviceBuffer d_transports(sizeof(Transport) * numRanks);
    CUDA_CHECK(cudaMemcpy(
        d_transports.get(),
        h_transports.data(),
        sizeof(Transport) * numRanks,
        cudaMemcpyHostToDevice));

    DeviceSpan<Transport> transports_span(
        static_cast<Transport*>(d_transports.get()), numRanks);

    // Prepare kernel launch parameters
    dim3 gridDim(config.numBlocks);
    dim3 blockDim(config.numThreads);

    void* buff_d = buffer.get();
    std::size_t nbytes = config.messageSize;
    void* args[] = {&buff_d, &globalRank, &rootRank, &transports_span, &nbytes};

    CudaEvent start, stop;

    // Warmup
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    NVTX_RANGE_PUSH("Pipes_Ring_Warmup");
    for (int i = 0; i < kWarmupIters; i++) {
      CUDA_CHECK(cudaLaunchKernel(
          (void*)broadcastRingKernel, gridDim, blockDim, args, 0, stream_));
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    NVTX_RANGE_POP();
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Benchmark
    NVTX_RANGE_PUSH("Pipes_Ring_Benchmark");
    CUDA_CHECK(cudaEventRecord(start.get(), stream_));
    for (int i = 0; i < kBenchmarkIters; i++) {
      NVTX_RANGE_PUSH("Pipes_Ring_Iter");
      CUDA_CHECK(cudaLaunchKernel(
          (void*)broadcastRingKernel, gridDim, blockDim, args, 0, stream_));
      NVTX_RANGE_POP();
    }
    CUDA_CHECK(cudaEventRecord(stop.get(), stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    NVTX_RANGE_POP();

    float totalTime_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime_ms, start.get(), stop.get()));
    float avgTime_ms = totalTime_ms / kBenchmarkIters;
    latencyUs = avgTime_ms * 1000.0f;

    // Broadcast bandwidth: data transferred / time
    float bandwidth_GBps = (config.messageSize / 1e9f) / (avgTime_ms / 1000.0f);

    // Verify data correctness if enabled
    if (FLAGS_verify_correctness) {
      // Re-initialize and run one more broadcast for verification
      if (globalRank == rootRank) {
        auto h_data = generateExpectedData(config.messageSize, rootRank);
        cudaMemcpy(
            buffer.get(),
            h_data.data(),
            config.messageSize,
            cudaMemcpyHostToDevice);
      } else {
        cudaMemset(buffer.get(), 0, config.messageSize);
      }

      buff_d = buffer.get();
      cudaLaunchKernel(
          (void*)broadcastRingKernel, gridDim, blockDim, args, 0, stream_);
      cudaStreamSynchronize(stream_);

      verified = verifyBroadcastData(
          buffer.get(), config.messageSize, rootRank, config.name, "Ring");
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    return bandwidth_GBps;
  }

  // --------------------------------------------------------------------------
  // Result Formatting and Output Utilities
  // --------------------------------------------------------------------------

  void printResultsTable(const std::vector<BroadcastBenchmarkResult>& results) {
    if (globalRank != 0) {
      return;
    }

    std::stringstream ss;
    ss << "\n";
    ss << "======================================================================================================================";
    if (FLAGS_verify_correctness) {
      ss << "================";
    }
    ss << "\n";
    ss << "                         NCCL vs Pipes Broadcast Benchmark Results";
    if (FLAGS_verify_correctness) {
      ss << " (with verification)";
    }
    ss << "\n";
    ss << "======================================================================================================================";
    if (FLAGS_verify_correctness) {
      ss << "================";
    }
    ss << "\n";
    ss << std::left << std::setw(22) << "Test Name" << std::right
       << std::setw(10) << "Size" << std::right << std::setw(8) << "Root"
       << std::right << std::setw(8) << "Ranks" << std::right << std::setw(12)
       << "NCCL BW" << std::right << std::setw(12) << "Pipes BW" << std::right
       << std::setw(10) << "Speedup" << std::right << std::setw(12)
       << "NCCL Lat" << std::right << std::setw(12) << "Pipes Lat" << std::right
       << std::setw(12) << "Lat Reduc";
    if (FLAGS_verify_correctness) {
      ss << std::right << std::setw(8) << "NCCL OK" << std::right
         << std::setw(8) << "Pipes OK";
    }
    ss << "\n";
    ss << std::left << std::setw(22) << "" << std::right << std::setw(10) << ""
       << std::right << std::setw(8) << "" << std::right << std::setw(8) << ""
       << std::right << std::setw(12) << "(GB/s)" << std::right << std::setw(12)
       << "(GB/s)" << std::right << std::setw(10) << "Pipes/NCCL" << std::right
       << std::setw(12) << "(us)" << std::right << std::setw(12) << "(us)"
       << std::right << std::setw(12) << "(us)";
    if (FLAGS_verify_correctness) {
      ss << std::right << std::setw(8) << "" << std::right << std::setw(8)
         << "";
    }
    ss << "\n";
    ss << "----------------------------------------------------------------------------------------------------------------------";
    if (FLAGS_verify_correctness) {
      ss << "----------------";
    }
    ss << "\n";

    for (const auto& r : results) {
      std::string msgSize = formatSize(r.messageSize);
      float latencyReduction = r.ncclLatency - r.pipesLatency;

      ss << std::left << std::setw(22) << r.testName << std::right
         << std::setw(10) << msgSize << std::right << std::setw(8) << r.rootRank
         << std::right << std::setw(8) << r.nRanks << std::right
         << std::setw(12) << std::fixed << std::setprecision(2)
         << r.ncclBandwidth << std::right << std::setw(12) << std::fixed
         << std::setprecision(2) << r.pipesBandwidth << std::right
         << std::setw(9) << std::fixed << std::setprecision(2) << r.speedup
         << "x" << std::right << std::setw(12) << std::fixed
         << std::setprecision(1) << r.ncclLatency << std::right << std::setw(12)
         << std::fixed << std::setprecision(1) << r.pipesLatency << std::right
         << std::setw(12) << std::fixed << std::setprecision(1)
         << latencyReduction;
      if (FLAGS_verify_correctness) {
        ss << std::right << std::setw(8) << (r.ncclVerified ? "PASS" : "FAIL")
           << std::right << std::setw(8) << (r.pipesVerified ? "PASS" : "FAIL");
      }
      ss << "\n";
    }

    ss << "======================================================================================================================";
    if (FLAGS_verify_correctness) {
      ss << "================";
    }
    ss << "\n";

    XLOG(INFO) << ss.str();

    // Print legend separately to avoid MPI buffer splitting issues
    XLOG(INFO) << "BW (Bandwidth) = Message size / time, in GB/s";
    XLOG(INFO) << "Lat (Latency) = Average transfer time per iteration, in us";
    XLOG(INFO)
        << "Lat Reduc = NCCL latency - Pipes latency (positive = Pipes faster)";
    XLOG(INFO) << "Speedup = Pipes Bandwidth / NCCL Bandwidth";
    if (FLAGS_verify_correctness) {
      XLOG(INFO) << "NCCL OK / Pipes OK = Data correctness verification result";
      XLOG(INFO)
          << "======================================================================================================================================";
    } else {
      XLOG(INFO)
          << "======================================================================================================================";
    }
  }

  std::string formatSize(std::size_t bytes) {
    std::stringstream ss;
    if (bytes >= 1024 * 1024 * 1024) {
      ss << std::fixed << std::setprecision(0)
         << (bytes / (1024.0 * 1024.0 * 1024.0)) << "GB";
    } else if (bytes >= 1024 * 1024) {
      ss << std::fixed << std::setprecision(0) << (bytes / (1024.0 * 1024.0))
         << "MB";
    } else if (bytes >= 1024) {
      ss << std::fixed << std::setprecision(0) << (bytes / 1024.0) << "KB";
    } else {
      ss << bytes << "B";
    }
    return ss.str();
  }

  ncclComm_t ncclComm_{};
  cudaStream_t stream_{};
};

// ============================================================================
// SECTION 3: TEST CASES
// ============================================================================

/**
 * Clustered Launch Comparison Benchmark
 *
 * Compares standard kernel launch vs clustered kernel launch with
 * cudaClusterSchedulingPolicySpread across various message sizes.
 *
 * Clustered launches group thread blocks into clusters that are scheduled
 * together on the same GPC, which can reduce L2 cache thrashing and improve
 * memory access patterns.
 */
TEST_F(BroadcastBenchmarkFixture, ClusteredLaunchComparison) {
  if (globalRank == 0) {
    XLOG(INFO) << "\n=== Clustered Launch vs Standard Launch Comparison ===\n";
    XLOG(INFO) << "Comparing cudaLaunchKernel vs cudaLaunchKernelExC with "
                  "cudaClusterSchedulingPolicySpread\n";
  }

  // Test configurations across message sizes
  std::vector<BroadcastBenchmarkConfig> configs = {
      // Small message
      {
          .messageSize = 64 * 1024,
          .stagingBufferSize = 64 * 1024,
          .pipelineDepth = 4,
          .chunkSize = 16 * 1024,
          .numBlocks = 8,
          .numThreads = 256,
          .rootRank = 0,
          .name = "64KB",
      },
      // Medium message
      {
          .messageSize = 1 * 1024 * 1024,
          .stagingBufferSize = 16 * 1024 * 1024,
          .pipelineDepth = 4,
          .chunkSize = 128 * 1024,
          .numBlocks = 32,
          .numThreads = 512,
          .rootRank = 0,
          .name = "1MB",
      },
      // Large message
      {
          .messageSize = 8 * 1024 * 1024,
          .stagingBufferSize = 16 * 1024 * 1024,
          .pipelineDepth = 4,
          .chunkSize = 128 * 1024,
          .numBlocks = 32,
          .numThreads = 512,
          .rootRank = 0,
          .name = "8MB",
      },
      // Very large message
      {
          .messageSize = 64 * 1024 * 1024,
          .stagingBufferSize = 64 * 1024 * 1024,
          .pipelineDepth = 4,
          .chunkSize = 128 * 1024,
          .numBlocks = 32,
          .numThreads = 512,
          .rootRank = 0,
          .name = "64MB",
      },
  };

  // Print header
  if (globalRank == 0) {
    std::stringstream ss;
    ss << "\n";
    ss << "========================================================================"
          "================================================\n";
    ss << "     Standard vs Clustered Launch Comparison (Flat-Tree Broadcast)\n";
    ss << "========================================================================"
          "================================================\n";
    ss << std::left << std::setw(12) << "MsgSize" << std::right << std::setw(12)
       << "NCCL BW" << std::right << std::setw(14) << "Standard BW"
       << std::right << std::setw(14) << "Clustered BW" << std::right
       << std::setw(12) << "Speedup" << std::right << std::setw(14)
       << "Std/NCCL" << std::right << std::setw(14) << "Clust/NCCL" << "\n";
    ss << std::left << std::setw(12) << "" << std::right << std::setw(12)
       << "(GB/s)" << std::right << std::setw(14) << "(GB/s)" << std::right
       << std::setw(14) << "(GB/s)" << std::right << std::setw(12)
       << "Clust/Std" << std::right << std::setw(14) << "" << std::right
       << std::setw(14) << "" << "\n";
    ss << "------------------------------------------------------------------------"
          "------------------------------------------------\n";
    XLOG(INFO) << ss.str();
  }

  for (const auto& config : configs) {
    int rootRank = config.rootRank;

    // NCCL baseline
    float ncclLatencyUs = 0.0f;
    bool ncclVerified = false;
    float ncclBandwidth = runNcclBroadcastBenchmark(
        config, rootRank, ncclLatencyUs, ncclVerified);

    // Standard launch
    float stdLatencyUs = 0.0f;
    bool stdVerified = false;
    float stdBandwidth = runPipesBroadcastFlatBenchmark(
        config, rootRank, stdLatencyUs, stdVerified);

    // Clustered launch (cluster size = kDefaultClusterSize)
    float clusteredLatencyUs = 0.0f;
    bool clusteredVerified = false;
    float clusteredBandwidth = runPipesClusteredBroadcastBenchmark(
        config,
        rootRank,
        clusteredLatencyUs,
        clusteredVerified,
        kDefaultClusterSize);

    if (globalRank == 0) {
      float clusterSpeedup =
          (stdBandwidth > 0) ? clusteredBandwidth / stdBandwidth : 0;
      float stdVsNccl = (ncclBandwidth > 0) ? stdBandwidth / ncclBandwidth : 0;
      float clustVsNccl =
          (ncclBandwidth > 0) ? clusteredBandwidth / ncclBandwidth : 0;

      std::stringstream ss;
      ss << std::left << std::setw(12) << config.name << std::right
         << std::setw(12) << std::fixed << std::setprecision(2) << ncclBandwidth
         << std::right << std::setw(14) << std::fixed << std::setprecision(2)
         << stdBandwidth << std::right << std::setw(14) << std::fixed
         << std::setprecision(2) << clusteredBandwidth << std::right
         << std::setw(11) << std::fixed << std::setprecision(2)
         << clusterSpeedup << "x" << std::right << std::setw(13) << std::fixed
         << std::setprecision(2) << stdVsNccl << "x" << std::right
         << std::setw(13) << std::fixed << std::setprecision(2) << clustVsNccl
         << "x";
      XLOG(INFO) << ss.str();
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  }

  if (globalRank == 0) {
    XLOG(INFO)
        << "========================================================================"
           "================================================";
    XLOG(INFO)
        << "Speedup > 1.0x indicates clustered launch is faster than standard";
    XLOG(INFO)
        << "========================================================================"
           "================================================\n";
  }
}

/**
 * Root Rank Sweep Benchmark
 *
 * Tests broadcast performance with each rank as root to identify
 * any topology-dependent performance variations.
 */
TEST_F(BroadcastBenchmarkFixture, RootRankSweep) {
  if (globalRank == 0) {
    XLOG(INFO) << "\n=== Root Rank Sweep Benchmark (All Algorithms) ===\n";
    XLOG(INFO) << "Testing broadcast with each rank as root to identify "
                  "topology-dependent performance variations.\n";
  }

  // Use a fixed medium message size for this sweep
  BroadcastBenchmarkConfig baseConfig{
      .messageSize = 4 * 1024 * 1024, // 4MB
      .stagingBufferSize = 16 * 1024 * 1024,
      .pipelineDepth = 4,
      .chunkSize = 128 * 1024,
      .numBlocks = 32,
      .numThreads = 512,
      .rootRank = 0, // Will be overridden
      .name = "4MB",
  };

  // Print header
  if (globalRank == 0) {
    std::stringstream ss;
    ss << "\n";
    ss << "========================================================================"
          "======================================\n";
    ss << "                         Root Rank Sweep (4MB Message, All Algorithms)\n";
    ss << "========================================================================"
          "======================================\n";
    ss << std::left << std::setw(8) << "Root" << std::left << std::setw(12)
       << "Algorithm" << std::right << std::setw(12) << "NCCL BW" << std::right
       << std::setw(12) << "Pipes BW" << std::right << std::setw(12)
       << "Speedup" << std::right << std::setw(12) << "NCCL Lat" << std::right
       << std::setw(12) << "Pipes Lat" << std::right << std::setw(14)
       << "Lat Diff" << "\n";
    ss << std::left << std::setw(8) << "Rank" << std::left << std::setw(12)
       << "" << std::right << std::setw(12) << "(GB/s)" << std::right
       << std::setw(12) << "(GB/s)" << std::right << std::setw(12)
       << "Pipes/NCCL" << std::right << std::setw(12) << "(us)" << std::right
       << std::setw(12) << "(us)" << std::right << std::setw(14) << "(us)"
       << "\n";
    ss << "------------------------------------------------------------------------"
          "--------------------------------------\n";
    XLOG(INFO) << ss.str();
  }

  // Sweep through all root ranks
  for (int rootRank = 0; rootRank < numRanks; rootRank++) {
    BroadcastBenchmarkConfig config = baseConfig;
    config.rootRank = rootRank;
    config.name = "Root" + std::to_string(rootRank);

    // Run NCCL once per root rank (baseline)
    float ncclLatencyUs = 0.0f;
    bool ncclVerified = false;
    float ncclBandwidth = runNcclBroadcastBenchmark(
        config, rootRank, ncclLatencyUs, ncclVerified);

    // Run each algorithm
    for (auto algo : kAllAlgorithms) {
      float pipesLatencyUs = 0.0f;
      bool pipesVerified = false;
      float pipesBandwidth = runPipesBroadcast(
          algo, config, rootRank, pipesLatencyUs, pipesVerified);

      if (globalRank == 0) {
        float speedup =
            (ncclBandwidth > 0) ? pipesBandwidth / ncclBandwidth : 0;
        float latDiff = ncclLatencyUs - pipesLatencyUs;

        std::stringstream ss;
        ss << std::left << std::setw(8) << rootRank << std::left
           << std::setw(12) << algorithmName(algo) << std::right
           << std::setw(12) << std::fixed << std::setprecision(2)
           << ncclBandwidth << std::right << std::setw(12) << std::fixed
           << std::setprecision(2) << pipesBandwidth << std::right
           << std::setw(11) << std::fixed << std::setprecision(2) << speedup
           << "x" << std::right << std::setw(12) << std::fixed
           << std::setprecision(1) << ncclLatencyUs << std::right
           << std::setw(12) << std::fixed << std::setprecision(1)
           << pipesLatencyUs << std::right << std::setw(14) << std::fixed
           << std::setprecision(1) << latDiff;
        XLOG(INFO) << ss.str();
      }

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    // Print separator between root ranks
    if (globalRank == 0 && rootRank < numRanks - 1) {
      XLOG(INFO)
          << "------------------------------------------------------------------------"
             "--------------------------------------";
    }
  }

  if (globalRank == 0) {
    XLOG(INFO)
        << "========================================================================"
           "======================================";
    XLOG(INFO) << "Speedup > 1.0x indicates Pipes is faster than NCCL for that "
                  "algorithm/root combination";
    XLOG(INFO)
        << "========================================================================"
           "======================================\n";
  }
}

/**
 * Extended Message Sizes Benchmark
 *
 * Tests a comprehensive range of message sizes from very small (64B)
 * to very large (256MB) to identify optimal message size ranges.
 */
TEST_F(BroadcastBenchmarkFixture, ExtendedMessageSizes) {
  if (globalRank == 0) {
    XLOG(INFO)
        << "\n=== Extended Message Sizes Benchmark (All Algorithms) ===\n";
    XLOG(INFO) << "Testing message sizes from 64B to 256MB.\n";
  }

  // Extended message size sweep
  std::vector<std::size_t> messageSizes = {
      64, // 64 bytes - latency bound
      256, // 256 bytes
      1 * 1024, // 1KB
      4 * 1024, // 4KB
      16 * 1024, // 16KB
      64 * 1024, // 64KB
      256 * 1024, // 256KB
      512 * 1024, // 512KB
      1 * 1024 * 1024, // 1MB
      2 * 1024 * 1024, // 2MB
      4 * 1024 * 1024, // 4MB
      8 * 1024 * 1024, // 8MB
      16 * 1024 * 1024, // 16MB
      32 * 1024 * 1024, // 32MB
      64 * 1024 * 1024, // 64MB
      128 * 1024 * 1024, // 128MB
      256 * 1024 * 1024, // 256MB
  };

  // Print header
  if (globalRank == 0) {
    std::stringstream ss;
    ss << "\n";
    ss << "========================================================================"
          "============================================\n";
    ss << "                  Extended Message Size Sweep (All Algorithms)\n";
    ss << "========================================================================"
          "============================================\n";
    ss << std::left << std::setw(10) << "MsgSize" << std::left << std::setw(12)
       << "Algorithm" << std::right << std::setw(12) << "NCCL BW" << std::right
       << std::setw(12) << "Pipes BW" << std::right << std::setw(12)
       << "Speedup" << std::right << std::setw(12) << "NCCL Lat" << std::right
       << std::setw(12) << "Pipes Lat" << std::right << std::setw(12) << "Chunk"
       << std::right << std::setw(10) << "Blocks" << "\n";
    ss << std::left << std::setw(10) << "" << std::left << std::setw(12) << ""
       << std::right << std::setw(12) << "(GB/s)" << std::right << std::setw(12)
       << "(GB/s)" << std::right << std::setw(12) << "Pipes/NCCL" << std::right
       << std::setw(12) << "(us)" << std::right << std::setw(12) << "(us)"
       << std::right << std::setw(12) << "Size" << std::right << std::setw(10)
       << "" << "\n";
    ss << "------------------------------------------------------------------------"
          "--------------------------------------------\n";
    XLOG(INFO) << ss.str();
  }

  for (std::size_t msgSize : messageSizes) {
    // Configure based on message size
    BroadcastBenchmarkConfig config;
    config.messageSize = msgSize;
    config.rootRank = 0;

    // Auto-tune configuration based on message size
    if (msgSize < kSmallMessageThreshold) {
      config.chunkSize = kSmallMsgChunkSize;
      config.stagingBufferSize = kSmallMsgStagingBuffer;
      config.numBlocks = kSmallMsgNumBlocks;
      config.numThreads = kSmallMsgNumThreads;
    } else if (msgSize < kMediumMessageThreshold) {
      config.chunkSize = kMediumMsgChunkSize;
      config.stagingBufferSize = kMediumMsgStagingBuffer;
      config.numBlocks = kMediumMsgNumBlocks;
      config.numThreads = kMediumMsgNumThreads;
    } else if (msgSize < kLargeMessageThreshold) {
      config.chunkSize = kLargeMsgChunkSize;
      config.stagingBufferSize = kLargeMsgStagingBuffer;
      config.numBlocks = kLargeMsgNumBlocks;
      config.numThreads = kLargeMsgNumThreads;
    } else {
      config.chunkSize = kVeryLargeMsgChunkSize;
      config.stagingBufferSize = std::min(msgSize, kMaxStagingBufferSize);
      config.numBlocks = kVeryLargeMsgNumBlocks;
      config.numThreads = kVeryLargeMsgNumThreads;
    }
    config.pipelineDepth = 4;
    config.name = formatSize(msgSize);

    // Run NCCL once per message size (baseline)
    float ncclLatencyUs = 0.0f;
    bool ncclVerified = false;
    float ncclBandwidth = runNcclBroadcastBenchmark(
        config, config.rootRank, ncclLatencyUs, ncclVerified);

    // Run each algorithm
    for (auto algo : kAllAlgorithms) {
      float pipesLatencyUs = 0.0f;
      bool pipesVerified = false;
      float pipesBandwidth = runPipesBroadcast(
          algo, config, config.rootRank, pipesLatencyUs, pipesVerified);

      if (globalRank == 0) {
        float speedup =
            (ncclBandwidth > 0) ? pipesBandwidth / ncclBandwidth : 0;

        std::stringstream ss;
        ss << std::left << std::setw(10) << formatSize(msgSize) << std::left
           << std::setw(12) << algorithmName(algo) << std::right
           << std::setw(12) << std::fixed << std::setprecision(2)
           << ncclBandwidth << std::right << std::setw(12) << std::fixed
           << std::setprecision(2) << pipesBandwidth << std::right
           << std::setw(11) << std::fixed << std::setprecision(2) << speedup
           << "x" << std::right << std::setw(12) << std::fixed
           << std::setprecision(1) << ncclLatencyUs << std::right
           << std::setw(12) << std::fixed << std::setprecision(1)
           << pipesLatencyUs << std::right << std::setw(12)
           << formatSize(config.chunkSize) << std::right << std::setw(10)
           << config.numBlocks;
        XLOG(INFO) << ss.str();
      }

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    // Print separator between message sizes
    if (globalRank == 0 && msgSize != messageSizes.back()) {
      XLOG(INFO)
          << "------------------------------------------------------------------------"
             "--------------------------------------------";
    }
  }

  if (globalRank == 0) {
    XLOG(INFO)
        << "========================================================================"
           "============================================";
    XLOG(INFO)
        << "Speedup > 1.0x indicates Pipes is faster than NCCL at that size";
    XLOG(INFO)
        << "========================================================================"
           "============================================\n";
  }
}

/**
 * Grid Configuration Sweep Benchmark
 *
 * Tests various block count and thread count configurations to identify
 * optimal launch parameters for different message sizes.
 */
TEST_F(BroadcastBenchmarkFixture, GridConfigSweep) {
  if (globalRank == 0) {
    XLOG(INFO) << "\n=== Grid Configuration Sweep ===\n";
    XLOG(INFO) << "Testing various block/thread configurations for 16MB "
                  "message.\n";
  }

  const std::size_t msgSize = 16 * 1024 * 1024; // 16MB

  std::vector<GridConfig> gridConfigs = {
      {4, 128, "4x128"},
      {4, 256, "4x256"},
      {4, 512, "4x512"},
      {8, 128, "8x128"},
      {8, 256, "8x256"},
      {8, 512, "8x512"},
      {16, 128, "16x128"},
      {16, 256, "16x256"},
      {16, 512, "16x512"},
      {32, 128, "32x128"},
      {32, 256, "32x256"},
      {32, 512, "32x512"},
      {64, 256, "64x256"},
      {64, 512, "64x512"},
  };

  // Get NCCL baseline
  BroadcastBenchmarkConfig ncclConfig{
      .messageSize = msgSize,
      .stagingBufferSize = 16 * 1024 * 1024,
      .pipelineDepth = 4,
      .chunkSize = 128 * 1024,
      .numBlocks = 32,
      .numThreads = 512,
      .rootRank = 0,
      .name = "baseline",
  };

  float ncclLatencyUs = 0.0f;
  bool ncclVerified = false;
  float ncclBandwidth =
      runNcclBroadcastBenchmark(ncclConfig, 0, ncclLatencyUs, ncclVerified);

  // Print header
  if (globalRank == 0) {
    std::stringstream ss;
    ss << "Message Size: " << formatSize(msgSize) << "\n";
    ss << "NCCL Baseline: " << std::fixed << std::setprecision(2)
       << ncclBandwidth << " GB/s\n\n";
    ss << std::left << std::setw(12) << "Config" << std::right << std::setw(10)
       << "Blocks" << std::right << std::setw(10) << "Threads" << std::right
       << std::setw(10) << "Total" << std::right << std::setw(14) << "Algorithm"
       << std::right << std::setw(12) << "Bandwidth" << std::right
       << std::setw(12) << "vs NCCL" << "\n";
    ss << std::left << std::setw(12) << "(BxT)" << std::right << std::setw(10)
       << "" << std::right << std::setw(10) << "" << std::right << std::setw(10)
       << "Threads" << std::right << std::setw(14) << "" << std::right
       << std::setw(12) << "(GB/s)" << std::right << std::setw(12) << ""
       << "\n";
    ss << "--------------------------------------------------------------------------------\n";
    XLOG(INFO) << ss.str();
  }

  float bestBandwidth = 0.0f;
  std::string bestConfig;
  std::string bestAlgorithm;

  for (const auto& grid : gridConfigs) {
    BroadcastBenchmarkConfig config{
        .messageSize = msgSize,
        .stagingBufferSize = 16 * 1024 * 1024,
        .pipelineDepth = 4,
        .chunkSize = 128 * 1024,
        .numBlocks = grid.numBlocks,
        .numThreads = grid.numThreads,
        .rootRank = 0,
        .name = grid.name,
    };

    for (auto algo : kAllAlgorithms) {
      float pipesLatencyUs = 0.0f;
      bool pipesVerified = false;
      float pipesBandwidth =
          runPipesBroadcast(algo, config, 0, pipesLatencyUs, pipesVerified);

      if (globalRank == 0) {
        float speedup =
            (ncclBandwidth > 0) ? pipesBandwidth / ncclBandwidth : 0;
        int totalThreads = grid.numBlocks * grid.numThreads;

        std::stringstream ss;
        ss << std::left << std::setw(12) << grid.name << std::right
           << std::setw(10) << grid.numBlocks << std::right << std::setw(10)
           << grid.numThreads << std::right << std::setw(10) << totalThreads
           << std::right << std::setw(14) << algorithmName(algo) << std::right
           << std::setw(12) << std::fixed << std::setprecision(2)
           << pipesBandwidth << std::right << std::setw(11) << std::fixed
           << std::setprecision(2) << speedup << "x";
        XLOG(INFO) << ss.str();

        if (pipesBandwidth > bestBandwidth) {
          bestBandwidth = pipesBandwidth;
          bestConfig = grid.name;
          bestAlgorithm = algorithmName(algo);
        }
      }

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    // Print separator between grid configs
    if (globalRank == 0) {
      XLOG(INFO) << "  ---";
    }
  }

  if (globalRank == 0) {
    XLOG(INFO)
        << "--------------------------------------------------------------------------------";
    XLOG(INFO) << "Best config: " << bestConfig << " (" << bestAlgorithm
               << ") at " << std::fixed << std::setprecision(2) << bestBandwidth
               << " GB/s (" << std::fixed << std::setprecision(2)
               << (bestBandwidth / ncclBandwidth) << "x of NCCL)";
    XLOG(INFO)
        << "================================================================================\n";
  }
}

/**
 * Staging Buffer Tuning Benchmark
 *
 * Sweeps staging buffer sizes and chunk sizes to find optimal
 * configuration for different message sizes.
 */
TEST_F(BroadcastBenchmarkFixture, StagingBufferTuning) {
  if (globalRank == 0) {
    XLOG(INFO)
        << "\n=== Staging Buffer and Chunk Size Tuning (All Algorithms) ===\n";
    XLOG(INFO) << "This test sweeps parameters across all algorithms.\n";
  }

  // Fixed message size for parameter sweep
  const std::size_t messageSize = 64 * 1024 * 1024; // 64MB

  // Parameter sweep configurations
  struct TuningConfig {
    std::size_t stagingBufferSize;
    std::size_t chunkSize;
    int numBlocks;
    std::string name;
  };

  std::vector<TuningConfig> tuningConfigs = {
      // Fine-grained chunk size sweep (smaller chunks = better!)
      // These use fixed 16MB staging buffer and 32 blocks
      {16 * 1024 * 1024, 16 * 1024, 32, "Chunk16KB"},
      {16 * 1024 * 1024, 32 * 1024, 32, "Chunk32KB"},
      {16 * 1024 * 1024, 64 * 1024, 32, "Chunk64KB"},
      {16 * 1024 * 1024, 128 * 1024, 32, "Chunk128KB"},
      {16 * 1024 * 1024, 256 * 1024, 32, "Chunk256KB"},
      {16 * 1024 * 1024, 512 * 1024, 32, "Chunk512KB"},
      {16 * 1024 * 1024, 1024 * 1024, 32, "Chunk1MB"},

      // Thread block sweep with optimal chunk size (128KB)
      {16 * 1024 * 1024, 128 * 1024, 8, "128KB_8blk"},
      {16 * 1024 * 1024, 128 * 1024, 16, "128KB_16blk"},
      {16 * 1024 * 1024, 128 * 1024, 32, "128KB_32blk"},
      {16 * 1024 * 1024, 128 * 1024, 64, "128KB_64blk"},
      {16 * 1024 * 1024, 128 * 1024, 128, "128KB_128blk"},

      // Staging buffer sweep with optimal chunk size (128KB)
      {4 * 1024 * 1024, 128 * 1024, 32, "4MB_128KB"},
      {8 * 1024 * 1024, 128 * 1024, 32, "8MB_128KB"},
      {16 * 1024 * 1024, 128 * 1024, 32, "16MB_128KB"},
      {32 * 1024 * 1024, 128 * 1024, 32, "32MB_128KB"},
  };

  // Get NCCL baseline
  BroadcastBenchmarkConfig ncclConfig{
      .messageSize = messageSize,
      .stagingBufferSize = 16 * 1024 * 1024,
      .pipelineDepth = 4,
      .chunkSize = 1024 * 1024,
      .numBlocks = 32,
      .numThreads = 512,
      .rootRank = 0,
      .name = "NCCL_baseline",
  };

  float ncclLatencyUs = 0.0f;
  bool ncclVerified = false;
  float ncclBandwidth =
      runNcclBroadcastBenchmark(ncclConfig, 0, ncclLatencyUs, ncclVerified);

  if (globalRank == 0) {
    std::stringstream ss;
    ss << "\n";
    ss << "Message Size: " << formatSize(messageSize) << "\n";
    ss << "NCCL Baseline: " << std::fixed << std::setprecision(2)
       << ncclBandwidth << " GB/s\n\n";
    ss << std::left << std::setw(18) << "Config" << std::left << std::setw(12)
       << "Algorithm" << std::right << std::setw(12) << "Staging" << std::right
       << std::setw(12) << "Chunk" << std::right << std::setw(10) << "Blocks"
       << std::right << std::setw(12) << "Bandwidth" << std::right
       << std::setw(12) << "vs NCCL" << "\n";
    ss << std::left << std::setw(18) << "" << std::left << std::setw(12) << ""
       << std::right << std::setw(12) << "Size" << std::right << std::setw(12)
       << "Size" << std::right << std::setw(10) << "" << std::right
       << std::setw(12) << "(GB/s)" << std::right << std::setw(12) << ""
       << "\n";
    ss << "------------------------------------------------------------------------"
          "--------------------\n";
    XLOG(INFO) << ss.str();
  }

  for (const auto& tuning : tuningConfigs) {
    BroadcastBenchmarkConfig config{
        .messageSize = messageSize,
        .stagingBufferSize = tuning.stagingBufferSize,
        .pipelineDepth = 4,
        .chunkSize = tuning.chunkSize,
        .numBlocks = tuning.numBlocks,
        .numThreads = 512,
        .rootRank = 0,
        .name = tuning.name,
    };

    // Run each algorithm
    for (auto algo : kAllAlgorithms) {
      float pipesLatencyUs = 0.0f;
      bool pipesVerified = false;
      float pipesBandwidth =
          runPipesBroadcast(algo, config, 0, pipesLatencyUs, pipesVerified);

      if (globalRank == 0) {
        float speedup =
            (ncclBandwidth > 0) ? pipesBandwidth / ncclBandwidth : 0;

        std::stringstream ss;
        ss << std::left << std::setw(18) << tuning.name << std::left
           << std::setw(12) << algorithmName(algo) << std::right
           << std::setw(12) << formatSize(tuning.stagingBufferSize)
           << std::right << std::setw(12) << formatSize(tuning.chunkSize)
           << std::right << std::setw(10) << tuning.numBlocks << std::right
           << std::setw(12) << std::fixed << std::setprecision(2)
           << pipesBandwidth << std::right << std::setw(11) << std::fixed
           << std::setprecision(2) << speedup << "x";
        XLOG(INFO) << ss.str();
      }

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    // Print separator between tuning configs
    if (globalRank == 0 && &tuning != &tuningConfigs.back()) {
      XLOG(INFO)
          << "------------------------------------------------------------------------"
             "--------------------";
    }
  }

  if (globalRank == 0) {
    XLOG(INFO)
        << "========================================================================"
           "====================";
    XLOG(INFO) << "Speedup > 1.0x indicates Pipes is faster than NCCL";
    XLOG(INFO)
        << "========================================================================"
           "====================\n";
  }
}

/**
 * Algorithm Comparison Benchmark
 *
 * Compares flat-tree vs binomial tree vs ring broadcast algorithms across
 * various message sizes. Includes bandwidth and latency metrics.
 *
 * This is the comprehensive algorithm comparison test. The ring algorithm
 * achieves best performance for large messages (77% of NCCL at 64MB).
 */
TEST_F(BroadcastBenchmarkFixture, AlgorithmComparison) {
  if (globalRank == 0) {
    XLOG(INFO)
        << "\n=== Flat-Tree vs Binomial Tree vs Ring Algorithm Comparison ===\n";
    XLOG(INFO) << "This test compares the three broadcast algorithms:\n";
    XLOG(INFO) << "  - Flat-Tree (Star): Root sends to all peers directly\n";
    XLOG(INFO) << "  - Binomial Tree: O(log N) rounds, distributes bandwidth\n";
    XLOG(INFO) << "  - Ring: Overlapping send/recv, near-optimal bandwidth\n";
  }

  // Test configurations focusing on message sizes where ring
  // should provide significant improvement (large messages)
  // NOTE: Using optimal configs (128KB chunks, adequate staging buffer)
  // to ensure fair comparison with profiling results
  std::vector<BroadcastBenchmarkConfig> configs;

  configs.push_back({
      .messageSize = 64 * 1024,
      .stagingBufferSize = 16 * 1024 * 1024, // 16MB staging for pipelining
      .pipelineDepth = 4,
      .chunkSize = 64 * 1024,
      .numBlocks = 16,
      .numThreads = 64,
      .rootRank = 0,
      .name = "64KB",
  });

  configs.push_back({
      .messageSize = 128 * 1024,
      .stagingBufferSize = 16 * 1024 * 1024, // 16MB staging for pipelining
      .pipelineDepth = 4,
      .chunkSize = 128 * 1024, // Optimal: 128KB chunks
      .numBlocks = 16,
      .numThreads = 128,
      .rootRank = 0,
      .name = "128KB",
  });

  // Medium messages - ring should start showing benefits
  configs.push_back({
      .messageSize = 256 * 1024,
      .stagingBufferSize = 16 * 1024 * 1024, // 16MB staging for pipelining
      .pipelineDepth = 4,
      .chunkSize = 128 * 1024, // Optimal: 128KB chunks
      .numBlocks = 16,
      .numThreads = 256,
      .rootRank = 0,
      .name = "256KB",
  });

  // Large messages - significant improvement expected
  configs.push_back({
      .messageSize = 1 * 1024 * 1024,
      .stagingBufferSize = 16 * 1024 * 1024, // 16MB staging
      .pipelineDepth = 4,
      .chunkSize = 128 * 1024, // Optimal: 128KB chunks
      .numBlocks = 32,
      .numThreads = 512,
      .rootRank = 0,
      .name = "1MB",
  });

  configs.push_back({
      .messageSize = 4 * 1024 * 1024,
      .stagingBufferSize = 16 * 1024 * 1024, // 16MB staging
      .pipelineDepth = 4,
      .chunkSize = 128 * 1024, // Optimal: 128KB chunks
      .numBlocks = 32,
      .numThreads = 512,
      .rootRank = 0,
      .name = "4MB",
  });

  configs.push_back({
      .messageSize = 8 * 1024 * 1024,
      .stagingBufferSize = 16 * 1024 * 1024, // 16MB staging
      .pipelineDepth = 4,
      .chunkSize = 128 * 1024, // Optimal: 128KB chunks
      .numBlocks = 32,
      .numThreads = 512,
      .rootRank = 0,
      .name = "8MB",
  });

  configs.push_back({
      .messageSize = 16 * 1024 * 1024,
      .stagingBufferSize = 16 * 1024 * 1024, // 16MB staging
      .pipelineDepth = 4,
      .chunkSize = 128 * 1024, // Optimal: 128KB chunks
      .numBlocks = 32,
      .numThreads = 512,
      .rootRank = 0,
      .name = "16MB",
  });

  configs.push_back({
      .messageSize = 32 * 1024 * 1024,
      .stagingBufferSize = 32 * 1024 * 1024, // Match message size
      .pipelineDepth = 4,
      .chunkSize = 128 * 1024, // Optimal: 128KB chunks
      .numBlocks = 32,
      .numThreads = 512,
      .rootRank = 0,
      .name = "32MB",
  });

  configs.push_back({
      .messageSize = 64 * 1024 * 1024,
      .stagingBufferSize = 64 * 1024 * 1024, // Match message size
      .pipelineDepth = 4,
      .chunkSize = 128 * 1024, // Optimal: 128KB chunks
      .numBlocks = 32,
      .numThreads = 512,
      .rootRank = 0,
      .name = "64MB",
  });

  // Print header
  if (globalRank == 0) {
    std::stringstream ss;
    ss << "\n";
    ss << "========================================================================"
          "========================================================================\n";
    ss << "           Flat-Tree vs Binomial Tree vs Ring Algorithm "
          "Comparison\n";
    ss << "========================================================================"
          "========================================================================\n";
    ss << std::left << std::setw(12) << "MsgSize" << std::right << std::setw(12)
       << "NCCL BW" << std::right << std::setw(11) << "Flat BW" << std::right
       << std::setw(12) << "Binom BW" << std::right << std::setw(11)
       << "Ring BW" << std::right << std::setw(13) << "Ring/Flat" << std::right
       << std::setw(13) << "Ring/NCCL" << std::right << std::setw(11)
       << "Flat Lat" << std::right << std::setw(12) << "Ring Lat" << "\n";
    ss << std::left << std::setw(12) << "" << std::right << std::setw(12)
       << "(GB/s)" << std::right << std::setw(11) << "(GB/s)" << std::right
       << std::setw(12) << "(GB/s)" << std::right << std::setw(11) << "(GB/s)"
       << std::right << std::setw(13) << "Speedup" << std::right
       << std::setw(13) << "Speedup" << std::right << std::setw(11) << "(us)"
       << std::right << std::setw(12) << "(us)" << "\n";
    ss << "------------------------------------------------------------------------"
          "------------------------------------------------------------------------\n";
    XLOG(INFO) << ss.str();
  }

  std::vector<BroadcastBenchmarkResult> flatTreeResults;
  std::vector<BroadcastBenchmarkResult> binomialTreeResults;
  std::vector<BroadcastBenchmarkResult> ringResults;

  for (const auto& config : configs) {
    int rootRank = config.rootRank;

    // Run NCCL benchmark (baseline)
    float ncclLatencyUs = 0.0f;
    bool ncclVerified = false;
    float ncclBandwidth = runNcclBroadcastBenchmark(
        config, rootRank, ncclLatencyUs, ncclVerified);

    // Run each algorithm using unified runner
    float flatTreeLatencyUs = 0.0f;
    bool flatTreeVerified = false;
    float flatTreeBandwidth = runPipesBroadcast(
        BroadcastAlgorithm::FlatTree,
        config,
        rootRank,
        flatTreeLatencyUs,
        flatTreeVerified);

    float binomialTreeLatencyUs = 0.0f;
    bool binomialTreeVerified = false;
    float binomialTreeBandwidth = runPipesBroadcast(
        BroadcastAlgorithm::BinomialTree,
        config,
        rootRank,
        binomialTreeLatencyUs,
        binomialTreeVerified);

    float ringLatencyUs = 0.0f;
    bool ringVerified = false;
    float ringBandwidth = runPipesBroadcast(
        BroadcastAlgorithm::Ring,
        config,
        rootRank,
        ringLatencyUs,
        ringVerified);

    if (globalRank == 0) {
      float ringVsFlat =
          (flatTreeBandwidth > 0) ? ringBandwidth / flatTreeBandwidth : 0;
      float ringVsNccl =
          (ncclBandwidth > 0) ? ringBandwidth / ncclBandwidth : 0;

      std::stringstream ss;
      ss << std::left << std::setw(12) << config.name << std::right
         << std::setw(12) << std::fixed << std::setprecision(2) << ncclBandwidth
         << std::right << std::setw(11) << std::fixed << std::setprecision(2)
         << flatTreeBandwidth << std::right << std::setw(12) << std::fixed
         << std::setprecision(2) << binomialTreeBandwidth << std::right
         << std::setw(11) << std::fixed << std::setprecision(2) << ringBandwidth
         << std::right << std::setw(12) << std::fixed << std::setprecision(2)
         << ringVsFlat << "x" << std::right << std::setw(12) << std::fixed
         << std::setprecision(2) << ringVsNccl << "x" << std::right
         << std::setw(11) << std::fixed << std::setprecision(1)
         << flatTreeLatencyUs << std::right << std::setw(12) << std::fixed
         << std::setprecision(1) << ringLatencyUs;
      XLOG(INFO) << ss.str();
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  }

  if (globalRank == 0) {
    XLOG(INFO)
        << "========================================================================"
           "========================================================================";
    XLOG(INFO) << "Ring/Flat Speedup: Values > 1.0x indicate ring is faster "
                  "than flat-tree";
    XLOG(INFO)
        << "Ring/NCCL Speedup: Target is >= 0.5x (50% of NCCL performance)";
    XLOG(INFO)
        << "========================================================================"
           "========================================================================\n";
  }
}

} // namespace

} // namespace comms::pipes::benchmark

// ============================================================================
// SECTION 4: MAIN FUNCTION - CLI Parsing and Test Execution
// ============================================================================

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);

  // Build test filter based on --benchmark flag
  // Valid values: "all", "optimal", "tuning", "algorithm", "clustered",
  //               "rootsweep", "extended", "gridconfig"
  // Can also be comma-separated: "optimal,algorithm"
  if (FLAGS_benchmark != "all") {
    std::string filter;
    std::istringstream iss(FLAGS_benchmark);
    std::string token;
    bool first = true;

    while (std::getline(iss, token, ',')) {
      // Trim whitespace
      token.erase(0, token.find_first_not_of(" \t"));
      token.erase(token.find_last_not_of(" \t") + 1);

      std::string testPattern;
      if (token == "optimal") {
        testPattern = "*OptimalConfigs*";
      } else if (token == "tuning") {
        testPattern = "*StagingBufferTuning*";
      } else if (token == "algorithm") {
        testPattern = "*AlgorithmComparison*";
      } else if (token == "clustered") {
        testPattern = "*ClusteredLaunchComparison*";
      } else if (token == "rootsweep") {
        testPattern = "*RootRankSweep*";
      } else if (token == "extended") {
        testPattern = "*ExtendedMessageSizes*";
      } else if (token == "gridconfig") {
        testPattern = "*GridConfigSweep*";
      } else {
        // Unknown benchmark name, print help and continue
        std::cerr
            << "Warning: Unknown benchmark '" << token << "'. "
            << "Valid values: rootsweep, extended, gridconfig, optimal, tuning, algorithm, clustered, all\n";
        continue;
      }

      if (!first) {
        filter += ":";
      }
      filter += testPattern;
      first = false;
    }

    if (!filter.empty()) {
      ::testing::GTEST_FLAG(filter) = filter;
    }
  }

  return RUN_ALL_TESTS();
}
