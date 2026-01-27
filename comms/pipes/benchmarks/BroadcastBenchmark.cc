// Copyright (c) Meta Platforms, Inc. and affiliates.

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
    "Which benchmark(s) to run: all, optimal, tuning, algorithm (comma-separated)");

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

namespace {

/**
 * Test configuration for Broadcast benchmark.
 */
struct BroadcastBenchmarkConfig {
  std::size_t messageSize;
  std::size_t stagingBufferSize;
  std::size_t pipelineDepth = 4;
  std::size_t chunkSize = 512 * 1024; // 512KB default
  int numBlocks;
  int numThreads;
  int rootRank; // Which rank is the broadcast source (use -1 for dynamic)
  std::string name;
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

      ncclBroadcast(
          buffer.get(),
          buffer.get(),
          config.messageSize,
          ncclChar,
          rootRank,
          ncclComm_,
          stream_);
      cudaStreamSynchronize(stream_);

      verified = verifyBroadcastData(
          buffer.get(), config.messageSize, rootRank, config.name, "NCCL");
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    return bandwidth_GBps;
  }

  /**
   * Run Pipes Broadcast benchmark.
   * Returns bandwidth in GB/s and sets latency in microseconds.
   */
  float runPipesBroadcastBenchmark(
      const BroadcastBenchmarkConfig& config,
      int rootRank,
      float& latencyUs,
      bool& verified) {
    XLOGF(
        DBG1,
        "Rank {}: Running Pipes Broadcast benchmark: {} (root={})",
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
    NVTX_RANGE_PUSH("Pipes_Broadcast_Warmup");
    for (int i = 0; i < kWarmupIters; i++) {
      CUDA_CHECK(cudaLaunchKernel(
          (void*)broadcastKernel, gridDim, blockDim, args, 0, stream_));
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    NVTX_RANGE_POP();
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Benchmark
    NVTX_RANGE_PUSH("Pipes_Broadcast_Benchmark");
    CUDA_CHECK(cudaEventRecord(start.get(), stream_));
    for (int i = 0; i < kBenchmarkIters; i++) {
      NVTX_RANGE_PUSH("Pipes_Broadcast_Iter");
      CUDA_CHECK(cudaLaunchKernel(
          (void*)broadcastKernel, gridDim, blockDim, args, 0, stream_));
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
          (void*)broadcastKernel, gridDim, blockDim, args, 0, stream_);
      cudaStreamSynchronize(stream_);

      verified = verifyBroadcastData(
          buffer.get(), config.messageSize, rootRank, config.name, "Pipes");
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    return bandwidth_GBps;
  }

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

  /**
   * Run Pipes Broadcast benchmark with adaptive algorithm selection.
   * Automatically selects the best algorithm based on message size:
   * - < 8MB: flat-tree (lowest latency)
   * - >= 8MB: ring (best bandwidth)
   * Returns bandwidth in GB/s and sets latency in microseconds.
   */
  float runPipesAdaptiveBenchmark(
      const BroadcastBenchmarkConfig& config,
      int rootRank,
      float& latencyUs,
      bool& verified) {
    XLOGF(
        DBG1,
        "Rank {}: Running Pipes Adaptive broadcast: {} (root={})",
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
    NVTX_RANGE_PUSH("Pipes_Adaptive_Warmup");
    for (int i = 0; i < kWarmupIters; i++) {
      CUDA_CHECK(cudaLaunchKernel(
          (void*)broadcastAdaptiveKernel, gridDim, blockDim, args, 0, stream_));
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    NVTX_RANGE_POP();
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Benchmark
    NVTX_RANGE_PUSH("Pipes_Adaptive_Benchmark");
    CUDA_CHECK(cudaEventRecord(start.get(), stream_));
    for (int i = 0; i < kBenchmarkIters; i++) {
      NVTX_RANGE_PUSH("Pipes_Adaptive_Iter");
      CUDA_CHECK(cudaLaunchKernel(
          (void*)broadcastAdaptiveKernel, gridDim, blockDim, args, 0, stream_));
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
          (void*)broadcastAdaptiveKernel, gridDim, blockDim, args, 0, stream_);
      cudaStreamSynchronize(stream_);

      verified = verifyBroadcastData(
          buffer.get(), config.messageSize, rootRank, config.name, "Adaptive");
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    return bandwidth_GBps;
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

  void printAlgorithmComparisonTable(
      const std::vector<BroadcastBenchmarkResult>& flatTreeResults,
      const std::vector<BroadcastBenchmarkResult>& binomialTreeResults) {
    if (globalRank != 0) {
      return;
    }

    std::stringstream ss;
    ss << "\n";
    ss << "========================================================"
          "========================================================\n";
    ss << "                Flat-Tree vs Binomial Tree Algorithm Comparison\n";
    ss << "========================================================"
          "========================================================\n";
    ss << std::left << std::setw(18) << "Test Name" << std::right
       << std::setw(10) << "Size" << std::right << std::setw(12) << "FlatTree"
       << std::right << std::setw(12) << "BinomTree" << std::right
       << std::setw(12) << "NCCL" << std::right << std::setw(12) << "Flat/NCCL"
       << std::right << std::setw(12) << "Binom/NCCL" << std::right
       << std::setw(14) << "Improvement" << "\n";
    ss << std::left << std::setw(18) << "" << std::right << std::setw(10) << ""
       << std::right << std::setw(12) << "(GB/s)" << std::right << std::setw(12)
       << "(GB/s)" << std::right << std::setw(12) << "(GB/s)" << std::right
       << std::setw(12) << "" << std::right << std::setw(12) << "" << std::right
       << std::setw(14) << "(Binom/Flat)" << "\n";
    ss << "--------------------------------------------------------"
          "--------------------------------------------------------\n";

    for (size_t i = 0; i < flatTreeResults.size(); i++) {
      const auto& flat = flatTreeResults[i];
      const auto& binom = binomialTreeResults[i];

      float improvement = (flat.pipesBandwidth > 0)
          ? binom.pipesBandwidth / flat.pipesBandwidth
          : 0;
      float binomSpeedup = (flat.ncclBandwidth > 0)
          ? binom.pipesBandwidth / flat.ncclBandwidth
          : 0;

      ss << std::left << std::setw(18) << flat.testName << std::right
         << std::setw(10) << formatSize(flat.messageSize) << std::right
         << std::setw(12) << std::fixed << std::setprecision(2)
         << flat.pipesBandwidth << std::right << std::setw(12) << std::fixed
         << std::setprecision(2) << binom.pipesBandwidth << std::right
         << std::setw(12) << std::fixed << std::setprecision(2)
         << flat.ncclBandwidth << std::right << std::setw(11) << std::fixed
         << std::setprecision(2) << flat.speedup << "x" << std::right
         << std::setw(11) << std::fixed << std::setprecision(2) << binomSpeedup
         << "x" << std::right << std::setw(13) << std::fixed
         << std::setprecision(2) << improvement << "x" << "\n";
    }

    ss << "========================================================"
          "========================================================\n";

    XLOG(INFO) << ss.str();
    XLOG(INFO) << "Improvement = Binomial Tree BW / Flat Tree BW";
    XLOG(INFO)
        << "========================================================================"
           "================================\n";
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

// clang-format off
/**
 * Broadcast Benchmark
 *
 * Tests Pipes Broadcast against NCCL ncclBroadcast across various:
 * - Message sizes: 64B to 64MB
 * - Root ranks: 0, middle, last
 * - Thread configurations
 *
 * The Pipes Broadcast uses adaptive algorithm selection:
 * - Small/medium messages (< 8MB): flat-tree (lowest latency)
 * - Large messages (>= 8MB): ring (best bandwidth, 77% of NCCL)
 *
 * Use --verify_correctness flag to enable data correctness verification.
 */
// clang-format on
TEST_F(BroadcastBenchmarkFixture, OptimalConfigs) {
  if (globalRank == 0) {
    XLOG(INFO)
        << "\n=== Pipes Broadcast vs NCCL Comparison (All Message Sizes) ===\n";
    if (FLAGS_verify_correctness) {
      XLOG(INFO) << "Data correctness verification ENABLED\n";
    }
  }

  std::vector<BroadcastBenchmarkConfig> configs;

  // === SMALL MESSAGES ===
  // 64B - latency bound
  configs.push_back({
      .messageSize = 64,
      .stagingBufferSize = 64 * 1024,
      .pipelineDepth = 4,
      .chunkSize = 8 * 1024,
      .numBlocks = 4,
      .numThreads = 256,
      .rootRank = 0,
      .name = "64B_root0",
  });

  // 1KB
  configs.push_back({
      .messageSize = 1 * 1024,
      .stagingBufferSize = 64 * 1024,
      .pipelineDepth = 4,
      .chunkSize = 8 * 1024,
      .numBlocks = 4,
      .numThreads = 256,
      .rootRank = 0,
      .name = "1KB_root0",
  });

  // 4KB
  configs.push_back({
      .messageSize = 4 * 1024,
      .stagingBufferSize = 64 * 1024,
      .pipelineDepth = 4,
      .chunkSize = 8 * 1024,
      .numBlocks = 4,
      .numThreads = 256,
      .rootRank = 0,
      .name = "4KB_root0",
  });

  // === MEDIUM MESSAGES ===
  // 64KB
  configs.push_back({
      .messageSize = 64 * 1024,
      .stagingBufferSize = 64 * 1024,
      .pipelineDepth = 4,
      .chunkSize = 16 * 1024,
      .numBlocks = 4,
      .numThreads = 256,
      .rootRank = 0,
      .name = "64KB_root0",
  });

  // 256KB
  configs.push_back({
      .messageSize = 256 * 1024,
      .stagingBufferSize = 256 * 1024,
      .pipelineDepth = 4,
      .chunkSize = 32 * 1024,
      .numBlocks = 8,
      .numThreads = 256,
      .rootRank = 0,
      .name = "256KB_root0",
  });

  // === LARGE MESSAGES ===
  // OPTIMIZED: Using empirically validated parameters from tuning sweep
  // Best config: 128KB chunks, 32 blocks, 16MB staging buffer
  // 1MB
  configs.push_back({
      .messageSize = 1 * 1024 * 1024,
      .stagingBufferSize = 16 * 1024 * 1024, // Optimal: 16MB staging
      .pipelineDepth = 4,
      .chunkSize = 128 * 1024, // Optimal: 128KB chunks
      .numBlocks = 32, // Optimal: 32 blocks
      .numThreads = 512,
      .rootRank = 0,
      .name = "1MB_root0",
  });

  // 4MB
  configs.push_back({
      .messageSize = 4 * 1024 * 1024,
      .stagingBufferSize = 16 * 1024 * 1024, // Optimal: 16MB staging
      .pipelineDepth = 4,
      .chunkSize = 128 * 1024, // Optimal: 128KB chunks
      .numBlocks = 32, // Optimal: 32 blocks
      .numThreads = 512,
      .rootRank = 0,
      .name = "4MB_root0",
  });

  // 8MB
  configs.push_back({
      .messageSize = 8 * 1024 * 1024,
      .stagingBufferSize = 16 * 1024 * 1024, // Optimal: 16MB staging
      .pipelineDepth = 4,
      .chunkSize = 128 * 1024, // Optimal: 128KB chunks
      .numBlocks = 32, // Optimal: 32 blocks
      .numThreads = 512,
      .rootRank = 0,
      .name = "8MB_root0",
  });

  // === VARYING ROOT RANK ===
  // Test with middle rank as root
  int middleRank = numRanks / 2;
  configs.push_back({
      .messageSize = 64 * 1024,
      .stagingBufferSize = 64 * 1024,
      .pipelineDepth = 4,
      .chunkSize = 16 * 1024,
      .numBlocks = 4,
      .numThreads = 256,
      .rootRank = middleRank,
      .name = "64KB_rootMiddle",
  });

  configs.push_back({
      .messageSize = 1 * 1024 * 1024,
      .stagingBufferSize = 16 * 1024 * 1024, // Optimal: 16MB staging
      .pipelineDepth = 4,
      .chunkSize = 128 * 1024, // Optimal: 128KB chunks
      .numBlocks = 32, // Optimal: 32 blocks
      .numThreads = 512,
      .rootRank = middleRank,
      .name = "1MB_rootMiddle",
  });

  // Test with last rank as root
  int lastRank = numRanks - 1;
  configs.push_back({
      .messageSize = 64 * 1024,
      .stagingBufferSize = 64 * 1024,
      .pipelineDepth = 4,
      .chunkSize = 16 * 1024,
      .numBlocks = 4,
      .numThreads = 256,
      .rootRank = lastRank,
      .name = "64KB_rootLast",
  });

  configs.push_back({
      .messageSize = 1 * 1024 * 1024,
      .stagingBufferSize = 16 * 1024 * 1024, // Optimal: 16MB staging
      .pipelineDepth = 4,
      .chunkSize = 128 * 1024, // Optimal: 128KB chunks
      .numBlocks = 32, // Optimal: 32 blocks
      .numThreads = 512,
      .rootRank = lastRank,
      .name = "1MB_rootLast",
  });

  // === LARGE MESSAGE SIZE SWEEP ===
  // OPTIMIZED: Using empirically validated parameters from tuning sweep
  // Best config: 128KB chunks, 32 blocks, staging buffer >= message size
  // Achieves 253 GB/s (0.77x NCCL) for 64MB with ring algorithm
  configs.push_back({
      .messageSize = 16 * 1024 * 1024,
      .stagingBufferSize = 16 * 1024 * 1024, // Match message size
      .pipelineDepth = 4,
      .chunkSize = 128 * 1024, // Optimal: 128KB chunks
      .numBlocks = 32, // Optimal: 32 blocks
      .numThreads = 512,
      .rootRank = 0,
      .name = "16MB_root0",
  });

  configs.push_back({
      .messageSize = 32 * 1024 * 1024,
      .stagingBufferSize = 32 * 1024 * 1024, // Match message size
      .pipelineDepth = 4,
      .chunkSize = 128 * 1024, // Optimal: 128KB chunks
      .numBlocks = 32, // Optimal: 32 blocks
      .numThreads = 512,
      .rootRank = 0,
      .name = "32MB_root0",
  });

  configs.push_back({
      .messageSize = 64 * 1024 * 1024,
      .stagingBufferSize = 64 * 1024 * 1024, // Match message size
      .pipelineDepth = 4,
      .chunkSize = 128 * 1024, // Optimal: 128KB chunks
      .numBlocks = 32, // Optimal: 32 blocks
      .numThreads = 512,
      .rootRank = 0,
      .name = "64MB_root0",
  });

  std::vector<BroadcastBenchmarkResult> results;

  for (const auto& config : configs) {
    int rootRank = config.rootRank;

    float ncclLatencyUs = 0.0f;
    bool ncclVerified = false;
    float ncclBandwidth = runNcclBroadcastBenchmark(
        config, rootRank, ncclLatencyUs, ncclVerified);

    float pipesLatencyUs = 0.0f;
    bool pipesVerified = false;
    float pipesBandwidth = runPipesAdaptiveBenchmark(
        config, rootRank, pipesLatencyUs, pipesVerified);

    if (globalRank == 0) {
      BroadcastBenchmarkResult result;
      result.testName = config.name;
      result.messageSize = config.messageSize;
      result.rootRank = rootRank;
      result.nRanks = numRanks;
      result.ncclBandwidth = ncclBandwidth;
      result.pipesBandwidth = pipesBandwidth;
      result.ncclLatency = ncclLatencyUs;
      result.pipesLatency = pipesLatencyUs;
      result.speedup = (ncclBandwidth > 0) ? pipesBandwidth / ncclBandwidth : 0;
      result.ncclVerified = ncclVerified;
      result.pipesVerified = pipesVerified;
      results.push_back(result);
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  }

  printResultsTable(results);

  // Assert verification results if enabled
  if (FLAGS_verify_correctness) {
    for (const auto& r : results) {
      EXPECT_TRUE(r.ncclVerified)
          << "NCCL verification failed for " << r.testName;
      EXPECT_TRUE(r.pipesVerified)
          << "Pipes verification failed for " << r.testName;
    }
  }
}

/**
 * Staging Buffer Tuning Benchmark
 *
 * Sweeps staging buffer sizes and chunk sizes to find optimal
 * configuration for different message sizes. Uses the adaptive
 * algorithm (ring for >= 8MB messages) to reflect real-world performance.
 *
 * EMPIRICAL FINDINGS (8-rank NVLink broadcast, 64MB message with ring):
 * - Smaller chunks = dramatically better performance!
 * - 128KB chunks with ring: 253 GB/s (0.77x NCCL)
 * - This is because more chunks = more warp parallelism + pipelining
 */
TEST_F(BroadcastBenchmarkFixture, StagingBufferTuning) {
  if (globalRank == 0) {
    XLOG(INFO) << "\n=== Staging Buffer and Chunk Size Tuning ===\n";
    XLOG(INFO)
        << "This test sweeps parameters using adaptive algorithm (ring for 64MB).\n";
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
    ss << std::left << std::setw(18) << "Config" << std::right << std::setw(12)
       << "Staging" << std::right << std::setw(12) << "Chunk" << std::right
       << std::setw(10) << "Blocks" << std::right << std::setw(12)
       << "Bandwidth" << std::right << std::setw(12) << "vs NCCL"
       << "\n";
    ss << std::left << std::setw(18) << "" << std::right << std::setw(12)
       << "Size" << std::right << std::setw(12) << "Size" << std::right
       << std::setw(10) << "" << std::right << std::setw(12) << "(GB/s)"
       << std::right << std::setw(12) << ""
       << "\n";
    ss << "--------------------------------------------------------------------------"
          "--\n";
    XLOG(INFO) << ss.str();
  }

  float bestBandwidth = 0.0f;
  std::string bestConfig;

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

    float pipesLatencyUs = 0.0f;
    bool pipesVerified = false;
    float pipesBandwidth =
        runPipesAdaptiveBenchmark(config, 0, pipesLatencyUs, pipesVerified);

    if (globalRank == 0) {
      float speedup = (ncclBandwidth > 0) ? pipesBandwidth / ncclBandwidth : 0;

      std::stringstream ss;
      ss << std::left << std::setw(18) << tuning.name << std::right
         << std::setw(12) << formatSize(tuning.stagingBufferSize) << std::right
         << std::setw(12) << formatSize(tuning.chunkSize) << std::right
         << std::setw(10) << tuning.numBlocks << std::right << std::setw(12)
         << std::fixed << std::setprecision(2) << pipesBandwidth << std::right
         << std::setw(11) << std::fixed << std::setprecision(2) << speedup
         << "x";
      XLOG(INFO) << ss.str();

      if (pipesBandwidth > bestBandwidth) {
        bestBandwidth = pipesBandwidth;
        bestConfig = tuning.name;
      }
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  }

  if (globalRank == 0) {
    XLOG(INFO)
        << "----------------------------------------------------------------------------";
    XLOG(INFO) << "Best config: " << bestConfig << " at " << std::fixed
               << std::setprecision(2) << bestBandwidth << " GB/s ("
               << std::fixed << std::setprecision(2)
               << (bestBandwidth / ncclBandwidth) << "x of NCCL)";
    XLOG(INFO)
        << "============================================================================\n";
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

    // Run flat-tree Pipes benchmark
    float flatTreeLatencyUs = 0.0f;
    bool flatTreeVerified = false;
    float flatTreeBandwidth = runPipesBroadcastBenchmark(
        config, rootRank, flatTreeLatencyUs, flatTreeVerified);

    // Run binomial tree Pipes benchmark
    float binomialTreeLatencyUs = 0.0f;
    bool binomialTreeVerified = false;
    float binomialTreeBandwidth = runPipesBinomialTreeBenchmark(
        config, rootRank, binomialTreeLatencyUs, binomialTreeVerified);

    // Run ring Pipes benchmark
    float ringLatencyUs = 0.0f;
    bool ringVerified = false;
    float ringBandwidth =
        runPipesRingBenchmark(config, rootRank, ringLatencyUs, ringVerified);

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

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);

  // Build test filter based on --benchmark flag
  // Valid values: "all", "optimal", "tuning", "algorithm"
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
      } else {
        // Unknown benchmark name, print help and continue
        std::cerr << "Warning: Unknown benchmark '" << token << "'. "
                  << "Valid values: optimal, tuning, algorithm, all\n";
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
