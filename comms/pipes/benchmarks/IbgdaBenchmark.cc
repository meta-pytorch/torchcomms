// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include <iomanip>
#include <memory>
#include <sstream>
#include <vector>

#include "comms/pipes/MultipeerIbgdaTransport.h"
#include "comms/pipes/benchmarks/IbgdaBenchmark.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::CudaEvent;
using meta::comms::DeviceBuffer;
using meta::comms::MpiBaseTestFixture;
using meta::comms::MPIEnvironmentBase;

namespace comms::pipes::benchmark {

// Benchmark iteration constants
// All benchmarks now use batched kernels to exclude kernel launch overhead
constexpr int kIbgdaBatchIters = 1000;

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

// CUDA error checking macro for functions returning a value
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

// Benchmark configuration
struct IbgdaBenchmarkConfig {
  std::size_t nBytes = 0;
  int numBlocks = 1;
  int numThreads = 32;
  std::string name;
};

// Result struct for collecting benchmark data
struct IbgdaBenchmarkResult {
  std::string testName;
  std::size_t messageSize{};
  float bandwidth{}; // GB/s
  float latency{}; // microseconds
};

class IbgdaBenchmarkFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDA_CHECK_VOID(cudaSetDevice(localRank));
    CUDA_CHECK_VOID(cudaStreamCreate(&stream_));

    // Get GPU clock rate for converting cycles to time
    int clockRateKHz;
    CUDA_CHECK_VOID(
        cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrClockRate, localRank));
    clockRateGHz_ = clockRateKHz / 1e6f;
  }

  void TearDown() override {
    CUDA_CHECK_VOID(cudaStreamDestroy(stream_));
    MpiBaseTestFixture::TearDown();
  }

  // Convert GPU cycles to microseconds
  float cyclesToUs(unsigned long long cycles) const {
    return cycles / (clockRateGHz_ * 1000.0f);
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

  // Run put_signal_non_adaptive + wait_local benchmark using batched kernel
  // Returns bandwidth, excludes kernel launch overhead
  float runPutSignalNonAdaptiveBenchmark(
      P2pIbgdaTransportDevice* deviceTransportPtr,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      const IbgdaBenchmarkConfig& config,
      unsigned long long* d_totalCycles,
      float& latencyUs) {
    constexpr int kSignalId = 0;

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Only rank 0 (sender) runs the batched benchmark
    if (globalRank == 0) {
      launchIbgdaPutSignalNonAdaptiveWaitLocalBatch(
          deviceTransportPtr,
          localBuf,
          remoteBuf,
          config.nBytes,
          kSignalId,
          kIbgdaBatchIters,
          d_totalCycles,
          stream_);
      CUDA_CHECK(cudaStreamSynchronize(stream_));

      unsigned long long totalCycles;
      CUDA_CHECK(cudaMemcpy(
          &totalCycles,
          d_totalCycles,
          sizeof(unsigned long long),
          cudaMemcpyDeviceToHost));

      latencyUs = cyclesToUs(totalCycles) / kIbgdaBatchIters;
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Unidirectional bandwidth
    float bandwidth_GBps = (config.nBytes / 1e9f) / (latencyUs / 1e6f);
    return bandwidth_GBps;
  }

  // Run put_signal (adaptive-safe) + wait_local benchmark using batched kernel
  // Returns bandwidth, excludes kernel launch overhead
  float runPutSignalBenchmark(
      P2pIbgdaTransportDevice* deviceTransportPtr,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      const IbgdaBenchmarkConfig& config,
      unsigned long long* d_totalCycles,
      float& latencyUs) {
    constexpr int kSignalId = 0;

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Only rank 0 (sender) runs the batched benchmark
    if (globalRank == 0) {
      launchIbgdaPutSignalWaitLocalBatch(
          deviceTransportPtr,
          localBuf,
          remoteBuf,
          config.nBytes,
          kSignalId,
          kIbgdaBatchIters,
          d_totalCycles,
          stream_);
      CUDA_CHECK(cudaStreamSynchronize(stream_));

      unsigned long long totalCycles;
      CUDA_CHECK(cudaMemcpy(
          &totalCycles,
          d_totalCycles,
          sizeof(unsigned long long),
          cudaMemcpyDeviceToHost));

      latencyUs = cyclesToUs(totalCycles) / kIbgdaBatchIters;
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Unidirectional bandwidth
    float bandwidth_GBps = (config.nBytes / 1e9f) / (latencyUs / 1e6f);
    return bandwidth_GBps;
  }

  void printResultsTable(
      const std::string& title,
      const std::vector<IbgdaBenchmarkResult>& results) {
    if (globalRank != 0) {
      return;
    }

    std::stringstream ss;
    ss << "\n";
    ss << "================================================================================\n";
    ss << "                    " << title << "\n";
    ss << "                    (Raw latency, no kernel launch overhead)\n";
    ss << "================================================================================\n";
    ss << std::left << std::setw(20) << "Test Name" << std::right
       << std::setw(12) << "Msg Size" << std::right << std::setw(14)
       << "BW (GB/s)" << std::right << std::setw(14) << "Latency (us)\n";
    ss << "--------------------------------------------------------------------------------\n";

    for (const auto& r : results) {
      std::string msgSize = formatSize(r.messageSize);

      ss << std::left << std::setw(20) << r.testName << std::right
         << std::setw(12) << msgSize << std::right << std::setw(14)
         << std::fixed << std::setprecision(2) << r.bandwidth << std::right
         << std::setw(14) << std::fixed << std::setprecision(2) << r.latency
         << "\n";
    }
    ss << "================================================================================\n";
    ss << "Measured using GPU cycle counters inside a single kernel launch\n";
    ss << "Batch iterations: " << kIbgdaBatchIters
       << ", GPU clock: " << clockRateGHz_ << " GHz\n";
    ss << "================================================================================\n\n";

    XLOG(INFO) << ss.str();
  }

  // Standard message size configurations
  std::vector<IbgdaBenchmarkConfig> getFullConfigs() {
    return {
        {.nBytes = 8, .name = "8B"},
        {.nBytes = 64, .name = "64B"},
        {.nBytes = 256, .name = "256B"},
        {.nBytes = 1024, .name = "1KB"},
        {.nBytes = 4 * 1024, .name = "4KB"},
        {.nBytes = 8 * 1024, .name = "8KB"},
        {.nBytes = 16 * 1024, .name = "16KB"},
        {.nBytes = 32 * 1024, .name = "32KB"},
        {.nBytes = 64 * 1024, .name = "64KB"},
        {.nBytes = 128 * 1024, .name = "128KB"},
        {.nBytes = 256 * 1024, .name = "256KB"},
        {.nBytes = 512 * 1024, .name = "512KB"},
        {.nBytes = 1024 * 1024, .name = "1MB"},
        {.nBytes = 2 * 1024 * 1024, .name = "2MB"},
        {.nBytes = 4 * 1024 * 1024, .name = "4MB"},
        {.nBytes = 8 * 1024 * 1024, .name = "8MB"},
        {.nBytes = 16 * 1024 * 1024, .name = "16MB"},
        {.nBytes = 32 * 1024 * 1024, .name = "32MB"},
        {.nBytes = 64 * 1024 * 1024, .name = "64MB"},
        {.nBytes = 128 * 1024 * 1024, .name = "128MB"},
    };
  }

  cudaStream_t stream_{};
  float clockRateGHz_{0.0f};
};

TEST_F(IbgdaBenchmarkFixture, PutWaitLocal) {
  // Measures raw RDMA Write latency (put + wait_local)
  if (numRanks != 2) {
    XLOGF(INFO, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;
  auto configs = getFullConfigs();

  std::size_t maxBufferSize = 0;
  for (const auto& config : configs) {
    maxBufferSize = std::max(maxBufferSize, config.nBytes);
  }

  std::vector<IbgdaBenchmarkResult> results;

  try {
    MultipeerIbgdaTransportConfig transportConfig{
        .cudaDevice = localRank,
        .signalCount = 1,
    };

    auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
    MultipeerIbgdaTransport transport(
        globalRank, numRanks, bootstrap, transportConfig);
    transport.exchange();

    DeviceBuffer dataBuffer(maxBufferSize);
    auto localDataBuf =
        transport.registerBuffer(dataBuffer.get(), maxBufferSize);

    auto remoteDataBufs = transport.exchangeBuffer(localDataBuf);
    int peerIndex = (peerRank < globalRank) ? peerRank : (peerRank - 1);
    auto remoteDataBuf = remoteDataBufs[peerIndex];

    P2pIbgdaTransportDevice* deviceTransportPtr =
        transport.getP2pTransportDevice(peerRank);

    // Allocate device memory for cycle counter output
    unsigned long long* d_totalCycles;
    CUDA_CHECK_VOID(cudaMalloc(&d_totalCycles, sizeof(unsigned long long)));

    XLOGF(
        INFO,
        "Rank {}: GPU clock rate = {:.2f} GHz",
        globalRank,
        clockRateGHz_);

    for (const auto& config : configs) {
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      // Only rank 0 sends
      if (globalRank == 0) {
        launchIbgdaPutWaitLocalBatch(
            deviceTransportPtr,
            localDataBuf,
            remoteDataBuf,
            config.nBytes,
            kIbgdaBatchIters,
            d_totalCycles,
            stream_);
        CUDA_CHECK_VOID(cudaStreamSynchronize(stream_));

        unsigned long long totalCycles;
        CUDA_CHECK_VOID(cudaMemcpy(
            &totalCycles,
            d_totalCycles,
            sizeof(unsigned long long),
            cudaMemcpyDeviceToHost));

        IbgdaBenchmarkResult result;
        result.testName = config.name;
        result.messageSize = config.nBytes;
        result.latency = cyclesToUs(totalCycles) / kIbgdaBatchIters;
        result.bandwidth = (config.nBytes / 1e9f) / (result.latency / 1e6f);

        results.push_back(result);

        XLOGF(
            INFO,
            "Rank {}: {} - Latency: {:.2f} us, BW: {:.2f} GB/s",
            globalRank,
            config.name,
            result.latency,
            result.bandwidth);
      }

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    CUDA_CHECK_VOID(cudaFree(d_totalCycles));

  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }

  printResultsTable("IBGDA Put+WaitLocal (RDMA Write)", results);
}

TEST_F(IbgdaBenchmarkFixture, PutSignalWaitLocal) {
  // Measures RDMA Write + atomic signal latency (put_signal + wait_local)
  if (numRanks != 2) {
    XLOGF(INFO, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;
  auto configs = getFullConfigs();
  constexpr int kSignalId = 0;

  std::size_t maxBufferSize = 0;
  for (const auto& config : configs) {
    maxBufferSize = std::max(maxBufferSize, config.nBytes);
  }

  std::vector<IbgdaBenchmarkResult> results;

  try {
    MultipeerIbgdaTransportConfig transportConfig{
        .cudaDevice = localRank,
        .signalCount = 1,
    };

    auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
    MultipeerIbgdaTransport transport(
        globalRank, numRanks, bootstrap, transportConfig);
    transport.exchange();

    DeviceBuffer dataBuffer(maxBufferSize);
    auto localDataBuf =
        transport.registerBuffer(dataBuffer.get(), maxBufferSize);

    auto remoteDataBufs = transport.exchangeBuffer(localDataBuf);
    int peerIndex = (peerRank < globalRank) ? peerRank : (peerRank - 1);
    auto remoteDataBuf = remoteDataBufs[peerIndex];

    P2pIbgdaTransportDevice* deviceTransportPtr =
        transport.getP2pTransportDevice(peerRank);

    // Allocate device memory for cycle counter output
    unsigned long long* d_totalCycles;
    CUDA_CHECK_VOID(cudaMalloc(&d_totalCycles, sizeof(unsigned long long)));

    XLOGF(
        INFO,
        "Rank {}: GPU clock rate = {:.2f} GHz",
        globalRank,
        clockRateGHz_);

    for (const auto& config : configs) {
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

      // Only rank 0 sends
      if (globalRank == 0) {
        launchIbgdaPutSignalWaitLocalBatch(
            deviceTransportPtr,
            localDataBuf,
            remoteDataBuf,
            config.nBytes,
            kSignalId,
            kIbgdaBatchIters,
            d_totalCycles,
            stream_);
        CUDA_CHECK_VOID(cudaStreamSynchronize(stream_));

        unsigned long long totalCycles;
        CUDA_CHECK_VOID(cudaMemcpy(
            &totalCycles,
            d_totalCycles,
            sizeof(unsigned long long),
            cudaMemcpyDeviceToHost));

        IbgdaBenchmarkResult result;
        result.testName = config.name;
        result.messageSize = config.nBytes;
        result.latency = cyclesToUs(totalCycles) / kIbgdaBatchIters;
        result.bandwidth = (config.nBytes / 1e9f) / (result.latency / 1e6f);

        results.push_back(result);

        XLOGF(
            INFO,
            "Rank {}: {} - Latency: {:.2f} us, BW: {:.2f} GB/s",
            globalRank,
            config.name,
            result.latency,
            result.bandwidth);
      }

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    }

    CUDA_CHECK_VOID(cudaFree(d_totalCycles));

  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }

  printResultsTable(
      "IBGDA Put+Signal+WaitLocal (RDMA Write + Atomic)", results);
}

TEST_F(IbgdaBenchmarkFixture, SignalOnly) {
  // Measures atomic signal-only latency (no data transfer)
  if (numRanks != 2) {
    XLOGF(INFO, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;
  constexpr int kSignalId = 0;

  try {
    MultipeerIbgdaTransportConfig transportConfig{
        .cudaDevice = localRank,
        .signalCount = 1,
    };

    auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
    MultipeerIbgdaTransport transport(
        globalRank, numRanks, bootstrap, transportConfig);
    transport.exchange();

    P2pIbgdaTransportDevice* deviceTransportPtr =
        transport.getP2pTransportDevice(peerRank);

    // Allocate device memory for cycle counter output
    unsigned long long* d_totalCycles;
    CUDA_CHECK_VOID(cudaMalloc(&d_totalCycles, sizeof(unsigned long long)));

    XLOGF(
        INFO,
        "Rank {}: GPU clock rate = {:.2f} GHz",
        globalRank,
        clockRateGHz_);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    float latencyUs = 0.0f;

    // Only rank 0 sends
    if (globalRank == 0) {
      launchIbgdaSignalOnlyBatch(
          deviceTransportPtr,
          kSignalId,
          kIbgdaBatchIters,
          d_totalCycles,
          stream_);
      CUDA_CHECK_VOID(cudaStreamSynchronize(stream_));

      unsigned long long totalCycles;
      CUDA_CHECK_VOID(cudaMemcpy(
          &totalCycles,
          d_totalCycles,
          sizeof(unsigned long long),
          cudaMemcpyDeviceToHost));

      latencyUs = cyclesToUs(totalCycles) / kIbgdaBatchIters;

      XLOGF(
          INFO,
          "\n=== Signal-Only Latency (Raw, no kernel launch overhead) ===");
      XLOGF(INFO, "Average latency: {:.2f} us", latencyUs);
      XLOGF(INFO, "Batch iterations: {}", kIbgdaBatchIters);
      XLOGF(
          INFO,
          "===========================================================\n");
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    CUDA_CHECK_VOID(cudaFree(d_totalCycles));

  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }
}

// Benchmark comparing put_signal (adaptive-safe) vs put_signal_non_adaptive
TEST_F(IbgdaBenchmarkFixture, PutSignalComparison) {
  if (numRanks != 2) {
    XLOGF(INFO, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;

  // Test a subset of sizes for comparison
  std::vector<IbgdaBenchmarkConfig> configs;
  configs.push_back(
      {.nBytes = 64, .numBlocks = 1, .numThreads = 32, .name = "64B"});
  configs.push_back(
      {.nBytes = 1024, .numBlocks = 1, .numThreads = 32, .name = "1KB"});
  configs.push_back(
      {.nBytes = 64 * 1024, .numBlocks = 1, .numThreads = 32, .name = "64KB"});
  configs.push_back(
      {.nBytes = 1024 * 1024, .numBlocks = 1, .numThreads = 32, .name = "1MB"});
  configs.push_back(
      {.nBytes = 16 * 1024 * 1024,
       .numBlocks = 1,
       .numThreads = 32,
       .name = "16MB"});

  std::size_t maxBufferSize = 0;
  for (const auto& config : configs) {
    maxBufferSize = std::max(maxBufferSize, config.nBytes);
  }

  try {
    MultipeerIbgdaTransportConfig transportConfig{
        .cudaDevice = localRank,
        .signalCount = 1,
    };

    auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
    MultipeerIbgdaTransport transport(
        globalRank, numRanks, bootstrap, transportConfig);
    transport.exchange();

    DeviceBuffer dataBuffer(maxBufferSize);
    auto localDataBuf =
        transport.registerBuffer(dataBuffer.get(), maxBufferSize);
    auto remoteDataBufs = transport.exchangeBuffer(localDataBuf);
    int peerIndex = (peerRank < globalRank) ? peerRank : (peerRank - 1);
    auto remoteDataBuf = remoteDataBufs[peerIndex];

    P2pIbgdaTransportDevice* deviceTransportPtr =
        transport.getP2pTransportDevice(peerRank);

    // Allocate device memory for cycle counter output
    unsigned long long* d_totalCycles;
    CUDA_CHECK_VOID(cudaMalloc(&d_totalCycles, sizeof(unsigned long long)));

    if (globalRank == 0) {
      XLOGF(
          INFO,
          "\n================================================================================");
      XLOGF(INFO, "        put_signal vs put_signal_non_adaptive Comparison");
      XLOGF(
          INFO, "        (Using batched kernels - no kernel launch overhead)");
      XLOGF(
          INFO,
          "================================================================================");
      XLOGF(
          INFO,
          "{:>10} {:>15} {:>15} {:>15} {:>15}",
          "Size",
          "Adaptive BW",
          "NonAdapt BW",
          "Adapt Lat",
          "NonAdapt Lat");
      XLOGF(
          INFO,
          "{:>10} {:>15} {:>15} {:>15} {:>15}",
          "",
          "(GB/s)",
          "(GB/s)",
          "(us)",
          "(us)");
      XLOGF(
          INFO,
          "--------------------------------------------------------------------------------");
    }

    for (const auto& config : configs) {
      float adaptiveLatency = 0.0f;
      float nonAdaptiveLatency = 0.0f;

      float adaptiveBw = runPutSignalBenchmark(
          deviceTransportPtr,
          localDataBuf,
          remoteDataBuf,
          config,
          d_totalCycles,
          adaptiveLatency);

      float nonAdaptiveBw = runPutSignalNonAdaptiveBenchmark(
          deviceTransportPtr,
          localDataBuf,
          remoteDataBuf,
          config,
          d_totalCycles,
          nonAdaptiveLatency);

      if (globalRank == 0) {
        XLOGF(
            INFO,
            "{:>10} {:>15.2f} {:>15.2f} {:>15.1f} {:>15.1f}",
            config.name,
            adaptiveBw,
            nonAdaptiveBw,
            adaptiveLatency,
            nonAdaptiveLatency);
      }
    }

    if (globalRank == 0) {
      XLOGF(
          INFO,
          "================================================================================");
      XLOGF(INFO, "Adaptive = put_signal (safe for adaptive routing networks)");
      XLOGF(
          INFO,
          "NonAdaptive = put_signal_non_adaptive (faster, but unsafe with adaptive routing)");
      XLOGF(
          INFO,
          "================================================================================\n");
    }

  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }
}

} // namespace comms::pipes::benchmark

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
