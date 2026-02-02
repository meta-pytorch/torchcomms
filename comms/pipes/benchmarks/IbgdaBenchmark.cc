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
constexpr int kWarmupIters = 20;
constexpr int kBenchmarkIters = 30;

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
  int numBlocks{};
  int numThreads{};
  float bandwidth{}; // GB/s
  float latency{}; // microseconds
};

class IbgdaBenchmarkFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDA_CHECK_VOID(cudaSetDevice(localRank));
    CUDA_CHECK_VOID(cudaStreamCreate(&stream_));
  }

  void TearDown() override {
    CUDA_CHECK_VOID(cudaStreamDestroy(stream_));
    MpiBaseTestFixture::TearDown();
  }

  // Run put_signal + wait_local benchmark - returns bandwidth
  // Sender issues kBenchmarkIters put_signal operations (each adds 1 to signal)
  // Receiver waits for final signal value
  float runPutSignalBenchmark(
      P2pIbgdaTransportDevice* deviceTransportPtr,
      const IbgdaLocalBuffer& localBuf,
      const IbgdaRemoteBuffer& remoteBuf,
      const IbgdaBenchmarkConfig& config,
      float& latencyUs) {
    CudaEvent start, stop;

    // Warmup - single iteration
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (globalRank == 0) {
      launchIbgdaPutSignalWaitLocal(
          deviceTransportPtr,
          localBuf,
          remoteBuf,
          config.nBytes,
          1,
          config.numBlocks,
          config.numThreads,
          stream_);
    } else {
      launchIbgdaWaitSignal(
          deviceTransportPtr, 1, config.numBlocks, config.numThreads, stream_);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Reset signal for benchmark
    if (globalRank == 1) {
      launchIbgdaResetSignal(deviceTransportPtr, stream_);
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Benchmark - sender issues all operations, receiver waits for final signal
    CUDA_CHECK(cudaEventRecord(start.get(), stream_));

    if (globalRank == 0) {
      // Sender: issue all put_signals (each adds 1 to cumulative signal)
      for (int i = 0; i < kBenchmarkIters; i++) {
        launchIbgdaPutSignalWaitLocal(
            deviceTransportPtr,
            localBuf,
            remoteBuf,
            config.nBytes,
            1, // Each operation adds 1 to signal
            config.numBlocks,
            config.numThreads,
            stream_);
      }
    } else {
      // Receiver: wait for final cumulative signal value
      launchIbgdaWaitSignal(
          deviceTransportPtr,
          kBenchmarkIters, // Wait for all signals
          config.numBlocks,
          config.numThreads,
          stream_);
    }

    CUDA_CHECK(cudaEventRecord(stop.get(), stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    float totalTime_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime_ms, start.get(), stop.get()));
    float avgTime_ms = totalTime_ms / kBenchmarkIters;
    latencyUs = avgTime_ms * 1000.0f;

    // Unidirectional bandwidth
    float bandwidth_GBps = (config.nBytes / 1e9f) / (avgTime_ms / 1000.0f);

    // Reset signal for next test
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (globalRank == 1) {
      launchIbgdaResetSignal(deviceTransportPtr, stream_);
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    return bandwidth_GBps;
  }

  void printResultsTable(const std::vector<IbgdaBenchmarkResult>& results) {
    if (globalRank != 0) {
      return;
    }

    std::stringstream ss;
    ss << "\n";
    ss << "================================================================================\n";
    ss << "                    IBGDA Put+Signal Benchmark Results\n";
    ss << "================================================================================\n";
    ss << std::left << std::setw(20) << "Test Name" << std::right
       << std::setw(12) << "Msg Size" << std::right << std::setw(10) << "Blocks"
       << std::right << std::setw(10) << "Threads" << std::right
       << std::setw(12) << "BW (GB/s)" << std::right << std::setw(14)
       << "Latency (us)\n";
    ss << "--------------------------------------------------------------------------------\n";

    for (const auto& r : results) {
      std::string msgSize = formatSize(r.messageSize);

      ss << std::left << std::setw(20) << r.testName << std::right
         << std::setw(12) << msgSize << std::right << std::setw(10)
         << r.numBlocks << std::right << std::setw(10) << r.numThreads
         << std::right << std::setw(12) << std::fixed << std::setprecision(2)
         << r.bandwidth << std::right << std::setw(14) << std::fixed
         << std::setprecision(1) << r.latency << "\n";
    }
    ss << "================================================================================\n";
    ss << "BW = Bandwidth (message size / time), Latency = Average time per transfer\n";
    ss << "Warmup iterations: " << kWarmupIters
       << ", Benchmark iterations: " << kBenchmarkIters << "\n";
    ss << "================================================================================\n\n";

    XLOG(INFO) << ss.str();
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

  cudaStream_t stream_{};
};

TEST_F(IbgdaBenchmarkFixture, PutSignalWaitLocal) {
  if (numRanks != 2) {
    XLOGF(INFO, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;

  // Test configurations - various message sizes
  std::vector<IbgdaBenchmarkConfig> configs;

  // Small messages (latency-sensitive)
  configs.push_back(
      {.nBytes = 8, .numBlocks = 1, .numThreads = 32, .name = "8B"});
  configs.push_back(
      {.nBytes = 64, .numBlocks = 1, .numThreads = 32, .name = "64B"});
  configs.push_back(
      {.nBytes = 256, .numBlocks = 1, .numThreads = 32, .name = "256B"});
  configs.push_back(
      {.nBytes = 1024, .numBlocks = 1, .numThreads = 32, .name = "1KB"});
  configs.push_back(
      {.nBytes = 4 * 1024, .numBlocks = 1, .numThreads = 32, .name = "4KB"});

  // Medium messages
  configs.push_back(
      {.nBytes = 8 * 1024, .numBlocks = 1, .numThreads = 32, .name = "8KB"});
  configs.push_back(
      {.nBytes = 16 * 1024, .numBlocks = 1, .numThreads = 32, .name = "16KB"});
  configs.push_back(
      {.nBytes = 32 * 1024, .numBlocks = 1, .numThreads = 32, .name = "32KB"});
  configs.push_back(
      {.nBytes = 64 * 1024, .numBlocks = 1, .numThreads = 32, .name = "64KB"});
  configs.push_back(
      {.nBytes = 128 * 1024,
       .numBlocks = 1,
       .numThreads = 32,
       .name = "128KB"});
  configs.push_back(
      {.nBytes = 256 * 1024,
       .numBlocks = 1,
       .numThreads = 32,
       .name = "256KB"});
  configs.push_back(
      {.nBytes = 512 * 1024,
       .numBlocks = 1,
       .numThreads = 32,
       .name = "512KB"});

  // Large messages
  configs.push_back(
      {.nBytes = 1024 * 1024, .numBlocks = 1, .numThreads = 32, .name = "1MB"});
  configs.push_back(
      {.nBytes = 2 * 1024 * 1024,
       .numBlocks = 1,
       .numThreads = 32,
       .name = "2MB"});
  configs.push_back(
      {.nBytes = 4 * 1024 * 1024,
       .numBlocks = 1,
       .numThreads = 32,
       .name = "4MB"});
  configs.push_back(
      {.nBytes = 8 * 1024 * 1024,
       .numBlocks = 1,
       .numThreads = 32,
       .name = "8MB"});
  configs.push_back(
      {.nBytes = 16 * 1024 * 1024,
       .numBlocks = 1,
       .numThreads = 32,
       .name = "16MB"});
  configs.push_back(
      {.nBytes = 32 * 1024 * 1024,
       .numBlocks = 1,
       .numThreads = 32,
       .name = "32MB"});
  configs.push_back(
      {.nBytes = 64 * 1024 * 1024,
       .numBlocks = 1,
       .numThreads = 32,
       .name = "64MB"});
  configs.push_back(
      {.nBytes = 128 * 1024 * 1024,
       .numBlocks = 1,
       .numThreads = 32,
       .name = "128MB"});

  // Find max buffer size needed
  std::size_t maxBufferSize = 0;
  for (const auto& config : configs) {
    maxBufferSize = std::max(maxBufferSize, config.nBytes);
  }

  std::vector<IbgdaBenchmarkResult> results;

  try {
    // Create transport with max buffer size
    MultipeerIbgdaTransportConfig transportConfig{
        .cudaDevice = localRank,
        .dataBufferSize = maxBufferSize,
        .signalCount = 1,
    };

    auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
    MultipeerIbgdaTransport transport(
        globalRank, numRanks, bootstrap, transportConfig);
    transport.exchange();

    auto localDataBuf = transport.getDataBuffer(peerRank);
    auto remoteDataBuf = transport.getRemoteDataBuffer(peerRank);
    P2pIbgdaTransportDevice* deviceTransportPtr =
        transport.getDeviceTransportPtr();

    XLOGF(
        INFO,
        "Rank {}: Transport initialized, localBuf ptr={} lkey={}, remoteBuf ptr={} rkey={}",
        globalRank,
        localDataBuf.ptr,
        localDataBuf.lkey,
        remoteDataBuf.ptr,
        remoteDataBuf.rkey);

    for (const auto& config : configs) {
      IbgdaBenchmarkResult result;
      result.testName = config.name;
      result.messageSize = config.nBytes;
      result.numBlocks = config.numBlocks;
      result.numThreads = config.numThreads;

      result.bandwidth = runPutSignalBenchmark(
          deviceTransportPtr,
          localDataBuf,
          remoteDataBuf,
          config,
          result.latency);

      results.push_back(result);

      XLOGF(
          INFO,
          "Rank {}: {} - BW: {:.2f} GB/s, Latency: {:.1f} us",
          globalRank,
          config.name,
          result.bandwidth,
          result.latency);
    }

  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }

  printResultsTable(results);
}

TEST_F(IbgdaBenchmarkFixture, SignalOnlyLatency) {
  if (numRanks != 2) {
    XLOGF(INFO, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  try {
    MultipeerIbgdaTransportConfig transportConfig{
        .cudaDevice = localRank,
        .dataBufferSize = 4096, // Minimal buffer
        .signalCount = 1,
    };

    auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
    MultipeerIbgdaTransport transport(
        globalRank, numRanks, bootstrap, transportConfig);
    transport.exchange();

    P2pIbgdaTransportDevice* deviceTransportPtr =
        transport.getDeviceTransportPtr();

    CudaEvent start, stop;

    // Warmup - single iteration
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (globalRank == 0) {
      launchIbgdaSignalOnly(deviceTransportPtr, 1, 1, 32, stream_);
    } else {
      launchIbgdaWaitSignal(deviceTransportPtr, 1, 1, 32, stream_);
    }
    CUDA_CHECK_VOID(cudaStreamSynchronize(stream_));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Reset signal for benchmark
    if (globalRank == 1) {
      launchIbgdaResetSignal(deviceTransportPtr, stream_);
      CUDA_CHECK_VOID(cudaStreamSynchronize(stream_));
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Benchmark - sender issues all signals, receiver waits for final value
    CUDA_CHECK_VOID(cudaEventRecord(start.get(), stream_));

    if (globalRank == 0) {
      // Sender: issue all signals (each adds 1 to cumulative signal)
      for (int i = 0; i < kBenchmarkIters; i++) {
        launchIbgdaSignalOnly(deviceTransportPtr, 1, 1, 32, stream_);
      }
    } else {
      // Receiver: wait for final cumulative signal value
      launchIbgdaWaitSignal(
          deviceTransportPtr, kBenchmarkIters, 1, 32, stream_);
    }

    CUDA_CHECK_VOID(cudaEventRecord(stop.get(), stream_));
    CUDA_CHECK_VOID(cudaStreamSynchronize(stream_));

    float totalTime_ms = 0.0f;
    CUDA_CHECK_VOID(
        cudaEventElapsedTime(&totalTime_ms, start.get(), stop.get()));
    float avgLatencyUs = (totalTime_ms / kBenchmarkIters) * 1000.0f;

    if (globalRank == 0) {
      XLOGF(INFO, "\n=== Signal-Only Latency ===");
      XLOGF(INFO, "Average latency: {:.2f} us", avgLatencyUs);
      XLOGF(INFO, "===========================\n");
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
