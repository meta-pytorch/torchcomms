// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>
#include <nccl.h>

#include "comms/common/CudaWrap.h"
#include "comms/pipes/MultiPeerNvlTransport.h"
#include "comms/pipes/benchmarks/BenchmarkKernel.cuh"
#include "comms/pipes/benchmarks/BenchmarkMacros.h"
#include "comms/pipes/benchmarks/P2pNvlBenchmarkUtils.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

using meta::comms::CudaEvent;
using meta::comms::DeviceBuffer;
using meta::comms::MpiBaseTestFixture;
using meta::comms::MPIEnvironmentBase;

namespace comms::pipes::benchmark {

class P2pSignalBenchmarkFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    // Use localRank for cudaSetDevice since each node has its own set of GPUs
    // globalRank would fail on multi-node setups where rank > num_gpus_per_node
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

  // Helper function to run P2P signal benchmark - returns latency
  // in microseconds
  float runSignalBenchmark(
      comms::pipes::P2pNvlTransportDevice& p2p,
      const BenchmarkConfig& config,
      int nSteps = 1000) {
    XLOGF(DBG1, "=== Running Signal benchmark: {} ===", config.name);

    dim3 gridDim(config.numBlocks);
    dim3 blockDim(config.numThreads);

    CudaEvent start, stop;

    int nStepsArg = nSteps;
    SyncScope groupScope = config.groupScope;
    void* args[] = {&p2p, &nStepsArg, &groupScope};
    void* kernelFunc = (void*)comms::pipes::benchmark::p2pSignalBenchKernel;

    // Synchronize both ranks before starting to ensure both GPUs launch
    // together
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Benchmark
    CUDA_CHECK(cudaEventRecord(start.get(), stream_));
    CUDA_CHECK(
        cudaLaunchKernel(kernelFunc, gridDim, blockDim, args, 0, stream_));
    CUDA_CHECK(cudaEventRecord(stop.get(), stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    float totalTime_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime_ms, start.get(), stop.get()));

    // Calculate per-signal latency in microseconds
    float avgLatencyUs = (totalTime_ms / nSteps) * 1000.0f;

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    return avgLatencyUs;
  }

  ncclComm_t ncclComm_{};
  cudaStream_t stream_{};
};

TEST_F(P2pSignalBenchmarkFixture, SignalBenchmark) {
  // Only test with 2 ranks
  if (numRanks != 2) {
    XLOGF(DBG1, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;

  // Signal benchmark configurations using BenchmarkConfig
  std::vector<BenchmarkConfig> configs = {
      {.numBlocks = 1,
       .numThreads = 32,
       .groupScope = SyncScope::WARP,
       .name = "1b_32t_warp"},
      {.numBlocks = 1,
       .numThreads = 128,
       .groupScope = SyncScope::WARP,
       .name = "1b_warp"},
      {.numBlocks = 2,
       .numThreads = 128,
       .groupScope = SyncScope::WARP,
       .name = "2b_warp"},
      {.numBlocks = 4,
       .numThreads = 128,
       .groupScope = SyncScope::WARP,
       .name = "4b_warp"},
      {.numBlocks = 8,
       .numThreads = 128,
       .groupScope = SyncScope::WARP,
       .name = "8b_warp"},
      {.numBlocks = 16,
       .numThreads = 128,
       .groupScope = SyncScope::WARP,
       .name = "16b_warp"},
      {.numBlocks = 32,
       .numThreads = 128,
       .groupScope = SyncScope::WARP,
       .name = "32b_warp"},
      {.numBlocks = 1,
       .numThreads = 128,
       .groupScope = SyncScope::BLOCK,
       .name = "1b_block"},
      {.numBlocks = 2,
       .numThreads = 128,
       .groupScope = SyncScope::BLOCK,
       .name = "2b_block"},
      {.numBlocks = 4,
       .numThreads = 128,
       .groupScope = SyncScope::BLOCK,
       .name = "4b_block"},
      {.numBlocks = 8,
       .numThreads = 128,
       .groupScope = SyncScope::BLOCK,
       .name = "8b_block"},
      {.numBlocks = 16,
       .numThreads = 128,
       .groupScope = SyncScope::BLOCK,
       .name = "16b_block"},
      {.numBlocks = 32,
       .numThreads = 128,
       .groupScope = SyncScope::BLOCK,
       .name = "32b_block"},
  };

  const int nSteps = 1000; // Number of signal iterations per kernel launch

  // GPU warmup phase - run one iteration to avoid cold start overhead
  {
    const auto& warmupConfig = configs[0];
    std::size_t signalCount = warmupConfig.groupScope == SyncScope::BLOCK
        ? warmupConfig.numBlocks
        : warmupConfig.numBlocks * (warmupConfig.numThreads / 32);

    comms::pipes::MultiPeerNvlTransportConfig p2pConfig{
        .dataBufferSize = 1,
        .chunkSize = 1,
        .pipelineDepth = 1,
        .signalCount = signalCount,
    };

    auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
    comms::pipes::MultiPeerNvlTransport transport(
        globalRank, numRanks, bootstrap, p2pConfig);
    transport.exchange();

    auto p2p = transport.getP2pTransportDevice(peerRank);
    runSignalBenchmark(p2p, warmupConfig, nSteps); // Discard result
  }

  std::vector<BenchmarkResult> results;

  for (const auto& config : configs) {
    // Calculate signalCount for this config
    // For warp groups: numBlocks * (numThreads / 32)
    // For block groups: numBlocks
    std::size_t signalCount = config.groupScope == SyncScope::BLOCK
        ? config.numBlocks
        : config.numBlocks * (config.numThreads / 32);

    // Create fresh P2P transport for each config to reset signal buffers
    comms::pipes::MultiPeerNvlTransportConfig p2pConfig{
        .dataBufferSize = 1,
        .chunkSize = 1,
        .pipelineDepth = 1,
        .signalCount = signalCount,
    };

    auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
    comms::pipes::MultiPeerNvlTransport transport(
        globalRank, numRanks, bootstrap, p2pConfig);
    transport.exchange();

    auto p2p = transport.getP2pTransportDevice(peerRank);

    BenchmarkResult result;
    result.testName = config.name;
    result.p2pTime = runSignalBenchmark(p2p, config, nSteps);
    results.push_back(result);
  }

  // Print results
  if (globalRank == 0) {
    std::stringstream ss;
    ss << "\n";
    ss << "================================================================\n";
    ss << "              P2P NVLink Signal Benchmark Results\n";
    ss << "================================================================\n";
    ss << std::left << std::setw(20) << "Config" << std::right << std::setw(15)
       << "Latency (us)\n";
    ss << "----------------------------------------------------------------\n";

    for (const auto& r : results) {
      ss << std::left << std::setw(20) << r.testName << std::right
         << std::setw(15) << std::fixed << std::setprecision(3) << r.p2pTime
         << "\n";
    }
    ss << "================================================================\n";
    ss << "Latency = Average time per signal/wait pair\n";
    ss << "Each measurement: 1 kernel launches x " << nSteps
       << " signal+wait/launch\n";
    ss << "================================================================\n\n";

    std::cout << ss.str();
  }
}

} // namespace comms::pipes::benchmark

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
