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

// Result struct for send/recv vs send_one/recv_one comparison
struct SendRecvOneResult {
  std::string testName;
  std::size_t messageSize{};
  std::size_t stagingBufferSize{};
  std::size_t pipelineDepth{};
  std::size_t chunkSize{};
  int numBlocks{};
  int numThreads{};
  float sendRecvLatencyUs{};
  float sendRecvOneLatencyUs{};
  float speedup{}; // sendRecvLatency / sendRecvOneLatency
  float latencyDiffUs{}; // sendRecvLatency - sendRecvOneLatency
};

class P2pSendRecvOneBenchmarkFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
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

  // Run send/recv benchmark - returns latency in microseconds
  float runSendRecvBenchmark(
      comms::pipes::P2pNvlTransportDevice& p2p,
      const BenchmarkConfig& config) {
    DeviceBuffer sendBuff(config.nBytes);
    DeviceBuffer recvBuff(config.nBytes);

    // Initialize buffers
    if (globalRank == 0) {
      CUDA_CHECK(cudaMemset(sendBuff.get(), 1, config.nBytes));
    }
    if (globalRank == 1) {
      CUDA_CHECK(cudaMemset(recvBuff.get(), 0, config.nBytes));
    }

    dim3 gridDim(config.numBlocks);
    dim3 blockDim(config.numThreads);

    CudaEvent start, stop;

    std::size_t nBytes = config.nBytes;
    bool isSend = (globalRank == 0);
    SyncScope groupScope = config.groupScope;
    void* devicePtr = (isSend ? sendBuff.get() : recvBuff.get());
    Timeout timeout; // Default timeout (disabled)
    void* args[] = {&p2p, &devicePtr, &nBytes, &groupScope, &timeout};
    void* kernelFunc = isSend ? (void*)comms::pipes::benchmark::p2pSend
                              : (void*)comms::pipes::benchmark::p2pRecv;

    // Use cluster launch for better SM utilization
    dim3 defaultClusterDim(comms::common::kDefaultClusterSize, 1, 1);
    std::optional<dim3> clusterDimOpt = config.spreadClusterLaunch
        ? std::optional{defaultClusterDim}
        : std::nullopt;

    // Warmup
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    for (int i = 0; i < kWarmupIters; i++) {
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
      CUDA_CHECK(
          comms::common::launchKernel(
              kernelFunc, gridDim, blockDim, args, nullptr, clusterDimOpt));
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Benchmark
    CUDA_CHECK(cudaEventRecord(start.get()));
    for (int i = 0; i < kBenchmarkIters; i++) {
      CUDA_CHECK(
          comms::common::launchKernel(
              kernelFunc, gridDim, blockDim, args, nullptr, clusterDimOpt));
    }
    CUDA_CHECK(cudaEventRecord(stop.get()));
    CUDA_CHECK(cudaDeviceSynchronize());

    float totalTime_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime_ms, start.get(), stop.get()));
    float avgTime_ms = totalTime_ms / kBenchmarkIters;
    float latencyUs = avgTime_ms * 1000.0f;

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    return latencyUs;
  }

  // Run send_one/recv_one benchmark - returns latency in microseconds
  float runSendRecvOneBenchmark(
      comms::pipes::P2pNvlTransportDevice& p2p,
      const BenchmarkConfig& config) {
    DeviceBuffer sendBuff(config.nBytes);
    DeviceBuffer recvBuff(config.nBytes);

    // Initialize buffers
    if (globalRank == 0) {
      CUDA_CHECK(cudaMemset(sendBuff.get(), 1, config.nBytes));
    }
    if (globalRank == 1) {
      CUDA_CHECK(cudaMemset(recvBuff.get(), 0, config.nBytes));
    }

    dim3 gridDim(config.numBlocks);
    dim3 blockDim(config.numThreads);

    CudaEvent start, stop;

    std::size_t nBytes = config.nBytes;
    bool isSend = (globalRank == 0);
    SyncScope groupScope = config.groupScope;
    void* devicePtr = (isSend ? sendBuff.get() : recvBuff.get());

    void* sendArgs[] = {&p2p, &devicePtr, &nBytes, &groupScope};
    void* recvArgs[] = {&p2p, &devicePtr, &groupScope};

    void* kernelFunc = isSend ? (void*)comms::pipes::benchmark::p2pSendOne
                              : (void*)comms::pipes::benchmark::p2pRecvOne;

    // Use cluster launch for better SM utilization
    dim3 defaultClusterDim(comms::common::kDefaultClusterSize, 1, 1);
    std::optional<dim3> clusterDimOpt = config.spreadClusterLaunch
        ? std::optional{defaultClusterDim}
        : std::nullopt;

    // Warmup
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    for (int i = 0; i < kWarmupIters; i++) {
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
      if (isSend) {
        CUDA_CHECK(
            comms::common::launchKernel(
                kernelFunc,
                gridDim,
                blockDim,
                sendArgs,
                nullptr,
                clusterDimOpt));
      } else {
        CUDA_CHECK(
            comms::common::launchKernel(
                kernelFunc,
                gridDim,
                blockDim,
                recvArgs,
                nullptr,
                clusterDimOpt));
      }
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Benchmark
    CUDA_CHECK(cudaEventRecord(start.get()));
    for (int i = 0; i < kBenchmarkIters; i++) {
      if (isSend) {
        CUDA_CHECK(
            comms::common::launchKernel(
                kernelFunc,
                gridDim,
                blockDim,
                sendArgs,
                nullptr,
                clusterDimOpt));
      } else {
        CUDA_CHECK(
            comms::common::launchKernel(
                kernelFunc,
                gridDim,
                blockDim,
                recvArgs,
                nullptr,
                clusterDimOpt));
      }
    }
    CUDA_CHECK(cudaEventRecord(stop.get()));
    CUDA_CHECK(cudaDeviceSynchronize());

    float totalTime_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime_ms, start.get(), stop.get()));
    float avgTime_ms = totalTime_ms / kBenchmarkIters;
    float latencyUs = avgTime_ms * 1000.0f;

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    return latencyUs;
  }

  ncclComm_t ncclComm_{};
  cudaStream_t stream_{};
};

TEST_F(P2pSendRecvOneBenchmarkFixture, SendRecvVsSendRecvOne) {
  // Only test with 2 ranks
  if (numRanks != 2) {
    XLOGF(DBG1, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;

  // Sweep message sizes from 128KB to 64MB
  // Using optimal settings based on P2pSendRecvBenchmark results
  std::vector<BenchmarkConfig> configs;

  // 128KB: 4 blocks, 8KB chunks
  configs.push_back({
      .nBytes = 128 * 1024,
      .stagedBufferSize = 128 * 1024,
      .numBlocks = 4,
      .numThreads = 128,
      .pipelineDepth = 2,
      .chunkSize = 8 * 1024,
      .groupScope = SyncScope::WARP,
      .spreadClusterLaunch = true,
      .name = "128KB",
  });

  // 256KB: 8 blocks, 8KB chunks
  configs.push_back({
      .nBytes = 256 * 1024,
      .stagedBufferSize = 256 * 1024,
      .numBlocks = 8,
      .numThreads = 128,
      .pipelineDepth = 2,
      .chunkSize = 8 * 1024,
      .groupScope = SyncScope::WARP,
      .spreadClusterLaunch = true,
      .name = "256KB",
  });

  // 512KB: 16 blocks, 16KB chunks
  configs.push_back({
      .nBytes = 512 * 1024,
      .stagedBufferSize = 512 * 1024,
      .numBlocks = 16,
      .numThreads = 128,
      .pipelineDepth = 2,
      .chunkSize = 16 * 1024,
      .groupScope = SyncScope::WARP,
      .spreadClusterLaunch = true,
      .name = "512KB",
  });

  // 1MB: 32 blocks, 32KB chunks
  configs.push_back({
      .nBytes = 1024 * 1024,
      .stagedBufferSize = 1024 * 1024,
      .numBlocks = 32,
      .numThreads = 128,
      .pipelineDepth = 2,
      .chunkSize = 32 * 1024,
      .groupScope = SyncScope::WARP,
      .spreadClusterLaunch = true,
      .name = "1MB",
  });

  // 2MB: 64 blocks, 16KB chunks
  configs.push_back({
      .nBytes = 2 * 1024 * 1024,
      .stagedBufferSize = 2 * 1024 * 1024,
      .numBlocks = 64,
      .numThreads = 128,
      .pipelineDepth = 2,
      .chunkSize = 16 * 1024,
      .groupScope = SyncScope::WARP,
      .spreadClusterLaunch = true,
      .name = "2MB",
  });

  // 4MB: 64 blocks, 32KB chunks
  configs.push_back({
      .nBytes = 4 * 1024 * 1024,
      .stagedBufferSize = 4 * 1024 * 1024,
      .numBlocks = 64,
      .numThreads = 128,
      .pipelineDepth = 2,
      .chunkSize = 32 * 1024,
      .groupScope = SyncScope::BLOCK,
      .spreadClusterLaunch = true,
      .name = "4MB",
  });

  // 8MB: 128 blocks, 64KB chunks
  configs.push_back({
      .nBytes = 8 * 1024 * 1024,
      .stagedBufferSize = 8 * 1024 * 1024,
      .numBlocks = 128,
      .numThreads = 128,
      .pipelineDepth = 4,
      .chunkSize = 64 * 1024,
      .groupScope = SyncScope::BLOCK,
      .spreadClusterLaunch = true,
      .name = "8MB",
  });

  // 16MB: 128 blocks, 128KB chunks
  configs.push_back({
      .nBytes = 16 * 1024 * 1024,
      .stagedBufferSize = 16 * 1024 * 1024,
      .numBlocks = 128,
      .numThreads = 128,
      .pipelineDepth = 4,
      .chunkSize = 128 * 1024,
      .groupScope = SyncScope::BLOCK,
      .spreadClusterLaunch = true,
      .name = "16MB",
  });

  // 32MB: 128 blocks, 128KB chunks
  configs.push_back({
      .nBytes = 32 * 1024 * 1024,
      .stagedBufferSize = 32 * 1024 * 1024,
      .numBlocks = 128,
      .numThreads = 128,
      .pipelineDepth = 4,
      .chunkSize = 128 * 1024,
      .groupScope = SyncScope::BLOCK,
      .spreadClusterLaunch = true,
      .name = "32MB",
  });

  // 64MB: 128 blocks, 128KB chunks
  configs.push_back({
      .nBytes = 64 * 1024 * 1024,
      .stagedBufferSize = 32 * 1024 * 1024,
      .numBlocks = 128,
      .numThreads = 128,
      .pipelineDepth = 4,
      .chunkSize = 128 * 1024,
      .groupScope = SyncScope::BLOCK,
      .spreadClusterLaunch = true,
      .name = "64MB",
  });

  // 128MB: 256 blocks, 256KB chunks
  configs.push_back({
      .nBytes = 128 * 1024 * 1024,
      .stagedBufferSize = 64 * 1024 * 1024,
      .numBlocks = 256,
      .numThreads = 128,
      .pipelineDepth = 4,
      .chunkSize = 256 * 1024,
      .groupScope = SyncScope::BLOCK,
      .spreadClusterLaunch = true,
      .name = "128MB",
  });

  // 256MB: 256 blocks, 512KB chunks
  configs.push_back({
      .nBytes = 256 * 1024 * 1024,
      .stagedBufferSize = 64 * 1024 * 1024,
      .numBlocks = 256,
      .numThreads = 128,
      .pipelineDepth = 4,
      .chunkSize = 512 * 1024,
      .groupScope = SyncScope::BLOCK,
      .spreadClusterLaunch = true,
      .name = "256MB",
  });

  std::vector<SendRecvOneResult> results;

  for (const auto& config : configs) {
    // Create P2P transport for this configuration
    comms::pipes::MultiPeerNvlTransportConfig p2pConfig{
        .dataBufferSize = config.stagedBufferSize,
        .chunkSize = config.chunkSize,
        .pipelineDepth = config.pipelineDepth,
    };

    auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
    comms::pipes::MultiPeerNvlTransport transport(
        globalRank, numRanks, bootstrap, p2pConfig);
    transport.exchange();

    auto p2p = transport.getP2pTransportDevice(peerRank);

    SendRecvOneResult result;
    result.testName = config.name;
    result.messageSize = config.nBytes;
    result.stagingBufferSize = config.stagedBufferSize;
    result.pipelineDepth = config.pipelineDepth;
    result.chunkSize = config.chunkSize;
    result.numBlocks = config.numBlocks;
    result.numThreads = config.numThreads;

    // Run send/recv benchmark
    result.sendRecvLatencyUs = runSendRecvBenchmark(p2p, config);

    // Run send_one/recv_one benchmark
    result.sendRecvOneLatencyUs = runSendRecvOneBenchmark(p2p, config);

    // Calculate speedup and latency difference
    result.speedup = (result.sendRecvOneLatencyUs > 0)
        ? result.sendRecvLatencyUs / result.sendRecvOneLatencyUs
        : 0;
    result.latencyDiffUs =
        result.sendRecvLatencyUs - result.sendRecvOneLatencyUs;

    results.push_back(result);
  }

  // Print results
  if (globalRank == 0) {
    std::stringstream ss;
    ss << "\n";
    ss << "============================================================================================================\n";
    ss << "                      send/recv vs send_one/recv_one Benchmark Results\n";
    ss << "============================================================================================================\n";
    ss << std::left << std::setw(12) << "Msg Size" << std::right
       << std::setw(12) << "Staging" << std::right << std::setw(6) << "PD"
       << std::right << std::setw(10) << "Chunk" << std::right << std::setw(8)
       << "Blocks" << std::right << std::setw(9) << "Threads" << std::right
       << std::setw(16) << "send/recv (us)" << std::right << std::setw(18)
       << "send_one/recv_one" << std::right << std::setw(10) << "Speedup"
       << std::right << std::setw(12) << "Diff (us)\n";
    ss << "------------------------------------------------------------------------------------------------------------\n";

    for (const auto& r : results) {
      std::string msgSize = formatSize(r.messageSize);
      std::string stagingSize = formatSize(r.stagingBufferSize);
      std::string chunkSizeStr = formatSize(r.chunkSize);

      ss << std::left << std::setw(12) << msgSize << std::right << std::setw(12)
         << stagingSize << std::right << std::setw(6) << r.pipelineDepth
         << std::right << std::setw(10) << chunkSizeStr << std::right
         << std::setw(8) << r.numBlocks << std::right << std::setw(9)
         << r.numThreads << std::right << std::setw(16) << std::fixed
         << std::setprecision(1) << r.sendRecvLatencyUs << std::right
         << std::setw(18) << std::fixed << std::setprecision(1)
         << r.sendRecvOneLatencyUs << std::right << std::setw(9) << std::fixed
         << std::setprecision(2) << r.speedup << "x" << std::right
         << std::setw(12) << std::fixed << std::setprecision(1)
         << r.latencyDiffUs << "\n";
    }
    ss << "============================================================================================================\n";
    ss << "Speedup = send/recv latency / send_one/recv_one latency\n";
    ss << "Diff = send/recv latency - send_one/recv_one latency (positive = send_one faster)\n";
    ss << "============================================================================================================\n\n";

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
