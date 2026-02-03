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

// Result struct for send/recv vs send_multiple/recv_multiple comparison
struct SendRecvMultipleResult {
  std::string testName;
  std::size_t messageSize{};
  std::size_t stagingBufferSize{};
  std::size_t pipelineDepth{};
  std::size_t chunkSize{};
  int numBlocks{};
  int numThreads{};
  int numChunks{};
  float sendRecvLatencyUs{};
  float sendRecvMultipleLatencyUs{};
  float speedup{}; // sendRecvLatency / sendRecvMultipleLatency
  float latencyDiffUs{}; // sendRecvLatency - sendRecvMultipleLatency
};

class P2pSendRecvMultipleBenchmarkFixture : public MpiBaseTestFixture {
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

  // Run send_multiple/recv_multiple benchmark - returns latency in microseconds
  float runSendRecvMultipleBenchmark(
      comms::pipes::P2pNvlTransportDevice& p2p,
      const BenchmarkConfig& config,
      int numChunks) {
    DeviceBuffer sendBuff(config.nBytes);
    DeviceBuffer recvBuff(config.nBytes);

    // Initialize buffers
    if (globalRank == 0) {
      CUDA_CHECK(cudaMemset(sendBuff.get(), 1, config.nBytes));
    }
    if (globalRank == 1) {
      CUDA_CHECK(cudaMemset(recvBuff.get(), 0, config.nBytes));
    }

    // Create chunk sizes and indices on device
    std::size_t chunkSizePerPiece = config.nBytes / numChunks;
    std::vector<std::size_t> hostChunkSizes(numChunks, chunkSizePerPiece);
    std::vector<std::size_t> hostChunkIndices(numChunks);
    for (int i = 0; i < numChunks; i++) {
      hostChunkIndices[i] = i;
    }

    // Allocate device memory for chunk sizes and indices
    std::size_t* deviceChunkSizes;
    std::size_t* deviceChunkIndices;
    CUDA_CHECK(cudaMalloc(&deviceChunkSizes, numChunks * sizeof(std::size_t)));
    CUDA_CHECK(
        cudaMalloc(&deviceChunkIndices, numChunks * sizeof(std::size_t)));
    CUDA_CHECK(cudaMemcpy(
        deviceChunkSizes,
        hostChunkSizes.data(),
        numChunks * sizeof(std::size_t),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        deviceChunkIndices,
        hostChunkIndices.data(),
        numChunks * sizeof(std::size_t),
        cudaMemcpyHostToDevice));

    DeviceSpan<const std::size_t> chunkSizesSpan(deviceChunkSizes, numChunks);
    DeviceSpan<const std::size_t> chunkIndicesSpan(
        deviceChunkIndices, numChunks);
    DeviceSpan<std::size_t> chunkSizesMutableSpan(deviceChunkSizes, numChunks);

    dim3 gridDim(config.numBlocks);
    dim3 blockDim(config.numThreads);

    CudaEvent start, stop;

    bool isSend = (globalRank == 0);
    SyncScope groupScope = config.groupScope;
    void* devicePtr = (isSend ? sendBuff.get() : recvBuff.get());

    void* sendArgs[] = {
        &p2p, &devicePtr, &chunkSizesSpan, &chunkIndicesSpan, &groupScope};
    void* recvArgs[] = {&p2p, &devicePtr, &chunkSizesMutableSpan, &groupScope};

    void* kernelFunc = isSend ? (void*)comms::pipes::benchmark::p2pSendMultiple
                              : (void*)comms::pipes::benchmark::p2pRecvMultiple;

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

    CUDA_CHECK(cudaFree(deviceChunkSizes));
    CUDA_CHECK(cudaFree(deviceChunkIndices));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    return latencyUs;
  }

  ncclComm_t ncclComm_{};
  cudaStream_t stream_{};
};

TEST_F(P2pSendRecvMultipleBenchmarkFixture, SendRecvVsSendRecvMultiple) {
  // Only test with 2 ranks
  if (numRanks != 2) {
    XLOGF(DBG1, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;

  // Test different message sizes with 2 and 4 chunks
  std::vector<std::size_t> messageSizes = {
      128 * 1024, // 128KB
      256 * 1024, // 256KB
      512 * 1024, // 512KB
      1024 * 1024, // 1MB
      2 * 1024 * 1024, // 2MB
      4 * 1024 * 1024, // 4MB
      8 * 1024 * 1024, // 8MB
  };

  std::vector<int> chunkCounts = {2, 4};

  std::vector<SendRecvMultipleResult> results;

  for (std::size_t msgSize : messageSizes) {
    for (int numChunks : chunkCounts) {
      XLOGF(
          INFO,
          "Rank {}: Testing {} with {} chunks...",
          globalRank,
          formatSize(msgSize),
          numChunks);

      // Determine config based on message size
      BenchmarkConfig config;
      config.nBytes = msgSize;
      config.numThreads = 128;
      config.pipelineDepth = (msgSize >= 8 * 1024 * 1024) ? 4 : 2;
      config.spreadClusterLaunch = true;

      if (msgSize <= 128 * 1024) {
        config.stagedBufferSize = msgSize;
        config.numBlocks = 4;
        config.chunkSize = 8 * 1024;
        config.groupScope = SyncScope::WARP;
      } else if (msgSize <= 256 * 1024) {
        config.stagedBufferSize = msgSize;
        config.numBlocks = 8;
        config.chunkSize = 8 * 1024;
        config.groupScope = SyncScope::WARP;
      } else if (msgSize <= 512 * 1024) {
        config.stagedBufferSize = msgSize;
        config.numBlocks = 16;
        config.chunkSize = 16 * 1024;
        config.groupScope = SyncScope::WARP;
      } else if (msgSize <= 1024 * 1024) {
        config.stagedBufferSize = msgSize;
        config.numBlocks = 32;
        config.chunkSize = 32 * 1024;
        config.groupScope = SyncScope::WARP;
      } else if (msgSize <= 2 * 1024 * 1024) {
        config.stagedBufferSize = msgSize;
        config.numBlocks = 64;
        config.chunkSize = 16 * 1024;
        config.groupScope = SyncScope::WARP;
      } else if (msgSize <= 4 * 1024 * 1024) {
        config.stagedBufferSize = msgSize;
        config.numBlocks = 64;
        config.chunkSize = 32 * 1024;
        config.groupScope = SyncScope::BLOCK;
      } else { // 8MB
        config.stagedBufferSize = msgSize;
        config.numBlocks = 128;
        config.chunkSize = 64 * 1024;
        config.groupScope = SyncScope::BLOCK;
      }

      // Create P2P transport
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

      SendRecvMultipleResult result;
      std::size_t chunkSizePerPiece = msgSize / numChunks;
      result.testName = formatSize(msgSize) + "_" + std::to_string(numChunks) +
          "x" + formatSize(chunkSizePerPiece);
      result.messageSize = msgSize;
      result.stagingBufferSize = config.stagedBufferSize;
      result.pipelineDepth = config.pipelineDepth;
      result.chunkSize = config.chunkSize;
      result.numBlocks = config.numBlocks;
      result.numThreads = config.numThreads;
      result.numChunks = numChunks;

      // Run send/recv benchmark
      result.sendRecvLatencyUs = runSendRecvBenchmark(p2p, config);

      // Run send_multiple benchmark
      result.sendRecvMultipleLatencyUs =
          runSendRecvMultipleBenchmark(p2p, config, numChunks);

      // Calculate speedup and latency difference
      result.speedup = (result.sendRecvMultipleLatencyUs > 0)
          ? result.sendRecvLatencyUs / result.sendRecvMultipleLatencyUs
          : 0;
      result.latencyDiffUs =
          result.sendRecvLatencyUs - result.sendRecvMultipleLatencyUs;

      results.push_back(result);

      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    }
  }

  // Print results
  if (globalRank == 0) {
    std::stringstream ss;
    ss << "\n";
    ss << "======================================================================================================================\n";
    ss << "                        send/recv vs send_multiple/recv_multiple Benchmark Results\n";
    ss << "======================================================================================================================\n";
    ss << std::left << std::setw(20) << "Test Name" << std::right
       << std::setw(10) << "Staging" << std::right << std::setw(5) << "PD"
       << std::right << std::setw(9) << "Chunk" << std::right << std::setw(7)
       << "Blocks" << std::right << std::setw(8) << "Threads" << std::right
       << std::setw(8) << "#Chunks" << std::right << std::setw(14)
       << "send/recv" << std::right << std::setw(16) << "send_multiple"
       << std::right << std::setw(9) << "Speedup" << std::right << std::setw(11)
       << "Diff (us)\n";
    ss << "----------------------------------------------------------------------------------------------------------------------\n";

    for (const auto& r : results) {
      std::string stagingSize = formatSize(r.stagingBufferSize);
      std::string chunkSizeStr = formatSize(r.chunkSize);

      ss << std::left << std::setw(20) << r.testName << std::right
         << std::setw(10) << stagingSize << std::right << std::setw(5)
         << r.pipelineDepth << std::right << std::setw(9) << chunkSizeStr
         << std::right << std::setw(7) << r.numBlocks << std::right
         << std::setw(8) << r.numThreads << std::right << std::setw(8)
         << r.numChunks << std::right << std::setw(14) << std::fixed
         << std::setprecision(1) << r.sendRecvLatencyUs << std::right
         << std::setw(16) << std::fixed << std::setprecision(1)
         << r.sendRecvMultipleLatencyUs << std::right << std::setw(8)
         << std::fixed << std::setprecision(2) << r.speedup << "x" << std::right
         << std::setw(11) << std::fixed << std::setprecision(1)
         << r.latencyDiffUs << "\n";
    }
    ss << "======================================================================================================================\n";
    ss << "Speedup = send/recv latency / send_multiple/recv_multiple latency\n";
    ss << "Diff = send/recv latency - send_multiple latency (positive = send_multiple faster)\n";
    ss << "======================================================================================================================\n\n";

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
