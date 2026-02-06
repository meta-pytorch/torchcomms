// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>
#include <nccl.h>

#include "comms/common/CudaWrap.h"
#include "comms/pipes/MultiPeerNvlTransport.h"
#include "comms/pipes/benchmarks/BenchmarkKernel.cuh"
#include "comms/pipes/benchmarks/BenchmarkMacros.h"
#include "comms/pipes/benchmarks/P2pNvlBenchmarkUtils.h"
#include "comms/testinfra/BenchmarkTestFixture.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

using meta::comms::CudaEvent;
using meta::comms::DeviceBuffer;

namespace comms::pipes::benchmark {

class P2pSendRecvBenchmarkFixture : public meta::comms::BenchmarkTestFixture {
 protected:
  void SetUp() override {
    // Initialize bootstrap and rank variables from base class
    BenchmarkTestFixture::SetUp();

    // Use localRank for cudaSetDevice since each node has its own set of GPUs
    // globalRank would fail on multi-node setups where rank > num_gpus_per_node
    CUDA_CHECK_VOID(cudaSetDevice(localRank));

    // Initialize NCCL
    NCCL_CHECK_VOID(
        ncclCommInitRank(&ncclComm_, worldSize, getNCCLId(), globalRank));
    CUDA_CHECK_VOID(cudaStreamCreate(&stream_));
  }

  void TearDown() override {
    NCCL_CHECK_VOID(ncclCommDestroy(ncclComm_));
    CUDA_CHECK_VOID(cudaStreamDestroy(stream_));
    BenchmarkTestFixture::TearDown();
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

    // Broadcast NCCL ID using bootstrap allGather
    std::vector<ncclUniqueId> allIds(worldSize);
    allIds[globalRank] = id;
    auto result =
        bootstrap
            ->allGather(
                allIds.data(), sizeof(ncclUniqueId), globalRank, worldSize)
            .get();
    if (result != 0) {
      XLOG(ERR) << "Bootstrap allGather for NCCL ID failed";
      std::abort();
    }
    id = allIds[0]; // Take rank 0's ID
    return id;
  }

  // Helper function to run NCCL benchmark - returns bandwidth
  float runNcclBenchmark(const BenchmarkConfig& config, float& timeUs) {
    XLOGF(DBG1, "=== Running NCCL benchmark: {} ===", config.name);

    DeviceBuffer sendBuff(config.nBytes);
    DeviceBuffer recvBuff(config.nBytes);

    // Initialize buffers
    if (globalRank == 0) {
      CUDA_CHECK(cudaMemset(sendBuff.get(), 1, config.nBytes));
    }
    if (globalRank == 1) {
      CUDA_CHECK(cudaMemset(recvBuff.get(), 0, config.nBytes));
    }

    CudaEvent start, stop;

    // Warmup
    bootstrap->barrierAll();
    for (int i = 0; i < kWarmupIters; i++) {
      if (globalRank == 0) {
        NCCL_CHECK(ncclSend(
            sendBuff.get(), config.nBytes, ncclChar, 1, ncclComm_, stream_));
      } else if (globalRank == 1) {
        NCCL_CHECK(ncclRecv(
            recvBuff.get(), config.nBytes, ncclChar, 0, ncclComm_, stream_));
      }
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    bootstrap->barrierAll();

    // Benchmark - measure time across all iterations
    // No barrier between iterations - rely on NCCL's internal synchronization
    CUDA_CHECK(cudaEventRecord(start.get(), stream_));
    for (int i = 0; i < kBenchmarkIters; i++) {
      if (globalRank == 0) {
        NCCL_CHECK(ncclSend(
            sendBuff.get(), config.nBytes, ncclChar, 1, ncclComm_, stream_));
      } else if (globalRank == 1) {
        NCCL_CHECK(ncclRecv(
            recvBuff.get(), config.nBytes, ncclChar, 0, ncclComm_, stream_));
      }
    }
    CUDA_CHECK(cudaEventRecord(stop.get(), stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    float totalTime_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime_ms, start.get(), stop.get()));
    float avgTime_ms = totalTime_ms / kBenchmarkIters;
    timeUs = avgTime_ms * 1000.0f;
    // Unidirectional bandwidth: data transferred in one direction / time
    float bandwidth_GBps = (config.nBytes / 1e9f) / (avgTime_ms / 1000.0f);

    bootstrap->barrierAll();

    return bandwidth_GBps;
  }

  // Helper function to run P2P NVL benchmark - returns bandwidth
  float runP2pNvlBenchmark(
      comms::pipes::P2pNvlTransportDevice& p2p,
      const BenchmarkConfig& config,
      float& timeUs) {
    XLOGF(DBG1, "=== Running P2P NVL benchmark: {} ===", config.name);

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

    cudaStream_t sendStream, recvStream;
    CUDA_CHECK(cudaStreamCreate(&sendStream));
    CUDA_CHECK(cudaStreamCreate(&recvStream));

    CudaEvent start, stop;

    std::size_t nBytes = config.nBytes;
    bool isSend = (globalRank == 0);
    SyncScope groupScope = config.groupScope;
    void* devicePtr = (isSend ? sendBuff.get() : recvBuff.get());
    Timeout timeout; // Default timeout (disabled)
    void* args[] = {&p2p, &devicePtr, &nBytes, &groupScope, &timeout};
    void* kernelFunc = isSend ? (void*)comms::pipes::benchmark::p2pSend
                              : (void*)comms::pipes::benchmark::p2pRecv;
    cudaStream_t stream = isSend ? sendStream : recvStream;

    // Warmup
    bootstrap->barrierAll();
    for (int i = 0; i < kWarmupIters; i++) {
      bootstrap->barrierAll();
      CUDA_CHECK(
          cudaLaunchKernel(kernelFunc, gridDim, blockDim, args, 0, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    bootstrap->barrierAll();

    // Benchmark - measure time across all iterations
    // No barrier between iterations - ChunkState provides synchronization
    CUDA_CHECK(cudaEventRecord(start.get(), stream));
    for (int i = 0; i < kBenchmarkIters; i++) {
      CUDA_CHECK(
          cudaLaunchKernel(kernelFunc, gridDim, blockDim, args, 0, stream));
    }
    CUDA_CHECK(cudaEventRecord(stop.get(), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float totalTime_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime_ms, start.get(), stop.get()));
    float avgTime_ms = totalTime_ms / kBenchmarkIters;
    timeUs = avgTime_ms * 1000.0f;
    // Unidirectional bandwidth: data transferred in one direction / time
    float bandwidth_GBps = (config.nBytes / 1e9f) / (avgTime_ms / 1000.0f);

    CUDA_CHECK(cudaStreamDestroy(sendStream));
    CUDA_CHECK(cudaStreamDestroy(recvStream));

    bootstrap->barrierAll();

    return bandwidth_GBps;
  }

  // Helper function to run NCCL bidirectional benchmark - returns algorithm BW
  float runNcclBidirectionalBenchmark(
      const BenchmarkConfig& config,
      float& timeUs) {
    XLOGF(
        DBG1,
        "Rank {}: Starting NCCL bidirectional benchmark: {}",
        globalRank,
        config.name);

    DeviceBuffer sendBuff(config.nBytes);
    DeviceBuffer recvBuff(config.nBytes);

    // Initialize buffers - each rank sends its own data
    CUDA_CHECK(cudaMemset(sendBuff.get(), globalRank, config.nBytes));
    CUDA_CHECK(cudaMemset(recvBuff.get(), 0, config.nBytes));

    int peerRank = (globalRank == 0) ? 1 : 0;

    CudaEvent start, stop;

    // Warmup
    XLOGF(DBG1, "Rank {}: NCCL bidi warmup starting", globalRank);
    bootstrap->barrierAll();
    for (int i = 0; i < kWarmupIters; i++) {
      NCCL_CHECK(ncclGroupStart());
      NCCL_CHECK(ncclSend(
          sendBuff.get(),
          config.nBytes,
          ncclChar,
          peerRank,
          ncclComm_,
          stream_));
      NCCL_CHECK(ncclRecv(
          recvBuff.get(),
          config.nBytes,
          ncclChar,
          peerRank,
          ncclComm_,
          stream_));
      NCCL_CHECK(ncclGroupEnd());
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    XLOGF(DBG1, "Rank {}: NCCL bidi warmup complete", globalRank);

    // Benchmark - measure time across all iterations
    // No barrier between iterations - rely on NCCL's internal synchronization
    CUDA_CHECK(cudaEventRecord(start.get(), stream_));
    for (int i = 0; i < kBenchmarkIters; i++) {
      NCCL_CHECK(ncclGroupStart());
      NCCL_CHECK(ncclSend(
          sendBuff.get(),
          config.nBytes,
          ncclChar,
          peerRank,
          ncclComm_,
          stream_));
      NCCL_CHECK(ncclRecv(
          recvBuff.get(),
          config.nBytes,
          ncclChar,
          peerRank,
          ncclComm_,
          stream_));
      NCCL_CHECK(ncclGroupEnd());
    }
    CUDA_CHECK(cudaEventRecord(stop.get(), stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    float totalTime_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime_ms, start.get(), stop.get()));
    float avgTime_ms = totalTime_ms / kBenchmarkIters;
    timeUs = avgTime_ms * 1000.0f;
    // Bidirectional bandwidth: 2x data (send + recv) / time
    float bandwidth_GBps =
        (2.0f * config.nBytes / 1e9f) / (avgTime_ms / 1000.0f);

    bootstrap->barrierAll();

    return bandwidth_GBps;
  }

  // Helper function to run P2P NVL bidirectional benchmark - returns algorithm
  // BW
  float runP2pNvlBidirectionalBenchmark(
      comms::pipes::P2pNvlTransportDevice& p2p,
      const BenchmarkConfig& config,
      float& timeUs) {
    XLOGF(
        DBG1,
        "Rank {}: Starting P2P NVL bidirectional benchmark: {}",
        globalRank,
        config.name);

    DeviceBuffer sendBuff(config.nBytes);
    DeviceBuffer recvBuff(config.nBytes);

    // Initialize buffers
    CUDA_CHECK(cudaMemset(sendBuff.get(), globalRank, config.nBytes));
    CUDA_CHECK(cudaMemset(recvBuff.get(), 0, config.nBytes));

    dim3 gridDim(config.numBlocks);
    dim3 blockDim(config.numThreads);

    CudaEvent start, stop;

    std::size_t nBytes = config.nBytes;
    void* sendPtr = sendBuff.get();
    void* recvPtr = recvBuff.get();
    SyncScope groupScope = config.groupScope;
    Timeout timeout; // Default timeout (disabled)
    void* args[] = {&p2p, &sendPtr, &recvPtr, &nBytes, &groupScope, &timeout};
    void* kernelFunc = (void*)comms::pipes::benchmark::p2pBidirectional;

    // Warmup - no reset needed, recv() signals -1 after each transfer
    bootstrap->barrierAll();

    // Use pointer to cluster dimension for clustered launch
    dim3 defaultClusterDim(comms::common::kDefaultClusterSize, 1, 1);
    std::optional<dim3> clusterDimOpt = config.spreadClusterLaunch
        ? std::optional{defaultClusterDim}
        : std::nullopt;

    for (int i = 0; i < kWarmupIters; i++) {
      CUDA_CHECK(
          comms::common::launchKernel(
              kernelFunc, gridDim, blockDim, args, nullptr, clusterDimOpt));
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    bootstrap->barrierAll();

    // Benchmark - measure time across all iterations
    // No barrier between iterations - ChunkState provides synchronization
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
    timeUs = avgTime_ms * 1000.0f;
    // Bidirectional bandwidth: 2x data (send + recv) / time
    float bandwidth_GBps =
        (2.0f * config.nBytes / 1e9f) / (avgTime_ms / 1000.0f);

    bootstrap->barrierAll();

    return bandwidth_GBps;
  }

  void printResultsTable(
      const std::vector<BenchmarkResult>& results,
      const std::string& title) {
    if (globalRank != 0) {
      return; // Only rank 0 prints the table
    }

    std::stringstream ss;
    ss << "\n";
    ss << "==============================================================================================================================\n";
    ss << "                              " << title << "\n";
    ss << "==============================================================================================================================\n";
    ss << std::left << std::setw(18) << "Test Name" << std::right
       << std::setw(10) << "Msg Size" << std::right << std::setw(12)
       << "Staging" << std::right << std::setw(5) << "PD" << std::right
       << std::setw(8) << "Chunk" << std::right << std::setw(7) << "Blocks"
       << std::right << std::setw(8) << "Threads" << std::right << std::setw(11)
       << "NCCL BW" << std::right << std::setw(11) << "P2P BW" << std::right
       << std::setw(9) << "Speedup" << std::right << std::setw(11) << "NCCL Lat"
       << std::right << std::setw(11) << "P2P Lat" << std::right
       << std::setw(11) << "Lat Reduc\n";
    ss << std::left << std::setw(18) << "" << std::right << std::setw(10) << ""
       << std::right << std::setw(12) << "" << std::right << std::setw(5) << ""
       << std::right << std::setw(8) << "" << std::right << std::setw(7) << ""
       << std::right << std::setw(8) << "" << std::right << std::setw(11)
       << "(GB/s)" << std::right << std::setw(11) << "(GB/s)" << std::right
       << std::setw(9) << "P2P/NCCL" << std::right << std::setw(11) << "(us)"
       << std::right << std::setw(11) << "(us)" << std::right << std::setw(11)
       << "(us)\n";
    ss << "------------------------------------------------------------------------------------------------------------------------------\n";

    for (const auto& r : results) {
      std::string msgSize = formatSize(r.messageSize);
      std::string stagingSize = formatSize(r.stagingBufferSize);
      std::string chunkSizeStr = formatSize(r.chunkSize);
      float latencyReduction = r.ncclTime - r.p2pTime;

      ss << std::left << std::setw(18) << r.testName << std::right
         << std::setw(10) << msgSize << std::right << std::setw(12)
         << stagingSize << std::right << std::setw(5) << r.pipelineDepth
         << std::right << std::setw(8) << chunkSizeStr << std::right
         << std::setw(7) << r.numBlocks << std::right << std::setw(8)
         << r.numThreads << std::right << std::setw(11) << std::fixed
         << std::setprecision(2) << r.ncclBandwidth << std::right
         << std::setw(11) << std::fixed << std::setprecision(2)
         << r.p2pBandwidth << std::right << std::setw(8) << std::fixed
         << std::setprecision(2) << r.p2pSpeedup << "x" << std::right
         << std::setw(11) << std::fixed << std::setprecision(1) << r.ncclTime
         << std::right << std::setw(11) << std::fixed << std::setprecision(1)
         << r.p2pTime << std::right << std::setw(11) << std::fixed
         << std::setprecision(1) << latencyReduction << "\n";
    }
    ss << "==============================================================================================================================\n";
    ss << "PD = Pipeline Depth, Chunk = Chunk Size, Blocks/Threads = P2P kernel launch config\n";
    ss << "BW (Bandwidth) = Data transferred / time, in GB/s\n";
    ss << "Lat (Latency) = Average transfer time per iteration, in microseconds\n";
    ss << "Lat Reduc = NCCL latency - P2P latency (positive = P2P faster)\n";
    ss << "Speedup = P2P Bandwidth / NCCL Bandwidth\n";
    ss << "==============================================================================================================================\n\n";

    std::cout << ss.str();
  }

  ncclComm_t ncclComm_{};
  cudaStream_t stream_{};
};

TEST_F(P2pSendRecvBenchmarkFixture, UnidirectionalBenchmark) {
  // Only test with 2 ranks
  if (worldSize != 2) {
    XLOGF(DBG1, "Skipping test: requires exactly 2 ranks, got {}", worldSize);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;

  // Test configurations
  // IMPORTANT: Balance threads with staging buffer size
  // - chunks = stagingBufferSize / chunkSize
  // - warps = numBlocks * numThreads / 32
  // - Aim for chunks >= warps for good utilization
  // - Small messages need smaller chunks to enable parallelism
  std::vector<BenchmarkConfig> configs;

  // === SMALL MESSAGES (8KB - 2MB) ===
  // Key insight: Use smaller chunks to enable warp parallelism

  configs.push_back({
      .nBytes = 8 * 1024,
      .stagedBufferSize = 8 * 1024,
      .numBlocks = 1,
      .numThreads = 128,
      .pipelineDepth = 2,
      .chunkSize = 8 * 1024,
      .name = "8KB",
  });

  // 32KB: 4 warps, 4 chunks (8KB each) -> 1 chunk/warp
  configs.push_back({
      .nBytes = 32 * 1024,
      .stagedBufferSize = 32 * 1024,
      .numBlocks = 1,
      .numThreads = 128,
      .pipelineDepth = 2,
      .chunkSize = 8 * 1024,
      .name = "32KB",
  });

  // 64KB: 8 warps, 8 chunks -> 1 chunk/warp
  configs.push_back({
      .nBytes = 64 * 1024,
      .stagedBufferSize = 64 * 1024,
      .numBlocks = 2,
      .numThreads = 128,
      .pipelineDepth = 2,
      .chunkSize = 8 * 1024,
      .name = "64KB",
  });

  // 128KB: 16 warps, 16 chunks -> 1 chunk/warp
  configs.push_back({
      .nBytes = 128 * 1024,
      .stagedBufferSize = 128 * 1024,
      .numBlocks = 4,
      .numThreads = 128,
      .pipelineDepth = 2,
      .chunkSize = 8 * 1024,
      .name = "128KB",
  });

  // 256KB: 32 warps, 32 chunks -> 1 chunk/warp
  configs.push_back({
      .nBytes = 256 * 1024,
      .stagedBufferSize = 256 * 1024,
      .numBlocks = 8,
      .numThreads = 128,
      .pipelineDepth = 2,
      .chunkSize = 8 * 1024,
      .name = "256KB",
  });

  // 512KB: 64 warps, 32 chunks (16KB each)
  configs.push_back({
      .nBytes = 512 * 1024,
      .stagedBufferSize = 512 * 1024,
      .numBlocks = 16,
      .numThreads = 128,
      .pipelineDepth = 2,
      .chunkSize = 16 * 1024,
      .name = "512KB",
  });

  // 1MB: 128 warps, 32 chunks (32KB each)
  configs.push_back({
      .nBytes = 1024 * 1024,
      .stagedBufferSize = 1024 * 1024,
      .numBlocks = 32,
      .numThreads = 128,
      .pipelineDepth = 2,
      .chunkSize = 32 * 1024,
      .name = "1MB",
  });

  // 2MB: 256 warps, 128 chunks (16KB each) -> 1.24x speedup!
  configs.push_back({
      .nBytes = 2 * 1024 * 1024,
      .stagedBufferSize = 2 * 1024 * 1024,
      .numBlocks = 64,
      .numThreads = 128,
      .pipelineDepth = 2,
      .chunkSize = 16 * 1024,
      .name = "2MB",
  });

  // === MEDIUM/LARGE MESSAGES (64MB - 1GB) ===
  // Key insight: Larger chunks work well, bandwidth is the limit

  // 64MB: 512 warps, 256 chunks (128KB each)
  configs.push_back({
      .nBytes = 64 * 1024 * 1024,
      .stagedBufferSize = 32 * 1024 * 1024,
      .numBlocks = 128,
      .numThreads = 128,
      .pipelineDepth = 4,
      .chunkSize = 128 * 1024,
      .name = "64MB",
  });

  // 256MB: 1024 warps, 128 chunks (512KB each)
  configs.push_back({
      .nBytes = 256 * 1024 * 1024,
      .stagedBufferSize = 64 * 1024 * 1024,
      .numBlocks = 256,
      .numThreads = 128,
      .pipelineDepth = 4,
      .chunkSize = 512 * 1024,
      .name = "256MB",
  });

  // 1GB with 256MB staging
  configs.push_back({
      .nBytes = 1024 * 1024 * 1024,
      .stagedBufferSize = 256 * 1024 * 1024,
      .numBlocks = 256,
      .numThreads = 128,
      .pipelineDepth = 4,
      .chunkSize = 512 * 1024,
      .name = "1GB",
  });

  // === BLOCK-BASED GROUPS (fewer coordination points) ===
  // With block groups: 256 blocks = 256 groups (vs 1024 warps)

  // 64MB with block groups
  configs.push_back({
      .nBytes = 64 * 1024 * 1024,
      .stagedBufferSize = 32 * 1024 * 1024,
      .numBlocks = 128,
      .numThreads = 128,
      .pipelineDepth = 4,
      .chunkSize = 128 * 1024,
      .groupScope = SyncScope::BLOCK,
      .name = "64MB_Block",
  });

  // 256MB with block groups
  configs.push_back({
      .nBytes = 256 * 1024 * 1024,
      .stagedBufferSize = 64 * 1024 * 1024,
      .numBlocks = 256,
      .numThreads = 128,
      .pipelineDepth = 4,
      .chunkSize = 512 * 1024,
      .groupScope = SyncScope::BLOCK,
      .name = "256MB_Block",
  });

  // 1GB with block groups
  configs.push_back({
      .nBytes = 1024 * 1024 * 1024,
      .stagedBufferSize = 256 * 1024 * 1024,
      .numBlocks = 256,
      .numThreads = 128,
      .pipelineDepth = 4,
      .chunkSize = 512 * 1024,
      .groupScope = SyncScope::BLOCK,
      .name = "1GB_Block",
  });

  std::vector<BenchmarkResult> results;

  for (const auto& config : configs) {
    // Create P2P transport for this configuration
    comms::pipes::MultiPeerNvlTransportConfig p2pConfig{
        .dataBufferSize = config.stagedBufferSize,
        .chunkSize = config.chunkSize,
        .pipelineDepth = config.pipelineDepth,
    };

    comms::pipes::MultiPeerNvlTransport transport(
        globalRank, worldSize, bootstrap, p2pConfig);
    transport.exchange();

    auto p2p = transport.getP2pTransportDevice(peerRank);

    BenchmarkResult result;
    result.testName = config.name;
    result.messageSize = config.nBytes;
    result.stagingBufferSize = config.stagedBufferSize;
    result.pipelineDepth = config.pipelineDepth;
    result.chunkSize = config.chunkSize;
    result.numBlocks = config.numBlocks;
    result.numThreads = config.numThreads;

    // Run NCCL benchmark
    result.ncclBandwidth = runNcclBenchmark(config, result.ncclTime);

    // Run P2P NVL benchmark
    result.p2pBandwidth = runP2pNvlBenchmark(p2p, config, result.p2pTime);

    // Calculate speedup
    result.p2pSpeedup = (result.ncclBandwidth > 0)
        ? result.p2pBandwidth / result.ncclBandwidth
        : 0;

    results.push_back(result);
  }

  printResultsTable(
      results, "NCCL vs P2P NVLink UNIDIRECTIONAL Benchmark Results");
}

TEST_F(P2pSendRecvBenchmarkFixture, BidirectionalBenchmark) {
  // Only test with 2 ranks
  if (worldSize != 2) {
    XLOGF(DBG1, "Skipping test: requires exactly 2 ranks, got {}", worldSize);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;

  // Bidirectional test configurations
  // Key insight: Same rules as unidirectional - small chunks for small messages
  std::vector<BenchmarkConfig> configs;

  // 2MB: 256 warps, 128 chunks (16KB each) - matches optimal unidirectional
  configs.push_back({
      .nBytes = 2 * 1024 * 1024,
      .stagedBufferSize = 2 * 1024 * 1024,
      .numBlocks = 64,
      .numThreads = 128,
      .pipelineDepth = 2,
      .chunkSize = 16 * 1024,
      .name = "Bidir_2MB",
  });

  // 64MB: 512 warps, 256 chunks (128KB each)
  configs.push_back({
      .nBytes = 64 * 1024 * 1024,
      .stagedBufferSize = 8 * 1024 * 1024,
      .numBlocks = 32,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 512 * 1024,
      .groupScope = SyncScope::BLOCK,
      .name = "Bidir_64MB",
  });

  configs.push_back({
      .nBytes = 128 * 1024 * 1024,
      .stagedBufferSize = 8 * 1024 * 1024,
      .numBlocks = 32,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 512 * 1024,
      .groupScope = SyncScope::BLOCK,
      .name = "Bidir_128MB",
  });

  // 256MB: 512 warps, 256 chunks (256KB each)
  configs.push_back({
      .nBytes = 256 * 1024 * 1024,
      .stagedBufferSize = 8 * 1024 * 1024,
      .numBlocks = 32,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 512 * 1024,
      .groupScope = SyncScope::BLOCK,
      .name = "Bidir_256MB",
  });

  configs.push_back({
      .nBytes = 512 * 1024 * 1024,
      .stagedBufferSize = 8 * 1024 * 1024,
      .numBlocks = 32,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 512 * 1024,
      .groupScope = SyncScope::BLOCK,
      .name = "Bidir_512MB",
  });

  // 1GB with 256MB staging
  configs.push_back({
      .nBytes = 1024 * 1024 * 1024,
      .stagedBufferSize = 8 * 1024 * 1024,
      .numBlocks = 32,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 512 * 1024,
      .groupScope = SyncScope::BLOCK,
      .name = "Bidir_1GB",
  });

  // === NCCL-LIKE CONFIGURATIONS ===
  // Helper function to add NCCL-like configs with consistent parameters
  // NCCL uses 16 blocks for < 512M messages, 32 blocks for >= 512M messages
  constexpr int kNcclBlocksSmall = 16; // For messages < 512MB
  constexpr int kNcclBlocksLarge = 32; // For messages >= 512MB
  constexpr int kNcclThreads = 512;
  constexpr std::size_t kNcclStagedBufferSize = 8 * 1024 * 1024; // 8MB
  constexpr std::size_t kChunkSize = 512 * 1024; // 512KB
  constexpr std::size_t kLargeMessageThreshold = 512 * 1024 * 1024; // 512MB

  // Helper function for adding NCCL-like config with auto-computed numBlocks
  // Uses 16 blocks for < 512M, 32 blocks for >= 512M (like NCCL)
  auto addNcclConfig = [&configs,
                        kNcclBlocksLarge,
                        kNcclStagedBufferSize,
                        kLargeMessageThreshold](
                           std::size_t sizeBytes,
                           const std::string& sizeName,
                           SyncScope scope,
                           const std::string& scopeName) {
    int numBlks = (sizeBytes >= kLargeMessageThreshold) ? kNcclBlocksLarge
                                                        : kNcclBlocksSmall;
    configs.push_back({
        .nBytes = sizeBytes,
        .stagedBufferSize = kNcclStagedBufferSize,
        .numBlocks = numBlks,
        .numThreads = kNcclThreads,
        .pipelineDepth = 2,
        .chunkSize = kChunkSize,
        .groupScope = scope,
        .spreadClusterLaunch = true,
        .name = "NCCL_" + sizeName + "_" + scopeName,
    });
  };

  // === BLOCK-BASED CONFIGURATIONS ===
  addNcclConfig(128 * 1024, "128K", SyncScope::BLOCK, "Block");
  addNcclConfig(256 * 1024, "256K", SyncScope::BLOCK, "Block");
  addNcclConfig(1 * 1024 * 1024, "1M", SyncScope::BLOCK, "Block");
  addNcclConfig(2 * 1024 * 1024, "2M", SyncScope::BLOCK, "Block");
  addNcclConfig(8 * 1024 * 1024, "8M", SyncScope::BLOCK, "Block");
  addNcclConfig(32 * 1024 * 1024, "32M", SyncScope::BLOCK, "Block");
  addNcclConfig(64 * 1024 * 1024, "64M", SyncScope::BLOCK, "Block");
  addNcclConfig(128 * 1024 * 1024, "128M", SyncScope::BLOCK, "Block");
  addNcclConfig(256 * 1024 * 1024, "256M", SyncScope::BLOCK, "Block");
  addNcclConfig(512 * 1024 * 1024, "512M", SyncScope::BLOCK, "Block");
  addNcclConfig(1024 * 1024 * 1024, "1G", SyncScope::BLOCK, "Block");

  // === CLUSTER-BASED CONFIGURATIONS ===
  addNcclConfig(128 * 1024, "128K", SyncScope::CLUSTER, "Cluster");
  addNcclConfig(256 * 1024, "256K", SyncScope::CLUSTER, "Cluster");
  addNcclConfig(1 * 1024 * 1024, "1M", SyncScope::CLUSTER, "Cluster");
  addNcclConfig(2 * 1024 * 1024, "2M", SyncScope::CLUSTER, "Cluster");
  addNcclConfig(8 * 1024 * 1024, "8M", SyncScope::CLUSTER, "Cluster");
  addNcclConfig(32 * 1024 * 1024, "32M", SyncScope::CLUSTER, "Cluster");
  addNcclConfig(64 * 1024 * 1024, "64M", SyncScope::CLUSTER, "Cluster");
  addNcclConfig(128 * 1024 * 1024, "128M", SyncScope::CLUSTER, "Cluster");
  addNcclConfig(256 * 1024 * 1024, "256M", SyncScope::CLUSTER, "Cluster");
  addNcclConfig(512 * 1024 * 1024, "512M", SyncScope::CLUSTER, "Cluster");
  addNcclConfig(1024 * 1024 * 1024, "1G", SyncScope::CLUSTER, "Cluster");

  std::vector<BenchmarkResult> results;

  for (const auto& config : configs) {
    // Create P2P transport for this configuration
    comms::pipes::MultiPeerNvlTransportConfig p2pConfig{
        .dataBufferSize = config.stagedBufferSize,
        .chunkSize = config.chunkSize,
        .pipelineDepth = config.pipelineDepth,
    };

    comms::pipes::MultiPeerNvlTransport transport(
        globalRank, worldSize, bootstrap, p2pConfig);
    transport.exchange();

    auto p2p = transport.getP2pTransportDevice(peerRank);

    BenchmarkResult result;
    result.testName = config.name;
    result.messageSize = config.nBytes;
    result.stagingBufferSize = config.stagedBufferSize;
    result.pipelineDepth = config.pipelineDepth;
    result.chunkSize = config.chunkSize;
    result.numBlocks = config.numBlocks;
    result.numThreads = config.numThreads;

    // Run NCCL bidirectional benchmark
    result.ncclBandwidth =
        runNcclBidirectionalBenchmark(config, result.ncclTime);

    // Run P2P NVL bidirectional benchmark
    result.p2pBandwidth =
        runP2pNvlBidirectionalBenchmark(p2p, config, result.p2pTime);

    // Calculate speedup
    result.p2pSpeedup = (result.ncclBandwidth > 0)
        ? result.p2pBandwidth / result.ncclBandwidth
        : 0;

    results.push_back(result);
  }

  // Print results with modified header for bidirectional
  if (globalRank == 0) {
    std::stringstream ss;
    ss << "\n";
    ss << "==============================================================================================================================\n";
    ss << "                         NCCL vs P2P NVLink BIDIRECTIONAL Benchmark Results\n";
    ss << "==============================================================================================================================\n";
    ss << std::left << std::setw(18) << "Test Name" << std::right
       << std::setw(10) << "Msg Size" << std::right << std::setw(12)
       << "Staging" << std::right << std::setw(5) << "PD" << std::right
       << std::setw(8) << "Chunk" << std::right << std::setw(7) << "Blocks"
       << std::right << std::setw(8) << "Threads" << std::right << std::setw(11)
       << "NCCL BW" << std::right << std::setw(11) << "P2P BW" << std::right
       << std::setw(9) << "Speedup" << std::right << std::setw(11) << "NCCL Lat"
       << std::right << std::setw(11) << "P2P Lat" << std::right
       << std::setw(11) << "Lat Reduc\n";
    ss << std::left << std::setw(18) << "" << std::right << std::setw(10) << ""
       << std::right << std::setw(12) << "" << std::right << std::setw(5) << ""
       << std::right << std::setw(8) << "" << std::right << std::setw(7) << ""
       << std::right << std::setw(8) << "" << std::right << std::setw(11)
       << "(GB/s)" << std::right << std::setw(11) << "(GB/s)" << std::right
       << std::setw(9) << "P2P/NCCL" << std::right << std::setw(11) << "(us)"
       << std::right << std::setw(11) << "(us)" << std::right << std::setw(11)
       << "(us)\n";
    ss << "------------------------------------------------------------------------------------------------------------------------------\n";

    for (const auto& r : results) {
      std::string msgSize = formatSize(r.messageSize);
      std::string stagingSize = formatSize(r.stagingBufferSize);
      std::string chunkSizeStr = formatSize(r.chunkSize);
      float latencyReduction = r.ncclTime - r.p2pTime;

      ss << std::left << std::setw(18) << r.testName << std::right
         << std::setw(10) << msgSize << std::right << std::setw(12)
         << stagingSize << std::right << std::setw(5) << r.pipelineDepth
         << std::right << std::setw(8) << chunkSizeStr << std::right
         << std::setw(7) << r.numBlocks << std::right << std::setw(8)
         << r.numThreads << std::right << std::setw(11) << std::fixed
         << std::setprecision(2) << r.ncclBandwidth << std::right
         << std::setw(11) << std::fixed << std::setprecision(2)
         << r.p2pBandwidth << std::right << std::setw(8) << std::fixed
         << std::setprecision(2) << r.p2pSpeedup << "x" << std::right
         << std::setw(11) << std::fixed << std::setprecision(1) << r.ncclTime
         << std::right << std::setw(11) << std::fixed << std::setprecision(1)
         << r.p2pTime << std::right << std::setw(11) << std::fixed
         << std::setprecision(1) << latencyReduction << "\n";
    }
    ss << "==============================================================================================================================\n";
    ss << "Bidirectional: Both ranks send AND receive simultaneously\n";
    ss << "BW = Algorithm bandwidth (2 x message size / time)\n";
    ss << "==============================================================================================================================\n\n";

    std::cout << ss.str();
  }
}

} // namespace comms::pipes::benchmark

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);

  // Set up distributed environment
  if (!meta::comms::isTcpEnvironment()) {
    ::testing::AddGlobalTestEnvironment(
        new meta::comms::BenchmarkEnvironment());
  }

  return RUN_ALL_TESTS();
}
