// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>
#include <nccl.h>

#include "comms/pipes/MultiPeerNvlTransport.h"
#include "comms/pipes/benchmarks/BenchmarkKernel.cuh"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

#include <iomanip>
#include <sstream>
#include <vector>

using meta::comms::CudaEvent;
using meta::comms::DeviceBuffer;
using meta::comms::MpiBaseTestFixture;
using meta::comms::MPIEnvironmentBase;

namespace comms::pipes::benchmark {

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

// NCCL error checking macro for void functions
#define NCCL_CHECK_VOID(call)        \
  do {                               \
    ncclResult_t res = call;         \
    if (res != ncclSuccess) {        \
      XLOGF(                         \
          ERR,                       \
          "NCCL error at {}:{}: {}", \
          __FILE__,                  \
          __LINE__,                  \
          ncclGetErrorString(res));  \
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

// NCCL error checking macro for float-returning functions
#define NCCL_CHECK(call)             \
  do {                               \
    ncclResult_t res = call;         \
    if (res != ncclSuccess) {        \
      XLOGF(                         \
          ERR,                       \
          "NCCL error at {}:{}: {}", \
          __FILE__,                  \
          __LINE__,                  \
          ncclGetErrorString(res));  \
      return 0.0f;                   \
    }                                \
  } while (0)

// Test configuration struct
struct BenchmarkConfig {
  std::size_t nBytes;
  std::size_t stagedBufferSize;
  int numBlocks;
  int numThreads;
  std::size_t pipelineDepth = 4;
  std::size_t chunkSize = 512 * 1024; // 512KB default
  bool useBlockGroups = false; // Use block-level groups instead of warp-level
  std::string name;
};

// Result struct for collecting benchmark data
struct BenchmarkResult {
  std::string testName;
  std::size_t messageSize{};
  std::size_t stagingBufferSize{};
  std::size_t pipelineDepth{};
  std::size_t chunkSize{};
  float ncclBandwidth{};
  float p2pBandwidth{};
  float ncclTime{};
  float p2pTime{};
  float p2pSpeedup{}; // P2P vs NCCL
};

class P2pNvlBenchmarkFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    cudaSetDevice(globalRank);

    // Set NCCL channels to 32 for maximum bandwidth
    setenv("NCCL_MIN_NCHANNELS", "32", 1);
    setenv("NCCL_MAX_NCHANNELS", "32", 1);

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

    const int nIter = 10;
    CudaEvent start, stop;

    // Warmup
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    for (int i = 0; i < 3; i++) {
      if (globalRank == 0) {
        NCCL_CHECK(ncclSend(
            sendBuff.get(), config.nBytes, ncclChar, 1, ncclComm_, stream_));
      } else if (globalRank == 1) {
        NCCL_CHECK(ncclRecv(
            recvBuff.get(), config.nBytes, ncclChar, 0, ncclComm_, stream_));
      }
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Benchmark - measure time across all iterations
    // No barrier between iterations - rely on NCCL's internal synchronization
    CUDA_CHECK(cudaEventRecord(start.get(), stream_));
    for (int i = 0; i < nIter; i++) {
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
    float avgTime_ms = totalTime_ms / nIter;
    timeUs = avgTime_ms * 1000.0f;
    // Unidirectional bandwidth: data transferred in one direction / time
    float bandwidth_GBps = (config.nBytes / (1024.0f * 1024.0f * 1024.0f)) /
        (avgTime_ms / 1000.0f);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

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
    bool useBlockGroups = config.useBlockGroups;
    void* devicePtr = (isSend ? sendBuff.get() : recvBuff.get());
    void* args[] = {&p2p, &devicePtr, &nBytes, &useBlockGroups};
    void* kernelFunc = isSend ? (void*)comms::pipes::benchmark::p2pSend
                              : (void*)comms::pipes::benchmark::p2pRecv;
    cudaStream_t stream = isSend ? sendStream : recvStream;

    const int nIter = 10;

    // Warmup
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    for (int i = 0; i < 3; i++) {
      MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
      CUDA_CHECK(
          cudaLaunchKernel(kernelFunc, gridDim, blockDim, args, 0, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Benchmark - measure time across all iterations
    // No barrier between iterations - ChunkState provides synchronization
    CUDA_CHECK(cudaEventRecord(start.get(), stream));
    for (int i = 0; i < nIter; i++) {
      CUDA_CHECK(
          cudaLaunchKernel(kernelFunc, gridDim, blockDim, args, 0, stream));
    }
    CUDA_CHECK(cudaEventRecord(stop.get(), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float totalTime_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime_ms, start.get(), stop.get()));
    float avgTime_ms = totalTime_ms / nIter;
    timeUs = avgTime_ms * 1000.0f;
    // Unidirectional bandwidth: data transferred in one direction / time
    float bandwidth_GBps = (config.nBytes / (1024.0f * 1024.0f * 1024.0f)) /
        (avgTime_ms / 1000.0f);

    CUDA_CHECK(cudaStreamDestroy(sendStream));
    CUDA_CHECK(cudaStreamDestroy(recvStream));

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    return bandwidth_GBps;
  }

  void printResultsTable(const std::vector<BenchmarkResult>& results) {
    if (globalRank != 0) {
      return; // Only rank 0 prints the table
    }

    std::stringstream ss;
    ss << "\n";
    ss << "====================================================================================================\n";
    ss << "                         NCCL vs P2P NVLink Benchmark Results\n";
    ss << "====================================================================================================\n";
    ss << std::left << std::setw(18) << "Test Name" << std::right
       << std::setw(10) << "Msg Size" << std::right << std::setw(12)
       << "Staging" << std::right << std::setw(5) << "PD" << std::right
       << std::setw(8) << "Chunk" << std::right << std::setw(11) << "NCCL BW"
       << std::right << std::setw(11) << "P2P BW" << std::right << std::setw(9)
       << "Speedup" << std::right << std::setw(11) << "NCCL Lat" << std::right
       << std::setw(11) << "P2P Lat" << std::right << std::setw(11)
       << "Lat Reduc\n";
    ss << std::left << std::setw(18) << "" << std::right << std::setw(10) << ""
       << std::right << std::setw(12) << "" << std::right << std::setw(5) << ""
       << std::right << std::setw(8) << "" << std::right << std::setw(11)
       << "(GB/s)" << std::right << std::setw(11) << "(GB/s)" << std::right
       << std::setw(9) << "P2P/NCCL" << std::right << std::setw(11) << "(us)"
       << std::right << std::setw(11) << "(us)" << std::right << std::setw(11)
       << "(us)\n";
    ss << "----------------------------------------------------------------------------------------------------\n";

    for (const auto& r : results) {
      std::string msgSize = formatSize(r.messageSize);
      std::string stagingSize = formatSize(r.stagingBufferSize);
      std::string chunkSizeStr = formatSize(r.chunkSize);
      float latencyReduction = r.ncclTime - r.p2pTime;

      ss << std::left << std::setw(18) << r.testName << std::right
         << std::setw(10) << msgSize << std::right << std::setw(12)
         << stagingSize << std::right << std::setw(5) << r.pipelineDepth
         << std::right << std::setw(8) << chunkSizeStr << std::right
         << std::setw(11) << std::fixed << std::setprecision(2)
         << r.ncclBandwidth << std::right << std::setw(11) << std::fixed
         << std::setprecision(2) << r.p2pBandwidth << std::right << std::setw(8)
         << std::fixed << std::setprecision(2) << r.p2pSpeedup << "x"
         << std::right << std::setw(11) << std::fixed << std::setprecision(1)
         << r.ncclTime << std::right << std::setw(11) << std::fixed
         << std::setprecision(1) << r.p2pTime << std::right << std::setw(11)
         << std::fixed << std::setprecision(1) << latencyReduction << "\n";
    }
    ss << "====================================================================================================\n";
    ss << "PD = Pipeline Depth, Chunk = Chunk Size\n";
    ss << "BW (Bandwidth) = Data transferred / time, in GB/s\n";
    ss << "Lat (Latency) = Average transfer time per iteration, in microseconds\n";
    ss << "Lat Reduc = NCCL latency - P2P latency (positive = P2P faster)\n";
    ss << "Speedup = P2P Bandwidth / NCCL Bandwidth\n";
    ss << "====================================================================================================\n\n";

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

  // Helper function to run NCCL bidirectional benchmark - returns algorithm BW
  float runNcclBidirectionalBenchmark(
      const BenchmarkConfig& config,
      float& timeUs) {
    XLOGF(
        INFO,
        "Rank {}: Starting NCCL bidirectional benchmark: {}",
        globalRank,
        config.name);

    DeviceBuffer sendBuff(config.nBytes);
    DeviceBuffer recvBuff(config.nBytes);

    // Initialize buffers - each rank sends its own data
    CUDA_CHECK(cudaMemset(sendBuff.get(), globalRank, config.nBytes));
    CUDA_CHECK(cudaMemset(recvBuff.get(), 0, config.nBytes));

    int peerRank = (globalRank == 0) ? 1 : 0;

    const int nIter = 10;
    CudaEvent start, stop;

    // Warmup
    XLOGF(INFO, "Rank {}: NCCL bidi warmup starting", globalRank);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    for (int i = 0; i < 3; i++) {
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
    XLOGF(INFO, "Rank {}: NCCL bidi warmup complete", globalRank);

    // Benchmark - measure time across all iterations
    // No barrier between iterations - rely on NCCL's internal synchronization
    CUDA_CHECK(cudaEventRecord(start.get(), stream_));
    for (int i = 0; i < nIter; i++) {
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
    float avgTime_ms = totalTime_ms / nIter;
    timeUs = avgTime_ms * 1000.0f;
    // Bidirectional bandwidth: 2x data (send + recv) / time
    float bandwidth_GBps =
        (2.0f * config.nBytes / (1024.0f * 1024.0f * 1024.0f)) /
        (avgTime_ms / 1000.0f);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    return bandwidth_GBps;
  }

  // Helper function to run P2P NVL bidirectional benchmark - returns algorithm
  // BW
  float runP2pNvlBidirectionalBenchmark(
      comms::pipes::P2pNvlTransportDevice& p2p,
      const BenchmarkConfig& config,
      float& timeUs) {
    XLOGF(
        INFO,
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
    bool useBlockGroups = config.useBlockGroups;
    void* args[] = {&p2p, &sendPtr, &recvPtr, &nBytes, &useBlockGroups};
    void* kernelFunc = (void*)comms::pipes::benchmark::p2pBidirectional;

    const int nIter = 10;

    // Warmup - no reset needed, recv() signals -1 after each transfer
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    for (int i = 0; i < 3; i++) {
      CUDA_CHECK(
          cudaLaunchKernel(kernelFunc, gridDim, blockDim, args, 0, nullptr));
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Benchmark - measure time across all iterations
    // No barrier between iterations - ChunkState provides synchronization
    CUDA_CHECK(cudaEventRecord(start.get()));
    for (int i = 0; i < nIter; i++) {
      CUDA_CHECK(
          cudaLaunchKernel(kernelFunc, gridDim, blockDim, args, 0, nullptr));
    }
    CUDA_CHECK(cudaEventRecord(stop.get()));
    CUDA_CHECK(cudaDeviceSynchronize());

    float totalTime_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime_ms, start.get(), stop.get()));
    float avgTime_ms = totalTime_ms / nIter;
    timeUs = avgTime_ms * 1000.0f;
    // Bidirectional bandwidth: 2x data (send + recv) / time
    float bandwidth_GBps =
        (2.0f * config.nBytes / (1024.0f * 1024.0f * 1024.0f)) /
        (avgTime_ms / 1000.0f);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    return bandwidth_GBps;
  }

  ncclComm_t ncclComm_{};
  cudaStream_t stream_{};
};

TEST_F(P2pNvlBenchmarkFixture, CompareNcclVsP2pNvl) {
  // Only test with 2 ranks
  if (numRanks != 2) {
    XLOGF(DBG1, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
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
      .useBlockGroups = true,
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
      .useBlockGroups = true,
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
      .useBlockGroups = true,
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

    auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
    comms::pipes::MultiPeerNvlTransport transport(
        globalRank, numRanks, bootstrap, p2pConfig);
    transport.exchange();

    auto p2p = transport.getP2pTransportDevice(peerRank);

    BenchmarkResult result;
    result.testName = config.name;
    result.messageSize = config.nBytes;
    result.stagingBufferSize = config.stagedBufferSize;
    result.pipelineDepth = config.pipelineDepth;
    result.chunkSize = config.chunkSize;

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

  printResultsTable(results);
}

TEST_F(P2pNvlBenchmarkFixture, BidirectionalBenchmark) {
  // Only test with 2 ranks
  if (numRanks != 2) {
    XLOGF(DBG1, "Skipping test: requires exactly 2 ranks, got {}", numRanks);
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
      .stagedBufferSize = 32 * 1024 * 1024,
      .numBlocks = 128,
      .numThreads = 128,
      .pipelineDepth = 2,
      .chunkSize = 128 * 1024,
      .name = "Bidir_64MB",
  });

  // 256MB: 512 warps, 256 chunks (256KB each)
  configs.push_back({
      .nBytes = 256 * 1024 * 1024,
      .stagedBufferSize = 64 * 1024 * 1024,
      .numBlocks = 128,
      .numThreads = 128,
      .pipelineDepth = 2,
      .chunkSize = 256 * 1024,
      .name = "Bidir_256MB",
  });

  // 1GB with 256MB staging
  configs.push_back({
      .nBytes = 1024 * 1024 * 1024,
      .stagedBufferSize = 256 * 1024 * 1024,
      .numBlocks = 256,
      .numThreads = 128,
      .pipelineDepth = 4,
      .chunkSize = 512 * 1024,
      .name = "Bidir_1GB",
  });

  // === BLOCK-BASED BIDIRECTIONAL ===
  // 1GB with block groups (256 groups vs 1024 warps)
  configs.push_back({
      .nBytes = 1024 * 1024 * 1024,
      .stagedBufferSize = 256 * 1024 * 1024,
      .numBlocks = 256,
      .numThreads = 128,
      .pipelineDepth = 4,
      .chunkSize = 512 * 1024,
      .useBlockGroups = true,
      .name = "Bidir_1GB_Block",
  });

  std::vector<BenchmarkResult> results;

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

    BenchmarkResult result;
    result.testName = config.name;
    result.messageSize = config.nBytes;
    result.stagingBufferSize = config.stagedBufferSize;
    result.pipelineDepth = config.pipelineDepth;
    result.chunkSize = config.chunkSize;

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
    ss << "====================================================================================================\n";
    ss << "                    NCCL vs P2P NVLink BIDIRECTIONAL Benchmark Results\n";
    ss << "====================================================================================================\n";
    ss << std::left << std::setw(18) << "Test Name" << std::right
       << std::setw(10) << "Msg Size" << std::right << std::setw(12)
       << "Staging" << std::right << std::setw(5) << "PD" << std::right
       << std::setw(8) << "Chunk" << std::right << std::setw(11) << "NCCL BW"
       << std::right << std::setw(11) << "P2P BW" << std::right << std::setw(9)
       << "Speedup" << std::right << std::setw(11) << "NCCL Lat" << std::right
       << std::setw(11) << "P2P Lat" << std::right << std::setw(11)
       << "Lat Reduc\n";
    ss << std::left << std::setw(18) << "" << std::right << std::setw(10) << ""
       << std::right << std::setw(12) << "" << std::right << std::setw(5) << ""
       << std::right << std::setw(8) << "" << std::right << std::setw(11)
       << "(GB/s)" << std::right << std::setw(11) << "(GB/s)" << std::right
       << std::setw(9) << "P2P/NCCL" << std::right << std::setw(11) << "(us)"
       << std::right << std::setw(11) << "(us)" << std::right << std::setw(11)
       << "(us)\n";
    ss << "----------------------------------------------------------------------------------------------------\n";

    for (const auto& r : results) {
      std::string msgSize = formatSize(r.messageSize);
      std::string stagingSize = formatSize(r.stagingBufferSize);
      std::string chunkSizeStr = formatSize(r.chunkSize);
      float latencyReduction = r.ncclTime - r.p2pTime;

      ss << std::left << std::setw(18) << r.testName << std::right
         << std::setw(10) << msgSize << std::right << std::setw(12)
         << stagingSize << std::right << std::setw(5) << r.pipelineDepth
         << std::right << std::setw(8) << chunkSizeStr << std::right
         << std::setw(11) << std::fixed << std::setprecision(2)
         << r.ncclBandwidth << std::right << std::setw(11) << std::fixed
         << std::setprecision(2) << r.p2pBandwidth << std::right << std::setw(8)
         << std::fixed << std::setprecision(2) << r.p2pSpeedup << "x"
         << std::right << std::setw(11) << std::fixed << std::setprecision(1)
         << r.ncclTime << std::right << std::setw(11) << std::fixed
         << std::setprecision(1) << r.p2pTime << std::right << std::setw(11)
         << std::fixed << std::setprecision(1) << latencyReduction << "\n";
    }
    ss << "====================================================================================================\n";
    ss << "Bidirectional: Both ranks send AND receive simultaneously\n";
    ss << "BW = Algorithm bandwidth (2 x message size / time)\n";
    ss << "====================================================================================================\n\n";

    XLOG(INFO) << ss.str();
  }
}

} // namespace comms::pipes::benchmark

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
