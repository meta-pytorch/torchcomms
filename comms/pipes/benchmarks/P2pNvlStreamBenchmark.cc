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
namespace {

class P2pStreamBenchmarkFixture : public meta::comms::BenchmarkTestFixture {
 protected:
  void SetUp() override {
    BenchmarkTestFixture::SetUp();
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
    id = allIds[0];
    return id;
  }

  float runNcclBenchmark(const BenchmarkConfig& config, float& timeUs) {
    XLOGF(DBG1, "=== Running NCCL benchmark: {} ===", config.name);

    DeviceBuffer sendBuff(config.nBytes);
    DeviceBuffer recvBuff(config.nBytes);

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

    // Benchmark
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
    float bandwidth_GBps = (config.nBytes / 1e9f) / (avgTime_ms / 1000.0f);

    bootstrap->barrierAll();

    return bandwidth_GBps;
  }

  float runStreamBenchmark(
      comms::pipes::P2pNvlTransportDevice& p2p,
      const BenchmarkConfig& config,
      float& timeUs) {
    XLOGF(DBG1, "=== Running Stream P2P benchmark: {} ===", config.name);

    DeviceBuffer sendBuff(config.nBytes);
    DeviceBuffer recvBuff(config.nBytes);

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
    Timeout timeout;
    void* args[] = {&p2p, &devicePtr, &nBytes, &groupScope, &timeout};
    void* kernelFunc = isSend ? (void*)comms::pipes::benchmark::p2pStreamSend
                              : (void*)comms::pipes::benchmark::p2pStreamRecv;
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

    // Benchmark
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
    float bandwidth_GBps = (config.nBytes / 1e9f) / (avgTime_ms / 1000.0f);

    CUDA_CHECK(cudaStreamDestroy(sendStream));
    CUDA_CHECK(cudaStreamDestroy(recvStream));

    bootstrap->barrierAll();

    return bandwidth_GBps;
  }

  void printResultsTable(
      const std::vector<BenchmarkResult>& results,
      const std::string& title) {
    if (globalRank != 0) {
      return;
    }

    std::stringstream ss;
    ss << "\n";
    ss << "====================================================================================================================================\n";
    ss << "                              " << title << "\n";
    ss << "====================================================================================================================================\n";
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
    ss << "------------------------------------------------------------------------------------------------------------------------------------\n";

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
    ss << "====================================================================================================================================\n";
    ss << "PD = Pipeline Depth, Chunk = Chunk Size, Blocks/Threads = P2P kernel launch config\n";
    ss << "BW (Bandwidth) = Data transferred / time, in GB/s\n";
    ss << "Lat (Latency) = Average transfer time per iteration, in microseconds\n";
    ss << "Lat Reduc = NCCL latency - P2P latency (positive = P2P faster)\n";
    ss << "Speedup = P2P Bandwidth / NCCL Bandwidth\n";
    ss << "====================================================================================================================================\n\n";

    std::cout << ss.str();
  }

  ncclComm_t ncclComm_{};
  cudaStream_t stream_{};
};

} // namespace

TEST_F(P2pStreamBenchmarkFixture, UnidirectionalStreamBenchmark) {
  if (worldSize != 2) {
    XLOGF(DBG1, "Skipping test: requires exactly 2 ranks, got {}", worldSize);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;

  std::vector<BenchmarkConfig> configs;

  // 8KB: 1 block, 128 threads, PD=2, chunk=8KB
  configs.push_back({
      .nBytes = 8 * 1024,
      .stagedBufferSize = 8 * 1024,
      .numBlocks = 1,
      .numThreads = 128,
      .pipelineDepth = 2,
      .chunkSize = 8 * 1024,
      .name = "Stream_8KB",
  });

  // 64KB: 2 blocks, 128 threads, PD=2, chunk=8KB
  configs.push_back({
      .nBytes = 64 * 1024,
      .stagedBufferSize = 64 * 1024,
      .numBlocks = 2,
      .numThreads = 128,
      .pipelineDepth = 2,
      .chunkSize = 8 * 1024,
      .name = "Stream_64KB",
  });

  // 1MB: 32 blocks, 128 threads, PD=2, chunk=32KB
  configs.push_back({
      .nBytes = 1024 * 1024,
      .stagedBufferSize = 1024 * 1024,
      .numBlocks = 32,
      .numThreads = 128,
      .pipelineDepth = 2,
      .chunkSize = 32 * 1024,
      .name = "Stream_1MB",
  });

  // 64MB: 128 blocks, 128 threads, PD=4, chunk=128KB
  configs.push_back({
      .nBytes = 64 * 1024 * 1024,
      .stagedBufferSize = 32 * 1024 * 1024,
      .numBlocks = 128,
      .numThreads = 128,
      .pipelineDepth = 4,
      .chunkSize = 128 * 1024,
      .name = "Stream_64MB",
  });

  // 256MB: 256 blocks, 128 threads, PD=4, chunk=512KB
  configs.push_back({
      .nBytes = 256 * 1024 * 1024,
      .stagedBufferSize = 64 * 1024 * 1024,
      .numBlocks = 256,
      .numThreads = 128,
      .pipelineDepth = 4,
      .chunkSize = 512 * 1024,
      .name = "Stream_256MB",
  });

  // 1GB: 256 blocks, 128 threads, PD=4, chunk=512KB
  configs.push_back({
      .nBytes = 1024 * 1024 * 1024,
      .stagedBufferSize = 256 * 1024 * 1024,
      .numBlocks = 256,
      .numThreads = 128,
      .pipelineDepth = 4,
      .chunkSize = 512 * 1024,
      .name = "Stream_1GB",
  });

  std::vector<BenchmarkResult> results;

  for (const auto& config : configs) {
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

    result.ncclBandwidth = runNcclBenchmark(config, result.ncclTime);
    result.p2pBandwidth = runStreamBenchmark(p2p, config, result.p2pTime);

    result.p2pSpeedup = (result.ncclBandwidth > 0)
        ? result.p2pBandwidth / result.ncclBandwidth
        : 0;

    results.push_back(result);
  }

  printResultsTable(
      results, "NCCL vs P2P NVLink STREAMING UNIDIRECTIONAL Benchmark Results");
}

} // namespace comms::pipes::benchmark

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);

  if (!meta::comms::isTcpEnvironment()) {
    ::testing::AddGlobalTestEnvironment(
        new meta::comms::BenchmarkEnvironment());
  }

  return RUN_ALL_TESTS();
}
