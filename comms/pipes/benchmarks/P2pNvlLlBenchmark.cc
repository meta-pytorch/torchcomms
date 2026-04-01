// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>
#include <nccl.h>

#include "comms/common/CudaWrap.h"
#include "comms/pipes/MultiPeerNvlTransport.h"
#include "comms/pipes/benchmarks/BenchmarkKernel.cuh"
#include "comms/pipes/benchmarks/BenchmarkMacros.h"
#include "comms/pipes/benchmarks/P2pNvlBenchmarkUtils.h"
#include "comms/pipes/ll/LlPacket.cuh"
#include "comms/pipes/ll128/Ll128AutoTune.cuh"
#include "comms/pipes/ll128/Ll128Packet.cuh"
#include "comms/testinfra/BenchmarkTestFixture.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

using meta::comms::CudaEvent;
using meta::comms::DeviceBuffer;

namespace comms::pipes::benchmark {

// Configuration for LL benchmark sweep points
struct LlConfig {
  std::size_t nBytes;
  int numBlocks;
  int numThreads;
  std::string name;
};

// Result struct for 3-way comparison (NCCL vs LL128 vs LL)
struct LlBenchmarkResult {
  std::string testName;
  std::size_t messageSize{};
  int numBlocks{};
  int numThreads{};
  float ncclBandwidth{};
  float ll128Bandwidth{};
  float llBandwidth{};
  float ncclTime{}; // microseconds
  float ll128Time{}; // microseconds
  float llTime{}; // microseconds
};

class P2pLlBenchmarkFixture : public meta::comms::BenchmarkTestFixture {
 protected:
  void SetUp() override {
    BenchmarkTestFixture::SetUp();
    CUDA_CHECK_VOID(cudaSetDevice(localRank));
    NCCL_CHECK_VOID(
        ncclCommInitRank(&ncclComm_, worldSize, getNCCLId(), globalRank));
    CUDA_CHECK_VOID(cudaStreamCreate(&stream_));
  }

  void TearDown() override {
    if (ncclComm_ != nullptr) {
      ncclResult_t res = ncclCommDestroy(ncclComm_);
      if (res != ncclSuccess) {
        XLOGF(ERR, "ncclCommDestroy failed: {}", ncclGetErrorString(res));
      }
    }
    if (stream_ != nullptr) {
      cudaError_t err = cudaStreamDestroy(stream_);
      if (err != cudaSuccess) {
        XLOGF(ERR, "cudaStreamDestroy failed: {}", cudaGetErrorString(err));
      }
    }
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

  float runNcclBenchmark(std::size_t nBytes, float& timeUs) {
    DeviceBuffer sendBuff(nBytes);
    DeviceBuffer recvBuff(nBytes);

    if (globalRank == 0) {
      CUDA_CHECK(cudaMemset(sendBuff.get(), 1, nBytes));
    }
    if (globalRank == 1) {
      CUDA_CHECK(cudaMemset(recvBuff.get(), 0, nBytes));
    }

    CudaEvent start, stop;

    // Warmup
    bootstrap->barrierAll();
    for (int i = 0; i < kWarmupIters; i++) {
      NCCL_CHECK(ncclGroupStart());
      if (globalRank == 0) {
        NCCL_CHECK(
            ncclSend(sendBuff.get(), nBytes, ncclChar, 1, ncclComm_, stream_));
      } else if (globalRank == 1) {
        NCCL_CHECK(
            ncclRecv(recvBuff.get(), nBytes, ncclChar, 0, ncclComm_, stream_));
      }
      NCCL_CHECK(ncclGroupEnd());
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    // Benchmark
    CUDA_CHECK(cudaEventRecord(start.get(), stream_));
    for (int i = 0; i < kBenchmarkIters; i++) {
      NCCL_CHECK(ncclGroupStart());
      if (globalRank == 0) {
        NCCL_CHECK(
            ncclSend(sendBuff.get(), nBytes, ncclChar, 1, ncclComm_, stream_));
      } else if (globalRank == 1) {
        NCCL_CHECK(
            ncclRecv(recvBuff.get(), nBytes, ncclChar, 0, ncclComm_, stream_));
      }
      NCCL_CHECK(ncclGroupEnd());
    }
    CUDA_CHECK(cudaEventRecord(stop.get(), stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    float totalTime_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime_ms, start.get(), stop.get()));
    float avgTime_ms = totalTime_ms / kBenchmarkIters;
    timeUs = avgTime_ms * 1000.0f;
    float bandwidth_GBps = (nBytes / 1e9f) / (avgTime_ms / 1000.0f);

    bootstrap->barrierAll();
    return bandwidth_GBps;
  }

  float runLl128Benchmark(
      comms::pipes::P2pNvlTransportDevice& p2p,
      const BenchmarkConfig& config,
      float& timeUs) {
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

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CudaEvent start, stop;

    std::size_t nBytes = config.nBytes;
    bool isSend = (globalRank == 0);
    void* devicePtr = isSend ? sendBuff.get() : recvBuff.get();
    Timeout timeout;
    void* args[] = {&p2p, &devicePtr, &nBytes, &timeout};
    void* kernelFunc = isSend ? (void*)comms::pipes::benchmark::p2pLl128Send
                              : (void*)comms::pipes::benchmark::p2pLl128Recv;

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

    CUDA_CHECK(cudaStreamDestroy(stream));

    bootstrap->barrierAll();
    return bandwidth_GBps;
  }

  float runLlBenchmark(
      comms::pipes::P2pNvlTransportDevice& p2p,
      const BenchmarkConfig& config,
      float& timeUs) {
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

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CudaEvent start, stop;

    std::size_t nBytes = config.nBytes;
    bool isSend = (globalRank == 0);
    void* devicePtr = isSend ? sendBuff.get() : recvBuff.get();
    Timeout timeout;
    void* args[] = {&p2p, &devicePtr, &nBytes, &timeout};
    void* kernelFunc = isSend ? (void*)comms::pipes::benchmark::p2pLlSend
                              : (void*)comms::pipes::benchmark::p2pLlRecv;

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

    CUDA_CHECK(cudaStreamDestroy(stream));

    bootstrap->barrierAll();
    return bandwidth_GBps;
  }

  void printResultsTable(
      const std::vector<LlBenchmarkResult>& results,
      const std::string& title) {
    if (globalRank != 0) {
      return;
    }

    std::stringstream ss;
    ss << "\n";
    ss << "======================================================================================================================================\n";
    ss << "                              " << title << "\n";
    ss << "======================================================================================================================================\n";
    ss << std::left << std::setw(16) << "Test Name" << std::right
       << std::setw(10) << "Msg Size" << std::right << std::setw(8) << "Blocks"
       << std::right << std::setw(9) << "Threads" << std::right << std::setw(11)
       << "NCCL BW" << std::right << std::setw(11) << "LL128 BW" << std::right
       << std::setw(11) << "LL BW" << std::right << std::setw(12) << "LL/NCCL"
       << std::right << std::setw(13) << "LL/LL128" << std::right
       << std::setw(11) << "NCCL Lat" << std::right << std::setw(11)
       << "LL128 Lat" << std::right << std::setw(11) << "LL Lat\n";
    ss << std::left << std::setw(16) << "" << std::right << std::setw(10) << ""
       << std::right << std::setw(8) << "" << std::right << std::setw(9) << ""
       << std::right << std::setw(11) << "(GB/s)" << std::right << std::setw(11)
       << "(GB/s)" << std::right << std::setw(11) << "(GB/s)" << std::right
       << std::setw(12) << "" << std::right << std::setw(13) << "" << std::right
       << std::setw(11) << "(us)" << std::right << std::setw(11) << "(us)"
       << std::right << std::setw(11) << "(us)\n";
    ss << "--------------------------------------------------------------------------------------------------------------------------------------\n";

    for (const auto& r : results) {
      std::string msgSize = formatSize(r.messageSize);
      float llVsNccl =
          (r.ncclBandwidth > 0) ? r.llBandwidth / r.ncclBandwidth : 0;
      float llVsLl128 =
          (r.ll128Bandwidth > 0) ? r.llBandwidth / r.ll128Bandwidth : 0;

      ss << std::left << std::setw(16) << r.testName << std::right
         << std::setw(10) << msgSize << std::right << std::setw(8)
         << r.numBlocks << std::right << std::setw(9) << r.numThreads
         << std::right << std::setw(11) << std::fixed << std::setprecision(2)
         << r.ncclBandwidth << std::right << std::setw(11) << std::fixed
         << std::setprecision(2) << r.ll128Bandwidth << std::right
         << std::setw(11) << std::fixed << std::setprecision(2) << r.llBandwidth
         << std::right << std::setw(11) << std::fixed << std::setprecision(2)
         << llVsNccl << "x" << std::right << std::setw(12) << std::fixed
         << std::setprecision(2) << llVsLl128 << "x" << std::right
         << std::setw(11) << std::fixed << std::setprecision(1) << r.ncclTime
         << std::right << std::setw(11) << std::fixed << std::setprecision(1)
         << r.ll128Time << std::right << std::setw(11) << std::fixed
         << std::setprecision(1) << r.llTime << "\n";
    }
    ss << "======================================================================================================================================\n";
    ss << "BW (Bandwidth) = Data transferred / time, in GB/s\n";
    ss << "Lat (Latency) = Average transfer time per iteration, in microseconds\n";
    ss << "LL/NCCL and LL/LL128 = LL Bandwidth / baseline Bandwidth (>1 = LL faster)\n";
    ss << "======================================================================================================================================\n";

    std::cout << ss.str();
  }

  float runNcclBidirectionalBenchmark(std::size_t nBytes, float& timeUs) {
    DeviceBuffer sendBuff(nBytes);
    DeviceBuffer recvBuff(nBytes);
    CUDA_CHECK(cudaMemset(sendBuff.get(), globalRank, nBytes));
    CUDA_CHECK(cudaMemset(recvBuff.get(), 0, nBytes));

    int ncclPeer = (globalRank == 0) ? 1 : 0;
    CudaEvent start, stop;

    bootstrap->barrierAll();
    for (int i = 0; i < kWarmupIters; i++) {
      NCCL_CHECK(ncclGroupStart());
      NCCL_CHECK(ncclSend(
          sendBuff.get(), nBytes, ncclChar, ncclPeer, ncclComm_, stream_));
      NCCL_CHECK(ncclRecv(
          recvBuff.get(), nBytes, ncclChar, ncclPeer, ncclComm_, stream_));
      NCCL_CHECK(ncclGroupEnd());
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    CUDA_CHECK(cudaEventRecord(start.get(), stream_));
    for (int i = 0; i < kBenchmarkIters; i++) {
      NCCL_CHECK(ncclGroupStart());
      NCCL_CHECK(ncclSend(
          sendBuff.get(), nBytes, ncclChar, ncclPeer, ncclComm_, stream_));
      NCCL_CHECK(ncclRecv(
          recvBuff.get(), nBytes, ncclChar, ncclPeer, ncclComm_, stream_));
      NCCL_CHECK(ncclGroupEnd());
    }
    CUDA_CHECK(cudaEventRecord(stop.get(), stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    float totalTime_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime_ms, start.get(), stop.get()));
    float avgTime_ms = totalTime_ms / kBenchmarkIters;
    timeUs = avgTime_ms * 1000.0f;
    float bandwidth_GBps = (2.0f * nBytes / 1e9f) / (avgTime_ms / 1000.0f);

    bootstrap->barrierAll();
    return bandwidth_GBps;
  }

  float runLl128BidirectionalBenchmark(
      comms::pipes::P2pNvlTransportDevice& p2p,
      const BenchmarkConfig& config,
      float& timeUs) {
    DeviceBuffer sendBuff(config.nBytes);
    DeviceBuffer recvBuff(config.nBytes);
    CUDA_CHECK(cudaMemset(sendBuff.get(), globalRank, config.nBytes));
    CUDA_CHECK(cudaMemset(recvBuff.get(), 0, config.nBytes));

    dim3 gridDim(config.numBlocks);
    dim3 blockDim(config.numThreads);
    CudaEvent start, stop;

    std::size_t nBytes = config.nBytes;
    void* sendPtr = sendBuff.get();
    void* recvPtr = recvBuff.get();
    Timeout timeout;
    void* args[] = {&p2p, &sendPtr, &recvPtr, &nBytes, &timeout};
    void* kernelFunc = (void*)comms::pipes::benchmark::p2pLl128Bidirectional;

    bootstrap->barrierAll();
    for (int i = 0; i < kWarmupIters; i++) {
      CUDA_CHECK(
          cudaLaunchKernel(kernelFunc, gridDim, blockDim, args, 0, stream_));
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    bootstrap->barrierAll();

    CUDA_CHECK(cudaEventRecord(start.get(), stream_));
    for (int i = 0; i < kBenchmarkIters; i++) {
      CUDA_CHECK(
          cudaLaunchKernel(kernelFunc, gridDim, blockDim, args, 0, stream_));
    }
    CUDA_CHECK(cudaEventRecord(stop.get(), stream_));
    CUDA_CHECK(cudaDeviceSynchronize());

    float totalTime_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime_ms, start.get(), stop.get()));
    float avgTime_ms = totalTime_ms / kBenchmarkIters;
    timeUs = avgTime_ms * 1000.0f;
    float bandwidth_GBps =
        (2.0f * config.nBytes / 1e9f) / (avgTime_ms / 1000.0f);

    bootstrap->barrierAll();
    return bandwidth_GBps;
  }

  float runLlBidirectionalBenchmark(
      comms::pipes::P2pNvlTransportDevice& p2p,
      const BenchmarkConfig& config,
      float& timeUs) {
    DeviceBuffer sendBuff(config.nBytes);
    DeviceBuffer recvBuff(config.nBytes);
    CUDA_CHECK(cudaMemset(sendBuff.get(), globalRank, config.nBytes));
    CUDA_CHECK(cudaMemset(recvBuff.get(), 0, config.nBytes));

    dim3 gridDim(config.numBlocks);
    dim3 blockDim(config.numThreads);
    CudaEvent start, stop;

    std::size_t nBytes = config.nBytes;
    void* sendPtr = sendBuff.get();
    void* recvPtr = recvBuff.get();
    Timeout timeout;
    void* args[] = {&p2p, &sendPtr, &recvPtr, &nBytes, &timeout};
    void* kernelFunc = (void*)comms::pipes::benchmark::p2pLlBidirectional;

    bootstrap->barrierAll();
    for (int i = 0; i < kWarmupIters; i++) {
      CUDA_CHECK(
          cudaLaunchKernel(kernelFunc, gridDim, blockDim, args, 0, stream_));
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    bootstrap->barrierAll();

    CUDA_CHECK(cudaEventRecord(start.get(), stream_));
    for (int i = 0; i < kBenchmarkIters; i++) {
      CUDA_CHECK(
          cudaLaunchKernel(kernelFunc, gridDim, blockDim, args, 0, stream_));
    }
    CUDA_CHECK(cudaEventRecord(stop.get(), stream_));
    CUDA_CHECK(cudaDeviceSynchronize());

    float totalTime_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime_ms, start.get(), stop.get()));
    float avgTime_ms = totalTime_ms / kBenchmarkIters;
    timeUs = avgTime_ms * 1000.0f;
    float bandwidth_GBps =
        (2.0f * config.nBytes / 1e9f) / (avgTime_ms / 1000.0f);

    bootstrap->barrierAll();
    return bandwidth_GBps;
  }

  float runNcclBenchmarkCached(std::size_t nBytes, float& timeUs) {
    auto it = ncclCache_.find(nBytes);
    if (it != ncclCache_.end()) {
      timeUs = it->second.second;
      return it->second.first;
    }
    float bw = runNcclBenchmark(nBytes, timeUs);
    ncclCache_[nBytes] = {bw, timeUs};
    return bw;
  }

  float runNcclBidirectionalBenchmarkCached(std::size_t nBytes, float& timeUs) {
    auto key = nBytes | (1ULL << 63);
    auto it = ncclCache_.find(key);
    if (it != ncclCache_.end()) {
      timeUs = it->second.second;
      return it->second.first;
    }
    float bw = runNcclBidirectionalBenchmark(nBytes, timeUs);
    ncclCache_[key] = {bw, timeUs};
    return bw;
  }

  void run_unidirectional_sweep(
      int peerRank,
      const std::vector<LlConfig>& configs,
      const std::string& title) {
    std::vector<LlBenchmarkResult> results;

    for (const auto& cfg : configs) {
      LlBenchmarkResult result;
      result.testName = cfg.name;
      result.messageSize = cfg.nBytes;
      result.numBlocks = cfg.numBlocks;
      result.numThreads = cfg.numThreads;

      result.ncclBandwidth =
          runNcclBenchmarkCached(cfg.nBytes, result.ncclTime);

      const bool ll128Eligible = (cfg.nBytes % 16 == 0);

      // Create transport with both LL128 and LL buffers enabled
      comms::pipes::MultiPeerNvlTransportConfig p2pConfig{
          .dataBufferSize = cfg.nBytes,
          .chunkSize = cfg.nBytes,
          .pipelineDepth = 2,
          .ll128BufferSize =
              ll128Eligible ? comms::pipes::ll128_buffer_size(cfg.nBytes) : 0,
          .llBufferSize = comms::pipes::ll_buffer_size(cfg.nBytes),
      };

      comms::pipes::MultiPeerNvlTransport transport(
          globalRank, worldSize, bootstrap, p2pConfig);
      transport.exchange();

      auto p2p = transport.getP2pTransportDevice(peerRank);

      BenchmarkConfig benchConfig{
          .nBytes = cfg.nBytes,
          .numBlocks = cfg.numBlocks,
          .numThreads = cfg.numThreads,
          .name = cfg.name,
      };

      if (ll128Eligible) {
        result.ll128Bandwidth =
            runLl128Benchmark(p2p, benchConfig, result.ll128Time);
      }
      result.llBandwidth = runLlBenchmark(p2p, benchConfig, result.llTime);

      results.push_back(result);
    }

    printResultsTable(results, title);
  }

  void run_bidirectional_sweep(
      int peerRank,
      const std::vector<LlConfig>& configs,
      const std::string& title) {
    std::vector<LlBenchmarkResult> results;

    for (const auto& cfg : configs) {
      LlBenchmarkResult result;
      result.testName = cfg.name;
      result.messageSize = cfg.nBytes;
      result.numBlocks = cfg.numBlocks;
      result.numThreads = cfg.numThreads;

      result.ncclBandwidth =
          runNcclBidirectionalBenchmarkCached(cfg.nBytes, result.ncclTime);

      // Bidirectional kernels call partition_interleaved(2), requiring >= 2
      // warp groups
      const bool bidirEligible = (cfg.numBlocks * (cfg.numThreads / 32) >= 2);

      if (!bidirEligible) {
        continue;
      }

      const bool ll128Eligible = (cfg.nBytes % 16 == 0);

      comms::pipes::MultiPeerNvlTransportConfig p2pConfig{
          .dataBufferSize = cfg.nBytes,
          .chunkSize = cfg.nBytes,
          .pipelineDepth = 2,
          .ll128BufferSize =
              ll128Eligible ? comms::pipes::ll128_buffer_size(cfg.nBytes) : 0,
          .llBufferSize = comms::pipes::ll_buffer_size(cfg.nBytes),
      };

      comms::pipes::MultiPeerNvlTransport transport(
          globalRank, worldSize, bootstrap, p2pConfig);
      transport.exchange();

      auto p2p = transport.getP2pTransportDevice(peerRank);

      BenchmarkConfig benchConfig{
          .nBytes = cfg.nBytes,
          .numBlocks = cfg.numBlocks,
          .numThreads = cfg.numThreads,
          .name = cfg.name,
      };

      if (ll128Eligible) {
        result.ll128Bandwidth =
            runLl128BidirectionalBenchmark(p2p, benchConfig, result.ll128Time);
      }
      result.llBandwidth =
          runLlBidirectionalBenchmark(p2p, benchConfig, result.llTime);

      results.push_back(result);
    }

    printResultsTable(results, title);
  }

  ncclComm_t ncclComm_{};
  cudaStream_t stream_{};
  std::unordered_map<std::size_t, std::pair<float, float>> ncclCache_;
};

TEST_F(P2pLlBenchmarkFixture, UnidirectionalBenchmark) {
  if (worldSize != 2) {
    GTEST_SKIP() << "Requires exactly 2 ranks, got " << worldSize;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;

  // LL sweet-spot configurations: very small messages, all multiples of 8
  std::vector<LlConfig> configs = {
      // Very small messages — LL sweet spot
      {8, 1, 32, "LL_8B"},
      {16, 1, 32, "LL_16B"},
      {32, 1, 32, "LL_32B"},
      {64, 1, 32, "LL_64B"},
      {128, 1, 32, "LL_128B"},
      {256, 1, 32, "LL_256B"},
      {512, 1, 32, "LL_512B"},
      {1024, 1, 32, "LL_1KB"},
      {2 * 1024, 1, 32, "LL_2KB"},
      {4 * 1024, 1, 32, "LL_4KB"},
      // 128-thread variants
      {64, 1, 128, "LL_64B_128t"},
      {256, 1, 128, "LL_256B_128t"},
      {1024, 1, 128, "LL_1KB_128t"},
      {4 * 1024, 1, 128, "LL_4KB_128t"},
      // 512-thread variants
      {64, 1, 512, "LL_64B_512t"},
      {256, 1, 512, "LL_256B_512t"},
      {1024, 1, 512, "LL_1KB_512t"},
      {4 * 1024, 1, 512, "LL_4KB_512t"},
      // Crossover region — medium messages
      {8 * 1024, 1, 128, "LL_8KB"},
      {16 * 1024, 1, 512, "LL_16KB"},
      {32 * 1024, 2, 512, "LL_32KB"},
      {64 * 1024, 4, 512, "LL_64KB"},
      {128 * 1024, 8, 512, "LL_128KB"},
      {256 * 1024, 16, 512, "LL_256KB"},
      // Multi-block variants for small messages
      {1024, 2, 128, "LL_1KB_2b"},
      {4 * 1024, 2, 128, "LL_4KB_2b"},
      {8 * 1024, 2, 128, "LL_8KB_2b"},
      {8 * 1024, 4, 512, "LL_8KB_4b"},
      {16 * 1024, 4, 512, "LL_16KB_4b"},
  };

  run_unidirectional_sweep(
      peerRank,
      configs,
      "NCCL vs LL128 vs LL UNIDIRECTIONAL Benchmark Results");
}

TEST_F(P2pLlBenchmarkFixture, BidirectionalBenchmark) {
  if (worldSize != 2) {
    GTEST_SKIP() << "Requires exactly 2 ranks, got " << worldSize;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;

  std::vector<LlConfig> configs = {
      // Very small messages
      {8, 1, 64, "Bidir_8B"},
      {16, 1, 64, "Bidir_16B"},
      {32, 1, 64, "Bidir_32B"},
      {64, 1, 64, "Bidir_64B"},
      {128, 1, 64, "Bidir_128B"},
      {256, 1, 64, "Bidir_256B"},
      {512, 1, 64, "Bidir_512B"},
      {1024, 1, 64, "Bidir_1KB"},
      {2 * 1024, 1, 64, "Bidir_2KB"},
      {4 * 1024, 1, 64, "Bidir_4KB"},
      // 128-thread variants
      {64, 1, 128, "Bidir_64B_128t"},
      {256, 1, 128, "Bidir_256B_128t"},
      {1024, 1, 128, "Bidir_1KB_128t"},
      {4 * 1024, 1, 128, "Bidir_4KB_128t"},
      // 512-thread variants
      {64, 1, 512, "Bidir_64B_512t"},
      {256, 1, 512, "Bidir_256B_512t"},
      {1024, 1, 512, "Bidir_1KB_512t"},
      {4 * 1024, 1, 512, "Bidir_4KB_512t"},
      // Crossover region
      {8 * 1024, 2, 128, "Bidir_8KB"},
      {16 * 1024, 2, 512, "Bidir_16KB"},
      {32 * 1024, 4, 512, "Bidir_32KB"},
      {64 * 1024, 8, 512, "Bidir_64KB"},
      {128 * 1024, 16, 512, "Bidir_128KB"},
      {256 * 1024, 32, 512, "Bidir_256KB"},
      // Multi-block variants
      {1024, 2, 128, "Bidir_1KB_2b"},
      {4 * 1024, 2, 128, "Bidir_4KB_2b"},
      {8 * 1024, 4, 512, "Bidir_8KB_4b"},
      {16 * 1024, 4, 512, "Bidir_16KB_4b"},
  };

  run_bidirectional_sweep(
      peerRank, configs, "NCCL vs LL128 vs LL BIDIRECTIONAL Benchmark Results");
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
