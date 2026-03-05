// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>
#include <nccl.h>

#include "comms/common/CudaWrap.h"
#include "comms/pipes/MultiPeerNvlTransport.h"
#include "comms/pipes/benchmarks/BenchmarkKernel.cuh"
#include "comms/pipes/benchmarks/BenchmarkMacros.h"
#include "comms/pipes/benchmarks/P2pNvlBenchmarkUtils.h"
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

// Configuration for LL128 benchmark sweep points
struct Ll128Config {
  std::size_t nBytes;
  int numBlocks;
  int numThreads;
  std::string name;
};

// Threshold above which Simple gets its own transport with optimal chunking
constexpr std::size_t kSimpleChunkThreshold = 8 * 1024;
constexpr std::size_t kSimpleChunkSize = 8 * 1024;

// Result struct for 3-way comparison (NCCL vs Simple vs LL128)
struct Ll128BenchmarkResult {
  std::string testName;
  std::size_t messageSize{};
  int numBlocks{};
  int numThreads{};
  float ncclBandwidth{};
  float simpleBandwidth{};
  float ll128Bandwidth{};
  float ncclTime{}; // microseconds
  float simpleTime{}; // microseconds
  float ll128Time{}; // microseconds
};

class P2pLl128BenchmarkFixture : public meta::comms::BenchmarkTestFixture {
 protected:
  void SetUp() override {
    BenchmarkTestFixture::SetUp();
    CUDA_CHECK_VOID(cudaSetDevice(localRank));
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
      if (globalRank == 0) {
        NCCL_CHECK(
            ncclSend(sendBuff.get(), nBytes, ncclChar, 1, ncclComm_, stream_));
      } else if (globalRank == 1) {
        NCCL_CHECK(
            ncclRecv(recvBuff.get(), nBytes, ncclChar, 0, ncclComm_, stream_));
      }
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
    bootstrap->barrierAll();

    // Benchmark
    CUDA_CHECK(cudaEventRecord(start.get(), stream_));
    for (int i = 0; i < kBenchmarkIters; i++) {
      if (globalRank == 0) {
        NCCL_CHECK(
            ncclSend(sendBuff.get(), nBytes, ncclChar, 1, ncclComm_, stream_));
      } else if (globalRank == 1) {
        NCCL_CHECK(
            ncclRecv(recvBuff.get(), nBytes, ncclChar, 0, ncclComm_, stream_));
      }
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

  float runSimpleBenchmark(
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
    SyncScope groupScope = config.groupScope;
    void* devicePtr = (isSend ? sendBuff.get() : recvBuff.get());
    Timeout timeout;
    void* args[] = {&p2p, &devicePtr, &nBytes, &groupScope, &timeout};
    void* kernelFunc = isSend ? (void*)comms::pipes::benchmark::p2pSend
                              : (void*)comms::pipes::benchmark::p2pRecv;

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
    int64_t flagValue = 1;
    Timeout timeout;
    void* args[] = {&p2p, &devicePtr, &nBytes, &flagValue, &timeout};
    void* kernelFunc = isSend ? (void*)comms::pipes::benchmark::p2pLl128Send
                              : (void*)comms::pipes::benchmark::p2pLl128Recv;

    // Warmup
    bootstrap->barrierAll();
    for (int i = 0; i < kWarmupIters; i++) {
      bootstrap->barrierAll();
      flagValue = static_cast<int64_t>(i + 1);
      CUDA_CHECK(
          cudaLaunchKernel(kernelFunc, gridDim, blockDim, args, 0, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    bootstrap->barrierAll();

    // Benchmark
    CUDA_CHECK(cudaEventRecord(start.get(), stream));
    for (int i = 0; i < kBenchmarkIters; i++) {
      flagValue = static_cast<int64_t>(kWarmupIters + i + 1);
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
      const std::vector<Ll128BenchmarkResult>& results,
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
       << "NCCL BW" << std::right << std::setw(11) << "Simple BW" << std::right
       << std::setw(11) << "LL128 BW" << std::right << std::setw(12)
       << "LL128/NCCL" << std::right << std::setw(13) << "LL128/Simple"
       << std::right << std::setw(11) << "NCCL Lat" << std::right
       << std::setw(11) << "Simple Lat" << std::right << std::setw(11)
       << "LL128 Lat\n";
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
      float ll128VsNccl =
          (r.ncclBandwidth > 0) ? r.ll128Bandwidth / r.ncclBandwidth : 0;
      float ll128VsSimple =
          (r.simpleBandwidth > 0) ? r.ll128Bandwidth / r.simpleBandwidth : 0;

      ss << std::left << std::setw(16) << r.testName << std::right
         << std::setw(10) << msgSize << std::right << std::setw(8)
         << r.numBlocks << std::right << std::setw(9) << r.numThreads
         << std::right << std::setw(11) << std::fixed << std::setprecision(2)
         << r.ncclBandwidth << std::right << std::setw(11) << std::fixed
         << std::setprecision(2) << r.simpleBandwidth << std::right
         << std::setw(11) << std::fixed << std::setprecision(2)
         << r.ll128Bandwidth << std::right << std::setw(11) << std::fixed
         << std::setprecision(2) << ll128VsNccl << "x" << std::right
         << std::setw(12) << std::fixed << std::setprecision(2) << ll128VsSimple
         << "x" << std::right << std::setw(11) << std::fixed
         << std::setprecision(1) << r.ncclTime << std::right << std::setw(11)
         << std::fixed << std::setprecision(1) << r.simpleTime << std::right
         << std::setw(11) << std::fixed << std::setprecision(1) << r.ll128Time
         << "\n";
    }
    ss << "======================================================================================================================================\n";
    ss << "BW (Bandwidth) = Data transferred / time, in GB/s\n";
    ss << "Lat (Latency) = Average transfer time per iteration, in microseconds\n";
    ss << "LL128/NCCL and LL128/Simple = LL128 Bandwidth / baseline Bandwidth (>1 = LL128 faster)\n";
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

  float runSimpleBidirectionalBenchmark(
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
    SyncScope groupScope = config.groupScope;
    Timeout timeout;
    void* args[] = {&p2p, &sendPtr, &recvPtr, &nBytes, &groupScope, &timeout};
    void* kernelFunc = (void*)comms::pipes::benchmark::p2pBidirectional;

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
    int64_t flagValue = 1;
    Timeout timeout;
    void* args[] = {&p2p, &sendPtr, &recvPtr, &nBytes, &flagValue, &timeout};
    void* kernelFunc = (void*)comms::pipes::benchmark::p2pLl128Bidirectional;

    bootstrap->barrierAll();
    for (int i = 0; i < kWarmupIters; i++) {
      flagValue = static_cast<int64_t>(i + 1);
      CUDA_CHECK(
          cudaLaunchKernel(kernelFunc, gridDim, blockDim, args, 0, stream_));
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    bootstrap->barrierAll();

    CUDA_CHECK(cudaEventRecord(start.get(), stream_));
    for (int i = 0; i < kBenchmarkIters; i++) {
      flagValue = static_cast<int64_t>(kWarmupIters + i + 1);
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
    // Use a distinct key by adding a sentinel bit to distinguish bidir
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

  ncclComm_t ncclComm_{};
  cudaStream_t stream_{};
  std::unordered_map<std::size_t, std::pair<float, float>> ncclCache_;
};

TEST_F(P2pLl128BenchmarkFixture, UnidirectionalBenchmark) {
  if (worldSize != 2) {
    XLOGF(DBG1, "Skipping test: requires exactly 2 ranks, got {}", worldSize);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;

  // LL128 sweet-spot configurations: small/medium messages, all multiples of 16
  std::vector<Ll128Config> configs = {
      // Small messages — LL128 sweet spot
      {64, 1, 128, "LL128_64B"},
      {128, 1, 128, "LL128_128B"},
      {256, 1, 128, "LL128_256B"},
      {512, 1, 128, "LL128_512B"},
      {1024, 1, 128, "LL128_1KB"},
      {2 * 1024, 1, 128, "LL128_2KB"},
      {3 * 1024, 1, 128, "LL128_3KB"},
      {4 * 1024, 1, 128, "LL128_4KB"},
      // Crossover region
      {5 * 1024, 1, 128, "LL128_5KB"},
      {6 * 1024, 1, 128, "LL128_6KB"},
      {8 * 1024, 1, 128, "LL128_8KB"},
      // Medium/large messages
      {16 * 1024, 1, 128, "LL128_16KB"},
      {32 * 1024, 2, 128, "LL128_32KB"},
      {64 * 1024, 4, 128, "LL128_64KB"},
      {128 * 1024, 4, 128, "LL128_128KB"},
      {256 * 1024, 8, 128, "LL128_256KB"},
      // Thread-count and block-count sweep
      {32 * 1024, 2, 256, "LL128_32KB_256t"},
      {32 * 1024, 4, 128, "LL128_32KB_4b"},
      {64 * 1024, 4, 256, "LL128_64KB_256t"},
      {64 * 1024, 8, 128, "LL128_64KB_8b"},
      {128 * 1024, 8, 128, "LL128_128KB_8b"},
      {128 * 1024, 8, 256, "LL128_128KB_256t"},
      {256 * 1024, 8, 256, "LL128_256KB_256t"},
      {256 * 1024, 16, 128, "LL128_256KB_16b"},
      // Max warp configs
      {128 * 1024, 16, 256, "LL128_128KB_max"},
      {256 * 1024, 16, 256, "LL128_256KB_max"},
  };

  std::vector<Ll128BenchmarkResult> results;

  for (const auto& cfg : configs) {
    Ll128BenchmarkResult result;
    result.testName = cfg.name;
    result.messageSize = cfg.nBytes;
    result.numBlocks = cfg.numBlocks;
    result.numThreads = cfg.numThreads;

    // Run NCCL benchmark (cached for sweep configs at the same size)
    result.ncclBandwidth = runNcclBenchmarkCached(cfg.nBytes, result.ncclTime);

    if (cfg.nBytes <= kSimpleChunkThreshold) {
      // Small messages: single transport works for both Simple and LL128
      comms::pipes::MultiPeerNvlTransportConfig p2pConfig{
          .dataBufferSize = cfg.nBytes,
          .chunkSize = cfg.nBytes,
          .pipelineDepth = 2,
          .ll128BufferSize = comms::pipes::ll128_buffer_size(cfg.nBytes),
      };

      comms::pipes::MultiPeerNvlTransport transport(
          globalRank, worldSize, bootstrap, p2pConfig);
      transport.exchange();

      auto p2p = transport.getP2pTransportDevice(peerRank);

      BenchmarkConfig benchConfig{
          .nBytes = cfg.nBytes,
          .stagedBufferSize = cfg.nBytes,
          .numBlocks = cfg.numBlocks,
          .numThreads = cfg.numThreads,
          .pipelineDepth = 2,
          .chunkSize = cfg.nBytes,
          .name = cfg.name,
      };

      result.simpleBandwidth =
          runSimpleBenchmark(p2p, benchConfig, result.simpleTime);
      result.ll128Bandwidth =
          runLl128Benchmark(p2p, benchConfig, result.ll128Time);
    } else {
      // Large messages: separate transports for fair Simple comparison.
      // LL128 transport uses ll128BufferSize; Simple transport uses optimal
      // chunking with BLOCK scope for better pipelining.

      // LL128 transport (created first — exchange() order must match)
      comms::pipes::MultiPeerNvlTransportConfig ll128Config{
          .dataBufferSize = cfg.nBytes,
          .chunkSize = cfg.nBytes,
          .pipelineDepth = 2,
          .ll128BufferSize = comms::pipes::ll128_buffer_size(cfg.nBytes),
      };
      comms::pipes::MultiPeerNvlTransport ll128Transport(
          globalRank, worldSize, bootstrap, ll128Config);
      ll128Transport.exchange();

      // Simple transport with optimal chunking
      comms::pipes::MultiPeerNvlTransportConfig simpleConfig{
          .dataBufferSize = cfg.nBytes,
          .chunkSize = kSimpleChunkSize,
          .pipelineDepth = 2,
      };
      comms::pipes::MultiPeerNvlTransport simpleTransport(
          globalRank, worldSize, bootstrap, simpleConfig);
      simpleTransport.exchange();

      auto ll128P2p = ll128Transport.getP2pTransportDevice(peerRank);
      auto simpleP2p = simpleTransport.getP2pTransportDevice(peerRank);

      BenchmarkConfig simpleBenchConfig{
          .nBytes = cfg.nBytes,
          .stagedBufferSize = cfg.nBytes,
          .numBlocks = cfg.numBlocks,
          .numThreads = cfg.numThreads,
          .pipelineDepth = 2,
          .chunkSize = kSimpleChunkSize,
          .groupScope = SyncScope::BLOCK,
          .name = cfg.name,
      };

      BenchmarkConfig ll128BenchConfig{
          .nBytes = cfg.nBytes,
          .stagedBufferSize = cfg.nBytes,
          .numBlocks = cfg.numBlocks,
          .numThreads = cfg.numThreads,
          .pipelineDepth = 2,
          .chunkSize = cfg.nBytes,
          .name = cfg.name,
      };

      result.simpleBandwidth =
          runSimpleBenchmark(simpleP2p, simpleBenchConfig, result.simpleTime);
      result.ll128Bandwidth =
          runLl128Benchmark(ll128P2p, ll128BenchConfig, result.ll128Time);
    }

    results.push_back(result);
  }

  printResultsTable(
      results, "NCCL vs Simple vs LL128 UNIDIRECTIONAL Benchmark Results");
}

TEST_F(P2pLl128BenchmarkFixture, BidirectionalBenchmark) {
  if (worldSize != 2) {
    XLOGF(DBG1, "Skipping test: requires exactly 2 ranks, got {}", worldSize);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;

  std::vector<Ll128Config> configs = {
      // Small messages
      {64, 1, 128, "Bidir_64B"},
      {128, 1, 128, "Bidir_128B"},
      {256, 1, 128, "Bidir_256B"},
      {512, 1, 128, "Bidir_512B"},
      {1024, 1, 128, "Bidir_1KB"},
      {2 * 1024, 1, 128, "Bidir_2KB"},
      {4 * 1024, 1, 128, "Bidir_4KB"},
      // Crossover region
      {5 * 1024, 1, 128, "Bidir_5KB"},
      {6 * 1024, 1, 128, "Bidir_6KB"},
      {8 * 1024, 1, 128, "Bidir_8KB"},
      // Medium/large messages
      {32 * 1024, 2, 128, "Bidir_32KB"},
      {64 * 1024, 4, 128, "Bidir_64KB"},
      {128 * 1024, 4, 128, "Bidir_128KB"},
      {256 * 1024, 8, 128, "Bidir_256KB"},
      // More blocks (partition_interleaved halves warps per direction)
      {32 * 1024, 4, 128, "Bidir_32KB_4b"},
      {64 * 1024, 8, 128, "Bidir_64KB_8b"},
      {128 * 1024, 8, 128, "Bidir_128KB_8b"},
      {256 * 1024, 16, 128, "Bidir_256KB_16b"},
      // 256-thread bidirectional sweep
      {32 * 1024, 4, 256, "Bidir_32KB_256t"},
      {64 * 1024, 8, 256, "Bidir_64KB_256t"},
      {128 * 1024, 8, 256, "Bidir_128KB_256t"},
      {256 * 1024, 16, 256, "Bidir_256KB_256t"},
  };

  std::vector<Ll128BenchmarkResult> results;

  for (const auto& cfg : configs) {
    Ll128BenchmarkResult result;
    result.testName = cfg.name;
    result.messageSize = cfg.nBytes;
    result.numBlocks = cfg.numBlocks;
    result.numThreads = cfg.numThreads;

    result.ncclBandwidth =
        runNcclBidirectionalBenchmarkCached(cfg.nBytes, result.ncclTime);

    if (cfg.nBytes <= kSimpleChunkThreshold) {
      comms::pipes::MultiPeerNvlTransportConfig p2pConfig{
          .dataBufferSize = cfg.nBytes,
          .chunkSize = cfg.nBytes,
          .pipelineDepth = 2,
          .ll128BufferSize = comms::pipes::ll128_buffer_size(cfg.nBytes),
      };

      comms::pipes::MultiPeerNvlTransport transport(
          globalRank, worldSize, bootstrap, p2pConfig);
      transport.exchange();

      auto p2p = transport.getP2pTransportDevice(peerRank);

      BenchmarkConfig benchConfig{
          .nBytes = cfg.nBytes,
          .stagedBufferSize = cfg.nBytes,
          .numBlocks = cfg.numBlocks,
          .numThreads = cfg.numThreads,
          .pipelineDepth = 2,
          .chunkSize = cfg.nBytes,
          .name = cfg.name,
      };

      result.simpleBandwidth =
          runSimpleBidirectionalBenchmark(p2p, benchConfig, result.simpleTime);
      result.ll128Bandwidth =
          runLl128BidirectionalBenchmark(p2p, benchConfig, result.ll128Time);
    } else {
      // LL128 transport (created first — exchange() order must match)
      comms::pipes::MultiPeerNvlTransportConfig ll128Config{
          .dataBufferSize = cfg.nBytes,
          .chunkSize = cfg.nBytes,
          .pipelineDepth = 2,
          .ll128BufferSize = comms::pipes::ll128_buffer_size(cfg.nBytes),
      };
      comms::pipes::MultiPeerNvlTransport ll128Transport(
          globalRank, worldSize, bootstrap, ll128Config);
      ll128Transport.exchange();

      // Simple transport with optimal chunking
      comms::pipes::MultiPeerNvlTransportConfig simpleConfig{
          .dataBufferSize = cfg.nBytes,
          .chunkSize = kSimpleChunkSize,
          .pipelineDepth = 2,
      };
      comms::pipes::MultiPeerNvlTransport simpleTransport(
          globalRank, worldSize, bootstrap, simpleConfig);
      simpleTransport.exchange();

      auto ll128P2p = ll128Transport.getP2pTransportDevice(peerRank);
      auto simpleP2p = simpleTransport.getP2pTransportDevice(peerRank);

      BenchmarkConfig simpleBenchConfig{
          .nBytes = cfg.nBytes,
          .stagedBufferSize = cfg.nBytes,
          .numBlocks = cfg.numBlocks,
          .numThreads = cfg.numThreads,
          .pipelineDepth = 2,
          .chunkSize = kSimpleChunkSize,
          .groupScope = SyncScope::BLOCK,
          .name = cfg.name,
      };

      BenchmarkConfig ll128BenchConfig{
          .nBytes = cfg.nBytes,
          .stagedBufferSize = cfg.nBytes,
          .numBlocks = cfg.numBlocks,
          .numThreads = cfg.numThreads,
          .pipelineDepth = 2,
          .chunkSize = cfg.nBytes,
          .name = cfg.name,
      };

      result.simpleBandwidth = runSimpleBidirectionalBenchmark(
          simpleP2p, simpleBenchConfig, result.simpleTime);
      result.ll128Bandwidth = runLl128BidirectionalBenchmark(
          ll128P2p, ll128BenchConfig, result.ll128Time);
    }

    results.push_back(result);
  }

  printResultsTable(
      results, "NCCL vs Simple vs LL128 BIDIRECTIONAL Benchmark Results");
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
