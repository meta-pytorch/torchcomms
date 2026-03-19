// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>
#include <nccl.h>

#include "comms/common/CudaWrap.h"
#include "comms/pipes/MultiPeerNvlTransport.h"
#include "comms/pipes/benchmarks/BenchmarkKernel.cuh"
#include "comms/pipes/benchmarks/BenchmarkMacros.h"
#include "comms/pipes/benchmarks/P2pNvlBenchmarkUtils.h"
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

// Configuration for LL128 benchmark sweep points
struct Ll128Config {
  std::size_t nBytes;
  int numBlocks;
  int numThreads;
  std::string name;
  std::size_t bufferNumPackets{
      0}; // 0 = buffer sized to fit message (no chunking)
};

// Threshold above which Simple gets its own transport with optimal chunking
constexpr std::size_t kSimpleChunkThreshold = 8 * 1024;
constexpr std::size_t kSimpleChunkSize = 8 * 1024;

// Message sizes used by auto-tuned benchmarks (both uni- and bidirectional).
static const std::vector<std::size_t> kAutoTuneMessageSizes = {
    64,
    256,
    1024,
    2 * 1024,
    4 * 1024,
    8 * 1024,
    16 * 1024,
    32 * 1024,
    64 * 1024,
    128 * 1024,
    256 * 1024,
    512 * 1024,
    1024 * 1024,
    2 * 1024 * 1024,
    4 * 1024 * 1024,
    8 * 1024 * 1024,
    16 * 1024 * 1024,
};

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

  void run_unidirectional_sweep(
      int peerRank,
      const std::vector<Ll128Config>& configs,
      const std::string& title) {
    std::vector<Ll128BenchmarkResult> results;

    for (const auto& cfg : configs) {
      Ll128BenchmarkResult result;
      result.testName = cfg.name;
      result.messageSize = cfg.nBytes;
      result.numBlocks = cfg.numBlocks;
      result.numThreads = cfg.numThreads;

      result.ncclBandwidth =
          runNcclBenchmarkCached(cfg.nBytes, result.ncclTime);

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
            runSimpleBenchmark(p2p, benchConfig, result.simpleTime);
        result.ll128Bandwidth =
            runLl128Benchmark(p2p, benchConfig, result.ll128Time);
      } else {
        comms::pipes::MultiPeerNvlTransportConfig ll128Config{
            .dataBufferSize = cfg.nBytes,
            .chunkSize = cfg.nBytes,
            .pipelineDepth = 2,
            .ll128BufferSize = comms::pipes::ll128_buffer_size(cfg.nBytes),
        };
        comms::pipes::MultiPeerNvlTransport ll128Transport(
            globalRank, worldSize, bootstrap, ll128Config);
        ll128Transport.exchange();

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

    printResultsTable(results, title);
  }

  void run_bidirectional_sweep(
      int peerRank,
      const std::vector<Ll128Config>& configs,
      const std::string& title) {
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

        result.simpleBandwidth = runSimpleBidirectionalBenchmark(
            p2p, benchConfig, result.simpleTime);
        result.ll128Bandwidth =
            runLl128BidirectionalBenchmark(p2p, benchConfig, result.ll128Time);
      } else {
        comms::pipes::MultiPeerNvlTransportConfig ll128Config{
            .dataBufferSize = cfg.nBytes,
            .chunkSize = cfg.nBytes,
            .pipelineDepth = 2,
            .ll128BufferSize = comms::pipes::ll128_buffer_size(cfg.nBytes),
        };
        comms::pipes::MultiPeerNvlTransport ll128Transport(
            globalRank, worldSize, bootstrap, ll128Config);
        ll128Transport.exchange();

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

    printResultsTable(results, title);
  }

  void run_chunked_ll128_sweep(
      int peerRank,
      const std::vector<Ll128Config>& configs,
      const std::string& title) {
    std::vector<Ll128BenchmarkResult> results;

    for (const auto& cfg : configs) {
      Ll128BenchmarkResult result;
      result.testName = cfg.name;
      result.messageSize = cfg.nBytes;
      result.numBlocks = cfg.numBlocks;
      result.numThreads = cfg.numThreads;

      result.ncclBandwidth =
          runNcclBenchmarkCached(cfg.nBytes, result.ncclTime);

      std::size_t ll128BufSize = (cfg.bufferNumPackets > 0)
          ? cfg.bufferNumPackets * comms::pipes::kLl128PacketSize
          : comms::pipes::ll128_buffer_size(cfg.nBytes);

      comms::pipes::MultiPeerNvlTransportConfig ll128Config{
          .dataBufferSize = cfg.nBytes,
          .chunkSize = cfg.nBytes,
          .pipelineDepth = 2,
          .ll128BufferSize = ll128BufSize,
      };
      comms::pipes::MultiPeerNvlTransport ll128Transport(
          globalRank, worldSize, bootstrap, ll128Config);
      ll128Transport.exchange();

      auto ll128P2p = ll128Transport.getP2pTransportDevice(peerRank);
      BenchmarkConfig ll128BenchConfig{
          .nBytes = cfg.nBytes,
          .stagedBufferSize = cfg.nBytes,
          .numBlocks = cfg.numBlocks,
          .numThreads = cfg.numThreads,
          .pipelineDepth = 2,
          .chunkSize = cfg.nBytes,
          .name = cfg.name,
      };

      result.simpleBandwidth = 0.0f;
      result.simpleTime = 0.0f;
      result.ll128Bandwidth =
          runLl128Benchmark(ll128P2p, ll128BenchConfig, result.ll128Time);

      results.push_back(result);
    }

    printResultsTable(results, title);
  }

  void run_chunked_ll128_bidir_sweep(
      int peerRank,
      const std::vector<Ll128Config>& configs,
      const std::string& title) {
    std::vector<Ll128BenchmarkResult> results;

    for (const auto& cfg : configs) {
      Ll128BenchmarkResult result;
      result.testName = cfg.name;
      result.messageSize = cfg.nBytes;
      result.numBlocks = cfg.numBlocks;
      result.numThreads = cfg.numThreads;

      result.ncclBandwidth =
          runNcclBidirectionalBenchmarkCached(cfg.nBytes, result.ncclTime);

      std::size_t ll128BufSize = (cfg.bufferNumPackets > 0)
          ? cfg.bufferNumPackets * comms::pipes::kLl128PacketSize
          : comms::pipes::ll128_buffer_size(cfg.nBytes);

      comms::pipes::MultiPeerNvlTransportConfig ll128Config{
          .dataBufferSize = cfg.nBytes,
          .chunkSize = cfg.nBytes,
          .pipelineDepth = 2,
          .ll128BufferSize = ll128BufSize,
      };
      comms::pipes::MultiPeerNvlTransport ll128Transport(
          globalRank, worldSize, bootstrap, ll128Config);
      ll128Transport.exchange();

      auto ll128P2p = ll128Transport.getP2pTransportDevice(peerRank);
      BenchmarkConfig ll128BenchConfig{
          .nBytes = cfg.nBytes,
          .stagedBufferSize = cfg.nBytes,
          .numBlocks = cfg.numBlocks,
          .numThreads = cfg.numThreads,
          .pipelineDepth = 2,
          .chunkSize = cfg.nBytes,
          .name = cfg.name,
      };

      result.simpleBandwidth = 0.0f;
      result.simpleTime = 0.0f;
      result.ll128Bandwidth = runLl128BidirectionalBenchmark(
          ll128P2p, ll128BenchConfig, result.ll128Time);

      results.push_back(result);
    }

    printResultsTable(results, title);
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
      // 512-thread variants for small messages (validate Recommendation 2)
      {64, 1, 512, "LL128_64B_512t"},
      {128, 1, 512, "LL128_128B_512t"},
      {256, 1, 512, "LL128_256B_512t"},
      {512, 1, 512, "LL128_512B_512t"},
      {1024, 1, 512, "LL128_1KB_512t"},
      {2 * 1024, 1, 512, "LL128_2KB_512t"},
      {3 * 1024, 1, 512, "LL128_3KB_512t"},
      {4 * 1024, 1, 512, "LL128_4KB_512t"},
      // Crossover region
      {5 * 1024, 1, 128, "LL128_5KB"},
      {6 * 1024, 1, 128, "LL128_6KB"},
      {8 * 1024, 1, 128, "LL128_8KB"},
      // Medium/large messages (512 threads default for ≥16KB)
      {16 * 1024, 1, 512, "LL128_16KB"},
      {32 * 1024, 2, 512, "LL128_32KB"},
      {64 * 1024, 4, 512, "LL128_64KB"},
      {128 * 1024, 4, 512, "LL128_128KB"},
      {256 * 1024, 8, 512, "LL128_256KB"},
      // Block-count sweep
      {32 * 1024, 4, 128, "LL128_32KB_4b"},
      {64 * 1024, 8, 128, "LL128_64KB_8b"},
      {128 * 1024, 8, 128, "LL128_128KB_8b"},
      {128 * 1024, 8, 512, "LL128_128KB_8b512t"},
      {256 * 1024, 16, 128, "LL128_256KB_16b"},
      // Max warp configs
      {128 * 1024, 16, 512, "LL128_128KB_max"},
      {128 * 1024, 32, 512, "LL128_128KB_32b"},
      {256 * 1024, 16, 512, "LL128_256KB_max"},
      {256 * 1024, 32, 512, "LL128_256KB_32b"},
      // Priority 1: Explore scaling ceiling (64b configs)
      {128 * 1024, 64, 512, "LL128_128KB_64b"},
      {256 * 1024, 64, 512, "LL128_256KB_64b"},
      {512 * 1024, 32, 512, "LL128_512KB_32b"},
      {512 * 1024, 64, 512, "LL128_512KB_64b"},
      // Priority 2: Close the 8-16KB gap with multi-block configs
      {8 * 1024, 2, 128, "LL128_8KB_2b"},
      {8 * 1024, 2, 512, "LL128_8KB_2b512t"},
      {16 * 1024, 2, 512, "LL128_16KB_2b"},
      {16 * 1024, 4, 512, "LL128_16KB_4b"},
      // Priority 3: Close the 32-64KB gap with higher block counts
      {32 * 1024, 8, 512, "LL128_32KB_8b"}, // 64 warps, ~4 pkts/warp
      {32 * 1024, 16, 512, "LL128_32KB_16b"}, // 128 warps, ~2 pkts/warp
      {64 * 1024, 16, 512, "LL128_64KB_16b"}, // 128 warps, ~4 pkts/warp
      {64 * 1024, 32, 512, "LL128_64KB_32b"}, // 256 warps, ~2 pkts/warp
      // Priority 4: Explore ceiling at 512KB+ and 1MB
      {512 * 1024, 128, 512, "LL128_512KB_128b"}, // 1024 warps, ~4 pkts/warp
      {1024 * 1024, 64, 512, "LL128_1MB_64b"}, // 512 warps, ~17 pkts/warp
      {1024 * 1024, 128, 512, "LL128_1MB_128b"}, // 1024 warps, ~9 pkts/warp
      // Round 4 Priority 1: Explore 32-64KB ceiling (~1 pkt/warp regime)
      {32 * 1024, 32, 512, "LL128_32KB_32b"}, // 256 warps, ~1 pkt/warp
      {64 * 1024, 64, 512, "LL128_64KB_64b"}, // 512 warps, ~1 pkt/warp
      // Round 4 Priority 2: 128KB 128b and 2MB scaling
      {128 * 1024, 128, 512, "LL128_128KB_128b"}, // 1024 warps
      {2 * 1024 * 1024, 128, 512, "LL128_2MB_128b"}, // verify ll128_buffer_size
      // Round 5 Priority 1: Close large-message parallelism gap (reach ~2
      // pkts/warp)
      {256 * 1024, 128, 512, "LL128_256KB_128b"}, // 1024w, ~2 pkts/warp
      {512 * 1024, 256, 512, "LL128_512KB_256b"}, // 2048w, ~2 pkts/warp
      {1024 * 1024, 256, 512, "LL128_1MB_256b"}, // 2048w, ~4 pkts/warp
      {1024 * 1024, 512, 512, "LL128_1MB_512b"}, // 4096w, ~2 pkts/warp
      // Round 5 Priority 2: Validate auto-tuning formula at small sizes
      {8 * 1024, 4, 512, "LL128_8KB_4b"}, // formula predicts ~4 blocks optimal
      {16 * 1024,
       8,
       256,
       "LL128_16KB_8b"}, // formula predicts ~8 blocks optimal
      // Round 5 Priority 3: 2MB higher blocks + 4MB exploration
      {2 * 1024 * 1024, 256, 512, "LL128_2MB_256b"}, // ~8 pkts/warp
      {2 * 1024 * 1024, 512, 512, "LL128_2MB_512b"}, // ~4 pkts/warp
      {4 * 1024 * 1024, 256, 512, "LL128_4MB_256b"}, // verify ll128_buffer_size
      {4 * 1024 * 1024, 512, 512, "LL128_4MB_512b"}, // verify ll128_buffer_size
      // Round 7: High-block-count unidirectional for large messages
      {2 * 1024 * 1024, 1024, 512, "LL128_2MB_1024b"},
      {4 * 1024 * 1024, 1024, 512, "LL128_4MB_1024b"},
      // Round 8: Validate 4KB auto-tune + extend sweep to 8MB/16MB
      {4 * 1024, 2, 512, "LL128_4KB_2b"},
      {8 * 1024 * 1024, 512, 512, "LL128_8MB_512b"},
      {8 * 1024 * 1024, 1024, 512, "LL128_8MB_1024b"},
      {16 * 1024 * 1024, 512, 512, "LL128_16MB_512b"},
      {16 * 1024 * 1024, 1024, 512, "LL128_16MB_1024b"},
  };

  run_unidirectional_sweep(
      peerRank,
      configs,
      "NCCL vs Simple vs LL128 UNIDIRECTIONAL Benchmark Results");
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
      // 512-thread variants for small messages (validate Recommendation 2)
      {64, 1, 512, "Bidir_64B_512t"},
      {128, 1, 512, "Bidir_128B_512t"},
      {256, 1, 512, "Bidir_256B_512t"},
      {512, 1, 512, "Bidir_512B_512t"},
      {1024, 1, 512, "Bidir_1KB_512t"},
      {2 * 1024, 1, 512, "Bidir_2KB_512t"},
      {4 * 1024, 1, 512, "Bidir_4KB_512t"},
      // Crossover region
      {5 * 1024, 1, 128, "Bidir_5KB"},
      {6 * 1024, 1, 128, "Bidir_6KB"},
      {8 * 1024, 1, 128, "Bidir_8KB"},
      // Medium/large messages (512 threads default for ≥16KB)
      {32 * 1024, 2, 512, "Bidir_32KB"},
      {64 * 1024, 4, 512, "Bidir_64KB"},
      {128 * 1024, 4, 512, "Bidir_128KB"},
      {256 * 1024, 8, 512, "Bidir_256KB"},
      // More blocks (partition_interleaved halves warps per direction)
      {32 * 1024, 4, 128, "Bidir_32KB_4b"},
      {64 * 1024, 8, 128, "Bidir_64KB_8b"},
      {128 * 1024, 8, 128, "Bidir_128KB_8b"},
      {256 * 1024, 16, 128, "Bidir_256KB_16b"},
      // 512-thread bidirectional sweep (more blocks)
      {32 * 1024, 4, 512, "Bidir_32KB_512t"},
      {64 * 1024, 8, 512, "Bidir_64KB_512t"},
      {128 * 1024, 8, 512, "Bidir_128KB_512t"},
      {256 * 1024, 16, 512, "Bidir_256KB_512t"},
      // Max warp configs
      {128 * 1024, 16, 512, "Bidir_128KB_max"},
      {256 * 1024, 32, 512, "Bidir_256KB_max"},
      // Priority 1: Explore scaling ceiling
      {128 * 1024, 32, 512, "Bidir_128KB_32b"},
      {256 * 1024, 64, 512, "Bidir_256KB_64b"},
      // Priority 2: Close 4-8KB bidir crossover gap
      {4 * 1024, 2, 512, "Bidir_4KB_2b"},
      {8 * 1024, 2, 512, "Bidir_8KB_2b"},
      // Priority 3: Close 32-64KB bidir gap
      {32 * 1024, 8, 512, "Bidir_32KB_8b"},
      {64 * 1024, 16, 512, "Bidir_64KB_16b"},
      // Priority 4: Bidir 512KB and 1MB coverage
      {512 * 1024, 32, 512, "Bidir_512KB_32b"},
      {512 * 1024, 64, 512, "Bidir_512KB_64b"},
      {1024 * 1024, 64, 512, "Bidir_1MB_64b"},
      {1024 * 1024, 128, 512, "Bidir_1MB_128b"},
      // Round 4 Priority 1: Close bidir 5-6KB and 16KB gaps
      {5 * 1024, 2, 512, "Bidir_5KB_2b"},
      {6 * 1024, 2, 512, "Bidir_6KB_2b"},
      {16 * 1024, 4, 512, "Bidir_16KB_4b"},
      // Round 4 Priority 1: Complete 32-64KB bidir block-count sweep
      {32 * 1024, 16, 512, "Bidir_32KB_16b"},
      {32 * 1024, 32, 512, "Bidir_32KB_32b"},
      {64 * 1024, 32, 512, "Bidir_64KB_32b"},
      {64 * 1024, 64, 512, "Bidir_64KB_64b"},
      // Round 4 Priority 2: 128KB ceiling and larger message scaling
      {128 * 1024, 64, 512, "Bidir_128KB_64b"},
      {128 * 1024, 128, 512, "Bidir_128KB_128b"},
      {512 * 1024, 128, 512, "Bidir_512KB_128b"},
      {2 * 1024 * 1024, 128, 512, "Bidir_2MB_128b"}, // verify ll128_buffer_size
      // Round 5 Priority 1: Close large-message parallelism gap (reach ~2
      // pkts/warp)
      {256 * 1024, 128, 512, "Bidir_256KB_128b"}, // 1024w, ~2 pkts/warp
      {256 * 1024, 256, 512, "Bidir_256KB_256b"}, // 2048w, past ceiling
      {512 * 1024, 256, 512, "Bidir_512KB_256b"}, // 2048w, ~2 pkts/warp
      {1024 * 1024, 256, 512, "Bidir_1MB_256b"}, // 2048w, ~4 pkts/warp
      {1024 * 1024, 512, 512, "Bidir_1MB_512b"}, // 4096w, ~2 pkts/warp
      // Round 5 Priority 2: Validate auto-tuning formula at small sizes
      {8 * 1024, 4, 512, "Bidir_8KB_4b"}, // formula predicts ~4 blocks optimal
      {16 * 1024,
       8,
       256,
       "Bidir_16KB_8b"}, // formula predicts ~8 blocks optimal
      // Round 5 Priority 3: 2MB higher blocks + 4MB exploration
      {2 * 1024 * 1024, 256, 512, "Bidir_2MB_256b"}, // ~8 pkts/warp
      {2 * 1024 * 1024, 512, 512, "Bidir_2MB_512b"}, // ~4 pkts/warp
      {4 * 1024 * 1024, 256, 512, "Bidir_4MB_256b"}, // verify ll128_buffer_size
      {4 * 1024 * 1024, 512, 512, "Bidir_4MB_512b"}, // verify ll128_buffer_size
      // Round 5 Priority 4: Bidir ceiling confirmation
      {32 * 1024, 64, 512, "Bidir_32KB_64b"}, // 32 per dir — past uni peak
      {64 * 1024, 128, 512, "Bidir_64KB_128b"}, // 64 per dir — past uni peak
      // Round 7: High-block-count bidirectional for large messages
      {2 * 1024 * 1024, 1024, 512, "Bidir_2MB_1024b"},
      {4 * 1024 * 1024, 1024, 512, "Bidir_4MB_1024b"},
      // Round 8: Validate 1MB/1024b cascade + extend sweep to 8MB/16MB
      {1024 * 1024, 1024, 512, "Bidir_1MB_1024b"},
      {8 * 1024 * 1024, 512, 512, "Bidir_8MB_512b"},
      {8 * 1024 * 1024, 1024, 512, "Bidir_8MB_1024b"},
      {16 * 1024 * 1024, 512, 512, "Bidir_16MB_512b"},
      {16 * 1024 * 1024, 1024, 512, "Bidir_16MB_1024b"},
      // Round 9: Close bidir sweep coverage gaps
      {4 * 1024, 4, 512, "Bidir_4KB_4b"}, // validate ATBi_4KB auto-tune
      {8 * 1024, 8, 512, "Bidir_8KB_8b"}, // validate ATBi_8KB auto-tune
      {512 * 1024,
       512,
       256,
       "Bidir_512KB_512b"}, // validate ATBi_512KB (peak ratio)
  };

  run_bidirectional_sweep(
      peerRank,
      configs,
      "NCCL vs Simple vs LL128 BIDIRECTIONAL Benchmark Results");
}

TEST_F(P2pLl128BenchmarkFixture, ChunkedUnidirectionalBenchmark) {
  if (worldSize != 2) {
    XLOGF(DBG1, "Skipping test: requires exactly 2 ranks, got {}", worldSize);
    return;
  }
  int peerRank = (globalRank == 0) ? 1 : 0;

  std::vector<Ll128Config> configs = {
      // --- Multiple-of-4 buffer packets (sub-groups naturally aligned) ---
      {32 * 1024, 1, 512, "Chunk_32KB_64pkt", 64},
      {64 * 1024, 2, 512, "Chunk_64KB_128pkt", 128},
      {128 * 1024, 4, 512, "Chunk_128KB_256pkt", 256},
      {256 * 1024, 8, 512, "Chunk_256KB_256pkt", 256},

      // --- Non-multiple-of-4 buffer packets (exercises __syncwarp fix) ---
      {32 * 1024, 1, 512, "Chunk_32KB_65pkt", 65},
      {64 * 1024, 2, 512, "Chunk_64KB_129pkt", 129},
      {128 * 1024, 4, 512, "Chunk_128KB_257pkt", 257},
      {256 * 1024, 8, 512, "Chunk_256KB_257pkt", 257},

      // --- Small buffer (high round count, stress-tests wrapping) ---
      {32 * 1024, 1, 512, "Chunk_32KB_16pkt", 16},
      {32 * 1024, 1, 512, "Chunk_32KB_17pkt", 17},

      // --- Aggressive non-multiple-of-4 (mirrors unit test bug case) ---
      {32 * 1024, 1, 512, "Chunk_32KB_5pkt", 5},
      {32 * 1024, 1, 512, "Chunk_32KB_7pkt", 7},
  };

  run_chunked_ll128_sweep(
      peerRank, configs, "LL128 CHUNKED UNIDIRECTIONAL Benchmark Results");
}

TEST_F(P2pLl128BenchmarkFixture, ChunkedBidirectionalBenchmark) {
  if (worldSize != 2) {
    XLOGF(DBG1, "Skipping test: requires exactly 2 ranks, got {}", worldSize);
    return;
  }
  int peerRank = (globalRank == 0) ? 1 : 0;

  std::vector<Ll128Config> configs = {
      // --- Multiple-of-4 buffer packets ---
      {32 * 1024, 2, 512, "ChBi_32KB_64pkt", 64},
      {64 * 1024, 4, 512, "ChBi_64KB_128pkt", 128},
      {128 * 1024, 8, 512, "ChBi_128KB_256pkt", 256},
      {256 * 1024, 16, 512, "ChBi_256KB_256pkt", 256},

      // --- Non-multiple-of-4 buffer packets ---
      {32 * 1024, 2, 512, "ChBi_32KB_65pkt", 65},
      {64 * 1024, 4, 512, "ChBi_64KB_129pkt", 129},
      {128 * 1024, 8, 512, "ChBi_128KB_257pkt", 257},
      {256 * 1024, 16, 512, "ChBi_256KB_257pkt", 257},

      // --- Small buffer ---
      {32 * 1024, 2, 512, "ChBi_32KB_16pkt", 16},
      {32 * 1024, 2, 512, "ChBi_32KB_17pkt", 17},

      // --- Aggressive non-multiple-of-4 ---
      {32 * 1024, 2, 512, "ChBi_32KB_5pkt", 5},
      {32 * 1024, 2, 512, "ChBi_32KB_7pkt", 7},
  };

  run_chunked_ll128_bidir_sweep(
      peerRank, configs, "LL128 CHUNKED BIDIRECTIONAL Benchmark Results");
}

TEST_F(P2pLl128BenchmarkFixture, AutoTunedBenchmark) {
  if (worldSize != 2) {
    XLOGF(DBG1, "Skipping test: requires exactly 2 ranks, got {}", worldSize);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;

  std::vector<Ll128Config> configs;
  for (auto nBytes : kAutoTuneMessageSizes) {
    auto cfg = comms::pipes::ll128_auto_tune(nBytes);
    configs.push_back(
        {nBytes, cfg.numBlocks, cfg.numThreads, "AT_" + formatSize(nBytes)});
  }

  run_unidirectional_sweep(
      peerRank, configs, "LL128 AUTO-TUNED Configuration Benchmark Results");
}

TEST_F(P2pLl128BenchmarkFixture, AutoTunedBidirectionalBenchmark) {
  if (worldSize != 2) {
    XLOGF(DBG1, "Skipping test: requires exactly 2 ranks, got {}", worldSize);
    return;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;

  std::vector<Ll128Config> configs;
  for (auto nBytes : kAutoTuneMessageSizes) {
    auto cfg = comms::pipes::ll128_auto_tune_bidirectional(nBytes);
    configs.push_back(
        {nBytes, cfg.numBlocks, cfg.numThreads, "ATBi_" + formatSize(nBytes)});
  }

  run_bidirectional_sweep(
      peerRank,
      configs,
      "LL128 AUTO-TUNED BIDIRECTIONAL Configuration Benchmark Results");
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
