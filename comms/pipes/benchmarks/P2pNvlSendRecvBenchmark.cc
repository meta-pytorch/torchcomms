// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>
#include <nccl.h>

#include "comms/common/CudaWrap.h"
#include "comms/pipes/MultiPeerNvlTransport.h"
#include "comms/pipes/TiledBuffer.cuh"
#include "comms/pipes/benchmarks/BenchmarkKernel.cuh"
#include "comms/pipes/benchmarks/BenchmarkMacros.h"
#include "comms/pipes/benchmarks/P2pNvlBenchmarkUtils.h"
#include "comms/pipes/benchmarks/TileSendRecv.cuh"
#include "comms/testinfra/BenchmarkTestFixture.h"
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

  // Verify correctness of a P2P transfer by filling src with a pattern,
  // running one send/recv, and checking the received data matches.
  bool verifyP2pCorrectness(
      comms::pipes::P2pNvlTransportDevice* p2pDevicePtr,
      const BenchmarkConfig& config) {
    DeviceBuffer sendBuff(config.nBytes);
    DeviceBuffer recvBuff(config.nBytes);

    bool isSend = (globalRank == 0);
    const int testValue = 0xAB;

    if (isSend) {
      CUDA_CHECK_BOOL(cudaMemset(sendBuff.get(), testValue, config.nBytes));
    } else {
      CUDA_CHECK_BOOL(cudaMemset(recvBuff.get(), 0, config.nBytes));
    }

    dim3 gridDim(config.numBlocks);
    dim3 blockDim(config.numThreads);
    SyncScope groupScope = config.groupScope;
    void* devicePtr = isSend ? sendBuff.get() : recvBuff.get();
    std::size_t nBytes = config.nBytes;
    Timeout timeout;
    void* args[] = {p2pDevicePtr, &devicePtr, &nBytes, &groupScope, &timeout};

    void* kernelFunc = isSend ? (void*)comms::pipes::benchmark::p2pSend
                              : (void*)comms::pipes::benchmark::p2pRecv;

    bootstrap->barrierAll();
    CUDA_CHECK_BOOL(
        cudaLaunchKernel(kernelFunc, gridDim, blockDim, args, 0, nullptr));
    CUDA_CHECK_BOOL(cudaDeviceSynchronize());
    bootstrap->barrierAll();

    if (!isSend) {
      // Verify received data
      std::vector<char> hostBuf(config.nBytes);
      CUDA_CHECK_BOOL(cudaMemcpy(
          hostBuf.data(),
          recvBuff.get(),
          config.nBytes,
          cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < config.nBytes; i++) {
        if (static_cast<unsigned char>(hostBuf[i]) != testValue) {
          XLOGF(
              ERR,
              "VERIFY FAILED {}: byte {} expected 0x{:02X} got 0x{:02X}",
              config.name,
              i,
              testValue,
              static_cast<unsigned int>(
                  static_cast<unsigned char>(hostBuf[i])));
          return false;
        }
      }
    }
    return true;
  }

  // Helper function to run P2P NVL benchmark - returns bandwidth
  // p2pDevicePtr must point to a P2pNvlTransportDevice in host memory
  // (e.g. obtained from buildP2pTransportDevice()).
  float runP2pNvlBenchmark(
      comms::pipes::P2pNvlTransportDevice* p2pDevicePtr,
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
    // p2pDevicePtr points to a host-side P2pNvlTransportDevice;
    // cudaLaunchKernel reads the struct by value from host memory for the
    // kernel parameter.
    void* args[] = {p2pDevicePtr, &devicePtr, &nBytes, &groupScope, &timeout};

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

  // Tile sendrecv — Triton-style head/tail counter protocol.
  // Uses 16MB slots with per-slot signaling (not per-chunk).
  float runTileBenchmark(
      comms::pipes::P2pNvlTransportDevice* p2pDevicePtr,
      const BenchmarkConfig& config,
      float& timeUs) {
    XLOGF(DBG1, "=== Running Tile benchmark: {} ===", config.name);

    DeviceBuffer sendBuff(config.nBytes);
    DeviceBuffer recvBuff(config.nBytes);

    // Fill send with rank-specific pattern, clear recv
    const int sendPattern = 0xA0 + globalRank;
    const int peerPattern = 0xA0 + (1 - globalRank);
    CUDA_CHECK(cudaMemset(sendBuff.get(), sendPattern, config.nBytes));
    CUDA_CHECK(cudaMemset(recvBuff.get(), 0, config.nBytes));

    int numSendBlocks = config.numBlocks;
    int totalBlocks = numSendBlocks * 2;
    dim3 gridDim(totalBlocks);
    dim3 blockDim(config.numThreads);

    CudaEvent start, stop;
    Timeout timeout;

    // Create TiledBuffer views for send and recv
    comms::pipes::TiledBuffer<char> sendTiles(
        static_cast<char*>(sendBuff.get()), config.nBytes, numSendBlocks);
    comms::pipes::TiledBuffer<char> recvTiles(
        static_cast<char*>(recvBuff.get()), config.nBytes, numSendBlocks);

    int chunksPerSlot = config.chunksPerSlot;
    void* kernelFunc = (void*)comms::pipes::benchmark::p2pTileSendRecv;
    void* args[] = {
        p2pDevicePtr,
        &sendTiles,
        &recvTiles,
        &numSendBlocks,
        &chunksPerSlot,
        &timeout};

    dim3 defaultClusterDim(comms::common::kDefaultClusterSize, 1, 1);
    std::optional<dim3> clusterDimOpt = config.spreadClusterLaunch
        ? std::optional{defaultClusterDim}
        : std::nullopt;

    // Correctness verification: run one exchange and check received data
    bootstrap->barrierAll();
    CUDA_CHECK(
        comms::common::launchKernel(
            kernelFunc, gridDim, blockDim, args, nullptr, clusterDimOpt));
    CUDA_CHECK(cudaDeviceSynchronize());
    bootstrap->barrierAll();

    // Verify recv buffer has peer's pattern
    {
      std::vector<char> hostBuf(config.nBytes);
      CUDA_CHECK(cudaMemcpy(
          hostBuf.data(),
          recvBuff.get(),
          config.nBytes,
          cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < config.nBytes; i++) {
        if (static_cast<unsigned char>(hostBuf[i]) !=
            static_cast<unsigned char>(peerPattern)) {
          XLOGF(
              ERR,
              "TILE VERIFY FAILED {}: byte {} expected 0x{:02X} got 0x{:02X}",
              config.name,
              i,
              peerPattern,
              static_cast<unsigned int>(
                  static_cast<unsigned char>(hostBuf[i])));
          bootstrap->barrierAll();
          timeUs = 0;
          return 0;
        }
      }
    }

    // Re-init buffers for benchmark
    CUDA_CHECK(cudaMemset(sendBuff.get(), sendPattern, config.nBytes));
    CUDA_CHECK(cudaMemset(recvBuff.get(), 0, config.nBytes));

    // Warmup
    bootstrap->barrierAll();
    for (int i = 0; i < kWarmupIters; i++) {
      CUDA_CHECK(
          comms::common::launchKernel(
              kernelFunc, gridDim, blockDim, args, nullptr, clusterDimOpt));
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    bootstrap->barrierAll();

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
    timeUs = avgTime_ms * 1000.0f;
    float bandwidth_GBps =
        (2.0f * config.nBytes / 1e9f) / (avgTime_ms / 1000.0f);

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
  // BW. p2pDevicePtr must point to a P2pNvlTransportDevice in host memory
  // (e.g. obtained from buildP2pTransportDevice()).
  float runP2pNvlBidirectionalBenchmark(
      comms::pipes::P2pNvlTransportDevice* p2pDevicePtr,
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
    // p2pDevicePtr points to a host-side P2pNvlTransportDevice;
    // cudaLaunchKernel reads the struct by value from host memory for the
    // kernel parameter.
    void* args[] = {
        p2pDevicePtr, &sendPtr, &recvPtr, &nBytes, &groupScope, &timeout};

    void* kernelFunc = (void*)comms::pipes::benchmark::p2pBidirectional;

    // Warmup - no reset needed, recv() signals -1 after each transfer
    bootstrap->barrierAll();

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

  // NCCL-like test configurations
  // NCCL uses 16 blocks for < 512M messages, 32 blocks for >= 512M messages
  std::vector<BenchmarkConfig> configs;

  constexpr int kNcclBlocksSmall = 16; // For messages < 512MB
  constexpr int kNcclBlocksLarge = 32; // For messages >= 512MB
  constexpr int kNcclThreads = 512;
  constexpr std::size_t kNcclStagedBufferSize = 8 * 1024 * 1024; // 8MB
  constexpr std::size_t kChunkSize = 512 * 1024; // 512KB
  constexpr std::size_t kLargeMessageThreshold = 512 * 1024 * 1024; // 512MB

  // Helper function for adding NCCL-like config with auto-computed numBlocks
  // Adds both single state and dual state variants
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
    // Single state buffer variant
    configs.push_back({
        .nBytes = sizeBytes,
        .stagedBufferSize = kNcclStagedBufferSize,
        .numBlocks = numBlks,
        .numThreads = kNcclThreads,
        .pipelineDepth = 2,
        .chunkSize = kChunkSize,
        .groupScope = scope,
        .spreadClusterLaunch = true,
        .useDualStateBuffer = false,
        .name = "NCCL_" + sizeName + "_" + scopeName + "_Single",
    });
    // Dual state buffer variant
    configs.push_back({
        .nBytes = sizeBytes,
        .stagedBufferSize = kNcclStagedBufferSize,
        .numBlocks = numBlks,
        .numThreads = kNcclThreads,
        .pipelineDepth = 2,
        .chunkSize = kChunkSize,
        .groupScope = scope,
        .spreadClusterLaunch = true,
        .useDualStateBuffer = true,
        .name = "NCCL_" + sizeName + "_" + scopeName + "_Dual",
    });
  };

  // === BLOCK-BASED CONFIGURATIONS ===
  addNcclConfig(4 * 1024, "4K", SyncScope::BLOCK, "Block");
  addNcclConfig(8 * 1024, "8K", SyncScope::BLOCK, "Block");
  addNcclConfig(16 * 1024, "16K", SyncScope::BLOCK, "Block");
  addNcclConfig(64 * 1024, "64K", SyncScope::BLOCK, "Block");
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
  addNcclConfig(4 * 1024, "4K", SyncScope::CLUSTER, "Cluster");
  addNcclConfig(8 * 1024, "8K", SyncScope::CLUSTER, "Cluster");
  addNcclConfig(16 * 1024, "16K", SyncScope::CLUSTER, "Cluster");
  addNcclConfig(64 * 1024, "64K", SyncScope::CLUSTER, "Cluster");
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
        .useDualStateBuffer = config.useDualStateBuffer,
    };

    comms::pipes::MultiPeerNvlTransport transport(
        globalRank, worldSize, bootstrap, p2pConfig);
    transport.exchange();

    // Build host-side P2pNvlTransportDevice (passed by value to kernel)
    auto p2pHost = transport.buildP2pTransportDevice(peerRank);

    BenchmarkResult result;
    result.testName = config.name;
    result.messageSize = config.nBytes;
    result.stagingBufferSize = config.stagedBufferSize;
    result.pipelineDepth = config.pipelineDepth;
    result.chunkSize = config.chunkSize;
    result.numBlocks = config.numBlocks;
    result.numThreads = config.numThreads;

    // Verify correctness before benchmarking
    if (!verifyP2pCorrectness(&p2pHost, config)) {
      XLOGF(ERR, "CORRECTNESS CHECK FAILED for config: {}", config.name);
      if (globalRank == 0) {
        std::cout << "*** VERIFY FAILED: " << config.name << " ***\n";
      }
      continue;
    }

    // Run NCCL benchmark
    result.ncclBandwidth = runNcclBenchmark(config, result.ncclTime);

    // Run P2P NVL benchmark
    result.p2pBandwidth = runP2pNvlBenchmark(&p2pHost, config, result.p2pTime);

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
  std::vector<BenchmarkConfig> configs;

  // NCCL-like bidirectional configs at all message sizes
  std::vector<std::pair<std::size_t, std::string>> bidiSizes = {
      {8 * 1024, "8K"},
      {16 * 1024, "16K"},
      {64 * 1024, "64K"},
      {128 * 1024, "128K"},
      {256 * 1024, "256K"},
      {512 * 1024, "512K"},
      {1024 * 1024, "1M"},
      {2 * 1024 * 1024, "2M"},
      {4 * 1024 * 1024, "4M"},
      {8 * 1024 * 1024, "8M"},
      {16 * 1024 * 1024, "16M"},
      {32 * 1024 * 1024, "32M"},
      {64 * 1024 * 1024, "64M"},
      {128 * 1024 * 1024, "128M"},
      {256 * 1024 * 1024, "256M"},
      {512 * 1024 * 1024, "512M"},
      {1024 * 1024 * 1024, "1G"},
  };
  for (const auto& [sz, nm] : bidiSizes) {
    configs.push_back({
        .nBytes = sz,
        .stagedBufferSize = 8 * 1024 * 1024,
        .numBlocks = 32,
        .numThreads = 512,
        .pipelineDepth = 2,
        .chunkSize = 512 * 1024,
        .groupScope = SyncScope::BLOCK,
        .name = "Bidir_" + nm,
    });
  }

  // === TILE CONFIGS (Triton-style: 8MB slots, head/tail counters) ===
  auto addTileConfig = [&](std::size_t sizeBytes, const std::string& name) {
    constexpr std::size_t kSlot = 8 * 1024 * 1024; // 8MB per slot
    configs.push_back({
        .nBytes = sizeBytes,
        .stagedBufferSize = kSlot,
        .numBlocks = 16,
        .numThreads = 512,
        .pipelineDepth = 2, // 2 slots = 16MB total
        .chunkSize = kSlot,
        .groupScope = SyncScope::BLOCK,
        .useTiled = true,
        .name = name,
    });
  };

  std::vector<std::pair<std::size_t, std::string>> tileSizes = {
      {8 * 1024, "8K"},
      {16 * 1024, "16K"},
      {64 * 1024, "64K"},
      {128 * 1024, "128K"},
      {256 * 1024, "256K"},
      {512 * 1024, "512K"},
      {1 * 1024 * 1024, "1M"},
      {2 * 1024 * 1024, "2M"},
      {4 * 1024 * 1024, "4M"},
      {8 * 1024 * 1024, "8M"},
      {16 * 1024 * 1024, "16M"},
      {32 * 1024 * 1024, "32M"},
      {64 * 1024 * 1024, "64M"},
      {128 * 1024 * 1024, "128M"},
      {256 * 1024 * 1024, "256M"},
      {512 * 1024 * 1024, "512M"},
      {1024 * 1024 * 1024, "1G"},
  };
  for (const auto& [sz, nm] : tileSizes) {
    addTileConfig(sz, "Tile_" + nm);
  }

  // === CLUSTERED TILE CONFIGS ===
  for (const auto& [sz, nm] : tileSizes) {
    constexpr std::size_t kSlot = 8 * 1024 * 1024;
    configs.push_back({
        .nBytes = sz,
        .stagedBufferSize = kSlot,
        .numBlocks = 16,
        .numThreads = 512,
        .pipelineDepth = 2,
        .chunkSize = kSlot,
        .groupScope = SyncScope::BLOCK,
        .spreadClusterLaunch = true,
        .useTiled = true,
        .name = "TileClus_" + nm,
    });
  }

  // === CHUNKS PER SLOT SWEEP (clustered, 8MB staging, pd=2) ===
  for (int cps : {2, 4, 8}) {
    for (const auto& [sz, nm] : tileSizes) {
      if (sz < 1 * 1024 * 1024) {
        continue; // only test >= 1MB
      }
      constexpr std::size_t kSlot2 = 8 * 1024 * 1024;
      configs.push_back({
          .nBytes = sz,
          .stagedBufferSize = kSlot2,
          .numBlocks = 16,
          .numThreads = 512,
          .pipelineDepth = 2,
          .chunkSize = kSlot2,
          .groupScope = SyncScope::BLOCK,
          .spreadClusterLaunch = true,
          .useTiled = true,
          .chunksPerSlot = cps,
          .name = "CPS" + std::to_string(cps) + "_" + nm,
      });
    }
  }

  // Sweep staging buffer configurations with 128KB signal granularity (best
  // from signal sweep) Baseline: 8MB slot × pd=2 = 16MB total (already covered
  // by Tile_*) Variant A: 16MB slot × pd=2 = 32MB total (larger slots, fewer
  // round-trips) Variant B: 8MB slot × pd=4 = 32MB total (deeper pipeline, same
  // slot size)
  struct StagingConfig {
    std::size_t slotSize;
    std::size_t pipelineDepth;
    std::string prefix;
  };
  std::vector<StagingConfig> stagingConfigs = {
      {16 * 1024 * 1024, 2, "Stg32M_"}, // 16MB×2 = 32MB total
      {8 * 1024 * 1024, 4, "Pd4_"}, // 8MB×4 = 32MB total
  };

  for (const auto& sc : stagingConfigs) {
    for (const auto& [sz, nm] : tileSizes) {
      if (sz < 4 * 1024 * 1024) {
        continue; // only test >= 4MB
      }

      configs.push_back({
          .nBytes = sz,
          .stagedBufferSize = sc.slotSize,
          .numBlocks = 16,
          .numThreads = 512,
          .pipelineDepth = sc.pipelineDepth,
          .chunkSize = 128 * 1024, // best signal granularity
          .groupScope = SyncScope::BLOCK,
          .useTiled = true,
          .name = sc.prefix + nm,
      });
    }
  }

  // Also test best signal granularity (128KB) with baseline staging for
  // comparison
  for (const auto& [sz, nm] : tileSizes) {
    if (sz < 4 * 1024 * 1024) {
      continue;
    }
    configs.push_back({
        .nBytes = sz,
        .stagedBufferSize = 8 * 1024 * 1024,
        .numBlocks = 16,
        .numThreads = 512,
        .pipelineDepth = 2,
        .chunkSize = 128 * 1024,
        .groupScope = SyncScope::BLOCK,
        .useTiled = true,
        .name = "Sig128K_" + nm,
    });
  }

  std::vector<BenchmarkResult> results;

  for (const auto& config : configs) {
    // Create P2P transport for this configuration
    comms::pipes::MultiPeerNvlTransportConfig p2pConfig{
        .dataBufferSize = config.stagedBufferSize,
        .chunkSize = config.chunkSize,
        .pipelineDepth = config.pipelineDepth,
        .useDualStateBuffer = config.useDualStateBuffer,
    };

    comms::pipes::MultiPeerNvlTransport transport(
        globalRank, worldSize, bootstrap, p2pConfig);
    transport.exchange();

    auto p2pHost = transport.buildP2pTransportDevice(peerRank);

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

    // Run P2P benchmark
    if (config.useTiled) {
      result.p2pBandwidth = runTileBenchmark(&p2pHost, config, result.p2pTime);
    } else {
      result.p2pBandwidth =
          runP2pNvlBidirectionalBenchmark(&p2pHost, config, result.p2pTime);
    }

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
