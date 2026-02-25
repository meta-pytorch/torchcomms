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

class P2pAsymmetricChunkBenchmarkFixture
    : public meta::comms::BenchmarkTestFixture {
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

  // NCCL unidirectional benchmark
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

  // NCCL bidirectional benchmark
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

    CUDA_CHECK(cudaMemset(sendBuff.get(), globalRank, config.nBytes));
    CUDA_CHECK(cudaMemset(recvBuff.get(), 0, config.nBytes));

    int peerRank = (globalRank == 0) ? 1 : 0;
    CudaEvent start, stop;

    // Warmup
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

    // Benchmark
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
    float bandwidth_GBps =
        (2.0f * config.nBytes / 1e9f) / (avgTime_ms / 1000.0f);

    bootstrap->barrierAll();
    return bandwidth_GBps;
  }

  /**
   * Unidirectional P2P benchmark helper.
   *
   * Runs warmup iterations (with inner barrier) followed by timed benchmark
   * iterations. The caller is responsible for allocating buffers and the
   * stream, and for constructing the correct kernelFunc and args array.
   *
   * @param config Benchmark configuration (used for grid/block dims, nBytes).
   * @param kernelFunc Pointer to the __global__ kernel to launch.
   * @param args Kernel argument array (pointers to caller-owned locals).
   * @param stream CUDA stream to launch on.
   * @param timeUs Output: average iteration time in microseconds.
   * @return Unidirectional bandwidth in GB/s.
   */
  float runUnidirectionalP2pBenchmark(
      const BenchmarkConfig& config,
      void* kernelFunc,
      void** args,
      cudaStream_t stream,
      float& timeUs) {
    XLOGF(DBG1, "=== Running P2P benchmark: {} ===", config.name);

    dim3 gridDim(config.numBlocks);
    dim3 blockDim(config.numThreads);

    CudaEvent start, stop;

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

    bootstrap->barrierAll();
    return bandwidth_GBps;
  }

  /**
   * Bidirectional P2P benchmark helper.
   *
   * Runs warmup iterations followed by timed benchmark iterations using
   * comms::common::launchKernel for optional cluster launch support.
   * The caller is responsible for allocating buffers and constructing the
   * correct kernelFunc and args array.
   *
   * @param config Benchmark configuration (used for grid/block dims, nBytes,
   *               spreadClusterLaunch).
   * @param kernelFunc Pointer to the __global__ kernel to launch.
   * @param args Kernel argument array (pointers to caller-owned locals).
   * @param timeUs Output: average iteration time in microseconds.
   * @return Bidirectional bandwidth in GB/s (2x data volume).
   */
  float runBidirectionalP2pBenchmark(
      const BenchmarkConfig& config,
      void* kernelFunc,
      void** args,
      float& timeUs) {
    XLOGF(
        DBG1,
        "Rank {}: Starting P2P bidirectional benchmark: {}",
        globalRank,
        config.name);

    dim3 gridDim(config.numBlocks);
    dim3 blockDim(config.numThreads);

    CudaEvent start, stop;

    dim3 defaultClusterDim(comms::common::kDefaultClusterSize, 1, 1);
    std::optional<dim3> clusterDimOpt = config.spreadClusterLaunch
        ? std::optional{defaultClusterDim}
        : std::nullopt;

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

  // Correctness check for asymmetric configs
  void runP2pCorrectnessCheck(
      comms::pipes::P2pNvlTransportDevice& p2p,
      const BenchmarkConfig& config) {
    XLOGF(DBG1, "=== Running correctness check: {} ===", config.name);

    DeviceBuffer sendBuff(config.nBytes);
    DeviceBuffer recvBuff(config.nBytes);

    // Sender fills with known pattern, receiver zeros
    if (globalRank == 0) {
      CUDA_CHECK_VOID(cudaMemset(sendBuff.get(), 0xAB, config.nBytes));
    } else {
      CUDA_CHECK_VOID(cudaMemset(recvBuff.get(), 0, config.nBytes));
    }

    dim3 gridDim(config.numBlocks);
    dim3 blockDim(config.numThreads);

    cudaStream_t sendStream, recvStream;
    CUDA_CHECK_VOID(cudaStreamCreate(&sendStream));
    CUDA_CHECK_VOID(cudaStreamCreate(&recvStream));

    std::size_t nBytes = config.nBytes;
    SyncScope groupScope = config.groupScope;
    Timeout timeout;
    bool isSend = (globalRank == 0);

    void* devicePtr = isSend ? sendBuff.get() : recvBuff.get();

    // Determine which kernel to use based on config
    void* kernelFunc = nullptr;
    void* symSendArgs[] = {&p2p, &devicePtr, &nBytes, &groupScope, &timeout};
    std::size_t sendChunkSize = config.sendChunkSize.value_or(0);
    void* asymSendArgs[] = {
        &p2p, &devicePtr, &nBytes, &sendChunkSize, &groupScope, &timeout};
    std::size_t recvChunkSize = config.recvChunkSize.value_or(0);
    void* asymRecvArgs[] = {
        &p2p, &devicePtr, &nBytes, &recvChunkSize, &groupScope, &timeout};

    void** args = nullptr;
    if (isSend) {
      if (config.sendChunkSize.has_value()) {
        kernelFunc = (void*)comms::pipes::benchmark::p2pAsymmetricSend;
        args = asymSendArgs;
      } else {
        kernelFunc = (void*)comms::pipes::benchmark::p2pSend;
        args = symSendArgs;
      }
    } else {
      if (config.recvChunkSize.has_value()) {
        kernelFunc = (void*)comms::pipes::benchmark::p2pAsymmetricRecv;
        args = asymRecvArgs;
      } else {
        kernelFunc = (void*)comms::pipes::benchmark::p2pRecv;
        args = symSendArgs; // same layout: (p2p, buf, nBytes, scope, timeout)
      }
    }
    cudaStream_t stream = isSend ? sendStream : recvStream;

    // Run one transfer
    bootstrap->barrierAll();
    CUDA_CHECK_VOID(
        cudaLaunchKernel(kernelFunc, gridDim, blockDim, args, 0, stream));
    CUDA_CHECK_VOID(cudaStreamSynchronize(stream));
    bootstrap->barrierAll();

    // Receiver verifies data
    if (globalRank == 1) {
      std::size_t checkSize = std::min(config.nBytes, (std::size_t)4096);
      std::vector<char> hostBuf(checkSize);
      CUDA_CHECK_VOID(cudaMemcpy(
          hostBuf.data(), recvBuff.get(), checkSize, cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < checkSize; ++i) {
        EXPECT_EQ(static_cast<unsigned char>(hostBuf[i]), 0xAB)
            << "Data mismatch at byte " << i;
      }
      XLOG(INFO) << "Correctness check passed for " << config.name << " ("
                 << checkSize << " bytes verified)";
    }

    CUDA_CHECK_VOID(cudaStreamDestroy(sendStream));
    CUDA_CHECK_VOID(cudaStreamDestroy(recvStream));
    bootstrap->barrierAll();
  }

  void printResultsTable(
      const std::vector<BenchmarkResult>& results,
      const std::string& title) {
    if (globalRank != 0) {
      return;
    }

    std::stringstream ss;
    ss << "\n";
    ss << "==========================================================================================================================================================\n";
    ss << "                              " << title << "\n";
    ss << "==========================================================================================================================================================\n";
    ss << std::left << std::setw(28) << "Test Name" << std::right
       << std::setw(10) << "Msg Size" << std::right << std::setw(12)
       << "Staging" << std::right << std::setw(5) << "PD" << std::right
       << std::setw(10) << "Snd Chunk" << std::right << std::setw(10)
       << "Rcv Chunk" << std::right << std::setw(7) << "Blocks" << std::right
       << std::setw(8) << "Threads" << std::right << std::setw(11) << "NCCL BW"
       << std::right << std::setw(11) << "P2P BW" << std::right << std::setw(9)
       << "Speedup" << std::right << std::setw(11) << "NCCL Lat" << std::right
       << std::setw(11) << "P2P Lat" << std::right << std::setw(11)
       << "Lat Reduc\n";
    ss << std::left << std::setw(28) << "" << std::right << std::setw(10) << ""
       << std::right << std::setw(12) << "" << std::right << std::setw(5) << ""
       << std::right << std::setw(10) << "" << std::right << std::setw(10) << ""
       << std::right << std::setw(7) << "" << std::right << std::setw(8) << ""
       << std::right << std::setw(11) << "(GB/s)" << std::right << std::setw(11)
       << "(GB/s)" << std::right << std::setw(9) << "P2P/NCCL" << std::right
       << std::setw(11) << "(us)" << std::right << std::setw(11) << "(us)"
       << std::right << std::setw(11) << "(us)\n";
    ss << "----------------------------------------------------------------------------------------------------------------------------------------------------------\n";

    for (const auto& r : results) {
      std::string msgSize = formatSize(r.messageSize);
      std::string stagingSize = formatSize(r.stagingBufferSize);
      std::string sendChunkStr = r.sendChunkSize.has_value()
          ? formatSize(r.sendChunkSize.value())
          : formatSize(r.chunkSize);
      std::string recvChunkStr = r.recvChunkSize.has_value()
          ? formatSize(r.recvChunkSize.value())
          : formatSize(r.chunkSize);
      float latencyReduction = r.ncclTime - r.p2pTime;

      ss << std::left << std::setw(28) << r.testName << std::right
         << std::setw(10) << msgSize << std::right << std::setw(12)
         << stagingSize << std::right << std::setw(5) << r.pipelineDepth
         << std::right << std::setw(10) << sendChunkStr << std::right
         << std::setw(10) << recvChunkStr << std::right << std::setw(7)
         << r.numBlocks << std::right << std::setw(8) << r.numThreads
         << std::right << std::setw(11) << std::fixed << std::setprecision(2)
         << r.ncclBandwidth << std::right << std::setw(11) << std::fixed
         << std::setprecision(2) << r.p2pBandwidth << std::right << std::setw(8)
         << std::fixed << std::setprecision(2) << r.p2pSpeedup << "x"
         << std::right << std::setw(11) << std::fixed << std::setprecision(1)
         << r.ncclTime << std::right << std::setw(11) << std::fixed
         << std::setprecision(1) << r.p2pTime << std::right << std::setw(11)
         << std::fixed << std::setprecision(1) << latencyReduction << "\n";
    }
    ss << "==========================================================================================================================================================\n";
    ss << "PD = Pipeline Depth, Snd Chunk = Send Chunk Size, Rcv Chunk = Recv Chunk Size\n";
    ss << "BW (Bandwidth) = Data transferred / time, in GB/s\n";
    ss << "Lat (Latency) = Average transfer time per iteration, in microseconds\n";
    ss << "Lat Reduc = NCCL latency - P2P latency (positive = P2P faster)\n";
    ss << "Speedup = P2P Bandwidth / NCCL Bandwidth\n";
    ss << "==========================================================================================================================================================\n\n";

    std::cout << ss.str();
  }

  ncclComm_t ncclComm_{};
  cudaStream_t stream_{};
};

// =============================================================================
// Unidirectional Asymmetric Chunk Benchmark
// =============================================================================
TEST_F(P2pAsymmetricChunkBenchmarkFixture, UnidirectionalAsymmetricChunk) {
  if (worldSize != 2) {
    GTEST_SKIP() << "Test requires exactly 2 ranks, got " << worldSize;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;

  // Message sizes to test
  const std::vector<std::pair<std::size_t, std::string>> messageSizes = {
      {64 * 1024 * 1024, "64MB"},
      {128 * 1024 * 1024, "128MB"},
      {256 * 1024 * 1024, "256MB"},
      {512 * 1024 * 1024, "512MB"},
      {1024 * 1024 * 1024, "1GB"},
  };

  constexpr std::size_t kStagedBufferSize = 16 * 1024 * 1024; // 16MB
  constexpr std::size_t kPipelineDepth = 2;
  constexpr int kNumBlocks = 32;
  constexpr int kNumThreads = 512;
  constexpr std::size_t kSmallChunk = 32 * 1024; // 32KB
  constexpr std::size_t kLargeChunk = 512 * 1024; // 512KB

  std::vector<BenchmarkResult> results;

  for (const auto& [msgBytes, sizeName] : messageSizes) {
    // Config 1: Symmetric 32KB
    BenchmarkConfig sym32k{
        .nBytes = msgBytes,
        .stagedBufferSize = kStagedBufferSize,
        .numBlocks = kNumBlocks,
        .numThreads = kNumThreads,
        .pipelineDepth = kPipelineDepth,
        .chunkSize = kSmallChunk,
        .groupScope = SyncScope::BLOCK,
        .name = "Sym_32K_" + sizeName,
    };

    // Config 2: Symmetric 512KB
    // NOTE: Despite fewer chunks per step (32 vs 512), Sym_512K consistently
    // outperforms Sym_32K. Fewer NVLink round-trips per step outweigh the
    // reduced parallelism.
    BenchmarkConfig sym512k{
        .nBytes = msgBytes,
        .stagedBufferSize = kStagedBufferSize,
        .numBlocks = kNumBlocks,
        .numThreads = kNumThreads,
        .pipelineDepth = kPipelineDepth,
        .chunkSize = kLargeChunk,
        .groupScope = SyncScope::BLOCK,
        .name = "Sym_512K_" + sizeName,
    };

    // Config 3: Asymmetric 32K send / 512K recv
    BenchmarkConfig asymRecv{
        .nBytes = msgBytes,
        .stagedBufferSize = kStagedBufferSize,
        .numBlocks = kNumBlocks,
        .numThreads = kNumThreads,
        .pipelineDepth = kPipelineDepth,
        .chunkSize = kSmallChunk, // transport configured at 32KB
        .groupScope = SyncScope::BLOCK,
        .recvChunkSize = kLargeChunk, // recv batches to 512KB
        .name = "Asym_Recv_32K_512K_" + sizeName,
    };

    // Config 4: Asymmetric 512K send / 32K recv
    BenchmarkConfig asymSend{
        .nBytes = msgBytes,
        .stagedBufferSize = kStagedBufferSize,
        .numBlocks = kNumBlocks,
        .numThreads = kNumThreads,
        .pipelineDepth = kPipelineDepth,
        .chunkSize = kSmallChunk, // transport configured at 32KB
        .groupScope = SyncScope::BLOCK,
        .sendChunkSize = kLargeChunk, // send batches to 512KB
        .name = "Asym_Send_512K_32K_" + sizeName,
    };

    // Config 5: Asymmetric both 512K send / 512K recv
    BenchmarkConfig asymBoth{
        .nBytes = msgBytes,
        .stagedBufferSize = kStagedBufferSize,
        .numBlocks = kNumBlocks,
        .numThreads = kNumThreads,
        .pipelineDepth = kPipelineDepth,
        .chunkSize = kSmallChunk, // transport configured at 32KB
        .groupScope = SyncScope::BLOCK,
        .sendChunkSize = kLargeChunk, // send batches to 512KB
        .recvChunkSize = kLargeChunk, // recv batches to 512KB
        .name = "Asym_Both_512K_" + sizeName,
    };

    // Run each config: create transport, run NCCL ref, then P2P
    for (auto& config : {sym32k, sym512k, asymRecv, asymSend, asymBoth}) {
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
      result.sendChunkSize = config.sendChunkSize;
      result.recvChunkSize = config.recvChunkSize;
      result.numBlocks = config.numBlocks;
      result.numThreads = config.numThreads;

      // NCCL reference
      result.ncclBandwidth = runNcclBenchmark(config, result.ncclTime);

      // Set up buffers and stream
      DeviceBuffer sendBuff(config.nBytes);
      DeviceBuffer recvBuff(config.nBytes);

      if (globalRank == 0) {
        CUDA_CHECK_VOID(cudaMemset(sendBuff.get(), 1, config.nBytes));
      }
      if (globalRank == 1) {
        CUDA_CHECK_VOID(cudaMemset(recvBuff.get(), 0, config.nBytes));
      }

      cudaStream_t stream;
      CUDA_CHECK_VOID(cudaStreamCreate(&stream));

      std::size_t nBytes = config.nBytes;
      SyncScope groupScope = config.groupScope;
      Timeout timeout;
      bool isSend = (globalRank == 0);
      void* devicePtr = isSend ? sendBuff.get() : recvBuff.get();

      // Pre-build all possible arg layouts (only the selected one is used)
      std::size_t sendChunkSize = config.sendChunkSize.value_or(0);
      std::size_t recvChunkSize = config.recvChunkSize.value_or(0);
      void* symArgs[] = {&p2p, &devicePtr, &nBytes, &groupScope, &timeout};
      void* asymSendArgs[] = {
          &p2p, &devicePtr, &nBytes, &sendChunkSize, &groupScope, &timeout};
      void* asymRecvArgs[] = {
          &p2p, &devicePtr, &nBytes, &recvChunkSize, &groupScope, &timeout};

      // P2P benchmark — dispatch based on asymmetric config
      bool hasSendAsym = config.sendChunkSize.has_value();
      bool hasRecvAsym = config.recvChunkSize.has_value();

      if (hasSendAsym || hasRecvAsym) {
        // Correctness check before benchmarking
        runP2pCorrectnessCheck(p2p, config);
      }

      void* kernelFunc = nullptr;
      void** args = nullptr;
      if (hasSendAsym && hasRecvAsym) {
        kernelFunc = isSend ? (void*)comms::pipes::benchmark::p2pAsymmetricSend
                            : (void*)comms::pipes::benchmark::p2pAsymmetricRecv;
        args = isSend ? asymSendArgs : asymRecvArgs;
      } else if (hasSendAsym) {
        kernelFunc = isSend ? (void*)comms::pipes::benchmark::p2pAsymmetricSend
                            : (void*)comms::pipes::benchmark::p2pRecv;
        args = isSend ? asymSendArgs : symArgs;
      } else if (hasRecvAsym) {
        kernelFunc = isSend ? (void*)comms::pipes::benchmark::p2pSend
                            : (void*)comms::pipes::benchmark::p2pAsymmetricRecv;
        args = isSend ? symArgs : asymRecvArgs;
      } else {
        kernelFunc = isSend ? (void*)comms::pipes::benchmark::p2pSend
                            : (void*)comms::pipes::benchmark::p2pRecv;
        args = symArgs;
      }

      result.p2pBandwidth = runUnidirectionalP2pBenchmark(
          config, kernelFunc, args, stream, result.p2pTime);

      CUDA_CHECK_VOID(cudaStreamDestroy(stream));

      result.p2pSpeedup = (result.ncclBandwidth > 0)
          ? result.p2pBandwidth / result.ncclBandwidth
          : 0;

      results.push_back(result);
    }
  }

  printResultsTable(
      results, "Asymmetric Chunk Size UNIDIRECTIONAL Benchmark Results");
}

// =============================================================================
// Bidirectional Asymmetric Chunk Benchmark
// =============================================================================
TEST_F(P2pAsymmetricChunkBenchmarkFixture, BidirectionalAsymmetricChunk) {
  if (worldSize != 2) {
    GTEST_SKIP() << "Test requires exactly 2 ranks, got " << worldSize;
  }

  int peerRank = (globalRank == 0) ? 1 : 0;

  const std::vector<std::pair<std::size_t, std::string>> messageSizes = {
      {64 * 1024 * 1024, "64MB"},
      {128 * 1024 * 1024, "128MB"},
      {256 * 1024 * 1024, "256MB"},
      {512 * 1024 * 1024, "512MB"},
      {1024 * 1024 * 1024, "1GB"},
  };

  constexpr std::size_t kStagedBufferSize = 16 * 1024 * 1024; // 16MB
  constexpr std::size_t kPipelineDepth = 2;
  constexpr int kNumBlocks = 32;
  constexpr int kNumThreads = 512;
  constexpr std::size_t kSmallChunk = 32 * 1024; // 32KB
  constexpr std::size_t kLargeChunk = 512 * 1024; // 512KB
  constexpr std::size_t kLargeMessageThreshold = 256 * 1024 * 1024; // 256MB

  std::vector<BenchmarkResult> results;

  for (const auto& [msgBytes, sizeName] : messageSizes) {
    // Config 1: Symmetric 32KB bidirectional
    BenchmarkConfig sym32k{
        .nBytes = msgBytes,
        .stagedBufferSize = kStagedBufferSize,
        .numBlocks = kNumBlocks,
        .numThreads = kNumThreads,
        .pipelineDepth = kPipelineDepth,
        .chunkSize = kSmallChunk,
        .groupScope = SyncScope::BLOCK,
        .name = "Sym_32K_" + sizeName,
    };

    // Config 2: Symmetric 512KB bidirectional
    // NOTE: Despite fewer chunks per step (32 vs 512), Sym_512K consistently
    // outperforms Sym_32K. Fewer NVLink round-trips per step outweigh the
    // reduced parallelism.
    BenchmarkConfig sym512k{
        .nBytes = msgBytes,
        .stagedBufferSize = kStagedBufferSize,
        .numBlocks = kNumBlocks,
        .numThreads = kNumThreads,
        .pipelineDepth = kPipelineDepth,
        .chunkSize = kLargeChunk,
        .groupScope = SyncScope::BLOCK,
        .name = "Sym_512K_" + sizeName,
    };

    // Config 3: Asymmetric recv bidirectional (BLOCK scope)
    BenchmarkConfig asymRecv{
        .nBytes = msgBytes,
        .stagedBufferSize = kStagedBufferSize,
        .numBlocks = kNumBlocks,
        .numThreads = kNumThreads,
        .pipelineDepth = kPipelineDepth,
        .chunkSize = kSmallChunk,
        .groupScope = SyncScope::BLOCK,
        .recvChunkSize = kLargeChunk,
        .name = "Asym_Recv_32K_512K_" + sizeName,
    };

    // Config 4: Asymmetric send bidirectional (BLOCK scope)
    BenchmarkConfig asymSend{
        .nBytes = msgBytes,
        .stagedBufferSize = kStagedBufferSize,
        .numBlocks = kNumBlocks,
        .numThreads = kNumThreads,
        .pipelineDepth = kPipelineDepth,
        .chunkSize = kSmallChunk,
        .groupScope = SyncScope::BLOCK,
        .sendChunkSize = kLargeChunk,
        .name = "Asym_Send_512K_32K_" + sizeName,
    };

    // Config 5: Asymmetric both bidirectional (BLOCK scope)
    BenchmarkConfig asymBoth{
        .nBytes = msgBytes,
        .stagedBufferSize = kStagedBufferSize,
        .numBlocks = kNumBlocks,
        .numThreads = kNumThreads,
        .pipelineDepth = kPipelineDepth,
        .chunkSize = kSmallChunk,
        .groupScope = SyncScope::BLOCK,
        .sendChunkSize = kLargeChunk,
        .recvChunkSize = kLargeChunk,
        .name = "Asym_Both_512K_" + sizeName,
    };

    std::vector<BenchmarkConfig> configs = {
        sym32k, sym512k, asymRecv, asymSend, asymBoth};

    // Config 6: Asymmetric with cluster scope for 256MB+ messages
    if (msgBytes >= kLargeMessageThreshold) {
      BenchmarkConfig asymCluster{
          .nBytes = msgBytes,
          .stagedBufferSize = kStagedBufferSize,
          .numBlocks = kNumBlocks,
          .numThreads = kNumThreads,
          .pipelineDepth = kPipelineDepth,
          .chunkSize = kSmallChunk,
          .groupScope = SyncScope::CLUSTER,
          .spreadClusterLaunch = true,
          .recvChunkSize = kLargeChunk,
          .name = "Asym_Cluster_" + sizeName,
      };
      configs.push_back(asymCluster);
    }

    for (auto& config : configs) {
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
      result.sendChunkSize = config.sendChunkSize;
      result.recvChunkSize = config.recvChunkSize;
      result.numBlocks = config.numBlocks;
      result.numThreads = config.numThreads;

      // NCCL reference
      result.ncclBandwidth =
          runNcclBidirectionalBenchmark(config, result.ncclTime);

      // Set up buffers
      DeviceBuffer sendBuff(config.nBytes);
      DeviceBuffer recvBuff(config.nBytes);

      CUDA_CHECK_VOID(cudaMemset(sendBuff.get(), globalRank, config.nBytes));
      CUDA_CHECK_VOID(cudaMemset(recvBuff.get(), 0, config.nBytes));

      std::size_t nBytes = config.nBytes;
      void* sendPtr = sendBuff.get();
      void* recvPtr = recvBuff.get();
      SyncScope groupScope = config.groupScope;
      Timeout timeout;

      // Pre-build all possible arg layouts (only the selected one is used)
      std::size_t sendChunkSize = config.sendChunkSize.value_or(0);
      std::size_t recvChunkSize = config.recvChunkSize.value_or(0);
      void* symArgs[] = {
          &p2p, &sendPtr, &recvPtr, &nBytes, &groupScope, &timeout};
      void* asymRecvArgs[] = {
          &p2p,
          &sendPtr,
          &recvPtr,
          &nBytes,
          &recvChunkSize,
          &groupScope,
          &timeout};
      void* asymSendArgs[] = {
          &p2p,
          &sendPtr,
          &recvPtr,
          &nBytes,
          &sendChunkSize,
          &groupScope,
          &timeout};
      void* asymBothArgs[] = {
          &p2p,
          &sendPtr,
          &recvPtr,
          &nBytes,
          &sendChunkSize,
          &recvChunkSize,
          &groupScope,
          &timeout};

      // P2P benchmark — dispatch based on asymmetric config
      bool hasSendAsym = config.sendChunkSize.has_value();
      bool hasRecvAsym = config.recvChunkSize.has_value();

      void* kernelFunc = nullptr;
      void** args = nullptr;
      if (hasSendAsym && hasRecvAsym) {
        kernelFunc =
            (void*)comms::pipes::benchmark::p2pAsymmetricBothBidirectional;
        args = asymBothArgs;
      } else if (hasSendAsym) {
        kernelFunc =
            (void*)comms::pipes::benchmark::p2pAsymmetricSendBidirectional;
        args = asymSendArgs;
      } else if (hasRecvAsym) {
        kernelFunc = (void*)comms::pipes::benchmark::p2pAsymmetricBidirectional;
        args = asymRecvArgs;
      } else {
        kernelFunc = (void*)comms::pipes::benchmark::p2pBidirectional;
        args = symArgs;
      }

      result.p2pBandwidth = runBidirectionalP2pBenchmark(
          config, kernelFunc, args, result.p2pTime);

      result.p2pSpeedup = (result.ncclBandwidth > 0)
          ? result.p2pBandwidth / result.ncclBandwidth
          : 0;

      results.push_back(result);
    }
  }

  printResultsTable(
      results, "Asymmetric Chunk Size BIDIRECTIONAL Benchmark Results");
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
