// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>
#include <nccl.h>
#include <chrono>

#include "comms/common/CudaWrap.h"
#include "comms/pipes/MultiPeerNvlTransport.h"
#include "comms/pipes/TimeoutUtils.h"
#include "comms/pipes/benchmarks/BenchmarkMacros.h"
#include "comms/pipes/collectives/AllToAllv.h"
#include "comms/pipes/collectives/AllToAllvLl128.h"
#include "comms/pipes/ll128/Ll128AutoTune.cuh"
#include "comms/pipes/ll128/Ll128Packet.cuh"
#include "comms/testinfra/BenchmarkTestFixture.h"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/CudaRAII.h"

#include <iomanip>
#include <sstream>
#include <vector>

using meta::comms::CudaEvent;
using meta::comms::DeviceBuffer;

namespace comms::pipes::benchmark {

namespace {

struct Ll128BenchmarkConfig {
  std::size_t bytesPerPeer;
  // Simple protocol settings
  int simpleNumBlocks;
  int simpleNumThreads;
  bool simpleSpreadCluster;
  std::size_t pipelineDepth;
  std::size_t chunkSize;
  std::size_t dataBufferSize;
  // LL128 protocol settings
  int ll128NumBlocks;
  int ll128NumThreads;
  std::string name;
};

struct Ll128BenchmarkResult {
  std::string testName;
  std::size_t bytesPerPeer;
  float ncclBandwidth;
  float simpleBandwidth;
  float ll128Bandwidth;
  float ncclLatency;
  float simpleLatency;
  float ll128Latency;
};

class AllToAllvLl128BenchmarkFixture
    : public meta::comms::BenchmarkTestFixture {
 protected:
  void SetUp() override {
    BenchmarkTestFixture::SetUp();
    CUDA_CHECK_VOID(cudaSetDevice(localRank));
    NCCL_CHECK_VOID(
        ncclCommInitRank(&ncclComm_, worldSize, get_nccl_id(), globalRank));
    CUDA_CHECK_VOID(cudaStreamCreate(&stream_));
  }

  void TearDown() override {
    NCCL_CHECK_VOID(ncclCommDestroy(ncclComm_));
    CUDA_CHECK_VOID(cudaStreamDestroy(stream_));
    BenchmarkTestFixture::TearDown();
  }

  ncclUniqueId get_nccl_id() {
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

  float run_nccl_benchmark(
      const Ll128BenchmarkConfig& config,
      float& latency_us) {
    const int nranks = worldSize;
    const std::size_t bytesPerPeer = config.bytesPerPeer;
    const std::size_t totalBytes = bytesPerPeer * nranks;

    DeviceBuffer sendBuffer(totalBytes);
    DeviceBuffer recvBuffer(totalBytes);
    CUDA_CHECK(cudaMemset(sendBuffer.get(), 1, totalBytes));
    CUDA_CHECK(cudaMemset(recvBuffer.get(), 0, totalBytes));

    std::vector<size_t> sendcounts(nranks, bytesPerPeer);
    std::vector<size_t> recvcounts(nranks, bytesPerPeer);
    std::vector<size_t> sdispls(nranks);
    std::vector<size_t> rdispls(nranks);
    for (int i = 0; i < nranks; i++) {
      sdispls[i] = i * bytesPerPeer;
      rdispls[i] = i * bytesPerPeer;
    }

    CudaEvent start, stop;
    constexpr int kNIter = 100;
    constexpr int kNWarmup = 5;

    bootstrap->barrierAll();
    for (int i = 0; i < kNWarmup; i++) {
      NCCL_CHECK(ncclAllToAllv(
          sendBuffer.get(),
          sendcounts.data(),
          sdispls.data(),
          recvBuffer.get(),
          recvcounts.data(),
          rdispls.data(),
          ncclChar,
          ncclComm_,
          stream_));
    }

    CUDA_CHECK(cudaEventRecord(start.get(), stream_));
    for (int i = 0; i < kNIter; i++) {
      NCCL_CHECK(ncclAllToAllv(
          sendBuffer.get(),
          sendcounts.data(),
          sdispls.data(),
          recvBuffer.get(),
          recvcounts.data(),
          rdispls.data(),
          ncclChar,
          ncclComm_,
          stream_));
    }
    CUDA_CHECK(cudaEventRecord(stop.get(), stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    float totalTime_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime_ms, start.get(), stop.get()));
    float avgTime_ms = totalTime_ms / kNIter;
    latency_us = avgTime_ms * 1000.0f;

    std::size_t totalDataMoved = 2 * totalBytes;
    float bandwidth_GBps = (totalDataMoved / (1000.0f * 1000.0f * 1000.0f)) /
        (avgTime_ms / 1000.0f);

    bootstrap->barrierAll();
    return bandwidth_GBps;
  }

  float run_simple_benchmark(
      const Ll128BenchmarkConfig& config,
      float& latency_us) {
    const int nranks = worldSize;
    const std::size_t bytesPerPeer = config.bytesPerPeer;
    const std::size_t totalBytes = bytesPerPeer * nranks;

    DeviceBuffer sendBuffer(totalBytes);
    DeviceBuffer recvBuffer(totalBytes);
    CUDA_CHECK(cudaMemset(sendBuffer.get(), 1, totalBytes));
    CUDA_CHECK(cudaMemset(recvBuffer.get(), 0, totalBytes));

    MultiPeerNvlTransportConfig nvlConfig{
        .dataBufferSize = config.dataBufferSize,
        .chunkSize = config.chunkSize,
        .pipelineDepth = config.pipelineDepth,
    };

    MultiPeerNvlTransport transport(globalRank, nranks, bootstrap, nvlConfig);
    transport.exchange();

    auto transports_span = transport.getDeviceTransports();

    std::vector<ChunkInfo> h_send_chunks, h_recv_chunks;
    for (int rank = 0; rank < nranks; rank++) {
      h_send_chunks.emplace_back(rank * bytesPerPeer, bytesPerPeer);
      h_recv_chunks.emplace_back(rank * bytesPerPeer, bytesPerPeer);
    }

    DeviceBuffer d_send_chunks(sizeof(ChunkInfo) * nranks);
    DeviceBuffer d_recv_chunks(sizeof(ChunkInfo) * nranks);
    CUDA_CHECK(cudaMemcpy(
        d_send_chunks.get(),
        h_send_chunks.data(),
        sizeof(ChunkInfo) * nranks,
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_recv_chunks.get(),
        h_recv_chunks.data(),
        sizeof(ChunkInfo) * nranks,
        cudaMemcpyHostToDevice));

    DeviceSpan<ChunkInfo> send_chunk_infos(
        static_cast<ChunkInfo*>(d_send_chunks.get()), nranks);
    DeviceSpan<ChunkInfo> recv_chunk_infos(
        static_cast<ChunkInfo*>(d_recv_chunks.get()), nranks);

    dim3 defaultClusterDim(comms::common::kDefaultClusterSize, 1, 1);
    std::optional<dim3> clusterDimOpt = config.simpleSpreadCluster
        ? std::optional{defaultClusterDim}
        : std::nullopt;

    // Pre-build Timeout once to avoid per-call
    // cudaGetDevice/cudaDeviceGetAttribute overhead.
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    Timeout timeout_config = makeTimeout(0, device);

    CudaEvent start, stop;
    constexpr int kNIter = 100;
    constexpr int kNWarmup = 5;

    bootstrap->barrierAll();
    for (int i = 0; i < kNWarmup; i++) {
      comms::pipes::all_to_allv(
          recvBuffer.get(),
          sendBuffer.get(),
          globalRank,
          transports_span,
          send_chunk_infos,
          recv_chunk_infos,
          timeout_config,
          nullptr,
          config.simpleNumBlocks,
          config.simpleNumThreads,
          clusterDimOpt);
    }

    CUDA_CHECK(cudaEventRecord(start.get()));
    for (int i = 0; i < kNIter; i++) {
      comms::pipes::all_to_allv(
          recvBuffer.get(),
          sendBuffer.get(),
          globalRank,
          transports_span,
          send_chunk_infos,
          recv_chunk_infos,
          timeout_config,
          nullptr,
          config.simpleNumBlocks,
          config.simpleNumThreads,
          clusterDimOpt);
    }
    CUDA_CHECK(cudaEventRecord(stop.get()));
    CUDA_CHECK(cudaDeviceSynchronize());

    float totalTime_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime_ms, start.get(), stop.get()));
    float avgTime_ms = totalTime_ms / kNIter;
    latency_us = avgTime_ms * 1000.0f;

    std::size_t totalDataMoved = 2 * totalBytes;
    float bandwidth_GBps = (totalDataMoved / (1000.0f * 1000.0f * 1000.0f)) /
        (avgTime_ms / 1000.0f);

    bootstrap->barrierAll();
    return bandwidth_GBps;
  }

  float run_ll128_benchmark(
      const Ll128BenchmarkConfig& config,
      float& latency_us) {
    const int nranks = worldSize;
    const std::size_t bytesPerPeer = config.bytesPerPeer;
    const std::size_t totalBytes = bytesPerPeer * nranks;

    DeviceBuffer sendBuffer(totalBytes);
    DeviceBuffer recvBuffer(totalBytes);
    CUDA_CHECK(cudaMemset(sendBuffer.get(), 1, totalBytes));
    CUDA_CHECK(cudaMemset(recvBuffer.get(), 0, totalBytes));

    MultiPeerNvlTransportConfig nvlConfig{
        .dataBufferSize = config.dataBufferSize,
        .chunkSize = config.chunkSize,
        .pipelineDepth = config.pipelineDepth,
        .ll128BufferSize = ll128_buffer_size(bytesPerPeer),
    };

    MultiPeerNvlTransport transport(globalRank, nranks, bootstrap, nvlConfig);
    transport.exchange();

    auto transports_span = transport.getDeviceTransports();

    std::vector<ChunkInfo> h_send_chunks, h_recv_chunks;
    for (int rank = 0; rank < nranks; rank++) {
      h_send_chunks.emplace_back(rank * bytesPerPeer, bytesPerPeer);
      h_recv_chunks.emplace_back(rank * bytesPerPeer, bytesPerPeer);
    }

    DeviceBuffer d_send_chunks(sizeof(ChunkInfo) * nranks);
    DeviceBuffer d_recv_chunks(sizeof(ChunkInfo) * nranks);
    CUDA_CHECK(cudaMemcpy(
        d_send_chunks.get(),
        h_send_chunks.data(),
        sizeof(ChunkInfo) * nranks,
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        d_recv_chunks.get(),
        h_recv_chunks.data(),
        sizeof(ChunkInfo) * nranks,
        cudaMemcpyHostToDevice));

    DeviceSpan<ChunkInfo> send_chunk_infos(
        static_cast<ChunkInfo*>(d_send_chunks.get()), nranks);
    DeviceSpan<ChunkInfo> recv_chunk_infos(
        static_cast<ChunkInfo*>(d_recv_chunks.get()), nranks);

    CudaEvent start, stop;
    constexpr int kNIter = 100;
    constexpr int kNWarmup = 5;

    // Create timeout ONCE outside the loop to avoid per-call
    // cudaGetDevice/cudaDeviceGetAttribute overhead.
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    Timeout timeout_config = makeTimeout(30000, device);

    // Warmup: per-iteration sync to ensure each iteration completes
    bootstrap->barrierAll();
    for (int i = 0; i < kNWarmup; i++) {
      comms::pipes::all_to_allv_ll128(
          recvBuffer.get(),
          sendBuffer.get(),
          globalRank,
          transports_span,
          send_chunk_infos,
          recv_chunk_infos,
          timeout_config,
          nullptr,
          config.ll128NumBlocks,
          config.ll128NumThreads);
    }

    // Timed loop
    CUDA_CHECK(cudaEventRecord(start.get()));
    for (int i = 0; i < kNIter; i++) {
      // No per-iteration sync: same-stream ordering guarantees sequential
      // kernel execution, and LL128 flag protocol handles inter-GPU sync.
      comms::pipes::all_to_allv_ll128(
          recvBuffer.get(),
          sendBuffer.get(),
          globalRank,
          transports_span,
          send_chunk_infos,
          recv_chunk_infos,
          timeout_config,
          nullptr,
          config.ll128NumBlocks,
          config.ll128NumThreads);
    }
    CUDA_CHECK(cudaEventRecord(stop.get()));
    CUDACHECK_TEST(cudaDeviceSynchronize());

    float totalTime_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime_ms, start.get(), stop.get()));
    float avgTime_ms = totalTime_ms / kNIter;
    latency_us = avgTime_ms * 1000.0f;

    std::size_t totalDataMoved = 2 * totalBytes;
    float bandwidth_GBps = (totalDataMoved / (1000.0f * 1000.0f * 1000.0f)) /
        (avgTime_ms / 1000.0f);

    bootstrap->barrierAll();
    return bandwidth_GBps;
  }

  void print_results_table(const std::vector<Ll128BenchmarkResult>& results) {
    if (globalRank != 0) {
      return;
    }

    auto format_bytes = [](std::size_t bytes) -> std::string {
      if (bytes < 1024) {
        return std::to_string(bytes) + "B";
      }
      if (bytes < 1024 * 1024) {
        return std::to_string(bytes / 1024) + "KB";
      }
      return std::to_string(bytes / (1024 * 1024)) + "MB";
    };

    std::stringstream ss;
    ss << "\n";
    ss << std::string(125, '=') << "\n";
    ss << "                                  NCCL vs Pipes Simple vs Pipes LL128 — AllToAllv Benchmark\n";
    ss << std::string(125, '=') << "\n";
    ss << std::left << std::setw(14) << "Test" << std::right << std::setw(12)
       << "Per-Peer" << std::right << std::setw(13) << "NCCL BW" << std::right
       << std::setw(13) << "Simple BW" << std::right << std::setw(13)
       << "LL128 BW" << std::right << std::setw(12) << "LL128/NCCL"
       << std::right << std::setw(12) << "LL128/Simp" << std::right
       << std::setw(12) << "NCCL Lat" << std::right << std::setw(12)
       << "Simp Lat" << std::right << std::setw(12) << "LL128 Lat\n";
    ss << std::left << std::setw(14) << "" << std::right << std::setw(12) << ""
       << std::right << std::setw(13) << "(GB/s)" << std::right << std::setw(13)
       << "(GB/s)" << std::right << std::setw(13) << "(GB/s)" << std::right
       << std::setw(12) << "" << std::right << std::setw(12) << "" << std::right
       << std::setw(12) << "(us)" << std::right << std::setw(12) << "(us)"
       << std::right << std::setw(12) << "(us)\n";
    ss << std::string(125, '-') << "\n";

    for (const auto& r : results) {
      float ll128VsNccl =
          r.ncclBandwidth > 0 ? r.ll128Bandwidth / r.ncclBandwidth : 0;
      float ll128VsSimple =
          r.simpleBandwidth > 0 ? r.ll128Bandwidth / r.simpleBandwidth : 0;

      ss << std::left << std::setw(14) << r.testName << std::right
         << std::setw(12) << format_bytes(r.bytesPerPeer) << std::right
         << std::setw(13) << std::fixed << std::setprecision(2)
         << r.ncclBandwidth << std::right << std::setw(13) << std::fixed
         << std::setprecision(2) << r.simpleBandwidth << std::right
         << std::setw(13) << std::fixed << std::setprecision(2)
         << r.ll128Bandwidth << std::right << std::setw(11) << std::fixed
         << std::setprecision(2) << ll128VsNccl << "x" << std::right
         << std::setw(11) << std::fixed << std::setprecision(2) << ll128VsSimple
         << "x" << std::right << std::setw(12) << std::fixed
         << std::setprecision(1) << r.ncclLatency << std::right << std::setw(12)
         << std::fixed << std::setprecision(1) << r.simpleLatency << std::right
         << std::setw(12) << std::fixed << std::setprecision(1)
         << r.ll128Latency << "\n";
    }

    ss << std::string(125, '=') << "\n";
    ss << "BW = Algorithm bandwidth (2 x total data / time), " << worldSize
       << " ranks\n";
    ss << "LL128/NCCL = LL128 BW / NCCL BW, LL128/Simp = LL128 BW / Simple BW\n";
    ss << std::string(125, '=') << "\n";

    XLOG(INFO) << ss.str();
  }

  ncclComm_t ncclComm_{};
  cudaStream_t stream_{};
};

TEST_F(AllToAllvLl128BenchmarkFixture, Ll128VsSimpleVsNccl) {
  if (globalRank == 0) {
    XLOG(INFO) << "\n=== LL128 vs Simple vs NCCL AllToAllv Comparison ===\n";
  }

  std::vector<Ll128BenchmarkConfig> configs;
  const std::size_t kDataBufferSize = 8 * 1024 * 1024; // 8MB

  // Message sizes focused on LL128's sweet spot
  // Simple protocol settings from AllToAllvBenchmark.cc for fair comparison

  // Helper to get auto-tuned LL128 block count for a given message size.
  auto auto_ll128_blocks = [&](std::size_t bytesPerPeer) {
    return ll128_auto_tune_alltoallv(bytesPerPeer, worldSize).numBlocks;
  };

  // 128B
  configs.push_back({
      .bytesPerPeer = 128,
      .simpleNumBlocks = 8,
      .simpleNumThreads = 512,
      .simpleSpreadCluster = true,
      .pipelineDepth = 2,
      .chunkSize = 64 * 1024,
      .dataBufferSize = kDataBufferSize,
      .ll128NumBlocks = auto_ll128_blocks(128),
      .ll128NumThreads = 512,
      .name = "128B",
  });

  // 256B
  configs.push_back({
      .bytesPerPeer = 256,
      .simpleNumBlocks = 8,
      .simpleNumThreads = 512,
      .simpleSpreadCluster = true,
      .pipelineDepth = 2,
      .chunkSize = 64 * 1024,
      .dataBufferSize = kDataBufferSize,
      .ll128NumBlocks = auto_ll128_blocks(256),
      .ll128NumThreads = 512,
      .name = "256B",
  });

  // 512B
  configs.push_back({
      .bytesPerPeer = 512,
      .simpleNumBlocks = 8,
      .simpleNumThreads = 512,
      .simpleSpreadCluster = true,
      .pipelineDepth = 2,
      .chunkSize = 64 * 1024,
      .dataBufferSize = kDataBufferSize,
      .ll128NumBlocks = auto_ll128_blocks(512),
      .ll128NumThreads = 512,
      .name = "512B",
  });

  // 1KB
  configs.push_back({
      .bytesPerPeer = 1024,
      .simpleNumBlocks = 8,
      .simpleNumThreads = 512,
      .simpleSpreadCluster = true,
      .pipelineDepth = 2,
      .chunkSize = 64 * 1024,
      .dataBufferSize = kDataBufferSize,
      .ll128NumBlocks = auto_ll128_blocks(1024),
      .ll128NumThreads = 512,
      .name = "1KB",
  });

  // 4KB
  configs.push_back({
      .bytesPerPeer = 4 * 1024,
      .simpleNumBlocks = 8,
      .simpleNumThreads = 512,
      .simpleSpreadCluster = true,
      .pipelineDepth = 2,
      .chunkSize = 64 * 1024,
      .dataBufferSize = kDataBufferSize,
      .ll128NumBlocks = auto_ll128_blocks(4 * 1024),
      .ll128NumThreads = 512,
      .name = "4KB",
  });

  // 16KB
  configs.push_back({
      .bytesPerPeer = 16 * 1024,
      .simpleNumBlocks = 8,
      .simpleNumThreads = 512,
      .simpleSpreadCluster = true,
      .pipelineDepth = 2,
      .chunkSize = 64 * 1024,
      .dataBufferSize = kDataBufferSize,
      .ll128NumBlocks = auto_ll128_blocks(16 * 1024),
      .ll128NumThreads = 512,
      .name = "16KB",
  });

  // 64KB
  configs.push_back({
      .bytesPerPeer = 64 * 1024,
      .simpleNumBlocks = 8,
      .simpleNumThreads = 512,
      .simpleSpreadCluster = true,
      .pipelineDepth = 2,
      .chunkSize = 64 * 1024,
      .dataBufferSize = kDataBufferSize,
      .ll128NumBlocks = auto_ll128_blocks(64 * 1024),
      .ll128NumThreads = 512,
      .name = "64KB",
  });

  // 256KB
  configs.push_back({
      .bytesPerPeer = 256 * 1024,
      .simpleNumBlocks = 8,
      .simpleNumThreads = 512,
      .simpleSpreadCluster = true,
      .pipelineDepth = 2,
      .chunkSize = 64 * 1024,
      .dataBufferSize = kDataBufferSize,
      .ll128NumBlocks = auto_ll128_blocks(256 * 1024),
      .ll128NumThreads = 512,
      .name = "256KB",
  });

  // 1MB
  configs.push_back({
      .bytesPerPeer = 1024 * 1024,
      .simpleNumBlocks = 16,
      .simpleNumThreads = 512,
      .simpleSpreadCluster = true,
      .pipelineDepth = 2,
      .chunkSize = 64 * 1024,
      .dataBufferSize = kDataBufferSize,
      .ll128NumBlocks = auto_ll128_blocks(1024 * 1024),
      .ll128NumThreads = 512,
      .name = "1MB",
  });

  std::vector<Ll128BenchmarkResult> results;

  for (const auto& config : configs) {
    float ncclLatency = 0.0f;
    float ncclBw = run_nccl_benchmark(config, ncclLatency);

    float simpleLatency = 0.0f;
    float simpleBw = run_simple_benchmark(config, simpleLatency);

    float ll128Latency = 0.0f;
    float ll128Bw = run_ll128_benchmark(config, ll128Latency);

    if (globalRank == 0) {
      results.push_back({
          .testName = config.name,
          .bytesPerPeer = config.bytesPerPeer,
          .ncclBandwidth = ncclBw,
          .simpleBandwidth = simpleBw,
          .ll128Bandwidth = ll128Bw,
          .ncclLatency = ncclLatency,
          .simpleLatency = simpleLatency,
          .ll128Latency = ll128Latency,
      });
    }

    bootstrap->barrierAll();
  }

  print_results_table(results);
}

TEST_F(AllToAllvLl128BenchmarkFixture, Ll128BlockThreadSweep) {
  if (globalRank == 0) {
    XLOG(INFO) << "\n=== LL128 AllToAllv Block Count Sweep (threads=512) ===\n";
  }

  // Message sizes that span medium-to-large range where block count matters
  const std::vector<std::size_t> messageSizes = {
      4 * 1024, // 4KB
      16 * 1024, // 16KB
      64 * 1024, // 64KB
      256 * 1024, // 256KB
      1024 * 1024, // 1MB
  };

  // Block counts to sweep (threads fixed at 256)
  const std::vector<int> blockCounts = {
      8, 16, 18, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512};

  struct SweepResult {
    std::size_t bytesPerPeer;
    int numBlocks;
    float bandwidth;
    float latency;
    int autoTuneBlocks; // what ll128_auto_tune_alltoallv recommends
  };

  std::vector<SweepResult> results;

  for (auto bytesPerPeer : messageSizes) {
    auto autoConfig = ll128_auto_tune_alltoallv(bytesPerPeer, worldSize);

    for (auto numBlocks : blockCounts) {
      Ll128BenchmarkConfig config{
          .bytesPerPeer = bytesPerPeer,
          .simpleNumBlocks = 4,
          .simpleNumThreads = 256,
          .simpleSpreadCluster = false,
          .pipelineDepth = 2,
          .chunkSize = 64 * 1024,
          .dataBufferSize = 8 * 1024 * 1024,
          .ll128NumBlocks = numBlocks,
          .ll128NumThreads = 512,
          .name = "",
      };

      float latency = 0.0f;
      float bw = run_ll128_benchmark(config, latency);

      if (globalRank == 0) {
        results.push_back({
            .bytesPerPeer = bytesPerPeer,
            .numBlocks = numBlocks,
            .bandwidth = bw,
            .latency = latency,
            .autoTuneBlocks = autoConfig.numBlocks,
        });
      }
      bootstrap->barrierAll();
    }
  }

  // Print results table
  if (globalRank == 0) {
    auto format_bytes = [](std::size_t bytes) -> std::string {
      if (bytes < 1024) {
        return std::to_string(bytes) + "B";
      }
      if (bytes < 1024 * 1024) {
        return std::to_string(bytes / 1024) + "KB";
      }
      return std::to_string(bytes / (1024 * 1024)) + "MB";
    };

    std::stringstream ss;
    ss << "\n";
    ss << std::string(80, '=') << "\n";
    ss << "  LL128 AllToAllv Block Count Sweep (" << worldSize
       << " ranks, threads=512)\n";
    ss << std::string(80, '=') << "\n";
    ss << std::left << std::setw(12) << "Per-Peer" << std::right
       << std::setw(10) << "Blocks" << std::right << std::setw(14)
       << "BW (GB/s)" << std::right << std::setw(14) << "Lat (us)" << std::right
       << std::setw(14) << "AutoTune" << std::right << std::setw(10)
       << "Match?\n";
    ss << std::string(80, '-') << "\n";

    for (const auto& r : results) {
      bool isAutoTune = (r.numBlocks == r.autoTuneBlocks);
      ss << std::left << std::setw(12) << format_bytes(r.bytesPerPeer)
         << std::right << std::setw(10) << r.numBlocks << std::right
         << std::setw(14) << std::fixed << std::setprecision(2) << r.bandwidth
         << std::right << std::setw(14) << std::fixed << std::setprecision(1)
         << r.latency << std::right << std::setw(14) << r.autoTuneBlocks
         << std::right << std::setw(10) << (isAutoTune ? " <--" : "") << "\n";
    }

    ss << std::string(80, '=') << "\n";
    XLOG(INFO) << ss.str();
  }
}

TEST_F(AllToAllvLl128BenchmarkFixture, Ll128ThreadSweep) {
  if (globalRank == 0) {
    XLOG(INFO)
        << "\n=== LL128 AllToAllv Thread Count Sweep (256 vs 512 threads) ===\n";
  }

  const std::vector<std::size_t> messageSizes = {
      4 * 1024, // 4KB
      16 * 1024, // 16KB
      64 * 1024, // 64KB
      256 * 1024, // 256KB
  };

  const std::vector<int> blockCounts = {8, 16, 32, 64, 128, 256, 384, 512};
  const std::vector<int> threadCounts = {256, 512};

  struct SweepResult {
    std::size_t bytesPerPeer;
    int numBlocks;
    int numThreads;
    float bandwidth;
    float latency;
  };

  std::vector<SweepResult> results;

  for (auto bytesPerPeer : messageSizes) {
    for (auto numThreads : threadCounts) {
      for (auto numBlocks : blockCounts) {
        Ll128BenchmarkConfig config{
            .bytesPerPeer = bytesPerPeer,
            .simpleNumBlocks = 4,
            .simpleNumThreads = 256,
            .simpleSpreadCluster = false,
            .pipelineDepth = 2,
            .chunkSize = 64 * 1024,
            .dataBufferSize = 8 * 1024 * 1024,
            .ll128NumBlocks = numBlocks,
            .ll128NumThreads = numThreads,
            .name = "",
        };

        float latency = 0.0f;
        float bw = run_ll128_benchmark(config, latency);

        if (globalRank == 0) {
          results.push_back({
              .bytesPerPeer = bytesPerPeer,
              .numBlocks = numBlocks,
              .numThreads = numThreads,
              .bandwidth = bw,
              .latency = latency,
          });
        }
        bootstrap->barrierAll();
      }
    }
  }

  if (globalRank == 0) {
    auto format_bytes = [](std::size_t bytes) -> std::string {
      if (bytes < 1024) {
        return std::to_string(bytes) + "B";
      }
      if (bytes < 1024 * 1024) {
        return std::to_string(bytes / 1024) + "KB";
      }
      return std::to_string(bytes / (1024 * 1024)) + "MB";
    };

    std::stringstream ss;
    ss << "\n";
    ss << std::string(80, '=') << "\n";
    ss << "  LL128 AllToAllv Thread Count Sweep (" << worldSize << " ranks)\n";
    ss << std::string(80, '=') << "\n";
    ss << std::left << std::setw(12) << "Per-Peer" << std::right
       << std::setw(10) << "Blocks" << std::right << std::setw(10) << "Threads"
       << std::right << std::setw(14) << "BW (GB/s)" << std::right
       << std::setw(14) << "Lat (us)\n";
    ss << std::string(80, '-') << "\n";

    for (const auto& r : results) {
      ss << std::left << std::setw(12) << format_bytes(r.bytesPerPeer)
         << std::right << std::setw(10) << r.numBlocks << std::right
         << std::setw(10) << r.numThreads << std::right << std::setw(14)
         << std::fixed << std::setprecision(2) << r.bandwidth << std::right
         << std::setw(14) << std::fixed << std::setprecision(1) << r.latency
         << "\n";
    }

    ss << std::string(80, '=') << "\n";
    XLOG(INFO) << ss.str();
  }
}

} // namespace

} // namespace comms::pipes::benchmark

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  ::testing::AddGlobalTestEnvironment(new meta::comms::BenchmarkEnvironment());
  return RUN_ALL_TESTS();
}
