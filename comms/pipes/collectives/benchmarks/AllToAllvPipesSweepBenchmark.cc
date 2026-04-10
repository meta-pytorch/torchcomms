// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <nccl.h>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/MultiPeerDeviceHandle.cuh"
#include "comms/pipes/MultiPeerTransport.h"
#include "comms/pipes/Transport.cuh"
#include "comms/pipes/collectives/AllToAllv.h"
#include "comms/pipes/collectives/AllToAllvAutoTuneConfig.h"
#include "comms/testinfra/BenchmarkTestFixture.h"
#include "comms/utils/CudaRAII.h"

#define CUDA_CHECK_SWEEP(cmd)                                \
  do {                                                       \
    cudaError_t err = (cmd);                                 \
    if (err != cudaSuccess) {                                \
      XLOGF(ERR, "CUDA error: {}", cudaGetErrorString(err)); \
      std::abort();                                          \
    }                                                        \
  } while (0)

#define NCCL_CHECK_SWEEP(cmd)                                \
  do {                                                       \
    ncclResult_t res = (cmd);                                \
    if (res != ncclSuccess) {                                \
      XLOGF(ERR, "NCCL error: {}", ncclGetErrorString(res)); \
      std::abort();                                          \
    }                                                        \
  } while (0)

using meta::comms::CudaEvent;
using meta::comms::DeviceBuffer;

namespace comms::pipes::benchmark {

namespace {

// Cap GPU buffer allocations to avoid OOM.
// ncclAllToAll needs linear buffers (msgSize × worldSize), so this cap
// determines the max msg size that AllToAll can benchmark at a given
// world size. Pipes AllToAllv uses modular wrapping so it always works.
// 32 GB allows AllToAll up to:
//   16 ranks (2×8):   256MB/peer (256MB × 16 = 4GB)
//   256 ranks (32×8): 128MB/peer (128MB × 256 = 32GB)
//   1024 ranks (128×8): 32MB/peer (32MB × 1024 = 32GB)
constexpr std::size_t kMaxBenchmarkBufSize = 32ULL * 1024 * 1024 * 1024;

enum class TopologyMode {
  NVL_ONLY,
  HYBRID,
};

std::string formatBytes(std::size_t bytes) {
  if (bytes >= 1024ULL * 1024 * 1024) {
    return std::to_string(bytes / (1024ULL * 1024 * 1024)) + "GB";
  }
  if (bytes >= 1024 * 1024) {
    return std::to_string(bytes / (1024 * 1024)) + "MB";
  }
  if (bytes >= 1024) {
    return std::to_string(bytes / 1024) + "KB";
  }
  return std::to_string(bytes) + "B";
}

int iterCountForMsgSize(std::size_t msgSize) {
  if (msgSize <= 64 * 1024) {
    return 100;
  }
  if (msgSize <= 4 * 1024 * 1024) {
    return 50;
  }
  return 10;
}

std::vector<std::size_t> allMessageSizes() {
  std::vector<std::size_t> sizes;
  for (std::size_t s = 1024; s <= 32ULL * 1024 * 1024; s *= 2) {
    sizes.push_back(s);
  }
  return sizes;
}

struct SweepResult {
  std::size_t msgSize;
  int numBlocks;
  int numThreads;
  // NCCL AllToAll (equal-count baseline)
  double ncclA2aLatencyUs;
  double ncclA2aAlgoBW;
  double ncclA2aBusBW;
  bool ncclA2aSkipped; // true when buffer exceeds cap
  // Pipes AllToAllv
  double pipesLatencyUs;
  double pipesAlgoBW;
  double pipesBusBW;
  // Speedup
  double speedupVsA2a;
};

class AllToAllvPipesSweepFixture : public meta::comms::BenchmarkTestFixture {
 protected:
  void SetUp() override {
    BenchmarkTestFixture::SetUp();
    CUDA_CHECK_SWEEP(cudaSetDevice(localRank));
    CUDA_CHECK_SWEEP(cudaStreamCreate(&stream_));

    ncclUniqueId id;
    if (globalRank == 0) {
      NCCL_CHECK_SWEEP(ncclGetUniqueId(&id));
    }
    std::vector<ncclUniqueId> allIds(worldSize);
    allIds[globalRank] = id;
    bootstrap
        ->allGather(allIds.data(), sizeof(ncclUniqueId), globalRank, worldSize)
        .get();
    id = allIds[0];
    NCCL_CHECK_SWEEP(ncclCommInitRank(&ncclComm_, worldSize, id, globalRank));

    bool isSingleNode = (localSize == worldSize);
    topoMode_ = isSingleNode ? TopologyMode::NVL_ONLY : TopologyMode::HYBRID;
  }

  void TearDown() override {
    NCCL_CHECK_SWEEP(ncclCommDestroy(ncclComm_));
    CUDA_CHECK_SWEEP(cudaStreamDestroy(stream_));
    BenchmarkTestFixture::TearDown();
  }

  // ─── NCCL AllToAll benchmark (equal-count) ─────────────────────────────
  double runNcclAllToAllBenchmark(
      std::size_t bytesPerPeer,
      int nIter,
      double& latencyUs,
      bool& skipped) {
    const std::size_t logicalTotalBytes = bytesPerPeer * worldSize;
    // ncclAllToAll uses implicit linear offsets (j*count) — no modular
    // wrapping. If the full allocation would exceed our cap, skip.
    if (logicalTotalBytes > kMaxBenchmarkBufSize) {
      latencyUs = 0;
      skipped = true;
      return 0;
    }
    skipped = false;
    DeviceBuffer sendBuffer(logicalTotalBytes);
    DeviceBuffer recvBuffer(logicalTotalBytes);
    CUDA_CHECK_SWEEP(
        cudaMemset(sendBuffer.get(), globalRank & 0xFF, logicalTotalBytes));
    CUDA_CHECK_SWEEP(cudaMemset(recvBuffer.get(), 0, logicalTotalBytes));

    size_t count = bytesPerPeer;

    CudaEvent start, stop;
    bootstrap->barrierAll();
    for (int i = 0; i < 5; ++i) {
      NCCL_CHECK_SWEEP(ncclAllToAll(
          sendBuffer.get(),
          recvBuffer.get(),
          count,
          ncclChar,
          ncclComm_,
          stream_));
    }
    CUDA_CHECK_SWEEP(cudaStreamSynchronize(stream_));
    bootstrap->barrierAll();

    CUDA_CHECK_SWEEP(cudaEventRecord(start.get(), stream_));
    for (int i = 0; i < nIter; ++i) {
      NCCL_CHECK_SWEEP(ncclAllToAll(
          sendBuffer.get(),
          recvBuffer.get(),
          count,
          ncclChar,
          ncclComm_,
          stream_));
    }
    CUDA_CHECK_SWEEP(cudaEventRecord(stop.get(), stream_));
    CUDA_CHECK_SWEEP(cudaStreamSynchronize(stream_));

    float totalMs = 0;
    CUDA_CHECK_SWEEP(cudaEventElapsedTime(&totalMs, start.get(), stop.get()));
    float avgMs = totalMs / nIter;
    latencyUs = avgMs * 1000.0;
    double algoBW =
        (logicalTotalBytes / (1000.0 * 1000.0 * 1000.0)) / (avgMs / 1000.0);
    return algoBW;
  }

  // ─── Print sweep results ──────────────────────────────────────────────
  void printResults(const std::vector<SweepResult>& results) {
    if (globalRank != 0 || results.empty()) {
      return;
    }

    int nnodes = worldSize / localSize;
    const char* topoStr = (topoMode_ == TopologyMode::HYBRID)
        ? "hybrid (NVLink + IBGDA)"
        : "NVLink-only";

    fprintf(stderr, "\n");
    fprintf(
        stderr,
        "  Topology: %s, %d nodes x %d GPUs = %d ranks\n",
        topoStr,
        nnodes,
        localSize,
        worldSize);
    fprintf(
        stderr,
        "============================================================"
        "============================================================\n");
    fprintf(
        stderr,
        "  %-8s %4s %4s | %8s %9s %9s | %9s %10s %10s | %7s\n",
        "MsgSize",
        "Blks",
        "Thds",
        "A2a Lat",
        "A2a AlgBW",
        "A2a BusBW",
        "Pipes Lat",
        "PipesAlgBW",
        "PipesBusBW",
        "vs A2a");
    fprintf(
        stderr,
        "  -------- ---- ---- | -------- --------- --------- "
        "| --------- ---------- ---------- | -------\n");

    double logSum = 0;
    int validCount = 0;
    double bestSpeedup = 0, worstSpeedup = 1e9;
    std::size_t bestMsg = 0, worstMsg = 0;

    for (const auto& r : results) {
      if (r.ncclA2aSkipped) {
        fprintf(
            stderr,
            "  %-8s %4d %4d | %8s %9s %9s | %9.1f %10.2f %10.2f | %7s\n",
            formatBytes(r.msgSize).c_str(),
            r.numBlocks,
            r.numThreads,
            "N/A",
            "N/A",
            "N/A",
            r.pipesLatencyUs,
            r.pipesAlgoBW,
            r.pipesBusBW,
            "N/A");
      } else {
        fprintf(
            stderr,
            "  %-8s %4d %4d | %8.1f %9.2f %9.2f | %9.1f %10.2f %10.2f | %6.2fx\n",
            formatBytes(r.msgSize).c_str(),
            r.numBlocks,
            r.numThreads,
            r.ncclA2aLatencyUs,
            r.ncclA2aAlgoBW,
            r.ncclA2aBusBW,
            r.pipesLatencyUs,
            r.pipesAlgoBW,
            r.pipesBusBW,
            r.speedupVsA2a);

        auto safeLog = [](double v) { return std::log(std::max(v, 0.01)); };
        logSum += safeLog(r.speedupVsA2a);
        ++validCount;

        if (r.speedupVsA2a > bestSpeedup) {
          bestSpeedup = r.speedupVsA2a;
          bestMsg = r.msgSize;
        }
        if (r.speedupVsA2a < worstSpeedup) {
          worstSpeedup = r.speedupVsA2a;
          worstMsg = r.msgSize;
        }
      }
    }

    fprintf(
        stderr,
        "============================================================"
        "============================================================\n");
    if (validCount > 0) {
      double geoMean = std::exp(logSum / validCount);
      fprintf(
          stderr,
          "  Geometric Mean Speedup (Pipes/NCCL A2a): %.3fx (%d sizes)\n",
          geoMean,
          validCount);
      fprintf(
          stderr,
          "  Best Speedup:  %.2fx at %s\n",
          bestSpeedup,
          formatBytes(bestMsg).c_str());
      fprintf(
          stderr,
          "  Worst Speedup: %.2fx at %s\n",
          worstSpeedup,
          formatBytes(worstMsg).c_str());
    }
    fprintf(
        stderr,
        "  BW = Algorithm bandwidth (total data / time), in GB/s"
        " — nccl-tests convention\n");
    fprintf(stderr, "  BusBW = Bus bandwidth = AlgBW × (nRanks-1)/nRanks\n");
    fprintf(
        stderr,
        "  N/A = NCCL AllToAll skipped (buffer would exceed %s cap)\n",
        formatBytes(kMaxBenchmarkBufSize).c_str());
    fprintf(
        stderr,
        "============================================================"
        "============================================================\n\n");
  }

  // ─── Compute busBW from algoBW ─────────────────────────────────────────
  double computeBusBW(double algoBW) const {
    return algoBW * static_cast<double>(worldSize - 1) /
        static_cast<double>(worldSize);
  }

  TopologyMode topoMode_{TopologyMode::NVL_ONLY};
  ncclComm_t ncclComm_{};
  cudaStream_t stream_{};
};

// ═══════════════════════════════════════════════════════════════════════════
// Sweep benchmark: NCCL AllToAll vs Pipes AllToAllv
//
// Runs NCCL AllToAll and Pipes AllToAllv at every message size from 1KB to
// 256MB, using autotune-derived optimal kernel parameters. Reports
// per-message algoBW, busBW, and speedups.
//
// Bandwidth conventions follow nccl-tests:
//   algoBW = totalData / time  (send direction only, 1x)
//   busBW  = algoBW × (nRanks - 1) / nRanks
//
// Launch examples:
//   Single config:
//     buck2 run
//     //comms/pipes/collectives/benchmarks:alltoallv_pipes_sweep_benchmark_2x8
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(AllToAllvPipesSweepFixture, NcclVsPipesSweep) {
  auto msgSizes = allMessageSizes();
  bool isHybrid = (topoMode_ == TopologyMode::HYBRID);

  auto bootstrapPtr = std::shared_ptr<meta::comms::IBootstrap>(
      bootstrap.get(), [](meta::comms::IBootstrap*) {});

  // Transport init config from autotune defaults
  AllToAllvAutoTuneConfig autoConfig;
  MultiPeerTransportConfig cfg{};
  cfg.ibgdaConfig.cudaDevice = localRank;

  if (isHybrid) {
    // Hybrid NVL config — must match the init config the autotune sweep
    // was run with (see PIPES_HYBRID_CONFIG_ENTRIES comment in
    // AllToAllvAutoTuneConfig.h: nvlChunk=32KB nvlBuf=2MB nvlPipe=2)
    cfg.nvlConfig.dataBufferSize = 2 * 1024 * 1024; // 2MB
    cfg.nvlConfig.chunkSize = 32 * 1024; // 32KB
    cfg.nvlConfig.pipelineDepth = 2;
    // IBGDA config from autotune defaults
    cfg.ibgdaSetupConfig.dataBufferSize =
        autoConfig.ibgdaInit.stagingBufferSize;
    cfg.ibgdaSetupConfig.chunkSize = autoConfig.ibgdaInit.ibgdaChunkSize;
    cfg.ibgdaSetupConfig.pipelineDepth = autoConfig.ibgdaInit.pipelineDepth;
  } else {
    // NVL-only config from autotune defaults
    cfg.nvlConfig.dataBufferSize = autoConfig.nvlInit.nvlDataBufferSize;
    cfg.nvlConfig.chunkSize = autoConfig.nvlInit.nvlChunkSize;
    cfg.nvlConfig.pipelineDepth = autoConfig.nvlInit.pipelineDepth;
    cfg.disableIb = true;
  }

  std::unique_ptr<MultiPeerTransport> transport;
  try {
    transport = std::make_unique<MultiPeerTransport>(
        globalRank, worldSize, localRank, bootstrapPtr, cfg);
    transport->exchange();
  } catch (const std::exception& e) {
    GTEST_SKIP() << "Transport setup failed: " << e.what();
  }

  auto deviceHandle = transport->get_device_handle();
  DeviceSpan<Transport> transports(
      deviceHandle.transports.data(), deviceHandle.transports.size());

  if (globalRank == 0) {
    int nnodes = worldSize / localSize;
    if (isHybrid) {
      XLOGF(
          INFO,
          "[SWEEP] Topology: hybrid (NVLink + IBGDA), {} nodes x {} GPUs = "
          "{} ranks, {} NVL peers, {} IBGDA peers",
          nnodes,
          localSize,
          worldSize,
          deviceHandle.numNvlPeers,
          deviceHandle.numIbPeers);
    } else {
      XLOGF(
          INFO,
          "[SWEEP] Topology: NVLink-only, {} nodes x {} GPUs = {} ranks",
          nnodes,
          localSize,
          worldSize);
    }
  }

  std::vector<SweepResult> results;

  for (std::size_t msgSize : msgSizes) {
    int nIter = iterCountForMsgSize(msgSize);

    // Get autotune kernel params for this message size
    int numBlocks, numThreads;
    WarpReserveDeviceConfig reserveDevCfg{};

    if (isHybrid) {
      auto hybridCfg = getHybridConfigForMsgSize(msgSize);
      numBlocks = hybridCfg.numBlocks;
      numThreads = hybridCfg.numThreads;

      // Warp reservation configs were tuned for 2x8=16 ranks. At different
      // world sizes, the fixed warp counts don't distribute correctly across
      // the different number of IBGDA/NVL peers (e.g., ibgdaSendWarps=8
      // for 248 IBGDA peers → most peers get zero warps). Fall back to
      // auto-partition (all zeros) when topology doesn't match.
      bool warpReserveTuned = (worldSize == 16 && localSize == 8);
      WarpReserveConfig wr{
          warpReserveTuned ? hybridCfg.nvlSendWarps : 0,
          warpReserveTuned ? hybridCfg.nvlRecvWarps : 0,
          warpReserveTuned ? hybridCfg.ibgdaSendWarps : 0,
          warpReserveTuned ? hybridCfg.ibgdaRecvWarps : 0,
          warpReserveTuned ? hybridCfg.selfWarps : 0};
      reserveDevCfg = resolveWarpReserve(
          wr,
          deviceHandle.numNvlPeers,
          deviceHandle.numIbPeers,
          deviceHandle.nvlPeerRanks.data(),
          deviceHandle.ibgdaPeerRanks.data());
    } else {
      auto nvlCfg = getNvlConfigForMsgSize(msgSize);
      numBlocks = nvlCfg.numBlocks;
      numThreads = nvlCfg.numThreads;
    }

    // Run NCCL AllToAll benchmark (equal-count)
    double ncclA2aLat = 0;
    bool ncclA2aSkipped = false;
    double ncclA2aAlgoBW =
        runNcclAllToAllBenchmark(msgSize, nIter, ncclA2aLat, ncclA2aSkipped);
    double ncclA2aBusBW = computeBusBW(ncclA2aAlgoBW);

    // Run Pipes benchmark
    // Buffer cap with modular wrapping: this is a performance benchmark
    // measuring communication latency/throughput, not data correctness.
    // Overlapping buffer regions are acceptable when the full
    // msgSize * worldSize allocation would exceed GPU memory.
    const std::size_t logicalTotalBytes = msgSize * worldSize;
    const std::size_t totalBytes =
        std::max(msgSize, std::min(logicalTotalBytes, kMaxBenchmarkBufSize));
    DeviceBuffer sendBuffer(totalBytes);
    DeviceBuffer recvBuffer(totalBytes);
    CUDA_CHECK_SWEEP(
        cudaMemset(sendBuffer.get(), globalRank & 0xFF, totalBytes));
    CUDA_CHECK_SWEEP(cudaMemset(recvBuffer.get(), 0, totalBytes));

    std::vector<ChunkInfo> h_chunks;
    h_chunks.reserve(worldSize);
    for (int r = 0; r < worldSize; ++r) {
      h_chunks.emplace_back((r * msgSize) % totalBytes, msgSize);
    }
    DeviceBuffer d_send(sizeof(ChunkInfo) * worldSize);
    DeviceBuffer d_recv(sizeof(ChunkInfo) * worldSize);
    CUDA_CHECK_SWEEP(cudaMemcpy(
        d_send.get(),
        h_chunks.data(),
        sizeof(ChunkInfo) * worldSize,
        cudaMemcpyHostToDevice));
    CUDA_CHECK_SWEEP(cudaMemcpy(
        d_recv.get(),
        h_chunks.data(),
        sizeof(ChunkInfo) * worldSize,
        cudaMemcpyHostToDevice));

    DeviceSpan<ChunkInfo> sendInfos(
        static_cast<ChunkInfo*>(d_send.get()), worldSize);
    DeviceSpan<ChunkInfo> recvInfos(
        static_cast<ChunkInfo*>(d_recv.get()), worldSize);

    CudaEvent start, stop;

    // Warmup
    bootstrap->barrierAll();
    for (int i = 0; i < 5; ++i) {
      if (isHybrid) {
        all_to_allv(
            recvBuffer.get(),
            sendBuffer.get(),
            globalRank,
            transports,
            sendInfos,
            recvInfos,
            std::chrono::milliseconds{0},
            stream_,
            numBlocks,
            numThreads,
            std::nullopt,
            reserveDevCfg);
      } else {
        all_to_allv(
            recvBuffer.get(),
            sendBuffer.get(),
            globalRank,
            transports,
            sendInfos,
            recvInfos,
            std::chrono::milliseconds{0},
            stream_,
            numBlocks,
            numThreads,
            std::nullopt);
      }
    }
    CUDA_CHECK_SWEEP(cudaStreamSynchronize(stream_));
    bootstrap->barrierAll();

    // Timed run
    CUDA_CHECK_SWEEP(cudaEventRecord(start.get(), stream_));
    for (int i = 0; i < nIter; ++i) {
      if (isHybrid) {
        all_to_allv(
            recvBuffer.get(),
            sendBuffer.get(),
            globalRank,
            transports,
            sendInfos,
            recvInfos,
            std::chrono::milliseconds{0},
            stream_,
            numBlocks,
            numThreads,
            std::nullopt,
            reserveDevCfg);
      } else {
        all_to_allv(
            recvBuffer.get(),
            sendBuffer.get(),
            globalRank,
            transports,
            sendInfos,
            recvInfos,
            std::chrono::milliseconds{0},
            stream_,
            numBlocks,
            numThreads,
            std::nullopt);
      }
    }
    CUDA_CHECK_SWEEP(cudaEventRecord(stop.get(), stream_));
    CUDA_CHECK_SWEEP(cudaStreamSynchronize(stream_));

    float totalMs = 0;
    CUDA_CHECK_SWEEP(cudaEventElapsedTime(&totalMs, start.get(), stop.get()));
    float avgMs = totalMs / nIter;
    double pipesLat = avgMs * 1000.0;
    // nccl-tests convention: send direction only (1x total data)
    double pipesAlgoBW =
        (logicalTotalBytes / (1000.0 * 1000.0 * 1000.0)) / (avgMs / 1000.0);
    double pipesBusBW = computeBusBW(pipesAlgoBW);
    double speedupVsA2a = (ncclA2aAlgoBW > 0) ? pipesAlgoBW / ncclA2aAlgoBW : 0;

    if (globalRank == 0) {
      results.push_back(
          SweepResult{
              msgSize,
              numBlocks,
              numThreads,
              ncclA2aLat,
              ncclA2aAlgoBW,
              ncclA2aBusBW,
              ncclA2aSkipped,
              pipesLat,
              pipesAlgoBW,
              pipesBusBW,
              speedupVsA2a});

      XLOGF(
          INFO,
          "[SWEEP] msgSize={} blocks={} threads={} -> "
          "A2a: {:.1f}us {:.2f}GB/s, "
          "Pipes: {:.1f}us {:.2f}GB/s, "
          "vsA2a={:.2f}x",
          formatBytes(msgSize),
          numBlocks,
          numThreads,
          ncclA2aLat,
          ncclA2aAlgoBW,
          pipesLat,
          pipesAlgoBW,
          speedupVsA2a);
    }

    bootstrap->barrierAll();
  }

  printResults(results);
}

} // namespace

} // namespace comms::pipes::benchmark

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  ::testing::AddGlobalTestEnvironment(new meta::comms::BenchmarkEnvironment());
  return RUN_ALL_TESTS();
}
