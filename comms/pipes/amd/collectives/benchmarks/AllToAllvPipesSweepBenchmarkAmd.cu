#if defined(__HIPCC__) || !defined(__CUDACC__)
// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// AMD port of
// comms/pipes/collectives/benchmarks/AllToAllvPipesSweepBenchmark.cc
//
// Sweep benchmark comparing RCCL AllToAllv vs Pipes AllToAllv on AMD GPUs.
// Runs both at every power-of-2 message size from 1KB to 32MB,
// reporting per-message latency, algorithm bandwidth, and speedup.
//
// Key AMD adaptations:
// - HIP APIs instead of CUDA
// - RCCL ncclAllToAllv instead of NCCL ncclAllToAll
// - MultiPeerNvlTransportAmd instead of MultiPeerTransport
// - HipDeviceBuffer instead of CudaRAII DeviceBuffer
// - Raw hipEvent_t instead of CudaEvent RAII
// - std::nullopt for cluster_dim (no clusters on AMD)
// - NVL_ONLY mode only (no hybrid/IBGDA yet)
// - Fixed default kernel params (no autotune config for AMD yet)

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include "HipDeviceBuffer.h" // @manual
#include "MultipeerIbgdaTransportAmd.h" // @manual
#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/IbgdaBuffer.h"
#include "comms/pipes/Transport.cuh"
#include "comms/pipes/amd/MultiPeerTransportAmd.h"
#include "comms/pipes/amd/collectives/AllToAllvAmd.h"
#include "comms/pipes/collectives/AllToAllv.h"
#include "comms/testinfra/BenchmarkTestFixture.h"

#define HIPCHECK_SWEEP(cmd)                                \
  do {                                                     \
    hipError_t err = (cmd);                                \
    if (err != hipSuccess) {                               \
      XLOGF(ERR, "HIP error: {}", hipGetErrorString(err)); \
      std::abort();                                        \
    }                                                      \
  } while (0)

#define RCCL_CHECK_SWEEP(cmd)                                \
  do {                                                       \
    ncclResult_t res = (cmd);                                \
    if (res != ncclSuccess) {                                \
      XLOGF(ERR, "RCCL error: {}", ncclGetErrorString(res)); \
      std::abort();                                          \
    }                                                        \
  } while (0)

using pipes_gda::HipDeviceBuffer;

namespace comms::pipes::benchmark {

namespace {

enum class TopologyMode {
  NVL_ONLY,
  HYBRID,
};

// Cap GPU buffer allocations to avoid OOM.
// RCCL ncclAllToAllv needs linear buffers (msgSize × worldSize), so this cap
// determines the max msg size that AllToAllv can benchmark at a given
// world size. Pipes AllToAllv uses modular wrapping so it always works.
// 32 GB allows AllToAllv up to:
//   8 ranks (1×8):   4GB/peer  (4GB × 8 = 32GB)
//   16 ranks (2×8):  2GB/peer  (2GB × 16 = 32GB)
constexpr std::size_t kMaxBenchmarkBufSize = 32ULL * 1024 * 1024 * 1024;

// NVL transport init params — match NVIDIA NvlInitConfig defaults.
// These were auto-tuned on 1x8 H100 NVLink exhaustive sweep.
constexpr std::size_t kDefaultDataBufferSize = 8ULL * 1024 * 1024; // 8MB
constexpr std::size_t kDefaultChunkSize = 32 * 1024; // 32KB
constexpr std::size_t kDefaultPipelineDepth = 4;

// Per-message-size kernel configs — exact match of NVIDIA kDefaultNvlConfigs[]
// from AllToAllvAutoTuneConfig.h, auto-tuned on 1x8 H100 NVLink.
struct NvlPerMsgConfig {
  std::size_t maxMsgSize;
  int numBlocks;
  int numThreads;
};

constexpr NvlPerMsgConfig kNvlConfigs[] = {
    {1024, 4, 128}, // 1KB
    {2 * 1024, 4, 128}, // 2KB
    {4 * 1024, 4, 128}, // 4KB
    {8 * 1024, 4, 128}, // 8KB
    {16 * 1024, 64, 128}, // 16KB
    {32 * 1024, 32, 128}, // 32KB
    {64 * 1024, 32, 128}, // 64KB
    {128 * 1024, 16, 128}, // 128KB
    {256 * 1024, 64, 128}, // 256KB
    {512 * 1024, 64, 128}, // 512KB
    {1024 * 1024, 64, 256}, // 1MB
    {2 * 1024 * 1024, 64, 256}, // 2MB
    {4 * 1024 * 1024, 64, 256}, // 4MB
    {8 * 1024 * 1024, 64, 256}, // 8MB
    {16 * 1024 * 1024, 64, 256}, // 16MB
    {32 * 1024 * 1024, 64, 256}, // 32MB
};

NvlPerMsgConfig get_nvl_config_for_msg_size(std::size_t msgSize) {
  for (const auto& cfg : kNvlConfigs) {
    if (msgSize <= cfg.maxMsgSize) {
      return cfg;
    }
  }
  // Fallback for sizes > 32MB
  return {msgSize, 64, 256};
}

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
  // RCCL AllToAllv baseline
  double rcclLatencyUs;
  double rcclAlgoBW;
  double rcclBusBW;
  bool rcclSkipped; // true when buffer exceeds cap
  // Pipes AllToAllv
  double pipesLatencyUs;
  double pipesAlgoBW;
  double pipesBusBW;
  // Speedup
  double speedupVsRccl;
};

class AllToAllvPipesSweepAmdFixture : public meta::comms::BenchmarkTestFixture {
 protected:
  void SetUp() override {
    BenchmarkTestFixture::SetUp();
    HIPCHECK_SWEEP(hipSetDevice(localRank));
    HIPCHECK_SWEEP(hipStreamCreate(&stream_));

    // Suppress verbose RCCL INFO logs (proxy connections, buffer imports, etc.)
    // unless the user explicitly set NCCL_DEBUG.
    if (!getenv("NCCL_DEBUG")) {
      setenv("NCCL_DEBUG", "WARN", 0);
    }

    ncclUniqueId id;
    if (globalRank == 0) {
      RCCL_CHECK_SWEEP(ncclGetUniqueId(&id));
    }
    std::vector<ncclUniqueId> allIds(worldSize);
    allIds[globalRank] = id;
    bootstrap
        ->allGather(allIds.data(), sizeof(ncclUniqueId), globalRank, worldSize)
        .get();
    id = allIds[0];
    RCCL_CHECK_SWEEP(ncclCommInitRank(&ncclComm_, worldSize, id, globalRank));

    bool isSingleNode = (localSize == worldSize);
    topoMode_ = isSingleNode ? TopologyMode::NVL_ONLY : TopologyMode::HYBRID;
  }

  void TearDown() override {
    RCCL_CHECK_SWEEP(ncclCommDestroy(ncclComm_));
    HIPCHECK_SWEEP(hipStreamDestroy(stream_));
    BenchmarkTestFixture::TearDown();
  }

  // ─── RCCL AllToAll benchmark (equal-count, matches NVIDIA ncclAllToAll) ──
  double runRcclAllToAllBenchmark(
      std::size_t bytesPerPeer,
      int nIter,
      double& latencyUs,
      bool& skipped) {
    const std::size_t logicalTotalBytes = bytesPerPeer * worldSize;
    if (logicalTotalBytes > kMaxBenchmarkBufSize) {
      latencyUs = 0;
      skipped = true;
      return 0;
    }
    skipped = false;
    HipDeviceBuffer sendBuffer(logicalTotalBytes);
    HipDeviceBuffer recvBuffer(logicalTotalBytes);
    HIPCHECK_SWEEP(
        hipMemset(sendBuffer.get(), globalRank & 0xFF, logicalTotalBytes));
    HIPCHECK_SWEEP(hipMemset(recvBuffer.get(), 0, logicalTotalBytes));

    size_t count = bytesPerPeer;

    hipEvent_t start, stop;
    HIPCHECK_SWEEP(hipEventCreate(&start));
    HIPCHECK_SWEEP(hipEventCreate(&stop));

    bootstrap->barrierAll();
    for (int i = 0; i < 5; ++i) {
      RCCL_CHECK_SWEEP(ncclAllToAll(
          sendBuffer.get(),
          recvBuffer.get(),
          count,
          ncclChar,
          ncclComm_,
          stream_));
    }
    HIPCHECK_SWEEP(hipStreamSynchronize(stream_));
    bootstrap->barrierAll();

    HIPCHECK_SWEEP(hipEventRecord(start, stream_));
    for (int i = 0; i < nIter; ++i) {
      RCCL_CHECK_SWEEP(ncclAllToAll(
          sendBuffer.get(),
          recvBuffer.get(),
          count,
          ncclChar,
          ncclComm_,
          stream_));
    }
    HIPCHECK_SWEEP(hipEventRecord(stop, stream_));
    HIPCHECK_SWEEP(hipStreamSynchronize(stream_));

    float totalMs = 0;
    HIPCHECK_SWEEP(hipEventElapsedTime(&totalMs, start, stop));
    float avgMs = totalMs / nIter;
    latencyUs = avgMs * 1000.0;
    double algoBW =
        (logicalTotalBytes / (1000.0 * 1000.0 * 1000.0)) / (avgMs / 1000.0);

    HIPCHECK_SWEEP(hipEventDestroy(start));
    HIPCHECK_SWEEP(hipEventDestroy(stop));

    return algoBW;
  }

  // ─── Print sweep results ──────────────────────────────────────────────
  void printResults(const std::vector<SweepResult>& results) {
    if (globalRank != 0 || results.empty()) {
      return;
    }

    int nnodes = worldSize / localSize;
    const char* topoStr = (topoMode_ == TopologyMode::HYBRID)
        ? "hybrid (NVLink + IBGDA, AMD)"
        : "NVLink-only (AMD)";

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
        "RCCL Lat",
        "RCCLAlgBW",
        "RCCLBusBW",
        "Pipes Lat",
        "PipesAlgBW",
        "PipesBusBW",
        "vs RCCL");
    fprintf(
        stderr,
        "  -------- ---- ---- | -------- --------- --------- "
        "| --------- ---------- ---------- | -------\n");

    double logSum = 0;
    int validCount = 0;
    double bestSpeedup = 0, worstSpeedup = 1e9;
    std::size_t bestMsg = 0, worstMsg = 0;

    for (const auto& r : results) {
      bool pipesSkipped = (r.speedupVsRccl == 0.0 && r.pipesLatencyUs == 0.0);
      if (r.rcclSkipped && pipesSkipped) {
        fprintf(
            stderr,
            "  %-8s %4d %4d | %8s %9s %9s | %9s %10s %10s | %7s\n",
            formatBytes(r.msgSize).c_str(),
            r.numBlocks,
            r.numThreads,
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A");
      } else if (r.rcclSkipped) {
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
      } else if (pipesSkipped) {
        fprintf(
            stderr,
            "  %-8s %4d %4d | %8.1f %9.2f %9.2f | %9s %10s %10s | %7s\n",
            formatBytes(r.msgSize).c_str(),
            r.numBlocks,
            r.numThreads,
            r.rcclLatencyUs,
            r.rcclAlgoBW,
            r.rcclBusBW,
            "N/A",
            "N/A",
            "N/A",
            "N/A");
      } else {
        fprintf(
            stderr,
            "  %-8s %4d %4d | %8.1f %9.2f %9.2f | %9.1f %10.2f %10.2f | %6.2fx\n",
            formatBytes(r.msgSize).c_str(),
            r.numBlocks,
            r.numThreads,
            r.rcclLatencyUs,
            r.rcclAlgoBW,
            r.rcclBusBW,
            r.pipesLatencyUs,
            r.pipesAlgoBW,
            r.pipesBusBW,
            r.speedupVsRccl);

        auto safeLog = [](double v) { return std::log(v > 0.01 ? v : 0.01); };
        logSum += safeLog(r.speedupVsRccl);
        ++validCount;

        if (r.speedupVsRccl > bestSpeedup) {
          bestSpeedup = r.speedupVsRccl;
          bestMsg = r.msgSize;
        }
        if (r.speedupVsRccl < worstSpeedup) {
          worstSpeedup = r.speedupVsRccl;
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
          "  Geometric Mean Speedup (Pipes/RCCL): %.3fx (%d sizes)\n",
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
        "  N/A = RCCL AllToAllv skipped (buffer would exceed %s cap)\n",
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
  hipStream_t stream_{};
};

// ═══════════════════════════════════════════════════════════════════════════
// Sweep benchmark: RCCL AllToAllv vs Pipes AllToAllv (AMD)
//
// Runs RCCL AllToAllv and Pipes AllToAllv at every message size from 1KB to
// 32MB, using fixed default kernel parameters. Reports per-message algoBW,
// busBW, and speedups.
//
// Bandwidth conventions follow nccl-tests:
//   algoBW = totalData / time  (send direction only, 1x)
//   busBW  = algoBW × (nRanks - 1) / nRanks
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(AllToAllvPipesSweepAmdFixture, RcclVsPipesSweep) {
  auto msgSizes = allMessageSizes();

  auto bootstrapPtr = std::shared_ptr<meta::comms::IBootstrap>(
      bootstrap.get(), [](meta::comms::IBootstrap*) {});

  // Transport init config with fixed defaults
  MultiPeerTransportAmdConfig cfg{
      .nvlConfig =
          {.dataBufferSize = kDefaultDataBufferSize,
           .chunkSize = kDefaultChunkSize,
           .pipelineDepth = kDefaultPipelineDepth},
      .hipDevice = localRank,
  };

  // Unified transport: NVL for intra-node, IBGDA for inter-node
  bool pipesAvailable = false;
  std::unique_ptr<MultiPeerTransportAmd> transport;
  try {
    transport = std::make_unique<MultiPeerTransportAmd>(
        globalRank, worldSize, localRank, localSize, bootstrapPtr, cfg);
    transport->exchange();
    pipesAvailable = true;
  } catch (const std::exception& e) {
    if (globalRank == 0) {
      XLOGF(
          INFO,
          "[SWEEP] Pipes transport unavailable ({}), running RCCL-only sweep",
          e.what());
    }
  }

  bool isHybrid = (topoMode_ == TopologyMode::HYBRID);

  if (globalRank == 0) {
    int nnodes = worldSize / localSize;
    const char* topoStr =
        isHybrid ? "hybrid (NVLink + IBGDA, AMD)" : "NVLink-only (AMD)";
    XLOGF(
        INFO,
        "[SWEEP] Topology: {}, {} nodes x {} GPUs = {} ranks{}",
        topoStr,
        nnodes,
        localSize,
        worldSize,
        pipesAvailable ? "" : " [RCCL-only]");
  }

  std::vector<SweepResult> results;

  for (std::size_t msgSize : msgSizes) {
    int nIter = iterCountForMsgSize(msgSize);

    // Per-message-size kernel config — matches NVIDIA autotune defaults,
    // adjusted for AMD wavefront size (64 vs NVIDIA warp size 32).
    // partition_interleaved(2) × partition_interleaved(nranks) needs at least
    // 2 × nranks wavefront groups = 2 × nranks × 64 total threads.
    auto nvlCfg = get_nvl_config_for_msg_size(msgSize);
    int numBlocks = nvlCfg.numBlocks;
    int numThreads = nvlCfg.numThreads;
    constexpr int kWavefrontSize = 64;
    int minTotalThreads = 2 * worldSize * kWavefrontSize;
    while (numBlocks * numThreads < minTotalThreads) {
      numBlocks *= 2;
    }

    // Run RCCL AllToAllv benchmark
    double rcclLat = 0;
    bool rcclSkipped = false;
    double rcclAlgoBW =
        runRcclAllToAllBenchmark(msgSize, nIter, rcclLat, rcclSkipped);
    double rcclBusBW = computeBusBW(rcclAlgoBW);

    double pipesLat = 0;
    double pipesAlgoBW = 0;
    double pipesBusBW = 0;
    double speedupVsRccl = 0;
    bool pipesSkipped = !pipesAvailable;

    if (pipesAvailable) {
      auto transports = transport->getDeviceTransports();
      // Run Pipes benchmark
      const std::size_t logicalTotalBytes = msgSize * worldSize;
      std::size_t cappedBytes = logicalTotalBytes < kMaxBenchmarkBufSize
          ? logicalTotalBytes
          : kMaxBenchmarkBufSize;
      const std::size_t totalBytes =
          msgSize > cappedBytes ? msgSize : cappedBytes;
      HipDeviceBuffer sendBuffer(totalBytes);
      HipDeviceBuffer recvBuffer(totalBytes);
      HIPCHECK_SWEEP(
          hipMemset(sendBuffer.get(), globalRank & 0xFF, totalBytes));
      HIPCHECK_SWEEP(hipMemset(recvBuffer.get(), 0, totalBytes));

      std::vector<ChunkInfo> h_chunks;
      h_chunks.reserve(worldSize);
      for (int r = 0; r < worldSize; ++r) {
        h_chunks.emplace_back((r * msgSize) % totalBytes, msgSize);
      }
      HipDeviceBuffer d_send(sizeof(ChunkInfo) * worldSize);
      HipDeviceBuffer d_recv(sizeof(ChunkInfo) * worldSize);
      HIPCHECK_SWEEP(hipMemcpy(
          d_send.get(),
          h_chunks.data(),
          sizeof(ChunkInfo) * worldSize,
          hipMemcpyHostToDevice));
      HIPCHECK_SWEEP(hipMemcpy(
          d_recv.get(),
          h_chunks.data(),
          sizeof(ChunkInfo) * worldSize,
          hipMemcpyHostToDevice));

      DeviceSpan<ChunkInfo> sendInfos(
          static_cast<ChunkInfo*>(d_send.get()), worldSize);
      DeviceSpan<ChunkInfo> recvInfos(
          static_cast<ChunkInfo*>(d_recv.get()), worldSize);

      // IBGDA buffer registration for multi-node (per message size)
      void* ibgdaTransportBase = nullptr;
      std::size_t ibgdaTransportStride = 0;
      int numIbgdaPeers = 0;
      int* d_ibgdaPeerRanksPtr = nullptr;
      IbgdaRemoteBuffer* d_ibgdaRecvBufsPtr = nullptr;
      IbgdaLocalBuffer ibgdaSendBuf;
      std::unique_ptr<HipDeviceBuffer> d_ibgdaPeerRanksBuf;
      std::unique_ptr<HipDeviceBuffer> d_ibgdaRecvBufsBuf;

      if (isHybrid) {
        auto* ibgdaTransport = transport->getIbgdaTransport();
        const auto& remotePeers = transport->getRemotePeerRanks();
        numIbgdaPeers = static_cast<int>(remotePeers.size());

        ibgdaSendBuf =
            ibgdaTransport->registerBuffer(sendBuffer.get(), totalBytes);
        auto remoteRecvBufs = ibgdaTransport->exchangeBuffer(
            ibgdaTransport->registerBuffer(recvBuffer.get(), totalBytes));

        ibgdaTransportBase =
            static_cast<void*>(ibgdaTransport->getDeviceTransportPtr());
        if (numIbgdaPeers >= 2) {
          auto* p0 = ibgdaTransport->getP2pTransportDevice(remotePeers[0]);
          auto* p1 = ibgdaTransport->getP2pTransportDevice(remotePeers[1]);
          ibgdaTransportStride =
              reinterpret_cast<char*>(p1) - reinterpret_cast<char*>(p0);
        }

        // Copy peer ranks and remote buffer descriptors to device
        d_ibgdaPeerRanksBuf =
            std::make_unique<HipDeviceBuffer>(numIbgdaPeers * sizeof(int));
        HIPCHECK_SWEEP(hipMemcpy(
            d_ibgdaPeerRanksBuf->get(),
            remotePeers.data(),
            numIbgdaPeers * sizeof(int),
            hipMemcpyHostToDevice));
        d_ibgdaPeerRanksPtr = static_cast<int*>(d_ibgdaPeerRanksBuf->get());

        d_ibgdaRecvBufsBuf = std::make_unique<HipDeviceBuffer>(
            numIbgdaPeers * sizeof(IbgdaRemoteBuffer));
        HIPCHECK_SWEEP(hipMemcpy(
            d_ibgdaRecvBufsBuf->get(),
            remoteRecvBufs.data(),
            numIbgdaPeers * sizeof(IbgdaRemoteBuffer),
            hipMemcpyHostToDevice));
        d_ibgdaRecvBufsPtr =
            static_cast<IbgdaRemoteBuffer*>(d_ibgdaRecvBufsBuf->get());
      }

      hipEvent_t start, stop;
      HIPCHECK_SWEEP(hipEventCreate(&start));
      HIPCHECK_SWEEP(hipEventCreate(&stop));

      auto dispatch_collective = [&]() {
        if (isHybrid) {
          all_to_allv_amd(
              recvBuffer.get(),
              sendBuffer.get(),
              globalRank,
              transports,
              sendInfos,
              recvInfos,
              ibgdaTransportBase,
              ibgdaTransportStride,
              d_ibgdaPeerRanksPtr,
              numIbgdaPeers,
              ibgdaSendBuf,
              d_ibgdaRecvBufsPtr,
              stream_,
              numBlocks,
              numThreads);
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
      };

      // Warmup
      bootstrap->barrierAll();
      for (int i = 0; i < 5; ++i) {
        dispatch_collective();
      }
      HIPCHECK_SWEEP(hipStreamSynchronize(stream_));
      bootstrap->barrierAll();

      // Timed run
      HIPCHECK_SWEEP(hipEventRecord(start, stream_));
      for (int i = 0; i < nIter; ++i) {
        dispatch_collective();
      }
      HIPCHECK_SWEEP(hipEventRecord(stop, stream_));
      HIPCHECK_SWEEP(hipStreamSynchronize(stream_));

      float totalMs = 0;
      HIPCHECK_SWEEP(hipEventElapsedTime(&totalMs, start, stop));
      float avgMs = totalMs / nIter;
      pipesLat = avgMs * 1000.0;
      const std::size_t logicalTotal = msgSize * worldSize;
      pipesAlgoBW =
          (logicalTotal / (1000.0 * 1000.0 * 1000.0)) / (avgMs / 1000.0);

      // Deregister IBGDA buffers for this iteration
      if (isHybrid) {
        auto* ibgdaTransport = transport->getIbgdaTransport();
        ibgdaTransport->deregisterBuffer(sendBuffer.get());
        ibgdaTransport->deregisterBuffer(recvBuffer.get());
      }
      pipesBusBW = computeBusBW(pipesAlgoBW);
      speedupVsRccl = (rcclAlgoBW > 0) ? pipesAlgoBW / rcclAlgoBW : 0;

      HIPCHECK_SWEEP(hipEventDestroy(start));
      HIPCHECK_SWEEP(hipEventDestroy(stop));
    }

    if (globalRank == 0) {
      results.push_back(
          SweepResult{
              msgSize,
              numBlocks,
              numThreads,
              rcclLat,
              rcclAlgoBW,
              rcclBusBW,
              rcclSkipped,
              pipesLat,
              pipesAlgoBW,
              pipesBusBW,
              pipesSkipped ? 0.0 : speedupVsRccl});

      if (pipesSkipped) {
        XLOGF(
            INFO,
            "[SWEEP] msgSize={} -> RCCL: {:.1f}us {:.2f}GB/s (Pipes: N/A)",
            formatBytes(msgSize),
            rcclLat,
            rcclAlgoBW);
      } else {
        XLOGF(
            INFO,
            "[SWEEP] msgSize={} blocks={} threads={} -> "
            "RCCL: {:.1f}us {:.2f}GB/s, "
            "Pipes: {:.1f}us {:.2f}GB/s, "
            "vsRCCL={:.2f}x",
            formatBytes(msgSize),
            numBlocks,
            numThreads,
            rcclLat,
            rcclAlgoBW,
            pipesLat,
            pipesAlgoBW,
            speedupVsRccl);
      }
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

#endif // defined(__HIPCC__) || !defined(__CUDACC__)
