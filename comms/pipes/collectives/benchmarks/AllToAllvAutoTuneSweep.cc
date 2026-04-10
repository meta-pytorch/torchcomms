// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <nccl.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
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
#include "comms/testinfra/BenchmarkTestFixture.h"
#include "comms/testinfra/mpi/MpiTestUtils.h" // @manual BenchmarkEnvironment
#include "comms/utils/CudaRAII.h"

#define CUDACHECK_SWEEP(cmd)                                 \
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
  for (std::size_t s = 1024; s <= 256ULL * 1024 * 1024; s *= 2) {
    sizes.push_back(s);
  }
  return sizes;
}

// Stratified message size ordering: round-robin across small/medium/large
// strata so that early termination decisions reflect all regimes, not just
// small messages.  This prevents the bias where chunk=64KB looks great at
// small sizes but collapses at large sizes — we catch it after just 3 evals.
std::vector<std::size_t> stratifiedMessageSizes() {
  return {
      1024, // 1KB   (small)
      256ULL * 1024 * 1024, // 256MB (large)
      4 * 1024 * 1024, // 4MB   (medium)
      64 * 1024, // 64KB  (small)
      64 * 1024 * 1024, // 64MB  (large)
      512 * 1024, // 512KB (medium)
      8 * 1024, // 8KB   (small)
      32 * 1024 * 1024, // 32MB  (large)
      1024 * 1024, // 1MB   (medium)
      4 * 1024, // 4KB   (small)
      128 * 1024 * 1024, // 128MB (large)
      128 * 1024, // 128KB (medium)
      16 * 1024, // 16KB  (small)
      16 * 1024 * 1024, // 16MB  (large)
      2 * 1024 * 1024, // 2MB   (medium)
      32 * 1024, // 32KB  (small)
      8 * 1024 * 1024, // 8MB   (large)
      256 * 1024, // 256KB (medium)
      2 * 1024, // 2KB   (small)
  };
}

struct PerMsgBest {
  std::size_t msgSize;
  int numBlocks;
  int numThreads;
  double latencyUs;
  double bandwidthGBps;
  double ncclLatencyUs;
  double ncclBandwidthGBps;
  double speedupVsNccl;
  int nvlSendWarps;
  int nvlRecvWarps;
  int ibgdaSendWarps;
  int ibgdaRecvWarps;
  int selfWarps;
};

using PerInitConfigResults =
    std::map<std::string, std::map<std::size_t, PerMsgBest>>;

struct InitTimeParams {
  std::size_t nvlChunkSize;
  std::size_t nvlDataBufSize;
  int nvlPipeDepth;
  std::size_t ibgdaStagingBufSize;
  int ibgdaPipeDepth;
  std::size_t ibgdaChunkSize;
};

// Generate warp reserve profiles scaled to actual peer counts.
// For hybrid, we use only 2 profiles (auto + 2×NVL/1×IBGDA) to keep
// the kernel combo count at 30 instead of 75, since IBGDA is
// latency-bound and insensitive to warp allocation.
std::vector<WarpReserveConfig>
hybridWarpReserveProfiles(int nNvl, int nIb, int totalWarps) {
  std::vector<WarpReserveConfig> profiles;
  profiles.push_back({0, 0, 0, 0, 0}); // auto (uniform partition)

  // 2× NVL, 1× IBGDA (matches auto defaults for the NVL-sensitive portion)
  int ns = 2 * nNvl;
  int nr = 2 * nNvl;
  int is = 1 * nIb;
  int ir = 1 * nIb;
  int selfW = 1;
  if (selfW + ns + nr + is + ir <= totalWarps) {
    profiles.push_back({ns, nr, is, ir, selfW});
  }

  return profiles;
}

std::string formatWarpReserve(const WarpReserveConfig& wr) {
  if (wr.nvlSendWarps == 0 && wr.nvlRecvWarps == 0 && wr.ibgdaSendWarps == 0 &&
      wr.ibgdaRecvWarps == 0 && wr.selfWarps == 0) {
    return "auto";
  }
  return "ns" + std::to_string(wr.nvlSendWarps) + "/nr" +
      std::to_string(wr.nvlRecvWarps) + "/is" +
      std::to_string(wr.ibgdaSendWarps) + "/ir" +
      std::to_string(wr.ibgdaRecvWarps) + "/s" + std::to_string(wr.selfWarps);
}

std::string makeInitLabel(const InitTimeParams& p, TopologyMode mode) {
  if (mode == TopologyMode::NVL_ONLY) {
    return "chunk=" + formatBytes(p.nvlChunkSize) +
        " pipe=" + std::to_string(p.nvlPipeDepth) +
        " buf=" + formatBytes(p.nvlDataBufSize);
  }
  return "ibStagBuf=" + formatBytes(p.ibgdaStagingBufSize) +
      " ibPipe=" + std::to_string(p.ibgdaPipeDepth) +
      " ibChunk=" + formatBytes(p.ibgdaChunkSize) +
      " nvlChunk=" + formatBytes(p.nvlChunkSize) +
      " nvlBuf=" + formatBytes(p.nvlDataBufSize) +
      " nvlPipe=" + std::to_string(p.nvlPipeDepth);
}

struct BenchResult {
  double latencyUs{0};
  double bandwidthGBps{0};
};

// ── Init config generation ──────────────────────────────────────────────────

std::vector<InitTimeParams> generateValidInitConfigs(TopologyMode mode) {
  std::vector<InitTimeParams> configs;

  // NVL param ranges
  const std::vector<std::size_t> nvlChunkSizes = {
      8 * 1024,
      16 * 1024,
      32 * 1024,
      64 * 1024,
      128 * 1024,
      256 * 1024,
      512 * 1024};
  const std::vector<int> nvlPipeDepths = {1, 2, 4, 8};
  const std::vector<std::size_t> nvlDataBufSizes = {
      256 * 1024,
      512 * 1024,
      1024 * 1024,
      2 * 1024 * 1024,
      4 * 1024 * 1024,
      8 * 1024 * 1024};

  if (mode == TopologyMode::NVL_ONLY) {
    for (auto chunk : nvlChunkSizes) {
      for (auto pipe : nvlPipeDepths) {
        for (auto buf : nvlDataBufSizes) {
          if (chunk * pipe <= buf) {
            configs.push_back({chunk, buf, pipe, 0, 0, 0});
          }
        }
      }
    }
  } else {
    // Hybrid: Use top NVL combos from NVL-only sweep results, then only
    // sweep the IBGDA dimension on top. This is based on two observations:
    //   1. NVL-only sweep (26 min) identifies the optimal NVL chunk size
    //      (chunk=32KB dominates all top 15 configs)
    //   2. pipe/buf settings barely matter within the winning chunk size
    //      (all within 0.5% of each other)
    //
    // UPDATE THESE after running the NVL-only sweep on your target hardware.
    // Use the top 3 from the ranked init config output.
    struct NvlCombo {
      std::size_t chunk;
      std::size_t buf;
      int pipe;
    };
    const std::vector<NvlCombo> topNvlCombos = {
        {32 * 1024, 8 * 1024 * 1024, 4}, // #1 from NVL-only: 1.103×
        {32 * 1024, 2 * 1024 * 1024, 2}, // #2 from NVL-only: 1.103×
        {32 * 1024, 1024 * 1024, 4}, // #3 from NVL-only: 1.101×
    };

    const std::vector<std::size_t> ibChunkSizes = {
        16 * 1024, 64 * 1024, 256 * 1024};
    const std::vector<int> ibPipeDepths = {1, 4, 8};
    const std::vector<std::size_t> ibStagBufSizes = {
        64 * 1024, 256 * 1024, 1024 * 1024};

    for (const auto& nvl : topNvlCombos) {
      for (auto ibChunk : ibChunkSizes) {
        for (auto ibPipe : ibPipeDepths) {
          for (auto ibStagBuf : ibStagBufSizes) {
            if (ibChunk * ibPipe > ibStagBuf) {
              continue;
            }
            configs.push_back(
                {nvl.chunk, nvl.buf, nvl.pipe, ibStagBuf, ibPipe, ibChunk});
          }
        }
      }
    }
  }

  return configs;
}

// ── Fixture ─────────────────────────────────────────────────────────────────

class AllToAllvAutoTuneSweepFixture : public meta::comms::BenchmarkTestFixture {
 protected:
  void SetUp() override {
    BenchmarkTestFixture::SetUp();
    CUDACHECK_SWEEP(cudaSetDevice(localRank));
    CUDACHECK_SWEEP(cudaStreamCreate(&stream_));

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

    auto bootstrapPtr = std::shared_ptr<meta::comms::IBootstrap>(
        bootstrap.get(), [](meta::comms::IBootstrap*) {});
    bool isSingleNode = (localSize == worldSize);
    MultiPeerTransportConfig topoConfig{};
    topoConfig.nvlConfig.dataBufferSize = 1024;
    topoConfig.nvlConfig.chunkSize = 1024;
    topoConfig.nvlConfig.pipelineDepth = 1;
    topoConfig.ibgdaConfig.cudaDevice = localRank;
    topoConfig.disableIb = isSingleNode;
    auto topoTransport = std::make_unique<MultiPeerTransport>(
        globalRank, worldSize, localRank, bootstrapPtr, topoConfig);

    bool hasIbgdaPeers = false;
    for (int r = 0; r < worldSize; ++r) {
      if (r != globalRank && !topoTransport->is_nvl_peer(r)) {
        hasIbgdaPeers = true;
        break;
      }
    }
    topoMode_ = hasIbgdaPeers ? TopologyMode::HYBRID : TopologyMode::NVL_ONLY;

    if (globalRank == 0) {
      int nvlCount = static_cast<int>(topoTransport->nvl_peer_ranks().size());
      int ibgdaOnlyCount = 0;
      for (int r = 0; r < worldSize; ++r) {
        if (r != globalRank && !topoTransport->is_nvl_peer(r)) {
          ibgdaOnlyCount++;
        }
      }
      XLOGF(
          INFO,
          "[SWEEP] Topology: {}, {} NVL peers, {} IBGDA-only peers, "
          "{} total ranks",
          topoMode_ == TopologyMode::NVL_ONLY ? "NVL-only" : "hybrid",
          nvlCount,
          ibgdaOnlyCount,
          worldSize);
    }

    topoTransport.reset();
    bootstrap->barrierAll();
  }

  void TearDown() override {
    NCCL_CHECK_SWEEP(ncclCommDestroy(ncclComm_));
    CUDACHECK_SWEEP(cudaStreamDestroy(stream_));
    BenchmarkTestFixture::TearDown();
  }

  double
  runNcclBenchmark(std::size_t bytesPerPeer, int nIter, double& latencyUs) {
    const std::size_t totalBytes = bytesPerPeer * worldSize;
    DeviceBuffer sendBuffer(totalBytes);
    DeviceBuffer recvBuffer(totalBytes);
    CUDACHECK_SWEEP(
        cudaMemset(sendBuffer.get(), globalRank & 0xFF, totalBytes));
    CUDACHECK_SWEEP(cudaMemset(recvBuffer.get(), 0, totalBytes));

    std::vector<size_t> sendcounts(worldSize, bytesPerPeer);
    std::vector<size_t> recvcounts(worldSize, bytesPerPeer);
    std::vector<size_t> sdispls(worldSize), rdispls(worldSize);
    for (int i = 0; i < worldSize; ++i) {
      sdispls[i] = i * bytesPerPeer;
      rdispls[i] = i * bytesPerPeer;
    }

    CudaEvent start, stop;
    int warmupIters = (bytesPerPeer <= 64 * 1024) ? 20 : 5;
    bootstrap->barrierAll();
    for (int i = 0; i < warmupIters; ++i) {
      NCCL_CHECK_SWEEP(ncclAllToAllv(
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
    CUDACHECK_SWEEP(cudaStreamSynchronize(stream_));
    bootstrap->barrierAll();

    CUDACHECK_SWEEP(cudaEventRecord(start.get(), stream_));
    for (int i = 0; i < nIter; ++i) {
      NCCL_CHECK_SWEEP(ncclAllToAllv(
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
    CUDACHECK_SWEEP(cudaEventRecord(stop.get(), stream_));
    CUDACHECK_SWEEP(cudaStreamSynchronize(stream_));

    float totalMs = 0;
    CUDACHECK_SWEEP(cudaEventElapsedTime(&totalMs, start.get(), stop.get()));
    float avgMs = totalMs / nIter;
    latencyUs = avgMs * 1000.0;
    std::size_t totalDataMoved = 2 * totalBytes;
    return (totalDataMoved / (1000.0 * 1000.0 * 1000.0)) / (avgMs / 1000.0);
  }

  BenchResult benchmarkKernel(
      DeviceSpan<Transport>& transports,
      bool isHybrid,
      int blocks,
      int threads,
      const WarpReserveConfig& wr,
      std::size_t msgSize,
      int nIter,
      int numNvlPeers,
      int numIbPeers,
      const int* nvlPeerRanks,
      const int* ibgdaPeerRanks) {
    int totalWarps = blocks * (threads / 32);
    if (totalWarps < 2 * worldSize) {
      return {};
    }

    WarpReserveDeviceConfig reserveDevCfg{};
    if (isHybrid) {
      int nNvl = numNvlPeers;
      int nIb = numIbPeers;

      if ((wr.nvlSendWarps > 0 && wr.nvlSendWarps < nNvl) ||
          (wr.nvlRecvWarps > 0 && wr.nvlRecvWarps < nNvl) ||
          (wr.ibgdaSendWarps > 0 && wr.ibgdaSendWarps < nIb) ||
          (wr.ibgdaRecvWarps > 0 && wr.ibgdaRecvWarps < nIb)) {
        return {};
      }

      int reqSelf = (wr.selfWarps > 0) ? wr.selfWarps : 1;
      int reqNvlSend = (wr.nvlSendWarps > 0) ? wr.nvlSendWarps : 2 * nNvl;
      int reqNvlRecv = (wr.nvlRecvWarps > 0) ? wr.nvlRecvWarps : 2 * nNvl;
      int reqIbSend = (wr.ibgdaSendWarps > 0) ? wr.ibgdaSendWarps : nIb;
      int reqIbRecv = (wr.ibgdaRecvWarps > 0) ? wr.ibgdaRecvWarps : nIb;
      int minWarps = reqSelf + reqNvlSend + reqNvlRecv + reqIbSend + reqIbRecv;

      if (totalWarps < minWarps) {
        return {};
      }

      reserveDevCfg = resolveWarpReserve(
          wr, nNvl, nIb, nvlPeerRanks, ibgdaPeerRanks, threads);
    }

    const std::size_t totalBytes = msgSize * worldSize;
    DeviceBuffer sendBuffer(totalBytes);
    DeviceBuffer recvBuffer(totalBytes);
    CUDACHECK_SWEEP(
        cudaMemset(sendBuffer.get(), globalRank & 0xFF, totalBytes));
    CUDACHECK_SWEEP(cudaMemset(recvBuffer.get(), 0, totalBytes));

    std::vector<ChunkInfo> h_chunks;
    h_chunks.reserve(worldSize);
    for (int r = 0; r < worldSize; ++r) {
      h_chunks.emplace_back(r * msgSize, msgSize);
    }
    DeviceBuffer d_send(sizeof(ChunkInfo) * worldSize);
    DeviceBuffer d_recv(sizeof(ChunkInfo) * worldSize);
    CUDACHECK_SWEEP(cudaMemcpy(
        d_send.get(),
        h_chunks.data(),
        sizeof(ChunkInfo) * worldSize,
        cudaMemcpyHostToDevice));
    CUDACHECK_SWEEP(cudaMemcpy(
        d_recv.get(),
        h_chunks.data(),
        sizeof(ChunkInfo) * worldSize,
        cudaMemcpyHostToDevice));

    DeviceSpan<ChunkInfo> sendInfos(
        static_cast<ChunkInfo*>(d_send.get()), worldSize);
    DeviceSpan<ChunkInfo> recvInfos(
        static_cast<ChunkInfo*>(d_recv.get()), worldSize);

    CudaEvent start, stop;

    bootstrap->barrierAll();
    bool launchOk = true;
    try {
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
              blocks,
              threads,
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
              blocks,
              threads,
              std::nullopt);
        }
      }
    } catch (const std::exception&) {
      launchOk = false;
    }
    cudaGetLastError();
    cudaStreamSynchronize(stream_);
    cudaGetLastError();
    bootstrap->barrierAll();

    if (!launchOk) {
      return {};
    }

    CUDACHECK_SWEEP(cudaEventRecord(start.get(), stream_));
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
            blocks,
            threads,
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
            blocks,
            threads,
            std::nullopt);
      }
    }
    CUDACHECK_SWEEP(cudaEventRecord(stop.get(), stream_));
    CUDACHECK_SWEEP(cudaStreamSynchronize(stream_));

    float totalMs = 0;
    CUDACHECK_SWEEP(cudaEventElapsedTime(&totalMs, start.get(), stop.get()));
    float avgMs = totalMs / nIter;
    double lat = avgMs * 1000.0;
    std::size_t totalDataMoved = 2 * totalBytes;
    double bw =
        (totalDataMoved / (1000.0 * 1000.0 * 1000.0)) / (avgMs / 1000.0);

    return {lat, bw};
  }

  double syncScalar(double localVal) {
    std::vector<double> all(worldSize, 0);
    all[globalRank] = localVal;
    bootstrap->allGather(all.data(), sizeof(double), globalRank, worldSize)
        .get();
    return all[0];
  }

  TopologyMode topoMode_{TopologyMode::NVL_ONLY};
  ncclComm_t ncclComm_{};
  cudaStream_t stream_{};
};

// ═══════════════════════════════════════════════════════════════════════════
// Exhaustive sweep: enumerate all valid init configs, run a full kernel
// param grid (blocks × threads × warp reserves) at every message size,
// with optimistic-bound pruning to cut configs that cannot win.
//
// Key improvement over the previous adaptive sweep: init config and kernel
// params are evaluated JOINTLY, capturing interactions like
// chunk=16KB + 512 threads >> chunk=64KB + 128 threads.
//
// Message sizes are evaluated in stratified order (small, large, medium,
// ...) so that pruning decisions reflect all regimes from the first few
// evaluations.
// ═══════════════════════════════════════════════════════════════════════════

TEST_F(AllToAllvAutoTuneSweepFixture, ExhaustiveSweep) {
  auto msgSizes = stratifiedMessageSizes();
  const int numSizes = static_cast<int>(msgSizes.size());
  int totalEvals = 0;
  bool isHybrid = (topoMode_ == TopologyMode::HYBRID);

  const std::vector<int> blockGrid = {4, 8, 16, 32, 64};
  const std::vector<int> threadGrid = {128, 256, 512};

  // ── Phase 1: Cache NCCL baselines ──
  // Sync baselines across ranks so all pruning decisions are identical.
  // Without this, measurement noise causes different ranks to compute
  // slightly different ncclBw → different speedup → pruning triggers on
  // some ranks but not others → collective count diverges → DEADLOCK.

  std::map<std::size_t, std::pair<double, double>> ncclBaselines;
  for (auto msgSize : msgSizes) {
    double lat = 0;
    double bw = runNcclBenchmark(msgSize, iterCountForMsgSize(msgSize), lat);
    bw = syncScalar(bw);
    lat = syncScalar(lat);
    ncclBaselines[msgSize] = {lat, bw};
  }

  if (globalRank == 0) {
    XLOGF(
        INFO,
        "[SWEEP] NCCL baselines cached for {} message sizes",
        msgSizes.size());
  }

  // ── Phase 2: Generate all valid init configs ──

  auto allConfigs = generateValidInitConfigs(topoMode_);

  // Sort configs so likely-good ones come first: NVL chunkSize near 32KB
  // (the typical NVL-only winner) establishes a strong baseline early,
  // enabling the optimistic bound to prune bad configs faster.
  std::sort(
      allConfigs.begin(),
      allConfigs.end(),
      [](const InitTimeParams& a, const InitTimeParams& b) {
        auto dist = [](std::size_t chunk) -> int {
          constexpr std::size_t target = 32 * 1024;
          if (chunk == target) {
            return 0;
          }
          if (chunk > target) {
            int d = 0;
            for (std::size_t v = target; v < chunk; v *= 2) {
              d++;
            }
            return d;
          }
          int d = 0;
          for (std::size_t v = chunk; v < target; v *= 2) {
            d++;
          }
          return d;
        };
        int da = dist(a.nvlChunkSize);
        int db = dist(b.nvlChunkSize);
        if (da != db) {
          return da < db;
        }
        if (a.nvlDataBufSize != b.nvlDataBufSize) {
          return a.nvlDataBufSize > b.nvlDataBufSize;
        }
        return a.nvlPipeDepth < b.nvlPipeDepth;
      });
  if (globalRank == 0) {
    XLOGF(
        INFO,
        "[SWEEP] Generated {} valid init configs for {} mode",
        allConfigs.size(),
        isHybrid ? "hybrid" : "NVL-only");
  }

  // ── Phase 3: Exhaustive sweep with optimistic-bound pruning ──

  double bestOverallGeoMean = 0;
  std::string bestOverallLabel;
  InitTimeParams bestOverallInit{};
  PerInitConfigResults perInitResults;
  std::vector<std::pair<std::string, double>> allEvaluatedConfigs;
  int configsEvaluated = 0;
  int configsPruned = 0;

  for (std::size_t cfgIdx = 0; cfgIdx < allConfigs.size(); ++cfgIdx) {
    const auto& initParams = allConfigs[cfgIdx];
    std::string initLabel = makeInitLabel(initParams, topoMode_);

    auto bootstrapPtr = std::shared_ptr<meta::comms::IBootstrap>(
        bootstrap.get(), [](meta::comms::IBootstrap*) {});

    MultiPeerTransportConfig cfg{};
    cfg.nvlConfig.chunkSize = initParams.nvlChunkSize;
    cfg.nvlConfig.dataBufferSize = initParams.nvlDataBufSize;
    cfg.nvlConfig.pipelineDepth = initParams.nvlPipeDepth;
    cfg.ibgdaConfig.cudaDevice = localRank;

    if (isHybrid) {
      cfg.ibgdaSetupConfig.dataBufferSize = initParams.ibgdaStagingBufSize;
      cfg.ibgdaSetupConfig.chunkSize = initParams.ibgdaChunkSize;
      cfg.ibgdaSetupConfig.pipelineDepth = initParams.ibgdaPipeDepth;
    } else {
      cfg.disableIb = true;
    }

    std::unique_ptr<MultiPeerTransport> transport;
    bool constructOk = true;
    try {
      transport = std::make_unique<MultiPeerTransport>(
          globalRank, worldSize, localRank, bootstrapPtr, cfg);
    } catch (const std::exception& e) {
      if (globalRank == 0) {
        XLOGF(
            WARN,
            "[SWEEP] Transport construction failed for {}: {}",
            initLabel,
            e.what());
      }
      constructOk = false;
    }

    // All ranks must agree before calling exchange() — if ANY rank's
    // constructor failed, skip this config on ALL ranks. Without this,
    // succeeding ranks enter exchange() (collective allGather) while
    // failing ranks call barrierAll() → different collectives → deadlock.
    int localOk = constructOk ? 1 : 0;
    std::vector<int> allOk(worldSize, 0);
    allOk[globalRank] = localOk;
    bootstrap->allGather(allOk.data(), sizeof(int), globalRank, worldSize)
        .get();
    bool anyFailed = false;
    for (int i = 0; i < worldSize; ++i) {
      if (allOk[i] == 0) {
        anyFailed = true;
        break;
      }
    }
    if (anyFailed) {
      transport.reset();
      configsPruned++;
      continue;
    }

    try {
      transport->exchange();
    } catch (const std::exception& e) {
      if (globalRank == 0) {
        XLOGF(WARN, "[SWEEP] Exchange failed for {}: {}", initLabel, e.what());
      }
      // Don't call barrierAll here — succeeding ranks are past exchange()
      // and heading to the kernel sweep. Use allGather consensus instead.
      constructOk = false;
    }

    // Post-exchange consensus: if ANY rank's exchange failed, all skip.
    // This prevents the deadlock where failing ranks call barrierAll
    // while succeeding ranks proceed to kernel sweep collectives.
    {
      int exchOk = constructOk ? 1 : 0;
      std::vector<int> allExchOk(worldSize, 0);
      allExchOk[globalRank] = exchOk;
      bootstrap->allGather(allExchOk.data(), sizeof(int), globalRank, worldSize)
          .get();
      bool anyExchFailed = false;
      for (int i = 0; i < worldSize; ++i) {
        if (allExchOk[i] == 0) {
          anyExchFailed = true;
          break;
        }
      }
      if (anyExchFailed) {
        transport.reset();
        configsPruned++;
        continue;
      }
    }

    auto deviceHandle = transport->get_device_handle();
    DeviceSpan<Transport> transports(
        deviceHandle.transports.data(), deviceHandle.transports.size());

    // Build warp reserve profiles for this transport
    int maxWarps = 64 * (512 / 32); // max possible from block/thread grid
    std::vector<WarpReserveConfig> wrProfiles;
    if (isHybrid) {
      wrProfiles = hybridWarpReserveProfiles(
          deviceHandle.numNvlPeers, deviceHandle.numIbPeers, maxWarps);
    } else {
      wrProfiles.push_back({0, 0, 0, 0, 0}); // auto only for NVL
    }

    // Per-msg results for this init config
    std::map<std::size_t, PerMsgBest> thisConfigResults;
    double sumLogSpeedup = 0;
    int sizesEvaluated = 0;
    bool pruned = false;

    for (int sIdx = 0; sIdx < numSizes; ++sIdx) {
      std::size_t msgSize = msgSizes[sIdx];
      int nIter = iterCountForMsgSize(msgSize);

      double ncclLat = ncclBaselines[msgSize].first;
      double ncclBw = ncclBaselines[msgSize].second;

      double bestBw = 0;
      double bestLat = 0;
      int bestBlocks = 0;
      int bestThreads = 0;
      WarpReserveConfig bestWr{0, 0, 0, 0, 0};

      // Full kernel param grid for this (init config, msg size)
      for (int blocks : blockGrid) {
        for (int threads : threadGrid) {
          for (const auto& wr : wrProfiles) {
            auto result = benchmarkKernel(
                transports,
                isHybrid,
                blocks,
                threads,
                wr,
                msgSize,
                nIter,
                deviceHandle.numNvlPeers,
                deviceHandle.numIbPeers,
                deviceHandle.nvlPeerRanks.data(),
                deviceHandle.ibgdaPeerRanks.data());
            double bw = syncScalar(result.bandwidthGBps);
            double lat = syncScalar(result.latencyUs);
            totalEvals++;

            if (bw > bestBw) {
              bestBw = bw;
              bestLat = lat;
              bestBlocks = blocks;
              bestThreads = threads;
              bestWr = wr;
            }
          }
        }
      }

      double speedup = (ncclBw > 0) ? bestBw / ncclBw : 0;
      double clampedSpeedup = std::max(speedup, 0.01);
      sumLogSpeedup += std::log(clampedSpeedup);
      sizesEvaluated++;

      thisConfigResults[msgSize] = PerMsgBest{
          msgSize,
          bestBlocks,
          bestThreads,
          bestLat,
          bestBw,
          ncclLat,
          ncclBw,
          speedup,
          bestWr.nvlSendWarps,
          bestWr.nvlRecvWarps,
          bestWr.ibgdaSendWarps,
          bestWr.ibgdaRecvWarps,
          bestWr.selfWarps};

      // ── Pruning checks ──

      // Magnitude gate: if best BW < 5% of NCCL at any size, config is broken
      if (ncclBw > 0 && bestBw < 0.05 * ncclBw) {
        if (globalRank == 0) {
          XLOGF(
              INFO,
              "[SWEEP] PRUNE (magnitude) config {} at {}: "
              "BW={:.2f} < 5% of NCCL {:.2f}",
              initLabel,
              formatBytes(msgSize),
              bestBw,
              ncclBw);
        }
        pruned = true;
        break;
      }

      // Fast reject: after 4 sizes, if running geo-mean < 0.3× → hopeless
      if (sizesEvaluated >= 4) {
        double runningGeoMean = std::exp(sumLogSpeedup / sizesEvaluated);
        if (runningGeoMean < 0.3) {
          if (globalRank == 0) {
            XLOGF(
                INFO,
                "[SWEEP] PRUNE (fast reject) config {} after {} sizes: "
                "geo-mean={:.3f}x",
                initLabel,
                sizesEvaluated,
                runningGeoMean);
          }
          pruned = true;
          break;
        }
      }

      // Optimistic bound: can this config possibly beat the current best?
      if (bestOverallGeoMean > 0 && sizesEvaluated >= 3) {
        int remaining = numSizes - sizesEvaluated;
        double optimisticLogSum = sumLogSpeedup + remaining * std::log(2.0);
        double bestPossible = std::exp(optimisticLogSum / numSizes);
        if (bestPossible < bestOverallGeoMean) {
          if (globalRank == 0) {
            XLOGF(
                INFO,
                "[SWEEP] PRUNE (optimistic bound) config {} after {} sizes: "
                "bestPossible={:.3f}x < current best {:.3f}x",
                initLabel,
                sizesEvaluated,
                bestPossible,
                bestOverallGeoMean);
          }
          pruned = true;
          break;
        }
      }
    }

    // Compute final geo-mean for this config
    double geoMean =
        (sizesEvaluated > 0) ? std::exp(sumLogSpeedup / sizesEvaluated) : 0;

    // For pruned configs, the geo-mean is only partial — penalize it
    // by assuming remaining sizes would be 0.5× NCCL (pessimistic)
    if (pruned && sizesEvaluated < numSizes) {
      int remaining = numSizes - sizesEvaluated;
      double penalizedLogSum = sumLogSpeedup + remaining * std::log(0.5);
      geoMean = std::exp(penalizedLogSum / numSizes);
    }

    configsEvaluated++;
    if (globalRank == 0) {
      allEvaluatedConfigs.emplace_back(initLabel, geoMean);
      XLOGF(
          INFO,
          "[SWEEP] Config {}/{}: {} -> geo-mean={:.3f}x ({}sizes={}) {}",
          cfgIdx + 1,
          allConfigs.size(),
          initLabel,
          geoMean,
          pruned ? "PRUNED " : "",
          sizesEvaluated,
          (geoMean > bestOverallGeoMean && !pruned) ? "NEW BEST" : "");
    }

    // Update overall best (only from fully evaluated configs)
    if (!pruned && geoMean > bestOverallGeoMean) {
      bestOverallGeoMean = geoMean;
      bestOverallLabel = initLabel;
      bestOverallInit = initParams;
      perInitResults[initLabel] = thisConfigResults;
    }

    bootstrap->barrierAll();
    transport.reset();
    // Ensure all async IBGDA/RDMA cleanup completes before creating the
    // next transport. Without this, RDMA resources (staging buffers,
    // memory registrations, QPs) from the previous transport may not be
    // fully released, causing the next exchange() to hang.
    CUDACHECK_SWEEP(cudaDeviceSynchronize());
    bootstrap->barrierAll();
  }

  // ── Phase 4: Reporting ──

  if (globalRank != 0) {
    return;
  }

  if (perInitResults.empty()) {
    fprintf(stderr, "\n  No valid configs found!\n");
    return;
  }

  const auto& bestPerMsg = perInitResults[bestOverallLabel];

  // Sort evaluated configs for ranked comparison
  std::sort(
      allEvaluatedConfigs.begin(),
      allEvaluatedConfigs.end(),
      [](const auto& a, const auto& b) { return a.second > b.second; });

  fprintf(stderr, "\n");
  fprintf(
      stderr,
      "==========================================="
      "===========================================\n");
  fprintf(stderr, "  BEST INIT-TIME CONFIG (Overall)\n");
  fprintf(
      stderr,
      "==========================================="
      "===========================================\n");
  fprintf(
      stderr,
      "  Winner: %s\n"
      "  Geometric mean speedup vs NCCL: %.3fx\n",
      bestOverallLabel.c_str(),
      bestOverallGeoMean);

  if (topoMode_ == TopologyMode::NVL_ONLY) {
    fprintf(
        stderr,
        "\n  NvlInitConfig {\n"
        "    .nvlChunkSize    = %s,\n"
        "    .nvlDataBufSize  = %s,\n"
        "    .pipelineDepth   = %d,\n"
        "  };\n",
        formatBytes(bestOverallInit.nvlChunkSize).c_str(),
        formatBytes(bestOverallInit.nvlDataBufSize).c_str(),
        bestOverallInit.nvlPipeDepth);
  } else {
    fprintf(
        stderr,
        "\n  IbgdaInitConfig {\n"
        "    .stagingBufferSize = %s,\n"
        "    .pipelineDepth     = %d,\n"
        "    .ibgdaChunkSize    = %s,\n"
        "  };\n"
        "  NvlInitConfig {\n"
        "    .nvlChunkSize      = %s,\n"
        "    .nvlDataBufSize    = %s,\n"
        "    .pipelineDepth     = %d,\n"
        "  };\n",
        formatBytes(bestOverallInit.ibgdaStagingBufSize).c_str(),
        bestOverallInit.ibgdaPipeDepth,
        formatBytes(bestOverallInit.ibgdaChunkSize).c_str(),
        formatBytes(bestOverallInit.nvlChunkSize).c_str(),
        formatBytes(bestOverallInit.nvlDataBufSize).c_str(),
        bestOverallInit.nvlPipeDepth);
  }

  // Per-msg kernel param table (sorted by msg size)
  fprintf(stderr, "\n  Best Per-Msg Kernel Params Under This Init Config:\n");
  if (topoMode_ == TopologyMode::NVL_ONLY) {
    fprintf(
        stderr,
        "  %-8s %6s %7s | %10s %10s | %10s %10s | %7s\n",
        "MsgSize",
        "Blocks",
        "Threads",
        "Lat(us)",
        "BW(GB/s)",
        "NCCL Lat",
        "NCCL BW",
        "vs NCCL");
    fprintf(stderr, "  %s\n", std::string(80, '-').c_str());
    for (const auto& [sz, c] : bestPerMsg) {
      fprintf(
          stderr,
          "  %-8s %6d %7d | %10.1f %10.2f | %10.1f %10.2f | %6.2fx\n",
          formatBytes(c.msgSize).c_str(),
          c.numBlocks,
          c.numThreads,
          c.latencyUs,
          c.bandwidthGBps,
          c.ncclLatencyUs,
          c.ncclBandwidthGBps,
          c.speedupVsNccl);
    }
  } else {
    fprintf(
        stderr,
        "  %-8s %6s %7s %-18s | %10s %10s | %10s %10s | %7s\n",
        "MsgSize",
        "Blocks",
        "Threads",
        " WarpReserve",
        "Lat(us)",
        "BW(GB/s)",
        "NCCL Lat",
        "NCCL BW",
        "vs NCCL");
    fprintf(stderr, "  %s\n", std::string(105, '-').c_str());
    for (const auto& [sz, c] : bestPerMsg) {
      WarpReserveConfig wr{
          c.nvlSendWarps,
          c.nvlRecvWarps,
          c.ibgdaSendWarps,
          c.ibgdaRecvWarps,
          c.selfWarps};
      fprintf(
          stderr,
          "  %-8s %6d %7d %-18s | %10.1f %10.2f | %10.1f %10.2f | %6.2fx\n",
          formatBytes(c.msgSize).c_str(),
          c.numBlocks,
          c.numThreads,
          formatWarpReserve(wr).c_str(),
          c.latencyUs,
          c.bandwidthGBps,
          c.ncclLatencyUs,
          c.ncclBandwidthGBps,
          c.speedupVsNccl);
    }
  }

  // C++ copy-paste code
  fprintf(stderr, "\n");
  fprintf(stderr, "  // -- Copy-paste into AllToAllvAutoTuneConfig.h --\n");
  fprintf(stderr, "  // Init-time config: %s\n", bestOverallLabel.c_str());

  if (topoMode_ == TopologyMode::NVL_ONLY) {
    fprintf(
        stderr,
        "  static constexpr NvlPerMsgConfig kDefaultNvlConfigs[] = {\n");
    for (const auto& [sz, c] : bestPerMsg) {
      fprintf(
          stderr,
          "      {%zu, %d, %d},  // <=%s  lat=%.1fus BW=%.2fGB/s %.2fx\n",
          c.msgSize,
          c.numBlocks,
          c.numThreads,
          formatBytes(c.msgSize).c_str(),
          c.latencyUs,
          c.bandwidthGBps,
          c.speedupVsNccl);
    }
    fprintf(stderr, "  };\n");
  } else {
    fprintf(
        stderr,
        "  static constexpr HybridPerMsgConfig "
        "kDefaultHybridConfigs[] = {\n");
    for (const auto& [sz, c] : bestPerMsg) {
      fprintf(
          stderr,
          "      {%zu, %d, %d, %d, %d, %d, %d, %d},"
          "  // <=%s  lat=%.1fus BW=%.2fGB/s %.2fx wr=%s\n",
          c.msgSize,
          c.numBlocks,
          c.numThreads,
          c.nvlSendWarps,
          c.nvlRecvWarps,
          c.ibgdaSendWarps,
          c.ibgdaRecvWarps,
          c.selfWarps,
          formatBytes(c.msgSize).c_str(),
          c.latencyUs,
          c.bandwidthGBps,
          c.speedupVsNccl,
          formatWarpReserve({c.nvlSendWarps,
                             c.nvlRecvWarps,
                             c.ibgdaSendWarps,
                             c.ibgdaRecvWarps,
                             c.selfWarps})
              .c_str());
    }
    fprintf(stderr, "  };\n");
  }

  // Ranked comparison
  fprintf(stderr, "\n  All Init Configs Evaluated (ranked by geo-mean):\n");
  fprintf(stderr, "  %4s  %-60s  %12s\n", "Rank", "Init Config", "GeoMean");
  fprintf(stderr, "  %s\n", std::string(80, '-').c_str());
  for (std::size_t i = 0; i < allEvaluatedConfigs.size(); ++i) {
    fprintf(
        stderr,
        "  %4zu  %-60s  %10.3fx\n",
        i + 1,
        allEvaluatedConfigs[i].first.c_str(),
        allEvaluatedConfigs[i].second);
    if (i >= 19) {
      fprintf(
          stderr,
          "        ... (%zu more configs)\n",
          allEvaluatedConfigs.size() - 20);
      break;
    }
  }

  fprintf(
      stderr,
      "\n  Total: %d configs evaluated, %d pruned, %d benchmark evals\n",
      configsEvaluated,
      configsPruned,
      totalEvals);

  fprintf(
      stderr,
      "==========================================="
      "===========================================\n\n");
}

} // namespace

} // namespace comms::pipes::benchmark

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  ::testing::AddGlobalTestEnvironment(new meta::comms::BenchmarkEnvironment());
  return RUN_ALL_TESTS();
}
