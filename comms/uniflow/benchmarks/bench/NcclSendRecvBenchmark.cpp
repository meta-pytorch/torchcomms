// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/benchmarks/bench/NcclSendRecvBenchmark.h"

#include <algorithm>
#include <chrono>
#include <cstring>

#include <cuda_runtime_api.h> // @manual=third-party//cuda:cuda-lazy
#include "nccl.h" // @manual

#include "comms/uniflow/benchmarks/Rendezvous.h"
#include "comms/uniflow/logging/Logger.h"

namespace uniflow::benchmark {

namespace {

#define CUDACHECK(cmd)                                            \
  do {                                                            \
    cudaError_t e = (cmd);                                        \
    if (e != cudaSuccess) {                                       \
      UNIFLOW_LOG_ERROR("CUDA error: {}", cudaGetErrorString(e)); \
      return {};                                                  \
    }                                                             \
  } while (0)

#define NCCLCHECK(cmd)                                                  \
  do {                                                                  \
    ncclResult_t r = (cmd);                                             \
    if (r != ncclSuccess) {                                             \
      UNIFLOW_LOG_ERROR("NCCL error {}: {}", r, ncclGetErrorString(r)); \
      return {};                                                        \
    }                                                                   \
  } while (0)

#define CUDACHECK_THROW(cmd)                                    \
  do {                                                          \
    cudaError_t e = (cmd);                                      \
    if (e != cudaSuccess) {                                     \
      throw std::runtime_error(                                 \
          std::string("CUDA error: ") + cudaGetErrorString(e)); \
    }                                                           \
  } while (0)

struct NcclResources {
  void* sendbuf{nullptr};
  void* recvbuf{nullptr};
  cudaStream_t stream{nullptr};
  ncclComm_t comm{nullptr};
  bool useGpu{false};

  NcclResources(ncclComm_t comm, bool useGpu, size_t sendSize, size_t recvSize)
      : comm(comm), useGpu(useGpu) {
    if (useGpu) {
      CUDACHECK_THROW(cudaMalloc(&sendbuf, sendSize));
      CUDACHECK_THROW(cudaMalloc(&recvbuf, recvSize));
      CUDACHECK_THROW(cudaMemset(sendbuf, 0xAB, sendSize));
      CUDACHECK_THROW(cudaMemset(recvbuf, 0x00, recvSize));
    } else {
      CUDACHECK_THROW(cudaMallocHost(&sendbuf, sendSize));
      CUDACHECK_THROW(cudaMallocHost(&recvbuf, recvSize));
      std::memset(sendbuf, 0xAB, sendSize);
      std::memset(recvbuf, 0x00, recvSize);
    }
    CUDACHECK_THROW(cudaStreamCreate(&stream));
  }

  ~NcclResources() {
    if (stream) {
      cudaStreamDestroy(stream);
    }
    if (useGpu) {
      if (recvbuf) {
        cudaFree(recvbuf);
      }
      if (sendbuf) {
        cudaFree(sendbuf);
      }
    } else {
      if (recvbuf) {
        cudaFreeHost(recvbuf);
      }
      if (sendbuf) {
        cudaFreeHost(sendbuf);
      }
    }
    if (comm) {
      ncclCommDestroy(comm);
    }
  }

  NcclResources(const NcclResources&) = delete;
  NcclResources& operator=(const NcclResources&) = delete;
  NcclResources(NcclResources&&) = delete;
  NcclResources& operator=(NcclResources&&) = delete;
};

} // namespace

std::vector<BenchmarkResult> NcclSendRecvBenchmark::run(
    const BenchmarkConfig& config,
    std::vector<PeerConnection>& peers,
    const BootstrapConfig& bootstrap) {
  const int rank = bootstrap.rank;
  const int worldSize = bootstrap.worldSize;
  const bool useGpu = config.cudaDevice >= 0;
  const auto& topology = config.topology;

  std::vector<int> sendPeers;
  std::vector<int> recvPeers;
  if (topology == "fanout") {
    if (bootstrap.isRank0()) {
      for (int r = 1; r < worldSize; ++r) {
        sendPeers.push_back(r);
      }
    } else {
      recvPeers.push_back(0);
    }
  } else if (topology == "fanin") {
    if (bootstrap.isRank0()) {
      for (int r = 1; r < worldSize; ++r) {
        recvPeers.push_back(r);
      }
    } else {
      sendPeers.push_back(0);
    }
  } else {
    if (bootstrap.isRank0()) {
      for (int r = 1; r < worldSize; ++r) {
        sendPeers.push_back(r);
        recvPeers.push_back(r);
      }
    } else {
      recvPeers.push_back(0);
      sendPeers.push_back(0);
    }
  }
  const int numPeers =
      static_cast<int>(std::max(sendPeers.size(), recvPeers.size()));

  // NCCL requires a CUDA context even for host memory transfers.
  CUDACHECK(cudaSetDevice(useGpu ? config.cudaDevice : rank));

  // --- Initialize NCCL communicator ---
  // Rank 0 generates unique ID, broadcasts to all peers via TCP control.
  ncclUniqueId ncclId;
  if (bootstrap.isRank0()) {
    NCCLCHECK(ncclGetUniqueId(&ncclId));
    std::vector<uint8_t> idBytes(sizeof(ncclId));
    std::memcpy(idBytes.data(), &ncclId, sizeof(ncclId));
    for (auto& peer : peers) {
      auto res = exchangeMetadata(*peer.ctrl, idBytes, true);
      if (!res) {
        UNIFLOW_LOG_ERROR(
            "NcclSendRecvBenchmark: failed to send NCCL ID to peer {}",
            peer.peerRank);
        return {};
      }
    }
  } else {
    if (peers.empty()) {
      UNIFLOW_LOG_ERROR(
          "NcclSendRecvBenchmark: rank {} has no control peer for NCCL ID exchange",
          rank);
      return {};
    }
    auto res = exchangeMetadata(
        *peers[0].ctrl, std::vector<uint8_t>(sizeof(ncclId)), false);
    if (!res) {
      UNIFLOW_LOG_ERROR("NcclSendRecvBenchmark: failed to recv NCCL ID");
      return {};
    }
    std::memcpy(&ncclId, res.value().data(), sizeof(ncclId));
  }

  ncclComm_t comm;
  NCCLCHECK(ncclCommInitRank(&comm, worldSize, ncclId, rank));

  UNIFLOW_LOG_WARN(
      "NcclSendRecvBenchmark: rank {} initialized (worldSize={}, "
      "topology={}, sends={}, recvs={}, memory={})",
      rank,
      worldSize,
      topology,
      sendPeers.size(),
      recvPeers.size(),
      useGpu ? "GPU" : "CPU");

  const size_t numRecvPeers = std::max(size_t{1}, recvPeers.size());
  const size_t recvBufSize = config.maxSize * numRecvPeers;

  NcclResources res(comm, useGpu, config.maxSize, recvBufSize);

  // --- Benchmark loop ---
  using Clock = std::chrono::steady_clock;
  auto sizes = generateSizes(config.minSize, config.maxSize);
  std::vector<BenchmarkResult> results;

  for (auto size : sizes) {
    auto barrierStatus = barrier(peers, bootstrap);
    if (!barrierStatus) {
      UNIFLOW_LOG_ERROR(
          "NcclSendRecvBenchmark: barrier failed: {}",
          barrierStatus.error().toString());
      break;
    }

    size_t count = size; // ncclChar = 1 byte

    // Warmup
    for (int w = 0; w < config.warmupIterations; ++w) {
      NCCLCHECK(ncclGroupStart());
      for (int p : sendPeers) {
        NCCLCHECK(
            ncclSend(res.sendbuf, count, ncclChar, p, res.comm, res.stream));
      }
      for (size_t i = 0; i < recvPeers.size(); ++i) {
        auto* dst = static_cast<char*>(res.recvbuf) + i * config.maxSize;
        NCCLCHECK(
            ncclRecv(dst, count, ncclChar, recvPeers[i], res.comm, res.stream));
      }
      NCCLCHECK(ncclGroupEnd());
      CUDACHECK(cudaStreamSynchronize(res.stream));
    }

    // Timed iterations
    CUDACHECK(cudaStreamSynchronize(res.stream));
    auto overallStart = Clock::now();

    for (int iter = 0; iter < config.iterations; ++iter) {
      NCCLCHECK(ncclGroupStart());
      for (int p : sendPeers) {
        NCCLCHECK(
            ncclSend(res.sendbuf, count, ncclChar, p, res.comm, res.stream));
      }
      for (size_t i = 0; i < recvPeers.size(); ++i) {
        auto* dst = static_cast<char*>(res.recvbuf) + i * config.maxSize;
        NCCLCHECK(
            ncclRecv(dst, count, ncclChar, recvPeers[i], res.comm, res.stream));
      }
      NCCLCHECK(ncclGroupEnd());
    }

    CUDACHECK(cudaStreamSynchronize(res.stream));
    auto overallEnd = Clock::now();

    double totalTimeSec =
        std::chrono::duration<double>(overallEnd - overallStart).count();

    double totalBytes =
        static_cast<double>(size) * config.iterations * numPeers;
    double bandwidthGBs =
        (totalTimeSec > 0) ? (totalBytes / totalTimeSec) / 1e9 : 0;
    double avgLatencyUs =
        (totalTimeSec > 0) ? (totalTimeSec * 1e6) / config.iterations : 0;

    results.push_back({
        .benchmarkName = name(),
        .transport = "nccl",
        .direction = topology,
        .messageSize = size,
        .iterations = config.iterations,
        .batchSize = 1,
        .bandwidthGBs = bandwidthGBs,
        .latency =
            {.min = avgLatencyUs,
             .max = avgLatencyUs,
             .avg = avgLatencyUs,
             .p50 = avgLatencyUs,
             .p99 = avgLatencyUs},
        .numPeers = numPeers,
    });

    UNIFLOW_LOG_WARN(
        "[rank {}] nccl_{} peers={} size={:<10} iters={:<6} "
        "aggBw={:.2f} GB/s  perLink={:.2f} GB/s  avg={:.1f} us",
        rank,
        topology,
        numPeers,
        size,
        config.iterations,
        bandwidthGBs,
        numPeers > 0 ? bandwidthGBs / numPeers : 0.0,
        avgLatencyUs);
  }

  return results;
}

} // namespace uniflow::benchmark
