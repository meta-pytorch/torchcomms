// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include <cuda_runtime.h>
#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include <memory>
#include <vector>

#include "comms/pipes/MultipeerIbgdaTransport.h"
#include "comms/pipes/benchmarks/IbgdaBenchmark.h"
#include "comms/pipes/rdma/NicDiscovery.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::DeviceBuffer;
using meta::comms::MpiBaseTestFixture;
using meta::comms::MPIEnvironmentBase;

namespace comms::pipes::benchmark {

constexpr int kScatterBenchIters = 1000;

#define CUDA_CHECK_VOID(call)        \
  do {                               \
    cudaError_t err = call;          \
    if (err != cudaSuccess) {        \
      XLOGF(                         \
          ERR,                       \
          "CUDA error at {}:{}: {}", \
          __FILE__,                  \
          __LINE__,                  \
          cudaGetErrorString(err));  \
      return;                        \
    }                                \
  } while (0)

class IbgdaScatterBenchmarkFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDA_CHECK_VOID(cudaSetDevice(localRank));
    CUDA_CHECK_VOID(cudaStreamCreate(&stream_));

    // Get GPU clock rate for converting cycles to time
    int clockRateKHz;
    CUDA_CHECK_VOID(
        cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrClockRate, localRank));
    clockRateGHz_ = clockRateKHz / 1e6f;
  }

  void TearDown() override {
    CUDA_CHECK_VOID(cudaStreamDestroy(stream_));
    MpiBaseTestFixture::TearDown();
  }

  // Convert GPU cycles to microseconds
  float cyclesToUs(unsigned long long cycles) const {
    return cycles / (clockRateGHz_ * 1000.0f);
  }

  cudaStream_t stream_{};
  float clockRateGHz_{1.0f};
};

TEST_F(IbgdaScatterBenchmarkFixture, ScatterSignal) {
  // Simulates AFD scatter: rank 0 sends 4 puts to each of N virtual peers,
  // each via its own QP. Virtual peers are created by instantiating multiple
  // MultipeerIbgdaTransport instances.
  if (numRanks < 2) {
    XLOGF(INFO, "Skipping test: requires at least 2 ranks, got {}", numRanks);
    return;
  }

  // Configuration from environment or defaults
  const char* envPeers = std::getenv("NUM_VIRTUAL_PEERS");
  int numVirtualPeers = envPeers ? std::atoi(envPeers) : (numRanks - 1);
  if (numVirtualPeers < 1) {
    numVirtualPeers = numRanks - 1;
  }

  // AFD FTA scenario configuration
  // Default: 4 FFN ranks, 565 attn ranks, D=3072, B=96, microbatch=4
  // tokens_per_peer = batch_size / num_microbatches / num_ffn_ranks
  //                 = 96 / 4 / 4 = 6
  constexpr int kNumMicrobatches = 4;
  constexpr int kNumFfnRanks = 4;

  const char* envBatch = std::getenv("BATCH_SIZE");
  int batchSize = envBatch ? std::atoi(envBatch) : 96;

  const char* envDim = std::getenv("MODEL_DIM");
  int modelDim = envDim ? std::atoi(envDim) : 3072;

  int tokensPerPeer = batchSize / kNumMicrobatches / kNumFfnRanks;

  constexpr int kSignalId = 0;

  // FTA send: 1 put (tokens) + 1 signal per peer
  // Data per peer = tokens_per_peer × model_dim × 2 (bf16)
  const std::size_t tokenBytes =
      static_cast<std::size_t>(tokensPerPeer) * modelDim * 2; // bf16

  // FTA pattern: only tokens are sent back (1 put + 1 signal)
  // ATF pattern would have 4 puts (tokens, scores, indices, splits)
  // Use sizes[0] = tokenBytes, sizes[1..3] = 0 for FTA
  const std::size_t totalPutSize = tokenBytes;

  int peersPerTransport = numRanks - 1;
  int numTransports =
      (numVirtualPeers + peersPerTransport - 1) / peersPerTransport;

  XLOGF(
      INFO,
      "Rank {}: ScatterSignal config: virtualPeers={}, B={}, D={}, "
      "ffn_ranks={}, tokens/peer={}, transports={}, putSize={} bytes/peer",
      globalRank,
      numVirtualPeers,
      batchSize,
      modelDim,
      kNumFfnRanks,
      tokensPerPeer,
      numTransports,
      totalPutSize);

  try {
    MultipeerIbgdaTransportConfig transportConfig{
        .cudaDevice = localRank,
        .qpDepth = 8192,
    };

    // Discover available NICs for this GPU and use multi-rail
    // (alternate transports across NICs for higher aggregate bandwidth).
    // Only use NICs on the same NUMA node as the GPU (same path type as best).
    comms::pipes::GpuNicDiscovery discovery(localRank);
    const auto& candidates = discovery.getCandidates();
    std::vector<std::string> nicNames;
    auto bestPath = candidates[0].pathType;
    for (const auto& c : candidates) {
      if (c.bandwidthGbps >= 400 && c.pathType == bestPath) {
        nicNames.push_back(c.name);
      }
    }
    if (nicNames.empty()) {
      nicNames.push_back(candidates[0].name);
    }
    int numNics = static_cast<int>(nicNames.size());
    XLOGF(INFO, "Rank {}: multi-rail with {} NICs", globalRank, numNics);
    for (int i = 0; i < numNics; i++) {
      // Find the candidate to get its path type
      for (const auto& c : candidates) {
        if (c.name == nicNames[i]) {
          XLOGF(
              INFO,
              "  NIC {}: {} ({}, {} Gb/s)",
              i,
              c.name,
              comms::pipes::pathTypeToString(c.pathType),
              c.bandwidthGbps);
          break;
        }
      }
    }

    // Create multiple transport instances to get enough QPs.
    // Share a single MpiBootstrap to avoid exhausting MPI communicators
    // (each MpiBootstrap creates a new MPI communicator; ~209 exhausts them).
    auto sharedBootstrap = std::make_shared<meta::comms::MpiBootstrap>();

    std::vector<std::unique_ptr<MultipeerIbgdaTransport>> transports;
    std::vector<DeviceBuffer> dataBuffers;
    std::vector<IbgdaLocalBuffer> localDataBufs;
    // remoteBufs[t][peerIndex] = remote buffer for transport t, peer peerIndex
    std::vector<std::vector<IbgdaRemoteBuffer>> allRemoteBufs;
    std::vector<DeviceBuffer> signalBuffers;
    std::vector<IbgdaLocalBuffer> localSignalBufs;
    std::vector<std::vector<IbgdaRemoteBuffer>> allRemoteSignalBufs;

    bool localFailed = false;
    std::string failReason;
    for (int t = 0; t < numTransports; t++) {
      // Alternate NICs across transports for multi-rail
      auto railConfig = transportConfig;
      railConfig.gpuNicMap[localRank] = {nicNames[t % numNics]};

      std::unique_ptr<MultipeerIbgdaTransport> transport;
      int localOk = 1;
      if (!localFailed) {
        try {
          transport = std::make_unique<MultipeerIbgdaTransport>(
              globalRank, numRanks, sharedBootstrap, railConfig);
        } catch (const std::exception& e) {
          localOk = 0;
          localFailed = true;
          failReason = e.what();
        }
      } else {
        localOk = 0;
      }

      // Coordinate across ranks: if any rank failed, all must stop together
      // to avoid deadlock on the collective exchange()/exchangeBuffer() calls.
      int globalOk = 0;
      MPI_CHECK(MPI_Allreduce(
          &localOk, &globalOk, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
      if (!globalOk) {
        XLOGF(
            INFO,
            "Rank {}: stopping transport creation at {}/{} (created {})",
            globalRank,
            t,
            numTransports,
            transports.size());
        break;
      }

      transport->exchange();

      // Each transport registers its own data and signal buffers
      dataBuffers.emplace_back(totalPutSize);
      auto localBuf =
          transport->registerBuffer(dataBuffers.back().get(), totalPutSize);
      localDataBufs.push_back(localBuf);

      auto remoteBufs = transport->exchangeBuffer(localBuf);
      allRemoteBufs.push_back(std::move(remoteBufs));

      signalBuffers.emplace_back(sizeof(uint64_t));
      CUDA_CHECK_VOID(
          cudaMemset(signalBuffers.back().get(), 0, sizeof(uint64_t)));
      auto localSignalBuf = transport->registerBuffer(
          signalBuffers.back().get(), sizeof(uint64_t));
      localSignalBufs.push_back(localSignalBuf);

      auto remoteSignalBufsForTransport =
          transport->exchangeBuffer(localSignalBuf);
      allRemoteSignalBufs.push_back(std::move(remoteSignalBufsForTransport));

      transports.push_back(std::move(transport));
    }

    if (transports.empty()) {
      GTEST_SKIP() << "No transports created successfully"
                   << (failReason.empty() ? "" : ": " + failReason);
    }

    // Adjust numVirtualPeers to match actual transports created
    int actualVirtualPeers =
        static_cast<int>(transports.size()) * peersPerTransport;
    if (actualVirtualPeers < numVirtualPeers) {
      XLOGF(
          INFO,
          "Rank {}: reduced virtualPeers from {} to {} ({} transports created)",
          globalRank,
          numVirtualPeers,
          actualVirtualPeers,
          transports.size());
      numVirtualPeers = actualVirtualPeers;
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Only rank 0 runs the scatter benchmark
    if (globalRank == 0) {
      // Collect P2pIbgdaTransportDevice*, local buffers, and remote buffers
      // for all virtual peers
      std::vector<P2pIbgdaTransportDevice*> peerHandles(numVirtualPeers);
      std::vector<IbgdaLocalBuffer> peerLocalBufs(numVirtualPeers);
      std::vector<IbgdaRemoteBuffer> peerRemoteBufs(numVirtualPeers);
      std::vector<IbgdaRemoteBuffer> peerRemoteSignalBufs(numVirtualPeers);

      for (int v = 0; v < numVirtualPeers; v++) {
        int tIdx = v / peersPerTransport;
        int peerRank = 1 + (v % peersPerTransport);
        peerHandles[v] = transports[tIdx]->getP2pTransportDevice(peerRank);
        peerLocalBufs[v] = localDataBufs[tIdx];

        // Map peerRank to peer index (ranks < myRank keep same index)
        int peerIndex = (peerRank < globalRank) ? peerRank : (peerRank - 1);
        assert(tIdx < allRemoteBufs.size());
        assert(tIdx < allRemoteSignalBufs.size());
        peerRemoteBufs[v] = allRemoteBufs[tIdx][peerIndex];
        peerRemoteSignalBufs[v] = allRemoteSignalBufs[tIdx][peerIndex];
      }

      // Copy arrays to device
      P2pIbgdaTransportDevice** d_transports;
      IbgdaLocalBuffer* d_localBufs;
      IbgdaRemoteBuffer* d_remoteBufs;
      IbgdaRemoteBuffer* d_remoteSignalBufs;
      unsigned long long* d_totalCycles;

      CUDA_CHECK_VOID(cudaMalloc(
          &d_transports, numVirtualPeers * sizeof(P2pIbgdaTransportDevice*)));
      CUDA_CHECK_VOID(
          cudaMalloc(&d_localBufs, numVirtualPeers * sizeof(IbgdaLocalBuffer)));
      CUDA_CHECK_VOID(cudaMalloc(
          &d_remoteBufs, numVirtualPeers * sizeof(IbgdaRemoteBuffer)));
      CUDA_CHECK_VOID(cudaMalloc(
          &d_remoteSignalBufs, numVirtualPeers * sizeof(IbgdaRemoteBuffer)));
      CUDA_CHECK_VOID(
          cudaMalloc(&d_totalCycles, 2 * sizeof(unsigned long long)));

      CUDA_CHECK_VOID(cudaMemcpy(
          d_transports,
          peerHandles.data(),
          numVirtualPeers * sizeof(P2pIbgdaTransportDevice*),
          cudaMemcpyHostToDevice));
      CUDA_CHECK_VOID(cudaMemcpy(
          d_localBufs,
          peerLocalBufs.data(),
          numVirtualPeers * sizeof(IbgdaLocalBuffer),
          cudaMemcpyHostToDevice));
      CUDA_CHECK_VOID(cudaMemcpy(
          d_remoteBufs,
          peerRemoteBufs.data(),
          numVirtualPeers * sizeof(IbgdaRemoteBuffer),
          cudaMemcpyHostToDevice));
      CUDA_CHECK_VOID(cudaMemcpy(
          d_remoteSignalBufs,
          peerRemoteSignalBufs.data(),
          numVirtualPeers * sizeof(IbgdaRemoteBuffer),
          cudaMemcpyHostToDevice));

      XLOGF(
          INFO,
          "Rank 0: launching scatter-signal benchmark with {} virtual peers, "
          "{} QPs, {} tokens/peer",
          numVirtualPeers,
          numVirtualPeers,
          tokensPerPeer);

      std::size_t totalBytes = totalPutSize * numVirtualPeers;

      XLOGF(INFO, "\n");
      XLOGF(
          INFO,
          "========================================================================");
      XLOGF(INFO, "  AFD FTA Scatter-Signal IBGDA Benchmark Results");
      XLOGF(
          INFO,
          "========================================================================");
      XLOGF(
          INFO,
          "  Scenario: 1 FFN rank → {} attn peers (FTA send)",
          numVirtualPeers);
      XLOGF(
          INFO,
          "  B={}, D={}, ffn_ranks={}, tokens/peer={}",
          batchSize,
          modelDim,
          kNumFfnRanks,
          tokensPerPeer);
      XLOGF(
          INFO,
          "  QPs: {},  Transports: {},  {} B/peer",
          numVirtualPeers,
          numTransports,
          tokenBytes);
      XLOGF(INFO, "  Total data per scatter: {:.2f} MB", totalBytes / 1e6);
      XLOGF(INFO, "  Batch iterations: {}", kScatterBenchIters);
      XLOGF(
          INFO,
          "------------------------------------------------------------------------");
      XLOGF(
          INFO,
          "  {:>20} {:>14} {:>14} {:>14}",
          "Mode",
          "Scatter (us)",
          "Per-peer (us)",
          "BW (GB/s)");
      XLOGF(
          INFO,
          "------------------------------------------------------------------------");

      // --- 1-put mode (FTA: 1 put tokens + 1 signal per peer) ---
      launchIbgdaScatterSignalSinglePutBatch(
          d_transports,
          d_localBufs,
          d_remoteBufs,
          d_remoteSignalBufs,
          totalPutSize,
          numVirtualPeers,
          kSignalId,
          kScatterBenchIters,
          d_totalCycles,
          stream_);
      CUDA_CHECK_VOID(cudaStreamSynchronize(stream_));

      {
        unsigned long long cycles[2];
        CUDA_CHECK_VOID(cudaMemcpy(
            cycles,
            d_totalCycles,
            2 * sizeof(unsigned long long),
            cudaMemcpyDeviceToHost));

        float lat = cyclesToUs(cycles[0]) / kScatterBenchIters;
        float postLat = cyclesToUs(cycles[1]) / kScatterBenchIters;
        XLOGF(
            INFO,
            "  {:>20} {:>14.2f} {:>14.2f} {:>14.2f}  (post={:.2f} us, wait={:.2f} us)",
            "1-put+signal",
            lat,
            lat / numVirtualPeers,
            (totalBytes / 1e9f) / (lat / 1e6f),
            postLat,
            lat - postLat);
      }

      XLOGF(
          INFO,
          "========================================================================\n");

      CUDA_CHECK_VOID(cudaFree(d_transports));
      CUDA_CHECK_VOID(cudaFree(d_localBufs));
      CUDA_CHECK_VOID(cudaFree(d_remoteBufs));
      CUDA_CHECK_VOID(cudaFree(d_remoteSignalBufs));
      CUDA_CHECK_VOID(cudaFree(d_totalCycles));
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }
}

} // namespace comms::pipes::benchmark

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
