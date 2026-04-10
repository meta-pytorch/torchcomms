// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>
#include <mpi.h>

#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include "comms/pipes/DeviceSpan.cuh"
#include "comms/pipes/MultiPeerIbgdaTransportSetup.h"
#include "comms/pipes/MultipeerIbgdaTransport.h"
#include "comms/pipes/P2pIbgdaTransportState.h"
#include "comms/pipes/P2pSelfTransportDevice.cuh"
#include "comms/pipes/Transport.cuh"
#include "comms/pipes/collectives/AllToAllv.h"
#include "comms/pipes/collectives/benchmarks/AllToAllvIbgdaBenchmarkKernels.cuh"
#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::DeviceBuffer;
using meta::comms::MpiBaseTestFixture;

#define CUDACHECK_BENCH(cmd)                                 \
  do {                                                       \
    cudaError_t err = (cmd);                                 \
    if (err != cudaSuccess) {                                \
      XLOGF(ERR, "CUDA error: {}", cudaGetErrorString(err)); \
      std::abort();                                          \
    }                                                        \
  } while (0)

namespace comms::pipes {

struct BenchmarkResult {
  std::string label;
  size_t perPeerBytes;
  int numBlocks;
  int numThreads;
  int numIterations;
  double latencyUs;
  double bandwidthGBps;
};

class AllToAllvIbgdaBenchmarkFixture : public MpiBaseTestFixture {
 protected:
  std::vector<BenchmarkResult> results_;

  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    CUDACHECK_BENCH(cudaSetDevice(localRank));
    results_.clear();
  }

  void TearDown() override {
    if (globalRank == 0 && !results_.empty()) {
      printResultsTable();
    }
    MpiBaseTestFixture::TearDown();
  }

  void printResultsTable() {
    fprintf(
        stderr,
        "\n"
        "╔══════════════════════════════╦════════════╦════════╦═════════╦═══════╦═══════════════╦═══════════════╗\n"
        "║ Label                        ║ Per-Peer   ║ Blocks ║ Threads ║ Iters ║ Latency (us)  ║ BW (GB/s)     ║\n"
        "╠══════════════════════════════╬════════════╬════════╬═════════╬═══════╬═══════════════╬═══════════════╣\n");
    for (const auto& r : results_) {
      std::string sizeStr;
      if (r.perPeerBytes >= 1048576) {
        sizeStr = std::to_string(r.perPeerBytes / 1048576) + " MB";
      } else if (r.perPeerBytes >= 1024) {
        sizeStr = std::to_string(r.perPeerBytes / 1024) + " KB";
      } else {
        sizeStr = std::to_string(r.perPeerBytes) + " B";
      }
      fprintf(
          stderr,
          "║ %-28s ║ %10s ║ %6d ║ %7d ║ %5d ║ %13.1f ║ %13.2f ║\n",
          r.label.c_str(),
          sizeStr.c_str(),
          r.numBlocks,
          r.numThreads,
          r.numIterations,
          r.latencyUs,
          r.bandwidthGBps);
    }
    fprintf(
        stderr,
        "╚══════════════════════════════╩════════════╩════════╩═════════╩═══════╩═══════════════╩═══════════════╝\n\n");
  }

  std::unique_ptr<MultipeerIbgdaTransport> createTransport() {
    MultipeerIbgdaTransportConfig config{
        .cudaDevice = localRank,
    };
    auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
    auto transport = std::make_unique<MultipeerIbgdaTransport>(
        globalRank, numRanks, bootstrap, config);
    transport->exchange();
    return transport;
  }

  Transport* buildTransportsArray(
      MultipeerIbgdaTransport& ibgdaTransport,
      MultiPeerIbgdaTransportSetup& setup,
      int myRank,
      int nRanks) {
    const size_t arrayBytes = nRanks * sizeof(Transport);
    auto* h_transports = static_cast<Transport*>(
        std::aligned_alloc(alignof(Transport), arrayBytes));
    const auto& hostStates = setup.getHostPeerStates();
    uint64_t* iterCounter = setup.getIterationCounter();
    for (int r = 0; r < nRanks; ++r) {
      if (r == myRank) {
        new (&h_transports[r]) Transport(P2pSelfTransportDevice{});
      } else {
        P2pIbgdaTransportDevice* devPtr =
            ibgdaTransport.buildP2pTransportDevice(
                r, hostStates[r], &iterCounter[r * 2], &iterCounter[r * 2 + 1]);
        new (&h_transports[r]) Transport(devPtr);
      }
    }
    Transport* d_transports = nullptr;
    cudaMalloc(&d_transports, arrayBytes);
    cudaMemcpy(d_transports, h_transports, arrayBytes, cudaMemcpyHostToDevice);
    for (int r = 0; r < nRanks; ++r) {
      h_transports[r].~Transport();
    }
    std::free(h_transports);
    return d_transports;
  }

  void runBenchmark(
      MultipeerIbgdaTransport& ibgdaTransport,
      MultiPeerIbgdaTransportSetup& setup,
      size_t perPeerBytes,
      int numBlocks,
      int numThreads,
      int numIterations,
      const std::string& label) {
    const size_t totalBytes = perPeerBytes * numRanks;

    Transport* d_transports =
        buildTransportsArray(ibgdaTransport, setup, globalRank, numRanks);

    DeviceBuffer sendBuffer(totalBytes);
    DeviceBuffer recvBuffer(totalBytes);

    // Build ChunkInfo
    std::vector<ChunkInfo> h_chunks;
    h_chunks.reserve(numRanks);
    for (int rank = 0; rank < numRanks; rank++) {
      h_chunks.emplace_back(rank * perPeerBytes, perPeerBytes);
    }
    DeviceBuffer d_send_chunks(sizeof(ChunkInfo) * numRanks);
    DeviceBuffer d_recv_chunks(sizeof(ChunkInfo) * numRanks);
    CUDACHECK_BENCH(cudaMemcpy(
        d_send_chunks.get(),
        h_chunks.data(),
        sizeof(ChunkInfo) * numRanks,
        cudaMemcpyHostToDevice));
    CUDACHECK_BENCH(cudaMemcpy(
        d_recv_chunks.get(),
        h_chunks.data(),
        sizeof(ChunkInfo) * numRanks,
        cudaMemcpyHostToDevice));

    DeviceSpan<Transport> transports_span(
        d_transports, static_cast<uint32_t>(numRanks));
    DeviceSpan<ChunkInfo> send_infos(
        static_cast<ChunkInfo*>(d_send_chunks.get()), numRanks);
    DeviceSpan<ChunkInfo> recv_infos(
        static_cast<ChunkInfo*>(d_recv_chunks.get()), numRanks);

    // Warmup
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < 5; i++) {
      all_to_allv(
          recvBuffer.get(),
          sendBuffer.get(),
          globalRank,
          transports_span,
          send_infos,
          recv_infos,
          std::chrono::milliseconds{0},
          nullptr,
          numBlocks,
          numThreads,
          std::nullopt);
    }
    CUDACHECK_BENCH(cudaDeviceSynchronize());

    // Timed run
    MPI_Barrier(MPI_COMM_WORLD);

    cudaEvent_t start, stop;
    CUDACHECK_BENCH(cudaEventCreate(&start));
    CUDACHECK_BENCH(cudaEventCreate(&stop));

    CUDACHECK_BENCH(cudaEventRecord(start));
    for (int i = 0; i < numIterations; i++) {
      all_to_allv(
          recvBuffer.get(),
          sendBuffer.get(),
          globalRank,
          transports_span,
          send_infos,
          recv_infos,
          std::chrono::milliseconds{0},
          nullptr,
          numBlocks,
          numThreads,
          std::nullopt);
    }
    CUDACHECK_BENCH(cudaEventRecord(stop));
    CUDACHECK_BENCH(cudaEventSynchronize(stop));

    float elapsedMs = 0;
    CUDACHECK_BENCH(cudaEventElapsedTime(&elapsedMs, start, stop));

    double avgLatencyUs = (elapsedMs * 1000.0) / numIterations;
    double totalDataBytes = static_cast<double>(perPeerBytes) * (numRanks - 1);
    double bandwidthGBps =
        (totalDataBytes / (avgLatencyUs * 1e-6)) / (1024.0 * 1024.0 * 1024.0);

    if (globalRank == 0) {
      XLOGF(
          INFO,
          "BENCHMARK [{}]: perPeer={}B, blocks={}, threads={}, "
          "iters={}, latency={:.1f}us, BW={:.2f}GB/s",
          label,
          perPeerBytes,
          numBlocks,
          numThreads,
          numIterations,
          avgLatencyUs,
          bandwidthGBps);

      results_.push_back(
          BenchmarkResult{
              .label = label,
              .perPeerBytes = perPeerBytes,
              .numBlocks = numBlocks,
              .numThreads = numThreads,
              .numIterations = numIterations,
              .latencyUs = avgLatencyUs,
              .bandwidthGBps = bandwidthGBps,
          });
    }

    CUDACHECK_BENCH(cudaEventDestroy(start));
    CUDACHECK_BENCH(cudaEventDestroy(stop));
    cudaFree(d_transports);

    MPI_Barrier(MPI_COMM_WORLD);
  }
};

// Message size sweep: 1KB → 128MB
TEST_F(AllToAllvIbgdaBenchmarkFixture, MessageSizeSweep) {
  std::unique_ptr<MultipeerIbgdaTransport> transport;
  try {
    transport = createTransport();
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }

  // Use large staging buffer for benchmark
  MultiPeerIbgdaTransportSetupConfig setupConfig{
      .dataBufferSize = 1024 * 1024, // 1MB staging
      .chunkSize = 1024 * 1024,
      .pipelineDepth = 4,
  };

  MultiPeerIbgdaTransportSetup setup(
      *transport, globalRank, numRanks, setupConfig);
  setup.exchangeBuffers();

  const std::vector<size_t> messageSizes = {
      1024,
      4096,
      16384,
      65536,
      262144,
      1048576,
      4194304,
      16777216,
      67108864,
      134217728,
  };

  // Use 1 block baseline. Minimum threads = 2 (send/recv) * nranks * 32
  // (warpSize) for partition_interleaved to have enough groups.
  const int minThreads = 2 * numRanks * 32;
  for (size_t msgSize : messageSizes) {
    std::string label = "MsgSweep_" + std::to_string(msgSize / 1024) + "KB";
    int iters = (msgSize <= 65536) ? 1000 : (msgSize <= 4194304 ? 100 : 10);
    runBenchmark(*transport, setup, msgSize, 1, minThreads, iters, label);
  }
}

// Block count sweep
TEST_F(AllToAllvIbgdaBenchmarkFixture, BlockCountSweep) {
  std::unique_ptr<MultipeerIbgdaTransport> transport;
  try {
    transport = createTransport();
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }

  MultiPeerIbgdaTransportSetupConfig setupConfig{
      .dataBufferSize = 1024 * 1024,
      .chunkSize = 1024 * 1024,
      .pipelineDepth = 4,
  };

  MultiPeerIbgdaTransportSetup setup(
      *transport, globalRank, numRanks, setupConfig);
  setup.exchangeBuffers();

  const size_t msgSize = 1048576; // 1MB per peer
  // Minimum threads = 2 (send/recv) * nranks * 32 (warpSize)
  const int minThreads = 2 * numRanks * 32;
  const std::vector<int> blockCounts = {1, 2, 4, 8, 16, 32};

  for (int blocks : blockCounts) {
    std::string label = "BlockSweep_" + std::to_string(blocks) + "b";
    runBenchmark(*transport, setup, msgSize, blocks, minThreads, 100, label);
  }
}

// Thread count sweep
TEST_F(AllToAllvIbgdaBenchmarkFixture, ThreadCountSweep) {
  std::unique_ptr<MultipeerIbgdaTransport> transport;
  try {
    transport = createTransport();
  } catch (const std::exception& e) {
    GTEST_SKIP() << "IBGDA transport not available: " << e.what();
  }

  MultiPeerIbgdaTransportSetupConfig setupConfig{
      .dataBufferSize = 256 * 1024,
      .chunkSize = 256 * 1024,
      .pipelineDepth = 4,
  };

  MultiPeerIbgdaTransportSetup setup(
      *transport, globalRank, numRanks, setupConfig);
  setup.exchangeBuffers();

  const size_t msgSize = 1048576; // 1MB per peer
  // Minimum threads = 2 (send/recv) * nranks * 32 (warpSize)
  const int minThreads = 2 * numRanks * 32;
  const std::vector<int> threadCounts = {minThreads, 256, 512};

  for (int totalThreads : threadCounts) {
    const int maxTPB = 256;
    const int tpb = std::min(totalThreads, maxTPB);
    const int blocks = std::max(1, totalThreads / tpb);
    std::string label = "ThreadSweep_" + std::to_string(totalThreads) + "t";
    runBenchmark(*transport, setup, msgSize, blocks, tpb, 100, label);
  }
}

// Pipeline depth sweep
TEST_F(AllToAllvIbgdaBenchmarkFixture, PipelineDepthSweep) {
  const size_t msgSize = 1048576; // 1MB per peer
  const std::vector<int> pipelineDepths = {1, 2, 4, 8};

  for (int depth : pipelineDepths) {
    // Create fresh transport per iteration to avoid NIC state conflicts
    // when deregistering/re-registering buffers with different sizes.
    std::unique_ptr<MultipeerIbgdaTransport> transport;
    try {
      transport = createTransport();
    } catch (const std::exception& e) {
      GTEST_SKIP() << "IBGDA transport not available: " << e.what();
    }

    MultiPeerIbgdaTransportSetupConfig setupConfig{
        .dataBufferSize = 256 * 1024,
        .chunkSize = 256 * 1024,
        .pipelineDepth = depth,
    };

    MultiPeerIbgdaTransportSetup setup(
        *transport, globalRank, numRanks, setupConfig);
    setup.exchangeBuffers();

    std::string label = "PipeDepth_" + std::to_string(depth);
    // Minimum threads = 2 (send/recv) * nranks * 32 (warpSize)
    int minThreads = 2 * numRanks * 32;
    runBenchmark(*transport, setup, msgSize, 1, minThreads, 100, label);
  }
}

// Chunk size sweep: fix message at 1MB, sweep chunkSize
TEST_F(AllToAllvIbgdaBenchmarkFixture, ChunkSizeSweep) {
  const size_t msgSize = 1048576; // 1MB per peer
  const std::vector<size_t> chunkSizes = {
      1024,
      4096,
      16384,
      65536,
      262144,
      1048576,
  };

  for (size_t chunkSize : chunkSizes) {
    std::unique_ptr<MultipeerIbgdaTransport> transport;
    try {
      transport = createTransport();
    } catch (const std::exception& e) {
      GTEST_SKIP() << "IBGDA transport not available: " << e.what();
    }

    MultiPeerIbgdaTransportSetupConfig setupConfig{
        .dataBufferSize = chunkSize,
        .chunkSize = chunkSize,
        .pipelineDepth = 4,
    };

    MultiPeerIbgdaTransportSetup setup(
        *transport, globalRank, numRanks, setupConfig);
    setup.exchangeBuffers();

    std::string sizeLabel;
    if (chunkSize >= 1048576) {
      sizeLabel = std::to_string(chunkSize / 1048576) + "MB";
    } else if (chunkSize >= 1024) {
      sizeLabel = std::to_string(chunkSize / 1024) + "KB";
    } else {
      sizeLabel = std::to_string(chunkSize) + "B";
    }
    std::string label = "ChunkSweep_" + sizeLabel;

    int minThreads = 2 * numRanks * 32;
    runBenchmark(*transport, setup, msgSize, 1, minThreads, 100, label);
  }
}

// Raw IBGDA P2P send/recv sweep: rank 0 → rank 7 via Pipes IBGDA
// Sweeps pipeline depth, thread count, and chunk size to find optimal config.
// Uses GPU 0 and GPU 7 which sit on opposite PCIe switches with different
// NICs, giving independent NIC bandwidth in each direction.
TEST_F(AllToAllvIbgdaBenchmarkFixture, P2pSendRecvSweep) {
  ASSERT_GE(numRanks, 2) << "P2pSendRecvSweep requires at least 2 ranks";

  constexpr int kSenderRank = 0;
  const int kRecverRank = numRanks - 1;
  constexpr size_t kMsgBytes = 1024ULL * 1024 * 1024; // 1GB
  constexpr int kWarmupIter = 10;
  constexpr int kTimedIter = 100;

  bool isSender = (globalRank == kSenderRank);
  bool isRecver = (globalRank == kRecverRank);
  bool isParticipant = isSender || isRecver;

  struct P2pConfig {
    std::string label;
    size_t chunkSize;
    int pipelineDepth;
    int numBlocks;
    int numThreads;
    int maxChannelsPerPeer = 1;
  };

  std::vector<P2pConfig> configs = {
      // Chunk size sweep (pipelineDepth=4, 2 blocks x 256 threads)
      {"chunk_64KB_pipe4_2x256_2ch", 64 * 1024, 4, 2, 256, 2},
      {"chunk_256KB_pipe4_2x256_2ch", 256 * 1024, 4, 2, 256, 2},
      {"chunk_512KB_pipe4_2x256_2ch", 512 * 1024, 4, 2, 256, 2},
      {"chunk_1MB_pipe4_2x256_2ch", 1024 * 1024, 4, 2, 256, 2},

      // Pipeline depth sweep (chunkSize=1MB, 2 blocks x 256 threads)
      {"chunk_1MB_pipe1_2x256_2ch", 1024 * 1024, 1, 2, 256, 2},
      {"chunk_1MB_pipe2_2x256_2ch", 1024 * 1024, 2, 2, 256, 2},
      {"chunk_1MB_pipe8_2x256_2ch", 1024 * 1024, 8, 2, 256, 2},

      // Thread count sweep (chunkSize=1MB, pipelineDepth=4, 256 tpb)
      {"chunk_1MB_pipe4_2x128_2ch", 1024 * 1024, 4, 2, 128, 2},
      {"chunk_1MB_pipe4_2x256_2ch_t", 1024 * 1024, 4, 2, 256, 2},
      {"chunk_1MB_pipe4_4x256_4ch", 1024 * 1024, 4, 4, 256, 4},

      // Multi-block sweep (chunkSize=1MB, pipelineDepth=4, 256 threads/block)
      {"chunk_1MB_pipe4_8x256_8ch", 1024 * 1024, 4, 8, 256, 8},
      {"chunk_1MB_pipe4_16x256_16ch", 1024 * 1024, 4, 16, 256, 16},

      // Multi-block per peer sweep: blocks x channels coverage
      // (chunkSize=1MB, pipelineDepth=4, 256 threads/block)
      // channelsPerBlock = min(warpsPerBlock, maxChannelsPerPeer)
      //   256 threads = 8 warps -> channelsPerBlock = min(8, maxCh)
      {"chunk_1MB_pipe4_1x256_1ch", 1024 * 1024, 4, 1, 256, 1},
      {"chunk_1MB_pipe4_2x256_2ch_b", 1024 * 1024, 4, 2, 256, 2},
      {"chunk_1MB_pipe4_4x256_4ch_b", 1024 * 1024, 4, 4, 256, 4},
      {"chunk_1MB_pipe4_2x256_8ch", 1024 * 1024, 4, 2, 256, 8},
      {"chunk_1MB_pipe4_4x256_8ch", 1024 * 1024, 4, 4, 256, 8},
      {"chunk_1MB_pipe4_8x256_8ch_b", 1024 * 1024, 4, 8, 256, 8},
  };

  for (const auto& cfg : configs) {
    std::unique_ptr<MultipeerIbgdaTransport> transport;
    try {
      transport = createTransport();
    } catch (const std::exception& e) {
      GTEST_SKIP() << "IBGDA transport not available: " << e.what();
    }

    MultiPeerIbgdaTransportSetupConfig setupConfig{
        .dataBufferSize = cfg.chunkSize,
        .chunkSize = cfg.chunkSize,
        .pipelineDepth = cfg.pipelineDepth,
    };

    MultiPeerIbgdaTransportSetup setup(
        *transport, globalRank, numRanks, setupConfig, cfg.maxChannelsPerPeer);
    setup.exchangeBuffers();

    P2pIbgdaTransportDevice* dev = nullptr;
    DeviceBuffer buf(isParticipant ? kMsgBytes : 1);

    if (isParticipant) {
      int peerRank = isSender ? kRecverRank : kSenderRank;
      const auto& hostStates = setup.getHostPeerStates();
      uint64_t* iterCounter = setup.getIterationCounter();
      int maxCh = hostStates[peerRank].maxChannelsPerPeer;
      dev = transport->buildP2pTransportDevice(
          peerRank,
          hostStates[peerRank],
          &iterCounter[peerRank * maxCh * 2],
          &iterCounter[peerRank * maxCh * 2 + maxCh]);
      CUDACHECK_BENCH(cudaMemset(buf.get(), globalRank & 0xFF, kMsgBytes));
    }

    // Warmup
    MPI_Barrier(MPI_COMM_WORLD);
    if (isSender) {
      benchmark::launchIbgdaSendBench(
          dev,
          buf.get(),
          kMsgBytes,
          kWarmupIter,
          nullptr,
          cfg.numBlocks,
          cfg.numThreads);
    } else if (isRecver) {
      benchmark::launchIbgdaRecvBench(
          dev,
          buf.get(),
          kMsgBytes,
          kWarmupIter,
          nullptr,
          cfg.numBlocks,
          cfg.numThreads);
    }
    if (isParticipant) {
      CUDACHECK_BENCH(cudaDeviceSynchronize());
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Timed run
    cudaEvent_t start, stop;
    if (isParticipant) {
      CUDACHECK_BENCH(cudaEventCreate(&start));
      CUDACHECK_BENCH(cudaEventCreate(&stop));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (isSender) {
      CUDACHECK_BENCH(cudaEventRecord(start));
      benchmark::launchIbgdaSendBench(
          dev,
          buf.get(),
          kMsgBytes,
          kTimedIter,
          nullptr,
          cfg.numBlocks,
          cfg.numThreads);
      CUDACHECK_BENCH(cudaEventRecord(stop));
      CUDACHECK_BENCH(cudaEventSynchronize(stop));
    } else if (isRecver) {
      CUDACHECK_BENCH(cudaEventRecord(start));
      benchmark::launchIbgdaRecvBench(
          dev,
          buf.get(),
          kMsgBytes,
          kTimedIter,
          nullptr,
          cfg.numBlocks,
          cfg.numThreads);
      CUDACHECK_BENCH(cudaEventRecord(stop));
      CUDACHECK_BENCH(cudaEventSynchronize(stop));
    }

    if (isSender) {
      float elapsedMs = 0;
      CUDACHECK_BENCH(cudaEventElapsedTime(&elapsedMs, start, stop));
      double avgLatUs = (elapsedMs * 1000.0) / kTimedIter;
      double bwGBps = (kMsgBytes / 1e9) / (avgLatUs * 1e-6);

      results_.push_back(
          BenchmarkResult{
              .label = cfg.label,
              .perPeerBytes = kMsgBytes,
              .numBlocks = cfg.numBlocks,
              .numThreads = cfg.numThreads,
              .numIterations = kTimedIter,
              .latencyUs = avgLatUs,
              .bandwidthGBps = bwGBps,
          });
    }

    if (isParticipant) {
      CUDACHECK_BENCH(cudaEventDestroy(start));
      CUDACHECK_BENCH(cudaEventDestroy(stop));
      if (dev) {
        cudaFree(dev);
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

} // namespace comms::pipes

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  // GTest takes ownership of the environment pointer via
  // AddGlobalTestEnvironment, so we must NOT wrap it in unique_ptr (double-free
  // → SIGSEGV on exit).
  ::testing::AddGlobalTestEnvironment(new meta::comms::MPIEnvironmentBase());
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
