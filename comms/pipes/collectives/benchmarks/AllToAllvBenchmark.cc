// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>
#include <nccl.h>

#include "comms/common/CudaWrap.h"
#include "comms/pipes/MultiPeerNvlTransport.h"
#include "comms/pipes/benchmarks/BenchmarkMacros.h"
#include "comms/pipes/collectives/benchmarks/CollectiveBenchmark.cuh"
#include "comms/testinfra/mpi/MpiBootstrap.h"
#include "comms/testinfra/mpi/MpiTestUtils.h"
#include "comms/utils/CudaRAII.h"

#include <iomanip>
#include <sstream>
#include <vector>

using meta::comms::CudaEvent;
using meta::comms::DeviceBuffer;
using meta::comms::MpiBaseTestFixture;
using meta::comms::MPIEnvironmentBase;

namespace comms::pipes::benchmark {

namespace {
/**
 * Test configuration for AllToAllv benchmark.
 */
struct AllToAllvBenchmarkConfig {
  std::size_t bytesPerPeer; // Message size per peer (equal for all peers)
  int numBlocks;
  int numThreads;
  std::size_t pipelineDepth = 4;
  std::size_t chunkSize = 512 * 1024; // 512KB default
  std::size_t dataBufferSize = 2048; // Data buffer size for P2P transport
  bool spreadClusterLaunch = false; // Use spread cluster kernel launch
  std::string name;
};

/**
 * Result struct for collecting benchmark data.
 */
struct AllToAllvBenchmarkResult {
  std::string testName;
  std::size_t bytesPerPeer{};
  std::size_t totalBytes{}; // Total across all peers
  std::size_t pipelineDepth{};
  std::size_t chunkSize{};
  float ncclBandwidth{}; // GB/s
  float alltoallvBandwidth{}; // GB/s
  float ncclLatency{}; // microseconds
  float alltoallvLatency{}; // microseconds
  float speedup{}; // AllToAllv / NCCL
};

class AllToAllvBenchmarkFixture : public MpiBaseTestFixture {
 protected:
  void SetUp() override {
    MpiBaseTestFixture::SetUp();
    // Use localRank for cudaSetDevice since each node has its own set of GPUs
    // globalRank would fail on multi-node setups where rank > num_gpus_per_node
    CUDA_CHECK_VOID(cudaSetDevice(localRank));

    // Initialize NCCL with default channel settings
    NCCL_CHECK_VOID(
        ncclCommInitRank(&ncclComm_, numRanks, getNCCLId(), globalRank));
    CUDA_CHECK_VOID(cudaStreamCreate(&stream_));
  }

  void TearDown() override {
    NCCL_CHECK_VOID(ncclCommDestroy(ncclComm_));
    CUDA_CHECK_VOID(cudaStreamDestroy(stream_));
    MpiBaseTestFixture::TearDown();
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
    MPI_CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    return id;
  }

  /**
   * Run NCCL AllToAllv benchmark using ncclAllToAllv API.
   * Returns bandwidth in GB/s and sets latency in microseconds.
   */
  float runNcclAllToAllvBenchmark(
      const AllToAllvBenchmarkConfig& config,
      float& latencyUs) {
    XLOGF(
        DBG1,
        "Rank {}: Running NCCL AllToAllv benchmark: {}",
        globalRank,
        config.name);

    const int nranks = numRanks;
    const std::size_t bytesPerPeer = config.bytesPerPeer;
    const std::size_t totalBytes = bytesPerPeer * nranks;

    // Allocate send and recv buffers
    DeviceBuffer sendBuffer(totalBytes);
    DeviceBuffer recvBuffer(totalBytes);

    // Initialize send buffer
    std::vector<char> h_send(totalBytes);
    for (int peer = 0; peer < nranks; peer++) {
      for (std::size_t i = 0; i < bytesPerPeer; i++) {
        h_send[peer * bytesPerPeer + i] =
            static_cast<char>(peer * 100 + globalRank * 10 + (i % 256));
      }
    }
    CUDA_CHECK(cudaMemcpy(
        sendBuffer.get(), h_send.data(), totalBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(recvBuffer.get(), 0, totalBytes));

    // Create send/recv counts and displacements
    std::vector<size_t> sendcounts(nranks, bytesPerPeer);
    std::vector<size_t> recvcounts(nranks, bytesPerPeer);
    std::vector<size_t> sdispls(nranks);
    std::vector<size_t> rdispls(nranks);

    for (int i = 0; i < nranks; i++) {
      sdispls[i] = i * bytesPerPeer;
      rdispls[i] = i * bytesPerPeer;
    }

    CudaEvent start, stop;
    const int nIter = 100;
    const int nIterWarmup = 5;

    // Warmup
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    for (int i = 0; i < nIterWarmup; i++) {
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

    // Benchmark
    CUDA_CHECK(cudaEventRecord(start.get(), stream_));
    for (int i = 0; i < nIter; i++) {
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
    float avgTime_ms = totalTime_ms / nIter;
    latencyUs = avgTime_ms * 1000.0f;

    // Algorithm bandwidth: total data moved (send + recv) / time
    std::size_t totalDataMoved = 2 * totalBytes; // send + recv
    float bandwidth_GBps = (totalDataMoved / (1000.0f * 1000.0f * 1000.0f)) /
        (avgTime_ms / 1000.0f);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    return bandwidth_GBps;
  }

  /**
   * Run AllToAllv benchmark.
   * Returns bandwidth in GB/s and sets latency in microseconds.
   */
  float runAllToAllvBenchmark(
      const AllToAllvBenchmarkConfig& config,
      float& latencyUs) {
    XLOGF(
        DBG1,
        "Rank {}: Running AllToAllv benchmark: {}",
        globalRank,
        config.name);

    const int nranks = numRanks;
    const std::size_t bytesPerPeer = config.bytesPerPeer;
    const std::size_t totalBytes = bytesPerPeer * nranks;

    // Allocate send and recv buffers
    DeviceBuffer sendBuffer(totalBytes);
    DeviceBuffer recvBuffer(totalBytes);

    // Initialize send buffer with pattern: peer * 1000 + globalRank * 100 +
    // offset
    std::vector<char> h_send(totalBytes);
    for (int peer = 0; peer < nranks; peer++) {
      for (std::size_t i = 0; i < bytesPerPeer; i++) {
        h_send[peer * bytesPerPeer + i] =
            static_cast<char>(peer * 100 + globalRank * 10 + (i % 256));
      }
    }
    CUDA_CHECK(cudaMemcpy(
        sendBuffer.get(), h_send.data(), totalBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(recvBuffer.get(), 0, totalBytes));

    // Setup P2P NVL transport
    MultiPeerNvlTransportConfig nvlConfig{
        .dataBufferSize = config.dataBufferSize,
        .chunkSize = config.chunkSize,
        .pipelineDepth = config.pipelineDepth,
    };

    // Create transport and exchange IPC handles
    auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
    MultiPeerNvlTransport transport(globalRank, nranks, bootstrap, nvlConfig);
    transport.exchange();

    // Get preallocated transport array from MultiPeerNvlTransport
    // (includes P2pSelfTransportDevice for self and P2pNvlTransportDevice for
    // peers)
    DeviceSpan<Transport> transports_span(
        transport.getTransportsArray(), nranks);

    // Create chunk info arrays (equal size for all peers)
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

    // Create device spans
    DeviceSpan<ChunkInfo> send_chunk_infos(
        static_cast<ChunkInfo*>(d_send_chunks.get()), nranks);
    DeviceSpan<ChunkInfo> recv_chunk_infos(
        static_cast<ChunkInfo*>(d_recv_chunks.get()), nranks);

    // Prepare kernel launch parameters
    dim3 gridDim(config.numBlocks);
    dim3 blockDim(config.numThreads);

    // Get device pointers from DeviceBuffer objects
    void* recvBuff_d = recvBuffer.get();
    void* sendBuff_d = sendBuffer.get();

    // Create timeout (default = no timeout)
    Timeout timeout;

    void* args[] = {
        &recvBuff_d,
        &sendBuff_d,
        &globalRank,
        &transports_span,
        &send_chunk_infos,
        &recv_chunk_infos,
        &timeout};

    CudaEvent start, stop;
    const int nIter = 100;
    const int nIterWarmup = 5;

    // Use pointer to cluster dimension for clustered launch
    dim3 defaultClusterDim(comms::common::kDefaultClusterSize, 1, 1);
    std::optional<dim3> clusterDimOpt = config.spreadClusterLaunch
        ? std::optional{defaultClusterDim}
        : std::nullopt;

    // Warmup
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    for (int i = 0; i < nIterWarmup; i++) {
      CUDA_CHECK(
          comms::common::launchKernel(
              (void*)all_to_allv_kernel,
              gridDim,
              blockDim,
              args,
              nullptr,
              clusterDimOpt));
      CUDA_CHECK(cudaDeviceSynchronize());
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    // Benchmark
    CUDA_CHECK(cudaEventRecord(start.get()));
    for (int i = 0; i < nIter; i++) {
      CUDA_CHECK(
          comms::common::launchKernel(
              (void*)all_to_allv_kernel,
              gridDim,
              blockDim,
              args,
              nullptr,
              clusterDimOpt));
    }
    CUDA_CHECK(cudaEventRecord(stop.get()));
    CUDA_CHECK(cudaDeviceSynchronize());

    float totalTime_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime_ms, start.get(), stop.get()));
    float avgTime_ms = totalTime_ms / nIter;
    latencyUs = avgTime_ms * 1000.0f;

    // Algorithm bandwidth: total data moved (send + recv) / time
    // Each rank sends nranks * bytesPerPeer and receives nranks * bytesPerPeer
    std::size_t totalDataMoved = 2 * totalBytes; // send + recv
    float bandwidth_GBps = (totalDataMoved / (1000.0f * 1000.0f * 1000.0f)) /
        (avgTime_ms / 1000.0f);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    return bandwidth_GBps;
  }

  void printResultsTable(const std::vector<AllToAllvBenchmarkResult>& results) {
    if (globalRank != 0) {
      return; // Only rank 0 prints
    }

    std::stringstream ss;
    ss << "\n";
    ss << "================================================================================================================\n";
    ss << "                         NCCL vs AllToAllv Benchmark Results\n";
    ss << "================================================================================================================\n";
    ss << std::left << std::setw(18) << "Test Name" << std::right
       << std::setw(12) << "Per-Peer" << std::right << std::setw(5) << "PD"
       << std::right << std::setw(10) << "Chunk" << std::right << std::setw(11)
       << "NCCL BW" << std::right << std::setw(11) << "A2A BW" << std::right
       << std::setw(9) << "Speedup" << std::right << std::setw(11) << "NCCL Lat"
       << std::right << std::setw(11) << "A2A Lat" << std::right
       << std::setw(11) << "Lat Reduc\n";
    ss << std::left << std::setw(18) << "" << std::right << std::setw(12) << ""
       << std::right << std::setw(5) << "" << std::right << std::setw(10) << ""
       << std::right << std::setw(11) << "(GB/s)" << std::right << std::setw(11)
       << "(GB/s)" << std::right << std::setw(9) << "A2A/NCCL" << std::right
       << std::setw(11) << "(us)" << std::right << std::setw(11) << "(us)"
       << std::right << std::setw(11) << "(us)\n";
    ss << "----------------------------------------------------------------------------------------------------------------\n";

    auto formatBytes = [](std::size_t bytes) -> std::string {
      if (bytes < 1024) {
        return std::to_string(bytes) + "B";
      }
      if (bytes < 1024 * 1024) {
        return std::to_string(bytes / 1024) + "KB";
      }
      if (bytes < 1024 * 1024 * 1024) {
        return std::to_string(bytes / (1024 * 1024)) + "MB";
      }
      return std::to_string(bytes / (1024 * 1024 * 1024)) + "GB";
    };

    for (const auto& r : results) {
      float latReduc = r.ncclLatency - r.alltoallvLatency;
      ss << std::left << std::setw(18) << r.testName << std::right
         << std::setw(12) << formatBytes(r.bytesPerPeer) << std::right
         << std::setw(5) << r.pipelineDepth << std::right << std::setw(10)
         << formatBytes(r.chunkSize) << std::right << std::setw(11)
         << std::fixed << std::setprecision(2) << r.ncclBandwidth << std::right
         << std::setw(11) << std::fixed << std::setprecision(2)
         << r.alltoallvBandwidth << std::right << std::setw(9) << std::fixed
         << std::setprecision(2) << r.speedup << "x" << std::right
         << std::setw(11) << std::fixed << std::setprecision(1) << r.ncclLatency
         << std::right << std::setw(11) << std::fixed << std::setprecision(1)
         << r.alltoallvLatency << std::right << std::setw(11) << std::fixed
         << std::setprecision(1) << latReduc << "\n";
    }

    ss << "================================================================================================================\n";
    ss << "Per-Peer: Message size per peer (equal for all peers), " << numRanks
       << " ranks\n";
    ss << "PD = Pipeline Depth, Chunk = Chunk Size\n";
    ss << "BW (Bandwidth) = Algorithm bandwidth (2 x total data / time), in GB/s\n";
    ss << "Lat (Latency) = Average transfer time per iteration, in microseconds\n";
    ss << "Lat Reduc = NCCL latency - AllToAllv latency (positive = AllToAllv faster)\n";
    ss << "Speedup = AllToAllv Bandwidth / NCCL Bandwidth\n";
    ss << "================================================================================================================\n";
    ss << "\n";

    XLOG(INFO) << ss.str();
  }

  ncclComm_t ncclComm_{};
  cudaStream_t stream_{};
};

// clang-format off
/**
 * Benchmark Results (H100 8-GPU, January 2025):
 * ================================================================================================================
 *                          NCCL vs AllToAllv Benchmark Results
 * ================================================================================================================
 * Test Name             Per-Peer   PD     Chunk    NCCL BW     A2A BW  Speedup   NCCL Lat    A2A Lat Lat Reduc
 *                                                  (GB/s)     (GB/s) A2A/NCCL       (us)       (us)      (us)
 * ----------------------------------------------------------------------------------------------------------------
 * 256K_8B                  256KB    2      64KB     186.34     135.43     0.73x       22.5       31.0       -8.5
 * 512K_16B                 512KB    2      64KB     221.06     262.48     1.19x       37.9       32.0        6.0
 * 1M_16B                     1MB    2      64KB     411.01     397.58     0.97x       40.8       42.2       -1.4
 * 2M_16B                     2MB    2     128KB     459.50     445.43     0.97x       73.0       75.3       -2.3
 * 4M_16B                     4MB    2     128KB     534.53     530.82     0.99x      125.5      126.4       -0.9
 * 8M_16B                     8MB    2     128KB     608.06     610.62     1.00x      220.7      219.8        0.9
 * 16M_16B                   16MB    2     128KB     657.45     657.70     1.00x      408.3      408.1        0.2
 * 32M_16B                   32MB    2     128KB     705.21     685.37     0.97x      761.3      783.3      -22.0
 * 64M_16B                   64MB    2     256KB     722.42     715.16     0.99x     1486.3     1501.4      -15.1
 * 128M_16B                 128MB    2     256KB     745.82     739.12     0.99x     2879.3     2905.5      -26.1
 * 512M_32B                 512MB    2     256KB     782.28     782.39     1.00x    10980.7    10979.2        1.5
 * 1G_32B                     1GB    2     256KB     788.04     786.64     1.00x    21800.8    21839.7      -38.8
 * ================================================================================================================
 *
 * Summary: AllToAllv achieves 0.97x-1.19x of NCCL bandwidth across all message sizes.
 * Best performance at 512KB (1.19x), parity at large messages (8MB+).
 */
// clang-format on
TEST_F(AllToAllvBenchmarkFixture, OptimalConfigs) {
  // Optimal configurations for multiple message sizes

  if (globalRank == 0) {
    XLOG(INFO)
        << "\n=== OPTIMAL AllToAllv vs NCCL Comparison (All Message Sizes) ===\n";
  }

  std::vector<AllToAllvBenchmarkConfig> configs;
  std::size_t kDataBufferSize = 8 * 1024 * 1024; // 8MB

  // === Block Count Tuning ===
  // Block counts are matched to NCCL channel counts for fair comparison.
  // NCCL channel tuning formula (from enqueue.cc:1913-1918):
  //   while (nBytes < nc * nt * threadThreshold) { nc--; }
  // Where:
  //   nc = number of channels (starts at 16)
  //   nt = number of threads (512)
  //   threadThreshold = NCCL_SIMPLE_THREAD_THRESHOLD = 64 (from comm.h:64)
  //
  // This means NCCL uses max channels when: nBytes >= nc * 512 * 64
  //   16 channels: nBytes >= 512KB
  //   8 channels:  nBytes >= 256KB
  //
  // === Chunk Size Tuning ===
  // 64KB for small messages (256KB-1MB), 128KB for medium (2MB-32MB),
  // 256KB for large (64MB+)

  // 256KB with 8 blocks (NCCL uses 8 channels), chunkSize = 64KB
  configs.push_back({
      .bytesPerPeer = 256 * 1024, // 256KB
      .numBlocks = 8,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 64 * 1024, // 64KB
      .dataBufferSize = kDataBufferSize,
      .spreadClusterLaunch = true,
      .name = "256K_8B",
  });

  // 512KB with 16 blocks (NCCL uses 16 channels), chunkSize = 64KB
  configs.push_back({
      .bytesPerPeer = 512 * 1024, // 512KB
      .numBlocks = 16,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 64 * 1024, // 64KB
      .dataBufferSize = kDataBufferSize,
      .spreadClusterLaunch = true,
      .name = "512K_16B",
  });

  // 1MB with 16 blocks (NCCL uses 16 channels), chunkSize = 64KB
  configs.push_back({
      .bytesPerPeer = 1 * 1024 * 1024, // 1MB
      .numBlocks = 16,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 64 * 1024, // 64KB
      .dataBufferSize = kDataBufferSize,
      .spreadClusterLaunch = true,
      .name = "1M_16B",
  });

  // 2MB with 16 blocks (NCCL uses 16 channels), chunkSize = 128KB
  configs.push_back({
      .bytesPerPeer = 2 * 1024 * 1024, // 2MB
      .numBlocks = 16,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 128 * 1024, // 128KB
      .dataBufferSize = kDataBufferSize,
      .spreadClusterLaunch = true,
      .name = "2M_16B",
  });

  // 4MB with 16 blocks (NCCL uses 16 channels), chunkSize = 128KB
  configs.push_back({
      .bytesPerPeer = 4 * 1024 * 1024, // 4MB
      .numBlocks = 16,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 128 * 1024, // 128KB
      .dataBufferSize = kDataBufferSize,
      .spreadClusterLaunch = true,
      .name = "4M_16B",
  });

  // 8MB with 16 blocks (NCCL uses 16 channels), chunkSize = 128KB
  configs.push_back({
      .bytesPerPeer = 8 * 1024 * 1024, // 8MB
      .numBlocks = 16,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 128 * 1024, // 128KB
      .dataBufferSize = kDataBufferSize,
      .spreadClusterLaunch = true,
      .name = "8M_16B",
  });

  // 16MB with 16 blocks, chunkSize = 128KB
  configs.push_back({
      .bytesPerPeer = 16 * 1024 * 1024, // 16MB
      .numBlocks = 16,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 128 * 1024, // 128KB
      .dataBufferSize = kDataBufferSize,
      .spreadClusterLaunch = true,
      .name = "16M_16B",
  });

  // 32MB with 16 blocks, chunkSize = 128KB
  configs.push_back({
      .bytesPerPeer = 32 * 1024 * 1024, // 32MB
      .numBlocks = 16,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 128 * 1024, // 128KB
      .dataBufferSize = kDataBufferSize,
      .spreadClusterLaunch = true,
      .name = "32M_16B",
  });

  // 64MB with 16 blocks, chunkSize = 256KB
  configs.push_back({
      .bytesPerPeer = 64 * 1024 * 1024, // 64MB
      .numBlocks = 16,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 256 * 1024, // 256KB
      .dataBufferSize = kDataBufferSize,
      .spreadClusterLaunch = true,
      .name = "64M_16B",
  });

  // 128MB with 16 blocks, chunkSize = 256KB
  configs.push_back({
      .bytesPerPeer = 128 * 1024 * 1024, // 128MB
      .numBlocks = 16,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 256 * 1024, // 256KB
      .dataBufferSize = kDataBufferSize,
      .spreadClusterLaunch = true,
      .name = "128M_16B",
  });

  // 512MB with 32 blocks, chunkSize = 256KB
  configs.push_back({
      .bytesPerPeer = 512 * 1024 * 1024, // 512MB
      .numBlocks = 32,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 256 * 1024, // 256KB
      .dataBufferSize = kDataBufferSize,
      .spreadClusterLaunch = true,
      .name = "512M_32B",
  });

  // 1GB with 32 blocks, chunkSize = 256KB
  configs.push_back({
      .bytesPerPeer = 1024 * 1024 * 1024, // 1GB
      .numBlocks = 32,
      .numThreads = 512,
      .pipelineDepth = 2,
      .chunkSize = 256 * 1024, // 256KB
      .dataBufferSize = kDataBufferSize,
      .spreadClusterLaunch = true,
      .name = "1G_32B",
  });

  std::vector<AllToAllvBenchmarkResult> results;

  for (const auto& config : configs) {
    float ncclLatencyUs = 0.0f;
    float ncclBandwidth = runNcclAllToAllvBenchmark(config, ncclLatencyUs);

    float alltoallvLatencyUs = 0.0f;
    float alltoallvBandwidth =
        runAllToAllvBenchmark(config, alltoallvLatencyUs);

    if (globalRank == 0) {
      AllToAllvBenchmarkResult result;
      result.testName = config.name;
      result.bytesPerPeer = config.bytesPerPeer;
      result.totalBytes = config.bytesPerPeer * numRanks * 2;
      result.pipelineDepth = config.pipelineDepth;
      result.chunkSize = config.chunkSize;
      result.ncclBandwidth = ncclBandwidth;
      result.alltoallvBandwidth = alltoallvBandwidth;
      result.ncclLatency = ncclLatencyUs;
      result.alltoallvLatency = alltoallvLatencyUs;
      result.speedup = alltoallvBandwidth / ncclBandwidth;
      results.push_back(result);
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  }

  printResultsTable(results);
}

} // namespace

} // namespace comms::pipes::benchmark

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new meta::comms::MPIEnvironmentBase);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
