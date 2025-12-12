// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>
#include <nccl.h>

#include "comms/pipes/MultiPeerNvlTransport.h"
#include "comms/pipes/benchmarks/BenchmarkKernel.cuh"
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

// CUDA error checking macro for void functions
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

// NCCL error checking macro for void functions
#define NCCL_CHECK_VOID(call)        \
  do {                               \
    ncclResult_t res = call;         \
    if (res != ncclSuccess) {        \
      XLOGF(                         \
          ERR,                       \
          "NCCL error at {}:{}: {}", \
          __FILE__,                  \
          __LINE__,                  \
          ncclGetErrorString(res));  \
      return;                        \
    }                                \
  } while (0)

// CUDA error checking macro for float-returning functions
#define CUDA_CHECK(call)             \
  do {                               \
    cudaError_t err = call;          \
    if (err != cudaSuccess) {        \
      XLOGF(                         \
          ERR,                       \
          "CUDA error at {}:{}: {}", \
          __FILE__,                  \
          __LINE__,                  \
          cudaGetErrorString(err));  \
      return 0.0f;                   \
    }                                \
  } while (0)

// NCCL error checking macro for float-returning functions
#define NCCL_CHECK(call)             \
  do {                               \
    ncclResult_t res = call;         \
    if (res != ncclSuccess) {        \
      XLOGF(                         \
          ERR,                       \
          "NCCL error at {}:{}: {}", \
          __FILE__,                  \
          __LINE__,                  \
          ncclGetErrorString(res));  \
      return 0.0f;                   \
    }                                \
  } while (0)

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
    CUDA_CHECK_VOID(cudaSetDevice(globalRank));

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
    const int nIter = 500;
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
    float bandwidth_GBps = (totalDataMoved / (1024.0f * 1024.0f * 1024.0f)) /
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

    // Create transport with MPI bootstrap and exchange IPC handles
    auto bootstrap = std::make_shared<meta::comms::MpiBootstrap>();
    MultiPeerNvlTransport transport(globalRank, nranks, bootstrap, nvlConfig);
    transport.exchange();

    // Create transport array: self for my rank, P2P for others
    P2pSelfTransportDevice selfTransport;
    std::vector<Transport> h_transports;
    h_transports.reserve(nranks);

    for (int rank = 0; rank < nranks; rank++) {
      if (rank == globalRank) {
        h_transports.emplace_back(selfTransport);
      } else {
        h_transports.emplace_back(transport.getP2pTransportDevice(rank));
      }
    }

    // Copy transports to device
    DeviceBuffer d_transports(sizeof(Transport) * nranks);
    CUDA_CHECK(cudaMemcpy(
        d_transports.get(),
        h_transports.data(),
        sizeof(Transport) * nranks,
        cudaMemcpyHostToDevice));

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
    DeviceSpan<Transport> transports_span(
        static_cast<Transport*>(d_transports.get()), nranks);
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

    void* args[] = {
        &recvBuff_d,
        &sendBuff_d,
        &globalRank,
        &transports_span,
        &send_chunk_infos,
        &recv_chunk_infos};

    CudaEvent start, stop;
    const int nIter = 500;
    const int nIterWarmup = 5;

    // Warmup
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    for (int i = 0; i < nIterWarmup; i++) {
      CUDA_CHECK(cudaLaunchKernel(
          (void*)allToAllvKernel, gridDim, blockDim, args, 0, nullptr));
    }

    // Benchmark
    CUDA_CHECK(cudaEventRecord(start.get()));
    for (int i = 0; i < nIter; i++) {
      CUDA_CHECK(cudaLaunchKernel(
          (void*)allToAllvKernel, gridDim, blockDim, args, 0, nullptr));
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
    float bandwidth_GBps = (totalDataMoved / (1024.0f * 1024.0f * 1024.0f)) /
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

TEST_F(AllToAllvBenchmarkFixture, OptimalConfigs) {
  // Optimal configurations for multiple message sizes

  if (globalRank == 0) {
    XLOG(INFO)
        << "\n=== OPTIMAL AllToAllv vs NCCL Comparison (All Message Sizes) ===\n";
  }

  std::vector<AllToAllvBenchmarkConfig> configs;
  std::size_t kDataBufferSize = 8 * 1024 * 1024; // 8MB

  // Focus on the most promising configs only
  // Baseline: 16 blocks, 8 warps/send, 1 chunk/warp
  configs.push_back({
      .bytesPerPeer = 8 * 1024,
      .numBlocks = 16,
      .numThreads = 256,
      .pipelineDepth = 2,
      .chunkSize = 1 * 1024,
      .dataBufferSize = kDataBufferSize,
      .name = "Baseline_16b_1k",
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
