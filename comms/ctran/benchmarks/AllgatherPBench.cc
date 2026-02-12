// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <nccl.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>
#include "comms/ctran/Ctran.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/testinfra/TestsCuUtils.h"

// Benchmark configuration flags
DEFINE_int64(
    min_bytes,
    16 * 1024,
    "Minimum message size in bytes (default: 16KB)");
DEFINE_int64(
    max_bytes,
    1024 * 1024 * 1024,
    "Maximum message size in bytes (default: 1GB)");
DEFINE_int32(warmup_iters, 5, "Number of warmup iterations (default: 5)");
DEFINE_int32(bench_iters, 50, "Number of benchmark iterations (default: 20)");
DEFINE_string(
    algo,
    "all",
    "Algorithm to benchmark: 'ctdirect', 'ctpipeline', 'nccl', or 'all' (default: all)");
DEFINE_bool(in_place, false, "Run in-place allgather (default: false)");
DEFINE_string(
    mem_type,
    "cudaMalloc",
    "Memory allocation type: 'cuMem' or 'cudamalloc' (default: cudamalloc)");

#define NCCLCHECK_TEST(cmd)                  \
  do {                                       \
    ncclResult_t r = cmd;                    \
    if (r != ncclSuccess) {                  \
      printf(                                \
          "Failed, NCCL error %s:%d '%s'\n", \
          __FILE__,                          \
          __LINE__,                          \
          ncclGetErrorString(r));            \
      exit(EXIT_FAILURE);                    \
    }                                        \
  } while (0)

// Benchmark result structure
struct BenchmarkResult {
  size_t sizeBytes;
  size_t count;
  double minTimeMs;
  double maxTimeMs;
  double avgTimeMs;
  double algoBwGBps;
  double busBwGBps;
  std::string algoName;
};

class CtranAllgatherPBenchTestEnv : public ctran::CtranEnvironmentBase {
 public:
  void SetUp() override {
    ctran::CtranEnvironmentBase::SetUp();

    // set logging level to WARN
    setenv("NCCL_DEBUG", "WARN", 1);
  }
};

class AllgatherPBenchmark : public ctran::CtranDistTestFixture {
 public:
  AllgatherPBenchmark() = default;

  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    CtranDistTestFixture::SetUp();

    ctranComm_ = makeCtranComm();

    CUDACHECK_TEST(cudaStreamCreate(&stream_));

    // Check if AllGatherP is supported
    EXPECT_TRUE(ctran::allGatherPSupport(ctranComm_.get()))
        << "allGatherP algo is not supported!";

    // Determine memory type
    memType_ =
        (FLAGS_mem_type == "cudaMalloc") ? kMemCudaMalloc : kMemNcclMemAlloc;

    // Check if CuMem is supported when using ncclmem
    EXPECT_TRUE(memType_ != kMemNcclMemAlloc || ncclIsCuMemSupported())
        << "CuMem is not supported!";

    dt_ = commBfloat16;
    typeSize_ = commTypeSize(dt_);

    // Initialize NCCL communicator for baseline benchmarking
    ncclUniqueId ncclId;
    if (globalRank == 0) {
      NCCLCHECK_TEST(ncclGetUniqueId(&ncclId));
    }
    // Broadcast NCCL unique ID to all ranks
    MPI_Bcast(&ncclId, sizeof(ncclId), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Initialize NCCL communicator
    CUDACHECK_TEST(cudaSetDevice(localRank));
    NCCLCHECK_TEST(ncclCommInitRank(&ncclComm_, numRanks, ncclId, globalRank));
  }

  void TearDown() override {
    // Destroy NCCL communicator
    if (ncclComm_ != nullptr) {
      ncclCommDestroy(ncclComm_);
    }
    CUDACHECK_TEST(cudaStreamDestroy(stream_));
    CtranDistTestFixture::TearDown();
  }

  // Allocate memory based on memory type
  void* allocateBuffer(size_t size) {
    void* buf = nullptr;
    if (memType_ == kMemCudaMalloc) {
      CUDACHECK_TEST(cudaMalloc(&buf, size));
    } else {
      NCCLCHECK_TEST(ncclMemAlloc(&buf, size));
    }
    return buf;
  }

  // Free memory based on memory type
  void freeBuffer(void* buf) {
    if (memType_ == kMemCudaMalloc) {
      CUDACHECK_TEST(cudaFree(buf));
    } else {
      ncclMemFree(buf);
    }
  }

  // Calculate bandwidth metrics
  void calculateBandwidth(
      size_t sizeBytes,
      double timeMs,
      double& algoBwGBps,
      double& busBwGBps) {
    // Algorithm bandwidth: total data moved
    algoBwGBps = (sizeBytes * numRanks) / (timeMs / 1000.0) / 1e9;
    // Bus bandwidth: accounts for data already present on each rank
    busBwGBps = algoBwGBps * (numRanks - 1) / numRanks;
  }

  // Run AllgatherP benchmark for a specific algorithm using a pre-initialized
  // request
  BenchmarkResult benchmarkAllgatherPWithRequest(
      size_t count,
      const std::string& algoName,
      CtranPersistentRequest* request,
      void* sendbuf,
      void* recvbuf) {
    const size_t sendBytes = count * typeSize_;

    // Determine send buffer (in-place or out-of-place)
    void* usedSendBuf = FLAGS_in_place
        ? static_cast<char*>(recvbuf) + globalRank * sendBytes
        : sendbuf;

    // Warmup iterations
    for (int i = 0; i < FLAGS_warmup_iters; ++i) {
      COMMCHECK_TEST(ctran::allGatherPExec(usedSendBuf, count, dt_, request));
      CUDACHECK_TEST(cudaStreamSynchronize(stream_));
    }

    // Benchmark iterations with timing
    CUDACHECK_TEST(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < FLAGS_bench_iters; ++i) {
      COMMCHECK_TEST(ctran::allGatherPExec(usedSendBuf, count, dt_, request));
      CUDACHECK_TEST(cudaStreamSynchronize(stream_));
    }

    CUDACHECK_TEST(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate total time and average per iteration
    double totalTimeMs =
        std::chrono::duration<double, std::milli>(end - start).count();
    double avgTime = totalTimeMs / FLAGS_bench_iters;

    // Allreduce timing measurements across all ranks
    double minTime, maxTime, avgTimeAllRanks;
    MPI_Allreduce(&avgTime, &minTime, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&avgTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(
        &avgTime, &avgTimeAllRanks, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    avgTimeAllRanks /= numRanks;

    double algoBw, busBw;
    calculateBandwidth(sendBytes, avgTimeAllRanks, algoBw, busBw);

    BenchmarkResult result;
    result.sizeBytes = sendBytes;
    result.count = count;
    result.minTimeMs = minTime;
    result.maxTimeMs = maxTime;
    result.avgTimeMs = avgTimeAllRanks;
    result.algoBwGBps = algoBw;
    result.busBwGBps = busBw;
    result.algoName = algoName;

    return result;
  }

  // Run NCCL baseline allgather benchmark using standard NCCL APIs
  BenchmarkResult benchmarkNcclAllgather(size_t count) {
    const size_t sendBytes = count * typeSize_;
    const size_t recvBytes = sendBytes * numRanks;

    // Allocate buffers using cudaMalloc (NCCL requires CUDA memory)
    void* sendbuf = nullptr;
    void* recvbuf = nullptr;
    CUDACHECK_TEST(cudaMalloc(&sendbuf, sendBytes));
    CUDACHECK_TEST(cudaMalloc(&recvbuf, recvBytes));

    // Initialize buffers
    CUDACHECK_TEST(cudaMemset(sendbuf, globalRank, sendBytes));
    CUDACHECK_TEST(cudaMemset(recvbuf, 0, recvBytes));
    CUDACHECK_TEST(cudaDeviceSynchronize());

    // Determine send buffer (in-place or out-of-place)
    void* usedSendBuf = FLAGS_in_place
        ? static_cast<char*>(recvbuf) + globalRank * sendBytes
        : sendbuf;

    // Warmup iterations
    for (int i = 0; i < FLAGS_warmup_iters; ++i) {
      NCCLCHECK_TEST(ncclAllGather(
          usedSendBuf, recvbuf, count, ncclBfloat16, ncclComm_, stream_));
      CUDACHECK_TEST(cudaStreamSynchronize(stream_));
    }

    // Benchmark iterations with timing
    CUDACHECK_TEST(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < FLAGS_bench_iters; ++i) {
      NCCLCHECK_TEST(ncclAllGather(
          usedSendBuf, recvbuf, count, ncclBfloat16, ncclComm_, stream_));
      CUDACHECK_TEST(cudaStreamSynchronize(stream_));
    }

    CUDACHECK_TEST(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate total time and average per iteration
    double totalTimeMs =
        std::chrono::duration<double, std::milli>(end - start).count();
    double avgTime = totalTimeMs / FLAGS_bench_iters;

    // Allreduce timing measurements across all ranks
    double minTime, maxTime, avgTimeAllRanks;
    MPI_Allreduce(&avgTime, &minTime, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&avgTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(
        &avgTime, &avgTimeAllRanks, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    avgTimeAllRanks /= numRanks;

    double algoBw, busBw;
    calculateBandwidth(sendBytes, avgTimeAllRanks, algoBw, busBw);

    // Cleanup
    CUDACHECK_TEST(cudaFree(sendbuf));
    CUDACHECK_TEST(cudaFree(recvbuf));

    BenchmarkResult result;
    result.sizeBytes = sendBytes;
    result.count = count;
    result.minTimeMs = minTime;
    result.maxTimeMs = maxTime;
    result.avgTimeMs = avgTimeAllRanks;
    result.algoBwGBps = algoBw;
    result.busBwGBps = busBw;
    result.algoName = "NCCL_AllGather";

    return result;
  }

  // Print benchmark header
  void printHeader() {
    if (globalRank == 0) {
      std::cout << "\n"
                << "=========================================="
                << "==========================================" << std::endl;
      std::cout << "AllgatherP Performance Benchmark" << std::endl;
      std::cout << "=========================================="
                << "==========================================" << std::endl;
      std::cout << "Configuration:" << std::endl;
      std::cout << "  Ranks: " << numRanks << std::endl;
      std::cout << "  Min Size: " << FLAGS_min_bytes << " bytes" << std::endl;
      std::cout << "  Max Size: " << FLAGS_max_bytes << " bytes" << std::endl;
      std::cout << "  Warmup Iterations: " << FLAGS_warmup_iters << std::endl;
      std::cout << "  Benchmark Iterations: " << FLAGS_bench_iters << std::endl;
      std::cout << "  Memory Type: " << FLAGS_mem_type << std::endl;
      std::cout << "  In-Place: " << (FLAGS_in_place ? "Yes" : "No")
                << std::endl;
      std::cout << "=========================================="
                << "==========================================" << std::endl;
      std::cout << std::endl;
    }
  }

  // Print result row
  void printResult(const BenchmarkResult& result) {
    if (globalRank == 0) {
      std::cout << std::left << std::setw(25) << result.algoName << std::right
                << std::setw(12) << result.sizeBytes << std::setw(12)
                << result.count << std::fixed << std::setprecision(3)
                << std::setw(12) << result.avgTimeMs << std::setw(12)
                << result.minTimeMs << std::setw(12) << result.maxTimeMs
                << std::setw(12) << result.algoBwGBps << std::setw(12)
                << result.busBwGBps << std::endl;
    }
  }

  // Print table header
  void printTableHeader() {
    if (globalRank == 0) {
      std::cout << std::left << std::setw(25) << "Algorithm" << std::right
                << std::setw(12) << "Size(B)" << std::setw(12) << "Count"
                << std::setw(12) << "Avg(ms)" << std::setw(12) << "Min(ms)"
                << std::setw(12) << "Max(ms)" << std::setw(12) << "AlgoBW(GB/s)"
                << std::setw(12) << "BusBW(GB/s)" << std::endl;
      std::cout << std::string(133, '-') << std::endl;
    }
  }

  // Run full benchmark suite
  void runBenchmark() {
    printHeader();
    printTableHeader();

    // Check nNodes for pipeline support
    const auto statex = ctranComm_->statex_.get();
    const auto nNodes = statex->nNodes();
    const bool pipelineSupported = nNodes >= 1;

    // Generate size range (powers of 2)
    std::vector<size_t> sizes;
    for (size_t size = FLAGS_min_bytes; size <= FLAGS_max_bytes; size *= 2) {
      sizes.push_back(size);
    }
    // Ensure max_bytes is included if not a power of 2
    if (sizes.back() != static_cast<size_t>(FLAGS_max_bytes)) {
      sizes.push_back(FLAGS_max_bytes);
    }

    // Allocate buffers ONCE for maximum size - reused for all tests
    const size_t maxSizeBytes = FLAGS_max_bytes;
    const size_t maxCount = maxSizeBytes / typeSize_;
    const size_t maxRecvBytes = maxSizeBytes * numRanks;

    void* sendbuf = nullptr;
    void* recvbuf = nullptr;
    void *sendHdl = nullptr, *recvHdl = nullptr;
    CtranPersistentRequest* request = nullptr;

    // Only allocate and initialize if running AllGatherP algorithms
    if (FLAGS_algo == "ctdirect" || FLAGS_algo == "ctpipeline" ||
        FLAGS_algo == "all") {
      // Allocate and initialize buffers ONCE for max size
      sendbuf = allocateBuffer(maxSizeBytes);
      recvbuf = allocateBuffer(maxRecvBytes);

      CUDACHECK_TEST(cudaMemset(sendbuf, globalRank, maxSizeBytes));
      CUDACHECK_TEST(cudaMemset(recvbuf, 0, maxRecvBytes));
      CUDACHECK_TEST(cudaDeviceSynchronize());

      // Register memory ONCE for max size
      COMMCHECK_TEST(
          ctranComm_->ctran_->commRegister(recvbuf, maxRecvBytes, &recvHdl));
      if (!FLAGS_in_place) {
        COMMCHECK_TEST(
            ctranComm_->ctran_->commRegister(sendbuf, maxSizeBytes, &sendHdl));
      }

      // Initialize AllGatherP ONCE globally with max size
      meta::comms::Hints hints;
      COMMCHECK_TEST(
          ctran::allGatherPInit(
              recvbuf,
              maxCount * numRanks,
              hints,
              dt_,
              ctranComm_.get(),
              stream_,
              request));

      // Wait for async init to complete
      CUDACHECK_TEST(cudaStreamSynchronize(stream_));
      CUDACHECK_TEST(cudaDeviceSynchronize());
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // Run benchmarks for all sizes using the same persistent request
    for (size_t sizeBytes : sizes) {
      size_t count = sizeBytes / typeSize_;

      // Reinitialize buffers for this size
      if (sendbuf && recvbuf) {
        CUDACHECK_TEST(cudaMemset(sendbuf, globalRank, sizeBytes));
        CUDACHECK_TEST(cudaMemset(recvbuf, 0, sizeBytes * numRanks));
        CUDACHECK_TEST(cudaDeviceSynchronize());
      }

      // Benchmark AllgatherP Direct (reuses same request)
      if (FLAGS_algo == "ctdirect" || FLAGS_algo == "all") {
        EnvRAII algoEnv(NCCL_ALLGATHER_P_ALGO, NCCL_ALLGATHER_P_ALGO::ctdirect);
        BenchmarkResult result = benchmarkAllgatherPWithRequest(
            count, "AllGatherP_Direct", request, sendbuf, recvbuf);
        printResult(result);
        MPI_Barrier(MPI_COMM_WORLD);
      }

      // Benchmark AllgatherP Pipeline (reuses same request)
      if ((FLAGS_algo == "ctpipeline" || FLAGS_algo == "all") &&
          pipelineSupported) {
        EnvRAII algoEnv(
            NCCL_ALLGATHER_P_ALGO, NCCL_ALLGATHER_P_ALGO::ctpipeline);
        BenchmarkResult result = benchmarkAllgatherPWithRequest(
            count, "AllGatherP_Pipeline", request, sendbuf, recvbuf);
        printResult(result);
        MPI_Barrier(MPI_COMM_WORLD);
      }

      // Benchmark NCCL baseline
      if (FLAGS_algo == "nccl" || FLAGS_algo == "all") {
        BenchmarkResult result = benchmarkNcclAllgather(count);
        printResult(result);
        MPI_Barrier(MPI_COMM_WORLD);
      }

      if (globalRank == 0) {
        std::cout << "====" << std::endl;
      }
    }

    // Cleanup ONCE at the very end after all sizes
    if (request != nullptr) {
      COMMCHECK_TEST(ctran::allGatherPDestroy(request));
      delete request;

      CUDACHECK_TEST(cudaStreamSynchronize(stream_));
      CUDACHECK_TEST(cudaDeviceSynchronize());
      MPI_Barrier(MPI_COMM_WORLD);
    }

    // Deregister memory ONCE
    if (recvHdl != nullptr) {
      COMMCHECK_TEST(ctranComm_->ctran_->commDeregister(recvHdl));
    }
    if (sendHdl != nullptr) {
      COMMCHECK_TEST(ctranComm_->ctran_->commDeregister(sendHdl));
    }

    // Free buffers ONCE
    if (sendbuf != nullptr) {
      freeBuffer(sendbuf);
    }
    if (recvbuf != nullptr) {
      freeBuffer(recvbuf);
    }

    if (globalRank == 0) {
      std::cout << "=========================================="
                << "==========================================" << std::endl;
      std::cout << "Benchmark completed successfully!" << std::endl;
      std::cout << "=========================================="
                << "==========================================" << std::endl;
    }
  }

 private:
  std::unique_ptr<CtranComm> ctranComm_;
  ncclComm_t ncclComm_{nullptr};
  cudaStream_t stream_{nullptr};
  commDataType_t dt_;
  size_t typeSize_;
  MemAllocType memType_;
};

// Benchmark test that runs the full suite
TEST_F(AllgatherPBenchmark, BenchmarkSuite) {
  runBenchmark();
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new CtranAllgatherPBenchTestEnv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
