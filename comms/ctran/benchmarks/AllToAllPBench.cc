// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <folly/init/Init.h>
#include <folly/json/json.h>
#include <gflags/gflags.h>
#include <nccl.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>
#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/AllToAll/AllToAllPImpl.h"
#include "comms/ctran/regcache/RegCache.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/ctran/tests/CtranTestUtils.h"
#include "comms/testinfra/TestsCuUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

#if defined(A2AP_BENCH_NOLOCAL)
static constexpr bool kNoLocal = true;
#else
static constexpr bool kNoLocal = false;
#endif

// Cap on total bytes for each persistent buffer (sendbuf_/recvbuf_) so high
// rank counts do not exhaust device memory; the per-peer max is derived by
// dividing this by numRanks.
static constexpr size_t kMaxTotalBytesPerBuffer = 8ULL * 1024 * 1024 * 1024;

// Benchmark configuration flags
DEFINE_int64(
    min_bytes,
    16 * 1024,
    "Minimum per-peer message size in bytes (default: 16KB)");
DEFINE_int64(
    max_bytes,
    1024 * 1024 * 1024,
    "Maximum per-peer message size in bytes (default: 1GB)");
DEFINE_int32(warmup_iters, 5, "Number of warmup iterations (default: 5)");
DEFINE_int32(bench_iters, 50, "Number of benchmark iterations (default: 50)");
DEFINE_string(
    algo,
    "all",
    "Algorithm to benchmark: 'ctran', 'nccl', or 'all' (default: all)");
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

class CtranAllToAllPBenchTestEnv : public ctran::CtranEnvironmentBase {
 public:
  void SetUp() override {
    ctran::CtranEnvironmentBase::SetUp();

    // set logging level to WARN
    setenv("NCCL_DEBUG", "WARN", 1);
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    setenv("NCCL_CTRAN_ALLOW_CUDA_GRAPH", "1", 0);
    // Turn on colltrace to validate used algorithm
    setenv("NCCL_COLLTRACE", "algostat", 1);
  }
};

class AllToAllPBenchmark : public ctran::CtranDistTestFixture {
 public:
  AllToAllPBenchmark() = default;

  void SetUp() override {
    CtranDistTestFixture::SetUp();

    ctranComm_ = makeCtranComm(kNoLocal);

    CUDACHECK_TEST(cudaStreamCreate(&stream_));

    // Check if AllToAllP is supported
    EXPECT_TRUE(ctran::AllToAllPSupport(ctranComm_.get()))
        << "AllToAllP algo is not supported!";

    // Determine memory type
    memType_ =
        (FLAGS_mem_type == "cudaMalloc") ? kMemCudaMalloc : kMemNcclMemAlloc;

    // Check if CuMem is supported when using ncclmem
    EXPECT_TRUE(memType_ != kMemNcclMemAlloc || ncclIsCuMemSupported())
        << "CuMem is not supported!";

    dt_ = commBfloat16;
    typeSize_ = commTypeSize(dt_);

    // Initialize NCCL communicator for baseline benchmarking
    // Use bootstrap allGather to broadcast ncclUniqueId to all ranks
    ncclUniqueId ncclId;
    if (globalRank == 0) {
      NCCLCHECK_TEST(ncclGetUniqueId(&ncclId));
    }
    std::vector<char> idBuf(numRanks * sizeof(ncclId));
    memcpy(idBuf.data() + globalRank * sizeof(ncclId), &ncclId, sizeof(ncclId));
    auto rc =
        ctranComm_->bootstrap_
            ->allGather(idBuf.data(), sizeof(ncclId), globalRank, numRanks)
            .get();
    EXPECT_EQ(rc, 0) << "Bootstrap allGather for ncclUniqueId failed";
    memcpy(&ncclId, idBuf.data(), sizeof(ncclId));

    // Initialize NCCL communicator
    CUDACHECK_TEST(cudaSetDevice(localRank));
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    ncclx::Hints hints;
    if constexpr (kNoLocal) {
      NCCLCHECK_TEST(hints.set("noLocal", "1"));
      config.hints = &hints;
    }

    NCCLCHECK_TEST(ncclCommInitRankConfig(
        &ncclComm_, numRanks, ncclId, globalRank, &config));
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
    // Mimic the real CCA allocator hook: cache (but do not force-register)
    // every allocation. Eager A2AP relies on the recvbuf being cached so its
    // scoped registration in AllToAllPInit can acquire it; the force-reg path
    // does not satisfy that.
    COMMCHECK_TEST(ctran::RegCache::getInstance()->globalRegister(buf, size));
    return buf;
  }

  // Free memory based on memory type
  void freeBuffer(void* buf, size_t size) {
    // Release the CCA cache entry acquired in allocateBuffer before freeing.
    COMMCHECK_TEST(ctran::RegCache::getInstance()->globalDeregister(buf, size));
    if (memType_ == kMemCudaMalloc) {
      CUDACHECK_TEST(cudaFree(buf));
    } else {
      ncclMemFree(buf);
    }
  }

  // Calculate bandwidth metrics. For alltoall, each rank sends (nRanks-1)
  // chunks off-node/off-rank and receives (nRanks-1) chunks, so both algo and
  // bus bandwidth reflect the (nRanks-1) chunks that actually cross ranks.
  void calculateBandwidth(
      size_t perPeerBytes,
      double timeMs,
      double& algoBwGBps,
      double& busBwGBps) {
    // Algorithm bandwidth: total data moved per rank across all peers.
    algoBwGBps = (perPeerBytes * numRanks) / (timeMs / 1000.0) / 1e9;
    // Bus bandwidth: excludes the self chunk that never leaves the rank.
    busBwGBps = algoBwGBps * (numRanks - 1) / numRanks;
  }

  // Run AllToAllP benchmark using a pre-initialized persistent request. count
  // is the per-peer chunk count; sendbuf holds numRanks distinct chunks and
  // recvbuf receives numRanks chunks.
  BenchmarkResult benchmarkAllToAllPWithRequest(
      size_t count,
      const std::string& algoName,
      CtranPersistentRequest* request) {
    const size_t perPeerBytes = count * typeSize_;

    // Warmup iterations
    for (int i = 0; i < FLAGS_warmup_iters; ++i) {
      COMMCHECK_TEST(ctran::AllToAllPExec(sendbuf_, count, request));
      CUDACHECK_TEST(cudaStreamSynchronize(stream_));
    }

    // Benchmark iterations with timing
    CUDACHECK_TEST(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < FLAGS_bench_iters; ++i) {
      COMMCHECK_TEST(ctran::AllToAllPExec(sendbuf_, count, request));
    }

    // Sync once after all iterations
    CUDACHECK_TEST(cudaStreamSynchronize(stream_));
    auto end = std::chrono::high_resolution_clock::now();

    return finalizeResult(count, perPeerBytes, algoName, start, end);
  }

  // Capture the same persistent AllToAllPExec into a plain CUDA graph via
  // stream capture and replay it, reusing the warmed-up eager request. CE-copy
  // batching is disabled under capture, so this tracks the eager-vs-graph
  // delta. Re-captures per call since the graph bakes in sendbuf_ ptr + count.
  BenchmarkResult benchmarkAllToAllPGraph(
      size_t count,
      CtranPersistentRequest* request) {
    const size_t perPeerBytes = count * typeSize_;

    // Warmup eager execs (per-iter sync) so async init completes before
    // capture.
    for (int i = 0; i < FLAGS_warmup_iters; ++i) {
      COMMCHECK_TEST(ctran::AllToAllPExec(sendbuf_, count, request));
      CUDACHECK_TEST(cudaStreamSynchronize(stream_));
    }

    // Capture the persistent exec into an instantiated graph.
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    CUDACHECK_TEST(
        cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal));
    COMMCHECK_TEST(ctran::AllToAllPExec(sendbuf_, count, request));
    CUDACHECK_TEST(cudaStreamEndCapture(stream_, &graph));
    CUDACHECK_TEST(cudaGraphInstantiate(&graphExec, graph, 0));

    // Graph launch warmups
    for (int i = 0; i < FLAGS_warmup_iters; ++i) {
      CUDACHECK_TEST(cudaGraphLaunch(graphExec, stream_));
    }
    CUDACHECK_TEST(cudaStreamSynchronize(stream_));

    // Benchmark iterations with timing
    CUDACHECK_TEST(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < FLAGS_bench_iters; ++i) {
      CUDACHECK_TEST(cudaGraphLaunch(graphExec, stream_));
    }

    // Sync once after all iterations
    CUDACHECK_TEST(cudaStreamSynchronize(stream_));
    auto end = std::chrono::high_resolution_clock::now();

    BenchmarkResult result =
        finalizeResult(count, perPeerBytes, "AllToAllP_Graph", start, end);

    CUDACHECK_TEST(cudaGraphExecDestroy(graphExec));
    CUDACHECK_TEST(cudaGraphDestroy(graph));

    return result;
  }

  // Run NCCL baseline alltoall benchmark using grouped ncclSend/ncclRecv.
  BenchmarkResult benchmarkNcclAllToAll(size_t count) {
    const size_t perPeerBytes = count * typeSize_;
    const size_t totalBytes = perPeerBytes * numRanks;

    // Allocate buffers using cudaMalloc (NCCL requires CUDA memory)
    void* sendbuf = nullptr;
    void* recvbuf = nullptr;
    CUDACHECK_TEST(cudaMalloc(&sendbuf, totalBytes));
    CUDACHECK_TEST(cudaMalloc(&recvbuf, totalBytes));

    // Initialize buffers
    CUDACHECK_TEST(cudaMemset(sendbuf, globalRank, totalBytes));
    CUDACHECK_TEST(cudaMemset(recvbuf, 0, totalBytes));
    CUDACHECK_TEST(cudaDeviceSynchronize());

    auto runOnce = [&]() {
      NCCLCHECK_TEST(ncclGroupStart());
      for (int peer = 0; peer < numRanks; ++peer) {
        NCCLCHECK_TEST(ncclSend(
            static_cast<char*>(sendbuf) + peer * perPeerBytes,
            count,
            ncclBfloat16,
            peer,
            ncclComm_,
            stream_));
        NCCLCHECK_TEST(ncclRecv(
            static_cast<char*>(recvbuf) + peer * perPeerBytes,
            count,
            ncclBfloat16,
            peer,
            ncclComm_,
            stream_));
      }
      NCCLCHECK_TEST(ncclGroupEnd());
    };

    // Warmup iterations
    for (int i = 0; i < FLAGS_warmup_iters; ++i) {
      runOnce();
      CUDACHECK_TEST(cudaStreamSynchronize(stream_));
    }

    // Benchmark iterations with timing
    CUDACHECK_TEST(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < FLAGS_bench_iters; ++i) {
      runOnce();
    }

    // Sync once after all iterations
    CUDACHECK_TEST(cudaStreamSynchronize(stream_));
    auto end = std::chrono::high_resolution_clock::now();

    BenchmarkResult result =
        finalizeResult(count, perPeerBytes, "NCCL_AllToAll", start, end);

    // Cleanup
    CUDACHECK_TEST(cudaFree(sendbuf));
    CUDACHECK_TEST(cudaFree(recvbuf));

    return result;
  }

  // Reduce a timed window into an all-rank BenchmarkResult: average the
  // per-rank per-iteration times via bootstrap allGather, then derive
  // bandwidth. Shared by the ctran and NCCL paths.
  BenchmarkResult finalizeResult(
      size_t count,
      size_t perPeerBytes,
      const std::string& algoName,
      std::chrono::high_resolution_clock::time_point start,
      std::chrono::high_resolution_clock::time_point end) {
    // Calculate total time and average per iteration
    double totalTimeMs =
        std::chrono::duration<double, std::milli>(end - start).count();
    double avgTime = totalTimeMs / FLAGS_bench_iters;

    // Gather timing measurements across all ranks using bootstrap allGather
    std::vector<double> allAvgTimes(numRanks);
    {
      std::vector<char> buf(numRanks * sizeof(double));
      memcpy(
          buf.data() + globalRank * sizeof(double), &avgTime, sizeof(double));
      auto rc =
          ctranComm_->bootstrap_
              ->allGather(buf.data(), sizeof(double), globalRank, numRanks)
              .get();
      EXPECT_EQ(rc, 0) << "Bootstrap allGather for timing failed";
      for (int i = 0; i < numRanks; ++i) {
        memcpy(
            &allAvgTimes[i], buf.data() + i * sizeof(double), sizeof(double));
      }
    }

    double minTime = *std::min_element(allAvgTimes.begin(), allAvgTimes.end());
    double maxTime = *std::max_element(allAvgTimes.begin(), allAvgTimes.end());
    double avgTimeAllRanks = 0.0;
    for (double t : allAvgTimes) {
      avgTimeAllRanks += t;
    }
    avgTimeAllRanks /= numRanks;

    double algoBw, busBw;
    calculateBandwidth(perPeerBytes, avgTimeAllRanks, algoBw, busBw);

    BenchmarkResult result;
    result.sizeBytes = perPeerBytes;
    result.count = count;
    result.minTimeMs = minTime;
    result.maxTimeMs = maxTime;
    result.avgTimeMs = avgTimeAllRanks;
    result.algoBwGBps = algoBw;
    result.busBwGBps = busBw;
    result.algoName = algoName;

    return result;
  }

  // Print benchmark header
  void printHeader(const size_t effectiveMaxBytes) {
    if (globalRank == 0) {
      std::cout << "\n"
                << "=========================================="
                << "==========================================" << std::endl;
      std::cout << "AllToAllP Performance Benchmark" << std::endl;
      std::cout << "=========================================="
                << "==========================================" << std::endl;
      std::cout << "Configuration:" << std::endl;
      std::cout << "  Ranks: " << numRanks << std::endl;
      std::cout << "  Min Size (per peer): " << FLAGS_min_bytes << " bytes"
                << std::endl;
      std::cout << "  Max Size (per peer): " << effectiveMaxBytes << " bytes"
                << std::endl;
      std::cout << "  Warmup Iterations: " << FLAGS_warmup_iters << std::endl;
      std::cout << "  Benchmark Iterations: " << FLAGS_bench_iters << std::endl;
      std::cout << "  Memory Type: " << FLAGS_mem_type << std::endl;
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

  // Confirm at runtime that each selected path ran its intended algorithm:
  // the ctran path via colltrace CT_pastColls (algoName == CtranAllToAllP,
  // topology-independent) and the NCCL baseline via AlgoStats (grouped
  // send/recv recorded as SendRecv / Baseline_SendRecv). Runs outside any
  // timed region.
  void confirmAlgorithms() {
    const bool ranCtran = (FLAGS_algo == "ctran" || FLAGS_algo == "all");
    const bool ranNccl = (FLAGS_algo == "nccl" || FLAGS_algo == "all");

    if (ranCtran) {
      const auto dumpMap = ctran::waitForCollTraceDrain(ctranComm_.get());
      const auto pastCollsJson = folly::parseJson(dumpMap.at("CT_pastColls"));
      const std::string expectedAlgo =
          ctran::alltoallp::AlgoImpl::algoName(NCCL_ALLTOALL_ALGO::ctran);
      int numMatched = 0;
      for (const auto& coll : pastCollsJson) {
        if (coll["algoName"].asString() == expectedAlgo) {
          numMatched++;
        }
      }
      EXPECT_GT(numMatched, 0) << "ctran path did not run " << expectedAlgo;
      if (globalRank == 0) {
        std::cout << "[algo-confirm] ctran path ran: " << expectedAlgo << " ("
                  << numMatched << " records)" << std::endl;
      }
    }

    if (ranNccl) {
#ifdef NCCL_HAS_DUMP_ALGO_STAT
      // Baseline nccl comm records algo stats via NCCL_COLLTRACE=algostat;
      // query them directly through the version-agnostic dumpAlgoStat API. Map
      // layout: collective name -> algo name -> call count.
      std::unordered_map<std::string, std::unordered_map<std::string, int64_t>>
          algoStat;
      ncclx::colltrace::dumpAlgoStat(ncclComm_, algoStat);
      const auto collIt = algoStat.find("SendRecv");
      EXPECT_NE(collIt, algoStat.end())
          << "baseline NCCL SendRecv not found in AlgoStats";
      std::string matchedAlgo;
      int64_t matchedCount = 0;
      if (collIt != algoStat.end()) {
        for (const auto& [algoName, callCount] : collIt->second) {
          if (callCount > 0 &&
              algoName.find("Baseline_SendRecv") != std::string::npos) {
            matchedAlgo = algoName;
            matchedCount = callCount;
            break;
          }
        }
      }
      EXPECT_FALSE(matchedAlgo.empty())
          << "baseline NCCL did not run an algo containing Baseline_SendRecv";
      if (globalRank == 0) {
        std::cout << "[algo-confirm] baseline NCCL ran: " << matchedAlgo
                  << " (count " << matchedCount << ")" << std::endl;
      }
#else
#error \
    "NCCL_HAS_DUMP_ALGO_STAT undefined; baseline algo confirmation unavailable"
#endif
    }
  }

  // Run full benchmark suite
  void runBenchmark() {
    // Per-peer max is capped so each persistent buffer (per-peer * numRanks)
    // stays within kMaxTotalBytesPerBuffer, preventing OOM at high rank counts.
    const size_t effectiveMaxBytes =
        std::min<size_t>(FLAGS_max_bytes, kMaxTotalBytesPerBuffer / numRanks);

    printHeader(effectiveMaxBytes);
    printTableHeader();

    const bool wantCtran = (FLAGS_algo == "ctran" || FLAGS_algo == "all");

    // Generate per-peer size range (powers of 2)
    std::vector<size_t> sizes;
    for (size_t size = FLAGS_min_bytes; size <= effectiveMaxBytes; size *= 2) {
      sizes.push_back(size);
    }
    // Ensure max_bytes is included if not a power of 2, and guarantee at least
    // one size when FLAGS_min_bytes exceeds the effective cap.
    if (sizes.empty() || sizes.back() != effectiveMaxBytes) {
      sizes.push_back(effectiveMaxBytes);
    }

    // Allocate buffers ONCE for maximum size - reused for all tests. Both the
    // per-peer sendbuf and recvbuf hold numRanks chunks (count*numRanks).
    const size_t maxPerPeerBytes = effectiveMaxBytes;
    const size_t maxPerPeerCount = maxPerPeerBytes / typeSize_;
    const size_t maxTotalBytes = maxPerPeerBytes * numRanks;
    const size_t maxRecvCount = maxPerPeerCount * numRanks;

    CtranPersistentRequest* request = nullptr;

    // Only allocate and initialize if running AllToAllP.
    if (wantCtran) {
      // allocateBuffer caches each buffer via the allocator hook; eager
      // AllToAllPInit scoped-registers recvbuf, and sendbuf is resolved from
      // the cache at exec, so no explicit commRegister is needed.
      sendbuf_ = allocateBuffer(maxTotalBytes);
      recvbuf_ = allocateBuffer(maxTotalBytes);

      CUDACHECK_TEST(cudaMemset(sendbuf_, globalRank, maxTotalBytes));
      CUDACHECK_TEST(cudaMemset(recvbuf_, 0, maxTotalBytes));
      CUDACHECK_TEST(cudaDeviceSynchronize());

      // One persistent request bound to recvbuf, sized for the max recv count.
      // Do NOT share a persistent request across variants.
      meta::comms::Hints hints;
      COMMCHECK_TEST(
          ctran::AllToAllPInit(
              recvbuf_,
              maxRecvCount,
              hints,
              dt_,
              ctranComm_.get(),
              stream_,
              request));

      // Wait for async init to complete
      CUDACHECK_TEST(cudaStreamSynchronize(stream_));
      CUDACHECK_TEST(cudaDeviceSynchronize());
      ctranComm_->bootstrap_->barrier(globalRank, numRanks).get();
    }

    // Run benchmarks for all sizes using the same persistent request
    for (size_t sizeBytes : sizes) {
      const size_t count = sizeBytes / typeSize_;

      // Reinitialize buffers for this size
      if (sendbuf_ && recvbuf_) {
        CUDACHECK_TEST(cudaMemset(sendbuf_, globalRank, sizeBytes * numRanks));
        CUDACHECK_TEST(cudaMemset(recvbuf_, 0, sizeBytes * numRanks));
        CUDACHECK_TEST(cudaDeviceSynchronize());
      }

      // Benchmark AllToAllP (reuses same request), eager then graph
      if (wantCtran) {
        BenchmarkResult eagerResult =
            benchmarkAllToAllPWithRequest(count, "AllToAllP", request);
        printResult(eagerResult);
        ctranComm_->bootstrap_->barrier(globalRank, numRanks).get();

        BenchmarkResult graphResult = benchmarkAllToAllPGraph(count, request);
        printResult(graphResult);
        ctranComm_->bootstrap_->barrier(globalRank, numRanks).get();

        if (globalRank == 0) {
          const double delta = graphResult.algoBwGBps - eagerResult.algoBwGBps;
          const double pct = eagerResult.algoBwGBps > 0.0
              ? (delta / eagerResult.algoBwGBps) * 100.0
              : 0.0;
          std::cout << "[eager-vs-graph] size=" << sizeBytes << ": eager "
                    << std::fixed << std::setprecision(3)
                    << eagerResult.algoBwGBps << " GB/s, graph "
                    << graphResult.algoBwGBps << " GB/s, delta " << delta
                    << " GB/s (" << pct << "%)" << std::endl;
        }
      }

      // Benchmark NCCL baseline
      if (FLAGS_algo == "nccl" || FLAGS_algo == "all") {
        BenchmarkResult result = benchmarkNcclAllToAll(count);
        printResult(result);
        ctranComm_->bootstrap_->barrier(globalRank, numRanks).get();
      }

      if (globalRank == 0) {
        std::cout << std::endl;
      }
    }

    // Cleanup ONCE at the very end after all sizes
    if (request != nullptr) {
      COMMCHECK_TEST(ctran::AllToAllPDestroy(request));
      delete request;
      CUDACHECK_TEST(cudaStreamSynchronize(stream_));
      CUDACHECK_TEST(cudaDeviceSynchronize());
      ctranComm_->bootstrap_->barrier(globalRank, numRanks).get();
    }

    // Free buffers ONCE (releases their CCA cache entries), after
    // AllToAllPDestroy.
    if (sendbuf_ != nullptr) {
      freeBuffer(sendbuf_, maxTotalBytes);
    }
    if (recvbuf_ != nullptr) {
      freeBuffer(recvbuf_, maxTotalBytes);
    }

    confirmAlgorithms();

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
  void* sendbuf_{nullptr};
  void* recvbuf_{nullptr};
};

// Benchmark test that runs the full suite
TEST_F(AllToAllPBenchmark, BenchmarkSuite) {
  runBenchmark();
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new CtranAllToAllPBenchTestEnv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
