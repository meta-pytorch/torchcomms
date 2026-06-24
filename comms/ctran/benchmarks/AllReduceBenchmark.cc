// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/AllReduce/AllReduceImpl.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/testinfra/TestsCuUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

DEFINE_int32(warmup_iters, 5, "Number of warmup iterations");
DEFINE_int32(bench_iters, 20, "Number of benchmark iterations");
DEFINE_string(
    sizes,
    "",
    "Comma-separated message sizes, e.g. 1K,8M,1G. Empty uses defaults.");
DEFINE_bool(
    include_offset_case,
    true,
    "Include the unaligned offset benchmark cases");

namespace {

struct BenchCase {
  std::string order;
  size_t sizeBytes{};
  bool inPlace{};
  size_t offsetBytes{};
  size_t sequenceIndex{};
  bool includeSequenceIndex{};
};

std::vector<size_t> gBenchmarkSizes;
std::vector<BenchCase> gBenchmarkCases;

std::vector<size_t> defaultBenchmarkSizes() {
  return {
      1024UL,
      8UL * 1024 * 1024,
      1024UL * 1024 * 1024,
  };
}

bool parseDecimalSize(
    const std::string& token,
    size_t multiplier,
    size_t* value,
    std::string* error) {
  if (token.empty()) {
    *error = "size token is missing a decimal value";
    return false;
  }

  size_t parsed = 0;
  for (char c : token) {
    if (!std::isdigit(static_cast<unsigned char>(c))) {
      *error = "size token `" + token + "` must contain only decimal digits";
      return false;
    }
    const size_t digit = static_cast<size_t>(c - '0');
    if (parsed > (std::numeric_limits<size_t>::max() - digit) / 10) {
      *error = "size token `" + token + "` is too large";
      return false;
    }
    parsed = parsed * 10 + digit;
  }

  if (parsed > std::numeric_limits<size_t>::max() / multiplier) {
    *error = "size token `" + token + "` overflows after applying suffix";
    return false;
  }
  *value = parsed * multiplier;
  return true;
}

bool parseSizeToken(std::string token, size_t* value, std::string* error) {
  if (token.empty()) {
    *error = "empty size token in --sizes";
    return false;
  }

  size_t multiplier = 1;
  const char suffix = token.back();
  if (suffix == 'k' || suffix == 'K') {
    multiplier = 1024UL;
    token.pop_back();
  } else if (suffix == 'm' || suffix == 'M') {
    multiplier = 1024UL * 1024;
    token.pop_back();
  } else if (suffix == 'g' || suffix == 'G') {
    multiplier = 1024UL * 1024 * 1024;
    token.pop_back();
  } else if (!std::isdigit(static_cast<unsigned char>(suffix))) {
    *error =
        "size token has unsupported suffix `" + std::string(1, suffix) + "`";
    return false;
  }
  return parseDecimalSize(token, multiplier, value, error);
}

bool parseSizeList(
    const std::string& value,
    std::vector<size_t>* sizes,
    std::string* error) {
  sizes->clear();
  size_t start = 0;
  while (start <= value.size()) {
    const size_t comma = value.find(',', start);
    const size_t end = comma == std::string::npos ? value.size() : comma;
    if (end == start) {
      *error = "empty size token in --sizes";
      return false;
    }
    size_t size = 0;
    const std::string token = value.substr(start, end - start);
    if (!parseSizeToken(token, &size, error)) {
      *error = "invalid --sizes token `" + token + "`: " + *error;
      return false;
    }
    sizes->push_back(size);
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  if (sizes->empty()) {
    *error = "--sizes must contain at least one size";
    return false;
  }
  return true;
}

std::string formatSize(size_t bytes) {
  if (bytes >= 1024UL * 1024 * 1024) {
    return std::to_string(bytes / (1024UL * 1024 * 1024)) + "G";
  }
  if (bytes >= 1024UL * 1024) {
    return std::to_string(bytes / (1024UL * 1024)) + "M";
  }
  if (bytes >= 1024UL) {
    return std::to_string(bytes / 1024UL) + "K";
  }
  return std::to_string(bytes) + "B";
}

const char* topologyName() {
#if defined(CTRAN_ALLREDUCE_BENCH_IB_ONLY)
  return "IB_ONLY";
#elif defined(CTRAN_ALLREDUCE_BENCH_NVL_ONLY)
  return "NVL_ONLY";
#else
  return "UNKNOWN_TOPOLOGY";
#endif
}

std::vector<BenchCase> makeBenchmarkCases(const std::vector<size_t>& sizes) {
  std::vector<BenchCase> cases;
  for (bool inPlace : {false, true}) {
    for (size_t size : sizes) {
      cases.push_back(
          BenchCase{.order = "exact", .sizeBytes = size, .inPlace = inPlace});
    }
  }

  const std::vector<size_t> mixedOrder{
      1024UL,
      1024UL * 1024 * 1024,
      8UL * 1024 * 1024,
      1024UL,
      8UL * 1024 * 1024,
      1024UL * 1024 * 1024,
  };
  for (bool inPlace : {false, true}) {
    for (size_t i = 0; i < mixedOrder.size(); ++i) {
      cases.push_back(
          BenchCase{
              .order = "mixed",
              .sizeBytes = mixedOrder[i],
              .inPlace = inPlace,
              .sequenceIndex = i,
              .includeSequenceIndex = true});
    }
  }

  if (FLAGS_include_offset_case) {
    for (bool inPlace : {false, true}) {
      cases.push_back(
          BenchCase{
              .order = "offset",
              .sizeBytes = 100,
              .inPlace = inPlace,
              .offsetBytes = sizeof(float),
          });
    }
  }
  return cases;
}

std::string benchmarkName(const BenchCase& benchCase) {
  std::string name = std::string("CtranAllReduce/") + topologyName() + "/" +
      benchCase.order + "/";
  if (benchCase.includeSequenceIndex) {
    name += "case_" + std::to_string(benchCase.sequenceIndex) + "/";
  }
  name += formatSize(benchCase.sizeBytes) + "/" +
      (benchCase.inPlace ? "InPlace" : "OutOfPlace");
  if (benchCase.offsetBytes != 0) {
    name += "/offset_" + std::to_string(benchCase.offsetBytes);
  }
  return name;
}

bool initializeBenchmarkConfig() {
  if (FLAGS_warmup_iters < 0) {
    std::cerr << "--warmup_iters must be >= 0" << std::endl;
    return false;
  }
  if (FLAGS_bench_iters <= 0) {
    std::cerr << "--bench_iters must be > 0" << std::endl;
    return false;
  }
  if (FLAGS_sizes.empty()) {
    gBenchmarkSizes = defaultBenchmarkSizes();
  } else {
    std::string error;
    if (!parseSizeList(FLAGS_sizes, &gBenchmarkSizes, &error)) {
      std::cerr << error << std::endl;
      return false;
    }
  }

  for (size_t size : gBenchmarkSizes) {
    if (size == 0 || size % sizeof(float) != 0) {
      std::cerr << "All benchmark sizes must be non-zero multiples of "
                << sizeof(float) << " bytes; got " << size << std::endl;
      return false;
    }
  }
  gBenchmarkCases = makeBenchmarkCases(gBenchmarkSizes);
  return true;
}

ctran::CtranEnvs benchmarkEnvOverrides() {
  ctran::CtranEnvs envs{
      {"NCCL_ALLREDUCE_ALGO", "ctree"},
      {"NCCL_CTRAN_USE_PIPES", "1"},
      {"NCCL_CTRAN_IBGDA_SENDRECV_ENABLE", "1"},
      {"NCCL_CTRAN_IBGDA_DATA_BUFFER_SIZE", "33554432"},
      {"NCCL_DEBUG", "WARN"},
  };
#if defined(CTRAN_ALLREDUCE_BENCH_IB_ONLY)
  envs.emplace_back("NCCL_COMM_STATE_DEBUG_TOPO", "nolocal");
  envs.emplace_back("NCCL_IGNORE_TOPO_LOAD_FAILURE", "1");
  envs.emplace_back("NCCL_P2P_DISABLE", "1");
#endif
  return envs;
}

class CtranAllReduceBenchmarkHarness final
    : public ctran::CtranDistTestFixture {
 public:
  ~CtranAllReduceBenchmarkHarness() {
    teardown();
  }

  void setup() {
    ctran::CtranDistTestFixture::SetUp(benchmarkEnvOverrides());
    isSetUp_ = true;
    CUDACHECK_TEST(cudaSetDevice(localRank));
    ctranComm_ = makeCtranComm();
    CUDACHECK_TEST(cudaStreamCreate(&stream_));

    if (!ctranAllReduceSupport(ctranComm_.get(), NCCL_ALLREDUCE_ALGO::ctree)) {
      throw std::runtime_error("ctree is not supported on this topology");
    }
  }

  void teardown() {
    if (!isSetUp_) {
      return;
    }
    if (stream_ != nullptr) {
      CUDACHECK_TEST(cudaStreamDestroy(stream_));
      stream_ = nullptr;
    }
    ctranComm_.reset();
    ctran::CtranDistTestFixture::TearDown();
    ncclCvarInit();
    isSetUp_ = false;
  }

  int rank() const {
    return globalRank;
  }

  int ranks() const {
    return numRanks;
  }

  void runBenchmark(benchmark::State& state, const BenchCase& benchCase) {
    void* sendBase = nullptr;
    void* recvBase = nullptr;
    const size_t allocBytes = benchCase.sizeBytes + benchCase.offsetBytes;
    CUDACHECK_TEST(cudaMalloc(&sendBase, allocBytes));
    if (!benchCase.inPlace) {
      CUDACHECK_TEST(cudaMalloc(&recvBase, allocBytes));
    }

    float* const sendbuf = reinterpret_cast<float*>(
        static_cast<char*>(sendBase) + benchCase.offsetBytes);
    float* const recvbuf = benchCase.inPlace
        ? sendbuf
        : reinterpret_cast<float*>(
              static_cast<char*>(recvBase) + benchCase.offsetBytes);
    const size_t count = benchCase.sizeBytes / sizeof(float);

    initializeBuffers(sendbuf, recvbuf, benchCase);
    barrier();
    for (int i = 0; i < FLAGS_warmup_iters; ++i) {
      runOnce(sendbuf, recvbuf, count);
    }
    CUDACHECK_TEST(cudaStreamSynchronize(stream_));
    barrier();

    cudaEvent_t start;
    cudaEvent_t stop;
    CUDACHECK_TEST(cudaEventCreate(&start));
    CUDACHECK_TEST(cudaEventCreate(&stop));
    int64_t actualIterations = 0;
    double totalMaxSec = 0.0;
    for (auto _ : state) {
      CUDACHECK_TEST(cudaEventRecord(start, stream_));
      runOnce(sendbuf, recvbuf, count);
      CUDACHECK_TEST(cudaEventRecord(stop, stream_));
      CUDACHECK_TEST(cudaEventSynchronize(stop));

      float iterMs = 0.0f;
      CUDACHECK_TEST(cudaEventElapsedTime(&iterMs, start, stop));
      const double maxIterSec = maxAcrossRanks(iterMs) / 1000.0;
      state.SetIterationTime(maxIterSec);
      totalMaxSec += maxIterSec;
      ++actualIterations;
    }
    CUDACHECK_TEST(cudaEventDestroy(start));
    CUDACHECK_TEST(cudaEventDestroy(stop));
    barrier();

    if (actualIterations > 0) {
      const double avgSec = totalMaxSec / static_cast<double>(actualIterations);
      const double algGbps =
          static_cast<double>(benchCase.sizeBytes) / avgSec / 1e9;
      const double busFactor = 2.0 * static_cast<double>(numRanks - 1) /
          static_cast<double>(numRanks);
      state.counters["Alg_GBps"] = algGbps;
      state.counters["Bus_GBps"] = algGbps * busFactor;
      state.counters["Offset_B"] = static_cast<double>(benchCase.offsetBytes);
    }

    CUDACHECK_TEST(cudaFree(sendBase));
    if (recvBase != nullptr) {
      CUDACHECK_TEST(cudaFree(recvBase));
    }
  }

 private:
  void TestBody() override {}

  void barrier() {
    const auto rc = ctranComm_->bootstrap_->barrier(globalRank, numRanks).get();
    CHECK_EQ(rc, 0) << "Bootstrap barrier failed";
  }

  double maxAcrossRanks(double value) {
    std::vector<char> buf(numRanks * sizeof(double));
    std::memcpy(
        buf.data() + globalRank * sizeof(double), &value, sizeof(double));
    const auto rc =
        ctranComm_->bootstrap_
            ->allGather(buf.data(), sizeof(double), globalRank, numRanks)
            .get();
    CHECK_EQ(rc, 0) << "Bootstrap allGather for timing failed";

    double maxValue = 0.0;
    for (int rank = 0; rank < numRanks; ++rank) {
      double rankValue = 0.0;
      std::memcpy(
          &rankValue, buf.data() + rank * sizeof(double), sizeof(double));
      maxValue = std::max(maxValue, rankValue);
    }
    return maxValue;
  }

  void initializeBuffers(
      float* sendbuf,
      float* recvbuf,
      const BenchCase& benchCase) {
    CUDACHECK_TEST(cudaMemset(sendbuf, globalRank, benchCase.sizeBytes));
    if (!benchCase.inPlace) {
      CUDACHECK_TEST(cudaMemset(recvbuf, 0, benchCase.sizeBytes));
    }
    CUDACHECK_TEST(cudaStreamSynchronize(stream_));
  }

  void runOnce(float* sendbuf, float* recvbuf, size_t count) {
    COMMCHECK_TEST(ctranAllReduceTree(
        sendbuf,
        recvbuf,
        count,
        commFloat32,
        commSum,
        ctranComm_.get(),
        stream_,
        /*timeout=*/std::nullopt));
  }

  std::unique_ptr<CtranComm> ctranComm_;
  cudaStream_t stream_{nullptr};
  bool isSetUp_{false};
};

void registerBenchmarks(CtranAllReduceBenchmarkHarness* harness) {
  if (harness->rank() == 0) {
    std::cout << "Registering CTRAN AllReduce benchmarks for " << topologyName()
              << " 1x" << harness->ranks() << std::endl;
  }
  for (const BenchCase& benchCase : gBenchmarkCases) {
    const std::string name = benchmarkName(benchCase);
    benchmark::RegisterBenchmark(
        name.c_str(),
        [harness, benchCase](benchmark::State& state) {
          harness->runBenchmark(state, benchCase);
        })
        ->Iterations(FLAGS_bench_iters)
        ->UseManualTime()
        ->Unit(benchmark::kMicrosecond);
  }
}

} // namespace

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::benchmark::Initialize(&argc, argv);
  folly::Init init(&argc, &argv);

  if (!initializeBenchmarkConfig()) {
    return EXIT_FAILURE;
  }

  ctran::CtranDistEnvironment distEnv;
  distEnv.SetUp();

  CtranAllReduceBenchmarkHarness harness;
  harness.setup();
  registerBenchmarks(&harness);
  ::benchmark::RunSpecifiedBenchmarks();
  harness.teardown();
  distEnv.TearDown();
  ::benchmark::Shutdown();
  return EXIT_SUCCESS;
}
