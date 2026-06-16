// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <fmt/core.h>
#include <folly/init/Init.h>
#include <nccl.h>
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/algos/AllGather/AllGatherImpl.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"
#include "comms/testinfra/TestsCuUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

int gWarmupIters = 5;
int gBenchIters = 50;
std::string gCtranAlgo = "cthierarchical_ring";
std::optional<size_t> gSizeBytes;
bool gTotalSizeBytes = false;
// When 0, the bench does not touch NCCL_CTRAN_PIPES_TRACE_ENABLE (so an
// explicit env override still traces every iteration). When > 0, the bench owns
// the cvar, enables tracing during warmup, and sets it to (i % gTraceEvery ==
// 0) before each timed ctranAllGather call.
int gTraceEvery = 0;

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

namespace {

struct Result {
  size_t sizeBytes{};
  double ncclGbps{};
  double ctranGbps{};
  double ncclLatencyUs{};
  double ctranLatencyUs{};
};

std::string formatSize(size_t bytes) {
  if (bytes >= 1024UL * 1024 * 1024) {
    return fmt::format("{} GB", bytes / (1024UL * 1024 * 1024));
  }
  if (bytes >= 1024UL * 1024) {
    return fmt::format("{} MB", bytes / (1024UL * 1024));
  }
  return fmt::format("{} KB", bytes / 1024UL);
}

std::vector<size_t> benchmarkSizes() {
  if (gSizeBytes.has_value()) {
    return {*gSizeBytes};
  }
  std::vector<size_t> sizes;
  for (size_t size = 8 * 1024; size <= 1024UL * 1024 * 1024; size *= 2) {
    sizes.push_back(size);
  }
  return sizes;
}

std::optional<std::string> parseFlagValue(
    const std::string& arg,
    const std::string& name) {
  const std::string prefix = "--" + name + "=";
  if (arg.rfind(prefix, 0) == 0) {
    return arg.substr(prefix.size());
  }
  return std::nullopt;
}

bool parseBoolValue(const std::string& value) {
  return value == "1" || value == "true" || value == "TRUE" || value == "yes" ||
      value == "YES" || value == "on" || value == "ON";
}

class ScopedEnvVar {
 public:
  ScopedEnvVar(std::string name, std::optional<std::string> value)
      : name_(std::move(name)) {
    const char* oldValue = std::getenv(name_.c_str());
    oldValue_ = oldValue ? std::optional<std::string>(oldValue) : std::nullopt;
    if (value.has_value()) {
      setenv(name_.c_str(), value->c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

  ~ScopedEnvVar() {
    if (oldValue_.has_value()) {
      setenv(name_.c_str(), oldValue_->c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

 private:
  std::string name_;
  std::optional<std::string> oldValue_;
};

void parseBenchmarkFlags(int* argc, char** argv) {
  int write = 1;
  for (int read = 1; read < *argc; ++read) {
    const std::string arg = argv[read];
    if (auto value = parseFlagValue(arg, "warmup_iters")) {
      gWarmupIters = std::stoi(*value);
      continue;
    }
    if (auto value = parseFlagValue(arg, "bench_iters")) {
      gBenchIters = std::stoi(*value);
      continue;
    }
    if (auto value = parseFlagValue(arg, "ctran_algo")) {
      gCtranAlgo = *value;
      continue;
    }
    if (auto value = parseFlagValue(arg, "size_bytes")) {
      gSizeBytes = std::stoull(*value);
      continue;
    }
    if (auto value = parseFlagValue(arg, "total_size_bytes")) {
      gTotalSizeBytes = parseBoolValue(*value);
      continue;
    }
    if (auto value = parseFlagValue(arg, "trace_every")) {
      gTraceEvery = std::stoi(*value);
      continue;
    }
    argv[write++] = argv[read];
  }
  *argc = write;
}

void parseBenchmarkEnv() {
  if (const char* value = std::getenv("CTRAN_ALLGATHER_BENCH_WARMUP_ITERS")) {
    gWarmupIters = std::stoi(value);
  }
  if (const char* value = std::getenv("CTRAN_ALLGATHER_BENCH_ITERS")) {
    gBenchIters = std::stoi(value);
  }
  if (const char* value = std::getenv("CTRAN_ALLGATHER_BENCH_ALGO")) {
    gCtranAlgo = value;
  }
  if (const char* value = std::getenv("CTRAN_ALLGATHER_BENCH_SIZE_BYTES")) {
    gSizeBytes = std::stoull(value);
  }
  if (const char* value =
          std::getenv("CTRAN_ALLGATHER_BENCH_TOTAL_SIZE_BYTES")) {
    gTotalSizeBytes = parseBoolValue(value);
  }
  if (const char* value = std::getenv("CTRAN_ALLGATHER_BENCH_TRACE_EVERY")) {
    gTraceEvery = std::stoi(value);
  }
}

bool validateBenchmarkConfig() {
  if (gWarmupIters < 0) {
    std::cerr
        << "--warmup_iters / CTRAN_ALLGATHER_BENCH_WARMUP_ITERS must be >= 0"
        << std::endl;
    return false;
  }
  if (gBenchIters <= 0) {
    std::cerr << "--bench_iters / CTRAN_ALLGATHER_BENCH_ITERS must be > 0"
              << std::endl;
    return false;
  }
  if (gTraceEvery < 0) {
    std::cerr
        << "--trace_every / CTRAN_ALLGATHER_BENCH_TRACE_EVERY must be >= 0"
        << std::endl;
    return false;
  }
  return true;
}

enum NCCL_ALLGATHER_ALGO parseCtranAlgo(const std::string& algo) {
  if (algo == "cthierarchical_ring") {
    return NCCL_ALLGATHER_ALGO::cthierarchical_ring;
  }
  if (algo == "ctring") {
    return NCCL_ALLGATHER_ALGO::ctring;
  }
  if (algo == "ctdirect") {
    return NCCL_ALLGATHER_ALGO::ctdirect;
  }
  if (algo == "ctbrucks") {
    return NCCL_ALLGATHER_ALGO::ctbrucks;
  }
  std::cerr << "Unsupported --ctran_algo=" << algo << std::endl;
  std::abort();
}

} // namespace

class CtranAllGatherBenchmark : public ctran::CtranDistTestFixture {
 public:
  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    setenv("NCCL_CTRAN_USE_PIPES", "1", 0);
    setenv("NCCL_CTRAN_IBGDA_SENDRECV_ENABLE", "1", 0);
    setenv("NCCL_CTRAN_IBGDA_DATA_BUFFER_SIZE", "8388608", 0);
    setenv("NCCL_DEBUG", "WARN", 0);

    ctran::CtranDistTestFixture::SetUp();
    CUDACHECK_TEST(cudaSetDevice(localRank));
    ctranComm_ = makeCtranComm();
    CUDACHECK_TEST(cudaStreamCreate(&stream_));

    ctranAlgo_ = parseCtranAlgo(gCtranAlgo);
    if (!ctranAllGatherSupport(ctranComm_.get(), ctranAlgo_, stream_)) {
      GTEST_SKIP() << allGatherAlgoName(ctranAlgo_)
                   << " is not supported on this topology";
    }

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

    {
      ScopedEnvVar disableCtran(
          "NCCL_CTRAN_ENABLE", std::optional<std::string>{"0"});
      ScopedEnvVar disablePipes(
          "NCCL_CTRAN_USE_PIPES", std::optional<std::string>{"0"});
      ScopedEnvVar unsetAllGatherAlgo("NCCL_ALLGATHER_ALGO", std::nullopt);
      ncclCvarInit();
      NCCLCHECK_TEST(
          ncclCommInitRank(&ncclComm_, numRanks, ncclId, globalRank));
    }
    ncclCvarInit();
  }

  void TearDown() override {
    if (ncclComm_ != nullptr) {
      ncclCommDestroy(ncclComm_);
    }
    if (stream_ != nullptr) {
      CUDACHECK_TEST(cudaStreamDestroy(stream_));
    }
    ctran::CtranDistTestFixture::TearDown();
  }

  double maxAcrossRanks(double value) {
    std::vector<char> buf(numRanks * sizeof(double));
    memcpy(buf.data() + globalRank * sizeof(double), &value, sizeof(double));
    auto rc = ctranComm_->bootstrap_
                  ->allGather(buf.data(), sizeof(double), globalRank, numRanks)
                  .get();
    EXPECT_EQ(rc, 0) << "Bootstrap allGather for timing failed";

    double maxValue = 0.0;
    for (int rank = 0; rank < numRanks; ++rank) {
      double rankValue;
      memcpy(&rankValue, buf.data() + rank * sizeof(double), sizeof(double));
      maxValue = std::max(maxValue, rankValue);
    }
    return maxValue;
  }

  void barrier() {
    auto rc = ctranComm_->bootstrap_->barrier(globalRank, numRanks).get();
    EXPECT_EQ(rc, 0) << "Bootstrap barrier failed";
  }

  double runNccl(size_t sizeBytes, void* sendbuf, void* recvbuf) {
    CUDACHECK_TEST(cudaMemset(sendbuf, globalRank, sizeBytes));
    CUDACHECK_TEST(cudaMemset(recvbuf, 0, sizeBytes * numRanks));
    CUDACHECK_TEST(cudaStreamSynchronize(stream_));

    barrier();
    for (int i = 0; i < gWarmupIters; ++i) {
      NCCLCHECK_TEST(ncclAllGather(
          sendbuf, recvbuf, sizeBytes, ncclChar, ncclComm_, stream_));
    }
    CUDACHECK_TEST(cudaStreamSynchronize(stream_));
    barrier();

    cudaEvent_t start, stop;
    CUDACHECK_TEST(cudaEventCreate(&start));
    CUDACHECK_TEST(cudaEventCreate(&stop));
    CUDACHECK_TEST(cudaEventRecord(start, stream_));
    for (int i = 0; i < gBenchIters; ++i) {
      NCCLCHECK_TEST(ncclAllGather(
          sendbuf, recvbuf, sizeBytes, ncclChar, ncclComm_, stream_));
    }
    CUDACHECK_TEST(cudaEventRecord(stop, stream_));
    CUDACHECK_TEST(cudaStreamSynchronize(stream_));

    float totalMs = 0.0f;
    CUDACHECK_TEST(cudaEventElapsedTime(&totalMs, start, stop));
    CUDACHECK_TEST(cudaEventDestroy(start));
    CUDACHECK_TEST(cudaEventDestroy(stop));
    return maxAcrossRanks(totalMs / gBenchIters);
  }

  double runCtran(size_t sizeBytes, void* sendbuf, void* recvbuf) {
    CUDACHECK_TEST(cudaMemset(sendbuf, globalRank, sizeBytes));
    CUDACHECK_TEST(cudaMemset(recvbuf, 0, sizeBytes * numRanks));
    CUDACHECK_TEST(cudaStreamSynchronize(stream_));

    barrier();
    if (gTraceEvery > 0) {
      NCCL_CTRAN_PIPES_TRACE_ENABLE = true;
    }
    for (int i = 0; i < gWarmupIters; ++i) {
      COMMCHECK_TEST(ctranAllGather(
          sendbuf,
          recvbuf,
          sizeBytes,
          commChar,
          ctranComm_.get(),
          stream_,
          ctranAlgo_));
    }
    CUDACHECK_TEST(cudaStreamSynchronize(stream_));
    barrier();

    cudaEvent_t start, stop;
    CUDACHECK_TEST(cudaEventCreate(&start));
    CUDACHECK_TEST(cudaEventCreate(&stop));
    CUDACHECK_TEST(cudaEventRecord(start, stream_));
    for (int i = 0; i < gBenchIters; ++i) {
      if (gTraceEvery > 0) {
        NCCL_CTRAN_PIPES_TRACE_ENABLE = (i % gTraceEvery == 0);
      }
      COMMCHECK_TEST(ctranAllGather(
          sendbuf,
          recvbuf,
          sizeBytes,
          commChar,
          ctranComm_.get(),
          stream_,
          ctranAlgo_));
    }
    CUDACHECK_TEST(cudaEventRecord(stop, stream_));
    CUDACHECK_TEST(cudaStreamSynchronize(stream_));

    float totalMs = 0.0f;
    CUDACHECK_TEST(cudaEventElapsedTime(&totalMs, start, stop));
    CUDACHECK_TEST(cudaEventDestroy(start));
    CUDACHECK_TEST(cudaEventDestroy(stop));
    return maxAcrossRanks(totalMs / gBenchIters);
  }

  void printResults(const std::vector<Result>& results) {
    if (globalRank != 0) {
      return;
    }

    std::cout << "\nNCCL vs ctran AllGather " << allGatherAlgoName(ctranAlgo_)
              << "\n";
    std::cout << std::left << std::setw(10) << "Size" << std::right
              << std::setw(12) << "NCCL GB/s" << std::setw(13) << "Ctran GB/s"
              << std::setw(9) << "Ratio" << std::setw(14) << "NCCL us"
              << std::setw(14) << "Ctran us" << "\n";
    for (const auto& result : results) {
      const double ratio = result.ctranGbps / result.ncclGbps;
      std::cout << std::left << std::setw(10) << formatSize(result.sizeBytes)
                << std::right << std::fixed << std::setprecision(2)
                << std::setw(12) << result.ncclGbps << std::setw(13)
                << result.ctranGbps << std::setw(8) << ratio << "x"
                << std::setprecision(1) << std::setw(14) << result.ncclLatencyUs
                << std::setw(14) << result.ctranLatencyUs << "\n";
    }
    std::cout << std::flush;
  }

  void runBenchmark() {
    if (globalRank == 0 && gTraceEvery > 0) {
      std::cout << "Pipes trace sampling: every " << gTraceEvery
                << " timed iterations (algo=" << allGatherAlgoName(ctranAlgo_)
                << ")\n"
                << std::flush;
    }
    const auto sizes = benchmarkSizes();
    const auto sendSizeBytes = [&](size_t sizeBytes) {
      if (!gTotalSizeBytes) {
        return sizeBytes;
      }
      EXPECT_EQ(sizeBytes % numRanks, 0)
          << "Total size must be divisible by numRanks";
      return sizeBytes / numRanks;
    };

    const size_t maxSizeBytes = sendSizeBytes(sizes.back());
    void* sendbuf = nullptr;
    void* recvbuf = nullptr;
    CUDACHECK_TEST(cudaMalloc(&sendbuf, maxSizeBytes));
    CUDACHECK_TEST(cudaMalloc(&recvbuf, maxSizeBytes * numRanks));

    std::vector<Result> results;
    for (const size_t resultSizeBytes : sizes) {
      const size_t sizeBytes = sendSizeBytes(resultSizeBytes);
      const double ncclMs = runNccl(sizeBytes, sendbuf, recvbuf);
      const double ctranMs = runCtran(sizeBytes, sendbuf, recvbuf);
      const double totalBytes = gTotalSizeBytes
          ? static_cast<double>(resultSizeBytes)
          : static_cast<double>(sizeBytes) * numRanks;
      results.push_back(
          Result{
              .sizeBytes = resultSizeBytes,
              .ncclGbps = totalBytes / (ncclMs / 1000.0) / 1e9,
              .ctranGbps = totalBytes / (ctranMs / 1000.0) / 1e9,
              .ncclLatencyUs = ncclMs * 1000.0,
              .ctranLatencyUs = ctranMs * 1000.0,
          });
      if (globalRank == 0) {
        const auto& result = results.back();
        std::cout << "Finished " << formatSize(resultSizeBytes)
                  << ": NCCL=" << result.ncclGbps
                  << " GB/s ctran=" << result.ctranGbps << " GB/s\n";
      }
    }

    printResults(results);
    CUDACHECK_TEST(cudaFree(sendbuf));
    CUDACHECK_TEST(cudaFree(recvbuf));
  }

 private:
  std::unique_ptr<CtranComm> ctranComm_;
  ncclComm_t ncclComm_{nullptr};
  cudaStream_t stream_{nullptr};
  enum NCCL_ALLGATHER_ALGO ctranAlgo_ {
    NCCL_ALLGATHER_ALGO::cthierarchical_ring
  };
};

TEST_F(CtranAllGatherBenchmark, VsNccl) {
  runBenchmark();
}

int main(int argc, char* argv[]) {
  parseBenchmarkEnv();
  parseBenchmarkFlags(&argc, argv);
  if (!validateBenchmarkConfig()) {
    return EXIT_FAILURE;
  }
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new ctran::CtranDistEnvironment);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
