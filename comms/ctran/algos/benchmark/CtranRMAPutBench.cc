// Copyright (c) Meta Platforms, Inc. and affiliates.

// Benchmark test for ctranPutSignal() with signal=true
// Measures latency and bandwidth between a single sender (rank 0) and
// single receiver (rank 1) over 200 iterations for various message sizes.

#include <chrono>
#include <thread>
#include <vector>

#include <folly/Benchmark.h>
#include <folly/BenchmarkUtil.h>
#include <folly/init/Init.h>
#include <gtest/gtest.h>

#include "comms/ctran/Ctran.h"
#include "comms/ctran/tests/CtranDistTestUtils.h"

using namespace ctran;

// Benchmark configuration
constexpr int kNumIterations = 200;
constexpr int kWarmupIterations = 10;
constexpr int kSenderRank = 0;
constexpr int kReceiverRank = 1;

// Global rank storage for use in main() after tests complete
static int g_globalRank = -1;

// Message sizes to benchmark (in bytes) - up to 1GB
const std::vector<size_t> kMessageSizes = {
    64,
    256,
    1024,
    4 * 1024,
    8 * 1024,
    16 * 1024,
    32 * 1024,
    64 * 1024,
    128 * 1024,
    256 * 1024,
    512 * 1024,
    1024 * 1024,
    2 * 1024 * 1024,
    4 * 1024 * 1024,
    8 * 1024 * 1024,
    16 * 1024 * 1024,
    32 * 1024 * 1024,
    64 * 1024 * 1024,
    128 * 1024 * 1024,
    256 * 1024 * 1024,
    512 * 1024 * 1024,
    1024ULL * 1024 * 1024, // 1GB
};

namespace {

// Format bytes as human-readable string (e.g., "64B", "1KB", "1MB", "1GB")
std::string formatSize(size_t bytes) {
  if (bytes >= 1024ULL * 1024 * 1024) {
    return std::to_string(bytes / (1024ULL * 1024 * 1024)) + "GB";
  } else if (bytes >= 1024 * 1024) {
    return std::to_string(bytes / (1024 * 1024)) + "MB";
  } else if (bytes >= 1024) {
    return std::to_string(bytes / 1024) + "KB";
  } else {
    return std::to_string(bytes) + "B";
  }
}

// Register a pre-measured benchmark result with folly's infrastructure
// Uses folly::detail::addBenchmarkImpl to inject timing data directly
void registerBenchmarkResult(
    const std::string& name,
    std::chrono::nanoseconds duration,
    unsigned int iterations,
    double bandwidthGBps,
    double msgRatePerSec) {
  folly::UserCounters counters;
  // Use Type::METRIC to get proper floating point formatting in output
  // (Type::CUSTOM uses PRId64 which truncates to integer)
  counters["bw_GBps"] =
      folly::UserMetric{bandwidthGBps, folly::UserMetric::Type::METRIC};
  counters["msg_rate"] =
      folly::UserMetric{msgRatePerSec, folly::UserMetric::Type::METRIC};

  folly::detail::addBenchmarkImpl(
      __FILE__,
      name,
      [duration, iterations, counters](unsigned int /* n */) {
        return folly::detail::TimeIterData{duration, iterations, counters};
      },
      true /* useCounter */);
}

} // namespace

class CtranRMAPutBenchEnv : public ctran::CtranEnvironmentBase {
 public:
  void SetUp() override {
    ctran::CtranEnvironmentBase::SetUp();
    // Suppress INFO and WARN level logs for cleaner benchmark output
    setenv("NCCL_DEBUG", "ERROR", 1);
    // Suppress folly/glog INFO level logs (0=INFO, 1=WARNING, 2=ERROR, 3=FATAL)
    setenv("GLOG_minloglevel", "2", 1);
  }
};

class CtranRMAPutBench : public ctran::CtranDistTestFixture {
 public:
  CtranRMAPutBench() = default;

 protected:
  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE", "1", 0);
    ctran::CtranDistTestFixture::SetUp();

    // Store the global rank for use in main() after tests complete
    g_globalRank = this->globalRank;

    ctranComm_ = this->makeCtranComm();
    ASSERT_NE(ctranComm_, nullptr);

    CUDACHECK_TEST(
        cudaStreamCreateWithFlags(&putStream_, cudaStreamNonBlocking));
    CUDACHECK_TEST(
        cudaStreamCreateWithFlags(&waitStream_, cudaStreamNonBlocking));
  }

  void TearDown() override {
    CUDACHECK_TEST(cudaStreamDestroy(putStream_));
    CUDACHECK_TEST(cudaStreamDestroy(waitStream_));
    ctranComm_.reset();
    ctran::CtranDistTestFixture::TearDown();
  }

  void barrier() {
    auto resFuture = ctranComm_->bootstrap_->barrier(
        ctranComm_->statex_->rank(), ctranComm_->statex_->nRanks());
    ASSERT_EQ(
        static_cast<commResult_t>(std::move(resFuture).get()), commSuccess);
  }

  template <typename T>
  bool verifyBuffer(T* buf, size_t count, T expectedValue) {
    std::vector<T> hostData(count);
    CUDACHECK_TEST(cudaMemcpy(
        hostData.data(), buf, count * sizeof(T), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < count; ++i) {
      if (hostData[i] != expectedValue) {
        return false;
      }
    }
    return true;
  }

  std::unique_ptr<CtranComm> ctranComm_;
  cudaStream_t putStream_{nullptr};
  cudaStream_t waitStream_{nullptr};
};

TEST_F(CtranRMAPutBench, BenchmarkPutSignal) {
  ASSERT_EQ(this->numRanks, 2) << "This benchmark requires exactly 2 ranks";

  auto statex = ctranComm_->statex_.get();
  ASSERT_NE(statex, nullptr);

  size_t maxMessageSize =
      *std::max_element(kMessageSizes.begin(), kMessageSizes.end());

  // Allocate window buffer on receiver
  CtranWin* win = nullptr;
  void* winBase = nullptr;
  COMMCHECK_TEST(
      ctranWinAllocate(maxMessageSize, ctranComm_.get(), &winBase, &win));

  // Allocate and register send buffer to avoid "not pre-registered" warnings
  char* sendBuf = nullptr;
  void* sendHdl = nullptr;
  CUDACHECK_TEST(cudaMalloc(&sendBuf, maxMessageSize));
  COMMCHECK_TEST(
      ctranComm_->ctran_->commRegister(sendBuf, maxMessageSize, &sendHdl));

  // Initialize buffers
  CUDACHECK_TEST(cudaMemset(sendBuf, this->globalRank + 1, maxMessageSize));
  CUDACHECK_TEST(cudaMemset(winBase, 0, maxMessageSize));

  // Create CUDA events for GPU-side timing
  cudaEvent_t startEvent, endEvent;
  CUDACHECK_TEST(cudaEventCreate(&startEvent));
  CUDACHECK_TEST(cudaEventCreate(&endEvent));

  barrier();

  bool allPassed = true;

  for (size_t msgSize : kMessageSizes) {
    // Reset window buffer
    CUDACHECK_TEST(cudaMemset(winBase, 0, msgSize));
    CUDACHECK_TEST(cudaDeviceSynchronize());
    barrier();

    // Warmup iterations
    for (int iter = 0; iter < kWarmupIterations; ++iter) {
      if (this->globalRank == kSenderRank) {
        COMMCHECK_TEST(ctranPutSignal(
            sendBuf,
            msgSize,
            commInt8,
            kReceiverRank,
            0,
            win,
            putStream_,
            true));
      }
      if (this->globalRank == kReceiverRank) {
        COMMCHECK_TEST(ctranWaitSignal(kSenderRank, win, waitStream_));
      }
    }
    CUDACHECK_TEST(cudaStreamSynchronize(putStream_));
    CUDACHECK_TEST(cudaStreamSynchronize(waitStream_));
    barrier();

    // Record start event on the appropriate stream
    cudaStream_t timedStream =
        (this->globalRank == kSenderRank) ? putStream_ : waitStream_;
    CUDACHECK_TEST(cudaEventRecord(startEvent, timedStream));

    // Timed iterations
    for (int iter = 0; iter < kNumIterations; ++iter) {
      if (this->globalRank == kSenderRank) {
        COMMCHECK_TEST(ctranPutSignal(
            sendBuf,
            msgSize,
            commInt8,
            kReceiverRank,
            0,
            win,
            putStream_,
            true));
      }
      if (this->globalRank == kReceiverRank) {
        COMMCHECK_TEST(ctranWaitSignal(kSenderRank, win, waitStream_));
      }
    }

    // Record end event and synchronize
    CUDACHECK_TEST(cudaEventRecord(endEvent, timedStream));
    CUDACHECK_TEST(cudaEventSynchronize(endEvent));

    // barrier();

    // Calculate elapsed time using CUDA events (in milliseconds)
    float elapsedMs = 0.0f;
    CUDACHECK_TEST(cudaEventElapsedTime(&elapsedMs, startEvent, endEvent));

    // Convert to microseconds and calculate metrics
    double elapsedUs = static_cast<double>(elapsedMs) * 1000.0;
    double avgLatencyUs = elapsedUs / kNumIterations;
    double totalBytes =
        static_cast<double>(msgSize) * static_cast<double>(kNumIterations);
    double elapsedSec = elapsedUs / 1e6;
    double bandwidthGBps = (totalBytes / elapsedSec) / 1e9;
    double msgRatePerSec = static_cast<double>(kNumIterations) / elapsedSec;

    // Register this benchmark result with folly's benchmark infrastructure
    if (this->globalRank == kSenderRank) {
      registerBenchmarkResult(
          "OneWayPutSignal_" + formatSize(msgSize),
          std::chrono::nanoseconds(
              static_cast<int64_t>(avgLatencyUs * 1000.0 * kNumIterations)),
          kNumIterations,
          bandwidthGBps,
          msgRatePerSec);
    }

    // Verify data on receiver
    if (this->globalRank == kReceiverRank) {
      bool verified = verifyBuffer(
          reinterpret_cast<char*>(winBase),
          msgSize,
          static_cast<char>(kSenderRank + 1));
      if (!verified) {
        allPassed = false;
      }
    }

    barrier();
  }

  // Cleanup
  CUDACHECK_TEST(cudaEventDestroy(startEvent));
  CUDACHECK_TEST(cudaEventDestroy(endEvent));
  COMMCHECK_TEST(ctranComm_->ctran_->commDeregister(sendHdl));
  CUDACHECK_TEST(cudaFree(sendBuf));
  COMMCHECK_TEST(ctranWinFree(win));

  EXPECT_TRUE(allPassed) << "Data verification failed for one or more sizes";
}

TEST_F(CtranRMAPutBench, BenchmarkTwoWayPutSignal) {
  ASSERT_EQ(this->numRanks, 2) << "This benchmark requires exactly 2 ranks";

  auto statex = ctranComm_->statex_.get();
  ASSERT_NE(statex, nullptr);

  size_t maxMessageSize =
      *std::max_element(kMessageSizes.begin(), kMessageSizes.end());

  // Allocate window buffer on both ranks (both will receive)
  CtranWin* win = nullptr;
  void* winBase = nullptr;
  COMMCHECK_TEST(
      ctranWinAllocate(maxMessageSize, ctranComm_.get(), &winBase, &win));

  // Allocate and register send buffer on both ranks
  char* sendBuf = nullptr;
  void* sendHdl = nullptr;
  CUDACHECK_TEST(cudaMalloc(&sendBuf, maxMessageSize));
  COMMCHECK_TEST(
      ctranComm_->ctran_->commRegister(sendBuf, maxMessageSize, &sendHdl));

  // Initialize buffers
  CUDACHECK_TEST(cudaMemset(sendBuf, this->globalRank + 1, maxMessageSize));
  CUDACHECK_TEST(cudaMemset(winBase, 0, maxMessageSize));

  // Create CUDA events for GPU-side timing
  cudaEvent_t startEvent, endEvent;
  CUDACHECK_TEST(cudaEventCreate(&startEvent));
  CUDACHECK_TEST(cudaEventCreate(&endEvent));

  // Determine peer rank for two-way communication
  int peerRank =
      (this->globalRank == kSenderRank) ? kReceiverRank : kSenderRank;

  barrier();

  bool allPassed = true;

  for (size_t msgSize : kMessageSizes) {
    // Reset window buffer
    CUDACHECK_TEST(cudaMemset(winBase, 0, msgSize));
    CUDACHECK_TEST(cudaDeviceSynchronize());
    barrier();

    // Warmup iterations - both ranks send and receive simultaneously
    for (int iter = 0; iter < kWarmupIterations; ++iter) {
      // Both ranks send to each other
      COMMCHECK_TEST(ctranPutSignal(
          sendBuf, msgSize, commInt8, peerRank, 0, win, putStream_, true));
      // Both ranks wait for data from peer
      COMMCHECK_TEST(ctranWaitSignal(peerRank, win, waitStream_));
    }
    CUDACHECK_TEST(cudaStreamSynchronize(putStream_));
    CUDACHECK_TEST(cudaStreamSynchronize(waitStream_));
    barrier();

    // Record start event
    CUDACHECK_TEST(cudaEventRecord(startEvent, putStream_));

    // Timed iterations - both ranks send and receive simultaneously
    for (int iter = 0; iter < kNumIterations; ++iter) {
      // Both ranks send to each other
      COMMCHECK_TEST(ctranPutSignal(
          sendBuf, msgSize, commInt8, peerRank, 0, win, putStream_, true));
      // Both ranks wait for data from peer
      COMMCHECK_TEST(ctranWaitSignal(peerRank, win, waitStream_));
    }

    // Record end event on putStream_ and synchronize waitStream_ to ensure
    // full bidirectional communication latency is captured (both send and
    // receive)
    CUDACHECK_TEST(cudaEventRecord(endEvent, putStream_));
    CUDACHECK_TEST(cudaStreamSynchronize(waitStream_));
    CUDACHECK_TEST(cudaEventSynchronize(endEvent));

    // Calculate elapsed time using CUDA events (in milliseconds)
    float elapsedMs = 0.0f;
    CUDACHECK_TEST(cudaEventElapsedTime(&elapsedMs, startEvent, endEvent));

    // Convert to microseconds and calculate metrics
    // For two-way, we measure bidirectional bandwidth (2x data transferred)
    double elapsedUs = static_cast<double>(elapsedMs) * 1000.0;
    double avgLatencyUs = elapsedUs / kNumIterations;
    double totalBytes = 2.0 * static_cast<double>(msgSize) *
        static_cast<double>(kNumIterations);
    double elapsedSec = elapsedUs / 1e6;
    double bandwidthGBps = (totalBytes / elapsedSec) / 1e9;
    double msgRatePerSec =
        2.0 * static_cast<double>(kNumIterations) / elapsedSec;

    // Register this benchmark result with folly's benchmark infrastructure
    // Only register on rank 0 to avoid duplicate registrations
    if (this->globalRank == kSenderRank) {
      registerBenchmarkResult(
          "TwoWayPutSignal_" + formatSize(msgSize),
          std::chrono::nanoseconds(
              static_cast<int64_t>(avgLatencyUs * 1000.0 * kNumIterations)),
          kNumIterations,
          bandwidthGBps,
          msgRatePerSec);
    }

    // Verify data on both ranks
    bool verified = verifyBuffer(
        reinterpret_cast<char*>(winBase),
        msgSize,
        static_cast<char>(peerRank + 1));
    if (!verified) {
      allPassed = false;
    }

    barrier();
  }

  // Cleanup
  CUDACHECK_TEST(cudaEventDestroy(startEvent));
  CUDACHECK_TEST(cudaEventDestroy(endEvent));
  COMMCHECK_TEST(ctranComm_->ctran_->commDeregister(sendHdl));
  CUDACHECK_TEST(cudaFree(sendBuf));
  COMMCHECK_TEST(ctranWinFree(win));

  EXPECT_TRUE(allPassed) << "Data verification failed for one or more sizes";
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new CtranRMAPutBenchEnv);
  folly::Init init(&argc, &argv);
  int result = RUN_ALL_TESTS();

  // Output benchmark results in folly's standard format
  // Only print on rank 0 to avoid output from multiple ranks
  if (g_globalRank == 0) {
    // Wait for any background activity to settle before printing results
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    folly::runBenchmarks();
  }

  return result;
}
