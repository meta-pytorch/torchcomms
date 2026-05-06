// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <folly/init/Init.h>
#include <folly/logging/xlog.h>

#include <cstddef>
#include <string>
#include <vector>

#include "comms/pipes/benchmarks/CopyDualBench.h"
#include "comms/utils/CudaRAII.h"

using meta::comms::DeviceBuffer;

namespace comms::pipes::benchmark {

class CopyDualBenchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    cudaSetDevice(0);
    cudaStreamCreate(&stream_);
  }

  void TearDown() override {
    cudaStreamDestroy(stream_);
  }

  cudaStream_t stream_{};
};

TEST_F(CopyDualBenchTest, Bandwidth) {
  constexpr int kNumBlocks = 8;
  constexpr int kNumThreads = 512;
  constexpr int kWarmupIters = 10;
  constexpr int kBenchIters = 100;
  constexpr int kRunsPerIter = 50;

  std::vector<std::size_t> messageSizes;
  for (std::size_t sz = 1ULL << 20; sz <= 1ULL << 27; sz <<= 1) {
    messageSizes.push_back(sz);
  }

  std::size_t maxBytes = messageSizes.back();
  DeviceBuffer srcBuf(maxBytes);
  DeviceBuffer dst1Buf(maxBytes);
  DeviceBuffer dst2Buf(maxBytes);
  cudaMemset(srcBuf.get(), 0xAA, maxBytes);
  cudaMemset(dst1Buf.get(), 0, maxBytes);
  cudaMemset(dst2Buf.get(), 0, maxBytes);
  cudaDeviceSynchronize();

  auto* src = static_cast<char*>(srcBuf.get());
  auto* dst1 = static_cast<char*>(dst1Buf.get());
  auto* dst2 = static_cast<char*>(dst2Buf.get());

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  XLOGF(INFO, "");
  XLOGF(
      INFO, "================================================================");
  XLOGF(
      INFO,
      "  memcpy_dual vs 2x memcpy_vectorized (D2D, {} blocks x {} threads)",
      kNumBlocks,
      kNumThreads);
  XLOGF(
      INFO, "================================================================");
  XLOGF(
      INFO,
      "{:>10s}  {:>14s}  {:>14s}  {:>10s}",
      "MsgSize",
      "dual GB/s",
      "2x_seq GB/s",
      "Speedup");
  XLOGF(INFO, "--------------------------------------------------------------");

  for (auto nBytes : messageSizes) {
    float dualBw = 0;
    float seqBw = 0;

    // --- memcpy_dual ---
    {
      for (int i = 0; i < kWarmupIters; ++i) {
        launch_copy_dual(
            dst1,
            dst2,
            src,
            nBytes,
            kRunsPerIter,
            kNumBlocks,
            kNumThreads,
            stream_);
      }
      cudaStreamSynchronize(stream_);

      cudaEventRecord(start, stream_);
      for (int i = 0; i < kBenchIters; ++i) {
        launch_copy_dual(
            dst1,
            dst2,
            src,
            nBytes,
            kRunsPerIter,
            kNumBlocks,
            kNumThreads,
            stream_);
      }
      cudaEventRecord(stop, stream_);
      cudaEventSynchronize(stop);

      float totalMs = 0;
      cudaEventElapsedTime(&totalMs, start, stop);
      float avgMs = totalMs / (kBenchIters * kRunsPerIter);
      dualBw = (nBytes / 1e9f) / (avgMs / 1000.0f);
    }

    // --- 2x memcpy_vectorized ---
    {
      for (int i = 0; i < kWarmupIters; ++i) {
        launch_copy_two_sequential(
            dst1,
            dst2,
            src,
            nBytes,
            kRunsPerIter,
            kNumBlocks,
            kNumThreads,
            stream_);
      }
      cudaStreamSynchronize(stream_);

      cudaEventRecord(start, stream_);
      for (int i = 0; i < kBenchIters; ++i) {
        launch_copy_two_sequential(
            dst1,
            dst2,
            src,
            nBytes,
            kRunsPerIter,
            kNumBlocks,
            kNumThreads,
            stream_);
      }
      cudaEventRecord(stop, stream_);
      cudaEventSynchronize(stop);

      float totalMs = 0;
      cudaEventElapsedTime(&totalMs, start, stop);
      float avgMs = totalMs / (kBenchIters * kRunsPerIter);
      seqBw = (nBytes / 1e9f) / (avgMs / 1000.0f);
    }

    std::string sizeStr;
    if (nBytes >= 1ULL << 20) {
      sizeStr = fmt::format("{}MB", nBytes >> 20);
    } else {
      sizeStr = fmt::format("{}KB", nBytes >> 10);
    }
    float speedup = (seqBw > 0) ? (dualBw / seqBw) : 0;
    XLOGF(
        INFO,
        "{:>10s}  {:>14.2f}  {:>14.2f}  {:>9.2f}x",
        sizeStr,
        dualBw,
        seqBw,
        speedup);
  }

  XLOGF(
      INFO, "================================================================");

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

} // namespace comms::pipes::benchmark

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
