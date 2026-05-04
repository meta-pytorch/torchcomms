// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>

#include <atomic>
#include <thread>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "comms/ctran/algos/common/GpeKernelSync.h"
#include "comms/testinfra/TestXPlatUtils.h"

using ctran::algos::GpeKernelSync;

extern __global__ void
GpeKernelSyncKernel(GpeKernelSync* sync, int* data, int numElem, int nSteps);
extern __global__ void GpeKernelSyncResetKernel(
    GpeKernelSync* sync,
    const int nworkers);

class CtranGpeKernelSyncTest : public ::testing::Test {
 public:
  CtranGpeKernelSyncTest() = default;

 protected:
  static void SetUpTestCase() {}

  void SetUp() override {
    CUDACHECK_TEST(cudaSetDevice(cudaDev_));
    CUDACHECK_TEST(cudaEventCreate(&start_));
    CUDACHECK_TEST(cudaEventCreate(&stop_));
  }
  void TearDown() override {
    CUDACHECK_TEST(cudaEventDestroy(start_));
    CUDACHECK_TEST(cudaEventDestroy(stop_));
  }

 protected:
  int cudaDev_{0};
  cudaEvent_t start_;
  cudaEvent_t stop_;
};

enum class ResetMode {
  kFromHost,
  kFromDevice,
};

class CtranGpeKernelSyncTestParamFixture
    : public CtranGpeKernelSyncTest,
      // number of thread blocks for each sync group
      public ::testing::WithParamInterface<std::tuple<int, ResetMode>> {};

TEST_P(CtranGpeKernelSyncTestParamFixture, kernelSync) {
  auto [nWorkers, resetType] = GetParam();

  int numElem = 8192;
  int niter = 10;
  void* ptr = nullptr;
  ASSERT_EQ(
      cudaHostAlloc(
          &ptr,
          sizeof(GpeKernelSync) + sizeof(int) * numElem * 3,
          cudaHostAllocDefault),
      cudaSuccess);

  // Assign sync and data pointers from the allocated memory
  GpeKernelSync* sync = reinterpret_cast<GpeKernelSync*>(ptr);

  int* data = reinterpret_cast<int*>(
      reinterpret_cast<char*>(ptr) + sizeof(GpeKernelSync));
  for (int e = 0; e < numElem; ++e) {
    data[e] = e;
  }

  if (resetType == ResetMode::kFromDevice) {
    std::array<void*, 2> resetArgs;
    resetArgs.at(0) = &sync;
    resetArgs.at(1) = &nWorkers;
    CUDACHECK_ASSERT(cudaLaunchKernel(
        (const void*)GpeKernelSyncResetKernel,
        {1, 1, 1},
        {32, 1, 1},
        resetArgs.data(),
        0,
        0));
    CUDACHECK_ASSERT(cudaDeviceSynchronize());
  } else {
    new (sync) GpeKernelSync(nWorkers);
  }

  std::array<void*, 4> kernArgs;
  kernArgs.at(0) = &sync;
  kernArgs.at(1) = &data;
  kernArgs.at(2) = &numElem;
  kernArgs.at(3) = &niter;
  dim3 grid = {(unsigned int)nWorkers, 1, 1};
  dim3 blocks = {128, 1, 1};
  ASSERT_EQ(
      cudaFuncSetAttribute(
          (const void*)GpeKernelSyncKernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          sizeof(CtranAlgoDeviceState)),
      cudaSuccess);
  ASSERT_EQ(
      cudaLaunchKernel(
          (const void*)GpeKernelSyncKernel,
          grid,
          blocks,
          kernArgs.data(),
          sizeof(CtranAlgoDeviceState),
          0),
      cudaSuccess);

  for (int i = 0; i < niter; ++i) {
    for (int e = 0; e < numElem; ++e) {
      data[e] += i;
    }

    sync->post(i);

    while (!sync->isComplete(i)) {
      std::this_thread::yield();
    }

    for (int e = 0; e < numElem; ++e) {
      ASSERT_EQ(data[e], e + i * (i + 1)) << " at " << e << " iteration " << i;
    }
  }

  ASSERT_EQ(cudaFreeHost(ptr), cudaSuccess);
}

INSTANTIATE_TEST_SUITE_P(
    CtranTest,
    CtranGpeKernelSyncTestParamFixture,
    ::testing::Combine(
        ::testing::Values(1, 2, 4, 8, 16, 32, 64),
        ::testing::Values(ResetMode::kFromHost, ResetMode::kFromDevice)),
    [&](const testing::TestParamInfo<
        CtranGpeKernelSyncTestParamFixture::ParamType>& info) {
      const auto resetTypeStr = std::get<1>(info.param) == ResetMode::kFromHost
          ? "ResetFromHost"
          : "ResetFromDevice";
      return std::to_string(std::get<0>(info.param)) + "numWorkers_" +
          resetTypeStr;
    });

// =============================================================================
// AcquireRace test — exercises the GpeKernelSync host poll → consumer-data-read
// pattern that the production bug at S651852 hits on aarch64 (GB200).
//
// Producer threads simulate kernel blocks: each writes its own portion of a
// shared payload, then release-stores its `completeFlag[b]` slot to signal
// completion of step `iter`. The release-fence + volatile-store pattern
// mirrors the GPU's `st.release.sys` in `GpeKernelSyncDev::complete()`.
//
// The consumer thread is the host: it calls `sync->isComplete(iter)` (the
// production code path) and immediately reads the payload to verify
// consistency.
//
// Iteration handshake uses a separate `std::atomic<int> goSignal` (a proper
// C++11 release/acquire pair) so this test does NOT race on
// `sync->postFlag[]`. The only intentionally race-relevant interaction is the
// producer's release-store of `completeFlag` paired with the consumer's poll
// inside `sync->isComplete()`.
//
// On aarch64 without `wcAcquireFence()` after the polling load, the consumer
// can observe `completeFlag` as set while a load-load reorder lets the
// payload read return stale values, producing per-(iter, block) mismatches.
//
// On x86 (TSO) the race cannot trigger naturally — load->load ordering is
// architecturally guaranteed. ThreadSanitizer is also not a useful validator
// here: TSan models the C++11 memory model, where the production
// `volatile + thread_fence` pattern is not a recognised release/acquire pair,
// so TSan reports the bare volatile read as a race regardless of whether
// `wcAcquireFence()` is present. The fix is hardware-correct but cannot be
// validated by TSan without refactoring `GpeKernelSync` to use `std::atomic`.
//
// With `wcAcquireFence()` in `GpeKernelSync::isComplete()` / `waitComplete()`
// (see D103040491), this test passes on aarch64 GB200. Without it the race
// triggers and `mismatches` is non-zero. On x86 the test always passes
// regardless of the fence (TSO masks the race architecturally).
//
// Run on aarch64 GB200:
//
//   buck2 run @fbcode//mode/opt -c hpc_comms.use_ncclx=stable \
//     -c nvcc.arch=b200 -c fbcode.arch=aarch64 \
//     fbcode//comms/ctran/algos/common/tests:ctran_algo_gpe_kernel_sync_test \
//     -- --gtest_filter='*AcquireRaceManualOnAarch64*'
TEST_F(CtranGpeKernelSyncTest, AcquireRaceManualOnAarch64) {
  constexpr int kNumWorkers = 8;
  constexpr int kPayloadPerWorker = 64;
  constexpr int kPayloadSize = kNumWorkers * kPayloadPerWorker;
  constexpr int kIters = 50000;

  // Match production allocation: pinned host memory for the sync.
  void* syncPtr = nullptr;
  ASSERT_EQ(
      cudaHostAlloc(&syncPtr, sizeof(GpeKernelSync), cudaHostAllocDefault),
      cudaSuccess);
  GpeKernelSync* sync = new (syncPtr) GpeKernelSync(kNumWorkers);

  // Per-(iter, block) magic value, recomputed in both producer and consumer.
  auto magicFor = [](int iter, int b) {
    return (iter * 13) ^ ((b + 1) * 0x9e3779b1);
  };

  std::vector<int> payload(kPayloadSize, -1);
  std::atomic<int> mismatches{0};
  // Iteration barrier: consumer sets goSignal[b] = iter, producer waits
  // for goSignal[b] >= iter. Uses std::atomic so TSan recognises the
  // release/acquire pair and does NOT flag this part of the test.
  std::vector<std::atomic<int>> goSignal(kNumWorkers);
  for (auto& g : goSignal) {
    g.store(-1, std::memory_order_relaxed);
  }

  // Producer threads: simulate kernel blocks. Wait for the consumer to
  // release goSignal[b] for `iter`, then write the per-block payload and
  // release-store completeFlag[b].
  std::vector<std::thread> producers;
  producers.reserve(kNumWorkers);
  for (int b = 0; b < kNumWorkers; ++b) {
    producers.emplace_back([&, b]() {
      for (int iter = 0; iter < kIters; ++iter) {
        // Acquire the consumer's "go" signal — TSan-clean handshake.
        while (goSignal[b].load(std::memory_order_acquire) < iter) {
          // friendly spin
        }

        // Write our portion of the payload.
        const int magic = magicFor(iter, b);
        const int start = b * kPayloadPerWorker;
        const int end = start + kPayloadPerWorker;
        for (int i = start; i < end; ++i) {
          payload[i] = magic;
        }

        // Mirror the GPU's `st.release.sys` on completeFlag — release fence
        // then volatile store. This is intentionally the SAME pattern used
        // in production (GpeKernelSync::post via wcStoreFence + volatile),
        // so TSan correctly identifies the missing acquire on the consumer
        // side as a data race.
        std::atomic_thread_fence(std::memory_order_release);
        volatile int* completeFlag = &sync->completeFlag[b];
        *completeFlag = iter;
      }
    });
  }

  // Consumer (host): release goSignal then call the production
  // `sync->isComplete(iter)`. After it returns true, read the payload back
  // and verify it matches the expected per-block magic.
  std::thread consumer([&]() {
    for (int iter = 0; iter < kIters; ++iter) {
      // Release goSignal[b] for `iter` so all producers can proceed.
      for (int b = 0; b < kNumWorkers; ++b) {
        goSignal[b].store(iter, std::memory_order_release);
      }

      while (!sync->isComplete(iter)) {
        // friendly spin
      }

      // Verify every block's payload portion. On aarch64 (or under TSan),
      // without the `wcAcquireFence()` in `isComplete()`, this read can
      // see stale data.
      for (int b = 0; b < kNumWorkers; ++b) {
        const int magic = magicFor(iter, b);
        const int start = b * kPayloadPerWorker;
        const int end = start + kPayloadPerWorker;
        for (int i = start; i < end; ++i) {
          if (payload[i] != magic) {
            mismatches.fetch_add(1, std::memory_order_relaxed);
            break; // one count per (iter, block)
          }
        }
      }
    }
  });

  consumer.join();
  for (auto& t : producers) {
    t.join();
  }

  EXPECT_EQ(mismatches.load(), 0)
      << "Acquire race detected: payload was inconsistent at "
      << mismatches.load() << " (iter, block) sites across " << kIters
      << " iterations × " << kNumWorkers << " blocks. "
      << "If running on aarch64 without wcAcquireFence() in "
      << "GpeKernelSync::isComplete(), this is the expected pre-fix failure.";

  ASSERT_EQ(cudaFreeHost(syncPtr), cudaSuccess);
}
