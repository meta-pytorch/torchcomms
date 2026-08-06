// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <thread>

#include <gtest/gtest.h>

#include "comms/common/fault_tolerance/Abort.h"
#include "comms/common/fault_tolerance/tests/AbortDeviceTest.cuh"

namespace comms::fault_tolerance::testing {
namespace {

constexpr int kDeviceTimeoutPollIterations = 10000000;
constexpr int kDeviceTimeoutExpectedMs = 1000;
constexpr float kDeviceTimeoutAccuracyMs = 100.0F;

struct CudaFreeDeleter {
  template <typename T>
  void operator()(T* ptr) const {
    if (ptr != nullptr) {
      (void)cudaFree(ptr);
    }
  }
};

template <typename T>
using DeviceValue = std::unique_ptr<T, CudaFreeDeleter>;

template <typename T>
DeviceValue<T> makeDeviceValue(T value = 0) {
  T* ptr = nullptr;
  EXPECT_EQ(cudaMalloc(&ptr, sizeof(T)), cudaSuccess);
  if (ptr == nullptr) {
    return nullptr;
  }

  EXPECT_EQ(
      cudaMemcpy(ptr, &value, sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
  return DeviceValue<T>{ptr};
}

template <typename T>
T readDeviceValue(const DeviceValue<T>& ptr) {
  T value = 0;
  EXPECT_EQ(
      cudaMemcpy(&value, ptr.get(), sizeof(T), cudaMemcpyDeviceToHost),
      cudaSuccess);
  return value;
}

void destroyEvent(cudaEvent_t event) {
  if (event != nullptr) {
    EXPECT_EQ(cudaEventDestroy(event), cudaSuccess);
  }
}

} // namespace

TEST(AbortDeviceTest, hostProducerHostConsumer) {
  Abort abort{/*enabled=*/true};

  EXPECT_FALSE(abort.isAborted());

  abort.setAbort();

  EXPECT_TRUE(abort.isAborted());
}

TEST(AbortDeviceTest, hostProducerDeviceConsumer) {
  Abort abort{/*enabled=*/true};
  auto observed = makeDeviceValue<int>();
  auto observedMode = makeDeviceValue<int>();
  ASSERT_NE(observed, nullptr);
  ASSERT_NE(observedMode, nullptr);

  abort.setAbort();

  EXPECT_EQ(
      launchDeviceReadAbort(
          abort.getDeviceHandle(),
          observed.get(),
          observedMode.get(),
          /*stream=*/nullptr),
      cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(readDeviceValue(observed), 1);
  EXPECT_EQ(
      readDeviceValue(observedMode), static_cast<int>(AbortReason::ABORTED));
}

TEST(AbortDeviceTest, checkExpiredSeesHostAbort) {
  Abort abort{/*enabled=*/true};
  auto observedCheckExpired = makeDeviceValue<int>();
  auto observedReason = makeDeviceValue<int>();
  ASSERT_NE(observedCheckExpired, nullptr);
  ASSERT_NE(observedReason, nullptr);

  abort.setAbort();

  EXPECT_EQ(
      launchDeviceReadCheckExpired(
          abort.getDeviceHandle(),
          observedCheckExpired.get(),
          observedReason.get(),
          /*stream=*/nullptr),
      cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(readDeviceValue(observedCheckExpired), 1);
  EXPECT_EQ(
      readDeviceValue(observedReason), static_cast<int>(AbortReason::ABORTED));
}

TEST(AbortDeviceTest, deviceCheckContinuesBeforeAbort) {
  Abort abort{/*enabled=*/true};
  auto observedCheckResult = makeDeviceValue<int>();
  ASSERT_NE(observedCheckResult, nullptr);

  EXPECT_EQ(
      launchDeviceReadCheckResult(
          abort.getDeviceHandle(), observedCheckResult.get(), nullptr),
      cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(
      readDeviceValue(observedCheckResult),
      static_cast<int>(AbortCheckResult::CONTINUE));
}

TEST(AbortDeviceTest, deviceCheckDefaultsToSkipOnAbort) {
  Abort abort{/*enabled=*/true};
  auto observedCheckResult = makeDeviceValue<int>();
  ASSERT_NE(observedCheckResult, nullptr);

  abort.setAbort();

  EXPECT_EQ(
      launchDeviceReadCheckResult(
          abort.getDeviceHandle(), observedCheckResult.get(), nullptr),
      cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(
      readDeviceValue(observedCheckResult),
      static_cast<int>(AbortCheckResult::SKIP));
}

TEST(AbortDeviceTest, deviceCheckReturnsTrapWhenConfigured) {
  Abort abort{/*enabled=*/true, AbortBehavior::TRAP};
  auto observedCheckResult = makeDeviceValue<int>();
  ASSERT_NE(observedCheckResult, nullptr);

  abort.setAbort();

  EXPECT_EQ(
      launchDeviceReadCheckResult(
          abort.getDeviceHandle(), observedCheckResult.get(), nullptr),
      cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(
      readDeviceValue(observedCheckResult),
      static_cast<int>(AbortCheckResult::TRAP));
}

TEST(AbortDeviceTest, deviceProducerHostConsumer) {
  Abort abort{/*enabled=*/true};

  EXPECT_EQ(
      launchDeviceSetAbort(
          abort.getDeviceHandle(), AbortReason::ABORTED, /*stream=*/nullptr),
      cudaSuccess);

  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!abort.isAborted() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }

  EXPECT_TRUE(abort.isAborted());
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

TEST(AbortDeviceTest, deviceProducerDeviceConsumer) {
  Abort abort{/*enabled=*/true};
  auto observed = makeDeviceValue<int>();
  auto observedMode = makeDeviceValue<int>();
  ASSERT_NE(observed, nullptr);
  ASSERT_NE(observedMode, nullptr);

  EXPECT_EQ(
      launchDeviceSetAbort(
          abort.getDeviceHandle(), AbortReason::ABORTED, /*stream=*/nullptr),
      cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(
      launchDeviceReadAbort(
          abort.getDeviceHandle(),
          observed.get(),
          observedMode.get(),
          /*stream=*/nullptr),
      cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(readDeviceValue(observed), 1);
  EXPECT_EQ(
      readDeviceValue(observedMode), static_cast<int>(AbortReason::ABORTED));
}

TEST(AbortDeviceTest, hostDefaultTimeoutDeviceConsumer) {
  Abort abort{/*enabled=*/true};
  auto observedTimeoutMs = makeDeviceValue<int64_t>();
  ASSERT_NE(observedTimeoutMs, nullptr);

  abort.setDefaultTimeout(std::chrono::milliseconds{1234});

  EXPECT_EQ(
      launchDeviceReadDefaultTimeoutMs(
          abort.getDeviceHandle(), observedTimeoutMs.get(), /*stream=*/nullptr),
      cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(readDeviceValue(observedTimeoutMs), 1234);
}

TEST(AbortDeviceTest, deviceHandleSeesHostDefaultTimeoutUpdates) {
  Abort abort{/*enabled=*/true};
  auto handle = abort.getDeviceHandle();
  auto observedTimeoutMs = makeDeviceValue<int64_t>();
  ASSERT_NE(observedTimeoutMs, nullptr);

  abort.setDefaultTimeout(std::chrono::milliseconds{4321});

  EXPECT_EQ(
      launchDeviceReadDefaultTimeoutMs(
          handle, observedTimeoutMs.get(), /*stream=*/nullptr),
      cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(readDeviceValue(observedTimeoutMs), 4321);
}

TEST(AbortDeviceTest, timedOutModeIsAbortedOnDevice) {
  Abort abort{/*enabled=*/true};
  auto observedIsAborted = makeDeviceValue<int>();
  auto observedReason = makeDeviceValue<int>();
  ASSERT_NE(observedIsAborted, nullptr);
  ASSERT_NE(observedReason, nullptr);

  abort.startTimeout(std::chrono::milliseconds{0});
  ASSERT_TRUE(abort.isAborted());

  EXPECT_EQ(
      launchDeviceReadAbortPredicate(
          abort.getDeviceHandle(),
          observedIsAborted.get(),
          observedReason.get(),
          /*stream=*/nullptr),
      cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(readDeviceValue(observedIsAborted), 1);
  EXPECT_EQ(
      readDeviceValue(observedReason),
      static_cast<int>(AbortReason::TIMED_OUT));
}

TEST(AbortDeviceTest, deviceTimeoutProducerHostAndDeviceConsumer) {
  Abort abort{/*enabled=*/true};
  auto observedMode = makeDeviceValue<int>();
  auto observedIsAborted = makeDeviceValue<int>();
  ASSERT_NE(observedMode, nullptr);
  ASSERT_NE(observedIsAborted, nullptr);

  abort.setDefaultTimeout(std::chrono::milliseconds{1});

  EXPECT_EQ(
      launchDeviceWaitForTimeout(
          abort.getDeviceHandle(),
          observedMode.get(),
          observedIsAborted.get(),
          kDeviceTimeoutPollIterations,
          /*stream=*/nullptr),
      cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(
      readDeviceValue(observedMode), static_cast<int>(AbortReason::TIMED_OUT));
  EXPECT_EQ(readDeviceValue(observedIsAborted), 1);
  EXPECT_TRUE(abort.isAborted());
  EXPECT_TRUE(abort.isTimedOut());
}

TEST(AbortDeviceTest, hostAbortWinsOverDeviceTimeout) {
  Abort abort{/*enabled=*/true};
  auto observedMode = makeDeviceValue<int>();
  auto observedIsAborted = makeDeviceValue<int>();
  ASSERT_NE(observedMode, nullptr);
  ASSERT_NE(observedIsAborted, nullptr);

  abort.setAbort(AbortReason::ABORTED);
  abort.setDefaultTimeout(std::chrono::milliseconds{1});

  EXPECT_EQ(
      launchDeviceWaitForTimeout(
          abort.getDeviceHandle(),
          observedMode.get(),
          observedIsAborted.get(),
          kDeviceTimeoutPollIterations,
          /*stream=*/nullptr),
      cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(
      readDeviceValue(observedMode), static_cast<int>(AbortReason::ABORTED));
  EXPECT_EQ(readDeviceValue(observedIsAborted), 1);
  EXPECT_TRUE(abort.isAborted());
  EXPECT_FALSE(abort.isTimedOut());
}

TEST(AbortDeviceTest, deviceAbortWinsOverHostTimeout) {
  Abort abort{/*enabled=*/true};

  EXPECT_EQ(
      launchDeviceSetAbort(
          abort.getDeviceHandle(), AbortReason::ABORTED, /*stream=*/nullptr),
      cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  abort.startTimeout(std::chrono::milliseconds{0});

  EXPECT_TRUE(abort.isAborted());
  EXPECT_FALSE(abort.isTimedOut());
}

TEST(AbortDeviceTest, hostTimeoutWinsOverDeviceAbort) {
  Abort abort{/*enabled=*/true};

  abort.startTimeout(std::chrono::milliseconds{0});
  ASSERT_TRUE(abort.isTimedOut());

  EXPECT_EQ(
      launchDeviceSetAbort(
          abort.getDeviceHandle(), AbortReason::ABORTED, /*stream=*/nullptr),
      cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_TRUE(abort.isAborted());
  EXPECT_TRUE(abort.isTimedOut());
}

TEST(AbortDeviceTest, deviceTimeoutWinsOverHostAbort) {
  Abort abort{/*enabled=*/true};
  auto observedMode = makeDeviceValue<int>();
  auto observedIsAborted = makeDeviceValue<int>();
  ASSERT_NE(observedMode, nullptr);
  ASSERT_NE(observedIsAborted, nullptr);

  abort.setDefaultTimeout(std::chrono::milliseconds{1});

  EXPECT_EQ(
      launchDeviceWaitForTimeout(
          abort.getDeviceHandle(),
          observedMode.get(),
          observedIsAborted.get(),
          kDeviceTimeoutPollIterations,
          /*stream=*/nullptr),
      cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  abort.setAbort(AbortReason::ABORTED);

  EXPECT_EQ(
      readDeviceValue(observedMode), static_cast<int>(AbortReason::TIMED_OUT));
  EXPECT_EQ(readDeviceValue(observedIsAborted), 1);
  EXPECT_TRUE(abort.isAborted());
  EXPECT_TRUE(abort.isTimedOut());
}

TEST(AbortDeviceTest, startAliasAndCheckExpiredRecordTimeout) {
  Abort abort{/*enabled=*/true};
  auto observedMode = makeDeviceValue<int>();
  auto observedCheckExpired = makeDeviceValue<int>();
  ASSERT_NE(observedMode, nullptr);
  ASSERT_NE(observedCheckExpired, nullptr);

  abort.setDefaultTimeout(std::chrono::milliseconds{1});

  EXPECT_EQ(
      launchDeviceWaitForTimeoutStartAlias(
          abort.getDeviceHandle(),
          observedMode.get(),
          observedCheckExpired.get(),
          kDeviceTimeoutPollIterations,
          /*stream=*/nullptr),
      cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(readDeviceValue(observedCheckExpired), 1);
  EXPECT_EQ(
      readDeviceValue(observedMode), static_cast<int>(AbortReason::TIMED_OUT));
  EXPECT_TRUE(abort.isTimedOut());
}

TEST(AbortDeviceTest, deviceTimeoutCanBeCancelledAndRestarted) {
  Abort abort{/*enabled=*/true};
  auto observedAfterCancel = makeDeviceValue<int>();
  auto observedMode = makeDeviceValue<int>();
  ASSERT_NE(observedAfterCancel, nullptr);
  ASSERT_NE(observedMode, nullptr);

  abort.setDefaultTimeout(std::chrono::milliseconds{1});

  EXPECT_EQ(
      launchDeviceCancelAndRestartTimeout(
          abort.getDeviceHandle(),
          observedAfterCancel.get(),
          observedMode.get(),
          kDeviceTimeoutPollIterations,
          /*stream=*/nullptr),
      cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(readDeviceValue(observedAfterCancel), 0);
  EXPECT_EQ(
      readDeviceValue(observedMode), static_cast<int>(AbortReason::TIMED_OUT));
  EXPECT_TRUE(abort.isAborted());
  EXPECT_TRUE(abort.isTimedOut());
}

TEST(AbortDeviceTest, deviceTimeoutAccuracyMeasuredWithCudaEvents) {
  Abort abort{/*enabled=*/true};
  auto observedMode = makeDeviceValue<int>();
  auto observedIsAborted = makeDeviceValue<int>();
  ASSERT_NE(observedMode, nullptr);
  ASSERT_NE(observedIsAborted, nullptr);

  abort.setDefaultTimeout(std::chrono::milliseconds{kDeviceTimeoutExpectedMs});

  cudaEvent_t start = nullptr;
  cudaEvent_t end = nullptr;
  ASSERT_EQ(cudaEventCreate(&start), cudaSuccess);
  ASSERT_EQ(cudaEventCreate(&end), cudaSuccess);

  ASSERT_EQ(cudaEventRecord(start, /*stream=*/nullptr), cudaSuccess);
  EXPECT_EQ(
      launchDeviceWaitForTimeout(
          abort.getDeviceHandle(),
          observedMode.get(),
          observedIsAborted.get(),
          kDeviceTimeoutPollIterations,
          /*stream=*/nullptr),
      cudaSuccess);
  ASSERT_EQ(cudaEventRecord(end, /*stream=*/nullptr), cudaSuccess);
  ASSERT_EQ(cudaEventSynchronize(end), cudaSuccess);

  float elapsedMs = 0.0F;
  ASSERT_EQ(cudaEventElapsedTime(&elapsedMs, start, end), cudaSuccess);
  destroyEvent(end);
  destroyEvent(start);

  std::fprintf(
      stderr,
      "AbortDevice timeout accuracy expected_ms=%d observed_ms=%.3f tolerance_ms=%.3f\n",
      kDeviceTimeoutExpectedMs,
      elapsedMs,
      kDeviceTimeoutAccuracyMs);
  EXPECT_EQ(
      readDeviceValue(observedMode), static_cast<int>(AbortReason::TIMED_OUT));
  EXPECT_EQ(readDeviceValue(observedIsAborted), 1);
  EXPECT_GE(
      elapsedMs,
      static_cast<float>(kDeviceTimeoutExpectedMs) - kDeviceTimeoutAccuracyMs);
  EXPECT_LE(
      elapsedMs,
      static_cast<float>(kDeviceTimeoutExpectedMs) + kDeviceTimeoutAccuracyMs);
}

TEST(AbortDeviceTest, disabledAbortDeviceHandleIsNoop) {
  auto abort = createAbort(/*enabled=*/false);
  auto observed = makeDeviceValue<int>();
  auto observedMode = makeDeviceValue<int>();
  auto observedTimeoutMs = makeDeviceValue<int64_t>();
  ASSERT_NE(observed, nullptr);
  ASSERT_NE(observedMode, nullptr);
  ASSERT_NE(observedTimeoutMs, nullptr);

  auto handle = abort->getDeviceHandle();

  EXPECT_EQ(
      launchDeviceReadAbort(
          handle, observed.get(), observedMode.get(), /*stream=*/nullptr),
      cudaSuccess);
  EXPECT_EQ(
      launchDeviceReadDefaultTimeoutMs(
          handle, observedTimeoutMs.get(), /*stream=*/nullptr),
      cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(readDeviceValue(observed), 0);
  EXPECT_EQ(readDeviceValue(observedMode), static_cast<int>(AbortReason::NONE));
  EXPECT_EQ(readDeviceValue(observedTimeoutMs), -1);
}

TEST(AbortDeviceTest, defaultConstructedHandleIsDisabledNoop) {
  AbortDevice handle;
  auto observed = makeDeviceValue<int>();
  auto observedMode = makeDeviceValue<int>();
  ASSERT_NE(observed, nullptr);
  ASSERT_NE(observedMode, nullptr);

  EXPECT_FALSE(handle.isEnabled());
  EXPECT_EQ(
      launchDeviceReadAbort(
          handle, observed.get(), observedMode.get(), /*stream=*/nullptr),
      cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(readDeviceValue(observed), 0);
  EXPECT_EQ(readDeviceValue(observedMode), static_cast<int>(AbortReason::NONE));
}

} // namespace comms::fault_tolerance::testing
