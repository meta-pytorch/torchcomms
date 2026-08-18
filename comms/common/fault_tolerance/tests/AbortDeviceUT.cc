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

// The communicator timeout must stay late-bound. Transports cache one device
// handle for the communicator's lifetime (MultiPeerTransport builds it in its
// constructor), so a handle created before setDefaultTimeout() must still honor
// the new value. Note this asserts the DEADLINE path, not a raw
// getTimeoutMs() read: deviceHandleSeesHostDefaultTimeoutUpdates covers the
// latter and would not catch a stale value cached inside startTimeout().
TEST(AbortDeviceTest, deadlineHonorsDefaultTimeoutSetAfterHandleCreation) {
  Abort abort{/*enabled=*/true};
  // Handle created BEFORE any timeout exists, then again after one is set, to
  // cover both orderings a communicator can produce.
  abort.setDefaultTimeout(std::chrono::milliseconds{60000});
  auto handle = abort.getDeviceHandle();

  auto observedMode = makeDeviceValue<int>();
  auto observedIsAborted = makeDeviceValue<int>();
  ASSERT_NE(observedMode, nullptr);
  ASSERT_NE(observedIsAborted, nullptr);

  // Shorten well below the value present at handle creation. If the handle
  // cached that value, this waits ~60s and the bound below fails.
  abort.setDefaultTimeout(std::chrono::milliseconds{kDeviceTimeoutExpectedMs});

  cudaEvent_t start = nullptr;
  cudaEvent_t end = nullptr;
  ASSERT_EQ(cudaEventCreate(&start), cudaSuccess);
  ASSERT_EQ(cudaEventCreate(&end), cudaSuccess);

  ASSERT_EQ(cudaEventRecord(start, /*stream=*/nullptr), cudaSuccess);
  EXPECT_EQ(
      launchDeviceWaitForTimeout(
          handle,
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

  EXPECT_EQ(
      readDeviceValue(observedMode), static_cast<int>(AbortReason::TIMED_OUT));
  EXPECT_LE(
      elapsedMs,
      static_cast<float>(kDeviceTimeoutExpectedMs) + kDeviceTimeoutAccuracyMs)
      << "deadline used a stale timeout captured at handle creation";
}

// A per-op override beats the communicator default, and clearing it falls back
// to shared state.
TEST(AbortDeviceTest, perOpTimeoutOverridesCommunicatorDefault) {
  Abort abort{/*enabled=*/true};
  abort.setDefaultTimeout(std::chrono::milliseconds{60000});

  auto handle = abort.getDeviceHandle();
  EXPECT_LT(handle.opTimeoutMs(), 0) << "override must default to unset";
  handle.setOpTimeoutMs(kDeviceTimeoutExpectedMs);

  auto observedMode = makeDeviceValue<int>();
  auto observedIsAborted = makeDeviceValue<int>();
  ASSERT_NE(observedMode, nullptr);
  ASSERT_NE(observedIsAborted, nullptr);

  cudaEvent_t start = nullptr;
  cudaEvent_t end = nullptr;
  ASSERT_EQ(cudaEventCreate(&start), cudaSuccess);
  ASSERT_EQ(cudaEventCreate(&end), cudaSuccess);

  ASSERT_EQ(cudaEventRecord(start, /*stream=*/nullptr), cudaSuccess);
  EXPECT_EQ(
      launchDeviceWaitForTimeout(
          handle,
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

  EXPECT_EQ(
      readDeviceValue(observedMode), static_cast<int>(AbortReason::TIMED_OUT));
  EXPECT_GE(
      elapsedMs,
      static_cast<float>(kDeviceTimeoutExpectedMs) - kDeviceTimeoutAccuracyMs);
  EXPECT_LE(
      elapsedMs,
      static_cast<float>(kDeviceTimeoutExpectedMs) + kDeviceTimeoutAccuracyMs)
      << "per-op override did not take precedence over the 60s comm default";
}

TEST(AbortDeviceTest, perOpTimeoutUnsetFallsBackToCommunicatorDefault) {
  Abort abort{/*enabled=*/true};
  abort.setDefaultTimeout(std::chrono::milliseconds{1234});

  auto handle = abort.getDeviceHandle();
  handle.setOpTimeoutMs(4321);
  EXPECT_EQ(handle.opTimeoutMs(), 4321);

  // Negative clears the override; the deadline reverts to shared state.
  handle.setOpTimeoutMs(-1);
  EXPECT_LT(handle.opTimeoutMs(), 0);

  auto observedTimeoutMs = makeDeviceValue<int64_t>();
  ASSERT_NE(observedTimeoutMs, nullptr);
  EXPECT_EQ(
      launchDeviceReadDefaultTimeoutMs(
          handle, observedTimeoutMs.get(), /*stream=*/nullptr),
      cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(readDeviceValue(observedTimeoutMs), 1234);
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

// --- FT_ABORT_* macros ----------------------------------------------------
//
// Each case bounds its loop, so a macro that fails to terminate reports the
// bound rather than hanging.

constexpr int kMacroLoopBound = 1000;

// The timeout case paces itself at roughly a microsecond per iteration, so the
// bound has to outlast the deadline by a wide margin for "ended early" to mean
// the deadline ended it. Reaching the bound caps the kernel at about a second.
constexpr auto kMacroTimeoutMs = std::chrono::milliseconds{20};
constexpr int kMacroTimeoutLoopBound = 1'000'000;
// A correctly armed 20 ms deadline takes thousands of iterations to reach, so
// anything this small means the deadline was already expired on entry.
constexpr int kMacroTimeoutMinIterations = 100;

TEST(AbortMacrosTest, BreakLeavesLoopWhenAborted) {
  Abort abort{/*enabled=*/true};
  abort.setAbort();

  auto iterations = makeDeviceValue<int>();
  ASSERT_NE(iterations, nullptr);
  EXPECT_EQ(
      launchMacroBreakLoop(
          abort.getDeviceHandle(),
          iterations.get(),
          kMacroLoopBound,
          /*stream=*/nullptr),
      cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(readDeviceValue(iterations), 1)
      << "FT_ABORT_BREAK must leave the loop on its first check";
}

TEST(AbortMacrosTest, BreakRunsToCompletionWhenNotAborted) {
  Abort abort{/*enabled=*/true};

  auto iterations = makeDeviceValue<int>();
  ASSERT_NE(iterations, nullptr);
  EXPECT_EQ(
      launchMacroBreakLoop(
          abort.getDeviceHandle(),
          iterations.get(),
          kMacroLoopBound,
          /*stream=*/nullptr),
      cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(readDeviceValue(iterations), kMacroLoopBound)
      << "a healthy handle must not terminate the loop";
}

TEST(AbortMacrosTest, BreakIsANoOpForDisabledHandle) {
  AbortDevice disabled;

  auto iterations = makeDeviceValue<int>();
  ASSERT_NE(iterations, nullptr);
  EXPECT_EQ(
      launchMacroBreakLoop(
          disabled, iterations.get(), kMacroLoopBound, /*stream=*/nullptr),
      cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(readDeviceValue(iterations), kMacroLoopBound);
}

// FT_ABORT_BREAK expands to an `if`. If it does not consume a trailing `else`,
// the caller's `else` binds to the macro, and the damage is silent: `fallback`
// runs precisely when the guard held and nothing had aborted.
//
// The healthy case is the one that discriminates. With a pre-aborted handle the
// macro's check is true on the first iteration, so the loop breaks and
// `fallback` stays 0 under either expansion -- that case only pins the break
// itself. Reaching the caller's `else` at all requires the check to be false,
// which is why both cases are here.
TEST(AbortMacrosTest, BreakDoesNotCaptureACallerElseWhenHealthy) {
  Abort abort{/*enabled=*/true};

  auto iterations = makeDeviceValue<int>();
  auto fallback = makeDeviceValue<int>();
  ASSERT_NE(iterations, nullptr);
  ASSERT_NE(fallback, nullptr);
  EXPECT_EQ(
      launchMacroBreakInIfElse(
          abort.getDeviceHandle(),
          iterations.get(),
          fallback.get(),
          kMacroLoopBound,
          /*stream=*/nullptr),
      cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(readDeviceValue(iterations), kMacroLoopBound)
      << "a healthy handle must not terminate the caller's loop";
  EXPECT_EQ(readDeviceValue(fallback), 0)
      << "the caller's else belongs to the caller's if, not to the macro; a "
         "naked-if expansion sets this to 1";
}

TEST(AbortMacrosTest, BreakDoesNotCaptureACallerElseWhenAborted) {
  Abort abort{/*enabled=*/true};
  abort.setAbort();

  auto iterations = makeDeviceValue<int>();
  auto fallback = makeDeviceValue<int>();
  ASSERT_NE(iterations, nullptr);
  ASSERT_NE(fallback, nullptr);
  EXPECT_EQ(
      launchMacroBreakInIfElse(
          abort.getDeviceHandle(),
          iterations.get(),
          fallback.get(),
          kMacroLoopBound,
          /*stream=*/nullptr),
      cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(readDeviceValue(iterations), 1)
      << "the break must still leave the caller's loop on the first check";
  EXPECT_EQ(readDeviceValue(fallback), 0);
}

TEST(AbortMacrosTest, CheckReportsStopToTheCaller) {
  Abort abort{/*enabled=*/true};
  abort.setAbort();

  auto iterations = makeDeviceValue<int>();
  auto stop = makeDeviceValue<int>();
  ASSERT_NE(iterations, nullptr);
  ASSERT_NE(stop, nullptr);
  EXPECT_EQ(
      launchMacroCheckLoop(
          abort.getDeviceHandle(),
          iterations.get(),
          stop.get(),
          kMacroLoopBound,
          /*stream=*/nullptr),
      cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(readDeviceValue(stop), 1)
      << "FT_ABORT_CHECK must report the terminal result";
  EXPECT_EQ(readDeviceValue(iterations), 1);
}

TEST(AbortMacrosTest, CheckReportsContinueWhenHealthy) {
  Abort abort{/*enabled=*/true};

  auto iterations = makeDeviceValue<int>();
  auto stop = makeDeviceValue<int>();
  ASSERT_NE(iterations, nullptr);
  ASSERT_NE(stop, nullptr);
  EXPECT_EQ(
      launchMacroCheckLoop(
          abort.getDeviceHandle(),
          iterations.get(),
          stop.get(),
          kMacroLoopBound,
          /*stream=*/nullptr),
      cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(readDeviceValue(stop), 0);
  EXPECT_EQ(readDeviceValue(iterations), kMacroLoopBound);
}

TEST(AbortMacrosTest, ReturnYieldsTheCallerSuppliedValue) {
  Abort abort{/*enabled=*/true};
  abort.setAbort();

  auto observed = makeDeviceValue<int>(0);
  ASSERT_NE(observed, nullptr);
  EXPECT_EQ(
      launchMacroReturnValue(
          abort.getDeviceHandle(), observed.get(), /*stream=*/nullptr),
      cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(readDeviceValue(observed), -1)
      << "FT_ABORT_RETURN must return the value the caller supplied";
}

TEST(AbortMacrosTest, ReturnFallsThroughWhenHealthy) {
  Abort abort{/*enabled=*/true};

  auto observed = makeDeviceValue<int>(0);
  ASSERT_NE(observed, nullptr);
  EXPECT_EQ(
      launchMacroReturnValue(
          abort.getDeviceHandle(), observed.get(), /*stream=*/nullptr),
      cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(readDeviceValue(observed), 7);
}

TEST(AbortMacrosTest, TimeoutTerminatesTheLoop) {
  Abort abort{/*enabled=*/true};
  abort.setDefaultTimeout(kMacroTimeoutMs);
  auto handle = abort.getDeviceHandle();

  auto iterations = makeDeviceValue<int>();
  ASSERT_NE(iterations, nullptr);
  // The kernel arms the deadline itself: startTimeout() is device-only and
  // reads the device clock, so arming it here would make the first check see
  // an already-expired deadline and the loop would end for the wrong reason.
  EXPECT_EQ(
      launchMacroTimeoutLoop(
          handle, iterations.get(), kMacroTimeoutLoopBound, /*stream=*/nullptr),
      cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // The deadline is what ends it, so the exact iteration is timing dependent.
  // Both bounds matter: reaching kMacroTimeoutLoopBound means the macro never
  // observed the timeout, while stopping in the first few iterations means the
  // deadline was already expired when the kernel started -- which is what an
  // accidental host-side startTimeout() produces, and it would otherwise pass
  // every assertion here.
  const int observed = readDeviceValue(iterations);
  EXPECT_LT(observed, kMacroTimeoutLoopBound);
  EXPECT_GT(observed, kMacroTimeoutMinIterations);
  EXPECT_TRUE(abort.isTimedOut());
}

} // namespace comms::fault_tolerance::testing
