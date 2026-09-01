// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>

#include <gmock/gmock.h>
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

// The greppable contract, spelled out rather than taken from the macro. A test
// that builds its expectation from `FT_ABORT_FIRST_WRITER_` would follow the
// macro anywhere it went, including somewhere no existing log search would find
// it. This is the string oncall greps for.
constexpr const char* kFirstWriterMarker = "COMMS FT ABORT FIRST WRITER: ";

// Runs `launch`, drains the device printf FIFO, and returns everything the
// process wrote to stdout meanwhile.
//
// The synchronize has to happen *inside* the capture window. Device `printf`
// appends to a per-context FIFO that the runtime drains only on kernel
// completion, synchronization, or context destruction, so reading the capture
// before syncing returns an empty string and the test silently proves nothing.
template <typename Launch>
std::string captureDeviceStdout(Launch&& launch) {
  ::testing::internal::CaptureStdout();
  const cudaError_t launched = launch();
  const cudaError_t synced = cudaDeviceSynchronize();
  std::string captured = ::testing::internal::GetCapturedStdout();
  // Asserted after the capture closes: a failure message emitted inside the
  // window would be swallowed by the capture instead of reported.
  EXPECT_EQ(launched, cudaSuccess);
  EXPECT_EQ(synced, cudaSuccess);
  return captured;
}

size_t countSubstr(const std::string& haystack, const std::string& needle) {
  size_t count = 0;
  for (size_t pos = haystack.find(needle); pos != std::string::npos;
       pos = haystack.find(needle, pos + needle.size())) {
    ++count;
  }
  return count;
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

TEST(AbortDeviceTest, deviceObservesDetailedHostReasonWithoutContext) {
  Abort abort{/*enabled=*/true};
  auto observed = makeDeviceValue<int>();
  auto observedReason = makeDeviceValue<int>();
  ASSERT_NE(observed, nullptr);
  ASSERT_NE(observedReason, nullptr);

  abort.setAbort(AbortReason::BOOTSTRAP_POLL, "socket health poll");

  EXPECT_EQ(
      launchDeviceReadAbort(
          abort.getDeviceHandle(),
          observed.get(),
          observedReason.get(),
          /*stream=*/nullptr),
      cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(readDeviceValue(observed), 1);
  EXPECT_EQ(
      readDeviceValue(observedReason),
      static_cast<int>(AbortReason::BOOTSTRAP_POLL));
  EXPECT_EQ(
      abort.getAbortInfo(),
      (AbortInfo{
          .reason = AbortReason::BOOTSTRAP_POLL,
          .context = "socket health poll",
      }));
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

TEST(AbortDeviceTest, deviceProducerSupportsDetailedTerminalReasons) {
  for (const auto reason : {
           AbortReason::BOOTSTRAP_POLL,
           AbortReason::NETWORK_ERROR,
           AbortReason::INTERNAL_ERROR,
           AbortReason::IBRC_PROXY_TIMEOUT,
       }) {
    Abort abort{/*enabled=*/true};

    EXPECT_EQ(
        launchDeviceSetAbort(
            abort.getDeviceHandle(), reason, /*stream=*/nullptr),
        cudaSuccess);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto info = abort.getAbortInfo();
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->reason, reason);
    EXPECT_TRUE(info->context.empty());
  }
}

TEST(AbortDeviceTest, deviceContextLogsOnlyForReasonCasWinner) {
  Abort abort{/*enabled=*/true};
  auto firstWon = makeDeviceValue<int>();
  auto secondWon = makeDeviceValue<int>();
  ASSERT_NE(firstWon, nullptr);
  ASSERT_NE(secondWon, nullptr);

  const std::string firstOut = captureDeviceStdout([&] {
    return launchDeviceSetAbortWithContext(
        abort.getDeviceHandle(),
        AbortReason::NETWORK_ERROR,
        /*useContext=*/true,
        firstWon.get(),
        /*stream=*/nullptr);
  });

  const std::string secondOut = captureDeviceStdout([&] {
    return launchDeviceSetAbortWithContext(
        abort.getDeviceHandle(),
        AbortReason::INTERNAL_ERROR,
        /*useContext=*/true,
        secondWon.get(),
        /*stream=*/nullptr);
  });

  EXPECT_EQ(readDeviceValue(firstWon), 1);
  EXPECT_EQ(readDeviceValue(secondWon), 0);
  EXPECT_EQ(
      abort.getAbortInfo(),
      (AbortInfo{
          .reason = AbortReason::NETWORK_ERROR,
          .context = "",
      }));

  // The winner's line, in full. Asserting the rendered text rather than only
  // the CAS result is the point: the boolean is unchanged if the printf is
  // deleted or its arguments are wrong.
  EXPECT_THAT(
      firstOut,
      ::testing::HasSubstr(
          std::string{kFirstWriterMarker} + "device reason=" +
          std::string{abortReasonToString(AbortReason::NETWORK_ERROR)} +
          " context=AbortDeviceTest callsite"))
      << "captured: " << firstOut;

  // The loser is silent, and silence is the property that keeps one aborted
  // communicator from producing one line per observing thread.
  EXPECT_EQ(countSubstr(secondOut, kFirstWriterMarker), 0U)
      << "captured: " << secondOut;
  EXPECT_EQ(countSubstr(firstOut + secondOut, kFirstWriterMarker), 1U);
}

TEST(AbortDeviceTest, deviceNullContextCanLogForReasonCasWinner) {
  Abort abort{/*enabled=*/true};
  auto won = makeDeviceValue<int>();
  ASSERT_NE(won, nullptr);

  const std::string out = captureDeviceStdout([&] {
    return launchDeviceSetAbortWithContext(
        abort.getDeviceHandle(),
        AbortReason::INTERNAL_ERROR,
        /*useContext=*/false,
        won.get(),
        /*stream=*/nullptr);
  });

  EXPECT_EQ(readDeviceValue(won), 1);
  EXPECT_EQ(
      abort.getAbortInfo(),
      (AbortInfo{
          .reason = AbortReason::INTERNAL_ERROR,
          .context = "",
      }));

  // The log belongs to the CAS win, not to whether a diagnostic string was
  // supplied. This is the regression guard for the `context != nullptr` guard
  // that used to sit on the printf: with it restored, the line disappears
  // entirely and this fails.
  EXPECT_THAT(
      out,
      ::testing::HasSubstr(
          std::string{kFirstWriterMarker} + "device reason=" +
          std::string{abortReasonToString(AbortReason::INTERNAL_ERROR)} +
          " context=\n"))
      << "captured: " << out;
}

// `AbortFlag` is the other device writer of the shared reason -- the
// poll-state-free handle the IBRC transport keeps in device memory, and the one
// its proxy watchdogs abort through. It must produce the same first-writer line
// as `AbortDevice`, or a watchdog abort leaves no greppable origin at all.
TEST(AbortDeviceTest, flagSetAbortEmitsFirstWriterMarker) {
  Abort abort{/*enabled=*/true};
  auto won = makeDeviceValue<int>();
  ASSERT_NE(won, nullptr);

  const std::string out = captureDeviceStdout([&] {
    return launchFlagSetAbortWithContext(
        abort.getDeviceHandle(),
        AbortReason::IBRC_PROXY_TIMEOUT,
        /*useContext=*/true,
        won.get(),
        /*stream=*/nullptr);
  });

  EXPECT_EQ(readDeviceValue(won), 1);
  EXPECT_EQ(abort.reason(), AbortReason::IBRC_PROXY_TIMEOUT);

  // Including the context. The IBRC watchdogs already pass one naming which
  // watchdog fired; before this it was accepted and dropped.
  EXPECT_THAT(
      out,
      ::testing::HasSubstr(
          std::string{kFirstWriterMarker} + "device reason=" +
          std::string{abortReasonToString(AbortReason::IBRC_PROXY_TIMEOUT)} +
          " context=AbortFlagTest callsite"))
      << "captured: " << out;
}

TEST(AbortDeviceTest, flagSetAbortLoserIsSilent) {
  Abort abort{/*enabled=*/true};
  auto won = makeDeviceValue<int>();
  ASSERT_NE(won, nullptr);

  // The host takes the reason first, so the flag's CAS loses.
  EXPECT_TRUE(abort.setAbort(AbortReason::ABORTED, "host got there first"));

  const std::string out = captureDeviceStdout([&] {
    return launchFlagSetAbortWithContext(
        abort.getDeviceHandle(),
        AbortReason::NETWORK_ERROR,
        /*useContext=*/true,
        won.get(),
        /*stream=*/nullptr);
  });

  EXPECT_EQ(readDeviceValue(won), 0);
  EXPECT_EQ(abort.reason(), AbortReason::ABORTED);
  EXPECT_EQ(countSubstr(out, kFirstWriterMarker), 0U) << "captured: " << out;
}

TEST(AbortDeviceTest, deviceWinnerDoesNotExposeLosingHostContext) {
  Abort abort{/*enabled=*/true};

  EXPECT_EQ(
      launchDeviceSetAbort(
          abort.getDeviceHandle(),
          AbortReason::INTERNAL_ERROR,
          /*stream=*/nullptr),
      cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  abort.setAbort(AbortReason::NETWORK_ERROR, "losing host context");

  EXPECT_EQ(
      abort.getAbortInfo(),
      (AbortInfo{
          .reason = AbortReason::INTERNAL_ERROR,
          .context = "",
      }));
}

TEST(AbortDeviceTest, hostWinnerPreservesContextAgainstDeviceAbort) {
  Abort abort{/*enabled=*/true};
  abort.setAbort(AbortReason::NETWORK_ERROR, "winning host context");

  EXPECT_EQ(
      launchDeviceSetAbort(
          abort.getDeviceHandle(),
          AbortReason::INTERNAL_ERROR,
          /*stream=*/nullptr),
      cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(
      abort.getAbortInfo(),
      (AbortInfo{
          .reason = AbortReason::NETWORK_ERROR,
          .context = "winning host context",
      }));
}

TEST(AbortDeviceTest, hostDeviceRaceNeverMismatchesContext) {
  constexpr int kIterations = 100;
  for (int i = 0; i < kIterations; ++i) {
    Abort abort{/*enabled=*/true};

    EXPECT_EQ(
        launchDeviceSetAbort(
            abort.getDeviceHandle(),
            AbortReason::INTERNAL_ERROR,
            /*stream=*/nullptr),
        cudaSuccess);
    abort.setAbort(AbortReason::NETWORK_ERROR, "host context");
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto info = abort.getAbortInfo();
    ASSERT_TRUE(info.has_value());
    if (info->reason == AbortReason::NETWORK_ERROR) {
      EXPECT_EQ(info->context, "host context");
    } else {
      EXPECT_EQ(info->reason, AbortReason::INTERNAL_ERROR);
      EXPECT_TRUE(info->context.empty());
    }
  }
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

// The abort context log reports `timeout_ms` and `elapsed_ms` as arithmetic
// over the arm-site clock state rather than as stored values. This checks that
// derivation against the timeout the caller actually asked for -- a log line
// that silently reports the wrong deadline is worse than one that reports none.
TEST(AbortDeviceTest, armedClockStateRecoversTheRequestedTimeout) {
  constexpr int64_t kRequestedTimeoutMs = 2500;
  constexpr uint64_t kOpId = 987654321;

  Abort abort{/*enabled=*/true};
  abort.setDefaultTimeout(std::chrono::milliseconds{kRequestedTimeoutMs});

  auto handle = abort.getDeviceHandle();
  handle.setOpId(kOpId);

  auto startCycles = makeDeviceValue<unsigned long long>();
  auto deadlineCycles = makeDeviceValue<unsigned long long>();
  auto cyclesPerMs = makeDeviceValue<unsigned long long>();
  auto opId = makeDeviceValue<unsigned long long>();
  ASSERT_NE(startCycles, nullptr);
  ASSERT_NE(deadlineCycles, nullptr);
  ASSERT_NE(cyclesPerMs, nullptr);
  ASSERT_NE(opId, nullptr);

  EXPECT_EQ(
      launchDeviceReadArmedClockState(
          handle,
          startCycles.get(),
          deadlineCycles.get(),
          cyclesPerMs.get(),
          opId.get(),
          /*stream=*/nullptr),
      cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const auto observedStart = readDeviceValue(startCycles);
  const auto observedDeadline = readDeviceValue(deadlineCycles);
  const auto observedCyclesPerMs = readDeviceValue(cyclesPerMs);

  EXPECT_EQ(readDeviceValue(opId), kOpId) << "op number must survive the copy";
  EXPECT_GT(observedCyclesPerMs, 0U);
  EXPECT_GT(observedStart, 0U) << "arming must stamp an origin to measure from";
  EXPECT_GT(observedDeadline, observedStart);
  EXPECT_EQ(
      static_cast<int64_t>(
          (observedDeadline - observedStart) / observedCyclesPerMs),
      kRequestedTimeoutMs);
}

// An unarmed handle has no origin, so the log must be able to tell "never
// armed" from "armed at clock zero" and report -1 rather than an elapsed time
// counted from the start of the device's uptime.
TEST(AbortDeviceTest, unarmedHandleReportsNoArmSite) {
  Abort abort{/*enabled=*/true};
  const auto handle = abort.getDeviceHandle();

  // `startCycles == 0` is the field that actually encodes "never armed", and it
  // is what makes the log's `armed` predicate false and its `elapsed_ms` -1.
  // Asserting only `opId`/`cyclesPerMs` would leave this test passing through a
  // regression in the arm-site origin, which is the thing it exists to protect.
  EXPECT_EQ(handle.startCycles(), 0U)
      << "an unarmed handle must have no origin to measure from";
  EXPECT_EQ(handle.deadlineCycles(), 0U);
  EXPECT_EQ(handle.opId(), 0U);
  EXPECT_GT(handle.cyclesPerMs(), 0U)
      << "the clock conversion is captured at handle creation, not at arm time";
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

// The third way a device abort is declared: not `setAbort()` from either
// handle, but a deadline lapsing inside `FT_ABORT_CHECK`. The thread that wins
// the timeout CAS logs through the same marker, and it carries the caller's own
// message and source location so the line says which wait gave up.
//
// Deliberately on the SKIP path. Under TRAP the marker is unassertable:
// `__trap()` faults the context and the printf FIFO is not reliably drained.
TEST(AbortMacrosTest, TimeoutFirstWriterEmitsTheMarker) {
  Abort abort{/*enabled=*/true};
  abort.setDefaultTimeout(kMacroTimeoutMs);
  ASSERT_EQ(abort.getDeviceHandle().behavior(), AbortBehavior::SKIP);

  auto iterations = makeDeviceValue<int>();
  ASSERT_NE(iterations, nullptr);

  const std::string out = captureDeviceStdout([&] {
    return launchMacroTimeoutLoop(
        abort.getDeviceHandle(),
        iterations.get(),
        kMacroTimeoutLoopBound,
        /*stream=*/nullptr);
  });

  ASSERT_TRUE(abort.isTimedOut());
  EXPECT_THAT(
      out,
      ::testing::HasSubstr(
          std::string{kFirstWriterMarker} +
          "device macroTimeoutLoop iteration"))
      << "captured: " << out;
  // The source location the macro concatenates at compile time, which is what
  // makes the line attributable to a wait rather than only to a communicator.
  EXPECT_THAT(out, ::testing::HasSubstr("AbortDeviceTest.cu:"))
      << "captured: " << out;
  // Exactly one: every later observer of the same terminal reason stays silent.
  EXPECT_EQ(countSubstr(out, kFirstWriterMarker), 1U) << "captured: " << out;
}

} // namespace comms::fault_tolerance::testing
