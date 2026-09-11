// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/common/fault_tolerance/Abort.h"
#include "comms/common/fault_tolerance/tests/AbortDeviceTest.cuh"
#include "comms/common/fault_tolerance/tests/AbortLogMarkers.h"

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

// Captured stdout plus the CUDA status of the launch that produced it.
//
// The status has to travel with the text: `ASSERT_*` cannot be used in a
// value-returning helper, and reporting a launch or sync failure with
// `EXPECT_*` inside the helper lets the test carry on and match against a
// capture that may be empty -- so the real failure is a CUDA error but what
// gets reported is a missing substring.
struct DeviceCapture {
  std::string out;
  cudaError_t status{cudaSuccess};
};

// Runs `launch`, drains the device printf FIFO, and returns everything the
// process wrote to stdout meanwhile.
//
// The synchronize has to happen *inside* the capture window. Device `printf`
// appends to a per-context FIFO that the runtime drains only on kernel
// completion, synchronization, or context destruction, so reading the capture
// before syncing returns an empty string and the test silently proves nothing.
template <typename Launch>
DeviceCapture captureDeviceStdoutWithStatus(Launch&& launch) {
  ::testing::internal::CaptureStdout();
  const cudaError_t launched = launch();
  const cudaError_t synced = cudaDeviceSynchronize();
  std::string captured = ::testing::internal::GetCapturedStdout();
  return DeviceCapture{
      std::move(captured), launched != cudaSuccess ? launched : synced};
}

// The device first-writer line as it should render, built in one place.
//
// Every assertion on the rendered line goes through this, so the field layout
// lives here rather than in each test. Reconstructing it inline meant a change
// to the line's shape had to be mirrored at every assertion, and missing one
// left a stale expectation that still compiled.
std::string expectedDeviceFirstWriterLine(
    AbortReason reason,
    std::string_view context) {
  return std::string{kFirstWriterMarker} +
      "device reason=" + std::string{abortReasonToString(reason)} +
      " context=" + std::string{context};
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

  const auto firstOutCapture = captureDeviceStdoutWithStatus([&] {
    return launchDeviceSetAbortWithContext(
        abort.getDeviceHandle(),
        AbortReason::NETWORK_ERROR,
        /*useContext=*/true,
        firstWon.get(),
        /*stream=*/nullptr);
  });
  ASSERT_EQ(firstOutCapture.status, cudaSuccess);
  const std::string& firstOut = firstOutCapture.out;

  const auto secondOutCapture = captureDeviceStdoutWithStatus([&] {
    return launchDeviceSetAbortWithContext(
        abort.getDeviceHandle(),
        AbortReason::INTERNAL_ERROR,
        /*useContext=*/true,
        secondWon.get(),
        /*stream=*/nullptr);
  });
  ASSERT_EQ(secondOutCapture.status, cudaSuccess);
  const std::string& secondOut = secondOutCapture.out;

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
      ::testing::HasSubstr(expectedDeviceFirstWriterLine(
          AbortReason::NETWORK_ERROR, "AbortDeviceTest callsite")))
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

  const auto outCapture = captureDeviceStdoutWithStatus([&] {
    return launchDeviceSetAbortWithContext(
        abort.getDeviceHandle(),
        AbortReason::INTERNAL_ERROR,
        /*useContext=*/false,
        won.get(),
        /*stream=*/nullptr);
  });
  ASSERT_EQ(outCapture.status, cudaSuccess);
  const std::string& out = outCapture.out;

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
          expectedDeviceFirstWriterLine(AbortReason::INTERNAL_ERROR, "") +
          " op="))
      << "captured: " << out;
}

// `AbortFlag` is the other device writer of the shared reason -- the
// poll-state-free handle the IBRC transport keeps in device memory, and the one
// its proxy watchdogs abort through. It must produce the same first-writer line
// as `AbortDevice`, or a watchdog abort leaves no greppable origin at all.
TEST(AbortDeviceTest, flagSetAbortEmitsFirstWriterMarker) {
  Abort abort{/*enabled=*/true};
  auto won = makeDeviceValue<int>();
  auto contextReady = makeDeviceValue<int>();
  ASSERT_NE(won, nullptr);
  ASSERT_NE(contextReady, nullptr);

  const auto outCapture = captureDeviceStdoutWithStatus([&] {
    return launchAbortFlagSetAbort(
        abort.getDeviceHandle(),
        AbortReason::IBRC_PROXY_TIMEOUT,
        /*useContext=*/true,
        won.get(),
        contextReady.get(),
        /*stream=*/nullptr);
  });
  ASSERT_EQ(outCapture.status, cudaSuccess);
  const std::string& out = outCapture.out;

  EXPECT_EQ(readDeviceValue(won), 1);
  EXPECT_EQ(abort.reason(), AbortReason::IBRC_PROXY_TIMEOUT);

  // Including the context. The IBRC watchdogs already pass one naming which
  // watchdog fired; before this it was accepted and dropped.
  EXPECT_THAT(
      out,
      ::testing::HasSubstr(expectedDeviceFirstWriterLine(
          AbortReason::IBRC_PROXY_TIMEOUT, "AbortFlagTest callsite")))
      << "captured: " << out;
}

TEST(AbortDeviceTest, flagSetAbortLoserIsSilent) {
  Abort abort{/*enabled=*/true};
  auto won = makeDeviceValue<int>();
  auto contextReady = makeDeviceValue<int>();
  ASSERT_NE(won, nullptr);
  ASSERT_NE(contextReady, nullptr);

  // The host takes the reason first, so the flag's CAS loses.
  EXPECT_TRUE(abort.setAbort(AbortReason::ABORTED, "host got there first"));

  const auto outCapture = captureDeviceStdoutWithStatus([&] {
    return launchAbortFlagSetAbort(
        abort.getDeviceHandle(),
        AbortReason::NETWORK_ERROR,
        /*useContext=*/true,
        won.get(),
        contextReady.get(),
        /*stream=*/nullptr);
  });
  ASSERT_EQ(outCapture.status, cudaSuccess);
  const std::string& out = outCapture.out;

  EXPECT_EQ(readDeviceValue(won), 0);
  EXPECT_EQ(abort.reason(), AbortReason::ABORTED);
  EXPECT_EQ(countSubstr(out, kFirstWriterMarker), 0U) << "captured: " << out;
}

// The disabled-handle guard, from device code.
//
// This diff collapsed an `if (!isEnabled())` check out of both `setAbort`
// bodies into the single `state == nullptr` return in `deviceTrySetAbort`. The
// substitution is exact -- `isEnabled()` *is* `state_ != nullptr` -- but the
// only disabled-handle device tests drive `reason()` and `getTimeoutMs()`,
// which never reach that helper. Deleting the guard outright would have left
// the suite green.
TEST(AbortDeviceTest, flagSetAbortOnDisabledHandleIsSilentNoop) {
  const AbortDevice handle; // default-constructed: no shared state
  ASSERT_FALSE(handle.isEnabled());

  auto won = makeDeviceValue<int>();
  auto contextReady = makeDeviceValue<int>();
  ASSERT_NE(won, nullptr);
  ASSERT_NE(contextReady, nullptr);

  const auto outCapture = captureDeviceStdoutWithStatus([&] {
    return launchAbortFlagSetAbort(
        handle,
        AbortReason::NETWORK_ERROR,
        /*useContext=*/true,
        won.get(),
        contextReady.get(),
        /*stream=*/nullptr);
  });
  ASSERT_EQ(outCapture.status, cudaSuccess);
  const std::string& out = outCapture.out;

  EXPECT_EQ(readDeviceValue(won), 0) << "a disabled handle cannot win the CAS";
  EXPECT_EQ(readDeviceValue(contextReady), -1)
      << "a disabled handle has no state to publish readiness into";
  EXPECT_EQ(countSubstr(out, kFirstWriterMarker), 0U) << "captured: " << out;
}

// Contended `NONE -> TIMED_OUT`, which is the shape production actually
// produces: one `groupAborted()` leader per block, all polling the same shared
// reason, all reaching their deadline together.
//
// One leader wins the CAS. Every loser must still return true, which it can
// only do by reading the winner's reason back out of the CAS through
// `observed`. A loser that returned false would broadcast "not aborted" to its
// whole block and let it cross the abort gate into a peer-visible action while
// the communicator is already aborted -- Principle 4 in FAULT_TOLERANCE.md.
//
// The existing coverage cannot catch a regression here: both timeout kernels
// run a single poller, and `directTimeoutNonWinnerIsSilent` seeds the reason on
// the host, so the device never contends for the CAS at all.
TEST(AbortDeviceTest, contendedTimeoutLosersStillReportExpiry) {
  constexpr int kBlocks = 64;

  Abort abort{/*enabled=*/true};
  abort.setDefaultTimeout(std::chrono::milliseconds{1});

  // Mapped pinned, not device memory: the host releases the blocks with a plain
  // store while the kernel is spinning. A `cudaMemcpy` would be ordered behind
  // the very kernel it is meant to unblock, and the two would deadlock.
  int* hostGate = nullptr;
  ASSERT_EQ(
      cudaHostAlloc(&hostGate, sizeof(int), cudaHostAllocMapped), cudaSuccess);
  ASSERT_NE(hostGate, nullptr);
  *hostGate = 0;
  int* deviceGate = nullptr;
  ASSERT_EQ(cudaHostGetDevicePointer(&deviceGate, hostGate, 0), cudaSuccess);

  int* observedExpired = nullptr;
  int* observedWon = nullptr;
  ASSERT_EQ(cudaMalloc(&observedExpired, kBlocks * sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&observedWon, kBlocks * sizeof(int)), cudaSuccess);
  ASSERT_NE(observedExpired, nullptr);
  ASSERT_NE(observedWon, nullptr);
  ASSERT_EQ(cudaMemset(observedExpired, 0, kBlocks * sizeof(int)), cudaSuccess);
  ASSERT_EQ(cudaMemset(observedWon, 0, kBlocks * sizeof(int)), cudaSuccess);

  const auto outCapture = captureDeviceStdoutWithStatus([&] {
    const cudaError_t launched = launchDeviceContendedTimeout(
        abort.getDeviceHandle(),
        deviceGate,
        observedExpired,
        observedWon,
        kBlocks,
        kDeviceTimeoutPollIterations,
        /*stream=*/nullptr);
    if (launched != cudaSuccess) {
      return launched;
    }
    // Give every block time to reach the gate with its deadline armed, then
    // release them together so the CAS is genuinely contended rather than
    // serialized by launch skew.
    std::this_thread::sleep_for(std::chrono::milliseconds{50});
    __atomic_store_n(hostGate, 1, __ATOMIC_RELEASE);
    return cudaSuccess;
  });
  ASSERT_EQ(outCapture.status, cudaSuccess);
  const std::string& out = outCapture.out;

  std::vector<int> expired(kBlocks, 0);
  std::vector<int> won(kBlocks, 0);
  EXPECT_EQ(
      cudaMemcpy(
          expired.data(),
          observedExpired,
          kBlocks * sizeof(int),
          cudaMemcpyDeviceToHost),
      cudaSuccess);
  EXPECT_EQ(
      cudaMemcpy(
          won.data(),
          observedWon,
          kBlocks * sizeof(int),
          cudaMemcpyDeviceToHost),
      cudaSuccess);
  EXPECT_EQ(cudaFree(observedExpired), cudaSuccess);
  EXPECT_EQ(cudaFree(observedWon), cudaSuccess);
  EXPECT_EQ(cudaFreeHost(hostGate), cudaSuccess);

  EXPECT_EQ(abort.reason(), AbortReason::TIMED_OUT);
  // The property that matters: nobody reports "not aborted".
  EXPECT_EQ(std::count(expired.begin(), expired.end(), 1), kBlocks)
      << "every contending block must observe expiry, winner or loser";
  // And the split is real -- one transition, many non-winners. Without this the
  // test would still pass if every block somehow won its own CAS.
  EXPECT_EQ(std::count(won.begin(), won.end(), 1), 1)
      << "exactly one block may perform the NONE -> TIMED_OUT transition";
  // Still exactly one line: the CAS is what gates it, and only one block can
  // perform the transition no matter how many observe it.
  EXPECT_EQ(countSubstr(out, kFirstWriterMarker), 1U) << "captured: " << out;
}

TEST(AbortDeviceTest, abortFlagPublishesEmptyContext) {
  Abort abort{/*enabled=*/true};
  auto won = makeDeviceValue<int>();
  auto contextReady = makeDeviceValue<int>();
  ASSERT_NE(won, nullptr);
  ASSERT_NE(contextReady, nullptr);

  EXPECT_EQ(
      launchAbortFlagSetAbort(
          abort.getDeviceHandle(),
          AbortReason::INTERNAL_ERROR,
          /*useContext=*/false,
          won.get(),
          contextReady.get(),
          /*stream=*/nullptr),
      cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  EXPECT_EQ(readDeviceValue(won), 1);
  EXPECT_EQ(readDeviceValue(contextReady), 1);
  EXPECT_EQ(
      abort.getAbortInfo(),
      (AbortInfo{
          .reason = AbortReason::INTERNAL_ERROR,
          .context = "",
      }));
}

TEST(AbortDeviceTest, missingContextPublicationReturnsBestEffortSnapshot) {
  Abort abort{/*enabled=*/true};
  auto won = makeDeviceValue<int>();
  ASSERT_NE(won, nullptr);

  EXPECT_EQ(
      launchDevicePublishReasonWithoutContext(
          abort.getDeviceHandle(),
          AbortReason::NETWORK_ERROR,
          won.get(),
          /*stream=*/nullptr),
      cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(readDeviceValue(won), 1);

  const auto start = std::chrono::steady_clock::now();
  EXPECT_EQ(
      abort.getAbortInfo(),
      (AbortInfo{
          .reason = AbortReason::NETWORK_ERROR,
          .context = "",
      }));
  EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds{2});
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
  EXPECT_EQ(
      abort.getAbortInfo(),
      (AbortInfo{.reason = AbortReason::TIMED_OUT, .context = ""}));
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

// The timeout path has two entry shapes and neither goes through a macro. These
// are how production actually observes a deadline: `groupAborted()` in the IB
// and NVL send/recv/forward loops and in Ring AllReduce calls `checkExpired()`,
// and the IBGDA warp proxy calls `isAborted()` directly. Neither has a macro
// above it to name a callsite, so unless the transition speaks for itself the
// abort has no greppable origin -- and no later observer can supply one,
// because they all take the `reason() != NONE` return without transitioning.
// Routing the deadline CAS through `detail::deviceTrySetAbort` is what makes
// these two identical to `setAbort()` in what they emit.

TEST(AbortDeviceTest, directTimeoutViaIsAbortedEmitsTheMarker) {
  Abort abort{/*enabled=*/true};
  auto observedMode = makeDeviceValue<int>();
  auto observedIsAborted = makeDeviceValue<int>();
  ASSERT_NE(observedMode, nullptr);
  ASSERT_NE(observedIsAborted, nullptr);

  abort.setDefaultTimeout(std::chrono::milliseconds{1});

  const auto outCapture = captureDeviceStdoutWithStatus([&] {
    return launchDeviceWaitForTimeout(
        abort.getDeviceHandle(),
        observedMode.get(),
        observedIsAborted.get(),
        kDeviceTimeoutPollIterations,
        /*stream=*/nullptr);
  });
  ASSERT_EQ(outCapture.status, cudaSuccess);
  const std::string& out = outCapture.out;

  ASSERT_TRUE(abort.isTimedOut());
  EXPECT_THAT(
      out,
      ::testing::HasSubstr(expectedDeviceFirstWriterLine(
          AbortReason::TIMED_OUT, kDeadlineExpiredContext)))
      << "captured: " << out;
  // Exactly one, even though the kernel polls in a loop: the line is gated on
  // the CAS, and only one iteration can perform the transition.
  EXPECT_EQ(countSubstr(out, kFirstWriterMarker), 1U) << "captured: " << out;
}

TEST(AbortDeviceTest, directTimeoutViaCheckExpiredEmitsTheMarker) {
  Abort abort{/*enabled=*/true};
  auto observedMode = makeDeviceValue<int>();
  auto observedCheckExpired = makeDeviceValue<int>();
  ASSERT_NE(observedMode, nullptr);
  ASSERT_NE(observedCheckExpired, nullptr);

  abort.setDefaultTimeout(std::chrono::milliseconds{1});

  const auto outCapture = captureDeviceStdoutWithStatus([&] {
    return launchDeviceWaitForTimeoutStartAlias(
        abort.getDeviceHandle(),
        observedMode.get(),
        observedCheckExpired.get(),
        kDeviceTimeoutPollIterations,
        /*stream=*/nullptr);
  });
  ASSERT_EQ(outCapture.status, cudaSuccess);
  const std::string& out = outCapture.out;

  ASSERT_TRUE(abort.isTimedOut());
  EXPECT_THAT(
      out,
      ::testing::HasSubstr(expectedDeviceFirstWriterLine(
          AbortReason::TIMED_OUT, kDeadlineExpiredContext)))
      << "captured: " << out;
  EXPECT_EQ(countSubstr(out, kFirstWriterMarker), 1U) << "captured: " << out;
}

// A deadline that lapses after someone else already declared the reason is an
// observation, not a transition, and must stay silent. Without the CAS gate
// every thread that later notices the abort would emit its own line.
TEST(AbortDeviceTest, directTimeoutNonWinnerIsSilent) {
  Abort abort{/*enabled=*/true};
  auto observedMode = makeDeviceValue<int>();
  auto observedIsAborted = makeDeviceValue<int>();
  ASSERT_NE(observedMode, nullptr);
  ASSERT_NE(observedIsAborted, nullptr);

  // Host wins first. Its own marker goes to stderr, so it cannot be mistaken
  // for a device line in the stdout capture below.
  ASSERT_TRUE(abort.setAbort(AbortReason::ABORTED, "host got there first"));
  abort.setDefaultTimeout(std::chrono::milliseconds{1});

  const auto outCapture = captureDeviceStdoutWithStatus([&] {
    return launchDeviceWaitForTimeout(
        abort.getDeviceHandle(),
        observedMode.get(),
        observedIsAborted.get(),
        kDeviceTimeoutPollIterations,
        /*stream=*/nullptr);
  });
  ASSERT_EQ(outCapture.status, cudaSuccess);
  const std::string& out = outCapture.out;

  EXPECT_EQ(readDeviceValue(observedIsAborted), 1);
  EXPECT_EQ(abort.reason(), AbortReason::ABORTED);
  EXPECT_EQ(countSubstr(out, kFirstWriterMarker), 0U) << "captured: " << out;
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

// The two tests above check the arm-site state the context fields are derived
// from. This checks the fields as *rendered*, which is a different claim: the
// derivation can be correct while the printf that reports it is missing an
// argument, has them out of order, or was never reached. The clock-state tests
// pass in all three cases.
//
// It is also what pins the merge. `op`, `timeout_ms` and `elapsed_ms` used to
// be a second `COMMS FT ABORT CONTEXT:` line; asserting they appear on the
// first-writer line itself is what fails if anything splits them back apart.
TEST(AbortDeviceTest, deviceFirstWriterLineCarriesTheAbortContext) {
  constexpr int64_t kCommunicatorTimeoutMs = 2500;
  constexpr uint64_t kOpId = 987654321;

  Abort abort{/*enabled=*/true};
  abort.setDefaultTimeout(std::chrono::milliseconds{kCommunicatorTimeoutMs});

  auto handle = abort.getDeviceHandle();
  handle.setOpId(kOpId);

  auto won = makeDeviceValue<int>();
  ASSERT_NE(won, nullptr);

  const auto outCapture = captureDeviceStdoutWithStatus([&] {
    return launchDeviceSetAbortWithContext(
        handle,
        AbortReason::NETWORK_ERROR,
        /*useContext=*/true,
        won.get(),
        /*stream=*/nullptr);
  });
  ASSERT_EQ(outCapture.status, cudaSuccess);
  const std::string& out = outCapture.out;

  EXPECT_EQ(readDeviceValue(won), 1);
  // The kernel aborts without arming, so this operation has no deadline -- and
  // the line says so, even though the communicator default is 2500ms.
  //
  // That distinction is the whole point of deriving `timeout_ms` from the arm
  // site instead of reading the communicator default live. A live read would
  // print `timeout_ms=2500` here and claim a deadline that was never in force,
  // next to a `deadline_cycles=0` saying the opposite.
  EXPECT_THAT(
      out,
      ::testing::HasSubstr(
          std::string{kFirstWriterMarker} + "device reason=" +
          std::string{abortReasonToString(AbortReason::NETWORK_ERROR)} +
          " context=AbortDeviceTest callsite op=" + std::to_string(kOpId) +
          " timeout_ms=-1 elapsed_ms=-1 elapsed_cycles=0 deadline_cycles=0"))
      << "captured: " << out;
  // Still one line, carrying all of it.
  EXPECT_EQ(countSubstr(out, kFirstWriterMarker), 1U) << "captured: " << out;
}

// The other half of that claim: when a deadline *is* armed, the line reports
// the deadline actually in force.
//
// `timeout_ms` is recovered from `deadline_cycles - startCycles`, which is what
// `startTimeout()` built the deadline from, so it stays truthful even if the
// host moves the communicator default while the operation is in flight. Reading
// the default live would print the new value beside a deadline derived from the
// old one, and the line would contradict itself.
TEST(AbortDeviceTest, deviceFirstWriterLineReportsTheArmedDeadline) {
  // A is what the deadline is built from; B is what the shared default holds by
  // the time the line is emitted. They have to differ, and the handshake below
  // has to order them, or a live re-read of the default would print the same
  // number as the armed value and this test would prove nothing.
  constexpr int64_t kArmedTimeoutMs = 7;
  constexpr int64_t kChangedTimeoutMs = 4321;
  constexpr uint64_t kOpId = 24680;

  Abort abort{/*enabled=*/true};
  abort.setDefaultTimeout(std::chrono::milliseconds{kArmedTimeoutMs});

  auto handle = abort.getDeviceHandle();
  handle.setOpId(kOpId);

  // Mapped pinned both ways: the device publishes "armed" and the host releases
  // the deadline, both while the kernel is running.
  int* armedFlag = nullptr;
  int* startGate = nullptr;
  ASSERT_EQ(
      cudaHostAlloc(&armedFlag, sizeof(int), cudaHostAllocMapped), cudaSuccess);
  ASSERT_EQ(
      cudaHostAlloc(&startGate, sizeof(int), cudaHostAllocMapped), cudaSuccess);
  ASSERT_NE(armedFlag, nullptr);
  ASSERT_NE(startGate, nullptr);
  *armedFlag = 0;
  *startGate = 0;
  int* deviceArmedFlag = nullptr;
  int* deviceStartGate = nullptr;
  ASSERT_EQ(
      cudaHostGetDevicePointer(&deviceArmedFlag, armedFlag, 0), cudaSuccess);
  ASSERT_EQ(
      cudaHostGetDevicePointer(&deviceStartGate, startGate, 0), cudaSuccess);

  auto observedIsAborted = makeDeviceValue<int>();
  ASSERT_NE(observedIsAborted, nullptr);

  const auto outCapture = captureDeviceStdoutWithStatus([&] {
    const cudaError_t launched = launchDeviceArmThenAwaitHostThenTimeout(
        handle,
        deviceArmedFlag,
        deviceStartGate,
        observedIsAborted.get(),
        kDeviceTimeoutPollIterations,
        /*stream=*/nullptr);
    if (launched != cudaSuccess) {
      return launched;
    }
    // Wait until the deadline is actually built from A...
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds{5};
    while (__atomic_load_n(armedFlag, __ATOMIC_ACQUIRE) == 0) {
      if (std::chrono::steady_clock::now() > deadline) {
        return cudaErrorUnknown;
      }
      std::this_thread::yield();
    }
    // ...then move the shared default out from under it before releasing the
    // device. Anything that re-reads the default now sees B.
    abort.setDefaultTimeout(std::chrono::milliseconds{kChangedTimeoutMs});
    __atomic_store_n(startGate, 1, __ATOMIC_RELEASE);
    return cudaSuccess;
  });
  ASSERT_EQ(outCapture.status, cudaSuccess);
  const std::string& out = outCapture.out;

  EXPECT_EQ(cudaFreeHost(armedFlag), cudaSuccess);
  EXPECT_EQ(cudaFreeHost(startGate), cudaSuccess);

  ASSERT_TRUE(abort.isTimedOut());
  EXPECT_EQ(readDeviceValue(observedIsAborted), 1);

  // Delimited on both sides: an unterminated `timeout_ms=7` would also accept
  // 70 or 4321 truncated to a prefix, which is most of what this test is for.
  EXPECT_THAT(
      out,
      ::testing::HasSubstr(
          expectedDeviceFirstWriterLine(
              AbortReason::TIMED_OUT, kDeadlineExpiredContext) +
          " op=" + std::to_string(kOpId) +
          " timeout_ms=" + std::to_string(kArmedTimeoutMs) + " elapsed_ms="))
      << "captured: " << out;
  // And the value the rejected live read would have produced is absent.
  EXPECT_THAT(
      out,
      ::testing::Not(
          ::testing::HasSubstr(
              " timeout_ms=" + std::to_string(kChangedTimeoutMs))))
      << "captured: " << out;
  EXPECT_EQ(countSubstr(out, kFirstWriterMarker), 1U) << "captured: " << out;
}

// `AbortFlag` has no handle to report on -- no `opId`, no arm-site clock. The
// line still has to appear and keep its shape, with the timing fields reported
// as unarmed, or the IBRC watchdogs that abort only through this type produce a
// line that parses differently from every other abort.
TEST(AbortDeviceTest, flagFirstWriterLineReportsUnarmedContext) {
  Abort abort{/*enabled=*/true};
  auto won = makeDeviceValue<int>();
  auto contextReady = makeDeviceValue<int>();
  ASSERT_NE(won, nullptr);
  ASSERT_NE(contextReady, nullptr);

  const auto outCapture = captureDeviceStdoutWithStatus([&] {
    return launchAbortFlagSetAbort(
        abort.getDeviceHandle(),
        AbortReason::IBRC_PROXY_TIMEOUT,
        /*useContext=*/true,
        won.get(),
        contextReady.get(),
        /*stream=*/nullptr);
  });
  ASSERT_EQ(outCapture.status, cudaSuccess);
  const std::string& out = outCapture.out;

  EXPECT_EQ(readDeviceValue(won), 1);
  EXPECT_THAT(
      out,
      ::testing::HasSubstr(
          expectedDeviceFirstWriterLine(
              AbortReason::IBRC_PROXY_TIMEOUT, "AbortFlagTest callsite") +
          " op=-1 timeout_ms=-1 elapsed_ms=-1"))
      << "captured: " << out;
  EXPECT_EQ(countSubstr(out, kFirstWriterMarker), 1U) << "captured: " << out;
}

// `op=0` is a real operation number -- the INIT lifecycle record consumes trace
// sequence 0 -- so absence cannot be spelled the same way, or an `AbortFlag`
// line joins to a legitimate record instead of being recognized as having no
// operation identity. `-1` is the sentinel, matching `timeout_ms`/`elapsed_ms`
// beside it.
TEST(AbortDeviceTest, absentOpIdIsDistinguishableFromOperationZero) {
  Abort abort{/*enabled=*/true};
  auto won = makeDeviceValue<int>();
  auto contextReady = makeDeviceValue<int>();
  ASSERT_NE(won, nullptr);
  ASSERT_NE(contextReady, nullptr);

  // A real handle whose operation number genuinely is zero.
  auto handle = abort.getDeviceHandle();
  ASSERT_EQ(handle.opId(), 0U);

  const auto zeroCapture = captureDeviceStdoutWithStatus([&] {
    return launchDeviceSetAbortWithContext(
        handle,
        AbortReason::NETWORK_ERROR,
        /*useContext=*/false,
        won.get(),
        /*stream=*/nullptr);
  });
  ASSERT_EQ(zeroCapture.status, cudaSuccess);
  EXPECT_THAT(zeroCapture.out, ::testing::HasSubstr(" op=0 "))
      << "captured: " << zeroCapture.out;

  // And a writer with no operation identity at all.
  Abort flagAbort{/*enabled=*/true};
  const auto flagCapture = captureDeviceStdoutWithStatus([&] {
    return launchAbortFlagSetAbort(
        flagAbort.getDeviceHandle(),
        AbortReason::IBRC_PROXY_TIMEOUT,
        /*useContext=*/false,
        won.get(),
        contextReady.get(),
        /*stream=*/nullptr);
  });
  ASSERT_EQ(flagCapture.status, cudaSuccess);
  EXPECT_THAT(flagCapture.out, ::testing::HasSubstr(" op=-1 "))
      << "captured: " << flagCapture.out;
  EXPECT_THAT(flagCapture.out, ::testing::Not(::testing::HasSubstr(" op=0 ")))
      << "captured: " << flagCapture.out;
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
// handle, but a deadline lapsing inside `FT_ABORT_CHECK`.
//
// This is the case that produces both lines. The transition emits the
// first-writer line from inside the CAS exactly as the direct-timeout kernels
// above do, and the macro adds a site line naming the wait that noticed it.
// Asserting both is what pins the split: a regression that folds the caller's
// message back into the first-writer marker, or that drops the site line,
// changes one of these without changing the other.
//
// Deliberately on the SKIP path. Under TRAP the markers are unassertable:
// `__trap()` faults the context and the printf FIFO is not reliably drained.
TEST(AbortMacrosTest, TimeoutFirstWriterEmitsTheMarker) {
  Abort abort{/*enabled=*/true};
  abort.setDefaultTimeout(kMacroTimeoutMs);
  ASSERT_EQ(abort.getDeviceHandle().behavior(), AbortBehavior::SKIP);

  auto iterations = makeDeviceValue<int>();
  ASSERT_NE(iterations, nullptr);

  const auto outCapture = captureDeviceStdoutWithStatus([&] {
    return launchMacroTimeoutLoop(
        abort.getDeviceHandle(),
        iterations.get(),
        kMacroTimeoutLoopBound,
        /*stream=*/nullptr);
  });
  ASSERT_EQ(outCapture.status, cudaSuccess);
  const std::string& out = outCapture.out;

  ASSERT_TRUE(abort.isTimedOut());
  // The transition. Same line the direct-timeout kernels produce, because it
  // comes from the same place.
  EXPECT_THAT(
      out,
      ::testing::HasSubstr(expectedDeviceFirstWriterLine(
          AbortReason::TIMED_OUT, kDeadlineExpiredContext)))
      << "captured: " << out;
  // The observation, carrying what only the callsite knows.
  EXPECT_THAT(
      out,
      ::testing::HasSubstr(
          std::string{kSiteMarker} + "macroTimeoutLoop iteration"))
      << "captured: " << out;
  // The source location the macro concatenates at compile time, which is what
  // makes the site line attributable to a wait rather than to a communicator.
  EXPECT_THAT(out, ::testing::HasSubstr("AbortDeviceTest.cu:"))
      << "captured: " << out;
  // One of each: every later observer of the same terminal reason stays silent,
  // and the site line is gated on the same CAS win.
  EXPECT_EQ(countSubstr(out, kFirstWriterMarker), 1U) << "captured: " << out;
  EXPECT_EQ(countSubstr(out, kSiteMarker), 1U) << "captured: " << out;
}

} // namespace comms::fault_tolerance::testing
