// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <chrono>
#include <thread>

#include "comms/common/fault_tolerance/Abort.h"
#include "comms/prims/tests/P2pIbTransportDeviceAbortTest.cuh"
#include "comms/testinfra/TestXPlatUtils.h"

namespace comms::prims {

namespace {

using Launcher = void (*)(
    uint64_t*,
    bool*,
    uint64_t,
    comms::fault_tolerance::AbortDevice,
    uint32_t*);

// Host-mapped flag the kernel raises just before entering the wait.
//
// This replaces a fixed sleep. A sleep only bounds how long the host waits; it
// proves nothing about the kernel, so a slow launch silently turns an
// abort-during-wait test into a pre-abort one and every assertion still passes.
// Polling the flag makes that failure loud instead.
class EnteredWaitFlag {
 public:
  EnteredWaitFlag() {
    CUDACHECK_TEST(cudaSetDevice(0));
    CUDACHECK_TEST(cudaHostAlloc(
        reinterpret_cast<void**>(&host_),
        sizeof(uint32_t),
        cudaHostAllocMapped));
    *host_ = 0;
    CUDACHECK_TEST(
        cudaHostGetDevicePointer(reinterpret_cast<void**>(&device_), host_, 0));
  }

  ~EnteredWaitFlag() {
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaFreeHost(host_);
  }

  EnteredWaitFlag(const EnteredWaitFlag&) = delete;
  EnteredWaitFlag& operator=(const EnteredWaitFlag&) = delete;
  EnteredWaitFlag(EnteredWaitFlag&&) = delete;
  EnteredWaitFlag& operator=(EnteredWaitFlag&&) = delete;

  uint32_t* device() {
    return device_;
  }

  // Bounded because a kernel that never reaches the wait must fail the test
  // rather than hang it.
  bool await(std::chrono::milliseconds bound) const {
    const auto deadline = std::chrono::steady_clock::now() + bound;
    while (std::chrono::steady_clock::now() < deadline) {
      if (__atomic_load_n(host_, __ATOMIC_ACQUIRE) != 0U) {
        return true;
      }
      // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return false;
  }

 private:
  uint32_t* host_{nullptr};
  uint32_t* device_{nullptr};
};

// Device-side signal the wait polls, plus the slot it reports its result in.
class WaitFixture {
 public:
  WaitFixture() {
    CUDACHECK_TEST(cudaSetDevice(0));
    CUDACHECK_TEST(cudaMalloc(&signal_, sizeof(uint64_t)));
    CUDACHECK_TEST(cudaMalloc(&waitResult_, sizeof(bool)));
    CUDACHECK_TEST(cudaMemset(signal_, 0, sizeof(uint64_t)));
    CUDACHECK_TEST(cudaMemset(waitResult_, 0, sizeof(bool)));
  }

  ~WaitFixture() {
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaFree(waitResult_);
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaFree(signal_);
  }

  WaitFixture(const WaitFixture&) = delete;
  WaitFixture& operator=(const WaitFixture&) = delete;
  WaitFixture(WaitFixture&&) = delete;
  WaitFixture& operator=(WaitFixture&&) = delete;

  uint64_t* signal() {
    return signal_;
  }

  bool* waitResult() {
    return waitResult_;
  }

  bool readWaitResult() {
    bool value = false;
    CUDACHECK_TEST(
        cudaMemcpy(&value, waitResult_, sizeof(bool), cudaMemcpyDeviceToHost));
    return value;
  }

 private:
  uint64_t* signal_{nullptr};
  bool* waitResult_{nullptr};
};

// A satisfied wait must report success and must not touch abort state.
void runAlreadySatisfiedWait(Launcher launcher) {
  WaitFixture fixture;
  comms::fault_tolerance::AbortDevice disabled;

  launcher(
      fixture.signal(),
      fixture.waitResult(),
      /*expected=*/0,
      disabled,
      /*enteredWait=*/nullptr);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  EXPECT_TRUE(fixture.readWaitResult());
}

// An abort recorded before launch must make the wait give up immediately.
void runPreAbortedWait(Launcher launcher) {
  WaitFixture fixture;
  comms::fault_tolerance::Abort abort(/*enabled=*/true);
  abort.setAbort();

  launcher(
      fixture.signal(),
      fixture.waitResult(),
      /*expected=*/1,
      abort.getDeviceHandle(),
      /*enteredWait=*/nullptr);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  EXPECT_FALSE(fixture.readWaitResult());
}

// The liveness property: a wait already spinning on a signal that will never
// arrive still exits once the host aborts. Without this the kernel runs until
// the CUDA launch timeout, which is the hang this stack exists to remove.
void runAbortDuringWait(Launcher launcher) {
  WaitFixture fixture;
  EnteredWaitFlag entered;
  comms::fault_tolerance::Abort abort(/*enabled=*/true);

  launcher(
      fixture.signal(),
      fixture.waitResult(),
      /*expected=*/1,
      abort.getDeviceHandle(),
      entered.device());
  const bool reachedWait = entered.await(std::chrono::seconds(10));
  // Abort and drain before asserting on the handshake. `setAbort()` is what
  // releases the kernel, so failing out first would run `abort`'s and
  // `fixture`'s destructors -- unmapping the abort state and freeing the signal
  // buffer -- while the kernel is still polling both.
  abort.setAbort();
  CUDACHECK_TEST(cudaDeviceSynchronize());
  ASSERT_TRUE(reachedWait)
      << "kernel never reached the wait, so this would silently degenerate "
         "into the pre-abort case";

  EXPECT_FALSE(fixture.readWaitResult());
  EXPECT_TRUE(abort.isAborted());
  EXPECT_FALSE(abort.isTimedOut())
      << "an explicit abort must win the first-writer race";
}

// The device deadline reaches the same wait with no host involvement, and the
// expiry is recorded in shared state where host code can see it.
void runTimeoutDuringWait(Launcher launcher) {
  constexpr std::chrono::milliseconds kTimeout{500};
  WaitFixture fixture;
  comms::fault_tolerance::Abort abort(/*enabled=*/true);
  abort.setDefaultTimeout(kTimeout);

  const auto start = std::chrono::steady_clock::now();
  launcher(
      fixture.signal(),
      fixture.waitResult(),
      /*expected=*/1,
      abort.getDeviceHandle(),
      /*enteredWait=*/nullptr);
  CUDACHECK_TEST(cudaDeviceSynchronize());
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start);

  EXPECT_FALSE(fixture.readWaitResult());
  EXPECT_TRUE(abort.isTimedOut());
  // Only an upper bound is asserted: the lower bound is timing-sensitive under
  // load, while "finished long before any watchdog" is the property at stake.
  EXPECT_LT(elapsed, std::chrono::seconds(10));
}

} // namespace

TEST(P2pIbTransportDeviceAbortTest, WrapperWaitSignalSucceedsWhenSatisfied) {
  runAlreadySatisfiedWait(test::launchIbWrapperWaitSignal);
}

TEST(P2pIbTransportDeviceAbortTest, IbrcWaitSignalSucceedsWhenSatisfied) {
  runAlreadySatisfiedWait(test::launchIbrcWaitSignal);
}

TEST(P2pIbTransportDeviceAbortTest, WrapperWaitSignalSkipsWhenPreAborted) {
  runPreAbortedWait(test::launchIbWrapperWaitSignal);
}

TEST(P2pIbTransportDeviceAbortTest, IbrcWaitSignalSkipsWhenPreAborted) {
  runPreAbortedWait(test::launchIbrcWaitSignal);
}

TEST(P2pIbTransportDeviceAbortTest, WrapperWaitSignalExitsOnAbortDuringWait) {
  runAbortDuringWait(test::launchIbWrapperWaitSignal);
}

TEST(P2pIbTransportDeviceAbortTest, IbrcWaitSignalExitsOnAbortDuringWait) {
  runAbortDuringWait(test::launchIbrcWaitSignal);
}

TEST(P2pIbTransportDeviceAbortTest, WrapperWaitSignalExitsOnDeviceTimeout) {
  runTimeoutDuringWait(test::launchIbWrapperWaitSignal);
}

TEST(P2pIbTransportDeviceAbortTest, IbrcWaitSignalExitsOnDeviceTimeout) {
  runTimeoutDuringWait(test::launchIbrcWaitSignal);
}

} // namespace comms::prims
