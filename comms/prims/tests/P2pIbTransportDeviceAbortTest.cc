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

// Backing buffer plus result slot for the queue-full put test.
class PutFixture {
 public:
  PutFixture() {
    CUDACHECK_TEST(cudaSetDevice(0));
    CUDACHECK_TEST(cudaMalloc(&data_, sizeof(uint64_t)));
    CUDACHECK_TEST(cudaMalloc(&posted_, sizeof(uint32_t)));
    CUDACHECK_TEST(cudaMemset(data_, 0, sizeof(uint64_t)));
    CUDACHECK_TEST(cudaMemset(posted_, 0, sizeof(uint32_t)));
  }

  ~PutFixture() {
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaFree(posted_);
    // NOLINTNEXTLINE(facebook-cuda-safe-api-call-check)
    cudaFree(data_);
  }

  PutFixture(const PutFixture&) = delete;
  PutFixture& operator=(const PutFixture&) = delete;
  PutFixture(PutFixture&&) = delete;
  PutFixture& operator=(PutFixture&&) = delete;

  uint64_t* data() {
    return data_;
  }

  uint32_t* posted() {
    return posted_;
  }

  uint32_t readPosted() {
    uint32_t value = 0;
    CUDACHECK_TEST(
        cudaMemcpy(&value, posted_, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    return value;
  }

 private:
  uint64_t* data_{nullptr};
  uint32_t* posted_{nullptr};
};

} // namespace

// A put that finds the ring full has to unwind, not trap: under fault tolerance
// a device trap takes down the CUDA context for the whole process, which is the
// failure the abort path exists to prevent.
TEST(P2pIbTransportDeviceAbortTest, IbrcPutSkipsWhenQueueFullAndAborted) {
  PutFixture fixture;
  comms::fault_tolerance::Abort abort(/*enabled=*/true);
  const uint32_t depth = test::ibrcTestQueueDepth();

  test::launchIbrcPutUntilQueueFull(
      fixture.data(),
      fixture.posted(),
      /*attempts=*/depth + 2,
      abort.getDeviceHandle());
  // Deliberate, and not replaceable by a condition variable: the abort has to
  // land *while* the kernel is spinning on the full ring, and the only signal
  // that it got there is the device-side posted counter, which the host cannot
  // read mid-kernel without serialising behind the very kernel it is waiting
  // on. 200 ms is two orders of magnitude above the launch it is covering.
  // NOLINTNEXTLINE(facebook-hte-BadCall-sleep_for)
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  abort.setAbort();
  CUDACHECK_TEST(cudaDeviceSynchronize());

  // Exactly the puts that fit are posted. The one that blocked on the full ring
  // is dropped by the abort, and the one after it is dropped without ever
  // touching the ring because the abort is already latched.
  EXPECT_EQ(fixture.readPosted(), depth);
  EXPECT_TRUE(abort.isAborted());
}

// The stalled-CPU-proxy case running concurrently with a collective deadline.
// No `setAbort()` is called anywhere in this test.
//
// Two bounds are in play and they are deliberately independent. The collective
// block owns the armed handle and expires on kTimeout, latching TIMED_OUT --
// that is the reason the host reports, because it is recorded first. The parked
// producer block is bounded by the fixed SM-to-proxy watchdog instead: it does
// not read the abort flag inside its spin, so the collective's shorter deadline
// does not release it. Both terminate, which is what liveness requires; only
// the promptness differs.
//
// Distinct from the wait-signal timeout tests, which stall in a wait that holds
// the armed handle itself. Here the two roles sit in different blocks, which is
// the arrangement a real collective produces.
TEST(
    P2pIbTransportDeviceAbortTest,
    IbrcQueueFullUnwindsWhileCollectiveTimesOut) {
  constexpr std::chrono::milliseconds kTimeout{500};
  PutFixture fixture;
  // The signal the collective block waits on. Never set, so its wait can only
  // end on the deadline.
  WaitFixture waitFixture;
  comms::fault_tolerance::Abort abort(/*enabled=*/true);
  abort.setDefaultTimeout(kTimeout);
  const uint32_t depth = test::ibrcTestQueueDepth();

  const auto start = std::chrono::steady_clock::now();
  test::launchIbrcQueueFullReleasedByCollectiveDeadline(
      fixture.data(),
      fixture.posted(),
      /*attempts=*/depth + 2,
      waitFixture.signal(),
      abort.getDeviceHandle());
  CUDACHECK_TEST(cudaDeviceSynchronize());
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start);

  // Same accounting as the host-abort case: the puts that fit are posted, the
  // one that blocks is released by the latched flag, and the next is skipped
  // before it touches the ring.
  EXPECT_EQ(fixture.readPosted(), depth)
      << "below depth means the collective block expired before the ring "
         "filled, which is not the scenario under test";
  // isAborted() is the generic "flag is latched" predicate and is true for a
  // timeout too, so isTimedOut() is what distinguishes a deadline from an
  // explicit abort.
  EXPECT_TRUE(abort.isTimedOut())
      << "the collective deadline is the first terminal reason recorded, so it "
         "is what the host sees even though the producer left on its watchdog";
  // Upper bound only; the lower bound is timing-sensitive under load. The bound
  // is the fixed proxy watchdog, not kTimeout: the parked producer does not
  // read the flag in its loop, so the collective's shorter deadline does not
  // release it early.
  EXPECT_LT(elapsed, std::chrono::seconds(30));
}

// The same skip, decided before the kernel does any work at all.
TEST(P2pIbTransportDeviceAbortTest, IbrcPutSkipsWhenPreAborted) {
  PutFixture fixture;
  comms::fault_tolerance::Abort abort(/*enabled=*/true);
  abort.setAbort();

  test::launchIbrcPutUntilQueueFull(
      fixture.data(),
      fixture.posted(),
      /*attempts=*/test::ibrcTestQueueDepth(),
      abort.getDeviceHandle());
  CUDACHECK_TEST(cudaDeviceSynchronize());

  EXPECT_EQ(fixture.readPosted(), 0U);
}

// The gap @benrcarver identified: a kernel that *ends* in flush() rather than
// parking in reserve(). Before the watchdog became unconditional, enabling FT
// removed the bound here entirely -- the legacy cycle deadline was gated on
// `!abort.isEnabled()` and P2pIbTransportDevice::flush dropped the caller's
// deadline on the IBRC branch -- so this hung until an explicit host abort.
//
// Nothing calls setAbort(). The drain ends on the fixed proxy watchdog and
// latches IBRC_PROXY_TIMEOUT, which names the stalled proxy rather than
// pretending the collective's own deadline expired.
//
// The configured 500 ms timeout is here to show it is *not* what bounds this:
// the watchdog is a property of the SM-to-proxy contract, so a shorter
// collective deadline neither shortens nor lengthens it.
TEST(P2pIbTransportDeviceAbortTest, IbrcFlushBoundedWithFtEnabled) {
  PutFixture fixture;
  comms::fault_tolerance::Abort abort(/*enabled=*/true);
  abort.setDefaultTimeout(std::chrono::milliseconds(500));

  const auto start = std::chrono::steady_clock::now();
  test::launchIbrcFlushNeverDrains(
      fixture.data(), fixture.posted(), abort.getDeviceHandle());
  CUDACHECK_TEST(cudaDeviceSynchronize());
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start);

  EXPECT_LT(elapsed, std::chrono::seconds(30))
      << "flush must be bounded with FT enabled; an unbounded drain is the "
         "regression this case exists to catch";
  EXPECT_EQ(
      abort.reason(), comms::fault_tolerance::AbortReason::IBRC_PROXY_TIMEOUT)
      << "a stalled proxy must be attributed to the proxy, not reported as the "
         "collective's own deadline expiring";
}

// Negative control for the two above: with no handle wired there is no abort to
// observe, so every put up to the ring depth must still post normally. This is
// what keeps the skip path from silently swallowing traffic on the legacy path.
TEST(P2pIbTransportDeviceAbortTest, IbrcPutFillsQueueWithoutAbortHandle) {
  PutFixture fixture;
  comms::fault_tolerance::AbortDevice disabled;
  const uint32_t depth = test::ibrcTestQueueDepth();

  test::launchIbrcPutUntilQueueFull(
      fixture.data(), fixture.posted(), /*attempts=*/depth, disabled);
  CUDACHECK_TEST(cudaDeviceSynchronize());

  EXPECT_EQ(fixture.readPosted(), depth);
}

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
