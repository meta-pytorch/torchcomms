// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <atomic>
#include <exception>
#include <string>
#include <thread>

#include <c10/util/intrusive_ptr.h>

#include "comms/torchcomms/BackendWrapper.hpp"
#include "comms/torchcomms/TorchWork.hpp"

// "Completed exactly once" is asserted by these tests NOT throwing: a second
// markCompleted() raises. A callback counter cannot see it -- markCompleted
// swaps the callback list out before invoking, so it can never reach two.

namespace torch::comms::test {
namespace {

// Releases both threads within nanoseconds. A condvar wakeup -- or just
// starting a thread and hoping -- is too coarse to land inside the window.
class SpinRendezvous {
 public:
  void arriveAndWait() {
    arrived_.fetch_add(1, std::memory_order_acq_rel);
    while (arrived_.load(std::memory_order_acquire) < 2) {
    }
  }

 private:
  std::atomic<int> arrived_{0};
};

// wait() is stream-ordered and does not complete the work, like the real CUDA
// backends: the terminal status only arrives when the test calls setStatus().
class TestWork final : public TorchWork {
 public:
  void wait() override {
    runWaitPreHooks();
    if (waitRendezvous_ != nullptr) {
      waitRendezvous_->arriveAndWait();
    }
    runWaitPostHooks();
  }

  // Holds wait() open so the caller enters the completion path at the same
  // instant the end hook fires.
  void setWaitRendezvous(SpinRendezvous* rendezvous) {
    waitRendezvous_ = rendezvous;
  }

  using TorchWork::setStatus;

 private:
  SpinRendezvous* waitRendezvous_{nullptr};
};

} // namespace

// The end hook resolved the Future first; wait() must not resolve it again.
TEST(WorkWrapperTest, EndHookBeforeWaitCompletesFutureOnce) {
  auto work = c10::make_intrusive<TestWork>();
  auto wrapper = c10::make_intrusive<WorkWrapper>(work);

  work->setStatus(TorchWork::WorkStatus::COMPLETED);
  EXPECT_TRUE(wrapper->wait(kNoTimeout));

  EXPECT_TRUE(wrapper->getFuture()->completed());
}

// wait() resolved it first; the late end hook must not. Pre-fix, this ordering
// threw out of setStatus() on the watchdog thread.
TEST(WorkWrapperTest, EndHookAfterWaitCompletesFutureOnce) {
  auto work = c10::make_intrusive<TestWork>();
  auto wrapper = c10::make_intrusive<WorkWrapper>(work);

  EXPECT_TRUE(wrapper->wait(kNoTimeout));
  EXPECT_NO_THROW(work->setStatus(TorchWork::WorkStatus::COMPLETED));

  EXPECT_TRUE(wrapper->getFuture()->completed());
}

// Work already finished before construction (e.g. the TorchWorkCompleted
// sentinel from an empty coalescing window) resolves the Future in the ctor.
TEST(WorkWrapperTest, WorkCompletedBeforeConstructionCompletesFutureOnce) {
  auto work = c10::make_intrusive<TestWork>();
  work->setStatus(TorchWork::WorkStatus::COMPLETED);

  auto wrapper = c10::make_intrusive<WorkWrapper>(work);
  EXPECT_TRUE(wrapper->getFuture()->completed());

  EXPECT_TRUE(wrapper->wait(kNoTimeout));
}

// synchronize() shares the completion path with wait().
TEST(WorkWrapperTest, SynchronizeAfterEndHookCompletesFutureOnce) {
  auto work = c10::make_intrusive<TestWork>();
  auto wrapper = c10::make_intrusive<WorkWrapper>(work);

  work->setStatus(TorchWork::WorkStatus::COMPLETED);
  wrapper->synchronize();

  EXPECT_TRUE(wrapper->getFuture()->completed());
}

// The production failure: the watchdog's end hook fires as the trainer comes
// out of wait(). Both used to observe an incomplete Future and both marked it.
TEST(WorkWrapperTest, ConcurrentEndHookAndWaitCompleteFutureOnce) {
  constexpr int kIterations = 1000;

  for (int i = 0; i < kIterations; ++i) {
    SpinRendezvous rendezvous;
    auto work = c10::make_intrusive<TestWork>();
    work->setWaitRendezvous(&rendezvous);
    auto wrapper = c10::make_intrusive<WorkWrapper>(work);

    // Report rather than throw: an escaping exception would hit the
    // still-joinable thread and abort the process instead of failing the test.
    std::string watchdogError;
    std::thread watchdog([&]() {
      rendezvous.arriveAndWait();
      try {
        work->setStatus(TorchWork::WorkStatus::COMPLETED);
      } catch (const std::exception& exception) {
        watchdogError = exception.what();
      }
    });

    // Sampled before join(), which would resolve the Future by itself and make
    // the assertion below pass whether or not wait() honoured blockIfLost.
    bool completedOnWaitReturn = false;
    std::string waitError;
    try {
      // EXPECT_ not ASSERT_: an ASSERT_ here returns from the test body with
      // the thread still joinable, which terminates the process.
      EXPECT_TRUE(wrapper->wait(kNoTimeout));
      completedOnWaitReturn = wrapper->getFuture()->completed();
    } catch (const std::exception& exception) {
      waitError = exception.what();
    }
    watchdog.join();

    ASSERT_TRUE(waitError.empty()) << "iteration " << i << ": " << waitError;
    ASSERT_TRUE(watchdogError.empty())
        << "iteration " << i << ": " << watchdogError;
    // wait() must not return while the end hook is still marking the Future.
    ASSERT_TRUE(completedOnWaitReturn) << "iteration " << i;
  }
}

} // namespace torch::comms::test
