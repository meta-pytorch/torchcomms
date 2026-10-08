// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <exception>
#include <functional>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#include <c10/util/Exception.h>
#include <c10/util/intrusive_ptr.h>

#include "comms/torchcomms/BackendWrapper.hpp"
#include "comms/torchcomms/TorchWork.hpp"

namespace torch::comms::test {
namespace {

using namespace std::chrono_literals;

class SpinRendezvous {
 public:
  void arriveAndWait() {
    arrived_.fetch_add(1, std::memory_order_acq_rel);
    while (arrived_.load(std::memory_order_acquire) < 2) {
      std::this_thread::yield();
    }
  }

 private:
  std::atomic<int> arrived_{0};
};

class TestWork final : public TorchWork {
 public:
  explicit TestWork(bool* destroyed = nullptr) : destroyed_(destroyed) {}

  ~TestWork() override {
    if (destroyed_ != nullptr) {
      *destroyed_ = true;
    }
  }

  void wait() override {
    runWaitPreHooks();
    if (waitAction_) {
      waitAction_();
    }
    runWaitPostHooks();
  }

  void hostSynchronize() override {
    if (hostSynchronizeAction_) {
      hostSynchronizeAction_();
    }
  }

  void setWaitAction(std::function<void()> action) {
    waitAction_ = std::move(action);
  }

  void setHostSynchronizeAction(std::function<void()> action) {
    hostSynchronizeAction_ = std::move(action);
  }

  using TorchWork::setStatus;

 private:
  std::function<void()> waitAction_;
  std::function<void()> hostSynchronizeAction_;
  bool* destroyed_{nullptr};
};

c10d::WorkResult getWorkResult(
    const c10::intrusive_ptr<c10::ivalue::Future>& future) {
  return static_cast<c10d::WorkResult>(future->value().toInt());
}

std::string getExceptionMessage(const std::exception_ptr& exception) {
  try {
    std::rethrow_exception(exception);
  } catch (const std::exception& error) {
    return error.what();
  }
}

[[noreturn]] void throwLocalWaitFailure() {
  throw std::runtime_error("local wait failed");
}

[[noreturn]] void throwSyntheticBackendFailure() {
  throw std::runtime_error("synthetic backend failure");
}

[[noreturn]] void throwSecondaryWaitFailure() {
  throw std::runtime_error("secondary wait failure");
}

[[noreturn]] void throwSyntheticHostSyncFailure() {
  throw std::runtime_error("synthetic host sync failure");
}

void expectFutureCompletes(
    const c10::intrusive_ptr<c10::ivalue::Future>& future,
    const std::function<void()>& complete) {
  std::promise<void> callbackRan;
  auto callbackFuture = callbackRan.get_future();
  future->addCallback(
      [&callbackRan](c10::ivalue::Future&) { callbackRan.set_value(); });
  complete();
  ASSERT_EQ(callbackFuture.wait_for(2s), std::future_status::ready);
}

} // namespace

TEST(WorkWrapperTest, RejectsNullWork) {
  EXPECT_THROW(
      {
        auto wrapper =
            c10::make_intrusive<WorkWrapper>(c10::intrusive_ptr<TorchWork>());
        (void)wrapper;
      },
      c10::Error);
}

TEST(WorkWrapperTest, PendingWorkPreservesC10dPollingSemantics) {
  auto work = c10::make_intrusive<TestWork>();
  work->setStatus(TorchWork::WorkStatus::INPROGRESS);
  auto wrapper = c10::make_intrusive<WorkWrapper>(work);

  EXPECT_FALSE(wrapper->isCompleted());
  EXPECT_TRUE(wrapper->isSuccess());
  EXPECT_EQ(wrapper->exception(), nullptr);
}

TEST(WorkWrapperTest, CompletedWorkResolvesBothFutures) {
  auto work = c10::make_intrusive<TestWork>();
  work->setStatus(TorchWork::WorkStatus::COMPLETED);

  auto wrapper = c10::make_intrusive<WorkWrapper>(work);

  EXPECT_TRUE(wrapper->isCompleted());
  EXPECT_TRUE(wrapper->isSuccess());
  EXPECT_EQ(wrapper->exception(), nullptr);
  EXPECT_TRUE(wrapper->getFuture()->completed());
  auto resultFuture = wrapper->getFutureResult();
  ASSERT_TRUE(resultFuture->completed());
  EXPECT_EQ(getWorkResult(resultFuture), c10d::WorkResult::SUCCESS);
}

TEST(WorkWrapperTest, ErrorStatusMapsToC10dFailure) {
  auto work = c10::make_intrusive<TestWork>();
  work->setStatus(TorchWork::WorkStatus::ERROR);
  auto wrapper = c10::make_intrusive<WorkWrapper>(work);

  EXPECT_TRUE(wrapper->isCompleted());
  EXPECT_FALSE(wrapper->isSuccess());
  ASSERT_NE(wrapper->exception(), nullptr);
  EXPECT_THROW(
      std::rethrow_exception(wrapper->exception()), c10::DistBackendError);
  EXPECT_EQ(
      getWorkResult(wrapper->getFutureResult()), c10d::WorkResult::COMM_ERROR);
  EXPECT_THROW(wrapper->wait(kNoTimeout), c10::DistBackendError);
}

TEST(WorkWrapperTest, TimeoutStatusMapsToC10dFailure) {
  auto work = c10::make_intrusive<TestWork>();
  work->setStatus(TorchWork::WorkStatus::TIMEDOUT);
  auto wrapper = c10::make_intrusive<WorkWrapper>(work);

  EXPECT_TRUE(wrapper->isCompleted());
  EXPECT_FALSE(wrapper->isSuccess());
  ASSERT_NE(wrapper->exception(), nullptr);
  EXPECT_THROW(
      std::rethrow_exception(wrapper->exception()), c10::DistBackendError);
  EXPECT_EQ(
      getWorkResult(wrapper->getFutureResult()), c10d::WorkResult::TIMEOUT);
  EXPECT_THROW(wrapper->wait(kNoTimeout), c10::DistBackendError);
}

TEST(WorkWrapperTest, WaitAndSynchronizePreserveNonterminalStatus) {
  auto work = c10::make_intrusive<TestWork>();
  work->setStatus(TorchWork::WorkStatus::INPROGRESS);
  auto wrapper = c10::make_intrusive<WorkWrapper>(work);

  EXPECT_TRUE(wrapper->wait(kNoTimeout));
  wrapper->synchronize();

  EXPECT_FALSE(wrapper->isCompleted());
  EXPECT_TRUE(wrapper->isSuccess());
  EXPECT_TRUE(wrapper->getFuture()->completed());
}

TEST(WorkWrapperTest, RepeatedWaitAndSynchronizeCompleteFuturesOnce) {
  int waitCalls = 0;
  auto work = c10::make_intrusive<TestWork>();
  work->setStatus(TorchWork::WorkStatus::COMPLETED);
  work->setWaitAction([&waitCalls]() { ++waitCalls; });
  auto wrapper = c10::make_intrusive<WorkWrapper>(work);

  EXPECT_TRUE(wrapper->wait(kNoTimeout));
  wrapper->synchronize();
  EXPECT_TRUE(wrapper->wait(kNoTimeout));

  EXPECT_TRUE(wrapper->getFuture()->completed());
  EXPECT_EQ(
      getWorkResult(wrapper->getFutureResult()), c10d::WorkResult::SUCCESS);
  EXPECT_EQ(waitCalls, 3);
}

TEST(WorkWrapperTest, FiniteWaitRejectionDoesNotPoisonWork) {
  auto work = c10::make_intrusive<TestWork>();
  work->setStatus(TorchWork::WorkStatus::INPROGRESS);
  auto wrapper = c10::make_intrusive<WorkWrapper>(work);

  EXPECT_THROW(wrapper->wait(1ms), std::runtime_error);
  EXPECT_FALSE(wrapper->isCompleted());
  EXPECT_TRUE(wrapper->isSuccess());
  EXPECT_EQ(wrapper->exception(), nullptr);

  work->setWaitAction(
      [work]() { work->setStatus(TorchWork::WorkStatus::COMPLETED); });
  EXPECT_TRUE(wrapper->wait(kNoTimeout));
  EXPECT_TRUE(wrapper->isCompleted());
  EXPECT_TRUE(wrapper->isSuccess());
  EXPECT_EQ(
      getWorkResult(wrapper->getFutureResult()), c10d::WorkResult::SUCCESS);
}

TEST(WorkWrapperTest, WaitFailureRemainsStickyAfterBackendCompletion) {
  int waitCalls = 0;
  auto work = c10::make_intrusive<TestWork>();
  work->setStatus(TorchWork::WorkStatus::INPROGRESS);
  work->setWaitAction([&waitCalls]() {
    ++waitCalls;
    throw std::runtime_error("first wait failed");
  });
  auto wrapper = c10::make_intrusive<WorkWrapper>(work);

  EXPECT_THROW(wrapper->wait(kNoTimeout), c10::DistBackendError);
  const auto firstException = wrapper->exception();
  ASSERT_NE(firstException, nullptr);
  work->setStatus(TorchWork::WorkStatus::COMPLETED);
  work->setWaitAction([&waitCalls]() { ++waitCalls; });

  EXPECT_THROW(wrapper->wait(kNoTimeout), c10::DistBackendError);
  EXPECT_EQ(wrapper->exception(), firstException);
  EXPECT_FALSE(wrapper->isSuccess());
  EXPECT_EQ(waitCalls, 1);
  EXPECT_EQ(
      getWorkResult(wrapper->getFutureResult()), c10d::WorkResult::COMM_ERROR);
}

TEST(WorkWrapperTest, ConcurrentWaitDoesNotReportSuccessAfterFailure) {
  // TestWork makes its wait action thread-safe so this can exercise only the
  // c10d adapter's outcome arbitration; backend wait concurrency remains
  // outside TorchWork's contract.
  SpinRendezvous rendezvous;
  std::atomic<int> waitCalls{0};
  std::atomic<int> successfulWaits{0};
  std::atomic<int> failedWaits{0};
  auto work = c10::make_intrusive<TestWork>();
  work->setStatus(TorchWork::WorkStatus::INPROGRESS);
  auto wrapper = c10::make_intrusive<WorkWrapper>(work);
  auto* const wrapperPtr = wrapper.get();
  work->setWaitAction([&]() {
    const int call = waitCalls.fetch_add(1);
    rendezvous.arriveAndWait();
    if (call == 0) {
      throw std::runtime_error("concurrent wait failed");
    }
    while (wrapperPtr->exception() == nullptr) {
      std::this_thread::yield();
    }
  });

  const auto wait = [&]() {
    try {
      if (wrapper->wait(kNoTimeout)) {
        successfulWaits.fetch_add(1);
      }
    } catch (const c10::DistBackendError&) {
      failedWaits.fetch_add(1);
    }
  };
  std::thread first(wait);
  std::thread second(wait);
  first.join();
  second.join();

  EXPECT_EQ(waitCalls.load(), 2);
  EXPECT_EQ(successfulWaits.load(), 0);
  EXPECT_EQ(failedWaits.load(), 2);
  EXPECT_FALSE(wrapper->isSuccess());
}

TEST(WorkWrapperTest, ProducerPublishesTerminalResultAfterLocalWaitFailure) {
  const std::array cases{
      std::pair{TorchWork::WorkStatus::ERROR, c10d::WorkResult::COMM_ERROR},
      std::pair{TorchWork::WorkStatus::TIMEDOUT, c10d::WorkResult::TIMEOUT},
  };
  for (const auto& [status, expectedResult] : cases) {
    SCOPED_TRACE(static_cast<int>(status));
    auto work = c10::make_intrusive<TestWork>();
    work->setStatus(TorchWork::WorkStatus::INPROGRESS);
    work->enableTerminalStatusProducer();
    work->setWaitAction(throwLocalWaitFailure);
    auto wrapper = c10::make_intrusive<WorkWrapper>(work);
    auto resultFuture = wrapper->getFutureResult();

    EXPECT_THROW(wrapper->wait(kNoTimeout), c10::DistBackendError);
    EXPECT_FALSE(resultFuture->completed());
    work->setStatus(status);
    resultFuture->wait();

    EXPECT_EQ(getWorkResult(resultFuture), expectedResult);
    EXPECT_FALSE(wrapper->isSuccess());
    EXPECT_NE(wrapper->exception(), nullptr);
  }
}

TEST(WorkWrapperTest, ConcurrentHookInstallationCompletesResultOnce) {
  constexpr int kIterations = 200;
  for (int i = 0; i < kIterations; ++i) {
    SpinRendezvous rendezvous;
    auto work = c10::make_intrusive<TestWork>();
    work->setStatus(TorchWork::WorkStatus::INPROGRESS);
    work->enableTerminalStatusProducer();
    auto wrapper = c10::make_intrusive<WorkWrapper>(work);
    c10::intrusive_ptr<c10::ivalue::Future> resultFuture;
    std::exception_ptr threadException;

    std::thread installer([&]() {
      rendezvous.arriveAndWait();
      try {
        resultFuture = wrapper->getFutureResult();
      } catch (...) {
        threadException = std::current_exception();
      }
    });
    rendezvous.arriveAndWait();
    work->setStatus(TorchWork::WorkStatus::COMPLETED);
    installer.join();

    ASSERT_EQ(threadException, nullptr) << "iteration " << i;
    ASSERT_NE(resultFuture, nullptr) << "iteration " << i;
    resultFuture->wait();
    EXPECT_EQ(getWorkResult(resultFuture), c10d::WorkResult::SUCCESS);
  }
}

TEST(WorkWrapperTest, WaitReturnsAfterBothRequestedFuturesComplete) {
  SpinRendezvous rendezvous;
  std::promise<void> transitioned;
  auto transitionedFuture = transitioned.get_future().share();
  auto work = c10::make_intrusive<TestWork>();
  work->setStatus(TorchWork::WorkStatus::INPROGRESS);
  work->enableTerminalStatusProducer();
  work->setWaitAction([&]() {
    rendezvous.arriveAndWait();
    transitionedFuture.wait();
  });
  auto wrapper = c10::make_intrusive<WorkWrapper>(work);
  auto resultFuture = wrapper->getFutureResult();

  std::thread watchdog([&]() {
    rendezvous.arriveAndWait();
    work->setStatus(TorchWork::WorkStatus::COMPLETED);
    transitioned.set_value();
  });

  EXPECT_TRUE(wrapper->wait(kNoTimeout));
  const bool tensorFutureCompleted = wrapper->getFuture()->completed();
  const bool resultFutureCompleted = resultFuture->completed();
  watchdog.join();

  EXPECT_TRUE(tensorFutureCompleted);
  EXPECT_TRUE(resultFutureCompleted);
  EXPECT_EQ(getWorkResult(resultFuture), c10d::WorkResult::SUCCESS);
}

TEST(WorkWrapperTest, EndHookDoesNotRetainWork) {
  bool destroyed = false;
  auto work = c10::make_intrusive<TestWork>(&destroyed);
  work->enableTerminalStatusProducer();
  {
    auto wrapper = c10::make_intrusive<WorkWrapper>(work);
    wrapper->getFutureResult();
  }
  work.reset();
  EXPECT_TRUE(destroyed);
}

TEST(WorkWrapperTest, TensorFutureCallbackDoesNotRunUnderBackendLock) {
  std::timed_mutex backendLock;
  auto work = c10::make_intrusive<TestWork>();
  auto wrapper = c10::make_intrusive<WorkWrapper>(work);
  std::promise<bool> callbackAcquiredLock;
  auto callbackResult = callbackAcquiredLock.get_future();
  wrapper->getFuture()->addCallback([&](c10::ivalue::Future&) {
    const bool acquired = backendLock.try_lock_for(250ms);
    if (acquired) {
      backendLock.unlock();
    }
    callbackAcquiredLock.set_value(acquired);
  });

  backendLock.lock();
  work->setStatus(TorchWork::WorkStatus::COMPLETED);
  backendLock.unlock();

  ASSERT_EQ(callbackResult.wait_for(2s), std::future_status::ready);
  EXPECT_TRUE(callbackResult.get());
}

TEST(WorkWrapperTest, ResultFutureCallbackDoesNotRunUnderBackendLock) {
  std::timed_mutex backendLock;
  auto work = c10::make_intrusive<TestWork>();
  work->enableTerminalStatusProducer();
  auto wrapper = c10::make_intrusive<WorkWrapper>(work);
  auto resultFuture = wrapper->getFutureResult();
  std::promise<bool> callbackAcquiredLock;
  auto callbackResult = callbackAcquiredLock.get_future();
  resultFuture->addCallback([&](c10::ivalue::Future&) {
    const bool acquired = backendLock.try_lock_for(250ms);
    if (acquired) {
      backendLock.unlock();
    }
    callbackAcquiredLock.set_value(acquired);
  });

  backendLock.lock();
  work->setStatus(TorchWork::WorkStatus::COMPLETED);
  backendLock.unlock();

  ASSERT_EQ(callbackResult.wait_for(2s), std::future_status::ready);
  EXPECT_TRUE(callbackResult.get());
}

TEST(WorkWrapperTest, PendingWorkRequiresTerminalStatusProducer) {
  auto work = c10::make_intrusive<TestWork>();
  work->setStatus(TorchWork::WorkStatus::INPROGRESS);
  auto wrapper = c10::make_intrusive<WorkWrapper>(work);

  try {
    wrapper->getFutureResult();
    FAIL() << "expected pending work without a producer to be rejected";
  } catch (const c10::Error& error) {
    EXPECT_NE(
        std::string(error.what())
            .find("autonomous one-shot terminal-status producer"),
        std::string::npos);
  }
}

TEST(WorkWrapperTest, OptedInPendingWorkCompletesResultFuture) {
  auto work = c10::make_intrusive<TestWork>();
  work->setStatus(TorchWork::WorkStatus::INPROGRESS);
  work->enableTerminalStatusProducer();
  auto wrapper = c10::make_intrusive<WorkWrapper>(work);
  auto resultFuture = wrapper->getFutureResult();

  expectFutureCompletes(resultFuture, [&]() {
    work->setStatus(TorchWork::WorkStatus::COMPLETED);
  });

  EXPECT_EQ(getWorkResult(resultFuture), c10d::WorkResult::SUCCESS);
}

TEST(WorkWrapperTest, TerminalWorkDoesNotRequireProducerOptIn) {
  auto work = c10::make_intrusive<TestWork>();
  work->setStatus(TorchWork::WorkStatus::TIMEDOUT);
  auto wrapper = c10::make_intrusive<WorkWrapper>(work);

  EXPECT_EQ(
      getWorkResult(wrapper->getFutureResult()), c10d::WorkResult::TIMEOUT);
}

TEST(WorkWrapperTest, WaitDrivenWorkRejectsPendingFutureThenResolves) {
  auto work = c10::make_intrusive<TestWork>();
  work->setStatus(TorchWork::WorkStatus::INPROGRESS);
  work->setWaitAction(
      [work]() { work->setStatus(TorchWork::WorkStatus::COMPLETED); });
  auto wrapper = c10::make_intrusive<WorkWrapper>(work);

  EXPECT_THROW(wrapper->getFutureResult(), c10::Error);
  EXPECT_TRUE(wrapper->wait(kNoTimeout));
  EXPECT_EQ(
      getWorkResult(wrapper->getFutureResult()), c10d::WorkResult::SUCCESS);
}

TEST(WorkWrapperTest, PlainWaitExceptionBecomesDistBackendError) {
  auto work = c10::make_intrusive<TestWork>();
  work->setStatus(TorchWork::WorkStatus::INPROGRESS);
  work->setWaitAction(throwSyntheticBackendFailure);
  auto wrapper = c10::make_intrusive<WorkWrapper>(work);

  EXPECT_THROW(wrapper->wait(kNoTimeout), c10::DistBackendError);
  ASSERT_NE(wrapper->exception(), nullptr);
  EXPECT_NE(
      getExceptionMessage(wrapper->exception())
          .find("synthetic backend failure"),
      std::string::npos);
  EXPECT_FALSE(wrapper->isSuccess());
  EXPECT_EQ(
      getWorkResult(wrapper->getFutureResult()), c10d::WorkResult::COMM_ERROR);
}

TEST(WorkWrapperTest, TerminalTimeoutWinsOverWaitException) {
  auto work = c10::make_intrusive<TestWork>();
  work->setStatus(TorchWork::WorkStatus::TIMEDOUT);
  work->setWaitAction(throwSecondaryWaitFailure);
  auto wrapper = c10::make_intrusive<WorkWrapper>(work);

  EXPECT_THROW(wrapper->wait(kNoTimeout), c10::DistBackendError);
  EXPECT_EQ(
      getWorkResult(wrapper->getFutureResult()), c10d::WorkResult::TIMEOUT);
  EXPECT_NE(
      getExceptionMessage(wrapper->exception()).find("secondary wait failure"),
      std::string::npos);
}

TEST(WorkWrapperTest, SuccessfulWorkResultSurvivesHostSyncFailure) {
  auto work = c10::make_intrusive<TestWork>();
  work->setStatus(TorchWork::WorkStatus::COMPLETED);
  work->setHostSynchronizeAction(throwSyntheticHostSyncFailure);
  auto wrapper = c10::make_intrusive<WorkWrapper>(
      work, std::vector<at::Tensor>{}, /*hostBlocking=*/true);

  EXPECT_THROW(wrapper->wait(kNoTimeout), c10::DistBackendError);
  EXPECT_EQ(
      getWorkResult(wrapper->getFutureResult()), c10d::WorkResult::SUCCESS);
  ASSERT_NE(wrapper->exception(), nullptr);
  EXPECT_NE(
      getExceptionMessage(wrapper->exception())
          .find("synthetic host sync failure"),
      std::string::npos);
  EXPECT_FALSE(wrapper->isSuccess());
}

} // namespace torch::comms::test
