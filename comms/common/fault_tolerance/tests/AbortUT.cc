// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/common/fault_tolerance/Abort.h"

namespace comms::fault_tolerance::testing {

using ::comms::fault_tolerance::Abort;
using ::comms::fault_tolerance::AbortReason;

namespace {

template <typename Predicate>
bool eventually(
    Predicate&& predicate,
    std::chrono::milliseconds timeout = std::chrono::seconds{1}) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (!predicate()) {
    if (std::chrono::steady_clock::now() >= deadline) {
      return false;
    }
    std::this_thread::yield();
  }
  return true;
}

void waitFor(std::chrono::milliseconds duration) {
  const auto deadline = std::chrono::steady_clock::now() + duration;
  while (std::chrono::steady_clock::now() < deadline) {
    std::this_thread::yield();
  }
}

} // namespace

TEST(AbortTest, enabledDefaultNotAbort) {
  Abort abort{/*enabled=*/true};
  EXPECT_FALSE(abort.isAborted());
  EXPECT_EQ(abort.reason(), AbortReason::NONE);
}

TEST(AbortTest, disabledNoopDefaultNotAbort) {
  Abort abort{/*enabled=*/false};
  EXPECT_FALSE(abort.isAborted());
  EXPECT_EQ(abort.reason(), AbortReason::NONE);
}

TEST(AbortTest, enabled) {
  Abort abort{/*enabled=*/true};
  ASSERT_TRUE(abort.isEnabled());

  abort.setAbort();

  EXPECT_TRUE(abort.isAborted());
  EXPECT_EQ(abort.reason(), AbortReason::ABORTED);
}

TEST(AbortTest, disabledNoop) {
  Abort abort{/*enabled=*/false};
  ASSERT_FALSE(abort.isEnabled());

  abort.setAbort();

  EXPECT_FALSE(abort.isAborted());
}

TEST(AbortTest, DoubleAbort) {
  Abort abort{/*enabled=*/true};

  abort.setAbort();
  abort.setAbort();

  EXPECT_TRUE(abort.isAborted());
}

TEST(AbortTest, MultipleAbortTest) {
  Abort abort{/*enabled=*/true};

  auto timeout = std::chrono::milliseconds(2000);
  std::atomic<bool> start{false};
  std::atomic<bool> abortMarked{false};

  auto startTs = std::chrono::high_resolution_clock::now();

  std::thread producer([&]() {
    while (!start.load()) {
      ASSERT_LE((std::chrono::high_resolution_clock::now() - startTs), timeout)
          << "producer: test case did not start";
    }

    std::this_thread::yield();

    ASSERT_FALSE(abort.isAborted());
    abort.setAbort();
    abortMarked.store(true);
  });
  std::thread consumer([&]() {
    while (!start.load()) {
      ASSERT_LE((std::chrono::high_resolution_clock::now() - startTs), timeout)
          << "consumer: test case did not start";
    }

    bool abortMarkedLocal = false;
    while (std::chrono::high_resolution_clock::now() - startTs < timeout) {
      abortMarkedLocal = abortMarked.load();
      bool aborted = abort.isAborted();
      if (abortMarkedLocal) {
        EXPECT_TRUE(aborted);
      } else {
        continue;
      }
    }
    ASSERT_TRUE(abortMarkedLocal) << "consumer: did not consume";
  });

  start.store(true);
  producer.join();
  consumer.join();
}

TEST(AbortFactoryTest, enabled) {
  auto abort = ::comms::fault_tolerance::createAbort(/*enabled=*/true);
  ASSERT_TRUE(abort->isEnabled());

  abort->setAbort();

  EXPECT_TRUE(abort->isAborted());
}

TEST(AbortFactoryTest, disabledNoop) {
  auto abort = ::comms::fault_tolerance::createAbort(/*enabled=*/false);
  ASSERT_FALSE(abort->isEnabled());

  abort->setAbort();

  EXPECT_FALSE(abort->isAborted());
}

TEST(AbortFactoryTest, DisabledSingletonDoesNotStoreAbortInfo) {
  auto first = ::comms::fault_tolerance::createAbort(/*enabled=*/false);
  auto second = ::comms::fault_tolerance::createAbort(/*enabled=*/false);

  ASSERT_EQ(first.get(), second.get());

  EXPECT_FALSE(
      first->setAbort(AbortReason::NETWORK_ERROR, "ignored disabled abort"));

  EXPECT_FALSE(first->isAborted());
  EXPECT_EQ(first->getAbortInfo(), std::nullopt);
  EXPECT_FALSE(second->isAborted());
  EXPECT_EQ(second->getAbortInfo(), std::nullopt);
}

TEST(AbortTest, timeoutNotExpired) {
  Abort abort{/*enabled=*/true};

  abort.startTimeout(std::chrono::milliseconds(1000));

  // Test should return false immediately as timeout hasn't expired
  EXPECT_FALSE(abort.isAborted());
}

TEST(AbortTest, timeoutExpired) {
  Abort abort{/*enabled=*/true};

  abort.startTimeout(std::chrono::milliseconds(1));

  // Test should return true as timeout has expired
  EXPECT_TRUE(eventually([&]() { return abort.isAborted(); }));
}

TEST(AbortTest, timeoutDisabledNoop) {
  Abort abort{/*enabled=*/false};

  abort.startTimeout(std::chrono::milliseconds(1));

  // Test should return false as abort is disabled:w
  EXPECT_FALSE(abort.isAborted());
  EXPECT_FALSE(abort.isTimedOut());
}

TEST(AbortTest, explicitSetTakesPrecedenceOverTimeout) {
  Abort abort{/*enabled=*/true};

  abort.startTimeout(std::chrono::milliseconds(10000));
  abort.setAbort();

  // Test should return true immediately due to explicit set
  EXPECT_TRUE(abort.isAborted());
}

TEST(AbortTest, timeoutAndExplicitSetBothTrue) {
  Abort abort{/*enabled=*/true};

  abort.startTimeout(std::chrono::milliseconds(1));
  abort.setAbort();

  // Test should return true (both conditions are true)
  EXPECT_TRUE(abort.isAborted());
}

TEST(AbortTest, explicitAbortWinsOverExpiredTimeout) {
  Abort abort{/*enabled=*/true};

  abort.startTimeout(std::chrono::milliseconds(1));
  EXPECT_TRUE(eventually([&]() {
    return abort.getTimeRemaining() == std::chrono::milliseconds{0};
  }));
  abort.setAbort();

  EXPECT_TRUE(abort.isAborted());
  EXPECT_FALSE(abort.isTimedOut());
}

TEST(AbortTest, timeoutWinsBeforeExplicitAbort) {
  Abort abort{/*enabled=*/true};

  abort.startTimeout(std::chrono::milliseconds(1));

  EXPECT_TRUE(eventually([&]() { return abort.isTimedOut(); }));

  abort.setAbort();

  EXPECT_TRUE(abort.isAborted());
  EXPECT_TRUE(abort.isTimedOut());
}

TEST(AbortTest, multipleTimeoutCalls) {
  Abort abort{/*enabled=*/true};

  abort.startTimeout(std::chrono::milliseconds(10000));

  EXPECT_FALSE(abort.isAborted());

  abort.startTimeout(std::chrono::milliseconds(1));

  // Test should return true as the shorter timeout has expired
  EXPECT_TRUE(eventually([&]() { return abort.isAborted(); }));
}

TEST(AbortTest, timeoutThreadSafety) {
  Abort abort{/*enabled=*/true};

  std::atomic<bool> timeoutSet{false};
  std::atomic<bool> timeoutDetected{false};
  std::atomic<int> testCallCount{0};

  // Thread 1: Sets timeout
  std::thread timeoutSetter([&]() {
    abort.startTimeout(std::chrono::milliseconds(50));
    timeoutSet.store(true);
  });

  // Thread 2: Continuously tests for abort
  std::thread tester([&]() {
    while (!timeoutSet.load()) {
      std::this_thread::yield();
    }

    auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start <
           std::chrono::milliseconds(100)) {
      testCallCount.fetch_add(1);
      if (abort.isAborted()) {
        timeoutDetected.store(true);
        break;
      }
      std::this_thread::yield();
    }
  });

  timeoutSetter.join();
  tester.join();

  EXPECT_TRUE(timeoutSet.load());
  EXPECT_TRUE(timeoutDetected.load());
  EXPECT_GT(testCallCount.load(), 0);
}

TEST(AbortTest, cancelTimeoutBeforeExpiry) {
  Abort abort{/*enabled=*/true};

  abort.startTimeout(std::chrono::milliseconds(100));
  abort.cancelTimeout();

  waitFor(std::chrono::milliseconds(150));

  // Test should return false as timeout was cancelled
  EXPECT_FALSE(abort.isAborted());
}

TEST(AbortTest, cancelTimeoutAfterExpiry) {
  Abort abort{/*enabled=*/true};

  abort.startTimeout(std::chrono::milliseconds(1));

  // Verify timeout has expired
  EXPECT_TRUE(eventually([&]() { return abort.isAborted(); }));

  abort.cancelTimeout();

  // CancelTimeout does not reset timeout state
  EXPECT_TRUE(abort.isAborted());
}

TEST(AbortTest, cancelTimeoutDisabledNoop) {
  Abort abort{/*enabled=*/false};

  abort.startTimeout(std::chrono::milliseconds(1));
  abort.cancelTimeout();

  // Test should return false as abort is disabled
  EXPECT_FALSE(abort.isAborted());
}

TEST(AbortTest, cancelTimeoutAfterExplicitSet) {
  Abort abort{/*enabled=*/true};

  abort.startTimeout(std::chrono::milliseconds(1000));
  abort.setAbort();

  EXPECT_TRUE(abort.isAborted());

  abort.cancelTimeout();

  // Test should still return true due to explicit set
  EXPECT_TRUE(abort.isAborted());
}

TEST(AbortTest, setTimeoutAfterCancel) {
  Abort abort{/*enabled=*/true};

  abort.startTimeout(std::chrono::milliseconds(10000));
  abort.cancelTimeout();

  // Verify timeout is cancelled
  EXPECT_FALSE(abort.isAborted());

  abort.startTimeout(std::chrono::milliseconds(1));

  // Test should return true as new timeout has expired
  EXPECT_TRUE(eventually([&]() { return abort.isAborted(); }));
}

TEST(AbortTest, multipleCancelTimeoutCalls) {
  Abort abort{/*enabled=*/true};

  abort.startTimeout(std::chrono::milliseconds(100));
  abort.cancelTimeout();
  abort.cancelTimeout();
  abort.cancelTimeout();

  waitFor(std::chrono::milliseconds(150));

  // Test should return false as timeout was cancelled
  EXPECT_FALSE(abort.isAborted());
}

TEST(AbortTest, cancelTimeoutThreadSafety) {
  Abort abort{/*enabled=*/true};

  std::atomic<bool> timeoutSet{false};
  std::atomic<bool> timeoutCancelled{false};
  std::atomic<bool> timeoutDetected{false};
  std::atomic<int> testCallCount{0};

  // Thread 1: Sets timeout
  std::thread timeoutSetter([&]() {
    abort.startTimeout(std::chrono::milliseconds(100));
    timeoutSet.store(true);
  });

  // Thread 2: Cancels timeout after a brief delay
  std::thread timeoutCanceller([&]() {
    while (!timeoutSet.load()) {
      std::this_thread::yield();
    }

    // Wait a bit then cancel
    waitFor(std::chrono::milliseconds(10));
    abort.cancelTimeout();
    timeoutCancelled.store(true);
  });

  // Thread 3: Continuously tests for abort
  std::thread tester([&]() {
    while (!timeoutSet.load()) {
      std::this_thread::yield();
    }

    // Test for timeout for a reasonable duration
    auto start = std::chrono::steady_clock::now();

    while (std::chrono::steady_clock::now() - start <
           std::chrono::milliseconds(200)) {
      testCallCount.fetch_add(1);
      if (abort.isAborted()) {
        timeoutDetected.store(true);
        break;
      }
      std::this_thread::yield();
    }
  });

  timeoutSetter.join();
  timeoutCanceller.join();
  tester.join();

  EXPECT_TRUE(timeoutSet.load());
  EXPECT_TRUE(timeoutCancelled.load());
  // Timeout should not be detected as it was cancelled
  EXPECT_FALSE(timeoutDetected.load());
  EXPECT_GT(testCallCount.load(), 0);
}

TEST(AbortTest, hasTimeoutInitiallyFalse) {
  Abort abort{/*enabled=*/true};
  EXPECT_FALSE(abort.isTimeoutActive());
}

TEST(AbortTest, hasTimeoutTrueAfterSet) {
  Abort abort{/*enabled=*/true};
  abort.startTimeout(std::chrono::milliseconds(1000));
  EXPECT_TRUE(abort.isTimeoutActive());
}

TEST(AbortTest, hasTimeoutFalseAfterCancel) {
  Abort abort{/*enabled=*/true};
  abort.startTimeout(std::chrono::milliseconds(30000));
  EXPECT_TRUE(abort.isTimeoutActive());

  abort.cancelTimeout();
  EXPECT_FALSE(abort.isTimeoutActive());
}

TEST(AbortTest, hasTimeoutDisabledNoop) {
  Abort abort{/*enabled=*/false};
  abort.startTimeout(std::chrono::milliseconds(1000));
  EXPECT_FALSE(abort.isTimeoutActive());
}

TEST(AbortTest, timedOutInitiallyFalse) {
  Abort abort{/*enabled=*/true};
  EXPECT_FALSE(abort.isTimedOut());
}

TEST(AbortTest, timedOutFalseBeforeExpiry) {
  Abort abort{/*enabled=*/true};
  abort.startTimeout(std::chrono::milliseconds(1000));
  EXPECT_FALSE(abort.isTimedOut());
}

TEST(AbortTest, timedOutTrueAfterExpiry) {
  Abort abort{/*enabled=*/true};
  abort.startTimeout(std::chrono::milliseconds(1));

  EXPECT_TRUE(eventually([&]() { return abort.isTimedOut(); }));
}

TEST(AbortTest, timedOutFalseForExplicitSet) {
  Abort abort{/*enabled=*/true};
  abort.setAbort();
  EXPECT_TRUE(abort.isAborted());
  EXPECT_FALSE(abort.isTimedOut());
}

TEST(AbortTest, setAbortTimedOutRecordsTimeout) {
  Abort abort{/*enabled=*/true};

  EXPECT_TRUE(abort.setAbort(AbortReason::TIMED_OUT));

  EXPECT_TRUE(abort.isAborted());
  EXPECT_TRUE(abort.isTimedOut());
  EXPECT_EQ(abort.reason(), AbortReason::TIMED_OUT);
}

TEST(AbortTest, reasonOnlySetAbortIsPreserved) {
  Abort abort{/*enabled=*/true};

  abort.setAbort(AbortReason::NETWORK_ERROR);

  EXPECT_EQ(
      abort.getAbortInfo(),
      (AbortInfo{.reason = AbortReason::NETWORK_ERROR, .context = ""}));
}

TEST(AbortTest, setAbortRejectsNone) {
  Abort abort{/*enabled=*/true};

  EXPECT_THROW(abort.setAbort(AbortReason::NONE), std::invalid_argument);
  EXPECT_FALSE(abort.isAborted());
}

TEST(AbortTest, setAbortRejectsUnknownReason) {
  Abort abort{/*enabled=*/true};

  EXPECT_THROW(
      abort.setAbort(static_cast<AbortReason>(99)), std::invalid_argument);
  EXPECT_FALSE(abort.isAborted());

  abort.setAbort(AbortReason::ABORTED);

  EXPECT_TRUE(abort.isAborted());
  EXPECT_FALSE(abort.isTimedOut());
}

TEST(AbortTest, setAbortRejectsInvalidReasonsWhenDisabled) {
  Abort abort{/*enabled=*/false};

  EXPECT_THROW(abort.setAbort(AbortReason::NONE), std::invalid_argument);
  EXPECT_THROW(
      abort.setAbort(static_cast<AbortReason>(99)), std::invalid_argument);
  EXPECT_EQ(abort.getAbortInfo(), std::nullopt);
}

TEST(AbortTest, abortInfoInitiallyEmpty) {
  Abort abort{/*enabled=*/true};

  EXPECT_EQ(abort.getAbortInfo(), std::nullopt);
}

TEST(AbortTest, abortInfoDisabledRemainsEmpty) {
  Abort abort{/*enabled=*/false};

  EXPECT_FALSE(abort.setAbort(AbortReason::ABORTED, "ignored"));

  EXPECT_EQ(abort.getAbortInfo(), std::nullopt);
}

TEST(AbortTest, abortInfoRecordsEveryTerminalReasonAndContext) {
  const std::vector reasons{
      AbortReason::ABORTED,
      AbortReason::TIMED_OUT,
      AbortReason::BOOTSTRAP_POLL,
      AbortReason::NETWORK_ERROR,
      AbortReason::INTERNAL_ERROR,
      AbortReason::IBRC_PROXY_TIMEOUT,
  };

  for (const auto reason : reasons) {
    Abort abort{/*enabled=*/true};
    EXPECT_TRUE(abort.setAbort(reason, "details"));
    EXPECT_EQ(
        abort.getAbortInfo(),
        (AbortInfo{.reason = reason, .context = "details"}));
  }
}

TEST(AbortTest, abortInfoPreservesEmbeddedNullInContext) {
  Abort abort{/*enabled=*/true};
  const std::string context{"prefix\0suffix", 13};

  EXPECT_TRUE(abort.setAbort(AbortReason::INTERNAL_ERROR, context));
  EXPECT_EQ(
      abort.getAbortInfo(),
      (AbortInfo{
          .reason = AbortReason::INTERNAL_ERROR,
          .context = context,
      }));
}

TEST(AbortTest, firstTerminalReasonAndContextWinTogether) {
  Abort abort{/*enabled=*/true};

  EXPECT_TRUE(abort.setAbort(AbortReason::BOOTSTRAP_POLL, "first"));
  EXPECT_FALSE(abort.setAbort(AbortReason::INTERNAL_ERROR, "second"));

  EXPECT_EQ(
      abort.getAbortInfo(),
      (AbortInfo{
          .reason = AbortReason::BOOTSTRAP_POLL,
          .context = "first",
      }));
}

TEST(AbortTest, expiredTimeoutRecordsContext) {
  Abort abort{/*enabled=*/true};
  abort.startTimeout(std::chrono::milliseconds{0});

  EXPECT_EQ(
      abort.getAbortInfo(),
      (AbortInfo{
          .reason = AbortReason::TIMED_OUT, .context = "timeout expired"}));
}

TEST(AbortTest, concurrentWinnerKeepsMatchingContext) {
  for (int iteration = 0; iteration < 100; ++iteration) {
    Abort abort{/*enabled=*/true};
    std::thread network(
        [&]() { abort.setAbort(AbortReason::NETWORK_ERROR, "network"); });
    std::thread internal(
        [&]() { abort.setAbort(AbortReason::INTERNAL_ERROR, "internal"); });
    network.join();
    internal.join();

    const auto info = abort.getAbortInfo();
    ASSERT_TRUE(info.has_value());
    if (info->reason == AbortReason::NETWORK_ERROR) {
      EXPECT_EQ(info->context, "network");
    } else {
      EXPECT_EQ(info->reason, AbortReason::INTERNAL_ERROR);
      EXPECT_EQ(info->context, "internal");
    }
  }
}

TEST(AbortTest, firstTerminalReasonWins) {
  Abort abort{/*enabled=*/true};

  EXPECT_TRUE(abort.setAbort(AbortReason::TIMED_OUT));
  EXPECT_FALSE(abort.setAbort(AbortReason::ABORTED));

  EXPECT_TRUE(abort.isAborted());
  EXPECT_TRUE(abort.isTimedOut());
  EXPECT_EQ(abort.reason(), AbortReason::TIMED_OUT);
}

// `firstTerminalReasonWins` covers the sequential case. This is the contended
// one: many threads recording different reasons at once must still leave
// exactly one terminal reason behind, and it must not move afterwards. A reason
// that could be overwritten would make the first-writer log name a fault that
// is no longer the one being reported.
TEST(AbortTest, concurrentWritersLeaveOneStableReason) {
  constexpr int kThreadsPerReason = 16;
  Abort abort{/*enabled=*/true};

  std::atomic<bool> go{false};
  std::vector<std::thread> writers;
  writers.reserve(kThreadsPerReason * 2);
  for (int i = 0; i < kThreadsPerReason * 2; ++i) {
    const auto reason =
        (i % 2 == 0) ? AbortReason::ABORTED : AbortReason::TIMED_OUT;
    writers.emplace_back([&abort, &go, reason] {
      while (!go.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      abort.setAbort(reason);
    });
  }
  go.store(true, std::memory_order_release);
  for (auto& t : writers) {
    t.join();
  }

  const auto reason = abort.reason();
  EXPECT_TRUE(
      reason == AbortReason::ABORTED || reason == AbortReason::TIMED_OUT);
  EXPECT_TRUE(abort.isAborted());
  for (int i = 0; i < 1000; ++i) {
    ASSERT_EQ(abort.reason(), reason);
  }
}

TEST(AbortTest, abortReasonToString) {
  EXPECT_EQ(abortReasonToString(AbortReason::NONE), "none");
  EXPECT_EQ(abortReasonToString(AbortReason::ABORTED), "aborted");
  EXPECT_EQ(abortReasonToString(AbortReason::TIMED_OUT), "timed_out");
  EXPECT_EQ(abortReasonToString(AbortReason::BOOTSTRAP_POLL), "bootstrap_poll");
  EXPECT_EQ(abortReasonToString(AbortReason::NETWORK_ERROR), "network_error");
  EXPECT_EQ(abortReasonToString(AbortReason::INTERNAL_ERROR), "internal_error");
  EXPECT_EQ(
      abortReasonToString(AbortReason::IBRC_PROXY_TIMEOUT),
      "ibrc_proxy_timeout");
  EXPECT_EQ(abortReasonToString(static_cast<AbortReason>(99)), "unknown");
}

TEST(AbortTest, abortInfoReasonStringIsComputedFromReason) {
  const AbortInfo info{
      .reason = AbortReason::NETWORK_ERROR,
      .context = "",
  };

  EXPECT_EQ(info.reasonString(), "network_error");
}

TEST(AbortTest, timeRemainingNoTimeout) {
  Abort abort{/*enabled=*/true};
  EXPECT_EQ(abort.getTimeRemaining(), std::chrono::milliseconds{-1});
}

TEST(AbortTest, timeRemainingDisabled) {
  Abort abort{/*enabled=*/false};
  abort.startTimeout(std::chrono::milliseconds(1000));
  EXPECT_EQ(abort.getTimeRemaining(), std::chrono::milliseconds{-1});
}

TEST(AbortTest, timeRemainingAfterSet) {
  Abort abort{/*enabled=*/true};
  abort.startTimeout(std::chrono::milliseconds(100));

  auto remaining = abort.getTimeRemaining();
  EXPECT_GT(remaining, std::chrono::milliseconds{0});
  EXPECT_LE(remaining, std::chrono::milliseconds{100});
}

TEST(AbortTest, timeRemainingDecreases) {
  Abort abort{/*enabled=*/true};
  abort.startTimeout(std::chrono::milliseconds(100));

  auto remaining1 = abort.getTimeRemaining();
  EXPECT_TRUE(
      eventually([&]() { return abort.getTimeRemaining() < remaining1; }));
  auto remaining2 = abort.getTimeRemaining();

  EXPECT_LT(remaining2, remaining1);
}

TEST(AbortTest, timeRemainingZeroAfterExpiry) {
  Abort abort{/*enabled=*/true};
  abort.startTimeout(std::chrono::milliseconds(1));

  EXPECT_TRUE(eventually([&]() {
    return abort.getTimeRemaining() == std::chrono::milliseconds{0};
  }));
}

TEST(AbortTest, timeRemainingAfterCancel) {
  Abort abort{/*enabled=*/true};
  abort.startTimeout(std::chrono::milliseconds(1000));
  abort.cancelTimeout();
  EXPECT_EQ(abort.getTimeRemaining(), std::chrono::milliseconds{-1});
}

TEST(AbortTest, defaultTimeoutInitiallyUnset) {
  Abort abort{/*enabled=*/true};
  EXPECT_EQ(abort.getDefaultTimeout(), std::nullopt);
}

TEST(AbortTest, defaultTimeoutSetAndGet) {
  Abort abort{/*enabled=*/true};
  constexpr std::chrono::milliseconds kDuration{750};
  abort.setDefaultTimeout(kDuration);
  EXPECT_EQ(abort.getDefaultTimeout(), kDuration);
}

TEST(AbortTest, defaultTimeoutMutable) {
  Abort abort{/*enabled=*/true};
  abort.setDefaultTimeout(std::chrono::milliseconds(100));
  abort.setDefaultTimeout(std::chrono::milliseconds(5000));
  EXPECT_EQ(abort.getDefaultTimeout(), std::chrono::milliseconds(5000));
}

TEST(AbortTest, defaultTimeoutDisabledSetterNoop) {
  Abort abort{/*enabled=*/false};
  abort.setDefaultTimeout(std::chrono::milliseconds(1000));
  EXPECT_EQ(abort.getDefaultTimeout(), std::nullopt);
}

} // namespace comms::fault_tolerance::testing
