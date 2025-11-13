// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <chrono>
#include <thread>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/ctran/utils/Abort.h"

namespace ctran::testing {

using ::ctran::utils::Abort;

TEST(AbortTest, enabledDefaultNotAbort) {
  Abort abort{/*enabled=*/true};
  EXPECT_FALSE(abort.Test());
}

TEST(AbortTest, disabledNoopDefaultNotAbort) {
  Abort abort{/*enabled=*/false};
  EXPECT_FALSE(abort.Test());
}

TEST(AbortTest, enabled) {
  Abort abort{/*enabled=*/true};
  ASSERT_TRUE(abort.Enabled());

  abort.Set();

  EXPECT_TRUE(abort.Test());
}

TEST(AbortTest, disabledNoop) {
  Abort abort{/*enabled=*/false};
  ASSERT_FALSE(abort.Enabled());

  abort.Set();

  EXPECT_FALSE(abort.Test());
}

TEST(AbortTest, DoubleAbort) {
  Abort abort{/*enabled=*/true};

  abort.Set();
  abort.Set();

  EXPECT_TRUE(abort.Test());
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

    // delay a bit to allow consumer start working first
    std::this_thread::sleep_for(std::chrono::microseconds(20));

    ASSERT_FALSE(abort.Test());
    abort.Set();
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
      bool aborted = abort.Test();
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
  auto abort = ::ctran::utils::createAbort(/*enabled=*/true);
  ASSERT_TRUE(abort->Enabled());

  abort->Set();

  EXPECT_TRUE(abort->Test());
}

TEST(AbortFactoryTest, disabledNoop) {
  auto abort = ::ctran::utils::createAbort(/*enabled=*/false);
  ASSERT_FALSE(abort->Enabled());

  abort->Set();

  EXPECT_FALSE(abort->Test());
}

TEST(AbortTest, timeoutNotExpired) {
  Abort abort{/*enabled=*/true};

  abort.SetTimeout(std::chrono::milliseconds(1000));

  // Test should return false immediately as timeout hasn't expired
  EXPECT_FALSE(abort.Test());
}

TEST(AbortTest, timeoutExpired) {
  Abort abort{/*enabled=*/true};

  abort.SetTimeout(std::chrono::milliseconds(1));

  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // Test should return true as timeout has expired
  EXPECT_TRUE(abort.Test());
}

TEST(AbortTest, timeoutDisabledNoop) {
  Abort abort{/*enabled=*/false};

  abort.SetTimeout(std::chrono::milliseconds(1));

  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // Test should return false as abort is disabled:w
  EXPECT_FALSE(abort.Test());
}

TEST(AbortTest, explicitSetTakesPrecedenceOverTimeout) {
  Abort abort{/*enabled=*/true};

  abort.SetTimeout(std::chrono::milliseconds(10000));
  abort.Set();

  // Test should return true immediately due to explicit set
  EXPECT_TRUE(abort.Test());
}

TEST(AbortTest, timeoutAndExplicitSetBothTrue) {
  Abort abort{/*enabled=*/true};

  abort.SetTimeout(std::chrono::milliseconds(1));
  abort.Set();

  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // Test should return true (both conditions are true)
  EXPECT_TRUE(abort.Test());
}

TEST(AbortTest, multipleTimeoutCalls) {
  Abort abort{/*enabled=*/true};

  abort.SetTimeout(std::chrono::milliseconds(10000));

  EXPECT_FALSE(abort.Test());

  abort.SetTimeout(std::chrono::milliseconds(1));

  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // Test should return true as the shorter timeout has expired
  EXPECT_TRUE(abort.Test());
}

TEST(AbortTest, timeoutThreadSafety) {
  Abort abort{/*enabled=*/true};

  std::atomic<bool> timeoutSet{false};
  std::atomic<bool> timeoutDetected{false};
  std::atomic<int> testCallCount{0};

  // Thread 1: Sets timeout
  std::thread timeoutSetter([&]() {
    abort.SetTimeout(std::chrono::milliseconds(50));
    timeoutSet.store(true);
  });

  // Thread 2: Continuously tests for abort
  std::thread tester([&]() {
    while (!timeoutSet.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    auto start = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - start <
           std::chrono::milliseconds(100)) {
      testCallCount.fetch_add(1);
      if (abort.Test()) {
        timeoutDetected.store(true);
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
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

  abort.SetTimeout(std::chrono::milliseconds(100));
  abort.CancelTimeout();

  std::this_thread::sleep_for(std::chrono::milliseconds(150));

  // Test should return false as timeout was cancelled
  EXPECT_FALSE(abort.Test());
}

TEST(AbortTest, cancelTimeoutAfterExpiry) {
  Abort abort{/*enabled=*/true};

  abort.SetTimeout(std::chrono::milliseconds(1));

  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // Verify timeout has expired
  EXPECT_TRUE(abort.Test());

  abort.CancelTimeout();

  // CancelTimeout does not reset timeout state
  EXPECT_TRUE(abort.Test());
}

TEST(AbortTest, cancelTimeoutDisabledNoop) {
  Abort abort{/*enabled=*/false};

  abort.SetTimeout(std::chrono::milliseconds(1));
  abort.CancelTimeout();

  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // Test should return false as abort is disabled
  EXPECT_FALSE(abort.Test());
}

TEST(AbortTest, cancelTimeoutAfterExplicitSet) {
  Abort abort{/*enabled=*/true};

  abort.SetTimeout(std::chrono::milliseconds(1000));
  abort.Set();

  EXPECT_TRUE(abort.Test());

  abort.CancelTimeout();

  // Test should still return true due to explicit set
  EXPECT_TRUE(abort.Test());
}

TEST(AbortTest, setTimeoutAfterCancel) {
  Abort abort{/*enabled=*/true};

  abort.SetTimeout(std::chrono::milliseconds(10000));
  abort.CancelTimeout();

  // Verify timeout is cancelled
  EXPECT_FALSE(abort.Test());

  abort.SetTimeout(std::chrono::milliseconds(1));

  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  // Test should return true as new timeout has expired
  EXPECT_TRUE(abort.Test());
}

TEST(AbortTest, multipleCancelTimeoutCalls) {
  Abort abort{/*enabled=*/true};

  abort.SetTimeout(std::chrono::milliseconds(100));
  abort.CancelTimeout();
  abort.CancelTimeout();
  abort.CancelTimeout();

  std::this_thread::sleep_for(std::chrono::milliseconds(150));

  // Test should return false as timeout was cancelled
  EXPECT_FALSE(abort.Test());
}

TEST(AbortTest, cancelTimeoutThreadSafety) {
  Abort abort{/*enabled=*/true};

  std::atomic<bool> timeoutSet{false};
  std::atomic<bool> timeoutCancelled{false};
  std::atomic<bool> timeoutDetected{false};
  std::atomic<int> testCallCount{0};

  // Thread 1: Sets timeout
  std::thread timeoutSetter([&]() {
    abort.SetTimeout(std::chrono::milliseconds(100));
    timeoutSet.store(true);
  });

  // Thread 2: Cancels timeout after a brief delay
  std::thread timeoutCanceller([&]() {
    while (!timeoutSet.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Wait a bit then cancel
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    abort.CancelTimeout();
    timeoutCancelled.store(true);
  });

  // Thread 3: Continuously tests for abort
  std::thread tester([&]() {
    while (!timeoutSet.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Test for timeout for a reasonable duration
    auto start = std::chrono::steady_clock::now();

    while (std::chrono::steady_clock::now() - start <
           std::chrono::milliseconds(200)) {
      testCallCount.fetch_add(1);
      if (abort.Test()) {
        timeoutDetected.store(true);
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
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
  EXPECT_FALSE(abort.HasTimeout());
}

TEST(AbortTest, hasTimeoutTrueAfterSet) {
  Abort abort{/*enabled=*/true};
  abort.SetTimeout(std::chrono::milliseconds(1000));
  EXPECT_TRUE(abort.HasTimeout());
}

TEST(AbortTest, hasTimeoutFalseAfterCancel) {
  Abort abort{/*enabled=*/true};
  abort.SetTimeout(std::chrono::milliseconds(30000));
  EXPECT_TRUE(abort.HasTimeout());

  abort.CancelTimeout();
  EXPECT_FALSE(abort.HasTimeout());
}

TEST(AbortTest, hasTimeoutDisabledNoop) {
  Abort abort{/*enabled=*/false};
  abort.SetTimeout(std::chrono::milliseconds(1000));
  EXPECT_FALSE(abort.HasTimeout());
}

TEST(AbortTest, timedOutInitiallyFalse) {
  Abort abort{/*enabled=*/true};
  EXPECT_FALSE(abort.TimedOut());
}

TEST(AbortTest, timedOutFalseBeforeExpiry) {
  Abort abort{/*enabled=*/true};
  abort.SetTimeout(std::chrono::milliseconds(1000));
  EXPECT_FALSE(abort.TimedOut());
}

TEST(AbortTest, timedOutTrueAfterExpiry) {
  Abort abort{/*enabled=*/true};
  abort.SetTimeout(std::chrono::milliseconds(1));

  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_TRUE(abort.TimedOut());
}

TEST(AbortTest, timedOutFalseForExplicitSet) {
  Abort abort{/*enabled=*/true};
  abort.Set();
  EXPECT_TRUE(abort.Test());
  EXPECT_FALSE(abort.TimedOut());
}

TEST(AbortTest, timeRemainingNoTimeout) {
  Abort abort{/*enabled=*/true};
  EXPECT_EQ(abort.TimeRemaining(), std::chrono::milliseconds{-1});
}

TEST(AbortTest, timeRemainingDisabled) {
  Abort abort{/*enabled=*/false};
  abort.SetTimeout(std::chrono::milliseconds(1000));
  EXPECT_EQ(abort.TimeRemaining(), std::chrono::milliseconds{-1});
}

TEST(AbortTest, timeRemainingAfterSet) {
  Abort abort{/*enabled=*/true};
  abort.SetTimeout(std::chrono::milliseconds(100));

  auto remaining = abort.TimeRemaining();
  EXPECT_GT(remaining, std::chrono::milliseconds{0});
  EXPECT_LE(remaining, std::chrono::milliseconds{100});
}

TEST(AbortTest, timeRemainingDecreases) {
  Abort abort{/*enabled=*/true};
  abort.SetTimeout(std::chrono::milliseconds(100));

  auto remaining1 = abort.TimeRemaining();
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  auto remaining2 = abort.TimeRemaining();

  EXPECT_LT(remaining2, remaining1);
}

TEST(AbortTest, timeRemainingZeroAfterExpiry) {
  Abort abort{/*enabled=*/true};
  abort.SetTimeout(std::chrono::milliseconds(1));

  std::this_thread::sleep_for(std::chrono::milliseconds(10));

  EXPECT_EQ(abort.TimeRemaining(), std::chrono::milliseconds{0});
}

TEST(AbortTest, timeRemainingAfterCancel) {
  Abort abort{/*enabled=*/true};
  abort.SetTimeout(std::chrono::milliseconds(1000));
  abort.CancelTimeout();
  EXPECT_EQ(abort.TimeRemaining(), std::chrono::milliseconds{-1});
}

} // namespace ctran::testing
