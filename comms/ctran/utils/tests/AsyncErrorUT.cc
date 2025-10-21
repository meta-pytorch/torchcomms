// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/utils/AsyncError.h"
#include "comms/ctran/utils/Exception.h"

using ctran::utils::AsyncError;
using ctran::utils::Exception;

class AsyncErrorTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST(CtranAsyncErrorTest, SetAndGet) {
  constexpr auto numThreads = 10;
  std::vector<std::thread> threads;

  auto asyncErr = std::make_shared<AsyncError>(false, "AsyncErrorTest");
  for (auto i = 0; i < numThreads; i++) {
    std::thread t(
        [&](int tid) {
          if (tid == 0) {
            asyncErr->setAsyncException(
                Exception("test error on thread 0", commRemoteError));
          } else {
            // Except all threads should see the asyncError stop waiting
            while (asyncErr->getAsyncResult() == commSuccess) {
              std::this_thread::yield();
            }

            // Expect the asyncError is set.
            auto asyncError = asyncErr->getAsyncException();
            EXPECT_EQ(asyncError.result(), commRemoteError);
            EXPECT_THAT(
                asyncError.what(),
                ::testing::HasSubstr("test error on thread 0"));
          }
        },
        i);
    threads.push_back(std::move(t));
  }

  // Expect all threads should see the asyncError and exit.
  for (auto& t : threads) {
    t.join();
  }
}

TEST(AsyncErrorTest, AbortOnError) {
  // Do not check exit code since CLOGF(FATAL) may trigger core dump, and
  // changed SIGABRT to core dump
  EXPECT_DEATH(
      {
        auto asyncErr = std::make_shared<AsyncError>(true, "AsyncErrorTest");
        CTRAN_ASYNC_ERR_GUARD(asyncErr, {
          throw Exception("test error on thread 0", commInternalError);
        });
      },
      "test error on thread 0")
      << "Expected to abort on error";
}

TEST(AsyncErrorTest, FaultToleranceDisabled) {
  auto comm =
      std::make_unique<CtranComm>(ctran::utils::createAbort(/*enabled=*/false));
  EXPECT_THROW(
      {
        CTRAN_ASYNC_ERR_GUARD_FAULT_TOLERANCE(comm, {
          throw ::ctran::utils::Exception("UT error", commRemoteError);
        });
      },
      ::ctran::utils::Exception)
      << "Expected to throw exception on error";
}

TEST(AsyncErrorTest, FaultToleranceEnabled) {
  auto comm =
      std::make_unique<CtranComm>(ctran::utils::createAbort(/*enabled=*/true));
  CTRAN_ASYNC_ERR_GUARD_FAULT_TOLERANCE(
      comm, { throw ::ctran::utils::Exception("UT error", commRemoteError); });
  EXPECT_TRUE(comm->testAbort());
}
