// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/ctran/CtranComm.h"
#include "comms/ctran/utils/Exception.h"

using ctran::utils::Exception;

class CtranAsyncErrorTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST(CtranAsyncErrorTest, SetAndGet) {
  constexpr auto numThreads = 10;
  std::vector<std::thread> threads;

  constexpr int rank = 1;
  constexpr uint64_t commHash = 0x12345;

  auto comm = std::make_unique<CtranComm>();
  // Expect no asyncError before set
  EXPECT_EQ(comm->getAsyncResult(), commSuccess);

  for (auto i = 0; i < numThreads; i++) {
    std::thread t(
        [&](int tid) {
          if (tid == 0) {
            comm->setAsyncException(Exception(
                "test error on thread 0", commRemoteError, rank, commHash));
          } else {
            // Except all threads should see the asyncError stop waiting
            while (comm->getAsyncResult() == commSuccess) {
              std::this_thread::yield();
            }

            // Expect the asyncError is set.
            auto asyncError = comm->getAsyncException();
            EXPECT_EQ(asyncError.result(), commRemoteError);
            EXPECT_EQ(asyncError.rank(), rank);
            EXPECT_EQ(asyncError.commHash(), commHash);
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
