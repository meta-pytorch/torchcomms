// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <chrono>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <signal.h>

#include "comms/ncclx/meta/tests/ForkBasedTestDriver.h"

using namespace ncclx::test;
using namespace ::testing;

// --- Static method tests (no fixture needed) ---

TEST(ForkBasedTestDriverStaticTest, IsTestDriverByDefault) {
  EXPECT_TRUE(ForkBasedTestDriver::isTestDriverProcess());
}

// --- Fixture 1: Basic multi-rank success ---
// Workers exit normally; verifies rank/worldSize accessors and KV round-trip.
// KV test is included here (not a separate TEST_F) because the singleton only
// exec's workers for the first test in the fixture.

class ForkBasedTestDriverBasicTest : public ForkBasedTestDriver,
                                     public ::testing::Test {
 public:
  void SetUp() override {
    ForkBasedTestDriver::SetUp(Config{.numRanks = 2});
  }
};

TEST_F(ForkBasedTestDriverBasicTest, WorkersExitSuccessfully) {
  if (isTestDriverProcess()) {
    auto& state = getTestDriverState();
    EXPECT_EQ(state.workerExitCodes.size(), 2);
    EXPECT_THAT(state.workerExitCodes, Each(Eq(0)));
    return;
  }
  EXPECT_GE(getRank(), 0);
  EXPECT_LT(getRank(), 2);
  EXPECT_EQ(getWorldSize(), 2);

  // KV round-trip
  setKey("test_key", "test_value");
  EXPECT_EQ(waitForKey("test_key"), "test_value");

  // Timeout overload: success path
  setKey("timeout_test_key", "timeout_test_value");
  auto result = waitForKey("timeout_test_key", std::chrono::milliseconds(5000));
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, "timeout_test_value");

  // Timeout overload: timeout path
  auto missing =
      waitForKey("nonexistent_key_xyz", std::chrono::milliseconds(100));
  EXPECT_FALSE(missing.has_value());
}

// --- Fixture 2: Worker exit code capture ---
// Worker calls _exit(42). Must be its own fixture so the singleton captures
// exit code 42, not stale exit codes from another fixture's workers.

class ForkBasedTestDriverExitCodeTest : public ForkBasedTestDriver,
                                        public ::testing::Test {
 public:
  void SetUp() override {
    ForkBasedTestDriver::SetUp(
        Config{.numRanks = 1, .shouldExitOnFailure = false});
  }
};

TEST_F(ForkBasedTestDriverExitCodeTest, ExactExitCodeCaptured) {
  if (isTestDriverProcess()) {
    auto& state = getTestDriverState();
    EXPECT_EQ(state.workerExitCodes.size(), 1);
    EXPECT_EQ(state.workerExitCodes[0], 42);
    return;
  }
  _exit(42);
}

// --- Fixture 3: Signal-terminated worker ---
// Worker calls raise(SIGTERM). Must be its own fixture so the singleton
// captures 128+SIGTERM, not stale exit codes from another fixture.

class ForkBasedTestDriverSignalTest : public ForkBasedTestDriver,
                                      public ::testing::Test {
 public:
  void SetUp() override {
    ForkBasedTestDriver::SetUp(
        Config{.numRanks = 1, .shouldExitOnFailure = false});
  }
};

TEST_F(ForkBasedTestDriverSignalTest, SignalReportsAs128PlusSignal) {
  if (isTestDriverProcess()) {
    auto& state = getTestDriverState();
    EXPECT_EQ(state.workerExitCodes.size(), 1);
    EXPECT_EQ(state.workerExitCodes[0], 128 + SIGTERM);
    return;
  }
  raise(SIGTERM);
}
