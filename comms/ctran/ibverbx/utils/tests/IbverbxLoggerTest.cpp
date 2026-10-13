// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ctran/ibverbx/utils/IbverbxLogger.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace ibverbx::logging {
namespace {

struct Captured {
  Level level;
  std::string message;
};

std::vector<Captured>& captured() {
  static std::vector<Captured> lines;
  return lines;
}

void captureSink(
    const Level level,
    const char* /*file*/,
    int /*line*/,
    const char* /*function*/,
    const std::string& message) {
  captured().push_back({level, message});
}

class IbverbxLoggerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    captured().clear();
    setLogSink(captureSink);
  }

  void TearDown() override {
    setLogSink(nullptr);
  }
};

TEST_F(IbverbxLoggerTest, InstalledSinkReceivesFormattedMessage) {
  IBVERBX_LOG(ERR, "qp {} failed with {}", 7, "ENOMEM");

  const std::vector<Captured> expected{{Level::ERR, "qp 7 failed with ENOMEM"}};
  ASSERT_EQ(captured().size(), expected.size());
  EXPECT_EQ(captured()[0].level, expected[0].level);
  EXPECT_EQ(captured()[0].message, expected[0].message);
}

// A sink sees every level, including DBG which the built-in stderr gate drops,
// so a host can apply its own threshold rather than inheriting ibverbx's.
TEST_F(IbverbxLoggerTest, InstalledSinkReceivesLevelsTheStderrGateDrops) {
  IBVERBX_LOG(DBG, "verbose");

  ASSERT_EQ(captured().size(), 1u);
  EXPECT_EQ(captured()[0].level, Level::DBG);
}

TEST_F(IbverbxLoggerTest, LogIfEmitsOnlyWhenConditionHolds) {
  IBVERBX_LOG_IF(WARN, false, "suppressed");
  EXPECT_TRUE(captured().empty());

  IBVERBX_LOG_IF(WARN, true, "emitted");

  ASSERT_EQ(captured().size(), 1u);
  EXPECT_EQ(captured()[0].message, "emitted");
}

// The hot-path assertions rely on this: a false condition must not evaluate the
// message arguments at all.
TEST_F(IbverbxLoggerTest, LogIfDoesNotEvaluateArgumentsWhenConditionIsFalse) {
  int evaluations = 0;
  const auto countEvaluation = [&]() {
    ++evaluations;
    return 0;
  };

  IBVERBX_LOG_IF(FATAL, false, "unreachable {}", countEvaluation());

  EXPECT_EQ(evaluations, 0);
}

TEST(IbverbxLoggerDeathTest, FatalEmitsThenAborts) {
  setLogSink(nullptr);
  EXPECT_DEATH(
      IBVERBX_LOG(FATAL, "Check failed: {}", "invariant"),
      "IBVERBX FATAL Check failed: invariant");
}

// Without a sink the default threshold keeps DBG out while admitting INFO, so
// embedded consumers see diagnostics but not verbose output.
TEST(IbverbxLoggerDefaultTest, StderrGateAdmitsInfoAndDropsDbg) {
  setLogSink(nullptr);
  EXPECT_FALSE(shouldLog(Level::DBG));
  EXPECT_TRUE(shouldLog(Level::INFO));
  EXPECT_TRUE(shouldLog(Level::FATAL));
}

// A line the level gate drops must cost nothing: the gate sits in the macro,
// ahead of formatting, so the message arguments are never evaluated.
TEST(IbverbxLoggerDefaultTest, DroppedLevelDoesNotEvaluateArguments) {
  setLogSink(nullptr);

  int evaluations = 0;
  const auto countEvaluation = [&]() {
    ++evaluations;
    return 0;
  };

  IBVERBX_LOG(DBG, "dropped {}", countEvaluation());

  EXPECT_EQ(evaluations, 0);
}

} // namespace
} // namespace ibverbx::logging
