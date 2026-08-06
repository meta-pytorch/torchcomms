// Copyright (c) Meta Platforms, Inc. and affiliates.

// Verify that build-wide spdlog configuration is independent of include order.
#include <spdlog/spdlog.h>

#include "comms/utils/logger/SpdlogLogger.h"

#include <string>

#include <gtest/gtest.h>

using meta::comms::logger::getSpdlogLogger;

class LogLevelRestoringTest : public testing::Test {
 protected:
  void TearDown() override {
    getSpdlogLogger().set_level(spdlog::level::info);
  }
};

TEST(SpdlogLoggerTest, ReturnsStableNamedLogger) {
  EXPECT_EQ(&getSpdlogLogger(), &getSpdlogLogger());
  EXPECT_EQ(getSpdlogLogger().name(), "comms");
}

TEST(SpdlogLoggerTest, MapsFollyLevelNames) {
  EXPECT_NO_THROW({
    COMMS_LOG(DBG, "debug message: {}", 1);
    COMMS_LOG(INFO, "info message: {}", 2);
    COMMS_LOG(WARN, "warning message: {}", 3);
    COMMS_LOG(ERR, "error message: {}", 4);
    COMMS_LOG(CRITICAL, std::string{"critical message"});
  });
}

TEST(SpdlogLoggerTest, FatalTerminatesProcess) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_DEATH(
      {
        thread_local int threadContext = 7;
        meta::comms::logger::configureSpdlogLogger(
            "CTRAN", [&]() { return threadContext; });
        meta::comms::logger::setSpdlogThreadName("caller");
        COMMS_LOG(INFO, "message before fatal");
        COMMS_LOG(FATAL, "fatal message: {}", 5);
      },
      "message before fatal(.|\\n)*\\[7\\]\\[caller\\] CTRAN FATAL fatal message: 5");
}

TEST_F(LogLevelRestoringTest, FatalIsNotSuppressedByRuntimeLevel) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_DEATH(
      {
        getSpdlogLogger().set_level(spdlog::level::off);
        COMMS_LOG(FATAL, "fatal while logging is disabled");
      },
      "FATAL fatal while logging is disabled");
}

TEST(SpdlogLoggerTest, CompileTimeGateSkipsArguments) {
  int evaluationCount = 0;
  COMMS_LOG(DBG, "compiled out: {}", ++evaluationCount);
  EXPECT_EQ(evaluationCount, 0);

  COMMS_LOG(INFO, "compiled in: {}", ++evaluationCount);
  EXPECT_EQ(evaluationCount, 1);
}

TEST_F(LogLevelRestoringTest, RuntimeGateSkipsArguments) {
  int evaluationCount = 0;
  getSpdlogLogger().set_level(spdlog::level::warn);

  COMMS_LOG(INFO, "filtered at runtime: {}", ++evaluationCount);
  EXPECT_EQ(evaluationCount, 0);

  COMMS_LOG(WARN, "enabled at runtime: {}", ++evaluationCount);
  EXPECT_EQ(evaluationCount, 1);
}

TEST_F(LogLevelRestoringTest, EmptyThreadContextUsesDefault) {
  getSpdlogLogger().set_level(spdlog::level::info);
  meta::comms::logger::configureSpdlogLogger("TEST", {});

  EXPECT_NO_THROW(COMMS_LOG(INFO, "empty callback"));

  meta::comms::logger::configureSpdlogLogger("COMMS", []() { return 0; });
}
