// Copyright (c) Meta Platforms, Inc. and affiliates.

// Verify that build-wide spdlog configuration is independent of include order.
#include <spdlog/spdlog.h>

#include "comms/utils/logger/SpdlogLogger.h"

#include <gtest/gtest.h>

using meta::comms::logger::getSpdlogLogger;

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
  });
}

TEST(SpdlogLoggerTest, FatalTerminatesProcess) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_DEATH(COMMS_LOG(FATAL, "fatal message: {}", 5), "fatal message: 5");
}

TEST(SpdlogLoggerTest, CompileTimeGateSkipsArguments) {
  int evaluationCount = 0;
  COMMS_LOG(DBG, "compiled out: {}", ++evaluationCount);
  EXPECT_EQ(evaluationCount, 0);

  COMMS_LOG(INFO, "compiled in: {}", ++evaluationCount);
  EXPECT_EQ(evaluationCount, 1);
}
