// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/uniflow/logging/Logger.h"

#include <gtest/gtest.h>

using namespace uniflow::logging;

TEST(LoggerTest, GetLoggerReturnsSameInstance) {
  auto* logger1 = getLogger();
  auto* logger2 = getLogger();
  EXPECT_EQ(logger1, logger2);
}

TEST(LoggerTest, GetLoggerHasCorrectName) {
  auto* logger = getLogger();
  EXPECT_EQ(logger->name(), "uniflow");
}

TEST(LoggerTest, LogMacrosProduceOutput) {
  auto* logger = getLogger();

  // Verify the logger accepts INFO/WARN/ERROR at runtime — these levels
  // are also above the compile-time gate (SPDLOG_ACTIVE_LEVEL=INFO),
  // so the corresponding UNIFLOW_LOG_* macros will expand and log.
  EXPECT_TRUE(logger->should_log(spdlog::level::info));
  EXPECT_TRUE(logger->should_log(spdlog::level::warn));
  EXPECT_TRUE(logger->should_log(spdlog::level::err));
  EXPECT_TRUE(logger->should_log(spdlog::level::critical));

  // TRACE and DEBUG are below the compile-time gate and expand to (void)0.
  // Execute all macros to verify they compile and run without crashing.
  UNIFLOW_LOG_TRACE("trace message: {}", 1);
  UNIFLOW_LOG_DEBUG("debug message: {}", 2);
  UNIFLOW_LOG_INFO("info message: {}", 3);
  UNIFLOW_LOG_WARN("warn message: {}", 4);
  UNIFLOW_LOG_ERROR("error message: {}", 5);
  UNIFLOW_LOG_CRITICAL("critical message: {}", 6);
}
