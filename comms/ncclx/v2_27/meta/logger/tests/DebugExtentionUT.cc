// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <fmt/core.h>
#include <folly/ScopeGuard.h>
#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/Logger.h"

#include "meta/logger/DebugExt.h"

#include "debug.h" // @manual
#include "param.h" // @manual

#define CAPTURE_STDOUT_WITH_FAIL_SAFE()                                    \
  testing::internal::CaptureStdout();                                      \
  SCOPE_FAIL {                                                             \
    std::string output = testing::internal::GetCapturedStdout();           \
    std::cout << "Test failed with stdout being: " << output << std::endl; \
  };

class NcclLoggerTestEnv : public ::testing::Environment {
 public:
  void SetUp() override {
    initEnv();
    // close logger to force unregistration of folly logger factory
    NcclLogger::close();
  }

  void TearDown() override {}
};

class DebugExtTest : public ::testing::Test {
 public:
  DebugExtTest() = default;
  void SetUp() override {}

  void TearDown() override {}

  void finishLogging() {
    NcclLogger::close();
  }

  void initLogging() {
    ncclDebugLevel = -1;
    ncclDebugLogFileStr = "";
    initNcclLogger();
  }
};

TEST_F(DebugExtTest, TestWarnLogToLimit) {
  initEnv();
  NCCL_DEBUG = "WARN";
  NCCL_DEBUG_FILE = NCCL_DEBUG_FILE_DEFAULT;
  initLogging();
  constexpr int logCount = 3;
  constexpr int iterCount = 10;
  CAPTURE_STDOUT_WITH_FAIL_SAFE()
  for (int i = 0; i < iterCount; i++) {
    WARN_FIRST_N(logCount, "test warning for %d times", i);
  }
  sleep(1); // Wait for the xlog to actually log content
  std::string output = testing::internal::GetCapturedStdout();
  for (int i = 0; i < logCount; i++) {
    EXPECT_THAT(
        output,
        testing::HasSubstr(fmt::format("test warning for {} times", i)));
  }
  for (int i = logCount; i < iterCount; i++) {
    EXPECT_THAT(
        output,
        testing::Not(
            testing::HasSubstr(fmt::format("test warning for {} times", i))));
  }
  finishLogging();
}

TEST_F(DebugExtTest, TestWarnLogBelowLimit) {
  initEnv();
  NCCL_DEBUG = "WARN";
  NCCL_DEBUG_FILE = NCCL_DEBUG_FILE_DEFAULT;
  initLogging();
  constexpr int logCount = 20;
  constexpr int iterCount = 10;
  CAPTURE_STDOUT_WITH_FAIL_SAFE()
  for (int i = 0; i < iterCount; i++) {
    WARN_FIRST_N(logCount, "test warning for %d times", i);
  }
  sleep(1); // Wait for the xlog to actually log content
  std::string output = testing::internal::GetCapturedStdout();
  for (int i = 0; i < iterCount; i++) {
    EXPECT_THAT(
        output,
        testing::HasSubstr(fmt::format("test warning for {} times", i)));
  }
  finishLogging();
}

TEST_F(DebugExtTest, TestThreeSeperateWarnLog) {
  initEnv();
  NCCL_DEBUG = "WARN";
  NCCL_DEBUG_FILE = NCCL_DEBUG_FILE_DEFAULT;
  initLogging();
  constexpr int logCount = 3;
  constexpr int iterCount = 10;
  CAPTURE_STDOUT_WITH_FAIL_SAFE()
  for (int i = 0; i < iterCount; i++) {
    WARN_FIRST_N(logCount, "[first] test warning for %d times", i);
    WARN_FIRST_N(logCount, "[second] test warning for %d times", i);
    WARN_FIRST_N(logCount, "[third] test warning for %d times", i);
  }
  sleep(1); // Wait for the xlog to actually log content
  std::string output = testing::internal::GetCapturedStdout();
  for (int i = 0; i < logCount; i++) {
    EXPECT_THAT(
        output,
        testing::HasSubstr(
            fmt::format("[first] test warning for {} times", i)));
    EXPECT_THAT(
        output,
        testing::HasSubstr(
            fmt::format("[second] test warning for {} times", i)));
    EXPECT_THAT(
        output,
        testing::HasSubstr(
            fmt::format("[third] test warning for {} times", i)));
  }
  for (int i = logCount; i < iterCount; i++) {
    EXPECT_THAT(
        output,
        testing::Not(
            testing::HasSubstr(
                fmt::format("[first] test warning for {} times", i))));
    EXPECT_THAT(
        output,
        testing::Not(
            testing::HasSubstr(
                fmt::format("[second] test warning for {} times", i))));
    EXPECT_THAT(
        output,
        testing::Not(
            testing::HasSubstr(
                fmt::format("[third] test warning for {} times", i))));
  }
  finishLogging();
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::AddGlobalTestEnvironment(new NcclLoggerTestEnv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
