// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <fmt/core.h>
#include <folly/ScopeGuard.h>
#include <folly/init/Init.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/testinfra/TestXPlatUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"

#include "meta/NcclxLogger.h"
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
  }

  void TearDown() override {}
};

class DebugExtTest : public ::testing::Test {
 public:
  DebugExtTest() = default;
  void SetUp() override {}

  void TearDown() override {}

  void finishLogging() {
    meta::comms::logger::getSpdlogLogger(ncclx::logging::kNcclxLoggerName)
        .flush();
  }

  void initLogging() {
#ifdef NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED
    // 2.32+ gates on a per-level bitmask instead of a scalar level.
    ncclDebugLevelMask = NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED;
#else
    ncclDebugLevel = -1;
#endif
    initNcclLogger();
  }
};

TEST_F(DebugExtTest, TestWarnLogToLimit) {
  initEnv();
  NCCL_DEBUG = "WARN";
  NCCL_DEBUG_FILE = NCCL_DEBUG_FILE_DEFAULTCVARVALUE;
  SysEnvRAII debugEnv{"NCCL_DEBUG", "WARN"};
  SysEnvRAII debugFileEnv{"NCCL_DEBUG_FILE", ""};
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
  NCCL_DEBUG_FILE = NCCL_DEBUG_FILE_DEFAULTCVARVALUE;
  SysEnvRAII debugEnv{"NCCL_DEBUG", "WARN"};
  SysEnvRAII debugFileEnv{"NCCL_DEBUG_FILE", ""};
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
  NCCL_DEBUG_FILE = NCCL_DEBUG_FILE_DEFAULTCVARVALUE;
  SysEnvRAII debugEnv{"NCCL_DEBUG", "WARN"};
  SysEnvRAII debugFileEnv{"NCCL_DEBUG_FILE", ""};
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

// ncclGetLastError() must report the ERR() origin, not the last propagation
// WARN. ncclDebugLogV() rewrites upstream's ncclLastError[] buffer on every
// WARN, and NCCLCHECK emits one per enclosing frame, so reading that buffer
// yields the outermost "-> %d" trace instead of the root cause.
TEST_F(DebugExtTest, LastErrorKeepsRootCauseThroughPropagation) {
  initEnv();
  NCCL_DEBUG = "WARN";
  NCCL_DEBUG_FILE = NCCL_DEBUG_FILE_DEFAULTCVARVALUE;
  SysEnvRAII debugEnv{"NCCL_DEBUG", "WARN"};
  SysEnvRAII debugFileEnv{"NCCL_DEBUG_FILE", ""};
  initLogging();

  ERR(ncclSystemError, "root cause: connect refused on eth0");
  // What NCCLCHECK/NCCLCHECKGOTO emit as the error unwinds.
  WARN("-> %d", ncclSystemError);
  WARN("-> %d", ncclSystemError);

  const std::string lastError = ncclGetLastError(nullptr);
  EXPECT_THAT(
      lastError, testing::HasSubstr("root cause: connect refused on eth0"));
  EXPECT_THAT(
      lastError,
      testing::Not(testing::HasSubstr(fmt::format("-> {}", ncclSystemError))));
  finishLogging();
}

#ifdef NCCL_LOG_HAS_ATTN
// The sink must not drop lines the native level mask passes. ATTN and
// NCCL_DEBUG_LEVELS both widen the mask beyond what NCCL_DEBUG names.
TEST_F(DebugExtTest, AttnLevelLinesReachOutput) {
  initEnv();
  NCCL_DEBUG = "ATTN";
  NCCL_DEBUG_FILE = NCCL_DEBUG_FILE_DEFAULTCVARVALUE;
  SysEnvRAII debugEnv{"NCCL_DEBUG", "ATTN"};
  SysEnvRAII debugFileEnv{"NCCL_DEBUG_FILE", ""};
  initLogging();
  CAPTURE_STDOUT_WITH_FAIL_SAFE()
  ERR(ncclSystemError, "probe-err-at-attn");
  WARN("probe-warn-at-attn");
  ATTN("probe-attn-at-attn");
  sleep(1); // Wait for the xlog to actually log content
  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_THAT(output, testing::HasSubstr("probe-err-at-attn"));
  EXPECT_THAT(output, testing::HasSubstr("probe-warn-at-attn"));
  EXPECT_THAT(output, testing::HasSubstr("probe-attn-at-attn"));
  finishLogging();
}

TEST_F(DebugExtTest, DebugLevelsLinesReachOutput) {
  initEnv();
  NCCL_DEBUG = "WARN";
  NCCL_DEBUG_FILE = NCCL_DEBUG_FILE_DEFAULTCVARVALUE;
  SysEnvRAII debugEnv{"NCCL_DEBUG", "WARN"};
  SysEnvRAII debugLevelsEnv{"NCCL_DEBUG_LEVELS", "INFO"};
  SysEnvRAII debugFileEnv{"NCCL_DEBUG_FILE", ""};
  initLogging();
  CAPTURE_STDOUT_WITH_FAIL_SAFE()
  WARN("probe-warn-levels");
  INFO(NCCL_ALL, "probe-info-levels");
  sleep(1); // Wait for the xlog to actually log content
  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_THAT(output, testing::HasSubstr("probe-warn-levels"));
  EXPECT_THAT(output, testing::HasSubstr("probe-info-levels"));
  finishLogging();
}

TEST_F(DebugExtTest, DebugLevelsWithoutDebugReachOutput) {
  initEnv();
  NCCL_DEBUG = "";
  NCCL_DEBUG_FILE = NCCL_DEBUG_FILE_DEFAULTCVARVALUE;
  SysEnvRAII debugEnv{"NCCL_DEBUG", ""};
  SysEnvRAII debugLevelsEnv{"NCCL_DEBUG_LEVELS", "WARN,INFO"};
  SysEnvRAII debugFileEnv{"NCCL_DEBUG_FILE", ""};
  initLogging();
  CAPTURE_STDOUT_WITH_FAIL_SAFE()
  WARN("probe-warn-levels-only");
  INFO(NCCL_ALL, "probe-info-levels-only");
  sleep(1); // Wait for the xlog to actually log content
  std::string output = testing::internal::GetCapturedStdout();
  EXPECT_THAT(output, testing::HasSubstr("probe-warn-levels-only"));
  EXPECT_THAT(output, testing::HasSubstr("probe-info-levels-only"));
  finishLogging();
}
#endif

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::AddGlobalTestEnvironment(new NcclLoggerTestEnv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
