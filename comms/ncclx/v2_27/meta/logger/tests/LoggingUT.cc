// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <string_view>

#include <fmt/format.h>
#include <folly/FileUtil.h>
#include <folly/logging/LogMessage.h>
#include <folly/testing/TestUtil.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/testinfra/TestUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/Logger.h"
#include "comms/utils/logger/LoggingFormat.h"

#include "debug.h" // @manual
#include "param.h" // @manual

namespace {
void checkStringHasLogging(
    std::string_view output,
    std::string_view expectString,
    std::string_view logLevel) {
  EXPECT_THAT(output, testing::HasSubstr(expectString));
  EXPECT_THAT(output, testing::HasSubstr(fmt::format("NCCL {}", logLevel)));
}

void checkStringHasNoLogging(
    std::string_view output,
    std::string_view expectString,
    std::string_view logLevel) {
  EXPECT_THAT(output, testing::Not(testing::HasSubstr(expectString)));
}

} // namespace

class NcclLoggerTest : public ::testing::Test {
 public:
  NcclLoggerTest() = default;
  void SetUp() override {}

  void TearDown() override {}

  void finishLogging() {
    sleep(1); // wait for logging to finish
    NcclLogger::close();
  }

  void initLogging() {
    ncclDebugLevel = -1;
    ncclDebugLogFileStr = "";
    initNcclLogger();
  }
};

// Just for remembering the test format. Current test format example:
// P1783645719
TEST_F(NcclLoggerTest, LogDisplay) {
  ncclResetDebugInit();

  ncclCvarInit();
  auto debugGuard = EnvRAII(NCCL_DEBUG, std::string{"INFO"});
  // auto fileGuard = EnvRAII(NCCL_DEBUG_FILE, std::string{"/tmp/debug.test3"});

  initLogging();
  NcclLogger::init(
      // TODO: Change the context name when ctran is refactored out of NCCLX
      // Otherwise the logging will no longer work as intended.
      {.contextName = "comms.ncclx.v2_25.meta.logger.tests",
       .logPrefix = "LOGGER",
       .logFilePath =
           meta::comms::logger::parseDebugFile(NCCL_DEBUG_FILE.c_str()),
       .logLevel = meta::comms::logger::loggerLevelToFollyLogLevel(
           meta::comms::logger::getLoggerDebugLevel(NCCL_DEBUG)),
       .threadContextFn = []() {
         int cudaDev = -1;
         cudaGetDevice(&cudaDev);
         return cudaDev;
       }});

  std::string TestStr = "TESTING";

  XLOG(INFO) << "RAW LOG TEST";
  XLOG(WARN) << "RAW LOG TEST";
  XLOG(ERR) << "RAW LOG TEST";

  INFO(NCCL_ALL, "%s", TestStr.c_str());
  WARN("%s", TestStr.c_str());
  ERR("%s", TestStr.c_str());

  finishLogging();
}

TEST_F(NcclLoggerTest, WarnLogTest) {
  auto debugGuard = EnvRAII(NCCL_DEBUG, std::string{"WARN"});
  ncclResetDebugInit();

  initLogging();
  std::string TestStr = "TESTING";

  testing::internal::CaptureStdout();
  ERR("%s", TestStr.c_str());
  sleep(1);
  std::string output = testing::internal::GetCapturedStdout();
  checkStringHasLogging(output, TestStr, "ERROR");

  testing::internal::CaptureStdout();
  WARN("%s", TestStr.c_str());
  sleep(1);
  output = testing::internal::GetCapturedStdout();
  checkStringHasLogging(output, TestStr, "WARN");

  testing::internal::CaptureStdout();
  INFO(NCCL_ALL, "%s", TestStr.c_str());
  sleep(1);
  output = testing::internal::GetCapturedStdout();
  checkStringHasNoLogging(output, TestStr, "INFO");

  finishLogging();
}

TEST_F(NcclLoggerTest, InfoLogTest) {
  auto debugGuard = EnvRAII(NCCL_DEBUG, std::string{"INFO"});
  ncclResetDebugInit();

  initLogging();
  std::string TestStr = "TESTING";

  testing::internal::CaptureStdout();
  ERR("%s", TestStr.c_str());
  sleep(1);
  std::string output = testing::internal::GetCapturedStdout();
  checkStringHasLogging(output, TestStr, "ERROR");

  testing::internal::CaptureStdout();
  WARN("%s", TestStr.c_str());
  sleep(1);
  output = testing::internal::GetCapturedStdout();
  checkStringHasLogging(output, TestStr, "WARN");

  testing::internal::CaptureStdout();
  INFO(NCCL_ALL, "%s", TestStr.c_str());
  sleep(1);
  output = testing::internal::GetCapturedStdout();
  checkStringHasLogging(output, TestStr, "INFO");

  finishLogging();
}

TEST_F(NcclLoggerTest, InfoSubsysLogTest) {
  auto nccl_debug = EnvRAII(NCCL_DEBUG, std::string{"INFO"});
  auto debugSubsys = EnvRAII(NCCL_DEBUG_SUBSYS, std::string{"ENV,NET"});
  ncclResetDebugInit();

  std::string TestStr = "TESTING";

  initLogging();
  testing::internal::CaptureStdout();
  INFO(NCCL_ENV, "%s", TestStr.c_str());
  sleep(1);
  std::string output = testing::internal::GetCapturedStdout();
  checkStringHasLogging(output, TestStr, "INFO");

  testing::internal::CaptureStdout();
  INFO(NCCL_NET, "%s", TestStr.c_str());
  sleep(1);
  output = testing::internal::GetCapturedStdout();
  checkStringHasLogging(output, TestStr, "INFO");

  testing::internal::CaptureStdout();
  INFO(NCCL_COLL, "%s", TestStr.c_str());
  sleep(1);
  output = testing::internal::GetCapturedStdout();
  checkStringHasNoLogging(output, TestStr, "INFO");

  finishLogging();
}

TEST_F(NcclLoggerTest, InfoSubsysLogRevertTest) {
  auto nccl_debug = EnvRAII(NCCL_DEBUG, std::string{"INFO"});
  auto debugSubsys = EnvRAII(NCCL_DEBUG_SUBSYS, std::string{"^ENV,NET"});
  ncclResetDebugInit();

  std::string TestStr = "TESTING";

  initLogging();
  testing::internal::CaptureStdout();
  INFO(NCCL_ENV, "%s", TestStr.c_str());
  sleep(1);
  std::string output = testing::internal::GetCapturedStdout();
  checkStringHasNoLogging(output, TestStr, "INFO");

  testing::internal::CaptureStdout();
  INFO(NCCL_NET, "%s", TestStr.c_str());
  sleep(1);
  output = testing::internal::GetCapturedStdout();
  checkStringHasNoLogging(output, TestStr, "INFO");

  testing::internal::CaptureStdout();
  INFO(NCCL_COLL, "%s", TestStr.c_str());
  sleep(1);
  output = testing::internal::GetCapturedStdout();
  checkStringHasLogging(output, TestStr, "INFO");

  finishLogging();
}

TEST_F(NcclLoggerTest, DebugFilePathTest) {
  folly::test::TemporaryDirectory tmpDir;

  auto tempFile = tmpDir.path() / "tempFile";
  // Set nccl_debug to set log file
  auto nccl_debug = EnvRAII(NCCL_DEBUG, std::string{"WARN"});
  auto debugFileGuard = EnvRAII(NCCL_DEBUG_FILE, tempFile.string());
  ncclResetDebugInit();

  EXPECT_EQ(ncclDebugLogFileStr, tempFile.string());
}

TEST_F(NcclLoggerTest, DebugDefaultPathTest) {
  // Set nccl_debug to set log file
  auto nccl_debug = EnvRAII(NCCL_DEBUG, std::string{"WARN"});
  auto debugFileGuard = EnvRAII(NCCL_DEBUG_FILE, std::string{""});
  ncclResetDebugInit();

  EXPECT_EQ(ncclDebugLogFileStr, "");
}

TEST_F(NcclLoggerTest, DebugStderrPathTest) {
  // Set nccl_debug to set log file
  auto nccl_debug = EnvRAII(NCCL_DEBUG, std::string{"WARN"});
  auto debugFileGuard = EnvRAII(NCCL_DEBUG_FILE, std::string{"stderr"});
  ncclResetDebugInit();

  EXPECT_EQ(ncclDebugLogFileStr, "stderr");
}

TEST_F(NcclLoggerTest, DebugFileLoggingTest) {
  folly::test::TemporaryDirectory tmpDir;

  auto tempFile = tmpDir.path() / "tempFile";
  // Set nccl_debug to set log file
  auto nccl_debug = EnvRAII(NCCL_DEBUG, std::string{"INFO"});
  auto debugFileGuard = EnvRAII(NCCL_DEBUG_FILE, tempFile.string());
  ncclResetDebugInit();
  initLogging();

  constexpr std::string_view TestStr = "RAW TESTING";
  constexpr std::string_view TestStr2 = "TESTING";

  testing::internal::CaptureStderr();

  XLOG(INFO) << TestStr;
  XLOG(WARN) << TestStr;
  XLOG(ERR) << TestStr;

  INFO(NCCL_ALL, "%s", TestStr2.data());
  WARN("%s", TestStr2.data());
  ERR("%s", TestStr2.data());

  auto stderrOutput = testing::internal::GetCapturedStderr();

  std::string fileContents;
  ASSERT_TRUE(folly::readFile(tempFile.c_str(), fileContents));
  for (const auto& level :
       std::vector<std::string_view>{"INFO", "WARN", "ERROR"}) {
    EXPECT_THAT(
        fileContents,
        testing::HasSubstr(fmt::format("NCCL {} {}", level, TestStr)));
    EXPECT_THAT(
        fileContents,
        testing::HasSubstr(fmt::format("NCCL {} {}", level, TestStr2)));
    if (level != "INFO") {
      // When logging to file, we should also log to stderr for WARN and ERROR
      EXPECT_THAT(
          stderrOutput,
          testing::HasSubstr(fmt::format("NCCL {} {}", level, TestStr)));
      EXPECT_THAT(
          stderrOutput,
          testing::HasSubstr(fmt::format("NCCL {} {}", level, TestStr2)));
    }
  }
  finishLogging();
}

TEST_F(NcclLoggerTest, TestUtilsLogHandler) {
  ncclResetDebugInit();

  ncclCvarInit();
  auto debugGuard = EnvRAII(NCCL_DEBUG, std::string{"INFO"});

  initLogging();
  auto utilsCategory = folly::LoggerDB::get().getCategory("comms.utils");
  ASSERT_THAT(utilsCategory, ::testing::NotNull());
  EXPECT_EQ(utilsCategory->getHandlers().size(), 1);

  NcclLogger::init(
      // TODO: Change the context name when ctran is refactored out of NCCLX
      // Otherwise the logging will no longer work as intended.
      {.contextName = "comms.ncclx.v2_25.meta.logger.tests",
       .logPrefix = "LOGGER",
       .logFilePath =
           meta::comms::logger::parseDebugFile(NCCL_DEBUG_FILE.c_str()),
       .logLevel = meta::comms::logger::loggerLevelToFollyLogLevel(
           meta::comms::logger::getLoggerDebugLevel(NCCL_DEBUG)),
       .threadContextFn = []() {
         int cudaDev = -1;
         cudaGetDevice(&cudaDev);
         return cudaDev;
       }});
  auto utilsCategory2 = folly::LoggerDB::get().getCategory("comms.utils");
  ASSERT_THAT(utilsCategory, ::testing::NotNull());
  EXPECT_EQ(utilsCategory, utilsCategory2);
  EXPECT_EQ(utilsCategory->getHandlers().size(), 1);

  utilsCategory->admitMessage(folly::LogMessage(
      utilsCategory,
      folly::LogLevel::INFO,
      std::chrono::system_clock::now(),
      "test",
      123,
      "UtilsTest",
      "testing testing 123"));

  std::string TestStr = "TESTING";

  XLOG(INFO) << "RAW LOG TEST";
  XLOG(WARN) << "RAW LOG TEST";
  XLOG(ERR) << "RAW LOG TEST";

  INFO(NCCL_ALL, "%s", TestStr.c_str());
  WARN("%s", TestStr.c_str());
  ERR("%s", TestStr.c_str());

  finishLogging();
}
