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

TEST_F(NcclLoggerTest, GetLastCommsErrorTest) {
  auto debugGuard = EnvRAII(NCCL_DEBUG, std::string{"INFO"});
  ncclResetDebugInit();

  initLogging();

  // Initially, the last error should be empty with just stack trace header
  auto lastError = meta::comms::logger::getLastCommsError();
  EXPECT_THAT(lastError, ::testing::HasSubstr("NCCL Stack trace:"));

  // Log an info message - should not update last error
  std::string infoMsg = "INFO MESSAGE";
  INFO(NCCL_ALL, "%s", infoMsg.c_str());
  sleep(1);
  lastError = meta::comms::logger::getLastCommsError();
  EXPECT_THAT(lastError, ::testing::Not(::testing::HasSubstr(infoMsg)));

  // Log a warning message - should not update last error
  std::string warnMsg = "WARN MESSAGE";
  WARN("%s", warnMsg.c_str());
  sleep(1);
  lastError = meta::comms::logger::getLastCommsError();
  EXPECT_THAT(lastError, ::testing::Not(::testing::HasSubstr(warnMsg)));

  // Log an error message - should update last error
  std::string errorMsg = "ERROR MESSAGE";
  ERR("%s", errorMsg.c_str());
  sleep(1);
  lastError = meta::comms::logger::getLastCommsError();
  EXPECT_THAT(lastError, ::testing::HasSubstr(errorMsg));
  EXPECT_THAT(lastError, ::testing::HasSubstr("NCCL Stack trace:"));

  // Log another error message - should update to the new error
  std::string errorMsg2 = "SECOND ERROR MESSAGE";
  ERR("%s", errorMsg2.c_str());
  sleep(1);
  lastError = meta::comms::logger::getLastCommsError();
  EXPECT_THAT(lastError, ::testing::HasSubstr(errorMsg2));
  EXPECT_THAT(lastError, ::testing::HasSubstr("NCCL Stack trace:"));

  // Log info and warn - last error should remain unchanged
  INFO(NCCL_ALL, "Another info");
  WARN("Another warn");
  sleep(1);
  lastError = meta::comms::logger::getLastCommsError();
  EXPECT_THAT(lastError, ::testing::HasSubstr(errorMsg2));

  finishLogging();
}

TEST_F(NcclLoggerTest, GetLastCommsErrorMultilineTest) {
  auto debugGuard = EnvRAII(NCCL_DEBUG, std::string{"INFO"});
  ncclResetDebugInit();

  initLogging();

  // Log a multiline error message
  std::string multilineError = "First line\nSecond line\nThird line";
  ERR("%s", multilineError.c_str());
  sleep(1);

  auto lastError = meta::comms::logger::getLastCommsError();
  EXPECT_THAT(lastError, ::testing::StartsWith(multilineError));

  finishLogging();
}

TEST_F(NcclLoggerTest, GetLastCommsErrorLongMessageTest) {
  auto debugGuard = EnvRAII(NCCL_DEBUG, std::string{"INFO"});
  ncclResetDebugInit();

  initLogging();

  // Create a long error message (but within the 1024 char buffer)
  std::string longError(500, 'X');
  ERR("%s", longError.c_str());
  sleep(1);

  auto lastError = meta::comms::logger::getLastCommsError();
  EXPECT_THAT(lastError, ::testing::StartsWith(longError));

  finishLogging();
}

TEST_F(NcclLoggerTest, GetLastCommsErrorLongMessageTestXLOG) {
  auto debugGuard = EnvRAII(NCCL_DEBUG, std::string{"INFO"});
  ncclResetDebugInit();

  initLogging();

  // Create a long error message (but within the 1024 char buffer)
  std::string longError(500, 'X');
  XLOG(ERR) << longError;
  sleep(1);

  auto lastError = meta::comms::logger::getLastCommsError();
  EXPECT_THAT(lastError, ::testing::StartsWith(longError));

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

TEST_F(NcclLoggerTest, AppendErrorToStackTest) {
  auto debugGuard = EnvRAII(NCCL_DEBUG, std::string{"INFO"});
  ncclResetDebugInit();

  initLogging();

  // Log an error message first
  std::string errorMsg = "Base error message";
  ERR("%s", errorMsg.c_str());
  sleep(1);

  // Append stack frames
  meta::comms::logger::appendErrorToStack("Stack frame 1: function1()");
  meta::comms::logger::appendErrorToStack("Stack frame 2: function2()");
  meta::comms::logger::appendErrorToStack("Stack frame 3: function3()");

  auto lastError = meta::comms::logger::getLastCommsError();
  EXPECT_THAT(lastError, ::testing::HasSubstr(errorMsg));
  EXPECT_THAT(lastError, ::testing::HasSubstr("NCCL Stack trace:"));
  EXPECT_THAT(lastError, ::testing::HasSubstr("Stack frame 1: function1()"));
  EXPECT_THAT(lastError, ::testing::HasSubstr("Stack frame 2: function2()"));
  EXPECT_THAT(lastError, ::testing::HasSubstr("Stack frame 3: function3()"));

  finishLogging();
}

TEST_F(NcclLoggerTest, AppendErrorToStackOrderTest) {
  auto debugGuard = EnvRAII(NCCL_DEBUG, std::string{"INFO"});
  ncclResetDebugInit();

  initLogging();

  // Log an error message first
  std::string errorMsg = "Error for stack order test";
  ERR("%s", errorMsg.c_str());
  sleep(1);

  // Append stack frames in specific order
  meta::comms::logger::appendErrorToStack("First");
  meta::comms::logger::appendErrorToStack("Second");
  meta::comms::logger::appendErrorToStack("Third");

  auto lastError = meta::comms::logger::getLastCommsError();
  std::string errorStr(lastError);

  // Verify the order is preserved
  size_t firstPos = errorStr.find("First");
  size_t secondPos = errorStr.find("Second");
  size_t thirdPos = errorStr.find("Third");

  EXPECT_NE(firstPos, std::string::npos);
  EXPECT_NE(secondPos, std::string::npos);
  EXPECT_NE(thirdPos, std::string::npos);
  EXPECT_LT(firstPos, secondPos);
  EXPECT_LT(secondPos, thirdPos);

  finishLogging();
}

TEST_F(NcclLoggerTest, GetLastCommsErrorWithMultipleStackFrames) {
  auto debugGuard = EnvRAII(NCCL_DEBUG, std::string{"INFO"});
  ncclResetDebugInit();

  initLogging();

  // Log an error
  std::string errorMsg = "Critical error occurred";
  ERR("%s", errorMsg.c_str());
  sleep(1);

  // Add detailed stack trace
  meta::comms::logger::appendErrorToStack("at ncclCommInitRank()");
  meta::comms::logger::appendErrorToStack("at ncclGroupEnd()");
  meta::comms::logger::appendErrorToStack("at ncclAllReduce()");
  meta::comms::logger::appendErrorToStack("in application code");

  auto lastError = meta::comms::logger::getLastCommsError();
  std::string errorStr(lastError);

  EXPECT_THAT(errorStr, ::testing::HasSubstr(errorMsg));
  EXPECT_THAT(errorStr, ::testing::HasSubstr("NCCL Stack trace:"));
  EXPECT_THAT(errorStr, ::testing::HasSubstr("at ncclCommInitRank()"));
  EXPECT_THAT(errorStr, ::testing::HasSubstr("at ncclGroupEnd()"));
  EXPECT_THAT(errorStr, ::testing::HasSubstr("at ncclAllReduce()"));
  EXPECT_THAT(errorStr, ::testing::HasSubstr("in application code"));

  // Verify all frames are present and in order
  EXPECT_LT(
      errorStr.find("at ncclCommInitRank()"),
      errorStr.find("at ncclGroupEnd()"));
  EXPECT_LT(
      errorStr.find("at ncclGroupEnd()"), errorStr.find("at ncclAllReduce()"));
  EXPECT_LT(
      errorStr.find("at ncclAllReduce()"),
      errorStr.find("in application code"));

  finishLogging();
}

TEST_F(NcclLoggerTest, GetLastCommsErrorEmptyStackTrace) {
  auto debugGuard = EnvRAII(NCCL_DEBUG, std::string{"INFO"});
  ncclResetDebugInit();

  initLogging();

  // Log an error without adding any stack frames
  std::string errorMsg = "Simple error without stack";
  ERR("%s", errorMsg.c_str());
  sleep(1);

  auto lastError = meta::comms::logger::getLastCommsError();
  std::string errorStr(lastError);

  EXPECT_THAT(errorStr, ::testing::HasSubstr(errorMsg));
  EXPECT_THAT(errorStr, ::testing::HasSubstr("NCCL Stack trace:"));

  finishLogging();
}

TEST_F(NcclLoggerTest, WarnWithScubaAppendsToStackTrace) {
  auto debugGuard = EnvRAII(NCCL_DEBUG, std::string{"INFO"});
  ncclResetDebugInit();

  initLogging();

  // Log an error first to set the error message
  std::string errorMsg = "Initial error";
  ERR("%s", errorMsg.c_str());
  sleep(1);

  // Use WARN_WITH_SCUBA to append to stack trace
  WARN_WITH_SCUBA("Stack trace entry 1 from WARN_WITH_SCUBA");
  WARN_WITH_SCUBA("Stack trace entry 2 from WARN_WITH_SCUBA");

  auto lastError = meta::comms::logger::getLastCommsError();
  std::string errorStr(lastError);

  EXPECT_THAT(errorStr, ::testing::HasSubstr(errorMsg));
  EXPECT_THAT(errorStr, ::testing::HasSubstr("NCCL Stack trace:"));
  EXPECT_THAT(
      errorStr,
      ::testing::HasSubstr("Stack trace entry 1 from WARN_WITH_SCUBA"));
  EXPECT_THAT(
      errorStr,
      ::testing::HasSubstr("Stack trace entry 2 from WARN_WITH_SCUBA"));

  finishLogging();
}

TEST_F(NcclLoggerTest, ErrWithScubaAppendsToStackTrace) {
  auto debugGuard = EnvRAII(NCCL_DEBUG, std::string{"INFO"});
  ncclResetDebugInit();

  initLogging();

  // Use ERR_WITH_SCUBA which should both log error and append to stack
  ERR_WITH_SCUBA("Primary error from ERR_WITH_SCUBA");
  sleep(1);

  // Add more stack frames
  WARN_WITH_SCUBA("Additional context from WARN_WITH_SCUBA");
  WARN_WITH_SCUBA("More context details");

  auto lastError = meta::comms::logger::getLastCommsError();
  std::string errorStr(lastError);

  EXPECT_THAT(
      errorStr, ::testing::HasSubstr("Primary error from ERR_WITH_SCUBA"));
  EXPECT_THAT(errorStr, ::testing::HasSubstr("NCCL Stack trace:"));
  EXPECT_THAT(
      errorStr,
      ::testing::HasSubstr("Additional context from WARN_WITH_SCUBA"));
  EXPECT_THAT(errorStr, ::testing::HasSubstr("More context details"));

  finishLogging();
}

TEST_F(NcclLoggerTest, ScubaStackTraceOrder) {
  auto debugGuard = EnvRAII(NCCL_DEBUG, std::string{"INFO"});
  ncclResetDebugInit();

  initLogging();

  // Log an error
  ERR("Base error message");
  sleep(1);

  // Use WARN_WITH_SCUBA in sequence
  WARN_WITH_SCUBA("Frame A");
  WARN_WITH_SCUBA("Frame B");
  WARN_WITH_SCUBA("Frame C");

  auto lastError = meta::comms::logger::getLastCommsError();
  std::string errorStr(lastError);

  // Verify the frames appear in order
  size_t posA = errorStr.find("Frame A");
  size_t posB = errorStr.find("Frame B");
  size_t posC = errorStr.find("Frame C");

  EXPECT_NE(posA, std::string::npos);
  EXPECT_NE(posB, std::string::npos);
  EXPECT_NE(posC, std::string::npos);
  EXPECT_LT(posA, posB);
  EXPECT_LT(posB, posC);

  finishLogging();
}

TEST_F(NcclLoggerTest, MixedScubaAndDirectStackAppend) {
  auto debugGuard = EnvRAII(NCCL_DEBUG, std::string{"INFO"});
  ncclResetDebugInit();

  initLogging();

  // Log an error
  ERR("Error occurred in operation");
  sleep(1);

  // Mix WARN_WITH_SCUBA and direct appendErrorToStack calls
  WARN_WITH_SCUBA("From WARN_WITH_SCUBA 1");
  meta::comms::logger::appendErrorToStack("From appendErrorToStack 1");
  WARN_WITH_SCUBA("From WARN_WITH_SCUBA 2");
  meta::comms::logger::appendErrorToStack("From appendErrorToStack 2");

  auto lastError = meta::comms::logger::getLastCommsError();
  std::string errorStr(lastError);

  EXPECT_THAT(errorStr, ::testing::HasSubstr("Error occurred in operation"));
  EXPECT_THAT(errorStr, ::testing::HasSubstr("From WARN_WITH_SCUBA 1"));
  EXPECT_THAT(errorStr, ::testing::HasSubstr("From appendErrorToStack 1"));
  EXPECT_THAT(errorStr, ::testing::HasSubstr("From WARN_WITH_SCUBA 2"));
  EXPECT_THAT(errorStr, ::testing::HasSubstr("From appendErrorToStack 2"));

  // Verify order is maintained
  size_t pos1 = errorStr.find("From WARN_WITH_SCUBA 1");
  size_t pos2 = errorStr.find("From appendErrorToStack 1");
  size_t pos3 = errorStr.find("From WARN_WITH_SCUBA 2");
  size_t pos4 = errorStr.find("From appendErrorToStack 2");

  EXPECT_LT(pos1, pos2);
  EXPECT_LT(pos2, pos3);
  EXPECT_LT(pos3, pos4);

  finishLogging();
}

TEST_F(NcclLoggerTest, ScubaStackTraceWithMultipleErrors) {
  auto debugGuard = EnvRAII(NCCL_DEBUG, std::string{"INFO"});
  ncclResetDebugInit();

  initLogging();

  // First error with stack
  ERR("First error");
  sleep(1);
  WARN_WITH_SCUBA("Stack for first error");

  auto lastError1 = meta::comms::logger::getLastCommsError();
  std::string errorStr1(lastError1);
  EXPECT_THAT(errorStr1, ::testing::HasSubstr("First error"));
  EXPECT_THAT(errorStr1, ::testing::HasSubstr("Stack for first error"));

  // Second error should update the message but stack remains
  ERR("Second error");
  sleep(1);

  auto lastError2 = meta::comms::logger::getLastCommsError();
  std::string errorStr2(lastError2);
  EXPECT_THAT(errorStr2, ::testing::HasSubstr("Second error"));
  // The old stack should still be there
  EXPECT_THAT(errorStr2, ::testing::HasSubstr("Stack for first error"));

  // Add more stack for second error
  WARN_WITH_SCUBA("Stack for second error");

  auto lastError3 = meta::comms::logger::getLastCommsError();
  std::string errorStr3(lastError3);
  EXPECT_THAT(errorStr3, ::testing::HasSubstr("Second error"));
  EXPECT_THAT(errorStr3, ::testing::HasSubstr("Stack for first error"));
  EXPECT_THAT(errorStr3, ::testing::HasSubstr("Stack for second error"));

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

  utilsCategory->admitMessage(
      folly::LogMessage(
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
