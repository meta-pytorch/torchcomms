// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <algorithm>
#include <string_view>

#include <fmt/format.h>
#include <folly/FileUtil.h>
#include <folly/ScopeGuard.h>
#include <folly/logging/LogMessage.h>
#include <folly/testing/TestUtil.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/testinfra/TestUtils.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/Logger.h"
#include "comms/utils/logger/LoggingFormat.h"
#include "comms/utils/logger/ProcessGlobalErrorsUtil.h"

#include "debug.h" // @manual
#include "param.h" // @manual

namespace {
void inline checkStringHasLogging(
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
    initNcclLogger();
  }
};

// Just for remembering the test format. Current test format example:
// P1783645719
TEST_F(NcclLoggerTest, LogDisplay) {
  ncclResetDebugInit();

  ncclCvarInit();
  auto debugGuard = EnvRAII(NCCL_DEBUG, std::string{"INFO"});
  setenv("NCCL_DEBUG", "INFO", 1);
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
  ERR(ncclInternalError, "%s", TestStr.c_str());

  finishLogging();
}

TEST_F(NcclLoggerTest, GetLastCommsErrorTest) {
  auto debugGuard = EnvRAII(NCCL_DEBUG, std::string{"INFO"});
  setenv("NCCL_DEBUG", "INFO", 1);
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
  ERR(ncclInternalError, "%s", errorMsg.c_str());
  sleep(1);
  lastError = meta::comms::logger::getLastCommsError();
  EXPECT_THAT(lastError, ::testing::HasSubstr(errorMsg));
  EXPECT_THAT(lastError, ::testing::HasSubstr("NCCL Stack trace:"));

  // Log another error message - should update to the new error
  std::string errorMsg2 = "SECOND ERROR MESSAGE";
  ERR(ncclInternalError, "%s", errorMsg2.c_str());
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
  setenv("NCCL_DEBUG", "INFO", 1);
  ncclResetDebugInit();

  initLogging();

  // Log a multiline error message
  std::string multilineError = "First line\nSecond line\nThird line";
  ERR(ncclInternalError, "%s", multilineError.c_str());
  sleep(1);

  auto lastError = meta::comms::logger::getLastCommsError();
  EXPECT_THAT(lastError, ::testing::StartsWith(multilineError));

  finishLogging();
}

TEST_F(NcclLoggerTest, GetLastCommsErrorLongMessageTest) {
  auto debugGuard = EnvRAII(NCCL_DEBUG, std::string{"INFO"});
  setenv("NCCL_DEBUG", "INFO", 1);
  ncclResetDebugInit();

  initLogging();

  // Create a long error message (but within the 1024 char buffer)
  std::string longError(500, 'X');
  ERR(ncclInternalError, "%s", longError.c_str());
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
  setenv("NCCL_DEBUG", "WARN", 1);
  ncclResetDebugInit();

  initLogging();
  std::string TestStr = "TESTING";

  testing::internal::CaptureStdout();
  ERR(ncclInternalError, "%s", TestStr.c_str());
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
  setenv("NCCL_DEBUG", "INFO", 1);
  ncclResetDebugInit();

  initLogging();
  std::string TestStr = "TESTING";

  testing::internal::CaptureStdout();
  ERR(ncclInternalError, "%s", TestStr.c_str());
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
  setenv("NCCL_DEBUG", "INFO", 1);
  auto debugSubsys = EnvRAII(NCCL_DEBUG_SUBSYS, std::string{"ENV,NET"});
  setenv("NCCL_DEBUG_SUBSYS", "ENV,NET", 1);
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
  setenv("NCCL_DEBUG", "INFO", 1);
  auto debugSubsys = EnvRAII(NCCL_DEBUG_SUBSYS, std::string{"^ENV,NET"});
  setenv("NCCL_DEBUG_SUBSYS", "^ENV,NET", 1);
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

TEST_F(NcclLoggerTest, DebugFileLoggingTest) {
  folly::test::TemporaryDirectory tmpDir;

  auto tempFile = tmpDir.path() / "tempFile";
  // Set nccl_debug to set log file
  auto nccl_debug = EnvRAII(NCCL_DEBUG, std::string{"INFO"});
  setenv("NCCL_DEBUG", "INFO", 1);
  auto debugFileGuard = EnvRAII(NCCL_DEBUG_FILE, tempFile.string());
  setenv("NCCL_DEBUG_FILE", tempFile.c_str(), 1);
  // EnvRAII only restores the NCCL_DEBUG_FILE cvar, not the raw OS env var. The
  // tempFile lives in a TemporaryDirectory that is deleted at test end, so the
  // OS env var must be unset on scope exit (even on the ASSERT_TRUE
  // early-return below) or a later test's ncclCvarInit() would reload this
  // stale path and NcclLogger would fail to open the deleted file.
  auto debugFileEnvGuard =
      folly::makeGuard([]() { unsetenv("NCCL_DEBUG_FILE"); });
  ncclResetDebugInit();
  initLogging();

  INFO(NCCL_ALL, "Trigger DebugInit");

  constexpr std::string_view TestStr = "RAW TESTING";
  constexpr std::string_view TestStr2 = "TESTING";

  testing::internal::CaptureStderr();

  XLOG(INFO) << TestStr;
  XLOG(WARN) << TestStr;
  XLOG(ERR) << TestStr;

  INFO(NCCL_ALL, "%s", TestStr2.data());
  WARN("%s", TestStr2.data());
  ERR(ncclInternalError, "%s", TestStr2.data());

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
}

TEST_F(NcclLoggerTest, AppendErrorToStackTest) {
  auto debugGuard = EnvRAII(NCCL_DEBUG, std::string{"INFO"});
  setenv("NCCL_DEBUG", "INFO", 1);
  ncclResetDebugInit();

  initLogging();

  // Log an error message first
  std::string errorMsg = "Base error message";
  ERR(ncclInternalError, "%s", errorMsg.c_str());
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
  setenv("NCCL_DEBUG", "INFO", 1);
  ncclResetDebugInit();

  initLogging();

  // Log an error message first
  std::string errorMsg = "Error for stack order test";
  ERR(ncclInternalError, "%s", errorMsg.c_str());
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
  setenv("NCCL_DEBUG", "INFO", 1);
  ncclResetDebugInit();

  initLogging();

  // Log an error
  std::string errorMsg = "Critical error occurred";
  ERR(ncclInternalError, "%s", errorMsg.c_str());
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
  setenv("NCCL_DEBUG", "INFO", 1);
  ncclResetDebugInit();

  initLogging();

  // Log an error without adding any stack frames
  std::string errorMsg = "Simple error without stack";
  ERR(ncclInternalError, "%s", errorMsg.c_str());
  sleep(1);

  auto lastError = meta::comms::logger::getLastCommsError();
  std::string errorStr(lastError);

  EXPECT_THAT(errorStr, ::testing::HasSubstr(errorMsg));
  EXPECT_THAT(errorStr, ::testing::HasSubstr("NCCL Stack trace:"));

  finishLogging();
}

// The following tests assert v2_30-only error behavior: plain ERR routes to the
// Scuba error record / ProcessGlobalErrors, and WARN no longer contributes to
// the error stack. Older ncclx versions keep the legacy behavior, so gate these
// tests to v2_30+.
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)

TEST_F(NcclLoggerTest, WarnDoesNotContributeToErrorStack) {
  auto debugGuard = EnvRAII(NCCL_DEBUG, std::string{"INFO"});
  setenv("NCCL_DEBUG", "INFO", 1);
  ncclResetDebugInit();

  initLogging();

  // An ERR sets the last error message.
  const std::string errorMsg = "Initial error";
  ERR(ncclInternalError, "%s", errorMsg.c_str());
  sleep(1);

  // WARN is a propagator and must no longer append to the error stack.
  const std::string warnMsg = "Propagator WARN should not be recorded";
  WARN("%s", warnMsg.c_str());
  sleep(1);

  const std::string errorStr(meta::comms::logger::getLastCommsError());
  EXPECT_THAT(errorStr, ::testing::HasSubstr(errorMsg));
  EXPECT_THAT(errorStr, ::testing::HasSubstr("NCCL Stack trace:"));
  EXPECT_THAT(errorStr, ::testing::Not(::testing::HasSubstr(warnMsg)));

  finishLogging();
}

TEST_F(NcclLoggerTest, ErrorRecordsSingleScubaError) {
  ncclCvarInit();
  auto debugGuard = EnvRAII(NCCL_DEBUG, std::string{"INFO"});
  setenv("NCCL_DEBUG", "INFO", 1);
  ncclResetDebugInit();

  initLogging();

  // A single ERR produces exactly one error record (not one per stack frame).
  const std::string errorMsg = "Primary error from ERR";
  ERR(ncclInternalError, "%s", errorMsg.c_str());

  const auto errors =
      ProcessGlobalErrorsUtil::getAllState().errorAndStackTraces;
  // One ERR must produce exactly one record carrying this message (the old
  // per-propagator-frame behavior would produce several). Count by message so
  // the check is robust to the process-global ring buffer's cap and to records
  // left by earlier tests.
  const size_t matchCount =
      std::count_if(errors.begin(), errors.end(), [&](const auto& errorEntry) {
        return errorEntry.errorMessage.find(errorMsg) != std::string::npos;
      });
  EXPECT_EQ(matchCount, 1u);
  EXPECT_THAT(errors.back().errorMessage, ::testing::HasSubstr(errorMsg));

  finishLogging();
}

TEST_F(NcclLoggerTest, SecondErrorUpdatesLastError) {
  auto debugGuard = EnvRAII(NCCL_DEBUG, std::string{"INFO"});
  setenv("NCCL_DEBUG", "INFO", 1);
  ncclResetDebugInit();

  initLogging();

  ERR(ncclInternalError, "Base error message");
  sleep(1);
  ERR(ncclInternalError, "Newer error message");
  sleep(1);

  const std::string errorStr(meta::comms::logger::getLastCommsError());
  EXPECT_THAT(errorStr, ::testing::HasSubstr("Newer error message"));

  finishLogging();
}

TEST_F(NcclLoggerTest, WarnNotAppendedButDirectAppendIs) {
  auto debugGuard = EnvRAII(NCCL_DEBUG, std::string{"INFO"});
  setenv("NCCL_DEBUG", "INFO", 1);
  ncclResetDebugInit();

  initLogging();

  ERR(ncclInternalError, "Error occurred in operation");
  sleep(1);

  // WARN must not appear in the error stack; direct appends still do.
  WARN("This WARN must not appear in the error stack");
  meta::comms::logger::appendErrorToStack("Direct append frame");
  sleep(1);

  const std::string errorStr(meta::comms::logger::getLastCommsError());
  EXPECT_THAT(errorStr, ::testing::HasSubstr("Error occurred in operation"));
  EXPECT_THAT(errorStr, ::testing::HasSubstr("Direct append frame"));
  EXPECT_THAT(
      errorStr,
      ::testing::Not(::testing::HasSubstr("This WARN must not appear")));

  finishLogging();
}

TEST_F(NcclLoggerTest, SecondErrorUpdatesMessageKeepsAppendedStack) {
  auto debugGuard = EnvRAII(NCCL_DEBUG, std::string{"INFO"});
  setenv("NCCL_DEBUG", "INFO", 1);
  ncclResetDebugInit();

  initLogging();

  // First error with an appended stack frame.
  ERR(ncclInternalError, "First error");
  sleep(1);
  meta::comms::logger::appendErrorToStack("Stack for first error");

  const std::string errorStr1(meta::comms::logger::getLastCommsError());
  EXPECT_THAT(errorStr1, ::testing::HasSubstr("First error"));
  EXPECT_THAT(errorStr1, ::testing::HasSubstr("Stack for first error"));

  // A newer error updates the message; the appended stack remains.
  ERR(ncclInternalError, "Second error");
  sleep(1);

  const std::string errorStr2(meta::comms::logger::getLastCommsError());
  EXPECT_THAT(errorStr2, ::testing::HasSubstr("Second error"));
  EXPECT_THAT(errorStr2, ::testing::HasSubstr("Stack for first error"));

  finishLogging();
}

#endif // NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)

TEST_F(NcclLoggerTest, TestUtilsLogHandler) {
  ncclResetDebugInit();

  ncclCvarInit();
  auto debugGuard = EnvRAII(NCCL_DEBUG, std::string{"INFO"});
  setenv("NCCL_DEBUG", "INFO", 1);

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
  ERR(ncclInternalError, "%s", TestStr.c_str());

  finishLogging();
}
