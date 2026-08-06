// Copyright (c) Meta Platforms, Inc. and affiliates.

// Verify that build-wide spdlog configuration is independent of include order.
#include <spdlog/spdlog.h>

#include "comms/utils/logger/SpdlogLogger.h"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>

#include <gtest/gtest.h>

using meta::comms::logger::getSpdlogLogger;

class ScopedTestFile {
 public:
  explicit ScopedTestFile(std::string filename)
      : path_{std::filesystem::path{testing::TempDir()} / std::move(filename)} {
    std::filesystem::remove(path_);
  }

  ~ScopedTestFile() {
    removeNoexcept();
  }

  const std::filesystem::path& path() const {
    return path_;
  }

 private:
  void removeNoexcept() noexcept {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

  std::filesystem::path path_;
};

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

TEST(SpdlogLoggerTest, ReturnsStableLoggerPerContext) {
  auto& ctranLogger = getSpdlogLogger("comms.ctran");
  EXPECT_EQ(&ctranLogger, &getSpdlogLogger("comms.ctran"));
  EXPECT_NE(&ctranLogger, &getSpdlogLogger("comms.ncclx"));
  EXPECT_EQ(ctranLogger.name(), "comms.ctran");
}

TEST(SpdlogLoggerTest, MatchesLegacyStderrRouting) {
  EXPECT_TRUE(meta::comms::logger::shouldWriteCommsLogToStderr("WARN message"));
  EXPECT_TRUE(
      meta::comms::logger::shouldWriteCommsLogToStderr("ERROR message"));
  EXPECT_TRUE(
      meta::comms::logger::shouldWriteCommsLogToStderr("FATAL message"));
  EXPECT_FALSE(
      meta::comms::logger::shouldWriteCommsLogToStderr("CRITICAL message"));
  EXPECT_FALSE(
      meta::comms::logger::shouldWriteCommsLogToStderr("INFO message"));
  EXPECT_FALSE(meta::comms::logger::shouldWriteCommsLogToStderr(""));
}

TEST(SpdlogLoggerTest, SynchronousFileDeliveryMatchesLegacyRouting) {
  constexpr std::string_view kContext = "comms.synchronous_file_test";
  const ScopedTestFile scopedLogFile{"comms_spdlog_sync.log"};
  const auto logPath = scopedLogFile.path().string();
  meta::comms::logger::configureSpdlogLogger(
      kContext, "TEST", logPath, []() { return 0; }, {}, false);
  auto& logger = getSpdlogLogger(kContext);
  logger.set_level(spdlog::level::info);

  testing::internal::CaptureStderr();
  COMMS_LOG_NAMED(kContext, CRITICAL, "critical file message");
  COMMS_LOG_NAMED(kContext, WARN, "warning mirrored message");
  logger.set_level(spdlog::level::off);
  COMMS_LOG_NAMED(kContext, INFO, "filtered synchronous message");
  const auto stderrOutput = testing::internal::GetCapturedStderr();

  std::ifstream logFile{logPath};
  const std::string fileOutput{
      std::istreambuf_iterator<char>{logFile},
      std::istreambuf_iterator<char>{}};

  EXPECT_FALSE(logger.usesAsyncLogging());
  EXPECT_NE(fileOutput.find("critical file message"), std::string::npos);
  EXPECT_NE(fileOutput.find("warning mirrored message"), std::string::npos);
  EXPECT_EQ(fileOutput.find("filtered synchronous message"), std::string::npos);
  EXPECT_EQ(stderrOutput.find("critical file message"), std::string::npos);
  EXPECT_NE(stderrOutput.find("warning mirrored message"), std::string::npos);
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

TEST_F(LogLevelRestoringTest, CompileTimeGateIncludesDebug) {
  int evaluationCount = 0;
  getSpdlogLogger().set_level(spdlog::level::debug);

  COMMS_LOG(DBG, "compiled in: {}", ++evaluationCount);
  EXPECT_EQ(evaluationCount, 1);
}

TEST_F(LogLevelRestoringTest, RuntimeGateSkipsArguments) {
  int evaluationCount = 0;
  getSpdlogLogger().set_level(spdlog::level::warn);

  COMMS_LOG(DBG, "filtered at runtime: {}", ++evaluationCount);
  EXPECT_EQ(evaluationCount, 0);

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

TEST(SpdlogLoggerTest, ErrorCallbackReceivesFormattedUserMessage) {
  std::string errorMessage;
  auto& logger = getSpdlogLogger("comms.callback_test");
  logger.configure(
      "TEST",
      []() { return 0; },
      [&](std::string_view message) { errorMessage = message; });

  COMMS_LOG_NAMED("comms.callback_test", ERR, "error message: {}", 9);
  EXPECT_EQ(errorMessage, "error message: 9");
  logger.configure("TEST", []() { return 0; }, {});
}

TEST(SpdlogLoggerTest, ErrorCallbackDoesNotReenter) {
  constexpr std::string_view kContext = "comms.reentrant_callback_test";
  int callbackCount = 0;
  auto& logger = getSpdlogLogger(kContext);
  logger.configure(
      "TEST",
      []() { return 0; },
      [&](std::string_view) {
        ++callbackCount;
        COMMS_LOG_NAMED(kContext, ERR, "nested error");
      });

  COMMS_LOG_NAMED(kContext, ERR, "outer error");
  EXPECT_EQ(callbackCount, 1);
  logger.configure("TEST", []() { return 0; }, {});
}

TEST(SpdlogLoggerTest, ErrorCallbackGuardSpansContexts) {
  constexpr std::string_view kOuterContext = "comms.outer_callback_test";
  constexpr std::string_view kInnerContext = "comms.inner_callback_test";
  int outerCallbackCount = 0;
  int innerCallbackCount = 0;
  auto& outerLogger = getSpdlogLogger(kOuterContext);
  auto& innerLogger = getSpdlogLogger(kInnerContext);
  innerLogger.configure(
      "TEST",
      []() { return 0; },
      [&](std::string_view) { ++innerCallbackCount; });
  outerLogger.configure(
      "TEST",
      []() { return 0; },
      [&](std::string_view) {
        ++outerCallbackCount;
        COMMS_LOG_NAMED(kInnerContext, ERR, "nested error");
      });

  COMMS_LOG_NAMED(kOuterContext, ERR, "outer error");
  EXPECT_EQ(outerCallbackCount, 1);
  EXPECT_EQ(innerCallbackCount, 0);
  outerLogger.configure("TEST", []() { return 0; }, {});
  innerLogger.configure("TEST", []() { return 0; }, {});
}

TEST(SpdlogLoggerTest, ErrorCallbackExceptionDoesNotEscapeLogCall) {
  constexpr std::string_view kContext = "comms.throwing_callback_test";
  int callbackCount = 0;
  auto& logger = getSpdlogLogger(kContext);
  logger.configure(
      "TEST",
      []() { return 0; },
      [&](std::string_view) {
        ++callbackCount;
        throw std::runtime_error{"callback failure"};
      });

  EXPECT_NO_THROW(COMMS_LOG_NAMED(kContext, ERR, "first error"));
  EXPECT_NO_THROW(COMMS_LOG_NAMED(kContext, ERR, "second error"));
  EXPECT_EQ(callbackCount, 2);
  logger.configure("TEST", []() { return 0; }, {});
}

TEST(SpdlogLoggerTest, FileOpenFailureIncludesPath) {
  constexpr std::string_view kLogPath = "/proc/comms_spdlog_missing/logger.log";

  try {
    getSpdlogLogger("comms.file_failure_test").configureOutput(kLogPath);
    FAIL() << "Expected configureOutput to reject an invalid path";
  } catch (const spdlog::spdlog_ex& error) {
    EXPECT_NE(std::string{error.what()}.find(kLogPath), std::string::npos);
  }
}
