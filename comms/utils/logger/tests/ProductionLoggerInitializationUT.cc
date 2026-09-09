// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>

#include <folly/testing/TestUtil.h>
#include <gtest/gtest.h>

#include "comms/ctran/utils/CtranLogUtils.h"
#include "comms/ctran/utils/LogInit.h"
#include "comms/mccl/utils/McclLogger.h"
#include "comms/mccl/utils/Utils.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "comms/utils/logger/SpdlogLogger.h"

namespace {

constexpr int kEnvironmentSetupFailure = 2;
constexpr int kDisabledArgumentWasEvaluated = 3;
constexpr int kOutputValidationFailure = 4;
constexpr int kLoggerConfigurationFailure = 5;

using Initializer = void (*)();
using AsyncStateProbe = bool (*)();
using LogEmitter = void (*)(const std::string&);

struct LoggerProbe {
  const char* directoryName;
  const char* subsystem;
  bool asyncLogging;
  Initializer initialize;
  AsyncStateProbe usesAsyncLogging;
  LogEmitter emit;
  std::string_view expectedRecord;
};

[[noreturn]] void
exitProbe(int exitCode, const char* message, const std::string& logPath = {}) {
  if (message != nullptr) {
    std::fprintf(stderr, "%s\n", message);
    std::fflush(stderr);
  }
  if (!logPath.empty()) {
    std::error_code error;
    std::filesystem::remove_all(
        std::filesystem::path{logPath}.parent_path(), error);
  }
  std::_Exit(exitCode);
}

std::string readFile(const std::string& path) {
  std::ifstream file{path};
  if (!file) {
    throw std::runtime_error{"failed to open " + path};
  }
  return {
      std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
}

[[noreturn]] int failIfEvaluated(const std::string& logPath) {
  exitProbe(
      kDisabledArgumentWasEvaluated,
      "disabled logging argument was evaluated",
      logPath);
}

size_t countOccurrences(std::string_view output, std::string_view token) {
  if (token.empty()) {
    return 0;
  }

  size_t count = 0;
  size_t position = 0;
  while ((position = output.find(token, position)) != std::string_view::npos) {
    ++count;
    position += token.size();
  }
  return count;
}

[[noreturn]] void runLoggerProbe(
    const std::string& logPath,
    const LoggerProbe& probe) {
  if (::setenv("NCCL_DEBUG", "INFO", 1) != 0 ||
      ::setenv("NCCL_DEBUG_SUBSYS", probe.subsystem, 1) != 0 ||
      ::setenv("NCCL_DEBUG_FILE", logPath.c_str(), 1) != 0 ||
      ::setenv("NCCL_DEBUG_LOGGING_ASYNC", probe.asyncLogging ? "1" : "0", 1) !=
          0) {
    exitProbe(
        kEnvironmentSetupFailure,
        "failed to configure logging environment",
        logPath);
  }

  probe.initialize();
  if (probe.usesAsyncLogging() != probe.asyncLogging) {
    exitProbe(
        kLoggerConfigurationFailure,
        "logger async mode did not match NCCL_DEBUG_LOGGING_ASYNC",
        logPath);
  }
  probe.emit(logPath);
  meta::comms::logger::shutdownCommsLogging();

  try {
    const auto output = readFile(logPath);
    if (countOccurrences(output, probe.expectedRecord) != 1) {
      std::fprintf(stderr, "logger probe output:\n");
      std::fwrite(output.data(), sizeof(char), output.size(), stderr);
      std::fprintf(stderr, "\n");
      std::fflush(stderr);
      exitProbe(
          kOutputValidationFailure,
          "logger probe output did not match",
          logPath);
    }
  } catch (const std::exception& error) {
    std::fprintf(stderr, "logger probe output error: %s\n", error.what());
    std::fflush(stderr);
    exitProbe(kOutputValidationFailure, nullptr, logPath);
  } catch (...) {
    exitProbe(
        kOutputValidationFailure, "unknown logger probe output error", logPath);
  }
  exitProbe(0, nullptr, logPath);
}

void initializeCtranLogger() {
  ncclCvarInit();
  ctran::logging::initCtranLogging();
}

bool ctranUsesAsyncLogging() {
  return ctran::logging::getCtranLogger().usesAsyncLogging();
}

void emitCtranCollRecord(const std::string&) {
  CTRAN_LOG_SUBSYS(INFO, COLL, "ctran COLL initialization record");
}

void emitCtranInitAndSuppressedCollRecord(const std::string& logPath) {
  CTRAN_LOG_SUBSYS(INFO, INIT, "ctran INIT initialization record");
  CTRAN_LOG_SUBSYS(
      INFO, COLL, "ctran suppressed record {}", failIfEvaluated(logPath));
}

void initializeMcclLogger() {
  ncclCvarInit();
  mccl::utils::initMcclLogger();
}

bool mcclUsesAsyncLogging() {
  return mccl::logging::getMcclLogger().usesAsyncLogging();
}

void emitMcclCollRecord(const std::string&) {
  MCCL_LOG_SUBSYS(INFO, COLL, "mccl COLL initialization record");
}

void emitMcclInitAndSuppressedCollRecord(const std::string& logPath) {
  MCCL_LOG_SUBSYS(INFO, INIT, "mccl INIT initialization record");
  MCCL_LOG_SUBSYS(
      INFO, COLL, "mccl suppressed record {}", failIfEvaluated(logPath));
}

class ProductionLoggerInitializationDeathTest : public testing::Test {
 protected:
  void SetUp() override {
    originalDeathTestStyle_ = GTEST_FLAG_GET(death_test_style);
    GTEST_FLAG_SET(death_test_style, "threadsafe");
  }

  void TearDown() override {
    GTEST_FLAG_SET(death_test_style, originalDeathTestStyle_);
  }

  void expectLoggerProbe(const LoggerProbe& probe) {
    folly::test::TemporaryDirectory directory{probe.directoryName};
    const auto logPath = (directory.path() / "logger.log").string();
    EXPECT_EXIT(runLoggerProbe(logPath, probe), testing::ExitedWithCode(0), "");
  }

 private:
  std::string originalDeathTestStyle_;
};

TEST_F(
    ProductionLoggerInitializationDeathTest,
    CtranReadsEnvironmentAndDrainsAsyncOutput) {
  expectLoggerProbe(
      {.directoryName = "ctran_logger_initialization",
       .subsystem = "COLL",
       .asyncLogging = true,
       .initialize = initializeCtranLogger,
       .usesAsyncLogging = ctranUsesAsyncLogging,
       .emit = emitCtranCollRecord,
       .expectedRecord = "ctran COLL initialization record"});
}

TEST_F(
    ProductionLoggerInitializationDeathTest,
    CtranSuppressesDisabledSubsystemWithoutEvaluatingArguments) {
  expectLoggerProbe(
      {.directoryName = "ctran_logger_filter",
       .subsystem = "INIT",
       .asyncLogging = false,
       .initialize = initializeCtranLogger,
       .usesAsyncLogging = ctranUsesAsyncLogging,
       .emit = emitCtranInitAndSuppressedCollRecord,
       .expectedRecord = "ctran INIT initialization record"});
}

TEST_F(
    ProductionLoggerInitializationDeathTest,
    McclReadsEnvironmentAndDrainsAsyncOutput) {
  expectLoggerProbe(
      {.directoryName = "mccl_logger_initialization",
       .subsystem = "COLL",
       .asyncLogging = true,
       .initialize = initializeMcclLogger,
       .usesAsyncLogging = mcclUsesAsyncLogging,
       .emit = emitMcclCollRecord,
       .expectedRecord = "mccl COLL initialization record"});
}

TEST_F(
    ProductionLoggerInitializationDeathTest,
    McclSuppressesDisabledSubsystemWithoutEvaluatingArguments) {
  expectLoggerProbe(
      {.directoryName = "mccl_logger_filter",
       .subsystem = "INIT",
       .asyncLogging = false,
       .initialize = initializeMcclLogger,
       .usesAsyncLogging = mcclUsesAsyncLogging,
       .emit = emitMcclInitAndSuppressedCollRecord,
       .expectedRecord = "mccl INIT initialization record"});
}

} // namespace
