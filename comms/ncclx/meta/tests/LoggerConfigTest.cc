// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <folly/testing/TestUtil.h>
#include <gtest/gtest.h>

#include "comms/utils/logger/LogTypes.h"
#include "comms/utils/logger/SpdlogLogger.h"
#include "debug.h"
#include "env.h"
#include "meta/NcclxLogger.h"
#include "param.h"

namespace {

static_assert(noexcept(initNcclLogger()));
static_assert(noexcept(ncclRefreshDebugInitInternal()));

class ScopedEnvironmentVariable {
 public:
  ScopedEnvironmentVariable(std::string name, std::optional<std::string> value)
      : name_{std::move(name)} {
    if (const char* previousValue = std::getenv(name_.c_str())) {
      previousValue_ = previousValue;
    }

    const int result = value.has_value()
        ? ::setenv(name_.c_str(), value->c_str(), 1)
        : ::unsetenv(name_.c_str());
    if (result != 0) {
      throw std::runtime_error{"failed to update test environment variable"};
    }
  }

  ScopedEnvironmentVariable(const ScopedEnvironmentVariable&) = delete;
  ScopedEnvironmentVariable& operator=(const ScopedEnvironmentVariable&) =
      delete;
  ScopedEnvironmentVariable(ScopedEnvironmentVariable&&) = delete;
  ScopedEnvironmentVariable& operator=(ScopedEnvironmentVariable&&) = delete;

  ~ScopedEnvironmentVariable() {
    if (previousValue_.has_value()) {
      ::setenv(name_.c_str(), previousValue_->c_str(), 1);
    } else {
      ::unsetenv(name_.c_str());
    }
  }

 private:
  std::string name_;
  std::optional<std::string> previousValue_;
};

std::string readFile(const std::string& path) {
  std::ifstream file{path};
  return {
      std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
}

} // namespace

TEST(LoggerConfigTest, PreservesLazyNativeConfigurationBeforePluginInit) {
  folly::test::TemporaryDirectory directory{"nccl_logger_pre_plugin_config"};
  const auto configPath = directory.path() / "nccl.conf";
  const auto logPath = directory.path() / "nccl.log";
  {
    std::ofstream config{configPath};
    ASSERT_TRUE(config.is_open());
    config << "NCCL_DEBUG=info\n"
           << "NCCL_DEBUG_SUBSYS=COLL\n"
           << "NCCL_DEBUG_FILE=" << logPath.string() << "\n"
           << "NCCL_DEBUG_LOGGING_ASYNC=0\n";
  }

  const ScopedEnvironmentVariable debug{"NCCL_DEBUG", std::nullopt};
  const ScopedEnvironmentVariable debugSubsys{
      "NCCL_DEBUG_SUBSYS", std::nullopt};
  const ScopedEnvironmentVariable debugFile{"NCCL_DEBUG_FILE", std::nullopt};
  const ScopedEnvironmentVariable asyncLogging{
      "NCCL_DEBUG_LOGGING_ASYNC", std::nullopt};
  const ScopedEnvironmentVariable configFile{
      "NCCL_CONF_FILE", configPath.string()};

  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      ([&]() {
        ncclResetDebugInitInternal();
        initEnv();
        INFO(NCCL_COLL, "%s", "native record before plugin initialization");

        auto& logger = meta::comms::logger::getSpdlogLogger(
            ncclx::logging::kNcclxLoggerName);
        logger.flush();
        const auto output = readFile(logPath.string());
        EXPECT_NE(
            output.find("native record before plugin initialization"),
            std::string::npos);
        EXPECT_TRUE(
            meta::comms::logger::isEnabledSubSystemBitwise(
                meta::comms::logger::COLL));
        EXPECT_FALSE(
            meta::comms::logger::isEnabledSubSystemBitwise(
                meta::comms::logger::INIT));
        std::_Exit(::testing::Test::HasFailure() ? 3 : 0);
      }()),
      ::testing::ExitedWithCode(0),
      "");
}

TEST(LoggerConfigTest, ResetBeforePluginRefreshesNativeAndSpdlogLoggers) {
  folly::test::TemporaryDirectory directory{"nccl_logger_pre_plugin_reset"};
  const auto initialLogPath = directory.path() / "initial.log";
  const auto resetLogPath = directory.path() / "reset.log";

  const ScopedEnvironmentVariable debug{"NCCL_DEBUG", "INFO"};
  const ScopedEnvironmentVariable debugSubsys{"NCCL_DEBUG_SUBSYS", "COLL"};
  const ScopedEnvironmentVariable debugFile{
      "NCCL_DEBUG_FILE", initialLogPath.string()};
  const ScopedEnvironmentVariable asyncLogging{"NCCL_DEBUG_LOGGING_ASYNC", "0"};
  const ScopedEnvironmentVariable configFile{"NCCL_CONF_FILE", std::nullopt};

  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      ([&]() {
        initEnv();
        INFO(NCCL_COLL, "%s", "native record before pre-plugin reset");
        NCCLX_LOG_INFO("direct record before pre-plugin reset");
        auto& logger = meta::comms::logger::getSpdlogLogger(
            ncclx::logging::kNcclxLoggerName);
        logger.flush();

        if (setenv("NCCL_DEBUG_FILE", resetLogPath.c_str(), 1) != 0) {
          std::_Exit(2);
        }
        ncclResetDebugInitInternal();

        NCCLX_LOG_INFO("direct record after pre-plugin reset");
        INFO(NCCL_COLL, "%s", "native record after pre-plugin reset");
        logger.flush();

        const auto initialOutput = readFile(initialLogPath.string());
        const auto resetOutput = readFile(resetLogPath.string());
        EXPECT_NE(
            initialOutput.find("native record before pre-plugin reset"),
            std::string::npos);
        EXPECT_NE(
            initialOutput.find("direct record before pre-plugin reset"),
            std::string::npos);
        EXPECT_EQ(
            initialOutput.find("direct record after pre-plugin reset"),
            std::string::npos);
        EXPECT_NE(
            resetOutput.find("direct record after pre-plugin reset"),
            std::string::npos);
        EXPECT_NE(
            resetOutput.find("native record after pre-plugin reset"),
            std::string::npos);
        std::_Exit(::testing::Test::HasFailure() ? 3 : 0);
      }()),
      ::testing::ExitedWithCode(0),
      "");
}

TEST(LoggerConfigTest, UsesNativeNcclConfigurationSources) {
  folly::test::TemporaryDirectory directory{"nccl_logger_config"};
  const auto configPath = directory.path() / "nccl.conf";
  const auto logPath = directory.path() / "nccl.log";
  const auto resetLogPath = directory.path() / "nccl-reset.log";
  const auto sharedLogPath = directory.path() / "shared.log";
  {
    std::ofstream config{configPath};
    ASSERT_TRUE(config.is_open());
    config << "NCCL_DEBUG=info\n"
           << "NCCL_DEBUG_SUBSYS=COLL\n"
           << "NCCL_DEBUG_FILE=" << logPath.string() << "\n"
           << "NCCL_DEBUG_LOGGING_ASYNC=0\n";
  }

  const ScopedEnvironmentVariable debug{"NCCL_DEBUG", std::nullopt};
  const ScopedEnvironmentVariable debugSubsys{
      "NCCL_DEBUG_SUBSYS", std::nullopt};
  const ScopedEnvironmentVariable debugFile{"NCCL_DEBUG_FILE", std::nullopt};
  const ScopedEnvironmentVariable asyncLogging{
      "NCCL_DEBUG_LOGGING_ASYNC", std::nullopt};
  const ScopedEnvironmentVariable configFile{
      "NCCL_CONF_FILE", configPath.string()};

  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      ([&]() {
        ncclResetDebugInitInternal();
        initEnv();

        auto& logger = meta::comms::logger::getSpdlogLogger(
            ncclx::logging::kNcclxLoggerName);
        auto& sharedLogger = meta::comms::logger::getSpdlogLogger();
        EXPECT_TRUE(logger.should_log(spdlog::level::info));
        EXPECT_FALSE(logger.should_log(spdlog::level::debug));
        EXPECT_FALSE(logger.usesAsyncLogging());
        EXPECT_TRUE(sharedLogger.should_log(spdlog::level::info));
        EXPECT_FALSE(sharedLogger.should_log(spdlog::level::debug));
        EXPECT_FALSE(sharedLogger.usesAsyncLogging());
        EXPECT_TRUE(
            meta::comms::logger::isEnabledSubSystemBitwise(
                meta::comms::logger::COLL));
        EXPECT_FALSE(
            meta::comms::logger::isEnabledSubSystemBitwise(
                meta::comms::logger::INIT));

        INFO(
            NCCL_COLL,
            "%s",
            "native logger configuration came from NCCL_CONF_FILE");
        NCCLX_LOG_INFO("logger configuration came from NCCL_CONF_FILE");
        COMMS_LOG_INFO("shared logger configuration came from NCCL_CONF_FILE");
        logger.flush();
        sharedLogger.flush();
        std::ifstream logFile{logPath};
        const std::string output{
            std::istreambuf_iterator<char>{logFile},
            std::istreambuf_iterator<char>{}};
        EXPECT_NE(
            output.find("logger configuration came from NCCL_CONF_FILE"),
            std::string::npos);
        EXPECT_NE(
            output.find("shared logger configuration came from NCCL_CONF_FILE"),
            std::string::npos);
        EXPECT_NE(
            output.find("native logger configuration came from NCCL_CONF_FILE"),
            std::string::npos);

        sharedLogger.reconfigure(
            "SHARED",
            sharedLogPath.string(),
            []() { return 0; },
            {},
            false,
            spdlog::level::info);

        if (setenv("NCCL_DEBUG", "WARN", 1) != 0 ||
            setenv("NCCL_DEBUG_SUBSYS", "INIT", 1) != 0 ||
            ncclEnvPluginInit() != ncclSuccess) {
          std::_Exit(2);
        }

        NCCLX_LOG_WARN("direct record after plugin reset");
        logger.flush();
        WARN("native record after plugin reset");
        logger.flush();
        const auto postPluginOutput = readFile(logPath.string());
        EXPECT_NE(
            postPluginOutput.find("direct record after plugin reset"),
            std::string::npos);
        EXPECT_NE(
            postPluginOutput.find("native record after plugin reset"),
            std::string::npos);
        EXPECT_NE(
            postPluginOutput.find(
                "logger configuration came from NCCL_CONF_FILE"),
            std::string::npos);
        EXPECT_NE(
            postPluginOutput.find(
                "shared logger configuration came from NCCL_CONF_FILE"),
            std::string::npos);
        EXPECT_NE(
            postPluginOutput.find(
                "native logger configuration came from NCCL_CONF_FILE"),
            std::string::npos);

        EXPECT_TRUE(logger.should_log(spdlog::level::warn));
        EXPECT_FALSE(logger.should_log(spdlog::level::info));
        EXPECT_TRUE(sharedLogger.should_log(spdlog::level::warn));
        EXPECT_TRUE(sharedLogger.should_log(spdlog::level::info));
        EXPECT_TRUE(
            meta::comms::logger::isEnabledSubSystemBitwise(
                meta::comms::logger::INIT));
        EXPECT_FALSE(
            meta::comms::logger::isEnabledSubSystemBitwise(
                meta::comms::logger::COLL));

        COMMS_LOG_INFO("shared logger retained its owner configuration");
        std::ifstream sharedLogFile{sharedLogPath};
        const std::string sharedOutput{
            std::istreambuf_iterator<char>{sharedLogFile},
            std::istreambuf_iterator<char>{}};
        EXPECT_NE(
            sharedOutput.find("shared logger retained its owner configuration"),
            std::string::npos);

        {
          std::ofstream resetLog{resetLogPath};
          resetLog << "previous generation must be truncated\n";
        }
        if (setenv("NCCL_DEBUG_FILE", resetLogPath.c_str(), 1) != 0) {
          std::_Exit(4);
        }
        ncclResetDebugInitInternal();
        NCCLX_LOG_WARN("direct record after file reset");
        logger.flush();
        WARN("native record after file reset");
        logger.flush();
        const auto postFileResetOutput = readFile(resetLogPath.string());
        EXPECT_EQ(
            postFileResetOutput.find("previous generation must be truncated"),
            std::string::npos);
        EXPECT_NE(
            postFileResetOutput.find("direct record after file reset"),
            std::string::npos);
        EXPECT_NE(
            postFileResetOutput.find("native record after file reset"),
            std::string::npos);

        if (setenv("NCCL_DEBUG", "INFO", 1) != 0 ||
            setenv("NCCL_DEBUG_SUBSYS", "COLL", 1) != 0) {
          std::_Exit(4);
        }
        ncclResetDebugInitInternal();

        EXPECT_TRUE(logger.should_log(spdlog::level::info));
        EXPECT_TRUE(sharedLogger.should_log(spdlog::level::info));
        EXPECT_TRUE(
            meta::comms::logger::isEnabledSubSystemBitwise(
                meta::comms::logger::COLL));
        EXPECT_FALSE(
            meta::comms::logger::isEnabledSubSystemBitwise(
                meta::comms::logger::INIT));

        std::atomic<bool> start{false};
        std::thread loggingThread{[&] {
          while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
          }
          for (int iteration = 0; iteration < 64; ++iteration) {
            NCCLX_LOG_INFO("concurrent logger reset {}", iteration);
            INFO(
                NCCL_COLL,
                "%s %d",
                "concurrent native logger reset",
                iteration);
          }
        }};
        start.store(true, std::memory_order_release);
        for (int iteration = 0; iteration < 64; ++iteration) {
          ncclResetDebugInitInternal();
        }
        loggingThread.join();

        EXPECT_TRUE(logger.should_log(spdlog::level::info));
        EXPECT_TRUE(
            meta::comms::logger::isEnabledSubSystemBitwise(
                meta::comms::logger::COLL));
        std::_Exit(::testing::Test::HasFailure() ? 3 : 0);
      }()),
      ::testing::ExitedWithCode(0),
      "");
}

TEST(LoggerConfigTest, PluginRefreshTruncatesNewDebugFile) {
  folly::test::TemporaryDirectory directory{"nccl_logger_plugin_file"};
  const auto initialLogPath = directory.path() / "initial.log";
  const auto pluginLogPath = directory.path() / "plugin.log";
  {
    std::ofstream pluginLog{pluginLogPath};
    ASSERT_TRUE(pluginLog.is_open());
    pluginLog << "previous destination content must be truncated\n";
  }

  const ScopedEnvironmentVariable debug{"NCCL_DEBUG", "INFO"};
  const ScopedEnvironmentVariable debugSubsys{"NCCL_DEBUG_SUBSYS", "COLL"};
  const ScopedEnvironmentVariable debugFile{
      "NCCL_DEBUG_FILE", initialLogPath.string()};
  const ScopedEnvironmentVariable asyncLogging{"NCCL_DEBUG_LOGGING_ASYNC", "0"};
  const ScopedEnvironmentVariable configFile{"NCCL_CONF_FILE", std::nullopt};

  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      ([&]() {
        initEnv();
        INFO(NCCL_COLL, "%s", "native record before plugin path refresh");
        NCCLX_LOG_INFO("direct record before plugin path refresh");
        auto& logger = meta::comms::logger::getSpdlogLogger(
            ncclx::logging::kNcclxLoggerName);
        logger.flush();

        if (setenv("NCCL_DEBUG_FILE", pluginLogPath.c_str(), 1) != 0 ||
            ncclEnvPluginInit() != ncclSuccess) {
          std::_Exit(2);
        }
        NCCLX_LOG_INFO("direct record after plugin path refresh");
        INFO(NCCL_COLL, "%s", "native record after plugin path refresh");
        logger.flush();

        const auto initialOutput = readFile(initialLogPath.string());
        const auto pluginOutput = readFile(pluginLogPath.string());
        EXPECT_NE(
            initialOutput.find("native record before plugin path refresh"),
            std::string::npos);
        EXPECT_NE(
            initialOutput.find("direct record before plugin path refresh"),
            std::string::npos);
        EXPECT_EQ(
            pluginOutput.find("previous destination content must be truncated"),
            std::string::npos);
        EXPECT_NE(
            pluginOutput.find("direct record after plugin path refresh"),
            std::string::npos);
        EXPECT_NE(
            pluginOutput.find("native record after plugin path refresh"),
            std::string::npos);
        std::_Exit(::testing::Test::HasFailure() ? 3 : 0);
      }()),
      ::testing::ExitedWithCode(0),
      "");
}

TEST(LoggerConfigTest, PluginInitializationRefreshesNativeAndSpdlogGates) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  folly::test::TemporaryDirectory directory{"nccl_plugin_logger_config"};
  const auto capturedStdoutPath = directory.path() / "stdout.log";
  EXPECT_EXIT(
      {
        if (std::freopen(capturedStdoutPath.c_str(), "w", stdout) == nullptr) {
          std::_Exit(2);
        }
        if (setenv("NCCL_DEBUG", "WARN", 1) != 0 ||
            setenv("NCCL_DEBUG_SUBSYS", "INIT", 1) != 0 ||
            setenv("NCCL_DEBUG_LOGGING_ASYNC", "0", 1) != 0 ||
            unsetenv("NCCL_DEBUG_FILE") != 0 ||
            unsetenv("NCCL_CONF_FILE") != 0) {
          std::_Exit(2);
        }
        initEnv();

        INFO(NCCL_COLL, "suppressed before plugin initialization");
        if (setenv("NCCL_DEBUG", "INFO", 1) != 0 ||
            setenv("NCCL_DEBUG_SUBSYS", "COLL", 1) != 0 ||
            ncclEnvPluginInit() != ncclSuccess) {
          std::_Exit(3);
        }

        INFO(NCCL_COLL, "native and spdlog gates refreshed together");
        std::fflush(stdout);
        const auto output = readFile(capturedStdoutPath.string());
        EXPECT_NE(
            output.find("native and spdlog gates refreshed together"),
            std::string::npos);
        EXPECT_EQ(
            output.find("suppressed before plugin initialization"),
            std::string::npos);
        std::_Exit(::testing::Test::HasFailure() ? 4 : 0);
      },
      ::testing::ExitedWithCode(0),
      "");
}

TEST(LoggerConfigTest, InvalidDebugFileFallsBackToStdout) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  folly::test::TemporaryDirectory directory{"nccl_invalid_log_path"};
  const auto parentFile = directory.path() / "not_a_directory";
  const auto capturedStdoutPath = directory.path() / "stdout.log";
  ASSERT_TRUE(std::ofstream{parentFile}.is_open());
  const auto invalidLogPath = parentFile / "nccl.log";
  const ScopedEnvironmentVariable debug{"NCCL_DEBUG", "WARN"};
  const ScopedEnvironmentVariable debugSubsys{"NCCL_DEBUG_SUBSYS", "INIT"};
  const ScopedEnvironmentVariable debugFile{
      "NCCL_DEBUG_FILE", invalidLogPath.string()};
  const ScopedEnvironmentVariable asyncLogging{"NCCL_DEBUG_LOGGING_ASYNC", "0"};
  const ScopedEnvironmentVariable configFile{"NCCL_CONF_FILE", std::nullopt};

  EXPECT_EXIT(
      {
        if (std::freopen(capturedStdoutPath.c_str(), "w", stdout) == nullptr) {
          std::_Exit(2);
        }
        initEnv();
        EXPECT_TRUE(
            meta::comms::logger::isEnabledSubSystemBitwise(
                meta::comms::logger::INIT));
        EXPECT_FALSE(
            meta::comms::logger::isEnabledSubSystemBitwise(
                meta::comms::logger::COLL));
        NCCLX_LOG_WARN("invalid debug file fell back to stdout");
        COMMS_LOG_WARN("shared logger also fell back to stdout");
        std::fflush(stdout);
        const auto output = readFile(capturedStdoutPath.string());
        std::_Exit(
            output.find("invalid debug file fell back to stdout") !=
                        std::string::npos &&
                    output.find("shared logger also fell back to stdout") !=
                        std::string::npos
                ? 0
                : 3);
      },
      ::testing::ExitedWithCode(0),
      "");
}
