// Copyright (c) Meta Platforms, Inc. and affiliates.

// Verify that build-wide spdlog configuration is independent of include order.
#include <spdlog/spdlog.h>

#include "comms/utils/logger/SpdlogLogger.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "comms/utils/logger/CommsLogFormatter.h"
#include "comms/utils/logger/CudaLog.h"

using meta::comms::logger::getSpdlogLogger;

namespace meta::comms::logger::testing {
bool holdAsyncThreadPoolLeaseForTesting(const std::function<void()>& callback);
void waitForAsyncThreadPoolShutdownForTesting();
bool asyncThreadPoolLeaseAvailableForTesting();
} // namespace meta::comms::logger::testing

/*
 * Each instance owns a private directory. Concurrent copies of this binary --
 * how stress runs execute -- would otherwise share one path under TempDir() and
 * unlink each other's log file while the sink still holds the old inode.
 */
class ScopedTestFile {
 public:
  explicit ScopedTestFile(std::string filename)
      : directory_{makeUniqueDirectory()},
        path_{directory_ / std::move(filename)} {}

  ~ScopedTestFile() {
    removeNoexcept();
  }

  const std::filesystem::path& path() const {
    return path_;
  }

 private:
  static std::filesystem::path makeUniqueDirectory() {
    const auto pattern =
        (std::filesystem::path{testing::TempDir()} / "comms_spdlog_ut_XXXXXX")
            .string();
    std::vector<char> buffer{pattern.begin(), pattern.end()};
    buffer.push_back('\0');
    if (::mkdtemp(buffer.data()) == nullptr) {
      throw std::runtime_error{
          "Failed to create a temporary directory under " +
          std::string{testing::TempDir()}};
    }
    return std::filesystem::path{buffer.data()};
  }

  void removeNoexcept() noexcept {
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
  }

  std::filesystem::path directory_;
  std::filesystem::path path_;
};

std::string readFile(const std::filesystem::path& path) {
  std::ifstream file{path};
  return {
      std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
}

bool waitForFileToContain(
    const std::filesystem::path& path,
    std::string_view message) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds{5};
  do {
    if (readFile(path).find(message) != std::string::npos) {
      return true;
    }
    /* sleep override */
    std::this_thread::sleep_for(std::chrono::milliseconds{10});
  } while (std::chrono::steady_clock::now() < deadline);
  return false;
}

class LogLevelRestoringTest : public testing::Test {
 protected:
  void TearDown() override {
    const auto resetLogger = [](auto& logger, std::string prefix) {
      logger.configure(std::move(prefix), []() { return 0; }, {}, false);
      logger.set_level(spdlog::level::info);
    };
    resetLogger(getSpdlogLogger(), "COMMS");
    resetLogger(getSpdlogLogger("comms.paired_test"), "PAIRED");
    resetLogger(getSpdlogLogger("comms.named_only_test"), "NAMED_ONLY");
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

TEST(SpdlogLoggerTest, SupportsLoggerExpressionDbg5Stream) {
  auto& logger = getSpdlogLogger("comms.dbg5_stream_test");
  logger.set_level(spdlog::level::off);

  COMMS_LOGGER_STREAM(logger, DBG5) << "suppressed trace message";
}

TEST_F(LogLevelRestoringTest, ConfiguresSharedAndNamedLoggersTogether) {
  constexpr std::string_view kNamedLoggerName{"comms.paired_test"};
  std::vector<std::string> errors;

  meta::comms::logger::configureCommsAndNamedSpdlogLoggers(
      kNamedLoggerName,
      "NAMED",
      "",
      []() { return 0; },
      [&](std::string_view message) { errors.emplace_back(message); },
      false,
      spdlog::level::err);

  COMMS_LOG(ERR, "shared error");
  COMMS_LOG_NAMED(kNamedLoggerName, ERR, "named error");

  EXPECT_EQ(errors, (std::vector<std::string>{"shared error", "named error"}));
}

TEST_F(LogLevelRestoringTest, ConfiguresOnlyNamedLoggerWhenRequested) {
  constexpr std::string_view kNamedLoggerName{"comms.named_only_test"};
  std::vector<std::string> sharedErrors;
  std::vector<std::string> namedErrors;
  getSpdlogLogger().configure(
      "SHARED",
      []() { return 0; },
      [&](std::string_view message) { sharedErrors.emplace_back(message); },
      false);
  getSpdlogLogger().set_level(spdlog::level::err);

  meta::comms::logger::configureCommsAndNamedSpdlogLoggers(
      kNamedLoggerName,
      "NAMED_ONLY",
      "",
      []() { return 0; },
      [&](std::string_view message) { namedErrors.emplace_back(message); },
      false,
      spdlog::level::err,
      false);

  COMMS_LOG(ERR, "shared error");
  COMMS_LOG_NAMED(kNamedLoggerName, ERR, "named error");

  EXPECT_EQ(sharedErrors, (std::vector<std::string>{"shared error"}));
  EXPECT_EQ(namedErrors, (std::vector<std::string>{"named error"}));
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

TEST(SpdlogLoggerTest, AsyncErrorReachesFileWithoutExplicitFlush) {
  constexpr std::string_view kContext = "comms.async_flush_test";
  const ScopedTestFile scopedLogFile{"comms_spdlog_async_flush.log"};
  meta::comms::logger::configureSpdlogLogger(
      kContext,
      "TEST",
      scopedLogFile.path().string(),
      []() { return 0; },
      {},
      true);
  auto& logger = getSpdlogLogger(kContext);
  logger.set_level(spdlog::level::info);

  COMMS_LOG_NAMED(kContext, ERR, "asynchronous error flush");

  EXPECT_TRUE(
      waitForFileToContain(scopedLogFile.path(), "asynchronous error flush"));
}

TEST(SpdlogLoggerTest, AsyncInfoReachesFileViaPeriodicFlush) {
  constexpr std::string_view kContext = "comms.periodic_flush_test";
  const ScopedTestFile scopedLogFile{"comms_spdlog_periodic_flush.log"};
  meta::comms::logger::configureSpdlogLogger(
      kContext,
      "TEST",
      scopedLogFile.path().string(),
      []() { return 0; },
      {},
      true);
  auto& logger = getSpdlogLogger(kContext);
  logger.set_level(spdlog::level::info);

  COMMS_LOG_NAMED(kContext, INFO, "asynchronous periodic flush");

  EXPECT_TRUE(waitForFileToContain(
      scopedLogFile.path(), "asynchronous periodic flush"));
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

TEST_F(LogLevelRestoringTest, DebugIsAvailableWithInfoCompileGate) {
  static_assert(SPDLOG_ACTIVE_LEVEL == SPDLOG_LEVEL_INFO);
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

TEST_F(LogLevelRestoringTest, CudaFacadeFormatsAndGatesMessages) {
  const ScopedTestFile scopedLogFile{"comms_cuda_log.log"};
  auto& logger = getSpdlogLogger();
  logger.configureOutput(scopedLogFile.path().string());
  logger.configure("TEST", []() { return 0; }, {}, false);
  logger.set_level(spdlog::level::warn);

  int evaluationCount = 0;
  COMMS_CUDA_LOG(INFO, "filtered value %d", ++evaluationCount);
  COMMS_CUDA_LOG(WARN, "cuda facade value %d", 7);
  const std::string largeMessage(600, 'x');
  COMMS_CUDA_LOG(WARN, "large cuda facade value %s", largeMessage.c_str());
  logger.flush();

  const auto output = readFile(scopedLogFile.path());
  EXPECT_EQ(evaluationCount, 0);
  EXPECT_NE(output.find("cuda facade value 7"), std::string::npos);
  EXPECT_NE(
      output.find("large cuda facade value " + largeMessage),
      std::string::npos);

  logger.configureOutput({});
  logger.configure("COMMS", []() { return 0; });
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

/*
 * Reproduces an exit without destroy_process_group: the communicator's threads
 * are never joined, so they keep logging while static destructors run. The
 * atexit handler is registered before the first getSpdlogLogger() call, so LIFO
 * ordering runs it after everything the logging singletons would have
 * destroyed.
 */
TEST(SpdlogLoggerTest, UnjoinedThreadCanLogDuringStaticDestruction) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        static std::atomic<bool> destructionStarted{false};
        static std::atomic<bool> backgroundThreadDone{false};
        constexpr std::string_view kNamedContext{"comms.teardown_test"};
        constexpr std::string_view kLateNamedContext{
            "comms.teardown_late_test"};

        std::thread{[=]() {
          while (!destructionStarted.load()) {
            /* sleep override */
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
          }
          COMMS_LOG(WARN, "shared logger survived teardown");
          COMMS_LOG_NAMED(
              kNamedContext, WARN, "named logger survived teardown");
          COMMS_LOG_NAMED(
              kLateNamedContext,
              WARN,
              "late-created named logger survived teardown");
          backgroundThreadDone.store(true);
        }}.detach();

        std::atexit([]() {
          destructionStarted.store(true);
          const auto deadline =
              std::chrono::steady_clock::now() + std::chrono::seconds{5};
          while (!backgroundThreadDone.load() &&
                 std::chrono::steady_clock::now() < deadline) {
            /* sleep override */
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
          }
        });

        const auto configureForStderr = [](auto& logger) {
          // Async configuration also exercises exit-time shutdown of the
          // periodic sink flusher before the logging thread runs.
          logger.configure("TEST", []() { return 0; }, {}, true);
          logger.set_level(spdlog::level::info);
        };
        configureForStderr(getSpdlogLogger());
        configureForStderr(getSpdlogLogger(kNamedContext));
        std::exit(0);
      },
      ::testing::ExitedWithCode(0),
      "shared logger survived teardown(.|\\n)*named logger survived "
      "teardown(.|\\n)*late-created named logger survived teardown");
}

/*
 * Exercises logging from an atexit handler after non-trivial TLS teardown. A
 * maximum-length name prevents small-string optimization from hiding a
 * lifetime error and verifies that the name remains intact.
 */
TEST(SpdlogLoggerTest, ExitingThreadCanLogFromAtexitAfterTlsTeardown) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        const std::string longThreadName(
            meta::comms::logger::kMaxLogThreadNameLength, 'n');
        meta::comms::logger::setSpdlogThreadName(longThreadName);

        auto& logger = getSpdlogLogger();
        logger.configure("TEST", []() { return 0; }, {}, true);
        logger.set_level(spdlog::level::info);

        std::atexit([]() {
          COMMS_LOG(WARN, "logged from atexit on the exiting thread");
        });
        std::exit(0);
      },
      ::testing::ExitedWithCode(0),
      "\\[n{63}\\].*logged from atexit on the exiting thread");
}

TEST(SpdlogLoggerTest, ThreadNameIsTruncatedToStorageCapacity) {
  const std::string overlongName(
      meta::comms::logger::kMaxLogThreadNameLength + 17, 'z');
  meta::comms::logger::setSpdlogThreadName(overlongName);
  EXPECT_EQ(
      meta::comms::logger::getLogThreadName(),
      std::string(meta::comms::logger::kMaxLogThreadNameLength, 'z'));

  meta::comms::logger::setSpdlogThreadName("main");
  EXPECT_EQ(meta::comms::logger::getLogThreadName(), "main");
}

TEST(SpdlogLoggerTest, AsyncLoggingFallsBackWhenThreadPoolIsGone) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        auto& logger = getSpdlogLogger();
        logger.configure("TEST", []() { return 0; }, {}, true);
        logger.set_level(spdlog::level::info);
        // Run shutdown in the logger library's translation unit. Sanitizer
        // builds may link a separate spdlog registry into this test binary.
        meta::comms::logger::shutdownSpdlogForFatal();

        COMMS_LOG(WARN, "delivered after thread pool teardown");
        // flush() must also fall back rather than posting to the dead pool.
        logger.flush();
        std::exit(0);
      },
      ::testing::ExitedWithCode(0),
      "delivered after thread pool teardown");
}

TEST(SpdlogLoggerTest, ShutdownWaitsForActiveLeaseAndRejectsNewLeases) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_EXIT(
      {
        auto& logger = getSpdlogLogger();
        logger.configure("TEST", []() { return 0; }, {}, true);
        logger.set_level(spdlog::level::info);

        std::mutex mutex;
        std::condition_variable cv;
        bool leaseHeld = false;
        bool releaseLease = false;
        std::atomic<bool> shutdownDone{false};

        std::thread leaseHolder{[&]() {
          const auto acquired =
              meta::comms::logger::testing::holdAsyncThreadPoolLeaseForTesting(
                  [&]() {
                    std::unique_lock lock{mutex};
                    leaseHeld = true;
                    cv.notify_all();
                    cv.wait(lock, [&]() { return releaseLease; });
                  });
          if (!acquired) {
            std::_Exit(2);
          }
        }};

        {
          std::unique_lock lock{mutex};
          if (!cv.wait_for(
                  lock, std::chrono::seconds{5}, [&]() { return leaseHeld; })) {
            std::_Exit(3);
          }
        }

        std::thread shutdown{[&]() {
          meta::comms::logger::shutdownSpdlogForFatal();
          shutdownDone.store(true);
        }};
        meta::comms::logger::testing::
            waitForAsyncThreadPoolShutdownForTesting();

        if (shutdownDone.load()) {
          std::_Exit(4);
        }
        if (meta::comms::logger::testing::
                asyncThreadPoolLeaseAvailableForTesting()) {
          std::_Exit(5);
        }
        COMMS_LOG(WARN, "synchronous log while shutdown waits for lease");

        {
          std::lock_guard lock{mutex};
          releaseLease = true;
        }
        cv.notify_all();
        leaseHolder.join();
        shutdown.join();
        if (!shutdownDone.load()) {
          std::_Exit(6);
        }
        std::exit(0);
      },
      ::testing::ExitedWithCode(0),
      "synchronous log while shutdown waits for lease");
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
