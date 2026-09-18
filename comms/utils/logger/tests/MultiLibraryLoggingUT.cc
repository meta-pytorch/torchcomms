// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/logger/SpdlogLogger.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

using meta::comms::logger::configureCommsAndNamedSpdlogLoggers;
using meta::comms::logger::getSpdlogLogger;
using meta::comms::logger::kCommsLoggerName;

namespace {

/*
 * Mirrors the logger names and prefixes each library passes to
 * configureCommsAndNamedSpdlogLoggers from its own initialization path:
 * ncclx/src/misc/param.cc, ctran/utils/LogInit.cc and mccl/utils/Utils.cpp.
 * They are duplicated here rather than included because the ncclx facade has no
 * lightweight build target, and depending on ctran/mccl would invert the
 * logger's layering.
 */
struct Library {
  std::string_view loggerName;
  std::string_view prefix;
};

constexpr Library kNcclx{"comms.ncclx", "NCCL"};
constexpr Library kCtran{"comms.ctran", "CTRAN"};
constexpr Library kMccl{"comms.mccl", "MCCL"};

constexpr std::string_view kSharedPrefix{"COMM"};

enum class CapturedStream { Stdout, Stderr };

class ScopedStreamCapture {
 public:
  explicit ScopedStreamCapture(CapturedStream stream) : stream_{stream} {
    if (stream_ == CapturedStream::Stdout) {
      testing::internal::CaptureStdout();
    } else {
      testing::internal::CaptureStderr();
    }
  }

  ~ScopedStreamCapture() {
    if (!released_) {
      (void)release();
    }
  }

  ScopedStreamCapture(const ScopedStreamCapture&) = delete;
  ScopedStreamCapture& operator=(const ScopedStreamCapture&) = delete;

  std::string release() {
    if (released_) {
      return {};
    }
    auto output = stream_ == CapturedStream::Stdout
        ? testing::internal::GetCapturedStdout()
        : testing::internal::GetCapturedStderr();
    released_ = true;
    return output;
  }

 private:
  CapturedStream stream_;
  bool released_{false};
};

class ScopedTestFile {
 public:
  explicit ScopedTestFile(std::string filename)
      : directory_{makeUniqueDirectory()},
        path_{directory_ / std::move(filename)} {}

  ~ScopedTestFile() {
    cleanup();
  }

  void cleanup() noexcept {
    std::error_code error;
    std::filesystem::remove_all(directory_, error);
  }

  ScopedTestFile(const ScopedTestFile&) = delete;
  ScopedTestFile& operator=(const ScopedTestFile&) = delete;

  const std::filesystem::path& path() const {
    return path_;
  }

  std::string read() const {
    std::ifstream file{path_};
    if (!file) {
      throw std::runtime_error{"failed to open " + path_.string()};
    }
    return {
        std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
  }

 private:
  static std::filesystem::path makeUniqueDirectory() {
    const auto pattern = (std::filesystem::path{testing::TempDir()} /
                          "comms_multi_library_ut_XXXXXX")
                             .string();
    std::vector<char> buffer{pattern.begin(), pattern.end()};
    buffer.push_back('\0');
    if (::mkdtemp(buffer.data()) == nullptr) {
      throw std::runtime_error{
          "failed to create a temporary directory under " +
          std::string{testing::TempDir()}};
    }
    return std::filesystem::path{buffer.data()};
  }

  std::filesystem::path directory_;
  std::filesystem::path path_;
};

class MultiLibraryLoggingTest : public testing::Test {
 protected:
  void SetUp() override {
    deathTestStyle_ = GTEST_FLAG_GET(death_test_style);
    resetLoggers();
  }

  // Synchronous logging keeps every assertion below free of flush waits.
  void initLibrary(
      const Library& library,
      std::string_view logFilePath,
      spdlog::level::level_enum level,
      std::function<void(std::string_view)> errorCallback = {}) {
    configureCommsAndNamedSpdlogLoggers(
        library.loggerName,
        std::string{library.prefix},
        logFilePath,
        []() { return 0; },
        std::move(errorCallback),
        false,
        level);
  }

  void TearDown() override {
    GTEST_FLAG_SET(death_test_style, deathTestStyle_);
    resetLoggers("");
  }

  void cleanupResetLogFile() {
    resetLogFile_.cleanup();
  }

  void resetLoggers() {
    resetLoggers(resetLogFile_.path().string());
  }

  void resetLoggers(std::string_view logFilePath) {
    for (const auto name :
         {kCommsLoggerName,
          kNcclx.loggerName,
          kCtran.loggerName,
          kMccl.loggerName}) {
      auto& logger = getSpdlogLogger(name);
      logger.configureOutput(logFilePath);
      logger.configure("COMMS", []() { return 0; }, {}, false);
      logger.set_level(spdlog::level::info);
    }
  }

 private:
  ScopedTestFile resetLogFile_{"comms_multi_library_reset.log"};
  std::string deathTestStyle_;
};

TEST_F(MultiLibraryLoggingTest, EachLibraryInitTagsItsOwnLoggerAndTheShared) {
  for (const auto& library : {kNcclx, kCtran, kMccl}) {
    SCOPED_TRACE(library.loggerName);
    resetLoggers();
    const ScopedTestFile logFile{"comms_multi_library_prefix.log"};
    initLibrary(library, logFile.path().string(), spdlog::level::info);

    ScopedStreamCapture stderrCapture{CapturedStream::Stderr};
    COMMS_LOG_NAMED(library.loggerName, WARN, "library message");
    COMMS_LOG(WARN, "shared message");
    (void)stderrCapture.release();
    resetLoggers();

    const auto output = logFile.read();
    EXPECT_NE(
        output.find(std::string{library.prefix} + " WARN library message"),
        std::string::npos)
        << library.loggerName << " output: " << output;
    EXPECT_NE(
        output.find(std::string{kSharedPrefix} + " WARN shared message"),
        std::string::npos)
        << library.loggerName << " output: " << output;
  }
}

/*
 * Every library init also reconfigures the shared "comms" logger, so in a
 * process that initializes more than one library the last init owns the shared
 * logger's level, destination and error callback.
 */
TEST_F(MultiLibraryLoggingTest, LastLibraryInitOwnsTheSharedLogger) {
  const ScopedTestFile ncclxLogFile{"comms_multi_library_ncclx.log"};
  const ScopedTestFile mcclLogFile{"comms_multi_library_mccl.log"};
  const auto ncclxErrors = std::make_shared<std::vector<std::string>>();
  const auto mcclErrors = std::make_shared<std::vector<std::string>>();

  initLibrary(
      kNcclx,
      ncclxLogFile.path().string(),
      spdlog::level::err,
      [ncclxErrors](std::string_view message) {
        ncclxErrors->emplace_back(message);
      });
  initLibrary(
      kMccl,
      mcclLogFile.path().string(),
      spdlog::level::info,
      [mcclErrors](std::string_view message) {
        mcclErrors->emplace_back(message);
      });

  ScopedStreamCapture stderrCapture{CapturedStream::Stderr};
  COMMS_LOG(ERR, "shared error after both inits");
  COMMS_LOG(INFO, "shared info after both inits");
  (void)stderrCapture.release();
  resetLoggers();

  EXPECT_TRUE(ncclxErrors->empty());
  EXPECT_EQ(
      *mcclErrors, (std::vector<std::string>{"shared error after both inits"}));
  EXPECT_EQ(ncclxLogFile.read().find("after both inits"), std::string::npos);
  const auto mcclOutput = mcclLogFile.read();
  EXPECT_NE(
      mcclOutput.find("shared error after both inits"), std::string::npos);
  EXPECT_NE(mcclOutput.find("shared info after both inits"), std::string::npos);
}

/*
 * With NCCL_DEBUG_FILE unset, library init points the logger at stdout, which
 * matches the `FILE* ncclDebugFile = stdout` default in ncclx's debug.cc and
 * upstream NCCL.
 */
TEST_F(MultiLibraryLoggingTest, InitWithoutDebugFileWritesToStdout) {
  constexpr std::string_view kMessage{"ncclx warn without debug file"};
  ScopedStreamCapture stdoutCapture{CapturedStream::Stdout};
  ScopedStreamCapture stderrCapture{CapturedStream::Stderr};
  initLibrary(kNcclx, "", spdlog::level::info);
  COMMS_LOG_NAMED(kNcclx.loggerName, WARN, "{}", kMessage);
  const auto stdoutOutput = stdoutCapture.release();
  const auto stderrOutput = stderrCapture.release();

  EXPECT_NE(stdoutOutput.find(kMessage), std::string::npos)
      << "captured stdout: " << stdoutOutput;
  EXPECT_EQ(stderrOutput.find(kMessage), std::string::npos)
      << "captured stderr: " << stderrOutput;
}

/*
 * NCCL_DEBUG_FILE routing: every initialized library logger writes to the one
 * file, and only warnings and errors are mirrored to stderr.
 */
TEST_F(MultiLibraryLoggingTest, InitializedLibraryLoggersShareTheDebugFile) {
  const ScopedTestFile logFile{"comms_multi_library_shared_file.log"};
  const auto logPath = logFile.path().string();
  initLibrary(kNcclx, logPath, spdlog::level::info);
  initLibrary(kCtran, logPath, spdlog::level::info);
  initLibrary(kMccl, logPath, spdlog::level::info);

  ScopedStreamCapture stderrCapture{CapturedStream::Stderr};
  for (const auto& library : {kNcclx, kCtran, kMccl}) {
    COMMS_LOG_NAMED(library.loggerName, INFO, "{} info line", library.prefix);
    COMMS_LOG_NAMED(library.loggerName, WARN, "{} warn line", library.prefix);
  }
  const auto stderrOutput = stderrCapture.release();
  resetLoggers();

  const auto output = logFile.read();
  for (const auto& library : {kNcclx, kCtran, kMccl}) {
    std::string infoRecord{library.prefix};
    infoRecord.append(" INFO ").append(library.prefix);
    std::string warnRecord{library.prefix};
    warnRecord.append(" WARN ").append(library.prefix);
    std::string infoLine{library.prefix};
    infoLine.append(" info line");
    std::string warnLine{library.prefix};
    warnLine.append(" warn line");

    EXPECT_NE(output.find(infoRecord), std::string::npos)
        << "log file: " << output;
    EXPECT_NE(output.find(warnRecord), std::string::npos)
        << "log file: " << output;
    EXPECT_EQ(stderrOutput.find(infoLine), std::string::npos)
        << "captured stderr: " << stderrOutput;
    EXPECT_NE(stderrOutput.find(warnLine), std::string::npos)
        << "captured stderr: " << stderrOutput;
  }
}

/*
 * A fatal in one library shuts the async thread pool down for the whole
 * process. Every other library logger must keep delivering synchronously so
 * that the messages explaining the abort are not lost.
 */
class MultiLibraryLoggingDeathTest : public MultiLibraryLoggingTest {};

TEST_F(
    MultiLibraryLoggingDeathTest,
    AllLibraryLoggersDeliverAfterFatalShutdown) {
  ScopedTestFile logFile{"comms_multi_library_fatal.log"};
  const auto logPath = logFile.path().string();
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      {
        int exitCode = 0;
        try {
          // Asynchronous init is what shutdownSpdlogForFatal() has to undo.
          for (const auto& library : {kNcclx, kCtran, kMccl}) {
            configureCommsAndNamedSpdlogLoggers(
                library.loggerName,
                std::string{library.prefix},
                logPath,
                []() { return 0; },
                {},
                true,
                spdlog::level::info);
          }
          meta::comms::logger::shutdownSpdlogForFatal();

          COMMS_LOG_NAMED(kNcclx.loggerName, WARN, "ncclx after shutdown");
          COMMS_LOG_NAMED(kCtran.loggerName, WARN, "ctran after shutdown");
          COMMS_LOG_NAMED(kMccl.loggerName, WARN, "mccl after shutdown");
          COMMS_LOG(WARN, "shared after shutdown");

          const auto fileOutput = logFile.read();
          if (fileOutput.find("ncclx after shutdown") == std::string::npos ||
              fileOutput.find("ctran after shutdown") == std::string::npos ||
              fileOutput.find("mccl after shutdown") == std::string::npos ||
              fileOutput.find("shared after shutdown") == std::string::npos) {
            std::fprintf(
                stderr,
                "debug file missing post-shutdown records:\n%s",
                fileOutput.c_str());
            exitCode = 1;
          }
        } catch (const std::exception& error) {
          std::fprintf(
              stderr,
              "failed to inspect post-shutdown debug file: %s\n",
              error.what());
          exitCode = 1;
        }
        resetLoggers("");
        logFile.cleanup();
        cleanupResetLogFile();
        std::exit(exitCode);
      },
      ::testing::ExitedWithCode(0),
      "ncclx after shutdown(.|\\n)*ctran after shutdown(.|\\n)*mccl after "
      "shutdown(.|\\n)*shared after shutdown");
}

} // namespace
