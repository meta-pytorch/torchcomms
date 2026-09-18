/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <array>
#include <atomic>
#include <cstdlib>
#include <thread>
#include <vector>

#include <fmt/format.h>

#include <folly/Format.h>
#include <folly/init/Init.h>
#include <folly/logging/LogMessage.h>
#include <folly/logging/Logger.h>
#include <folly/logging/LoggerDB.h>
#include <folly/portability/GTest.h>
#include <folly/portability/Stdlib.h>
#include <folly/system/ThreadName.h>

#include "comms/utils/logger/CommsLogFormatter.h"
#include "comms/utils/logger/LoggingFormat.h"

FOLLY_GNU_DISABLE_WARNING("-Wdeprecated-declarations")

using folly::getOSThreadID;
using folly::LoggerDB;
using folly::LogLevel;
using folly::LogMessage;
using folly::StringPiece;

namespace {
/**
 * Helper function to format a LogMessage using the GlogStyleFormatter.
 *
 * formatMsg() accepts the timestamp as a plain integer simply to reduce the
 * verbosity of the test code.
 *
 * Note that in this test's main() function we set the timezone to "UTC"
 * so that the logged time values will be consistent regardless of the actual
 * local time on this host.
 */
std::string formatMsg(
    LogLevel level,
    StringPiece msg,
    StringPiece filename,
    unsigned int lineNumber,
    StringPiece functionName,
    // Default timestamp: 2017-04-17 13:45:56.123456 UTC
    uint64_t timestampNS = 1492436756123456789ULL,
    StringPiece prefix = "NCCL",
    int threadContext = 0) {
  LoggerDB db{LoggerDB::TESTING};
  auto* category = db.getCategory("test");
  meta::comms::logger::NcclLogFormatter formatter(
      prefix.str(), [threadContext]() { return threadContext; });

  std::chrono::system_clock::time_point logTimePoint{
      std::chrono::duration_cast<std::chrono::system_clock::duration>(
          std::chrono::nanoseconds{timestampNS})};
  LogMessage logMessage{
      category,
      level,
      logTimePoint,
      filename,
      lineNumber,
      functionName,
      msg.str()};

  return formatter.formatMessage(logMessage, category);
}

std::string getHostName(const char delim) {
  constexpr int maxlen = HOST_NAME_MAX + 1;
  char hostname[maxlen];
  if (gethostname(hostname, maxlen) != 0) {
    return "unknown";
  }
  int i = 0;
  while ((hostname[i] != delim) && (hostname[i] != '\0') && (i < maxlen - 1)) {
    i++;
  }
  hostname[i] = '\0';
  return std::string{hostname};
}

} // namespace

TEST(GlogFormatter, log) {
  auto tid = getOSThreadID();
  auto hostname = getHostName('.');
  auto pid = getpid();
  constexpr std::string_view kThreadName = "main";
  constexpr std::string_view kPrefix = "NCCL";

  // Test a very simple single-line log message
  auto expected = fmt::format(
      "W0417 13:45:56.123456 {:5d} myfile.cpp:1234] {}:{}:{} [{}][{}] {} WARN hello world\n",
      tid,
      hostname,
      pid,
      tid,
      0,
      kThreadName,
      kPrefix);
  EXPECT_EQ(
      expected,
      formatMsg(
          LogLevel::WARN, "hello world", "myfile.cpp", 1234, "testFunction"));
}

TEST(GlogFormatter, logCustomPrefixAndThreadContext) {
  const auto formatted = formatMsg(
      LogLevel::WARN,
      "hello world",
      "myfile.cpp",
      1234,
      "testFunction",
      1492436756123456789ULL,
      "CTRAN",
      7);
  EXPECT_NE(
      formatted.find(" [7][main] CTRAN WARN hello world\n"), std::string::npos);
}

TEST(CommsLogFormatter, emptyLevelUsesPlaceholder) {
  const meta::comms::logger::CommsLogMetadata metadata{
      .timestamp = std::chrono::system_clock::time_point{},
      .threadId = 1,
      .filename = "test.cpp",
      .lineNumber = 2,
      .hostname = "host",
      .processId = 3,
      .threadContext = 4,
      .threadName = "thread",
      .prefix = "NCCL",
  };

  const auto formatted =
      meta::comms::logger::formatCommsLogMessage("", "message", metadata);
  EXPECT_EQ(formatted.front(), '?');
  EXPECT_NE(formatted.find(" NCCL  message\n"), std::string::npos);
}

TEST(LoggingFormat, ParseDebugSubsysMaskPreservesLegacyBehavior) {
  using meta::comms::logger::ALL;
  using meta::comms::logger::BOOTSTRAP;
  using meta::comms::logger::COLL;
  using meta::comms::logger::DESTROY;
  using meta::comms::logger::ENV;
  using meta::comms::logger::INIT;
  using meta::comms::logger::P2P;
  using meta::comms::logger::parseDebugSubsysMask;

  struct TestCase {
    const char* input;
    uint64_t expected;
  };
  constexpr std::array<TestCase, 16> kCases{{
      {"", 0},
      {",", 0},
      {"^", ~0ULL},
      {"INIT", INIT},
      {"init", INIT},
      {"iNiT,cOlL", INIT | COLL},
      {"INIT,COLL,P2P", INIT | COLL | P2P},
      {"^INIT,COLL", ~(static_cast<uint64_t>(INIT | COLL))},
      {"ALL", static_cast<uint64_t>(ALL)},
      {"^ALL", 0},
      {"UNKNOWN,COLL", COLL},
      {"^UNKNOWN", ~0ULL},
      {"INIT,,COLL,", INIT | COLL},
      {" INIT", 0},
      {"DESTROY", DESTROY},
      {"destroy,init", DESTROY | INIT},
  }};

  EXPECT_EQ(
      parseDebugSubsysMask(nullptr),
      static_cast<uint64_t>(INIT | BOOTSTRAP | ENV));

  for (const auto& testCase : kCases) {
    EXPECT_EQ(parseDebugSubsysMask(testCase.input), testCase.expected)
        << "input: " << testCase.input;
  }
}

TEST(LoggingFormat, SharedDebugLevelParsingRemainsCaseSensitive) {
  using meta::comms::logger::getLoggerDebugLevel;
  using meta::comms::logger::LogLevel;

  EXPECT_EQ(getLoggerDebugLevel("INFO"), LogLevel::INFO);
  EXPECT_EQ(getLoggerDebugLevel("info"), LogLevel::NONE);
}

TEST(LoggingFormat, NcclDebugLevelParsingIsCaseInsensitive) {
  using meta::comms::logger::getNcclLoggerDebugLevel;
  using meta::comms::logger::LogLevel;

  EXPECT_EQ(getNcclLoggerDebugLevel("info"), LogLevel::INFO);
  EXPECT_EQ(getNcclLoggerDebugLevel("WaRn"), LogLevel::WARN);
  EXPECT_EQ(getNcclLoggerDebugLevel("trace"), LogLevel::TRACE);
  EXPECT_EQ(getNcclLoggerDebugLevel("invalid"), LogLevel::NONE);
}

TEST(LoggingFormat, ParsesAsyncDeliveryLikeGeneratedNcclCvar) {
  using meta::comms::logger::parseDebugLoggingAsync;

  EXPECT_TRUE(parseDebugLoggingAsync(nullptr, true));
  EXPECT_FALSE(parseDebugLoggingAsync(nullptr, false));
  EXPECT_TRUE(parseDebugLoggingAsync("", false));
  EXPECT_FALSE(parseDebugLoggingAsync("No", true));
  EXPECT_TRUE(parseDebugLoggingAsync("YES", false));
  EXPECT_TRUE(parseDebugLoggingAsync("invalid", true));
  EXPECT_TRUE(parseDebugLoggingAsync("invalid", false));
}

TEST(LoggingFormat, ParseDebugSubsysMaskSupportsConcurrentCalls) {
  using meta::comms::logger::ALL;
  using meta::comms::logger::ALLOC;
  using meta::comms::logger::BOOTSTRAP;
  using meta::comms::logger::CALL;
  using meta::comms::logger::COLL;
  using meta::comms::logger::ENV;
  using meta::comms::logger::GRAPH;
  using meta::comms::logger::INIT;
  using meta::comms::logger::NET;
  using meta::comms::logger::NVLS;
  using meta::comms::logger::P2P;
  using meta::comms::logger::parseDebugSubsysMask;
  using meta::comms::logger::PROFILE;
  using meta::comms::logger::PROXY;
  using meta::comms::logger::RAS;
  using meta::comms::logger::REG;
  using meta::comms::logger::SHM;
  using meta::comms::logger::TUNING;

  struct TestCase {
    const char* input;
    uint64_t expected;
  };
  constexpr std::array<TestCase, 4> kCases{{
      {"INIT,COLL,P2P,SHM,NET,GRAPH,TUNING,ENV",
       INIT | COLL | P2P | SHM | NET | GRAPH | TUNING | ENV},
      {"ALLOC,CALL,PROXY,NVLS,BOOTSTRAP,REG,PROFILE,RAS",
       ALLOC | CALL | PROXY | NVLS | BOOTSTRAP | REG | PROFILE | RAS},
      {"^INIT,COLL,P2P,SHM,NET,GRAPH,TUNING,ENV",
       ~(static_cast<uint64_t>(
           INIT | COLL | P2P | SHM | NET | GRAPH | TUNING | ENV))},
      {"ALL", static_cast<uint64_t>(ALL)},
  }};
  constexpr int kIterations = 10'000;

  std::atomic<bool> start{false};
  std::atomic<int> mismatches{0};
  std::vector<std::thread> threads;
  threads.reserve(kCases.size());
  for (const auto& testCase : kCases) {
    threads.emplace_back([&start, &mismatches, testCase] {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int iteration = 0; iteration < kIterations; ++iteration) {
        if (parseDebugSubsysMask(testCase.input) != testCase.expected) {
          mismatches.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  start.store(true, std::memory_order_release);
  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(mismatches.load(std::memory_order_relaxed), 0);
}

TEST(GlogFormatter, logThreadName) {
  auto tid = getOSThreadID();
  auto hostname = getHostName('.');
  auto pid = getpid();
  constexpr std::string_view kThreadName = "TestT1";
  constexpr std::string_view kPrefix = "NCCL";

  meta::comms::logger::initThreadMetaData(kThreadName);
  // Test a very simple single-line log message
  auto expected = fmt::format(
      "W0417 13:45:56.123456 {:5d} myfile.cpp:1234] {}:{}:{} [{}][{}] {} WARN hello world\n",
      tid,
      hostname,
      pid,
      tid,
      0,
      kThreadName,
      kPrefix);
  EXPECT_EQ(
      expected,
      formatMsg(
          LogLevel::WARN,
          "hello world",
          "myfile.cpp",
          1234,
          "testFunction",
          1492436756123456789ULL /* timestampNS */));
}

#ifndef _WIN32
TEST(GlogFormatter, logThreadNameChanged) {
  if (folly::canSetCurrentThreadName()) {
    std::string msg;
    std::string threadName = "foo";
    uint64_t otherThreadID;
    auto hostname = getHostName('.');
    auto pid = getpid();
    constexpr std::string_view kThreadName = "TestT1";
    constexpr std::string_view kPrefix = "NCCL";
    std::thread thread([&] {
      meta::comms::logger::initThreadMetaData(kThreadName);
      otherThreadID = getOSThreadID();
      msg = formatMsg(
          LogLevel::WARN,
          "hello world",
          "myfile.cpp",
          1234,
          "testFunction",
          1492436756123456789ULL /* timestampNS */);
    });
    thread.join();
    // Test a very simple single-line log message
    auto expected = fmt::format(
        "W0417 13:45:56.123456 {:5d} myfile.cpp:1234] {}:{}:{} [{}][{}] {} WARN hello world\n",
        otherThreadID,
        hostname,
        pid,
        otherThreadID,
        0,
        kThreadName,
        kPrefix);
    EXPECT_EQ(expected, msg);
  }
}
#endif

TEST(LoggingFormat, getLastCommsErrorBasic) {
  LoggerDB db{LoggerDB::TESTING};
  auto* category = db.getCategory("test.error");
  meta::comms::logger::NcclLogFormatter formatter("NCCL", []() { return 0; });

  std::chrono::system_clock::time_point logTimePoint{
      std::chrono::duration_cast<std::chrono::system_clock::duration>(
          std::chrono::nanoseconds{1492436756123456789ULL})};
  LogMessage errorMsg{
      category,
      LogLevel::ERR,
      logTimePoint,
      "test.cpp",
      100,
      "testFunc",
      "Test error message"};

  formatter.formatMessage(errorMsg, category);

  auto lastError = meta::comms::logger::getLastCommsError();
  EXPECT_FALSE(lastError.empty());
  const auto& errorStr = lastError;
  EXPECT_TRUE(errorStr.find("Test error message") != std::string::npos);
  EXPECT_TRUE(errorStr.find("NCCL Stack trace:") != std::string::npos);
}

TEST(LoggingFormat, getLastCommsErrorWithStack) {
  LoggerDB db{LoggerDB::TESTING};
  auto* category = db.getCategory("test.error");
  meta::comms::logger::NcclLogFormatter formatter("NCCL", []() { return 0; });

  std::chrono::system_clock::time_point logTimePoint{
      std::chrono::duration_cast<std::chrono::system_clock::duration>(
          std::chrono::nanoseconds{1492436756123456789ULL})};
  LogMessage errorMsg{
      category,
      LogLevel::ERR,
      logTimePoint,
      "test.cpp",
      100,
      "testFunc",
      "Error with stack"};

  formatter.formatMessage(errorMsg, category);

  meta::comms::logger::appendErrorToStack("Stack frame 1");
  meta::comms::logger::appendErrorToStack("Stack frame 2");
  meta::comms::logger::appendErrorToStack("Stack frame 3");

  auto lastError = meta::comms::logger::getLastCommsError();
  EXPECT_FALSE(lastError.empty());
  const auto& errorStr = lastError;
  EXPECT_TRUE(errorStr.find("Error with stack") != std::string::npos);
  EXPECT_TRUE(errorStr.find("NCCL Stack trace:") != std::string::npos);
  EXPECT_TRUE(errorStr.find("Stack frame 1") != std::string::npos);
  EXPECT_TRUE(errorStr.find("Stack frame 2") != std::string::npos);
  EXPECT_TRUE(errorStr.find("Stack frame 3") != std::string::npos);
}

TEST(LoggingFormat, appendErrorToStackOrder) {
  LoggerDB db{LoggerDB::TESTING};
  auto* category = db.getCategory("test.error");
  meta::comms::logger::NcclLogFormatter formatter("NCCL", []() { return 0; });

  std::chrono::system_clock::time_point logTimePoint{
      std::chrono::duration_cast<std::chrono::system_clock::duration>(
          std::chrono::nanoseconds{1492436756123456789ULL})};
  LogMessage errorMsg{
      category,
      LogLevel::ERR,
      logTimePoint,
      "test.cpp",
      200,
      "testFunc2",
      "Error for stack order test"};

  formatter.formatMessage(errorMsg, category);

  meta::comms::logger::appendErrorToStack("First");
  meta::comms::logger::appendErrorToStack("Second");
  meta::comms::logger::appendErrorToStack("Third");

  auto lastError = meta::comms::logger::getLastCommsError();
  EXPECT_FALSE(lastError.empty());
  const auto& errorStr = lastError;

  size_t firstPos = errorStr.find("First");
  size_t secondPos = errorStr.find("Second");
  size_t thirdPos = errorStr.find("Third");

  EXPECT_NE(firstPos, std::string::npos);
  EXPECT_NE(secondPos, std::string::npos);
  EXPECT_NE(thirdPos, std::string::npos);
  EXPECT_LT(firstPos, secondPos);
  EXPECT_LT(secondPos, thirdPos);
}

TEST(LoggingFormat, getLastCommsErrorEmptyStack) {
  LoggerDB db{LoggerDB::TESTING};
  auto* category = db.getCategory("test.error");
  meta::comms::logger::NcclLogFormatter formatter("NCCL", []() { return 0; });

  std::chrono::system_clock::time_point logTimePoint{
      std::chrono::duration_cast<std::chrono::system_clock::duration>(
          std::chrono::nanoseconds{1492436756123456789ULL})};
  LogMessage errorMsg{
      category,
      LogLevel::ERR,
      logTimePoint,
      "test.cpp",
      300,
      "testFunc3",
      "Error without stack"};

  formatter.formatMessage(errorMsg, category);

  auto lastError = meta::comms::logger::getLastCommsError();
  EXPECT_FALSE(lastError.empty());
  const auto& errorStr = lastError;
  EXPECT_TRUE(errorStr.find("Error without stack") != std::string::npos);
  EXPECT_TRUE(errorStr.find("NCCL Stack trace:") != std::string::npos);
}

TEST(LoggingFormat, setLastErrorPrefersNativeStack) {
  meta::comms::logger::setLastError(
      "net timeout", {"stackFrame1", "stackFrame2"});

  auto lastError = meta::comms::logger::getLastCommsError();
  const auto messagePos = lastError.find("net timeout");
  const auto headerPos = lastError.find("NCCL Stack trace:");
  const auto frame1Pos = lastError.find("stackFrame1");
  const auto frame2Pos = lastError.find("stackFrame2");

  EXPECT_NE(messagePos, std::string::npos);
  EXPECT_NE(headerPos, std::string::npos);
  EXPECT_NE(frame1Pos, std::string::npos);
  EXPECT_NE(frame2Pos, std::string::npos);

  // The native stack is recorded after the message, in order.
  EXPECT_LT(messagePos, frame1Pos);
  EXPECT_LT(frame1Pos, frame2Pos);
}

TEST(LoggingFormat, nonErrorMessageDoesNotUpdateLastError) {
  LoggerDB db{LoggerDB::TESTING};
  auto* category = db.getCategory("test.info");
  meta::comms::logger::NcclLogFormatter formatter("NCCL", []() { return 0; });

  std::chrono::system_clock::time_point logTimePoint{
      std::chrono::duration_cast<std::chrono::system_clock::duration>(
          std::chrono::nanoseconds{1492436756123456789ULL})};

  LogMessage errorMsg{
      category,
      LogLevel::ERR,
      logTimePoint,
      "test.cpp",
      100,
      "testFunc",
      "Initial error"};
  formatter.formatMessage(errorMsg, category);

  LogMessage infoMsg{
      category,
      LogLevel::INFO,
      logTimePoint,
      "test.cpp",
      200,
      "testFunc",
      "This is just info"};
  formatter.formatMessage(infoMsg, category);

  auto lastError = meta::comms::logger::getLastCommsError();
  EXPECT_FALSE(lastError.empty());
  const auto& errorStr = lastError;
  EXPECT_TRUE(errorStr.find("Initial error") != std::string::npos);
  EXPECT_FALSE(errorStr.find("This is just info") != std::string::npos);
}

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);

  // Some of our tests check timestamps emitted by the formatter.
  // Set the timezone to a consistent value so that the tests are not
  // affected by the local time of the user running the test.
  //
  // UTC is the only timezone that we can really rely on to work consistently.
  // This will work even in the absence of a proper tzdata installation on the
  // local system.
  setenv("TZ", "UTC", 1);

  return RUN_ALL_TESTS();
}
