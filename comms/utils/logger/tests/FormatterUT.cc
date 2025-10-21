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

#include <cstdlib>

#include <fmt/format.h>

#include <folly/Format.h>
#include <folly/init/Init.h>
#include <folly/logging/LogMessage.h>
#include <folly/logging/Logger.h>
#include <folly/logging/LoggerDB.h>
#include <folly/portability/GTest.h>
#include <folly/portability/Stdlib.h>
#include <folly/system/ThreadName.h>

#include "comms/utils/logger/LoggingFormat.h"

FOLLY_GNU_DISABLE_WARNING("-Wdeprecated-declarations")

using namespace fmt::literals;
using namespace folly;

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
    StringPiece prefix = "NCCL") {
  LoggerDB db{LoggerDB::TESTING};
  auto* category = db.getCategory("test");
  meta::comms::logger::NcclLogFormatter formatter(
      prefix.str(), []() { return 0; });

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
  auto expected = folly::sformat(
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

TEST(GlogFormatter, logThreadName) {
  auto tid = getOSThreadID();
  auto hostname = getHostName('.');
  auto pid = getpid();
  constexpr std::string_view kThreadName = "TestT1";
  constexpr std::string_view kPrefix = "NCCL";

  meta::comms::logger::initThreadMetaData(kThreadName);
  // Test a very simple single-line log message
  auto expected = folly::sformat(
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
    auto expected = folly::sformat(
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
