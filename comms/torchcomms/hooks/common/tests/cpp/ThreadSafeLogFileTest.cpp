// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Unit tests for ThreadSafeLogFile's buffered-flush behavior. The logger
// flushes on a bounded line/time cadence instead of once per line, so these
// verify that (a) no lines are lost across a normal close and (b) lines are
// buffered during writing rather than flushed one-per-line.

#include <comms/torchcomms/hooks/common/ThreadSafeLogFile.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace torch::comms {
namespace {

std::vector<std::string> readLines(const std::string& path) {
  std::vector<std::string> lines;
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty()) {
      lines.push_back(line);
    }
  }
  return lines;
}

} // namespace

TEST(ThreadSafeLogFileTest, PreservesAllLinesAcrossClose) {
  const std::string path =
      std::string(::testing::TempDir()) + "/tslf_preserve.log";
  constexpr int kNumLines = 5000;
  {
    ThreadSafeLogFile log;
    log.open(path);
    for (int i = 0; i < kNumLines; ++i) {
      log.writeLine("line-" + std::to_string(i));
    }
  } // destructor flushes and closes

  const auto lines = readLines(path);
  ASSERT_EQ(lines.size(), static_cast<size_t>(kNumLines));
  EXPECT_EQ(lines.front(), "line-0");
  EXPECT_EQ(lines.back(), "line-" + std::to_string(kNumLines - 1));
  std::filesystem::remove(path);
}

TEST(ThreadSafeLogFileTest, BuffersRatherThanFlushingPerLine) {
  const std::string path =
      std::string(::testing::TempDir()) + "/tslf_buffer.log";
  constexpr int kNumLines = 100;
  size_t onDiskBeforeClose = 0;
  {
    // Pin the cadence bounds above what this test writes (line bound >
    // kNumLines, effectively-infinite time bound) so no flush fires during the
    // loop; a per-line-flush logger would still have every line on disk.
    ThreadSafeLogFile log(
        /*flushEveryLines=*/kNumLines + 1,
        /*flushInterval=*/std::chrono::hours(1));
    log.open(path);
    for (int i = 0; i < kNumLines; ++i) {
      log.writeLine("x");
    }
    onDiskBeforeClose = readLines(path).size();
  } // destructor flushes and closes

  EXPECT_EQ(onDiskBeforeClose, 0u);
  EXPECT_EQ(readLines(path).size(), static_cast<size_t>(kNumLines));
  std::filesystem::remove(path);
}

} // namespace torch::comms
