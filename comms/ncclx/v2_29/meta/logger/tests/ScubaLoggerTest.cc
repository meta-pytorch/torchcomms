// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <filesystem>

#include <folly/stop_watch.h>
#include <gtest/gtest.h>

#include "ScubaLoggerTestMixin.h"
#include "comms/utils/logger/ScubaLogger.h"

class ScubaLoggerTest : public ::testing::Test, public ScubaLoggerTestMixin {
 public:
  void SetUp() override {
    ScubaLoggerTestMixin::SetUp();
  }
};

int numFiles(const std::string& path) {
  int count = 0;
  for (const auto& entry : std::filesystem::directory_iterator(path)) {
    if (entry.is_regular_file()) {
      ++count;
    }
  }
  return count;
}

// This simply verifies that the logging directory has some files created
TEST_F(ScubaLoggerTest, LoggingFilesCreated) {
  // Log a sample
  NcclScubaSample scubaSample("test");
  SCUBA_nccl_structured_logging.addSample(std::move(scubaSample));

  // Wait until the scuba directory is non-empty
  const auto& tmpDir = scubaDir();
  folly::stop_watch<std::chrono::milliseconds> timer;
  std::chrono::milliseconds timeout{180000};
  while (numFiles(tmpDir.path().string()) == 0 && timer.elapsed() < timeout) {
    /* sleep override */
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  EXPECT_GT(numFiles(tmpDir.path().string()), 0);
}
