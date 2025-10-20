// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/FileUtil.h>
#include <folly/json/json.h>
#include <folly/stop_watch.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <range/v3/core.hpp>
#include <range/v3/view/split.hpp>
#include <filesystem>
#include <string>

#include "comms/ncclx/v2_27/meta/logger/tests/ScubaLoggerTestMixin.h"
#include "comms/utils/commSpecs.h"
#include "comms/utils/cvars/nccl_cvars.h"
#include "meta/colltrace/CollTrace.h"
#include "meta/colltrace/CollTraceUtils.h"

using namespace std::string_view_literals;

static std::once_flag initCvarFlag;

std::optional<std::filesystem::directory_entry> getScubaFile(
    const std::string& path) {
  for (const auto& entry : std::filesystem::directory_iterator(path)) {
    if (entry.is_regular_file() &&
        entry.path().string().find("nccl_slow_coll") != std::string::npos) {
      return entry;
    }
  }
  return std::nullopt;
}

void initSlowCollThresholds(std::vector<std::string> curList) {
  std::call_once(initCvarFlag, ncclCvarInit);

  NCCL_COLLTRACE_SLOW_COLL_THRESHOLD_BY_PG = curList;
  SlowCollReporter::initThresholdMap();
}

TEST(SlowCollReporterUT, TPReportOnly) {
  initSlowCollThresholds({"TP:1000"});
  CollTraceColl slowColl;
  slowColl.latency = 10000;

  CollTraceColl normalColl;
  normalColl.latency = 100;

  SlowCollReporter tpReporter(CommLogData{.commDesc = "TP:1234"});
  SlowCollReporter ppReporter(CommLogData{.commDesc = "PP_P2P_0:4567"});
  SlowCollReporter epReporter(CommLogData{.commDesc = "EP:4567"});

  EXPECT_TRUE(tpReporter.shouldReportColl(slowColl));
  EXPECT_FALSE(tpReporter.shouldReportColl(normalColl));

  EXPECT_FALSE(ppReporter.shouldReportColl(slowColl));
  EXPECT_FALSE(ppReporter.shouldReportColl(normalColl));

  EXPECT_FALSE(epReporter.shouldReportColl(slowColl));
  EXPECT_FALSE(epReporter.shouldReportColl(normalColl));
}

TEST(SlowCollReporterUT, TPReportAndANY) {
  initSlowCollThresholds({"TP:1000", "ANY:20000"});
  CollTraceColl superSlowColl;
  superSlowColl.latency = 100000;

  CollTraceColl slowColl;
  slowColl.latency = 10000;

  CollTraceColl normalColl;
  normalColl.latency = 100;

  SlowCollReporter tpReporter(CommLogData{.commDesc = "TP:1234"});
  SlowCollReporter ppReporter(CommLogData{.commDesc = "PP_P2P_0:4567"});
  SlowCollReporter epReporter(CommLogData{.commDesc = "EP:4567"});

  EXPECT_TRUE(tpReporter.shouldReportColl(superSlowColl));
  EXPECT_TRUE(tpReporter.shouldReportColl(slowColl));
  EXPECT_FALSE(tpReporter.shouldReportColl(normalColl));

  EXPECT_TRUE(ppReporter.shouldReportColl(superSlowColl));
  EXPECT_FALSE(ppReporter.shouldReportColl(slowColl));
  EXPECT_FALSE(ppReporter.shouldReportColl(normalColl));

  EXPECT_TRUE(epReporter.shouldReportColl(superSlowColl));
  EXPECT_FALSE(epReporter.shouldReportColl(slowColl));
  EXPECT_FALSE(epReporter.shouldReportColl(normalColl));
}

TEST(SlowCollReporterUT, AllExceptPP) {
  initSlowCollThresholds({"TP:1000", "ANY:20000", "PP:-1"});
  CollTraceColl superSlowColl;
  superSlowColl.latency = 100000;

  CollTraceColl slowColl;
  slowColl.latency = 10000;

  CollTraceColl normalColl;
  normalColl.latency = 100;

  SlowCollReporter tpReporter(CommLogData{.commDesc = "TP:1234"});
  SlowCollReporter ppReporter(CommLogData{.commDesc = "PP_P2P_0:4567"});
  SlowCollReporter epReporter(CommLogData{.commDesc = "EP:4567"});

  EXPECT_TRUE(tpReporter.shouldReportColl(superSlowColl));
  EXPECT_TRUE(tpReporter.shouldReportColl(slowColl));
  EXPECT_FALSE(tpReporter.shouldReportColl(normalColl));

  EXPECT_FALSE(ppReporter.shouldReportColl(superSlowColl));
  EXPECT_FALSE(ppReporter.shouldReportColl(slowColl));
  EXPECT_FALSE(ppReporter.shouldReportColl(normalColl));

  EXPECT_TRUE(epReporter.shouldReportColl(superSlowColl));
  EXPECT_FALSE(epReporter.shouldReportColl(slowColl));
  EXPECT_FALSE(epReporter.shouldReportColl(normalColl));
}

TEST(SlowCollReporterUT, TestShouldReportUnfinishedColl) {
  initSlowCollThresholds({"PP_P2P:20000", "PP:1000"});
  // Report every 30 second;
  NCCL_COLLTRACE_REPORT_INTERVAL_SEC = 30;
  SlowCollReporter reporter(CommLogData{.commDesc = "TP:4567"});
  // First report should be true
  EXPECT_TRUE(reporter.shouldReportUnfinishedColl());
  reporter.updateLastReportTimeToNow();
  // Second report should be false after just 1 second;
  /* sleep override */
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  EXPECT_FALSE(reporter.shouldReportUnfinishedColl());
}

TEST(SlowCollReporterUT, TestLongestPrefix) {
  initSlowCollThresholds({"PP_P2P:20000", "PP:1000"});
  CollTraceColl superSlowColl;
  superSlowColl.latency = 100000;

  CollTraceColl slowColl;
  slowColl.latency = 10000;

  CollTraceColl normalColl;
  normalColl.latency = 100;

  SlowCollReporter ppReporter(CommLogData{.commDesc = "PP_P2P_0:4567"});

  EXPECT_TRUE(ppReporter.shouldReportColl(superSlowColl));
  EXPECT_FALSE(ppReporter.shouldReportColl(slowColl));
  EXPECT_FALSE(ppReporter.shouldReportColl(normalColl));
}

class SlowCollReporterScubaTest : public ::testing::Test,
                                  public ScubaLoggerTestMixin {
 public:
  void SetUp() override {
    ScubaLoggerTestMixin::SetUp();
  }
};

TEST_F(SlowCollReporterScubaTest, DISABLED_TestScubaLogging) {
  initSlowCollThresholds({"TP:1000"});
  CollTraceColl slowColl;
  slowColl.latency = 10000;
  slowColl.opCount = 0;
  slowColl.logMetaData.commDesc = "TP:1234";

  CollTraceColl normalColl;
  normalColl.latency = 100;
  normalColl.opCount = 1;
  normalColl.logMetaData.commDesc = "TP:1234";

  auto logMetaData = CommLogData{.commDesc = "TP:1234"};

  SlowCollReporter tpReporter(CommLogData{.commDesc = "TP:1234"});

  if (tpReporter.shouldReportColl(slowColl)) {
    ncclx::colltrace::reportCollToScuba("SlowColl", slowColl, logMetaData);
  }
  if (tpReporter.shouldReportColl(normalColl)) {
    ncclx::colltrace::reportCollToScuba("SlowColl", normalColl, logMetaData);
  }

  const auto& tmpDir = scubaDir();
  folly::stop_watch<std::chrono::milliseconds> timer;
  std::chrono::milliseconds timeout{180000};

  std::optional<std::filesystem::directory_entry> scubaFile = std::nullopt;
  while (scubaFile == std::nullopt && timer.elapsed() < timeout) {
    /* sleep override */
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    scubaFile = getScubaFile(tmpDir.path().string());
  }
  ASSERT_TRUE(scubaFile != std::nullopt);

  // Wait for the file to be written by the background thread.
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));

  std::string file_content;
  folly::readFile(scubaFile->path().string().c_str(), file_content);

  auto lines = file_content | ranges::views::split("\n"sv) |
      ranges::to<std::vector<std::string>>;
  ASSERT_EQ(lines.size(), 1);

  auto json = folly::parseJson(lines[0]);
  EXPECT_EQ(json["normal"]["commDesc"].asString(), "TP:1234");
  EXPECT_EQ(json["int"]["latencyUs"].asInt(), 10000 * 1000); // convert to us
}
