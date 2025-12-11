// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <folly/Random.h>
#include <folly/dynamic.h>
#include <folly/json.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/ctran/tests/CtranXPlatUtUtils.h"
#include "comms/ctran/utils/CtranTraceLogger.h"
#include "comms/ctran/utils/LogInit.h"
#include "comms/testinfra/TestXPlatUtils.h"

class CtranTraceLoggerTest : public ::testing::Test {
 public:
  double expectedDurMS;
  CtranTraceLoggerTest() = default;

 protected:
  void SetUp() override {
    setenv("NCCL_CTRAN_ENABLE_TRACE_LOGGER", "1", 1);

    // Ensure logger is initialized
    ncclCvarInit();
    ctran::logging::initCtranLogging(true /*alwaysInit*/);

    // A random duration between 0-5ms to test the timer
    srand(time(NULL));
    expectedDurMS = rand() % 5 + 1;
  }
  void TearDown() override {}
};

namespace ctran::utils {
TEST_F(CtranTraceLoggerTest, Simple) {
  auto dummyAlgo = "Ring";
  auto ts = std::make_unique<TraceRecord>(dummyAlgo);
  EXPECT_THAT(ts, testing::NotNull());
}

TEST_F(CtranTraceLoggerTest, Timer) {
  auto timeInterval = std::unique_ptr<TimeInterval>(new TimeInterval(0));
  EXPECT_THAT(timeInterval, testing::NotNull());

  int expectedDurMS;
  expectedDurMS = folly::Random::rand32() % 5 + 1;
  std::this_thread::sleep_for(std::chrono::milliseconds(expectedDurMS));

  timeInterval->end();
  int64_t durMs = timeInterval->durationMs();
  EXPECT_GE(durMs, expectedDurMS);

  timeInterval->end();
  int64_t durUs = timeInterval->durationUs();
  EXPECT_GE(durUs, expectedDurMS * 1000);
};

TEST_F(CtranTraceLoggerTest, TimestampPoint) {
  int peer = 0;
  auto tp = std::unique_ptr<TimestampPoint>(new TimestampPoint(peer));
  EXPECT_THAT(tp, testing::NotNull());
}

TEST_F(CtranTraceLoggerTest, Timestamp) {
  auto dummyAlgo = "Ring";
  auto ts = std::make_unique<TraceRecord>(dummyAlgo);
  EXPECT_THAT(ts, testing::NotNull());
}

TEST_F(CtranTraceLoggerTest, TraceTimestampPointWithSeqNum) {
  auto dummyAlgo = "Ring";
  auto ts = std::make_unique<TraceRecord>(dummyAlgo, 1 /* rank */);
  EXPECT_THAT(ts, testing::NotNull());

  // Insert two points with increasing seqNum
  int seqNum = 1;
  const int peer = 2;
  ts->addPoint("recvCtrl", seqNum, peer);

  seqNum++;
  ts->addPoint("recvCtrl", seqNum, peer);

  ts->end();

  // Dump the json report for verification
  int id = 0;
  std::string json = "[" + ts->toJsonEntry(id, 0) + "]";
  EXPECT_EQ(id, 3);

  folly::dynamic parsed = folly::parseJson(json);
  EXPECT_EQ(parsed.size(), 3);

  std::unordered_set<int> seqNums = {1, 2};

  // Expect first record is the entire algo
  EXPECT_EQ(parsed[0]["name"], "Ring");
  EXPECT_EQ(parsed[0]["tid"], -1);

  for (int i = 1; i < parsed.size(); i++) {
    auto& p = parsed[i];
    EXPECT_EQ(p["name"], "recvCtrl");
    EXPECT_EQ(p["tid"], peer);
    EXPECT_NE(p["args"], nullptr);
    EXPECT_NE(p["args"]["seqNum"], nullptr);

    // We don't know the order of the two points so have to check seqNum from a
    // set
    int seqNumExp = atoi(p["args"]["seqNum"].c_str());
    EXPECT_TRUE(seqNums.find(seqNumExp) != seqNums.end());
    seqNums.erase(seqNumExp);
  }
}

TEST_F(CtranTraceLoggerTest, TraceTimeIntervalWithSeqNum) {
  auto dummyAlgo = "Ring";
  auto ts = std::make_unique<TraceRecord>(dummyAlgo, 1 /* rank */);
  EXPECT_THAT(ts, testing::NotNull());

  // Insert two points with increasing seqNum
  int seqNum = 1;
  const int peer = 2;
  const int expectedDurMS = 10;

  ts->startInterval("sendTrans", seqNum, peer);
  // @lint-ignore
  std::this_thread::sleep_for(std::chrono::milliseconds(expectedDurMS));
  ts->endInterval("sendTrans", seqNum);

  EXPECT_TRUE(ts->hasInterval("sendTrans", seqNum));

  seqNum++;
  ts->startInterval("sendTrans", seqNum, peer);
  // @lint-ignore
  std::this_thread::sleep_for(std::chrono::milliseconds(expectedDurMS));
  ts->endInterval("sendTrans", seqNum);

  EXPECT_TRUE(ts->hasInterval("sendTrans", seqNum));

  ts->end();

  // Dump the json report for verification
  int id = 0;
  std::string json = "[" + ts->toJsonEntry(id, 0) + "]";
  EXPECT_EQ(id, 3);

  folly::dynamic parsed = folly::parseJson(json);
  EXPECT_EQ(parsed.size(), 3);

  std::unordered_set<int> seqNums = {1, 2};

  // Expect first record is the entire algo
  EXPECT_EQ(parsed[0]["name"], "Ring");
  EXPECT_EQ(parsed[0]["tid"], -1);

  for (int i = 1; i < parsed.size(); i++) {
    auto& p = parsed[i];
    EXPECT_EQ(p["name"], "sendTrans");
    EXPECT_EQ(p["tid"], peer);
    EXPECT_NE(p["args"], nullptr);
    EXPECT_NE(p["args"]["seqNum"], nullptr);

    // We don't know the order of the two points so have to check seqNum from a
    // set
    int seqNumExp = atoi(p["args"]["seqNum"].c_str());
    EXPECT_TRUE(seqNums.find(seqNumExp) != seqNums.end());
    seqNums.erase(seqNumExp);
  }
}

TEST_F(CtranTraceLoggerTest, TraceTimeIntervalWithMetaData) {
  auto dummyAlgo = "Ring";
  auto temp = getenv("NCCL_CTRAN_ENABLE_TRACE_LOGGER");
  EXPECT_EQ(std::string(temp), "1");

  auto ts = std::make_unique<TraceRecord>(dummyAlgo, 1 /* rank */);
  EXPECT_THAT(ts, testing::NotNull());

  // Insert two points with increasing seqNum
  int seqNum = 1;
  const int peer = 2;
  const int expectedDurMS = 10;

  std::map<std::string, std::string> metaData = {
      {"step", std::to_string(1)}, {"round", std::to_string(10)}};

  ts->startInterval("sendTrans", seqNum, peer, metaData);
  // @lint-ignore
  std::this_thread::sleep_for(std::chrono::milliseconds(expectedDurMS));
  ts->endInterval("sendTrans", seqNum);

  ts->end();

  // Dump the json report for verification
  int id = 0;
  std::string json = "[" + ts->toJsonEntry(id, 0) + "]";

  folly::dynamic parsed = folly::parseJson(json);
  EXPECT_EQ(parsed.size(), 2);

  // Skip first record which is the entire algo
  EXPECT_EQ(parsed[0]["name"], "Ring");

  for (int i = 1; i < parsed.size(); i++) {
    auto& p = parsed[i];
    EXPECT_EQ(p["name"], "sendTrans");
    EXPECT_EQ(p["tid"], peer);
    EXPECT_NE(p["args"], nullptr);
    EXPECT_EQ(p["args"]["seqNum"], "1");
    EXPECT_EQ(p["args"]["step"], "1");
    EXPECT_EQ(p["args"]["round"], "10");
  }
}

TEST_F(CtranTraceLoggerTest, TraceTimeIntervalWithNonexistStart) {
  auto dummyAlgo = "Ring";
  auto temp = getenv("NCCL_CTRAN_ENABLE_TRACE_LOGGER");
  EXPECT_EQ(std::string(temp), "1");

  auto ts = std::make_unique<TraceRecord>(dummyAlgo, 1 /* rank */);
  EXPECT_THAT(ts, testing::NotNull());

  EXPECT_FALSE(ts->hasInterval("sendTrans", 1));
  EXPECT_DEATH(ts->endInterval("sendTrans", 1), "");
}

TEST_F(CtranTraceLoggerTest, CtranTraceLoggerSimple) {
  auto temp = getenv("NCCL_CTRAN_ENABLE_TRACE_LOGGER");
  EXPECT_EQ(std::string(temp), "1");

  std::unique_ptr<TraceLogger> ctran_trace_logger{nullptr};
  ctran_trace_logger = std::make_unique<TraceLogger>(0);
  EXPECT_EQ(ctran_trace_logger->isTraceEnabled(), true);

  auto dummyAlgo = "Ring";
  auto ts = std::make_unique<TraceRecord>(dummyAlgo, 1 /* rank */);
  EXPECT_THAT(ts, testing::NotNull());

  EXPECT_FALSE(ts->hasInterval("sendTrans", 1));
  EXPECT_DEATH(ts->endInterval("sendTrans", 1), "");
  ctran_trace_logger->addTraceRecord(std::move(ts));
}

} // namespace ctran::utils
