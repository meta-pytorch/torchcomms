// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstdlib>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "ScubaLoggerTestMixin.h"
#include "comms/utils/logger/EventsScubaUtil.h"

class EventsScubaUtilTest : public ::testing::Test,
                            public ScubaLoggerTestMixin {
 public:
  void SetUp() override {
    ScubaLoggerTestMixin::SetUp();
  }
};

void a5() {
  auto sg = EVENTS_SCUBA_UTIL_SAMPLE_GUARD(__FUNCTION__);

  auto context = EventsScubaUtil::getAllContext();
  ASSERT_EQ(2, context.keyValuePairs.size());
  ASSERT_EQ("xyz", context.keyValuePairs["run_id"]);
  ASSERT_EQ("twshared", context.keyValuePairs["host_prefix"]);

  ASSERT_EQ(3, context.eventNameStack.size());
  ASSERT_EQ("a1", context.eventNameStack[0].eventName);
  ASSERT_EQ("a2", context.eventNameStack[1].eventName);
  ASSERT_EQ("a5", context.eventNameStack[2].eventName);
}

void a3() {
  auto sg1 = EVENTS_SCUBA_UTIL_SAMPLE_GUARD(__FUNCTION__);
  auto sg2 = EVENTS_SCUBA_UTIL_SAMPLE_GUARD("a4");

  auto context = EventsScubaUtil::getAllContext();
  ASSERT_EQ(2, context.keyValuePairs.size());
  ASSERT_EQ("xyz", context.keyValuePairs["run_id"]);
  ASSERT_EQ("twshared", context.keyValuePairs["host_prefix"]);

  ASSERT_EQ(4, context.eventNameStack.size());
  ASSERT_EQ("a1", context.eventNameStack[0].eventName);
  ASSERT_EQ("a2", context.eventNameStack[1].eventName);
  ASSERT_EQ("a3", context.eventNameStack[2].eventName);
  ASSERT_EQ("a4", context.eventNameStack[3].eventName);
}

void a2() {
  auto cg = EventsScubaUtil::StickyContextGuard("host_prefix", "twshared");
  auto sg = EVENTS_SCUBA_UTIL_SAMPLE_GUARD(__FUNCTION__);
  a3();
  a5();
}

void a1() {
  auto cg = EventsScubaUtil::StickyContextGuard("run_id", "xyz");
  auto sg = EVENTS_SCUBA_UTIL_SAMPLE_GUARD(__FUNCTION__);
  a2();

  auto context = EventsScubaUtil::getAllContext();
  ASSERT_EQ(1, context.keyValuePairs.size());
  ASSERT_EQ("xyz", context.keyValuePairs["run_id"]);

  ASSERT_EQ(1, context.eventNameStack.size());
  ASSERT_EQ("a1", context.eventNameStack[0].eventName);
}

TEST_F(EventsScubaUtilTest, StickyContextAndEvents) {
  a1();
}

void b2() {
  auto cg = EventsScubaUtil::StickyContextGuard("run_id", "abc");
  auto sg = EVENTS_SCUBA_UTIL_SAMPLE_GUARD(__FUNCTION__);

  auto context = EventsScubaUtil::getAllContext();
  ASSERT_EQ(1, context.keyValuePairs.size());
  ASSERT_EQ("abc", context.keyValuePairs["run_id"]);
}

void b1() {
  auto cg = EventsScubaUtil::StickyContextGuard("run_id", "xyz");
  auto sg = EVENTS_SCUBA_UTIL_SAMPLE_GUARD(__FUNCTION__);
  b2();
}

TEST_F(EventsScubaUtilTest, StickyContextOverwriteKey) {
  b1();
}

void test4() {
  auto sg = EVENTS_SCUBA_UTIL_SAMPLE_GUARD(__FUNCTION__);
  /* sleep override */
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));

  auto context = EventsScubaUtil::getAllContext();

  ASSERT_EQ(3, context.eventNameStack.size());
  ASSERT_EQ("test4", context.eventNameStack[2].eventName);
  ASSERT_EQ(
      *context.eventNameStack[2].childrenLatency, std::chrono::milliseconds(0));
}

void test3() {
  auto sg = EVENTS_SCUBA_UTIL_SAMPLE_GUARD(__FUNCTION__);
  /* sleep override */
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));

  auto context = EventsScubaUtil::getAllContext();

  ASSERT_EQ(3, context.eventNameStack.size());
  ASSERT_EQ("test3", context.eventNameStack[2].eventName);
  ASSERT_EQ(
      *context.eventNameStack[2].childrenLatency, std::chrono::milliseconds(0));
}

void test2() {
  auto sg = EVENTS_SCUBA_UTIL_SAMPLE_GUARD(__FUNCTION__);

  test3();
  test4();

  /* sleep override */
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));

  auto context = EventsScubaUtil::getAllContext();

  ASSERT_EQ(2, context.eventNameStack.size());
  ASSERT_EQ("test2", context.eventNameStack[1].eventName);
  ASSERT_GE(
      context.eventNameStack[1].childrenLatency->count(),
      std::chrono::milliseconds(2000).count());
}

void test1() {
  auto sg = EVENTS_SCUBA_UTIL_SAMPLE_GUARD(__FUNCTION__);

  test2();

  auto context = EventsScubaUtil::getAllContext();
  ASSERT_EQ(1, context.eventNameStack.size());
  ASSERT_EQ("test1", context.eventNameStack[0].eventName);
  ASSERT_GE(
      context.eventNameStack[0].childrenLatency->count(),
      std::chrono::milliseconds(3000).count());
}

TEST_F(EventsScubaUtilTest, LatencyTest) {
  test1();
}

void c3() {
  auto sg = EVENTS_SCUBA_UTIL_SAMPLE_GUARD(__FUNCTION__);
  throw std::runtime_error("test");
}

void c2() {
  auto sg = EVENTS_SCUBA_UTIL_SAMPLE_GUARD(__FUNCTION__);
  c3();
}

void c1() {
  auto sg = EVENTS_SCUBA_UTIL_SAMPLE_GUARD(__FUNCTION__);
  auto& sample = sg.sample();
  ASSERT_FALSE(sample.hasException);

  try {
    c2();
  } catch (const std::exception& ex) {
    sample.setExceptionInfo(ex);
  }

  ASSERT_TRUE(sample.hasException);
  ASSERT_FALSE(sample.exceptionMessage.empty());
  ASSERT_GT(sample.stackTrace.size(), 0);
}

TEST_F(EventsScubaUtilTest, CatchException) {
  c1();
}

void d3() {
  auto sg = EVENTS_SCUBA_UTIL_SAMPLE_GUARD(__FUNCTION__);
  sg.sample().setError("test error");
  ASSERT_GT(sg.sample().stackTrace.size(), 0);
}

void d2() {
  auto sg = EVENTS_SCUBA_UTIL_SAMPLE_GUARD(__FUNCTION__);
  d3();
}

void d1() {
  auto sg = EVENTS_SCUBA_UTIL_SAMPLE_GUARD(__FUNCTION__);
  d2();
}

TEST_F(EventsScubaUtilTest, IncludeStackTrace) {
  d1();
}
