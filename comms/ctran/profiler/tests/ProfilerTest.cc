// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/profiler/Profiler.h"
#include <gtest/gtest.h>
#include "comms/ctran/profiler/tests/MockProfilerReporter.h"
#include "comms/ctran/profiler/tests/ProfilerHarness.h"

using namespace ::testing;

namespace ctran {

class ProfilerTest : public ::testing::Test {
 public:
  void SetUp() override {
    comm_ = fakeComm_.get();
    profiler_ = std::make_unique<ctran::Profiler>(comm_);
  }
  void TearDown() override {
    comm_ = nullptr;
  }

  uint64_t getOpCount() {
    return profiler_->getOpCount();
  }

 protected:
  ctran::test::FakeProfilerComm fakeComm_;
  CtranComm* comm_{nullptr};
  std::unique_ptr<ctran::Profiler> profiler_{nullptr};
};

TEST_F(ProfilerTest, testInitForEachColl) {
  uint64_t opCount = 100;
  // test negative sampling weight
  profiler_->initForEachColl(opCount, -1);
  EXPECT_FALSE(profiler_->shouldTrace());
  EXPECT_NE(profiler_->getOpCount(), opCount);

  // test zero sampling weight
  profiler_->initForEachColl(opCount, 0);
  EXPECT_FALSE(profiler_->shouldTrace());
  EXPECT_NE(profiler_->getOpCount(), opCount);

  // test sampling weight = 1
  profiler_->initForEachColl(opCount, 1);
  EXPECT_TRUE(profiler_->shouldTrace());
  EXPECT_EQ(profiler_->getOpCount(), opCount);

  // test opCount is the multiple of sampling weight
  profiler_->initForEachColl(opCount, 20);
  EXPECT_TRUE(profiler_->shouldTrace());
  EXPECT_EQ(profiler_->getOpCount(), opCount);

  // test opCount is not the multiple of sampling weight
  ++opCount;
  profiler_->initForEachColl(opCount, 20);
  EXPECT_FALSE(profiler_->shouldTrace());
  EXPECT_NE(profiler_->getOpCount(), opCount);
}

TEST_F(ProfilerTest, testForceTraceOverridesSamplingWeight) {
  // Without forceTrace, opCount=101 with weight=20 should not trace
  profiler_->initForEachColl(101, 20);
  EXPECT_FALSE(profiler_->shouldTrace());

  // With forceTrace, same opCount/weight should trace
  profiler_->setForceTrace(true);
  profiler_->initForEachColl(101, 20);
  EXPECT_TRUE(profiler_->shouldTrace());
  EXPECT_EQ(getOpCount(), 101);

  // forceTrace persists until caller clears it
  profiler_->initForEachColl(102, 20);
  EXPECT_TRUE(profiler_->shouldTrace());

  // Caller clears forceTrace
  profiler_->setForceTrace(false);
  profiler_->initForEachColl(101, 20);
  EXPECT_FALSE(profiler_->shouldTrace());
}

TEST_F(ProfilerTest, testDefaultReporterType) {
  // Default constructor should use default reporter (no crash on reportToScuba)
  auto profiler = std::make_unique<ctran::Profiler>(comm_);
  EXPECT_NO_THROW(ctran::test::runOneCollective(*profiler, 100, 1));
}

TEST_F(ProfilerTest, testReportToScubaCallsReporter) {
  auto mockReporter = std::make_unique<StrictMock<MockProfilerReporter>>();
  auto* mockPtr = mockReporter.get();
  profiler_ = std::make_unique<ctran::Profiler>(comm_, std::move(mockReporter));

  // Set up profiler state
  profiler_->initForEachColl(100, 1);
  ASSERT_TRUE(profiler_->shouldTrace());

  // Set algo context
  profiler_->algoContext.algorithmName = "testAlgo";
  profiler_->algoContext.deviceName = "gpu0";
  profiler_->algoContext.sendContext.totalBytes = 1024;
  profiler_->algoContext.recvContext.totalBytes = 2048;
  profiler_->algoContext.peerRank = 3;

  // Simulate event timing
  profiler_->startEvent(ctran::ProfilerEvent::BUF_REG);
  profiler_->endEvent(ctran::ProfilerEvent::BUF_REG);
  profiler_->startEvent(ctran::ProfilerEvent::ALGO_CTRL);
  profiler_->endEvent(ctran::ProfilerEvent::ALGO_CTRL);
  profiler_->startEvent(ctran::ProfilerEvent::ALGO_DATA);
  profiler_->endEvent(ctran::ProfilerEvent::ALGO_DATA);

  // Expect exactly one call and capture the report
  AlgoProfilerReport capturedReport;
  AlgoContext capturedAlgoContext;
  EXPECT_CALL(*mockPtr, report(_))
      .WillOnce([&](const AlgoProfilerReport& report) {
        capturedReport = report;
        if (report.algoContext) {
          capturedAlgoContext = *report.algoContext;
        }
      });

  profiler_->reportToScuba();

  // Verify captured report contents
  EXPECT_EQ(capturedReport.opCount, 100);
  EXPECT_EQ(capturedAlgoContext.algorithmName, "testAlgo");
  EXPECT_EQ(capturedAlgoContext.deviceName, "gpu0");
  EXPECT_EQ(capturedAlgoContext.sendContext.totalBytes, 1024);
  EXPECT_EQ(capturedAlgoContext.recvContext.totalBytes, 2048);
  EXPECT_EQ(capturedAlgoContext.peerRank, 3);

  // shouldTrace should be reset after reportToScuba
  EXPECT_FALSE(profiler_->shouldTrace());
}

TEST_F(ProfilerTest, testReportToScubaNotCalledWhenNotTracing) {
  auto mockReporter = std::make_unique<StrictMock<MockProfilerReporter>>();
  profiler_ = std::make_unique<ctran::Profiler>(comm_, std::move(mockReporter));

  // Don't init tracing — StrictMock will fail if report() is called
  profiler_->reportToScuba();
}

} // namespace ctran
