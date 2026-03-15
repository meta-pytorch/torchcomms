// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/profiler/Profiler.h"
#include <gtest/gtest.h>
#include "comms/ctran/profiler/AlgoProfilerReport.h"
#include "comms/ctran/profiler/IAlgoProfilerReporter.h"

using namespace ::testing;

namespace ctran {

// Mock reporter that captures the report for verification
class MockAlgoProfilerReporter : public IAlgoProfilerReporter {
 public:
  void report(const AlgoProfilerReport& report) override {
    reportCalled_ = true;
    lastReport_ = report;
    // Deep-copy the AlgoContext since the pointer may become invalid
    if (report.algoContext) {
      capturedAlgoContext_ = *report.algoContext;
    }
    reportCount_++;
  }

  bool reportCalled_{false};
  AlgoProfilerReport lastReport_;
  AlgoContext capturedAlgoContext_;
  int reportCount_{0};
};

class ProfilerTest : public ::testing::Test {
 public:
  void SetUp() override {
    // Allocate a zero-initialized CtranComm-sized buffer to avoid pulling in
    // the heavy ctran_lib dependency. The Profiler only accesses
    // comm_->logMetaData_ which is a trivial POD struct, so zero-init is safe.
    commBuf_.resize(sizeof(CtranComm), 0);
    comm_ = reinterpret_cast<CtranComm*>(commBuf_.data());
    profiler_ = std::make_shared<ctran::Profiler>(comm_);
  }
  void TearDown() override {
    comm_ = nullptr;
  }

  uint64_t getOpCount() {
    return profiler_->opCount_;
  }

 protected:
  std::vector<char> commBuf_;
  CtranComm* comm_{nullptr};
  std::shared_ptr<ctran::Profiler> profiler_{nullptr};
};

TEST_F(ProfilerTest, testInitForEachColl) {
  uint64_t opCount = 100;
  // test negative sampling weight
  profiler_->initForEachColl(opCount, -1);
  EXPECT_FALSE(profiler_->shouldTrace());
  EXPECT_NE(getOpCount(), opCount);

  // test zero sampling weight
  profiler_->initForEachColl(opCount, 0);
  EXPECT_FALSE(profiler_->shouldTrace());
  EXPECT_NE(getOpCount(), opCount);

  // test sampling weight = 1
  profiler_->initForEachColl(opCount, 1);
  EXPECT_TRUE(profiler_->shouldTrace());
  EXPECT_EQ(getOpCount(), opCount);

  // test opCount is the multiple of sampling weight
  profiler_->initForEachColl(opCount, 20);
  EXPECT_TRUE(profiler_->shouldTrace());
  EXPECT_EQ(getOpCount(), opCount);

  // test opCount is not the multiple of sampling weight
  ++opCount;
  profiler_->initForEachColl(opCount, 20);
  EXPECT_FALSE(profiler_->shouldTrace());
  EXPECT_NE(getOpCount(), opCount);
}

TEST_F(ProfilerTest, testSetReporter) {
  auto mockReporter = std::make_unique<MockAlgoProfilerReporter>();
  // setReporter should not throw
  EXPECT_NO_THROW(profiler_->setReporter(std::move(mockReporter)));
}

TEST_F(ProfilerTest, testReportToScubaCallsReporter) {
  auto mockReporter = std::make_unique<MockAlgoProfilerReporter>();
  auto* mockPtr = mockReporter.get();
  profiler_->setReporter(std::move(mockReporter));

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

  // Report
  profiler_->reportToScuba();

  // Verify reporter was called
  EXPECT_TRUE(mockPtr->reportCalled_);
  EXPECT_EQ(mockPtr->reportCount_, 1);
  EXPECT_EQ(mockPtr->lastReport_.opCount, 100);
  EXPECT_EQ(mockPtr->capturedAlgoContext_.algorithmName, "testAlgo");
  EXPECT_EQ(mockPtr->capturedAlgoContext_.deviceName, "gpu0");
  EXPECT_EQ(mockPtr->capturedAlgoContext_.sendContext.totalBytes, 1024);
  EXPECT_EQ(mockPtr->capturedAlgoContext_.recvContext.totalBytes, 2048);
  EXPECT_EQ(mockPtr->capturedAlgoContext_.peerRank, 3);

  // shouldTrace should be reset after reportToScuba
  EXPECT_FALSE(profiler_->shouldTrace());
}

TEST_F(ProfilerTest, testReportToScubaNotCalledWhenNotTracing) {
  auto mockReporter = std::make_unique<MockAlgoProfilerReporter>();
  auto* mockPtr = mockReporter.get();
  profiler_->setReporter(std::move(mockReporter));

  // Don't init tracing
  profiler_->reportToScuba();

  EXPECT_FALSE(mockPtr->reportCalled_);
}

} // namespace ctran
