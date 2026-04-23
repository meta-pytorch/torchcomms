// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/profiler/Profiler.h"
#include <gtest/gtest.h>
#include "comms/ctran/profiler/tests/MockProfilerReporter.h"
#include "comms/utils/cvars/nccl_cvars.h"

using namespace ::testing;

namespace ctran {

class ProfilerTest : public ::testing::Test {
 public:
  void SetUp() override {
    // Allocate a zero-initialized CtranComm-sized buffer to avoid pulling in
    // the heavy ctran_lib dependency. The Profiler only accesses
    // comm_->logMetaData_ which is a trivial POD struct, so zero-init is safe.
    commBuf_.resize(sizeof(CtranComm), 0);
    comm_ = reinterpret_cast<CtranComm*>(commBuf_.data());
    // Default: sample every op
    NCCL_CTRAN_ALGO_PROFILING_SAMPLING_WEIGHT = 1;
    profiler_ = std::make_unique<ctran::Profiler>(comm_);
  }
  void TearDown() override {
    comm_ = nullptr;
  }

  // Create a Profiler with a specific sampling weight
  std::unique_ptr<ctran::Profiler> makeProfiler(int samplingWeight) {
    NCCL_CTRAN_ALGO_PROFILING_SAMPLING_WEIGHT = samplingWeight;
    return std::make_unique<ctran::Profiler>(comm_);
  }

 protected:
  std::vector<char> commBuf_;
  CtranComm* comm_{nullptr};
  std::unique_ptr<ctran::Profiler> profiler_{nullptr};
};

TEST_F(ProfilerTest, testInitForEachColl) {
  uint64_t opCount = 100;

  // test negative sampling weight
  auto p1 = makeProfiler(-1);
  p1->initForEachColl(opCount);
  EXPECT_FALSE(p1->shouldTrace());

  // test zero sampling weight
  auto p2 = makeProfiler(0);
  p2->initForEachColl(opCount);
  EXPECT_FALSE(p2->shouldTrace());

  // test sampling weight = 1
  auto p3 = makeProfiler(1);
  p3->initForEachColl(opCount);
  EXPECT_TRUE(p3->shouldTrace());
  EXPECT_EQ(p3->getOpCount(), opCount);

  // test opCount is the multiple of sampling weight
  auto p4 = makeProfiler(20);
  p4->initForEachColl(opCount);
  EXPECT_TRUE(p4->shouldTrace());
  EXPECT_EQ(p4->getOpCount(), opCount);

  // test opCount is not the multiple of sampling weight
  ++opCount;
  auto p5 = makeProfiler(20);
  p5->initForEachColl(opCount);
  EXPECT_FALSE(p5->shouldTrace());
  EXPECT_NE(p5->getOpCount(), opCount);
}

TEST_F(ProfilerTest, testDefaultReporterType) {
  // Default constructor should use default reporter (no crash on reportToScuba)
  profiler_->initForEachColl(100);
  profiler_->startEvent(ctran::ProfilerEvent::BUF_REG);
  profiler_->endEvent(ctran::ProfilerEvent::BUF_REG);
  profiler_->startEvent(ctran::ProfilerEvent::ALGO_CTRL);
  profiler_->endEvent(ctran::ProfilerEvent::ALGO_CTRL);
  profiler_->startEvent(ctran::ProfilerEvent::ALGO_DATA);
  profiler_->endEvent(ctran::ProfilerEvent::ALGO_DATA);
  EXPECT_NO_THROW(profiler_->reportToScuba());
}

TEST_F(ProfilerTest, testReportToScubaCallsReporter) {
  auto mockReporter = std::make_unique<StrictMock<MockProfilerReporter>>();
  auto* mockPtr = mockReporter.get();
  profiler_ = std::make_unique<ctran::Profiler>(comm_, std::move(mockReporter));

  // Set up profiler state
  profiler_->initForEachColl(100);
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
