// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/ctran/profiler/Profiler.h"
#include <folly/json/json.h>
#include <gtest/gtest.h>
#include "comms/ctran/profiler/tests/MockAlgoProfilerReporter.h"
#include "comms/mccl/utils/logger/McclAlgoProfilerReporter.h"
#include "comms/mccl/utils/logger/McclDataTableWrapper.h"
#include "comms/mccl/utils/logger/McclOperationTraceTypes.h"
#include "comms/mccl/utils/logger/tests/MockMcclDataTable.h"
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
    profiler_ = std::make_shared<ctran::Profiler>(comm_);
  }
  void TearDown() override {
    comm_ = nullptr;
  }

  uint64_t getOpCount() {
    return profiler_->opCount_;
  }

  void setReporter(std::unique_ptr<ctran::IAlgoProfilerReporter> reporter) {
    profiler_->setReporter(std::move(reporter));
  }

 protected:
  std::vector<char> commBuf_;
  CtranComm* comm_{nullptr};
  std::shared_ptr<ctran::Profiler> profiler_{nullptr};
};

TEST_F(ProfilerTest, testInitForEachColl) {
  constexpr uint64_t opCount = 42;

  // test negative sampling weight — never traces
  profiler_->initForEachColl(opCount, -1);
  EXPECT_FALSE(profiler_->shouldTrace());

  // test zero sampling weight — never traces
  profiler_->initForEachColl(opCount, 0);
  EXPECT_FALSE(profiler_->shouldTrace());

  // test sampling weight = 1 — traces every call
  profiler_->initForEachColl(opCount, 1);
  EXPECT_TRUE(profiler_->shouldTrace());
  EXPECT_EQ(getOpCount(), opCount);

  profiler_->initForEachColl(opCount, 1);
  EXPECT_TRUE(profiler_->shouldTrace());
}

TEST_F(ProfilerTest, testSamplingByInvocationCount) {
  constexpr uint64_t opCount = 0; // frozen opCount (CUDA graph replay)
  constexpr int samplingWeight = 3;
  int traceCount = 0;

  // Sampling is based on invocation count, not opCount.
  // With weight=3, every 3rd invocation should trace.
  for (int i = 0; i < 9; ++i) {
    profiler_->initForEachColl(opCount, samplingWeight);
    if (profiler_->shouldTrace()) {
      traceCount++;
      EXPECT_EQ(getOpCount(), opCount);
    }
  }
  EXPECT_EQ(traceCount, 3);
}

TEST_F(ProfilerTest, testDefaultReporterType) {
  // Default constructor should use NCCLX reporter (no crash on reportToScuba)
  auto profiler = std::make_shared<ctran::Profiler>(comm_);
  profiler->initForEachColl(100, 1);
  profiler->startEvent(ctran::ProfilerEvent::BUF_REG);
  profiler->endEvent(ctran::ProfilerEvent::BUF_REG);
  profiler->startEvent(ctran::ProfilerEvent::ALGO_CTRL);
  profiler->endEvent(ctran::ProfilerEvent::ALGO_CTRL);
  profiler->startEvent(ctran::ProfilerEvent::ALGO_DATA);
  profiler->endEvent(ctran::ProfilerEvent::ALGO_DATA);
  EXPECT_NO_THROW(profiler->reportToScuba());
}

TEST_F(ProfilerTest, testReportToScubaCallsReporter) {
  // Use a custom reporter injected via the test friend access
  auto mockReporter = std::make_unique<MockAlgoProfilerReporter>();
  auto* mockPtr = mockReporter.get();
  setReporter(std::move(mockReporter));

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
  setReporter(std::move(mockReporter));

  // Don't init tracing
  profiler_->reportToScuba();

  EXPECT_FALSE(mockPtr->reportCalled_);
}

// Verify that constructing a Profiler with ReporterType::MCCL
// creates a McclAlgoProfilerReporter that writes samples to the MCCL scuba
// table with correct comm metadata and algo profiling fields.
TEST_F(ProfilerTest, testMcclReporterFactoryWiring) {
  using mccl::logger::McclCommLogMetadata;
  using mccl::logger::McclDataTableWrapper;
  using mccl::logger::testing::MockMcclDataTableFactory;
  using mccl::logger::testing::ThreadSafeFakeTable;

  // Test constants
  constexpr int64_t kCommId = 12345;
  constexpr int64_t kCommHash = 67890;
  constexpr int kRank = 2;
  constexpr int kNRanks = 8;
  constexpr int kGpuId = 3;
  constexpr int64_t kMcclcommUuid = 999;
  const std::string kHostname = "devgpu001";
  const std::string kJobId = "job_42";
  constexpr int kOpCount = 100;
  constexpr int kSamplingWeight = 1;
  const std::string kAlgoName = "ctdirect";
  constexpr size_t kSendBytes = 1024;
  constexpr size_t kRecvBytes = 2048;

  // Enable MCCL scuba logging
  MCCL_SCUBA_ENABLED = true;
  MCCL_SCUBA_LOG_LEVEL = MCCL_SCUBA_LOG_LEVEL::LOW;

  // Set up fake scuba table to capture samples
  auto fakeTable = std::make_unique<ThreadSafeFakeTable>();
  auto* fakeTablePtr = fakeTable.get();
  McclDataTableWrapper::init(
      std::make_unique<MockMcclDataTableFactory>(std::move(fakeTable)));

  // Set up comm metadata on the CtranComm buffer
  McclCommLogMetadata commMeta;
  commMeta.commId = kCommId;
  commMeta.commHash = kCommHash;
  commMeta.rank = kRank;
  commMeta.nRanks = kNRanks;
  commMeta.gpuId = kGpuId;
  commMeta.mcclcommUuid = kMcclcommUuid;
  commMeta.hostname = kHostname;
  commMeta.jobId = kJobId;

  // Wire the context pointer, simulating what McclComm::init does
  comm_->algoProfilerReporterCtx_ = &commMeta;

  // Register the MCCL reporter factory before constructing the profiler
  ctran::registerMcclAlgoProfilerReporter();

  // Construct profiler with MCCL reporter type — exercises the factory
  auto mcclProfiler =
      std::make_shared<ctran::Profiler>(comm_, ctran::ReporterType::MCCL);

  // Set up profiler state and run through events
  mcclProfiler->initForEachColl(kOpCount, kSamplingWeight);
  ASSERT_TRUE(mcclProfiler->shouldTrace());

  mcclProfiler->algoContext.algorithmName = kAlgoName;
  mcclProfiler->algoContext.sendContext.totalBytes = kSendBytes;
  mcclProfiler->algoContext.sendContext.messageSizes =
      std::to_string(kSendBytes);
  mcclProfiler->algoContext.recvContext.totalBytes = kRecvBytes;
  mcclProfiler->algoContext.recvContext.messageSizes =
      std::to_string(kRecvBytes);

  mcclProfiler->startEvent(ctran::ProfilerEvent::BUF_REG);
  mcclProfiler->endEvent(ctran::ProfilerEvent::BUF_REG);
  mcclProfiler->startEvent(ctran::ProfilerEvent::ALGO_CTRL);
  mcclProfiler->endEvent(ctran::ProfilerEvent::ALGO_CTRL);
  mcclProfiler->startEvent(ctran::ProfilerEvent::ALGO_DATA);
  mcclProfiler->endEvent(ctran::ProfilerEvent::ALGO_DATA);

  mcclProfiler->reportToScuba();

  // Verify sample was written to the MCCL scuba table
  ASSERT_EQ(fakeTablePtr->getSampleCount(), 1);

  auto jsons = fakeTablePtr->getSampleJsons();
  auto json = folly::parseJson(jsons[0]);

  // Verify comm metadata was propagated through the factory
  EXPECT_EQ(json["int"]["comm_id"].getInt(), kCommId);
  EXPECT_EQ(json["int"]["comm_hash"].getInt(), kCommHash);
  EXPECT_EQ(json["int"]["rank"].getInt(), kRank);
  EXPECT_EQ(json["int"]["world_size"].getInt(), kNRanks);
  EXPECT_EQ(json["int"]["gpu_id"].getInt(), kGpuId);
  EXPECT_EQ(json["int"]["mcclcomm_uuid"].getInt(), kMcclcommUuid);
  EXPECT_EQ(json["normal"]["hostname"].getString(), kHostname);
  EXPECT_EQ(json["normal"]["job_id"].getString(), kJobId);

  // Verify algo profiling fields
  EXPECT_EQ(json["normal"]["ctran_algo"].getString(), kAlgoName);
  EXPECT_EQ(json["int"]["op_count"].getInt(), kOpCount);
  EXPECT_EQ(
      json["int"]["send_total_bytes"].getInt(),
      static_cast<int64_t>(kSendBytes));
  EXPECT_EQ(
      json["int"]["recv_total_bytes"].getInt(),
      static_cast<int64_t>(kRecvBytes));
  EXPECT_EQ(
      json["normal"]["send_message_sizes"].getString(),
      std::to_string(kSendBytes));
  EXPECT_EQ(
      json["normal"]["recv_message_sizes"].getString(),
      std::to_string(kRecvBytes));

  // Clean up
  mcclProfiler.reset();
  McclDataTableWrapper::shutdown();
  MCCL_SCUBA_ENABLED = false;
}

} // namespace ctran
