// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include "comms/utils/logger/OperationTraceWriter.h"

using comms::logger::IOperationTraceWriter;
using comms::logger::OperationTraceLogger;
using comms::logger::OperationTraceSample;
using comms::logger::OperationTraceWriterRegistry;

namespace {

class MockOperationTraceWriter : public IOperationTraceWriter {
 public:
  bool isEnabled() const override {
    return enabled;
  }

  void addSample(OperationTraceSample sample) override {
    samples.push_back(std::move(sample));
  }

  bool enabled{true};
  std::vector<OperationTraceSample> samples;
};

class OperationTraceWriterTest : public ::testing::Test {
 protected:
  void TearDown() override {
    OperationTraceWriterRegistry::set(nullptr);
  }
};

} // namespace

TEST_F(OperationTraceWriterTest, RegistryReturnsNullByDefault) {
  EXPECT_EQ(OperationTraceWriterRegistry::get(), nullptr);
}

TEST_F(OperationTraceWriterTest, RegisteredWriterReceivesSamples) {
  auto mock = std::make_shared<MockOperationTraceWriter>();
  OperationTraceWriterRegistry::set(mock);

  auto* writer = OperationTraceWriterRegistry::get();
  ASSERT_NE(writer, nullptr);
  EXPECT_TRUE(writer->isEnabled());

  OperationTraceSample sample;
  sample.mcclop = "TEST_OP";
  sample.event = "TEST_OP_END";
  sample.timestampUs = 1000000;
  sample.rank = 3;
  sample.commHash = 12345;
  sample.commId = 99;
  sample.worldSize = 8;
  sample.opCount = 42;
  sample.durationUs = 500;
  writer->addSample(std::move(sample));

  ASSERT_EQ(mock->samples.size(), 1);
  const auto& s = mock->samples[0];
  EXPECT_EQ(s.mcclop, "TEST_OP");
  EXPECT_EQ(s.event, "TEST_OP_END");
  EXPECT_EQ(s.timestampUs, 1000000);
  EXPECT_EQ(s.rank, 3);
  EXPECT_EQ(s.commHash, 12345);
  EXPECT_EQ(s.commId, 99);
  EXPECT_EQ(s.worldSize, 8);
  EXPECT_EQ(s.opCount, 42);
  ASSERT_TRUE(s.durationUs.has_value());
  EXPECT_EQ(s.durationUs.value(), 500);
}

TEST_F(OperationTraceWriterTest, DisabledWriterSkipsSamples) {
  auto mock = std::make_shared<MockOperationTraceWriter>();
  mock->enabled = false;
  OperationTraceWriterRegistry::set(mock);

  auto* writer = OperationTraceWriterRegistry::get();
  ASSERT_NE(writer, nullptr);
  EXPECT_FALSE(writer->isEnabled());

  if (writer->isEnabled()) {
    OperationTraceSample sample;
    writer->addSample(std::move(sample));
  }
  EXPECT_EQ(mock->samples.size(), 0);
}

// --- OperationTraceLogger tests ---

TEST_F(OperationTraceWriterTest, LoggerIsInactiveWithNoWriter) {
  OperationTraceLogger logger("GPE_EXECUTION", 0, 111, 222, 8, 1);
  EXPECT_FALSE(logger.isActive());
  // Should be a no-op, not crash
  logger.logEvent("GPE_EXECUTION_START");
}

TEST_F(OperationTraceWriterTest, LoggerIsInactiveWhenDisabled) {
  auto mock = std::make_shared<MockOperationTraceWriter>();
  mock->enabled = false;
  OperationTraceWriterRegistry::set(mock);

  OperationTraceLogger logger("GPE_EXECUTION", 0, 111, 222, 8, 1);
  EXPECT_FALSE(logger.isActive());
  logger.logEvent("GPE_EXECUTION_START");
  EXPECT_EQ(mock->samples.size(), 0);
}

TEST_F(OperationTraceWriterTest, LoggerLogEventAutoTimestamp) {
  auto mock = std::make_shared<MockOperationTraceWriter>();
  OperationTraceWriterRegistry::set(mock);

  OperationTraceLogger logger("GPE_EXECUTION", 3, 111, 222, 8, 42);
  ASSERT_TRUE(logger.isActive());

  logger.logEvent("GPE_EXECUTION_START");

  ASSERT_EQ(mock->samples.size(), 1);
  const auto& s = mock->samples[0];
  EXPECT_EQ(s.mcclop, "GPE_EXECUTION");
  EXPECT_EQ(s.event, "GPE_EXECUTION_START");
  EXPECT_GT(s.timestampUs, 0);
  EXPECT_EQ(s.rank, 3);
  EXPECT_EQ(s.commHash, 111);
  EXPECT_EQ(s.commId, 222);
  EXPECT_EQ(s.worldSize, 8);
  EXPECT_EQ(s.opCount, 42);
  EXPECT_FALSE(s.durationUs.has_value());
}

TEST_F(OperationTraceWriterTest, LoggerLogEventWithTimestampAndDuration) {
  auto mock = std::make_shared<MockOperationTraceWriter>();
  OperationTraceWriterRegistry::set(mock);

  OperationTraceLogger logger("GPE_EXECUTION", 0, 111, 222, 8, 1);
  logger.logEvent("GPE_EXECUTION_END", 5000, 3000);

  ASSERT_EQ(mock->samples.size(), 1);
  const auto& s = mock->samples[0];
  EXPECT_EQ(s.event, "GPE_EXECUTION_END");
  EXPECT_EQ(s.timestampUs, 5000);
  ASSERT_TRUE(s.durationUs.has_value());
  EXPECT_EQ(s.durationUs.value(), 3000);
}

TEST_F(OperationTraceWriterTest, LoggerGpeContextAppliedToAllEvents) {
  auto mock = std::make_shared<MockOperationTraceWriter>();
  OperationTraceWriterRegistry::set(mock);

  OperationTraceLogger logger("GPE_EXECUTION", 0, 111, 222, 8, 1);
  logger.setGpeContext("AllGather", 4, 256, false);

  logger.logEvent("GPE_EXECUTION_START");
  logger.logEvent("GPE_EXECUTION_END", 5000, 3000);

  ASSERT_EQ(mock->samples.size(), 2);
  for (const auto& s : mock->samples) {
    ASSERT_TRUE(s.gpeKernelType.has_value());
    EXPECT_EQ(s.gpeKernelType.value(), "AllGather");
    ASSERT_TRUE(s.gpeNumBlocks.has_value());
    EXPECT_EQ(s.gpeNumBlocks.value(), 4);
    ASSERT_TRUE(s.gpeNumThreads.has_value());
    EXPECT_EQ(s.gpeNumThreads.value(), 256);
    ASSERT_TRUE(s.gpePersistent.has_value());
    EXPECT_FALSE(s.gpePersistent.value());
  }
}

// Simulate the exact 6-event GPE tracing pattern from CtranGpeImpl.cc
// gpeThreadFn() to verify correctness of the instrumentation sequence.
TEST_F(OperationTraceWriterTest, GpeThreadTracingPattern) {
  auto mock = std::make_shared<MockOperationTraceWriter>();
  OperationTraceWriterRegistry::set(mock);

  // Simulated timestamps (microseconds)
  const int64_t tEnqueue = 1000;
  const int64_t tDequeue = 1050;
  const int64_t tKernelWaitStart = 1055;
  const int64_t tKernelWaitEnd = 1100;
  const int64_t tCpuStart = 1105;
  const int64_t tCpuEnd = 1400;
  const int64_t tTermStart = 1405;
  const int64_t tTermEnd = 1500;
  const int64_t tEnd = 1505;

  // Mirror gpeThreadFn() instrumentation
  OperationTraceLogger traceLogger("GPE_EXECUTION", 2, 0xABCD, 42, 16, 7);
  ASSERT_TRUE(traceLogger.isActive());
  traceLogger.setGpeContext("AllReduce", 8, 512, false);

  // 1. GPE_EXECUTION_START (at enqueue time, no duration)
  traceLogger.logEvent("GPE_EXECUTION_START", tEnqueue);

  // 2. GPE_QUEUE_WAIT_END (dequeue time, duration = dequeue - enqueue)
  traceLogger.logEvent("GPE_QUEUE_WAIT_END", tDequeue, tDequeue - tEnqueue);

  // 3. GPE_KERNEL_WAIT_END (kernel started, duration = wait time)
  traceLogger.logEvent(
      "GPE_KERNEL_WAIT_END", tKernelWaitEnd, tKernelWaitEnd - tKernelWaitStart);

  // 4. GPE_CPU_EXECUTION_END (collective done, duration = cpu time)
  traceLogger.logEvent("GPE_CPU_EXECUTION_END", tCpuEnd, tCpuEnd - tCpuStart);

  // 5. GPE_GPU_TERMINATE_END (kernel stopped, duration = term time)
  traceLogger.logEvent(
      "GPE_GPU_TERMINATE_END", tTermEnd, tTermEnd - tTermStart);

  // 6. GPE_EXECUTION_END (total, duration = end - enqueue)
  traceLogger.logEvent("GPE_EXECUTION_END", tEnd, tEnd - tEnqueue);

  // Verify all 6 events were logged
  ASSERT_EQ(mock->samples.size(), 6);

  // Verify event names in order
  EXPECT_EQ(mock->samples[0].event, "GPE_EXECUTION_START");
  EXPECT_EQ(mock->samples[1].event, "GPE_QUEUE_WAIT_END");
  EXPECT_EQ(mock->samples[2].event, "GPE_KERNEL_WAIT_END");
  EXPECT_EQ(mock->samples[3].event, "GPE_CPU_EXECUTION_END");
  EXPECT_EQ(mock->samples[4].event, "GPE_GPU_TERMINATE_END");
  EXPECT_EQ(mock->samples[5].event, "GPE_EXECUTION_END");

  // Verify timestamps
  EXPECT_EQ(mock->samples[0].timestampUs, tEnqueue);
  EXPECT_EQ(mock->samples[5].timestampUs, tEnd);

  // Verify durations
  EXPECT_FALSE(
      mock->samples[0].durationUs.has_value()); // START has no duration
  EXPECT_EQ(mock->samples[1].durationUs.value(), 50); // queue wait
  EXPECT_EQ(mock->samples[2].durationUs.value(), 45); // kernel wait
  EXPECT_EQ(mock->samples[3].durationUs.value(), 295); // cpu execution
  EXPECT_EQ(mock->samples[4].durationUs.value(), 95); // gpu terminate
  EXPECT_EQ(mock->samples[5].durationUs.value(), 505); // total

  // Verify identity fields consistent across all events
  for (const auto& s : mock->samples) {
    EXPECT_EQ(s.mcclop, "GPE_EXECUTION");
    EXPECT_EQ(s.rank, 2);
    EXPECT_EQ(s.commHash, 0xABCD);
    EXPECT_EQ(s.commId, 42);
    EXPECT_EQ(s.worldSize, 16);
    EXPECT_EQ(s.opCount, 7);
    ASSERT_TRUE(s.gpeKernelType.has_value());
    EXPECT_EQ(s.gpeKernelType.value(), "AllReduce");
    EXPECT_EQ(s.gpeNumBlocks.value(), 8);
    EXPECT_EQ(s.gpeNumThreads.value(), 512);
    EXPECT_FALSE(s.gpePersistent.value());
  }
}

// Verify GPE tracing is completely skipped when no writer is registered
TEST_F(OperationTraceWriterTest, GpeTracingNoOpWithoutWriter) {
  // No writer registered
  OperationTraceLogger traceLogger("GPE_EXECUTION", 0, 111, 222, 8, 1);
  EXPECT_FALSE(traceLogger.isActive());

  // All these should be no-ops
  traceLogger.setGpeContext("AllReduce", 4, 256, false);
  traceLogger.logEvent("GPE_EXECUTION_START", 1000);
  traceLogger.logEvent("GPE_QUEUE_WAIT_END", 1050, 50);
  traceLogger.logEvent("GPE_EXECUTION_END", 1100, 100);
  // No crash, no samples — verified by mock not existing
}
