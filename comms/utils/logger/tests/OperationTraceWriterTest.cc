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
