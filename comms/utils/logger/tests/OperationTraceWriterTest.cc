// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include "comms/utils/logger/OperationTraceWriter.h"

using namespace comms::logger;

namespace {

// Simple mock writer that records calls for verification.
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
    // Reset the registry after each test to avoid cross-test contamination.
    OperationTraceWriterRegistry::set(nullptr);
  }
};

} // namespace

// Verify get() returns nullptr when no writer has been registered.
TEST_F(OperationTraceWriterTest, RegistryReturnsNullByDefault) {
  EXPECT_EQ(OperationTraceWriterRegistry::get(), nullptr);
}

// Verify the full DI contract: register a writer, query it, send a sample,
// and confirm all named fields arrive correctly.
TEST_F(OperationTraceWriterTest, RegisteredWriterReceivesSamples) {
  auto mock = std::make_shared<MockOperationTraceWriter>();
  OperationTraceWriterRegistry::set(mock);

  auto* writer = OperationTraceWriterRegistry::get();
  ASSERT_NE(writer, nullptr);
  EXPECT_TRUE(writer->isEnabled());

  // Build and submit a sample (mirrors what ctran GPE tracing does)
  OperationTraceSample sample;
  sample.mcclop = "GPE_EXECUTION";
  sample.event = "GPE_EXECUTION_END";
  sample.timestampUs = 1000000;
  sample.rank = 3;
  sample.commHash = 12345;
  sample.commId = 99;
  sample.worldSize = 8;
  sample.opCount = 42;
  sample.durationUs = 500;
  sample.gpeKernelType = "AllGather";
  sample.gpeNumBlocks = 4;
  sample.gpeNumThreads = 256;
  sample.gpePersistent = false;
  writer->addSample(std::move(sample));

  ASSERT_EQ(mock->samples.size(), 1);
  const auto& s = mock->samples[0];
  EXPECT_EQ(s.mcclop, "GPE_EXECUTION");
  EXPECT_EQ(s.event, "GPE_EXECUTION_END");
  EXPECT_EQ(s.timestampUs, 1000000);
  EXPECT_EQ(s.rank, 3);
  EXPECT_EQ(s.commHash, 12345);
  EXPECT_EQ(s.commId, 99);
  EXPECT_EQ(s.worldSize, 8);
  EXPECT_EQ(s.opCount, 42);
  ASSERT_TRUE(s.durationUs.has_value());
  EXPECT_EQ(s.durationUs.value(), 500);
  ASSERT_TRUE(s.gpeKernelType.has_value());
  EXPECT_EQ(s.gpeKernelType.value(), "AllGather");
  ASSERT_TRUE(s.gpeNumBlocks.has_value());
  EXPECT_EQ(s.gpeNumBlocks.value(), 4);
  ASSERT_TRUE(s.gpeNumThreads.has_value());
  EXPECT_EQ(s.gpeNumThreads.value(), 256);
  ASSERT_TRUE(s.gpePersistent.has_value());
  EXPECT_FALSE(s.gpePersistent.value());
}

// Verify that when the writer reports disabled, callers can skip work.
TEST_F(OperationTraceWriterTest, DisabledWriterSkipsSamples) {
  auto mock = std::make_shared<MockOperationTraceWriter>();
  mock->enabled = false;
  OperationTraceWriterRegistry::set(mock);

  auto* writer = OperationTraceWriterRegistry::get();
  ASSERT_NE(writer, nullptr);
  EXPECT_FALSE(writer->isEnabled());

  // Caller checks isEnabled() before building the sample — no sample added
  if (writer->isEnabled()) {
    OperationTraceSample sample;
    writer->addSample(std::move(sample));
  }
  EXPECT_EQ(mock->samples.size(), 0);
}

// Verify OperationTraceGuard logs START on construction and END on
// destruction with computed duration and context fields.
TEST_F(OperationTraceWriterTest, OperationGuardLogsStartAndEnd) {
  auto mock = std::make_shared<MockOperationTraceWriter>();
  OperationTraceWriterRegistry::set(mock);

  {
    OperationTraceSample sample;
    sample.mcclop = "GPE_EXECUTION";
    sample.rank = 1;
    sample.commHash = 999;
    OperationTraceGuard guard(std::move(sample));
    EXPECT_TRUE(guard.isActive());
    guard.sample().gpeKernelType = "AllReduce";
  }

  ASSERT_EQ(mock->samples.size(), 2);

  // START logged on construction
  const auto& start = mock->samples[0];
  EXPECT_EQ(start.event, "GPE_EXECUTION_START");
  EXPECT_EQ(start.rank, 1);
  EXPECT_EQ(start.commHash, 999);

  // END logged on destruction with duration + context
  const auto& end = mock->samples[1];
  EXPECT_EQ(end.event, "GPE_EXECUTION_END");
  EXPECT_EQ(end.rank, 1);
  EXPECT_EQ(end.commHash, 999);
  ASSERT_TRUE(end.durationUs.has_value());
  EXPECT_GE(end.durationUs.value(), 0);
  ASSERT_TRUE(end.gpeKernelType.has_value());
  EXPECT_EQ(end.gpeKernelType.value(), "AllReduce");
}

// Verify EventLoggerGuard logs event START/END within a parent
// operation guard, sharing identity fields.
TEST_F(OperationTraceWriterTest, EventGuardLogsStartAndEnd) {
  auto mock = std::make_shared<MockOperationTraceWriter>();
  OperationTraceWriterRegistry::set(mock);

  {
    OperationTraceSample sample;
    sample.mcclop = "GPE_EXECUTION";
    sample.rank = 2;
    sample.commHash = 777;
    OperationTraceGuard opGuard(std::move(sample));

    {
      EventLoggerGuard evGuard(opGuard, "GPE_KERNEL_WAIT");
    }
    // Event guard destructor fires before operation guard
  }

  // Expected order: OP_START, EVENT_START, EVENT_END, OP_END
  ASSERT_EQ(mock->samples.size(), 4);

  EXPECT_EQ(mock->samples[0].event, "GPE_EXECUTION_START");
  EXPECT_EQ(mock->samples[1].event, "GPE_KERNEL_WAIT_START");
  EXPECT_EQ(mock->samples[1].rank, 2);
  EXPECT_EQ(mock->samples[1].commHash, 777);
  EXPECT_EQ(mock->samples[2].event, "GPE_KERNEL_WAIT_END");
  ASSERT_TRUE(mock->samples[2].durationUs.has_value());
  EXPECT_GE(mock->samples[2].durationUs.value(), 0);
  EXPECT_EQ(mock->samples[3].event, "GPE_EXECUTION_END");
}
