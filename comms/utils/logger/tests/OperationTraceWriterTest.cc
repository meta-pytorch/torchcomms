// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include "comms/utils/logger/OperationTraceWriter.h"

using comms::logger::IOperationTraceWriter;
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
