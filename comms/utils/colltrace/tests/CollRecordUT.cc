// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <chrono>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "comms/utils/colltrace/CollRecord.h"
#include "comms/utils/colltrace/tests/MockTypes.h"

using namespace meta::comms::colltrace;
using namespace std::chrono;
using namespace testing;

namespace {

class CollTimingRecordTest : public Test {
 protected:
  void SetUp() override {
    // Initialize with current time
    now_ = system_clock::now();

    // Set different timestamps for testing
    timingRecord_.setPreviousCollEndTs(now_ - minutes(5));
    timingRecord_.setCollEnqueueTs(now_ - minutes(3));
    timingRecord_.setCollStartTs(now_ - minutes(2));
    timingRecord_.setCollEndTs(now_ - minutes(1));
  }

  CollTimingRecord timingRecord_;
  system_clock::time_point now_;
};

TEST_F(CollTimingRecordTest, GettersAndSetters) {
  // Test that the getters return the values set in SetUp
  EXPECT_EQ(timingRecord_.getPreviousCollEndTs(), now_ - minutes(5));
  EXPECT_EQ(timingRecord_.getCollEnqueueTs(), now_ - minutes(3));
  EXPECT_EQ(timingRecord_.getCollStartTs(), now_ - minutes(2));
  EXPECT_EQ(timingRecord_.getCollEndTs(), now_ - minutes(1));

  // Test setting new values
  auto newTime = now_ + minutes(1);
  timingRecord_.setPreviousCollEndTs(newTime);
  EXPECT_EQ(timingRecord_.getPreviousCollEndTs(), newTime);

  newTime = now_ + minutes(2);
  timingRecord_.setCollEnqueueTs(newTime);
  EXPECT_EQ(timingRecord_.getCollEnqueueTs(), newTime);

  newTime = now_ + minutes(3);
  timingRecord_.setCollStartTs(newTime);
  EXPECT_EQ(timingRecord_.getCollStartTs(), newTime);

  newTime = now_ + minutes(4);
  timingRecord_.setCollEndTs(newTime);
  EXPECT_EQ(timingRecord_.getCollEndTs(), newTime);
}

TEST_F(CollTimingRecordTest, HashCalculation) {
  // Create a copy with the same values
  CollTimingRecord copy;
  copy.setPreviousCollEndTs(timingRecord_.getPreviousCollEndTs());
  copy.setCollEnqueueTs(timingRecord_.getCollEnqueueTs());
  copy.setCollStartTs(timingRecord_.getCollStartTs());
  copy.setCollEndTs(timingRecord_.getCollEndTs());

  // Hashes should be equal for equal objects
  EXPECT_EQ(timingRecord_.hash(), copy.hash());

  // Change one value, hash should be different
  copy.setCollEndTs(now_ + hours(1));
  EXPECT_NE(timingRecord_.hash(), copy.hash());
}

TEST_F(CollTimingRecordTest, EqualityComparison) {
  // Create a copy with the same values
  CollTimingRecord copy;
  copy.setPreviousCollEndTs(timingRecord_.getPreviousCollEndTs());
  copy.setCollEnqueueTs(timingRecord_.getCollEnqueueTs());
  copy.setCollStartTs(timingRecord_.getCollStartTs());
  copy.setCollEndTs(timingRecord_.getCollEndTs());

  // Objects should be equal
  EXPECT_TRUE(timingRecord_.equals(copy));
  EXPECT_TRUE(timingRecord_ == copy);

  // Change one value, objects should not be equal
  copy.setCollEndTs(now_ + hours(1));
  EXPECT_FALSE(timingRecord_.equals(copy));
  EXPECT_FALSE(timingRecord_ == copy);
}

TEST_F(CollTimingRecordTest, DynamicConversion) {
  // Convert to dynamic
  folly::dynamic result = timingRecord_.toDynamic();

  // Check that all expected fields are present
  EXPECT_TRUE(result.count("enqueueTs"));
  EXPECT_TRUE(result.count("startTs"));
  EXPECT_TRUE(result.count("endTs"));
  EXPECT_TRUE(result.count("previousEndTs"));
  EXPECT_TRUE(result.count("latencyUs"));
  EXPECT_TRUE(result.count("ExecutionTimeUs"));
  EXPECT_TRUE(result.count("QueueingTimeUs"));
  EXPECT_TRUE(result.count("InterCollTimeUs"));

  // Check that the values are as expected
  EXPECT_EQ(
      result["latencyUs"].asInt(),
      duration_cast<microseconds>(
          timingRecord_.getCollEndTs() - timingRecord_.getCollStartTs())
          .count());

  EXPECT_EQ(
      result["QueueingTimeUs"].asInt(),
      duration_cast<microseconds>(
          timingRecord_.getCollStartTs() - timingRecord_.getCollEnqueueTs())
          .count());

  EXPECT_EQ(
      result["InterCollTimeUs"].asInt(),
      duration_cast<microseconds>(
          timingRecord_.getCollStartTs() - timingRecord_.getPreviousCollEndTs())
          .count());

  // ExecutionTimeUs should be the same as latencyUs
  EXPECT_EQ(result["ExecutionTimeUs"].asInt(), result["latencyUs"].asInt());
}

class CollRecordTest : public Test {
 protected:
  void SetUp() override {
    // Create a mock metadata
    auto metadata = std::make_unique<NiceMock<MockCollMetadata>>();
    mockMetadata_ = metadata.get();

    // Set up expectations for the mock
    ON_CALL(*mockMetadata_, hash()).WillByDefault(Return(12345));
    ON_CALL(*mockMetadata_, equals(_)).WillByDefault(Return(true));
    ON_CALL(*mockMetadata_, getMetadataType())
        .WillByDefault(Return("MockMetadata"));

    folly::dynamic metadataDynamic = folly::dynamic::object;
    metadataDynamic["type"] = "MockMetadata";
    metadataDynamic["value"] = 42;
    ON_CALL(*mockMetadata_, toDynamic()).WillByDefault(Return(metadataDynamic));

    // Create the CollRecord
    collRecord_ = std::make_unique<CollRecord>(123, std::move(metadata));

    // Set up timing info
    now_ = system_clock::now();
    collRecord_->getTimingInfo().setPreviousCollEndTs(now_ - minutes(5));
    collRecord_->getTimingInfo().setCollEnqueueTs(now_ - minutes(3));
    collRecord_->getTimingInfo().setCollStartTs(now_ - minutes(2));
    collRecord_->getTimingInfo().setCollEndTs(now_ - minutes(1));
  }

  std::unique_ptr<CollRecord> collRecord_;
  MockCollMetadata* mockMetadata_; // Owned by collRecord_
  system_clock::time_point now_;
};

TEST_F(CollRecordTest, ConstructorAndGetters) {
  // Check that the getters return the expected values
  EXPECT_EQ(collRecord_->getCollId(), 123);
  EXPECT_EQ(collRecord_->getCollMetadata(), mockMetadata_);

  // Check that the timing info is accessible and has the expected values
  EXPECT_EQ(
      collRecord_->getTimingInfo().getPreviousCollEndTs(), now_ - minutes(5));
  EXPECT_EQ(collRecord_->getTimingInfo().getCollEnqueueTs(), now_ - minutes(3));
  EXPECT_EQ(collRecord_->getTimingInfo().getCollStartTs(), now_ - minutes(2));
  EXPECT_EQ(collRecord_->getTimingInfo().getCollEndTs(), now_ - minutes(1));
}

TEST_F(CollRecordTest, HashCalculation) {
  // Create a copy with the same values
  auto metadata = std::make_unique<NiceMock<MockCollMetadata>>();
  MockCollMetadata* mockMetadata2 = metadata.get();

  // Set up expectations for the mock
  ON_CALL(*mockMetadata2, hash()).WillByDefault(Return(12345));
  ON_CALL(*mockMetadata2, equals(_)).WillByDefault(Return(true));

  auto copy = std::make_unique<CollRecord>(123, std::move(metadata));
  copy->getTimingInfo().setPreviousCollEndTs(now_ - minutes(5));
  copy->getTimingInfo().setCollEnqueueTs(now_ - minutes(3));
  copy->getTimingInfo().setCollStartTs(now_ - minutes(2));
  copy->getTimingInfo().setCollEndTs(now_ - minutes(1));

  // Hashes should be equal for equal objects
  EXPECT_EQ(collRecord_->hash(), copy->hash());

  // Change the collId, hash should be different
  auto differentId = std::make_unique<CollRecord>(
      456, std::make_unique<NiceMock<MockCollMetadata>>());
  differentId->getTimingInfo().setPreviousCollEndTs(now_ - minutes(5));
  differentId->getTimingInfo().setCollEnqueueTs(now_ - minutes(3));
  differentId->getTimingInfo().setCollStartTs(now_ - minutes(2));
  differentId->getTimingInfo().setCollEndTs(now_ - minutes(1));

  EXPECT_NE(collRecord_->hash(), differentId->hash());
}

TEST_F(CollRecordTest, EqualityComparison) {
  // Create a copy with the same values
  auto metadata = std::make_unique<NiceMock<MockCollMetadata>>();
  MockCollMetadata* mockMetadata2 = metadata.get();

  // Set up expectations for the mock
  ON_CALL(*mockMetadata2, hash()).WillByDefault(Return(12345));
  ON_CALL(*mockMetadata2, equals(_)).WillByDefault(Return(true));

  auto copy = std::make_unique<CollRecord>(123, std::move(metadata));
  copy->getTimingInfo().setPreviousCollEndTs(now_ - minutes(5));
  copy->getTimingInfo().setCollEnqueueTs(now_ - minutes(3));
  copy->getTimingInfo().setCollStartTs(now_ - minutes(2));
  copy->getTimingInfo().setCollEndTs(now_ - minutes(1));

  // Objects should be equal
  EXPECT_TRUE(collRecord_->equals(*copy));
  EXPECT_TRUE(*collRecord_ == *copy);

  // Change the collId, objects should not be equal
  auto differentId = std::make_unique<CollRecord>(
      456, std::make_unique<NiceMock<MockCollMetadata>>());
  differentId->getTimingInfo().setPreviousCollEndTs(now_ - minutes(5));
  differentId->getTimingInfo().setCollEnqueueTs(now_ - minutes(3));
  differentId->getTimingInfo().setCollStartTs(now_ - minutes(2));
  differentId->getTimingInfo().setCollEndTs(now_ - minutes(1));

  EXPECT_FALSE(collRecord_->equals(*differentId));
  EXPECT_FALSE(*collRecord_ == *differentId);
}

TEST_F(CollRecordTest, DynamicConversion) {
  // Convert to dynamic
  folly::dynamic result = collRecord_->toDynamic();

  // Check that all expected fields are present
  EXPECT_TRUE(result.count("collId"));
  EXPECT_TRUE(result.count("opCount"));
  EXPECT_TRUE(result.count("type"));
  EXPECT_TRUE(result.count("value"));
  EXPECT_TRUE(result.count("enqueueTs"));
  EXPECT_TRUE(result.count("startTs"));
  EXPECT_TRUE(result.count("endTs"));
  EXPECT_TRUE(result.count("previousEndTs"));

  // Check that the values are as expected
  EXPECT_EQ(result["collId"].asInt(), 123);
  EXPECT_EQ(result["opCount"].asInt(), 123);
  EXPECT_EQ(result["type"].asString(), "MockMetadata");
  EXPECT_EQ(result["value"].asInt(), 42);
}

TEST_F(CollRecordTest, NullMetadata) {
  // Create a CollRecord with null metadata
  auto recordWithNullMetadata = std::make_unique<CollRecord>(123, nullptr);

  // Should not crash when accessing metadata
  EXPECT_EQ(recordWithNullMetadata->getCollMetadata(), nullptr);

  // Should be able to calculate hash
  std::size_t hash = recordWithNullMetadata->hash();
  EXPECT_NE(hash, 0);

  // Should be able to convert to dynamic
  folly::dynamic result = recordWithNullMetadata->toDynamic();
  EXPECT_TRUE(result.count("collId"));
  EXPECT_TRUE(result.count("opCount"));

  // Should be equal to another record with null metadata
  auto anotherNullMetadata = std::make_unique<CollRecord>(123, nullptr);
  EXPECT_TRUE(recordWithNullMetadata->equals(*anotherNullMetadata));

  // Should not be equal to a record with non-null metadata
  EXPECT_FALSE(recordWithNullMetadata->equals(*collRecord_));
  EXPECT_FALSE(collRecord_->equals(*recordWithNullMetadata));
}

} // namespace
