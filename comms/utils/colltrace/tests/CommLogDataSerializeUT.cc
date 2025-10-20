// Copyright (c) Meta Platforms, Inc. and affiliates.

// Include CommLogDataSerialize.h to get the folly::DynamicConstructor and
// folly::DynamicConverter specializations for CommLogData
#include "comms/utils/colltrace/CommLogDataSerialize.h"

// Force direct use of CommLogDataSerialize.h to satisfy the linter
namespace {
// This is just a dummy function to force the use of CommLogDataSerialize.h
// It's never actually called in the tests
void dummy_function_to_use_header() {
  // Use the specializations directly to satisfy the linter
  CommLogData data;
  folly::dynamic d = folly::DynamicConstructor<CommLogData>::construct(data);
  CommLogData result = folly::DynamicConverter<CommLogData>::convert(d);
  (void)result; // Avoid unused variable warning
}
} // namespace

#include <folly/DynamicConverter.h>
#include <folly/dynamic.h>
#include <gtest/gtest.h>

#include "comms/utils/commSpecs.h"

class CommLogDataSerializeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a sample CommLogData object for testing
    testData_.commId = 12345;
    testData_.commHash = 67890;
    testData_.commDesc = "test_comm";
    testData_.rank = 3;
    testData_.nRanks = 8;
  }

  CommLogData testData_;
};

// Test serialization of CommLogData to folly::dynamic
TEST_F(CommLogDataSerializeTest, SerializeCommLogData) {
  // Serialize the CommLogData object to folly::dynamic
  folly::dynamic serialized = folly::toDynamic(testData_);

  // Verify that the serialized object contains the expected values
  EXPECT_EQ(serialized["commId"], testData_.commId);
  EXPECT_EQ(serialized["commHash"], testData_.commHash);
  EXPECT_EQ(serialized["commDesc"], testData_.commDesc);
  EXPECT_EQ(serialized["rank"], testData_.rank);
  EXPECT_EQ(serialized["nRanks"], testData_.nRanks);
}

// Test deserialization of folly::dynamic to CommLogData
TEST_F(CommLogDataSerializeTest, DeserializeCommLogData) {
  // Serialize the CommLogData object to folly::dynamic
  folly::dynamic serialized = folly::toDynamic(testData_);

  // Deserialize the folly::dynamic object back to CommLogData
  CommLogData deserialized = folly::convertTo<CommLogData>(serialized);

  // Verify that the deserialized object matches the original
  EXPECT_EQ(deserialized.commId, testData_.commId);
  EXPECT_EQ(deserialized.commHash, testData_.commHash);
  EXPECT_EQ(deserialized.commDesc, testData_.commDesc);
  EXPECT_EQ(deserialized.rank, testData_.rank);
  EXPECT_EQ(deserialized.nRanks, testData_.nRanks);

  // Verify that the equality operator works correctly
  EXPECT_TRUE(deserialized == testData_);

  // Verify that the hash function works correctly
  EXPECT_EQ(deserialized.hash(), testData_.hash());
}

// Test serialization and deserialization with edge cases
TEST_F(CommLogDataSerializeTest, EdgeCases) {
  // Test with empty commDesc
  CommLogData emptyDesc;
  emptyDesc.commId = 1;
  emptyDesc.commHash = 2;
  emptyDesc.commDesc = "";
  emptyDesc.rank = 0;
  emptyDesc.nRanks = 1;

  folly::dynamic serializedEmpty = folly::toDynamic(emptyDesc);
  CommLogData deserializedEmpty =
      folly::convertTo<CommLogData>(serializedEmpty);

  EXPECT_EQ(deserializedEmpty.commId, emptyDesc.commId);
  EXPECT_EQ(deserializedEmpty.commHash, emptyDesc.commHash);
  EXPECT_EQ(deserializedEmpty.commDesc, emptyDesc.commDesc);
  EXPECT_EQ(deserializedEmpty.rank, emptyDesc.rank);
  EXPECT_EQ(deserializedEmpty.nRanks, emptyDesc.nRanks);
  EXPECT_TRUE(deserializedEmpty == emptyDesc);

  // Test with large values
  CommLogData largeValues;
  largeValues.commId = std::numeric_limits<uint64_t>::max();
  largeValues.commHash = std::numeric_limits<uint64_t>::max();
  largeValues.commDesc =
      "very_long_description_with_many_characters_to_test_serialization_of_long_strings";
  largeValues.rank = std::numeric_limits<int>::max();
  largeValues.nRanks = std::numeric_limits<int>::max();

  folly::dynamic serializedLarge = folly::toDynamic(largeValues);
  CommLogData deserializedLarge =
      folly::convertTo<CommLogData>(serializedLarge);

  EXPECT_EQ(deserializedLarge.commId, largeValues.commId);
  EXPECT_EQ(deserializedLarge.commHash, largeValues.commHash);
  EXPECT_EQ(deserializedLarge.commDesc, largeValues.commDesc);
  EXPECT_EQ(deserializedLarge.rank, largeValues.rank);
  EXPECT_EQ(deserializedLarge.nRanks, largeValues.nRanks);
  EXPECT_TRUE(deserializedLarge == largeValues);
}

// Test error handling for malformed dynamic objects
TEST_F(CommLogDataSerializeTest, ErrorHandling) {
  // Create a dynamic object with missing fields
  folly::dynamic missingFields = folly::dynamic::object;
  missingFields["commId"] = 123;
  // Missing commHash
  missingFields["commDesc"] = "test";
  missingFields["rank"] = 1;
  missingFields["nRanks"] = 4;

  // Attempt to deserialize should throw an exception
  EXPECT_THROW(folly::convertTo<CommLogData>(missingFields), std::exception);

  // Create a dynamic object with wrong types
  folly::dynamic wrongTypes = folly::dynamic::object;
  wrongTypes["commId"] = "not_an_integer"; // String instead of integer
  wrongTypes["commHash"] = 456;
  wrongTypes["commDesc"] = "test";
  wrongTypes["rank"] = 1;
  wrongTypes["nRanks"] = 4;

  // Attempt to deserialize should throw an exception
  EXPECT_THROW(folly::convertTo<CommLogData>(wrongTypes), std::exception);
}

// Test round-trip serialization and deserialization with multiple objects
TEST_F(CommLogDataSerializeTest, MultipleObjects) {
  // Create multiple CommLogData objects
  std::vector<CommLogData> originalObjects;

  for (int i = 0; i < 5; i++) {
    CommLogData data;
    data.commId = i * 1000;
    data.commHash = i * 2000;
    data.commDesc = "comm_" + std::to_string(i);
    data.rank = i;
    data.nRanks = 5;
    originalObjects.push_back(data);
  }

  // Serialize all objects
  std::vector<folly::dynamic> serializedObjects;
  for (const auto& obj : originalObjects) {
    serializedObjects.push_back(folly::toDynamic(obj));
  }

  // Deserialize all objects
  std::vector<CommLogData> deserializedObjects;
  for (const auto& serialized : serializedObjects) {
    deserializedObjects.push_back(folly::convertTo<CommLogData>(serialized));
  }

  // Verify that all deserialized objects match the originals
  ASSERT_EQ(deserializedObjects.size(), originalObjects.size());
  for (size_t i = 0; i < originalObjects.size(); i++) {
    EXPECT_TRUE(deserializedObjects[i] == originalObjects[i]);
    EXPECT_EQ(deserializedObjects[i].hash(), originalObjects[i].hash());
  }
}
