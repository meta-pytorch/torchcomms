// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <folly/dynamic.h>
#include <folly/json.h>

#include "comms/utils/colltrace/CollTraceEvent.h"
#include "comms/utils/colltrace/plugins/CommDumpPlugin.h"
#include "comms/utils/colltrace/tests/MockTypes.h"

using namespace meta::comms;
using namespace meta::comms::colltrace;

// Test fixture for commDumpToMap function tests
class CommDumpToMapTest : public ::testing::Test {
 protected:
  // Helper method to create a CollRecord with a given ID
  std::shared_ptr<CollRecord> createCollRecord(uint64_t collId) {
    auto metadata = std::make_unique<MockCollMetadata>();
    // Metadata will return an empty dynamic object
    EXPECT_CALL(*metadata, toDynamic()).WillRepeatedly(testing::Invoke([]() {
      return folly::dynamic::object();
    }));
    return std::make_shared<CollRecord>(collId, std::move(metadata));
  }
};

// Test commDumpToMap with empty CollTraceDump
TEST_F(CommDumpToMapTest, EmptyDump) {
  CollTraceDump dump;

  auto map = commDumpToMap(dump);

  // Verify the map has the expected keys
  EXPECT_EQ(map.size(), 3);
  EXPECT_TRUE(map.find("CT_pastColls") != map.end());
  EXPECT_TRUE(map.find("CT_pendingColls") != map.end());
  EXPECT_TRUE(map.find("CT_currentColl") != map.end());

  // Verify the values are as expected for an empty dump
  EXPECT_EQ(map["CT_pastColls"], "[]");
  EXPECT_EQ(map["CT_pendingColls"], "[]");
  EXPECT_EQ(map["CT_currentColl"], "null");
}

// Test commDumpToMap with only pastColls
TEST_F(CommDumpToMapTest, WithPastColls) {
  CollTraceDump dump;

  // Add some past collectives
  dump.pastColls.push_back(createCollRecord(1));
  dump.pastColls.push_back(createCollRecord(2));

  auto map = commDumpToMap(dump);

  // Verify the map has the expected keys
  EXPECT_EQ(map.size(), 3);

  // Parse the JSON for pastColls and verify it contains the expected data
  auto pastCollsJson = folly::parseJson(map["CT_pastColls"]);
  EXPECT_EQ(pastCollsJson.size(), 2);
  EXPECT_EQ(pastCollsJson[0]["collId"], 1);
  EXPECT_EQ(pastCollsJson[1]["collId"], 2);

  // Verify other values
  EXPECT_EQ(map["CT_pendingColls"], "[]");
  EXPECT_EQ(map["CT_currentColl"], "null");
}

// Test commDumpToMap with only currentColl
TEST_F(CommDumpToMapTest, WithCurrentColl) {
  CollTraceDump dump;

  // Set current collective
  dump.currentColl = createCollRecord(3);

  auto map = commDumpToMap(dump);

  // Verify the map has the expected keys
  EXPECT_EQ(map.size(), 3);

  // Verify pastColls and pendingColls are empty
  EXPECT_EQ(map["CT_pastColls"], "[]");
  EXPECT_EQ(map["CT_pendingColls"], "[]");

  // Parse the JSON for currentColl and verify it contains the expected data
  auto currentCollJson = folly::parseJson(map["CT_currentColl"]);
  EXPECT_EQ(currentCollJson["collId"], 3);
}

// Test commDumpToMap with only pendingColls
TEST_F(CommDumpToMapTest, WithPendingColls) {
  CollTraceDump dump;

  // Add some pending collectives
  dump.pendingColls.push_back(createCollRecord(4));
  dump.pendingColls.push_back(createCollRecord(5));
  dump.pendingColls.push_back(createCollRecord(6));

  auto map = commDumpToMap(dump);

  // Verify the map has the expected keys
  EXPECT_EQ(map.size(), 3);

  // Verify pastColls is empty and currentColl is null
  EXPECT_EQ(map["CT_pastColls"], "[]");
  EXPECT_EQ(map["CT_currentColl"], "null");

  // Parse the JSON for pendingColls and verify it contains the expected data
  auto pendingCollsJson = folly::parseJson(map["CT_pendingColls"]);
  EXPECT_EQ(pendingCollsJson.size(), 3);
  EXPECT_EQ(pendingCollsJson[0]["collId"], 4);
  EXPECT_EQ(pendingCollsJson[1]["collId"], 5);
  EXPECT_EQ(pendingCollsJson[2]["collId"], 6);
}

// Test commDumpToMap with all fields populated
TEST_F(CommDumpToMapTest, FullDump) {
  CollTraceDump dump;

  // Add past collectives
  dump.pastColls.push_back(createCollRecord(1));
  dump.pastColls.push_back(createCollRecord(2));

  // Set current collective
  dump.currentColl = createCollRecord(3);

  // Add pending collectives
  dump.pendingColls.push_back(createCollRecord(4));
  dump.pendingColls.push_back(createCollRecord(5));

  auto map = commDumpToMap(dump);

  // Verify the map has the expected keys
  EXPECT_EQ(map.size(), 3);

  // Parse the JSON for pastColls and verify it contains the expected data
  auto pastCollsJson = folly::parseJson(map["CT_pastColls"]);
  EXPECT_EQ(pastCollsJson.size(), 2);
  EXPECT_EQ(pastCollsJson[0]["collId"], 1);
  EXPECT_EQ(pastCollsJson[1]["collId"], 2);

  // Parse the JSON for currentColl and verify it contains the expected data
  auto currentCollJson = folly::parseJson(map["CT_currentColl"]);
  EXPECT_EQ(currentCollJson["collId"], 3);

  // Parse the JSON for pendingColls and verify it contains the expected data
  auto pendingCollsJson = folly::parseJson(map["CT_pendingColls"]);
  EXPECT_EQ(pendingCollsJson.size(), 2);
  EXPECT_EQ(pendingCollsJson[0]["collId"], 4);
  EXPECT_EQ(pendingCollsJson[1]["collId"], 5);
}
