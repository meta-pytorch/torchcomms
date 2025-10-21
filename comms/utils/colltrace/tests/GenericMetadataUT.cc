// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "comms/utils/colltrace/GenericMetadata.h"
#include "comms/utils/commSpecs.h"

using namespace meta::comms;
using namespace meta::comms::colltrace;

TEST(CollectiveMetadata, ConstructorAndProperties) {
  // Create a sample CollectiveMetadata for testing
  CollectiveMetadata metadata{
      .opName = "allreduce",
      .algoName = "ring",
      .opCount = 42,
      .sendbuff = 0x1234,
      .recvbuff = 0x5678,
      .dataType = commDataType_t::commFloat32,
      .count = 1024};

  EXPECT_EQ(metadata.opName, "allreduce");
  EXPECT_EQ(metadata.algoName, "ring");
  EXPECT_EQ(metadata.opCount, 42);
  EXPECT_EQ(metadata.sendbuff, 0x1234);
  EXPECT_EQ(metadata.recvbuff, 0x5678);
  EXPECT_EQ(metadata.dataType, commDataType_t::commFloat32);
  EXPECT_TRUE(metadata.count.has_value());
  EXPECT_EQ(metadata.count.value(), 1024);
}

TEST(CollectiveMetadata, EqualityOperators) {
  // Create a sample metadata object
  CollectiveMetadata metadata{
      .opName = "allreduce",
      .algoName = "ring",
      .opCount = 42,
      .sendbuff = 0x1234,
      .recvbuff = 0x5678,
      .dataType = commDataType_t::commFloat32,
      .count = 1024};

  // Create an identical metadata object
  CollectiveMetadata identical{
      .opName = "allreduce",
      .algoName = "ring",
      .opCount = 42,
      .sendbuff = 0x1234,
      .recvbuff = 0x5678,
      .dataType = commDataType_t::commFloat32,
      .count = 1024};

  // Create a different metadata object
  CollectiveMetadata different{
      .opName = "broadcast",
      .algoName = "ring",
      .opCount = 42,
      .sendbuff = 0x1234,
      .recvbuff = 0x5678,
      .dataType = commDataType_t::commFloat32,
      .count = 1024};

  // Create a metadata object with different opCount
  CollectiveMetadata differentOpCount{
      .opName = "allreduce",
      .algoName = "ring",
      .opCount = 99,
      .sendbuff = 0x1234,
      .recvbuff = 0x5678,
      .dataType = commDataType_t::commFloat32,
      .count = 1024};

  // Test equality operator
  EXPECT_TRUE(metadata == identical);
  EXPECT_FALSE(metadata == different);
  EXPECT_FALSE(metadata == differentOpCount);

  // Test inequality operator
  EXPECT_FALSE(metadata != identical);
  EXPECT_TRUE(metadata != different);
  EXPECT_TRUE(metadata != differentOpCount);
}

TEST(CollectiveMetadata, HashFunction) {
  // Create a sample metadata object
  CollectiveMetadata metadata{
      .opName = "allreduce",
      .algoName = "ring",
      .opCount = 42,
      .sendbuff = 0x1234,
      .recvbuff = 0x5678,
      .dataType = commDataType_t::commFloat32,
      .count = 1024};

  // Create an identical metadata object
  CollectiveMetadata identical{
      .opName = "allreduce",
      .algoName = "ring",
      .opCount = 42,
      .sendbuff = 0x1234,
      .recvbuff = 0x5678,
      .dataType = commDataType_t::commFloat32,
      .count = 1024};

  // Create a different metadata object
  CollectiveMetadata different{
      .opName = "broadcast",
      .algoName = "ring",
      .opCount = 42,
      .sendbuff = 0x1234,
      .recvbuff = 0x5678,
      .dataType = commDataType_t::commFloat32,
      .count = 1024};

  // Create a metadata object with different opCount
  CollectiveMetadata differentOpCount{
      .opName = "allreduce",
      .algoName = "ring",
      .opCount = 99,
      .sendbuff = 0x1234,
      .recvbuff = 0x5678,
      .dataType = commDataType_t::commFloat32,
      .count = 1024};

  // Test that identical objects have the same hash
  EXPECT_EQ(metadata.hash(), identical.hash());

  // Test that different objects have different hashes
  EXPECT_NE(metadata.hash(), different.hash());
  EXPECT_NE(metadata.hash(), differentOpCount.hash());
}

TEST(CollectiveMetadata, OptionalCount) {
  // Create a sample metadata object with count
  CollectiveMetadata metadata{
      .opName = "allreduce",
      .algoName = "ring",
      .opCount = 42,
      .sendbuff = 0x1234,
      .recvbuff = 0x5678,
      .dataType = commDataType_t::commFloat32,
      .count = 1024};

  // Create metadata without count
  CollectiveMetadata noCount{
      .opName = "allgather",
      .algoName = "ring",
      .opCount = 42,
      .sendbuff = 0x1234,
      .recvbuff = 0x5678,
      .dataType = commDataType_t::commFloat32,
      .count = std::nullopt};

  EXPECT_FALSE(noCount.count.has_value());

  // Test hash function with nullopt count
  std::size_t hashWithCount = metadata.hash();
  std::size_t hashWithoutCount = noCount.hash();
  EXPECT_NE(hashWithCount, hashWithoutCount);
}

TEST(CollectiveMetadata, ToDynamicAndFromDynamic) {
  // Create a sample metadata object
  CollectiveMetadata metadata{
      .opName = "allreduce",
      .algoName = "ring",
      .opCount = 42,
      .sendbuff = 0x1234,
      .recvbuff = 0x5678,
      .dataType = commDataType_t::commFloat32,
      .count = 1024};

  // Convert to dynamic
  folly::dynamic dynamicData = metadata.toDynamic();

  // Verify dynamic data
  EXPECT_EQ(dynamicData["opName"], "allreduce");
  EXPECT_EQ(dynamicData["algoName"], "ring");
  EXPECT_EQ(dynamicData["opCount"], 42);
  EXPECT_EQ(dynamicData["sendbuff"], 0x1234);
  EXPECT_EQ(dynamicData["recvbuff"], 0x5678);
  EXPECT_EQ(dynamicData["dataType"], "commFloat32");
  EXPECT_EQ(dynamicData["count"], 1024);

  // Convert back from dynamic
  CollectiveMetadata reconstructed =
      CollectiveMetadata::fromDynamic(dynamicData);

  // Verify reconstructed metadata
  EXPECT_EQ(reconstructed.opName, metadata.opName);
  EXPECT_EQ(reconstructed.algoName, metadata.algoName);
  EXPECT_EQ(reconstructed.opCount, metadata.opCount);
  EXPECT_EQ(reconstructed.sendbuff, metadata.sendbuff);
  EXPECT_EQ(reconstructed.recvbuff, metadata.recvbuff);
  EXPECT_EQ(reconstructed.dataType, metadata.dataType);
  EXPECT_TRUE(reconstructed.count.has_value());
  EXPECT_EQ(reconstructed.count.value(), metadata.count.value());

  // Verify equality
  EXPECT_EQ(reconstructed, metadata);
}

TEST(CollectiveMetadata, ToDynamicAndFromDynamicWithoutCount) {
  // Create metadata without count
  CollectiveMetadata noCount{
      .opName = "allgather",
      .algoName = "ring",
      .opCount = 42,
      .sendbuff = 0x1234,
      .recvbuff = 0x5678,
      .dataType = commDataType_t::commFloat32,
      .count = std::nullopt};

  // Convert to dynamic
  folly::dynamic dynamicData = noCount.toDynamic();

  // Verify dynamic data
  EXPECT_EQ(dynamicData["opName"], "allgather");
  EXPECT_EQ(dynamicData["algoName"], "ring");
  EXPECT_EQ(dynamicData["opCount"], 42);
  EXPECT_EQ(dynamicData["sendbuff"], 0x1234);
  EXPECT_EQ(dynamicData["recvbuff"], 0x5678);
  EXPECT_EQ(dynamicData["dataType"], "commFloat32");
  EXPECT_TRUE(dynamicData["count"].isNull());

  // Convert back from dynamic
  CollectiveMetadata reconstructed =
      CollectiveMetadata::fromDynamic(dynamicData);

  // Verify reconstructed metadata
  EXPECT_EQ(reconstructed.opName, noCount.opName);
  EXPECT_EQ(reconstructed.algoName, noCount.algoName);
  EXPECT_EQ(reconstructed.opCount, noCount.opCount);
  EXPECT_EQ(reconstructed.sendbuff, noCount.sendbuff);
  EXPECT_EQ(reconstructed.recvbuff, noCount.recvbuff);
  EXPECT_EQ(reconstructed.dataType, noCount.dataType);
  EXPECT_FALSE(reconstructed.count.has_value());

  // Verify equality
  EXPECT_EQ(reconstructed, noCount);
}

TEST(GroupedP2PMetaData, ConstructorAndProperties) {
  // Create a sample GroupedP2PMetaData for testing
  GroupedP2PMetaData metadata{
      .opName = "send_recv",
      .algoName = "p2p",
      .opCount = 123,
      .ranksInGroupedP2P = std::vector<int>{0, 1, 2, 3},
      .dataType = commDataType_t::commInt8,
      .count = 2048};

  EXPECT_EQ(metadata.opName, "send_recv");
  EXPECT_EQ(metadata.algoName, "p2p");
  EXPECT_EQ(metadata.opCount, 123);
  EXPECT_FALSE(metadata.ranksInGroupedP2P.empty());
  EXPECT_EQ(metadata.ranksInGroupedP2P, std::vector<int>({0, 1, 2, 3}));
  EXPECT_EQ(metadata.dataType, commDataType_t::commInt8);
  EXPECT_EQ(metadata.count, 2048);
}

TEST(GroupedP2PMetaData, EqualityOperators) {
  // Create a sample metadata object
  GroupedP2PMetaData metadata{
      .opName = "send_recv",
      .algoName = "p2p",
      .opCount = 123,
      .ranksInGroupedP2P = std::vector<int>{0, 1, 2, 3},
      .dataType = commDataType_t::commInt8,
      .count = 2048};

  // Create an identical metadata object
  GroupedP2PMetaData identical{
      .opName = "send_recv",
      .algoName = "p2p",
      .opCount = 123,
      .ranksInGroupedP2P = std::vector<int>{0, 1, 2, 3},
      .dataType = commDataType_t::commInt8,
      .count = 2048};

  // Create a different metadata object
  GroupedP2PMetaData different{
      .opName = "send_recv",
      .algoName = "p2p",
      .opCount = 123,
      .ranksInGroupedP2P = std::vector<int>{0, 1, 2, 4}, // Different rank
      .dataType = commDataType_t::commInt8,
      .count = 2048};

  // Create a metadata object with different opCount
  GroupedP2PMetaData differentOpCount{
      .opName = "send_recv",
      .algoName = "p2p",
      .opCount = 456,
      .ranksInGroupedP2P = std::vector<int>{0, 1, 2, 3},
      .dataType = commDataType_t::commInt8,
      .count = 2048};

  // Test equality operator
  EXPECT_TRUE(metadata == identical);
  EXPECT_FALSE(metadata == different);
  EXPECT_FALSE(metadata == differentOpCount);

  // Test inequality operator
  EXPECT_FALSE(metadata != identical);
  EXPECT_TRUE(metadata != different);
  EXPECT_TRUE(metadata != differentOpCount);
}

TEST(GroupedP2PMetaData, HashFunction) {
  // Create a sample metadata object
  GroupedP2PMetaData metadata{
      .opName = "send_recv",
      .algoName = "p2p",
      .opCount = 123,
      .ranksInGroupedP2P = std::vector<int>{0, 1, 2, 3},
      .dataType = commDataType_t::commInt8,
      .count = 2048};

  // Create an identical metadata object
  GroupedP2PMetaData identical{
      .opName = "send_recv",
      .algoName = "p2p",
      .opCount = 123,
      .ranksInGroupedP2P = std::vector<int>{0, 1, 2, 3},
      .dataType = commDataType_t::commInt8,
      .count = 2048};

  // Create a different metadata object
  GroupedP2PMetaData different{
      .opName = "send_recv",
      .algoName = "p2p",
      .opCount = 123,
      .ranksInGroupedP2P = std::vector<int>{0, 1, 2, 4}, // Different rank
      .dataType = commDataType_t::commInt8,
      .count = 2048};

  // Create a metadata object with different opCount
  GroupedP2PMetaData differentOpCount{
      .opName = "send_recv",
      .algoName = "p2p",
      .opCount = 456,
      .ranksInGroupedP2P = std::vector<int>{0, 1, 2, 3},
      .dataType = commDataType_t::commInt8,
      .count = 2048};

  // Test that identical objects have the same hash
  EXPECT_EQ(metadata.hash(), identical.hash());

  // Test that different objects have different hashes
  EXPECT_NE(metadata.hash(), different.hash());
  EXPECT_NE(metadata.hash(), differentOpCount.hash());
}

TEST(GroupedP2PMetaData, OptionalRanks) {
  // Create a sample metadata object with ranks
  GroupedP2PMetaData metadata{
      .opName = "send_recv",
      .algoName = "p2p",
      .opCount = 123,
      .ranksInGroupedP2P = std::vector<int>{0, 1, 2, 3},
      .dataType = commDataType_t::commInt8,
      .count = 2048};

  // Create metadata without ranks
  GroupedP2PMetaData noRanks{
      .opName = "send_recv",
      .algoName = "p2p",
      .opCount = 123,
      .dataType = commDataType_t::commInt8,
      .count = 2048};

  EXPECT_TRUE(noRanks.ranksInGroupedP2P.empty());

  // Test hash function with nullopt ranks
  std::size_t hashWithRanks = metadata.hash();
  std::size_t hashWithoutRanks = noRanks.hash();
  EXPECT_NE(hashWithRanks, hashWithoutRanks);
}

TEST(GroupedP2PMetaData, ToDynamicAndFromDynamic) {
  // Create a sample metadata object
  GroupedP2PMetaData metadata{
      .opName = "send_recv",
      .algoName = "p2p",
      .opCount = 123,
      .ranksInGroupedP2P = std::vector<int>{0, 1, 2, 3},
      .dataType = commDataType_t::commInt8,
      .count = 2048};

  // Convert to dynamic
  folly::dynamic dynamicData = metadata.toDynamic();

  // Verify dynamic data
  EXPECT_EQ(dynamicData["opName"], "send_recv");
  EXPECT_EQ(dynamicData["algoName"], "p2p");
  EXPECT_EQ(dynamicData["opCount"], 123);
  EXPECT_EQ(dynamicData["dataType"], "commInt8");
  EXPECT_EQ(dynamicData["count"], 2048);

  // Check ranks array
  EXPECT_TRUE(dynamicData["ranksInGroupedP2P"].isArray());
  EXPECT_EQ(dynamicData["ranksInGroupedP2P"].size(), 4);
  EXPECT_EQ(dynamicData["ranksInGroupedP2P"][0], 0);
  EXPECT_EQ(dynamicData["ranksInGroupedP2P"][1], 1);
  EXPECT_EQ(dynamicData["ranksInGroupedP2P"][2], 2);
  EXPECT_EQ(dynamicData["ranksInGroupedP2P"][3], 3);

  // Convert back from dynamic
  GroupedP2PMetaData reconstructed =
      GroupedP2PMetaData::fromDynamic(dynamicData);

  // Verify reconstructed metadata
  EXPECT_EQ(reconstructed.opName, metadata.opName);
  EXPECT_EQ(reconstructed.algoName, metadata.algoName);
  EXPECT_EQ(reconstructed.opCount, metadata.opCount);
  EXPECT_EQ(reconstructed.dataType, metadata.dataType);
  EXPECT_EQ(reconstructed.count, metadata.count);
  EXPECT_FALSE(reconstructed.ranksInGroupedP2P.empty());
  EXPECT_EQ(reconstructed.ranksInGroupedP2P, metadata.ranksInGroupedP2P);

  // Verify equality
  EXPECT_EQ(reconstructed, metadata);
}

TEST(GroupedP2PMetaData, ToDynamicAndFromDynamicWithoutRanks) {
  // Create metadata without ranks
  GroupedP2PMetaData noRanks{
      .opName = "send_recv",
      .algoName = "p2p",
      .opCount = 123,
      .dataType = commDataType_t::commInt8,
      .count = 2048};

  // Convert to dynamic
  folly::dynamic dynamicData = noRanks.toDynamic();

  // Verify dynamic data
  EXPECT_EQ(dynamicData["opName"], "send_recv");
  EXPECT_EQ(dynamicData["algoName"], "p2p");
  EXPECT_EQ(dynamicData["opCount"], 123);
  EXPECT_EQ(dynamicData["dataType"], "commInt8");
  EXPECT_EQ(dynamicData["count"], 2048);
  EXPECT_TRUE(dynamicData["ranksInGroupedP2P"].empty());

  // Convert back from dynamic
  GroupedP2PMetaData reconstructed =
      GroupedP2PMetaData::fromDynamic(dynamicData);

  // Verify reconstructed metadata
  EXPECT_EQ(reconstructed.opName, noRanks.opName);
  EXPECT_EQ(reconstructed.algoName, noRanks.algoName);
  EXPECT_EQ(reconstructed.opCount, noRanks.opCount);
  EXPECT_EQ(reconstructed.dataType, noRanks.dataType);
  EXPECT_EQ(reconstructed.count, noRanks.count);
  EXPECT_TRUE(reconstructed.ranksInGroupedP2P.empty());

  // Verify equality
  EXPECT_EQ(reconstructed, noRanks);
}

TEST(GroupedCollP2PMetaData, ConstructorAndProperties) {
  // Create sample CollectiveMetadata objects
  CollectiveMetadata coll1{
      .opName = "allreduce",
      .algoName = "ring",
      .opCount = 42,
      .sendbuff = 0x1234,
      .recvbuff = 0x5678,
      .dataType = commDataType_t::commFloat32,
      .count = 1024};

  CollectiveMetadata coll2{
      .opName = "broadcast",
      .algoName = "tree",
      .opCount = 43,
      .sendbuff = 0x9abc,
      .recvbuff = 0xdef0,
      .dataType = commDataType_t::commInt32,
      .count = 512};

  // Create sample GroupedP2PMetaData
  GroupedP2PMetaData p2p{
      .opName = "send_recv",
      .algoName = "p2p",
      .opCount = 123,
      .ranksInGroupedP2P = std::vector<int>{0, 1, 2, 3},
      .dataType = commDataType_t::commInt8,
      .count = 2048};

  // Create GroupedCollP2PMetaData
  GroupedCollP2PMetaData metadata{
      .colls = std::vector<CollectiveMetadata>{coll1, coll2}, .p2p = p2p};

  EXPECT_EQ(metadata.colls.size(), 2);
  EXPECT_EQ(metadata.colls[0], coll1);
  EXPECT_EQ(metadata.colls[1], coll2);
  EXPECT_EQ(metadata.p2p, p2p);
}

TEST(GroupedCollP2PMetaData, EmptyConstructor) {
  GroupedCollP2PMetaData metadata;

  EXPECT_TRUE(metadata.colls.empty());
  // Default constructed GroupedCollP2PMetaData should not have p2p data
  EXPECT_FALSE(metadata.p2p.has_value());
}

TEST(GroupedCollP2PMetaData, EqualityOperators) {
  // Create sample CollectiveMetadata objects
  CollectiveMetadata coll1{
      .opName = "allreduce",
      .algoName = "ring",
      .opCount = 42,
      .sendbuff = 0x1234,
      .recvbuff = 0x5678,
      .dataType = commDataType_t::commFloat32,
      .count = 1024};

  CollectiveMetadata coll2{
      .opName = "broadcast",
      .algoName = "tree",
      .opCount = 43,
      .sendbuff = 0x9abc,
      .recvbuff = 0xdef0,
      .dataType = commDataType_t::commInt32,
      .count = 512};

  // Create sample GroupedP2PMetaData
  GroupedP2PMetaData p2p{
      .opName = "send_recv",
      .algoName = "p2p",
      .opCount = 123,
      .ranksInGroupedP2P = std::vector<int>{0, 1, 2, 3},
      .dataType = commDataType_t::commInt8,
      .count = 2048};

  // Create identical GroupedCollP2PMetaData objects
  GroupedCollP2PMetaData metadata{
      .colls = std::vector<CollectiveMetadata>{coll1, coll2}, .p2p = p2p};

  GroupedCollP2PMetaData identical{
      .colls = std::vector<CollectiveMetadata>{coll1, coll2}, .p2p = p2p};

  // Create different GroupedCollP2PMetaData with different colls
  GroupedCollP2PMetaData differentColls{
      .colls = std::vector<CollectiveMetadata>{coll1}, // Only one collective
      .p2p = p2p};

  // Create different GroupedCollP2PMetaData with different p2p
  GroupedP2PMetaData differentP2P{
      .opName = "send_recv",
      .algoName = "p2p",
      .opCount = 456, // Different opCount
      .ranksInGroupedP2P = std::vector<int>{0, 1, 2, 3},
      .dataType = commDataType_t::commInt8,
      .count = 2048};

  GroupedCollP2PMetaData differentP2PMetadata{
      .colls = std::vector<CollectiveMetadata>{coll1, coll2},
      .p2p = differentP2P};

  // Test equality operator
  EXPECT_TRUE(metadata == identical);
  EXPECT_FALSE(metadata == differentColls);
  EXPECT_FALSE(metadata == differentP2PMetadata);

  // Test inequality operator
  EXPECT_FALSE(metadata != identical);
  EXPECT_TRUE(metadata != differentColls);
  EXPECT_TRUE(metadata != differentP2PMetadata);
}

TEST(GroupedCollP2PMetaData, HashFunction) {
  // Create sample CollectiveMetadata objects
  CollectiveMetadata coll1{
      .opName = "allreduce",
      .algoName = "ring",
      .opCount = 42,
      .sendbuff = 0x1234,
      .recvbuff = 0x5678,
      .dataType = commDataType_t::commFloat32,
      .count = 1024};

  CollectiveMetadata coll2{
      .opName = "broadcast",
      .algoName = "tree",
      .opCount = 43,
      .sendbuff = 0x9abc,
      .recvbuff = 0xdef0,
      .dataType = commDataType_t::commInt32,
      .count = 512};

  // Create sample GroupedP2PMetaData
  GroupedP2PMetaData p2p{
      .opName = "send_recv",
      .algoName = "p2p",
      .opCount = 123,
      .ranksInGroupedP2P = std::vector<int>{0, 1, 2, 3},
      .dataType = commDataType_t::commInt8,
      .count = 2048};

  // Create identical GroupedCollP2PMetaData objects
  GroupedCollP2PMetaData metadata{
      .colls = std::vector<CollectiveMetadata>{coll1, coll2}, .p2p = p2p};

  GroupedCollP2PMetaData identical{
      .colls = std::vector<CollectiveMetadata>{coll1, coll2}, .p2p = p2p};

  // Create different GroupedCollP2PMetaData with different colls
  GroupedCollP2PMetaData differentColls{
      .colls = std::vector<CollectiveMetadata>{coll1}, // Only one collective
      .p2p = p2p};

  // Create different GroupedCollP2PMetaData with different p2p
  GroupedP2PMetaData differentP2P{
      .opName = "send_recv",
      .algoName = "p2p",
      .opCount = 456, // Different opCount
      .ranksInGroupedP2P = std::vector<int>{0, 1, 2, 3},
      .dataType = commDataType_t::commInt8,
      .count = 2048};

  GroupedCollP2PMetaData differentP2PMetadata{
      .colls = std::vector<CollectiveMetadata>{coll1, coll2},
      .p2p = differentP2P};

  // Test that identical objects have the same hash
  EXPECT_EQ(metadata.hash(), identical.hash());

  // Test that different objects have different hashes
  EXPECT_NE(metadata.hash(), differentColls.hash());
  EXPECT_NE(metadata.hash(), differentP2PMetadata.hash());
}

TEST(GroupedCollP2PMetaData, HashWithEmptyColls) {
  // Create GroupedP2PMetaData
  GroupedP2PMetaData p2p{
      .opName = "send_recv",
      .algoName = "p2p",
      .opCount = 123,
      .ranksInGroupedP2P = std::vector<int>{0, 1, 2, 3},
      .dataType = commDataType_t::commInt8,
      .count = 2048};

  // Create GroupedCollP2PMetaData with empty colls
  GroupedCollP2PMetaData emptyColls{
      .colls = std::vector<CollectiveMetadata>{}, .p2p = p2p};

  // Create GroupedCollP2PMetaData with one collective
  CollectiveMetadata coll{
      .opName = "allreduce",
      .algoName = "ring",
      .opCount = 42,
      .sendbuff = 0x1234,
      .recvbuff = 0x5678,
      .dataType = commDataType_t::commFloat32,
      .count = 1024};

  GroupedCollP2PMetaData withColls{
      .colls = std::vector<CollectiveMetadata>{coll}, .p2p = p2p};

  // Test that different number of collectives results in different hashes
  EXPECT_NE(emptyColls.hash(), withColls.hash());

  // Test that empty colls objects are still handled correctly
  GroupedCollP2PMetaData anotherEmptyColls{
      .colls = std::vector<CollectiveMetadata>{}, .p2p = p2p};

  EXPECT_EQ(emptyColls.hash(), anotherEmptyColls.hash());
}

TEST(GroupedCollP2PMetaData, ToDynamicWithColls) {
  // Create sample CollectiveMetadata
  CollectiveMetadata coll{
      .opName = "allreduce",
      .algoName = "ring",
      .opCount = 42,
      .sendbuff = 0x1234,
      .recvbuff = 0x5678,
      .dataType = commDataType_t::commFloat32,
      .count = 1024};

  // Create sample GroupedP2PMetaData
  GroupedP2PMetaData p2p{
      .opName = "send_recv",
      .algoName = "p2p",
      .opCount = 123,
      .ranksInGroupedP2P = std::vector<int>{0, 1, 2, 3},
      .dataType = commDataType_t::commInt8,
      .count = 2048};

  GroupedCollP2PMetaData metadata{
      .colls = std::vector<CollectiveMetadata>{coll}, .p2p = p2p};

  // Convert to dynamic - should return the first collective's dynamic
  // representation
  folly::dynamic dynamicData = metadata.toDynamic();

  // Verify it matches the first collective's toDynamic output
  folly::dynamic expectedDynamic = coll.toDynamic();
  EXPECT_EQ(dynamicData, expectedDynamic);
}

TEST(GroupedCollP2PMetaData, ToDynamicWithEmptyColls) {
  // Create sample GroupedP2PMetaData
  GroupedP2PMetaData p2p{
      .opName = "send_recv",
      .algoName = "p2p",
      .opCount = 123,
      .ranksInGroupedP2P = std::vector<int>{0, 1, 2, 3},
      .dataType = commDataType_t::commInt8,
      .count = 2048};

  GroupedCollP2PMetaData metadata{
      .colls = std::vector<CollectiveMetadata>{}, .p2p = p2p};

  // Convert to dynamic - should return the p2p's dynamic representation
  folly::dynamic dynamicData = metadata.toDynamic();

  // Verify it matches the p2p's toDynamic output
  folly::dynamic expectedDynamic = p2p.toDynamic();
  EXPECT_EQ(dynamicData, expectedDynamic);
}

TEST(GroupedCollP2PMetaData, FromDynamicNotSupported) {
  // Create a sample dynamic object
  folly::dynamic dynamicData = folly::dynamic::object();
  dynamicData["opName"] = "allreduce";
  dynamicData["algoName"] = "ring";
  dynamicData["opCount"] = 42;

  // fromDynamic should return a default-constructed object and log an error
  GroupedCollP2PMetaData reconstructed =
      GroupedCollP2PMetaData::fromDynamic(dynamicData);

  // The function should return a default-constructed object
  GroupedCollP2PMetaData expected;
  EXPECT_EQ(reconstructed, expected);
  EXPECT_TRUE(reconstructed.colls.empty());
  EXPECT_FALSE(reconstructed.p2p.has_value());
}

TEST(GroupedCollP2PMetaData, OptionalP2PWithValue) {
  // Test GroupedCollP2PMetaData with p2p field having a value
  CollectiveMetadata coll{
      .opName = "allreduce",
      .algoName = "ring",
      .opCount = 42,
      .sendbuff = 0x1234,
      .recvbuff = 0x5678,
      .dataType = commDataType_t::commFloat32,
      .count = 1024};

  GroupedP2PMetaData p2p{
      .opName = "send_recv",
      .algoName = "p2p",
      .opCount = 123,
      .ranksInGroupedP2P = std::vector<int>{0, 1, 2, 3},
      .dataType = commDataType_t::commInt8,
      .count = 2048};

  GroupedCollP2PMetaData metadataWithP2P{
      .colls = std::vector<CollectiveMetadata>{coll}, .p2p = p2p};

  // Test that p2p has a value
  EXPECT_TRUE(metadataWithP2P.p2p.has_value());
  EXPECT_EQ(metadataWithP2P.p2p->opName, "send_recv");
  EXPECT_EQ(metadataWithP2P.p2p->ranksInGroupedP2P.size(), 4);

  // Test hash function works correctly with p2p value
  std::size_t hash1 = metadataWithP2P.hash();
  EXPECT_NE(hash1, 0); // Should have a non-zero hash

  // Test equality with identical structure
  GroupedCollP2PMetaData identicalWithP2P{
      .colls = std::vector<CollectiveMetadata>{coll}, .p2p = p2p};
  EXPECT_EQ(metadataWithP2P, identicalWithP2P);
  EXPECT_EQ(metadataWithP2P.hash(), identicalWithP2P.hash());
}

TEST(GroupedCollP2PMetaData, OptionalP2PWithoutValue) {
  // Test GroupedCollP2PMetaData without p2p field (nullopt)
  CollectiveMetadata coll{
      .opName = "allreduce",
      .algoName = "ring",
      .opCount = 42,
      .sendbuff = 0x1234,
      .recvbuff = 0x5678,
      .dataType = commDataType_t::commFloat32,
      .count = 1024};

  GroupedCollP2PMetaData metadataWithoutP2P{
      .colls = std::vector<CollectiveMetadata>{coll}, .p2p = std::nullopt};

  // Test that p2p does not have a value
  EXPECT_FALSE(metadataWithoutP2P.p2p.has_value());

  // Test hash function works correctly without p2p value
  std::size_t hash1 = metadataWithoutP2P.hash();
  EXPECT_NE(hash1, 0); // Should still have a non-zero hash from colls

  // Test equality with identical structure
  GroupedCollP2PMetaData identicalWithoutP2P{
      .colls = std::vector<CollectiveMetadata>{coll}, .p2p = std::nullopt};
  EXPECT_EQ(metadataWithoutP2P, identicalWithoutP2P);
  EXPECT_EQ(metadataWithoutP2P.hash(), identicalWithoutP2P.hash());
}

TEST(GroupedCollP2PMetaData, OptionalP2PComparisons) {
  // Test comparisons between objects with and without p2p
  CollectiveMetadata coll{
      .opName = "allreduce",
      .algoName = "ring",
      .opCount = 42,
      .sendbuff = 0x1234,
      .recvbuff = 0x5678,
      .dataType = commDataType_t::commFloat32,
      .count = 1024};

  GroupedP2PMetaData p2p{
      .opName = "send_recv",
      .algoName = "p2p",
      .opCount = 123,
      .ranksInGroupedP2P = std::vector<int>{0, 1, 2, 3},
      .dataType = commDataType_t::commInt8,
      .count = 2048};

  GroupedCollP2PMetaData metadataWithP2P{
      .colls = std::vector<CollectiveMetadata>{coll}, .p2p = p2p};

  GroupedCollP2PMetaData metadataWithoutP2P{
      .colls = std::vector<CollectiveMetadata>{coll}, .p2p = std::nullopt};

  // Test inequality between objects with and without p2p
  EXPECT_NE(metadataWithP2P, metadataWithoutP2P);
  EXPECT_NE(metadataWithP2P.hash(), metadataWithoutP2P.hash());
}

TEST(GroupedCollP2PMetaData, ToDynamicWithoutP2PAndEmptyColls) {
  // Test toDynamic when both colls is empty and p2p is nullopt
  GroupedCollP2PMetaData emptyMetadata{
      .colls = std::vector<CollectiveMetadata>{}, .p2p = std::nullopt};

  // Convert to dynamic - should return empty object
  folly::dynamic dynamicData = emptyMetadata.toDynamic();

  // Verify it returns an empty object
  EXPECT_TRUE(dynamicData.isObject());
  EXPECT_EQ(dynamicData.size(), 0);
}

TEST(GroupedCollP2PMetaData, HashFunctionWithOptionalP2P) {
  // Test that hash function behaves correctly with optional p2p
  CollectiveMetadata coll{
      .opName = "allreduce",
      .algoName = "ring",
      .opCount = 42,
      .sendbuff = 0x1234,
      .recvbuff = 0x5678,
      .dataType = commDataType_t::commFloat32,
      .count = 1024};

  GroupedP2PMetaData p2p{
      .opName = "send_recv",
      .algoName = "p2p",
      .opCount = 123,
      .ranksInGroupedP2P = std::vector<int>{0, 1, 2, 3},
      .dataType = commDataType_t::commInt8,
      .count = 2048};

  // Create objects with same colls but different p2p states
  GroupedCollP2PMetaData withP2P{
      .colls = std::vector<CollectiveMetadata>{coll}, .p2p = p2p};

  GroupedCollP2PMetaData withoutP2P{
      .colls = std::vector<CollectiveMetadata>{coll}, .p2p = std::nullopt};

  GroupedCollP2PMetaData emptyBoth{
      .colls = std::vector<CollectiveMetadata>{}, .p2p = std::nullopt};

  // Hash values should be different
  EXPECT_NE(withP2P.hash(), withoutP2P.hash());
  EXPECT_NE(withP2P.hash(), emptyBoth.hash());
  EXPECT_NE(withoutP2P.hash(), emptyBoth.hash());

  // Hash should be consistent for same object
  EXPECT_EQ(withP2P.hash(), withP2P.hash());
  EXPECT_EQ(withoutP2P.hash(), withoutP2P.hash());
  EXPECT_EQ(emptyBoth.hash(), emptyBoth.hash());
}

TEST(GroupedCollP2PMetaData, ComplexScenario) {
  // Test with multiple collectives and complex p2p data
  CollectiveMetadata coll1{
      .opName = "allreduce",
      .algoName = "ring",
      .opCount = 42,
      .sendbuff = 0x1234,
      .recvbuff = 0x5678,
      .dataType = commDataType_t::commFloat32,
      .count = 1024};

  CollectiveMetadata coll2{
      .opName = "broadcast",
      .algoName = "tree",
      .opCount = 43,
      .sendbuff = 0x9abc,
      .recvbuff = 0xdef0,
      .dataType = commDataType_t::commInt32,
      .count = 512};

  CollectiveMetadata coll3{
      .opName = "allgather",
      .algoName = "ring",
      .opCount = 44,
      .sendbuff = 0xabcd,
      .recvbuff = 0xef01,
      .dataType = commDataType_t::commFloat64,
      .count = std::nullopt}; // Test with nullopt count

  GroupedP2PMetaData p2p{
      .opName = "send_recv",
      .algoName = "p2p",
      .opCount = 123,
      .ranksInGroupedP2P = std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7},
      .dataType = commDataType_t::commInt8,
      .count = 4096};

  GroupedCollP2PMetaData metadata{
      .colls = std::vector<CollectiveMetadata>{coll1, coll2, coll3},
      .p2p = p2p};

  // Test properties
  EXPECT_EQ(metadata.colls.size(), 3);
  EXPECT_TRUE(metadata.p2p.has_value());
  EXPECT_EQ(metadata.p2p->ranksInGroupedP2P.size(), 8);

  // Test hash consistency
  GroupedCollP2PMetaData identical{
      .colls = std::vector<CollectiveMetadata>{coll1, coll2, coll3},
      .p2p = p2p};
  EXPECT_EQ(metadata.hash(), identical.hash());

  // Test toDynamic returns first collective
  folly::dynamic dynamicData = metadata.toDynamic();
  folly::dynamic expectedDynamic = coll1.toDynamic();
  EXPECT_EQ(dynamicData, expectedDynamic);
}
