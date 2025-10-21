// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "comms/utils/colltrace/CollMetadataImpl.h"
#include "comms/utils/commSpecs.h"

using namespace meta::comms;
using namespace meta::comms::colltrace;

TEST(BaselineMetadata, ConstructorAndProperties) {
  // Create a sample BaselineMetadata for testing
  BaselineMetadata metadata{
      .stream = reinterpret_cast<cudaStream_t>(0x1234),
      .coll = CommFunc::AllReduce,
      .algorithm = CommAlgo::Ring,
      .protocol = CommProtocol::Simple,
      .redOp = commRedOp_t::commSum,
      .root = 0};

  EXPECT_EQ(reinterpret_cast<uintptr_t>(metadata.stream), 0x1234);
  EXPECT_EQ(metadata.coll, CommFunc::AllReduce);
  EXPECT_EQ(metadata.algorithm, CommAlgo::Ring);
  EXPECT_EQ(metadata.protocol, CommProtocol::Simple);
  EXPECT_EQ(metadata.redOp, commRedOp_t::commSum);
  EXPECT_EQ(metadata.root, 0);
}

TEST(BaselineMetadata, EqualityOperators) {
  // Create a sample metadata object
  BaselineMetadata metadata{
      .stream = reinterpret_cast<cudaStream_t>(0x1234),
      .coll = CommFunc::AllReduce,
      .algorithm = CommAlgo::Ring,
      .protocol = CommProtocol::Simple,
      .redOp = commRedOp_t::commSum,
      .root = 0};

  // Create an identical metadata object
  BaselineMetadata identical{
      .stream = reinterpret_cast<cudaStream_t>(0x1234),
      .coll = CommFunc::AllReduce,
      .algorithm = CommAlgo::Ring,
      .protocol = CommProtocol::Simple,
      .redOp = commRedOp_t::commSum,
      .root = 0};

  // Create a different metadata object
  BaselineMetadata different{
      .stream = reinterpret_cast<cudaStream_t>(0x1234),
      .coll = CommFunc::Broadcast, // Different collective operation
      .algorithm = CommAlgo::Ring,
      .protocol = CommProtocol::Simple,
      .redOp = commRedOp_t::commSum,
      .root = 0};

  // Test equality operator
  EXPECT_TRUE(metadata == identical);
  EXPECT_FALSE(metadata == different);

  // Test inequality operator
  EXPECT_FALSE(metadata != identical);
  EXPECT_TRUE(metadata != different);
}

TEST(BaselineMetadata, HashFunction) {
  // Create a sample metadata object
  BaselineMetadata metadata{
      .stream = reinterpret_cast<cudaStream_t>(0x1234),
      .coll = CommFunc::AllReduce,
      .algorithm = CommAlgo::Ring,
      .protocol = CommProtocol::Simple,
      .redOp = commRedOp_t::commSum,
      .root = 0};

  // Create an identical metadata object
  BaselineMetadata identical{
      .stream = reinterpret_cast<cudaStream_t>(0x1234),
      .coll = CommFunc::AllReduce,
      .algorithm = CommAlgo::Ring,
      .protocol = CommProtocol::Simple,
      .redOp = commRedOp_t::commSum,
      .root = 0};

  // Create a different metadata object
  BaselineMetadata different{
      .stream = reinterpret_cast<cudaStream_t>(0x1234),
      .coll = CommFunc::Broadcast, // Different collective operation
      .algorithm = CommAlgo::Ring,
      .protocol = CommProtocol::Simple,
      .redOp = commRedOp_t::commSum,
      .root = 0};

  // Test that identical objects have the same hash
  EXPECT_EQ(metadata.hash(), identical.hash());

  // Test that different objects have different hashes
  EXPECT_NE(metadata.hash(), different.hash());
}

TEST(BaselineMetadata, ToDynamicAndFromDynamic) {
  // Create a sample metadata object
  BaselineMetadata metadata{
      .stream = reinterpret_cast<cudaStream_t>(0x1234),
      .coll = CommFunc::AllReduce,
      .algorithm = CommAlgo::Ring,
      .protocol = CommProtocol::Simple,
      .redOp = commRedOp_t::commSum,
      .root = 0};

  // Convert to dynamic
  folly::dynamic dynamicData = metadata.toDynamic();

  // Verify dynamic data
  EXPECT_EQ(dynamicData["stream"], 0x1234);
  EXPECT_EQ(dynamicData["coll"], "AllReduce");
  EXPECT_EQ(dynamicData["algorithm"], "Ring");
  EXPECT_EQ(dynamicData["protocol"], "Simple");
  EXPECT_EQ(dynamicData["redOp"], "Sum");
  EXPECT_EQ(dynamicData["root"], 0);

  // Convert back from dynamic
  BaselineMetadata reconstructed = BaselineMetadata::fromDynamic(dynamicData);

  // Verify reconstructed metadata
  EXPECT_EQ(
      reinterpret_cast<uintptr_t>(reconstructed.stream),
      reinterpret_cast<uintptr_t>(metadata.stream));
  EXPECT_EQ(reconstructed.coll, metadata.coll);
  EXPECT_EQ(reconstructed.algorithm, metadata.algorithm);
  EXPECT_EQ(reconstructed.protocol, metadata.protocol);
  EXPECT_EQ(reconstructed.redOp, metadata.redOp);
  EXPECT_EQ(reconstructed.root, metadata.root);

  // Verify equality
  EXPECT_EQ(reconstructed, metadata);
}

TEST(CtranMetadata, ConstructorAndProperties) {
  // Create a sample CtranMetadata for testing
  CtranMetadata metadata{.stream = reinterpret_cast<cudaStream_t>(0x5678)};

  EXPECT_EQ(reinterpret_cast<uintptr_t>(metadata.stream), 0x5678);
}

TEST(CtranMetadata, EqualityOperators) {
  // Create a sample metadata object
  CtranMetadata metadata{.stream = reinterpret_cast<cudaStream_t>(0x5678)};

  // Create an identical metadata object
  CtranMetadata identical{.stream = reinterpret_cast<cudaStream_t>(0x5678)};

  // Create a different metadata object (different stream)
  CtranMetadata differentStream{
      .stream = reinterpret_cast<cudaStream_t>(0x9ABC)};

  // Test equality operator
  EXPECT_TRUE(metadata == identical);
  EXPECT_FALSE(metadata == differentStream);

  // Test inequality operator
  EXPECT_FALSE(metadata != identical);
  EXPECT_TRUE(metadata != differentStream);
}

TEST(CtranMetadata, HashFunction) {
  // Create a sample metadata object
  CtranMetadata metadata{.stream = reinterpret_cast<cudaStream_t>(0x5678)};

  // Create an identical metadata object
  CtranMetadata identical{.stream = reinterpret_cast<cudaStream_t>(0x5678)};

  // Create a different metadata object (different stream)
  CtranMetadata differentStream{
      .stream = reinterpret_cast<cudaStream_t>(0x9ABC)};

  // Test that identical objects have the same hash
  EXPECT_EQ(metadata.hash(), identical.hash());

  // Test that different objects have different hashes
  EXPECT_NE(metadata.hash(), differentStream.hash());
}

TEST(CtranMetadata, ToDynamicAndFromDynamic) {
  // Create a sample metadata object
  CtranMetadata metadata{.stream = reinterpret_cast<cudaStream_t>(0x5678)};

  // Convert to dynamic
  folly::dynamic dynamicData = metadata.toDynamic();

  // Verify dynamic data
  EXPECT_EQ(dynamicData["stream"], 0x5678);

  // Convert back from dynamic
  CtranMetadata reconstructed = CtranMetadata::fromDynamic(dynamicData);

  // Verify reconstructed metadata
  EXPECT_EQ(
      reinterpret_cast<uintptr_t>(reconstructed.stream),
      reinterpret_cast<uintptr_t>(metadata.stream));

  // Verify equality
  EXPECT_EQ(reconstructed, metadata);
}

TEST(CollectiveMetadata, ConstructorAndProperties) {
  // Create sample components
  CommLogData commData{
      .commId = 123,
      .commHash = 456,
      .commDesc = "test_comm",
      .rank = 2,
      .nRanks = 8};

  BaselineMetadata baselineData{
      .stream = reinterpret_cast<cudaStream_t>(0x1234),
      .coll = CommFunc::AllReduce,
      .algorithm = CommAlgo::Ring,
      .protocol = CommProtocol::Simple,
      .redOp = commRedOp_t::commSum,
      .root = 0};

  CollectiveMetadata collData{
      .opName = "allreduce",
      .algoName = "ring",
      .sendbuff = 0x5678,
      .recvbuff = 0x9ABC,
      .dataType = commDataType_t::commFloat32,
      .count = 1024};

  // Create CollMetadata
  auto metadata = makeCollMetadata(commData, baselineData, collData);

  // Test hash function
  std::size_t expectedHash = folly::hash::hash_combine(
      folly::hash::hash_combine(
          commData.commId,
          commData.commHash,
          commData.commDesc,
          commData.rank,
          commData.nRanks),
      baselineData.hash(),
      collData.hash());

  EXPECT_EQ(metadata->hash(), expectedHash);

  // Test metadata type
  EXPECT_EQ(metadata->getMetadataType(), "CollectiveMetadata");
}

TEST(GroupedP2PMetadata, ConstructorAndProperties) {
  // Create sample components
  CommLogData commData{
      .commId = 123,
      .commHash = 456,
      .commDesc = "test_comm",
      .rank = 2,
      .nRanks = 8};

  BaselineMetadata baselineData{
      .stream = reinterpret_cast<cudaStream_t>(0x1234),
      .coll = CommFunc::SendRecv,
  };

  GroupedP2PMetaData p2pData{
      .opName = "sendrecv",
      .algoName = "Baseline_sendrecv_S4_R4",
      .dataType = commDataType_t::commInt8,
      .count = 1024};

  // Create CollMetadata
  auto metadata = makeCollMetadata(commData, baselineData, p2pData);

  // Test hash function
  std::size_t expectedHash = folly::hash::hash_combine(
      folly::hash::hash_combine(
          commData.commId,
          commData.commHash,
          commData.commDesc,
          commData.rank,
          commData.nRanks),
      baselineData.hash(),
      p2pData.hash());

  EXPECT_EQ(metadata->hash(), expectedHash);

  // Test metadata type
  EXPECT_EQ(metadata->getMetadataType(), "GroupedP2PMetaData");
}

TEST(CollectiveMetadata, EqualityOperator) {
  // Create sample components
  CommLogData commData{
      .commId = 123,
      .commHash = 456,
      .commDesc = "test_comm",
      .rank = 2,
      .nRanks = 8};

  BaselineMetadata baselineData{
      .stream = reinterpret_cast<cudaStream_t>(0x1234),
      .coll = CommFunc::AllReduce,
      .algorithm = CommAlgo::Ring,
      .protocol = CommProtocol::Simple,
      .redOp = commRedOp_t::commSum,
      .root = 0};

  CollectiveMetadata collData{
      .opName = "allreduce",
      .algoName = "ring",
      .sendbuff = 0x5678,
      .recvbuff = 0x9ABC,
      .dataType = commDataType_t::commFloat32,
      .count = 1024};

  // Create identical BaselineCollMetadata objects
  auto metadata1 = makeCollMetadata(commData, baselineData, collData);
  auto metadata2 = makeCollMetadata(commData, baselineData, collData);

  // Create different BaselineCollMetadata
  CommLogData differentCommData{
      .commId = 789, // Different commId
      .commHash = 456,
      .commDesc = "test_comm",
      .rank = 2,
      .nRanks = 8};
  auto different = makeCollMetadata(differentCommData, baselineData, collData);

  // Test equality
  EXPECT_TRUE(metadata1->equals(*metadata2));
  EXPECT_FALSE(metadata1->equals(*different));
}

TEST(GroupedP2PMetaData, EqualityOperator) {
  // Create sample components
  CommLogData commData{
      .commId = 123,
      .commHash = 456,
      .commDesc = "test_comm",
      .rank = 2,
      .nRanks = 8};

  BaselineMetadata baselineData{
      .stream = reinterpret_cast<cudaStream_t>(0x1234),
      .coll = CommFunc::SendRecv,
  };

  GroupedP2PMetaData p2pData{
      .opName = "sendrecv",
      .algoName = "Baseline_sendrecv_S4_R4",
      .dataType = commDataType_t::commInt8,
      .count = 1024};

  // Create identical BaselineCollMetadata objects
  auto metadata1 = makeCollMetadata(commData, baselineData, p2pData);
  auto metadata2 = makeCollMetadata(commData, baselineData, p2pData);

  // Create different BaselineCollMetadata
  CommLogData differentCommData{
      .commId = 789, // Different commId
      .commHash = 456,
      .commDesc = "test_comm",
      .rank = 2,
      .nRanks = 8};
  auto different = makeCollMetadata(differentCommData, baselineData, p2pData);

  // Test equality
  EXPECT_TRUE(metadata1->equals(*metadata2));
  EXPECT_FALSE(metadata1->equals(*different));
}

TEST(CollectiveMetadata, ToDynamicAndFromDynamic) {
  // Create sample components
  CommLogData commData{
      .commId = 123,
      .commHash = 456,
      .commDesc = "test_comm",
      .rank = 2,
      .nRanks = 8};

  BaselineMetadata baselineData{
      .stream = reinterpret_cast<cudaStream_t>(0x1234),
      .coll = CommFunc::AllReduce,
      .algorithm = CommAlgo::Ring,
      .protocol = CommProtocol::Simple,
      .redOp = commRedOp_t::commSum,
      .root = 0};

  CollectiveMetadata collData{
      .opName = "allreduce",
      .algoName = "ring",
      .sendbuff = 0x5678,
      .recvbuff = 0x9ABC,
      .dataType = commDataType_t::commFloat32,
      .count = 1024};

  // Create BaselineCollMetadata
  auto metadata = makeCollMetadata(commData, baselineData, collData);

  // Convert to dynamic
  folly::dynamic dynamicData = metadata->toDynamic();

  // Verify dynamic data contains expected fields
  EXPECT_EQ(dynamicData["commId"], 123);
  EXPECT_EQ(dynamicData["commHash"], 456);
  EXPECT_EQ(dynamicData["commDesc"], "test_comm");
  EXPECT_EQ(dynamicData["rank"], 2);
  EXPECT_EQ(dynamicData["nRanks"], 8);
  EXPECT_EQ(dynamicData["stream"], 0x1234);
  EXPECT_EQ(dynamicData["coll"], "AllReduce");
  EXPECT_EQ(dynamicData["algorithm"], "Ring");
  EXPECT_EQ(dynamicData["protocol"], "Simple");
  EXPECT_EQ(dynamicData["redOp"], "Sum");
  EXPECT_EQ(dynamicData["MetadataType"], "CollectiveMetadata");

  // Create a new CollMetadata object
  auto reconstructed =
      makeCollMetadata(CommLogData{}, BaselineMetadata{}, CollectiveMetadata{});

  // Populate from dynamic
  reconstructed->fromDynamic(dynamicData);

  // Test equality with original
  EXPECT_TRUE(metadata->equals(*reconstructed));
}

TEST(GroupedP2PMetaData, ToDynamicAndFromDynamic) {
  // Create sample components
  CommLogData commData{
      .commId = 123,
      .commHash = 456,
      .commDesc = "test_comm",
      .rank = 2,
      .nRanks = 8};

  BaselineMetadata baselineData{
      .stream = reinterpret_cast<cudaStream_t>(0x1234),
      .coll = CommFunc::SendRecv,
  };

  GroupedP2PMetaData p2pData{
      .opName = "SendRecv",
      .algoName = "Baseline_SendRecv_S4_R4",
      .ranksInGroupedP2P = std::vector{0, 1, 2, 3},
      .dataType = commDataType_t::commInt8,
      .count = 1024};

  // Create BaselineCollMetadata
  auto metadata = makeCollMetadata(commData, baselineData, p2pData);

  // Convert to dynamic
  folly::dynamic dynamicData = metadata->toDynamic();

  // Verify dynamic data contains expected fields
  EXPECT_EQ(dynamicData["commId"], 123);
  EXPECT_EQ(dynamicData["commHash"], 456);
  EXPECT_EQ(dynamicData["commDesc"], "test_comm");
  EXPECT_EQ(dynamicData["rank"], 2);
  EXPECT_EQ(dynamicData["nRanks"], 8);
  EXPECT_EQ(dynamicData["stream"], 0x1234);
  EXPECT_EQ(dynamicData["coll"], "SendRecv");
  EXPECT_EQ(dynamicData["MetadataType"], "GroupedP2PMetaData");
  EXPECT_EQ(
      dynamicData["ranksInGroupedP2P"], folly::dynamic::array(0, 1, 2, 3));

  // Create a new CollMetadata object
  auto reconstructed =
      makeCollMetadata(CommLogData{}, BaselineMetadata{}, GroupedP2PMetaData{});

  // Populate from dynamic
  reconstructed->fromDynamic(dynamicData);

  // Test equality with original
  EXPECT_TRUE(metadata->equals(*reconstructed));
}
