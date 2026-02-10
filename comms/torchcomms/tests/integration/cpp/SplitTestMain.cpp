// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "SplitTest.hpp"

#include <gtest/gtest.h>
#include <unordered_set>

// Test fixtures
TEST_F(SplitTest, ContigTwoGroup) {
  SCOPED_TRACE(::testing::Message() << "Testing contiguous two-group split");
  testContiguousGroup(2, {});
}

TEST_F(SplitTest, ContigThreeGroup) {
  SCOPED_TRACE(::testing::Message() << "Testing contiguous three-group split");
  testContiguousGroup(3, {});
}

TEST_F(SplitTest, ContigTwoGroupWithSkip) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing two-group split with second group skipped");
  testContiguousGroup(2, {1});
}

TEST_F(SplitTest, ContigThreeGroupWithSkip) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing three-group split with second group skipped");
  testContiguousGroup(3, {1});
}

TEST_F(SplitTest, DuplicateRanks) {
  SCOPED_TRACE(::testing::Message() << "Testing split with duplicate ranks");
  testDuplicateRanks();
}

TEST_F(SplitTest, RankNotInGroup) {
  SCOPED_TRACE(::testing::Message() << "Testing split with rank not in group");
  testRankNotInGroup();
}

TEST_F(SplitTest, NonContigTwoGroup) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing non-contiguous two-group split");
  testNonContiguousGroup(2, {});
}

TEST_F(SplitTest, NonContigThreeGroup) {
  SCOPED_TRACE(
      ::testing::Message() << "Testing non-contiguous three-group split");
  testNonContiguousGroup(3, {});
}

TEST_F(SplitTest, NonContigTwoGroupWithSkip) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing non-contiguous two-group split with second group skipped");
  testNonContiguousGroup(2, {1});
}

TEST_F(SplitTest, NonContigThreeGroupWithSkip) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing non-contiguous three-group split with second group skipped");
  testNonContiguousGroup(3, {1});
}

TEST_F(SplitTest, MultiLevelSplit) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing multi-level split with simultaneous communication");
  testMultiLevel();
}

TEST_F(SplitTest, MultipleSplitsSameRanks) {
  SCOPED_TRACE(
      ::testing::Message()
      << "Testing multiple splits with the same ranks to verify unique store prefixes");
  testMultipleSplitsSameRanks();
}

// This main function is provided by gtest
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
