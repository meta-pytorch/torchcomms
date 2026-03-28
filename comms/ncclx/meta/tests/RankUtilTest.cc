// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "meta/RankUtil.h"

#include <gtest/gtest.h>

TEST(RankUtilTest, GetGlobalRankEmpty) {
  unsetenv("RANK");
  ASSERT_FALSE(RankUtil::getGlobalRank().has_value());
}

TEST(RankUtilTest, GetGlobalRankZero) {
  setenv("RANK", "0", 1);
  auto globalRank = RankUtil::getGlobalRank();
  ASSERT_TRUE(globalRank.has_value());
  ASSERT_EQ(0, globalRank.value());
}

TEST(RankUtilTest, GetGlobalRank) {
  setenv("RANK", "10", 1);
  auto globalRank = RankUtil::getGlobalRank();
  ASSERT_TRUE(globalRank.has_value());
  ASSERT_EQ(10, globalRank.value());
}

TEST(RankUtilTest, GetWorldSizeEmpty) {
  unsetenv("WORLD_SIZE");
  ASSERT_FALSE(RankUtil::getWorldSize().has_value());
}

TEST(RankUtilTest, GetWorldSizeRank) {
  setenv("WORLD_SIZE", "10", 1);
  auto worldSize = RankUtil::getWorldSize();
  ASSERT_TRUE(worldSize.has_value());
  ASSERT_EQ(10, worldSize.value());
}

TEST(RankUtilTest, GetInt64FromEnvInvalid) {
  setenv("FOO", "NOT_AN_INT", 1);
  ASSERT_FALSE(RankUtil::getInt64FromEnv("FOO").has_value());
}

TEST(RankUtilTest, GetInt64FromEnvEmpty) {
  unsetenv("FOO");
  ASSERT_FALSE(RankUtil::getInt64FromEnv("FOO").has_value());
}

TEST(RankUtilTest, GetInt64FromEnv) {
  setenv("FOO", "10", 1);
  auto foo = RankUtil::getInt64FromEnv("FOO");
  ASSERT_TRUE(foo.has_value());
  ASSERT_EQ(10, foo.value());
}
