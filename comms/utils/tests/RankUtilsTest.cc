// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/utils/RankUtils.h"

#include <gtest/gtest.h>

TEST(RankUtilTest, GetGlobalRankEmpty) {
  unsetenv("RANK");
  ASSERT_FALSE(RankUtils::getGlobalRank().has_value());
}

TEST(RankUtilTest, GetGlobalRankZero) {
  setenv("RANK", "0", 1);
  auto globalRank = RankUtils::getGlobalRank();
  ASSERT_TRUE(globalRank.has_value());
  ASSERT_EQ(0, globalRank.value());
}

TEST(RankUtilTest, GetGlobalRank) {
  setenv("RANK", "10", 1);
  auto globalRank = RankUtils::getGlobalRank();
  ASSERT_TRUE(globalRank.has_value());
  ASSERT_EQ(10, globalRank.value());
}

TEST(RankUtilTest, GetWorldSizeEmpty) {
  unsetenv("WORLD_SIZE");
  ASSERT_FALSE(RankUtils::getWorldSize().has_value());
}

TEST(RankUtilTest, GetWorldSizeRank) {
  setenv("WORLD_SIZE", "10", 1);
  auto worldSize = RankUtils::getWorldSize();
  ASSERT_TRUE(worldSize.has_value());
  ASSERT_EQ(10, worldSize.value());
}

TEST(RankUtilTest, GetInt64FromEnvInvalid) {
  setenv("FOO", "NOT_AN_INT", 1);
  ASSERT_FALSE(RankUtils::getInt64FromEnv("FOO").has_value());
}

TEST(RankUtilTest, GetInt64FromEnvEmpty) {
  unsetenv("FOO");
  ASSERT_FALSE(RankUtils::getInt64FromEnv("FOO").has_value());
}

TEST(RankUtilTest, GetInt64FromEnv) {
  setenv("FOO", "10", 1);
  auto foo = RankUtils::getInt64FromEnv("FOO");
  ASSERT_TRUE(foo.has_value());
  ASSERT_EQ(10, foo.value());
}

TEST(RankUtilTest, GetLocalRankLocalRun) {
  unsetenv("LOCAL_RANK");
  unsetenv("SLURM_JOB_ID");
  ASSERT_FALSE(RankUtils::getLocalRank().has_value());
}

TEST(RankUtilTest, GetLocalRankTorchRun) {
  setenv("LOCAL_RANK", "5", 1);
  unsetenv("SLURM_JOB_ID");
  auto localRank = RankUtils::getLocalRank();
  ASSERT_TRUE(localRank.has_value());
  ASSERT_EQ(5, localRank.value());
}

TEST(RankUtilTest, GetLocalRankTorchRunZero) {
  setenv("LOCAL_RANK", "0", 1);
  unsetenv("SLURM_JOB_ID");
  auto localRank = RankUtils::getLocalRank();
  ASSERT_TRUE(localRank.has_value());
  ASSERT_EQ(0, localRank.value());
}

TEST(RankUtilTest, GetLocalRankTorchRunInvalid) {
  setenv("LOCAL_RANK", "NOT_AN_INT", 1);
  unsetenv("SLURM_JOB_ID");
  ASSERT_FALSE(RankUtils::getLocalRank().has_value());
}

TEST(RankUtilTest, GetLocalRankSlurmJob) {
  unsetenv("LOCAL_RANK");
  setenv("SLURM_JOB_ID", "12345", 1);
  setenv("SLURM_LOCALID", "3", 1);
  auto localRank = RankUtils::getLocalRank();
  ASSERT_TRUE(localRank.has_value());
  ASSERT_EQ(3, localRank.value());
}

TEST(RankUtilTest, GetLocalRankSlurmJobZero) {
  unsetenv("LOCAL_RANK");
  setenv("SLURM_JOB_ID", "12345", 1);
  setenv("SLURM_LOCALID", "0", 1);
  auto localRank = RankUtils::getLocalRank();
  ASSERT_TRUE(localRank.has_value());
  ASSERT_EQ(0, localRank.value());
}

TEST(RankUtilTest, GetLocalRankSlurmJobInvalid) {
  unsetenv("LOCAL_RANK");
  setenv("SLURM_JOB_ID", "12345", 1);
  setenv("SLURM_LOCALID", "NOT_AN_INT", 1);
  ASSERT_FALSE(RankUtils::getLocalRank().has_value());
}

TEST(RankUtilTest, GetLocalRankSlurmJobMissingLocalId) {
  unsetenv("LOCAL_RANK");
  setenv("SLURM_JOB_ID", "12345", 1);
  unsetenv("SLURM_LOCALID");
  ASSERT_FALSE(RankUtils::getLocalRank().has_value());
}

TEST(RankUtilTest, GetLocalRankTorchRunTakesPrecedenceOverSlurm) {
  setenv("LOCAL_RANK", "7", 1);
  setenv("SLURM_JOB_ID", "12345", 1);
  setenv("SLURM_LOCALID", "3", 1);
  auto localRank = RankUtils::getLocalRank();
  ASSERT_TRUE(localRank.has_value());
  ASSERT_EQ(7, localRank.value());
}
