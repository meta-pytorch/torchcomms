// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "comms/ctran/profiler/SamplingRegistry.h"
#include <gtest/gtest.h>

namespace ctran {

TEST(SamplingRegistryTest, NegativeWeightNeverTraces) {
  SamplingRegistry reg(-1);
  reg.setSampleCount(0);
  EXPECT_FALSE(reg.shouldTrace());
  reg.setSampleCount(1);
  EXPECT_FALSE(reg.shouldTrace());
  reg.setSampleCount(100);
  EXPECT_FALSE(reg.shouldTrace());
}

TEST(SamplingRegistryTest, ZeroWeightNeverTraces) {
  SamplingRegistry reg(0);
  reg.setSampleCount(0);
  EXPECT_FALSE(reg.shouldTrace());
  reg.setSampleCount(1);
  EXPECT_FALSE(reg.shouldTrace());
}

TEST(SamplingRegistryTest, WeightOneAlwaysTraces) {
  SamplingRegistry reg(1);
  reg.setSampleCount(0);
  EXPECT_TRUE(reg.shouldTrace());
  reg.setSampleCount(1);
  EXPECT_TRUE(reg.shouldTrace());
  reg.setSampleCount(999);
  EXPECT_TRUE(reg.shouldTrace());
}

TEST(SamplingRegistryTest, WeightNTracesMultiples) {
  SamplingRegistry reg(20);
  reg.setSampleCount(0);
  EXPECT_TRUE(reg.shouldTrace());
  reg.setSampleCount(20);
  EXPECT_TRUE(reg.shouldTrace());
  reg.setSampleCount(100);
  EXPECT_TRUE(reg.shouldTrace());
  reg.setSampleCount(1);
  EXPECT_FALSE(reg.shouldTrace());
  reg.setSampleCount(19);
  EXPECT_FALSE(reg.shouldTrace());
  reg.setSampleCount(101);
  EXPECT_FALSE(reg.shouldTrace());
}

TEST(SamplingRegistryTest, InitialStateFalse) {
  SamplingRegistry reg(1);
  // Before any setSampleCount call, shouldTrace is false
  EXPECT_FALSE(reg.shouldTrace());
}

TEST(SamplingRegistryTest, ShouldTraceReflectsCurrentSampleCount) {
  SamplingRegistry reg(10);
  reg.setSampleCount(10);
  EXPECT_TRUE(reg.shouldTrace());
  EXPECT_TRUE(reg.shouldTrace()); // idempotent
  reg.setSampleCount(11);
  EXPECT_FALSE(reg.shouldTrace());
}

} // namespace ctran
