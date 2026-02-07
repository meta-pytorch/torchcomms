// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include "comm.h"
#include "device.h"
#include "meta/algoconf/InfoExt.h"
#include "meta/algoconf/InfoExtOverride.h"

using ncclx::algoconf::infoExtOverride;
using ncclx::algoconf::ncclInfoExt;

// Test default initialization of ncclInfoExt
TEST(InfoExtTest, DefaultInitialization) {
  ncclInfoExt ext{};

  // Default values should indicate no override
  EXPECT_EQ(ext.algorithm, NCCL_ALGO_UNDEF);
  EXPECT_EQ(ext.protocol, NCCL_PROTO_UNDEF);
  EXPECT_FALSE(ext.opDevSet);
  EXPECT_EQ(ext.nMaxChannels, 0);
  EXPECT_EQ(ext.nWarps, 0);

  // No override set
  EXPECT_FALSE(ext.hasOverride());
  EXPECT_FALSE(ext.isComplete());
}

// Test partial override detection (only some fields set)
TEST(InfoExtTest, PartialOverrideDetection) {
  // Only algorithm set
  {
    ncclInfoExt ext{};
    ext.algorithm = NCCL_ALGO_RING;
    EXPECT_TRUE(ext.hasOverride());
    EXPECT_FALSE(ext.isComplete());
  }

  // Only protocol set
  {
    ncclInfoExt ext{};
    ext.protocol = NCCL_PROTO_SIMPLE;
    EXPECT_TRUE(ext.hasOverride());
    EXPECT_FALSE(ext.isComplete());
  }

  // Only nMaxChannels set
  {
    ncclInfoExt ext{};
    ext.nMaxChannels = 4;
    EXPECT_TRUE(ext.hasOverride());
    EXPECT_FALSE(ext.isComplete());
  }

  // Only nWarps set
  {
    ncclInfoExt ext{};
    ext.nWarps = 8;
    EXPECT_TRUE(ext.hasOverride());
    EXPECT_FALSE(ext.isComplete());
  }

  // Only opDevSet
  {
    ncclInfoExt ext{};
    ext.opDevSet = true;
    EXPECT_TRUE(ext.hasOverride());
    EXPECT_FALSE(ext.isComplete());
  }
}

// Test complete override detection (all required fields set)
TEST(InfoExtTest, CompleteOverrideDetection) {
  ncclInfoExt ext{};
  ext.algorithm = NCCL_ALGO_RING;
  ext.protocol = NCCL_PROTO_SIMPLE;
  ext.nMaxChannels = 4;
  ext.nWarps = 8;

  EXPECT_TRUE(ext.hasOverride());
  EXPECT_TRUE(ext.isComplete());

  // opDevSet is optional - not required for isComplete()
  ext.opDevSet = true;
  EXPECT_TRUE(ext.hasOverride());
  EXPECT_TRUE(ext.isComplete());
}

// Test that aggregate initialization with fewer fields works correctly
// (C++20 designated initializers with default member initializers)
TEST(InfoExtTest, AggregateInitializationDefaults) {
  // Empty brace initialization should use all defaults
  ncclInfoExt ext1{};
  EXPECT_FALSE(ext1.hasOverride());

  // Partial initialization - remaining fields use defaults
  ncclInfoExt ext2{.algorithm = NCCL_ALGO_PAT};
  EXPECT_EQ(ext2.algorithm, NCCL_ALGO_PAT);
  EXPECT_EQ(ext2.protocol, NCCL_PROTO_UNDEF); // default
  EXPECT_FALSE(ext2.opDevSet); // default
  EXPECT_EQ(ext2.nMaxChannels, 0); // default
  EXPECT_EQ(ext2.nWarps, 0); // default
  EXPECT_TRUE(ext2.hasOverride());
  EXPECT_FALSE(ext2.isComplete());
}

// Tests for infoExtOverride function

// Test infoExtOverride rejects grouped collectives
TEST(InfoExtOverrideTest, RejectsGroupedCollectives) {
  ncclInfoExt ext{
      .algorithm = NCCL_ALGO_RING,
      .protocol = NCCL_PROTO_SIMPLE,
      .nMaxChannels = 4,
      .nWarps = 8};

  ncclTaskColl task{};
  task.ext = ext;

  // isGrouped = true should fail
  EXPECT_EQ(infoExtOverride(&task, /*isGrouped=*/true), ncclInvalidUsage);
}

// Test infoExtOverride rejects partial override
TEST(InfoExtOverrideTest, RejectsPartialOverride) {
  // Only algorithm set (incomplete)
  ncclInfoExt ext{.algorithm = NCCL_ALGO_RING};

  ncclTaskColl task{};
  task.ext = ext;

  EXPECT_EQ(infoExtOverride(&task, /*isGrouped=*/false), ncclInvalidUsage);
}

// Test infoExtOverride applies complete override successfully
TEST(InfoExtOverrideTest, AppliesCompleteOverride) {
  ncclInfoExt ext{
      .algorithm = NCCL_ALGO_RING,
      .protocol = NCCL_PROTO_SIMPLE,
      .nMaxChannels = 4,
      .nWarps = 8};

  ncclTaskColl task{};
  task.ext = ext;

  EXPECT_EQ(infoExtOverride(&task, /*isGrouped=*/false), ncclSuccess);

  // Verify fields were copied
  EXPECT_EQ(task.algorithm, NCCL_ALGO_RING);
  EXPECT_EQ(task.protocol, NCCL_PROTO_SIMPLE);
  EXPECT_EQ(task.nMaxChannels, 4);
  EXPECT_EQ(task.nWarps, 8);
}

// Test infoExtOverride applies opDev when opDevSet is true
TEST(InfoExtOverrideTest, AppliesOpDevWhenSet) {
  ncclInfoExt ext{
      .algorithm = NCCL_ALGO_RING,
      .protocol = NCCL_PROTO_SIMPLE,
      .opDevSet = true,
      .nMaxChannels = 4,
      .nWarps = 8};
  ext.opDev.op = ncclDevSum;

  ncclTaskColl task{};
  task.ext = ext;
  task.opDev.op = ncclDevProd; // Different initial value

  EXPECT_EQ(infoExtOverride(&task, /*isGrouped=*/false), ncclSuccess);

  // Verify opDev was copied
  EXPECT_EQ(task.opDev.op, ncclDevSum);
}

// Test infoExtOverride does not modify opDev when opDevSet is false
TEST(InfoExtOverrideTest, PreservesOpDevWhenNotSet) {
  ncclInfoExt ext{
      .algorithm = NCCL_ALGO_RING,
      .protocol = NCCL_PROTO_SIMPLE,
      .opDevSet = false,
      .nMaxChannels = 4,
      .nWarps = 8};

  ncclTaskColl task{};
  task.ext = ext;
  task.opDev.op = ncclDevProd; // Initial value

  EXPECT_EQ(infoExtOverride(&task, /*isGrouped=*/false), ncclSuccess);

  // Verify opDev was NOT modified
  EXPECT_EQ(task.opDev.op, ncclDevProd);
}
