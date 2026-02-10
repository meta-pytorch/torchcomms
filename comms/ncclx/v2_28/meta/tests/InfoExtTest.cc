// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>

#include <optional>

#include "comm.h"
#include "device.h"
#include "meta/algoconf/InfoExt.h"
#include "meta/algoconf/InfoExtOverride.h"

using ncclx::algoconf::infoExtOverride;
using ncclx::algoconf::ncclInfoExt;

// Test constructor creates a valid ncclInfoExt with all required fields
TEST(InfoExtTest, ConstructorSetsAllFields) {
  ncclInfoExt ext(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, 4, 8);

  EXPECT_EQ(ext.algorithm, NCCL_ALGO_RING);
  EXPECT_EQ(ext.protocol, NCCL_PROTO_SIMPLE);
  EXPECT_EQ(ext.nMaxChannels, 4);
  EXPECT_EQ(ext.nWarps, 8);
  EXPECT_FALSE(ext.opDev.has_value());
}

// Test constructor with opDev parameter
TEST(InfoExtTest, ConstructorWithOpDev) {
  ncclDevRedOpFull opDev{};
  opDev.op = ncclDevSum;

  ncclInfoExt ext(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, 4, 8, opDev);

  EXPECT_EQ(ext.algorithm, NCCL_ALGO_RING);
  EXPECT_EQ(ext.protocol, NCCL_PROTO_SIMPLE);
  EXPECT_EQ(ext.nMaxChannels, 4);
  EXPECT_EQ(ext.nWarps, 8);
  EXPECT_TRUE(ext.opDev.has_value());
  EXPECT_EQ(ext.opDev->op, ncclDevSum);
}

// Tests for infoExtOverride function

// Test infoExtOverride rejects grouped collectives
TEST(InfoExtOverrideTest, RejectsGroupedCollectives) {
  ncclInfoExt ext(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, 4, 8);

  ncclTaskColl task{};
  task.ext = ext;

  // isGrouped = true should fail
  EXPECT_EQ(infoExtOverride(&task, /*isGrouped=*/true), ncclInvalidUsage);
}

// Test infoExtOverride applies override successfully
TEST(InfoExtOverrideTest, AppliesCompleteOverride) {
  ncclInfoExt ext(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, 4, 8);

  ncclTaskColl task{};
  task.ext = ext;

  EXPECT_EQ(infoExtOverride(&task, /*isGrouped=*/false), ncclSuccess);

  // Verify fields were copied
  EXPECT_EQ(task.algorithm, NCCL_ALGO_RING);
  EXPECT_EQ(task.protocol, NCCL_PROTO_SIMPLE);
  EXPECT_EQ(task.nMaxChannels, 4);
  EXPECT_EQ(task.nWarps, 8);
}

// Test infoExtOverride applies opDev when set
TEST(InfoExtOverrideTest, AppliesOpDevWhenSet) {
  ncclDevRedOpFull opDev{};
  opDev.op = ncclDevSum;
  ncclInfoExt ext(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, 4, 8, opDev);

  ncclTaskColl task{};
  task.ext = ext;
  task.opDev.op = ncclDevProd; // Different initial value

  EXPECT_EQ(infoExtOverride(&task, /*isGrouped=*/false), ncclSuccess);

  // Verify opDev was copied
  EXPECT_EQ(task.opDev.op, ncclDevSum);
}

// Test infoExtOverride does not modify opDev when not set
TEST(InfoExtOverrideTest, PreservesOpDevWhenNotSet) {
  ncclInfoExt ext(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, 4, 8);

  ncclTaskColl task{};
  task.ext = ext;
  task.opDev.op = ncclDevProd; // Initial value

  EXPECT_EQ(infoExtOverride(&task, /*isGrouped=*/false), ncclSuccess);

  // Verify opDev was NOT modified
  EXPECT_EQ(task.opDev.op, ncclDevProd);
}
