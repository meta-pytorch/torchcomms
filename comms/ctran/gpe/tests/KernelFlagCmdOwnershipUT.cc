// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gtest/gtest.h>
#include "comms/ctran/algos/AllToAll/AllToAllPImpl.h"
#include "comms/ctran/algos/AllToAll/AllToAllvDynamicPImpl.h"
#include "comms/ctran/algos/common/GpeKernel.h"
#include "comms/ctran/gpe/CtranGpeImpl.h"

class KernelFlagCmdOwnershipTest : public ::testing::Test {
 protected:
  void SetUp() override {
    EXPECT_EQ(cudaSetDevice(0), cudaSuccess);
  }
};

// Verifies the full lifecycle: a persistent cmd (graph mode) holds a
// KernelFlagItem via setPersistent(), preventing pool reclaim even after
// flags are UNSET. When the cmd is destroyed (~CtranGpeCmd calls
// clearPersistent() and reset()), the flag is released back to the pool.
TEST_F(KernelFlagCmdOwnershipTest, CmdDestructorReleasesFlag) {
  constexpr int poolSize = 4;
  auto pool = std::make_unique<KernelFlagPool>(poolSize);
  ASSERT_EQ(pool->size(), poolSize);

  // Pop a flag from the pool.
  auto* flag = pool->pop();
  ASSERT_NE(flag, nullptr);
  EXPECT_EQ(pool->size(), poolSize - 1);

  // Simulate what submit() does during graph capture.
  flag->setPersistent();

  // Simulate kernel writing KERNEL_UNSET after a replay.
  // Without persistent, this would make the flag reclaimable.
  flag->reset();
  EXPECT_TRUE(flag->inUse());

  // Create a persistent cmd that owns this flag (simulates graph capture).
  auto* cmd = new CtranGpeCmd;
  cmd->persistent = true;
  cmd->kernelFlag = flag;

  // Pool reclaim should NOT reclaim this flag.
  pool->reclaim();
  EXPECT_EQ(pool->size(), poolSize - 1);

  // Destroy the cmd (simulates graph destruction -> cmdDestroy -> delete cmd).
  delete cmd;
  EXPECT_FALSE(flag->inUse());

  // Now pool reclaim should recover the flag.
  pool->reclaim();
  EXPECT_EQ(pool->size(), poolSize);
}

// Verify that ~CtranGpeCmd invokes postKernelCleanup for persistent (graph)
// cmds. During graph replay, postKernelCleanup is deliberately skipped so
// resources persist across replays. On graph destruction, cmdDestroy deletes
// the cmd, and the destructor must run the cleanup to free those resources.
TEST_F(KernelFlagCmdOwnershipTest, CmdDestructorRunsPostKernelCleanup) {
  bool cleanupCalled = false;

  auto* cmd = new CtranGpeCmd;
  cmd->persistent = true;
  cmd->postKernelCleanup = [&cleanupCalled]() { cleanupCalled = true; };

  EXPECT_FALSE(cleanupCalled);
  delete cmd;
  EXPECT_TRUE(cleanupCalled);
}

// Verify that ~CtranGpeCmd also invokes postKernelCleanup for non-persistent
// cmds (e.g., if cleanup wasn't already called by the GPE thread).
TEST_F(KernelFlagCmdOwnershipTest, NonPersistentCmdDestructorRunsCleanup) {
  bool cleanupCalled = false;

  auto* cmd = new CtranGpeCmd;
  cmd->persistent = false;
  cmd->postKernelCleanup = [&cleanupCalled]() { cleanupCalled = true; };

  delete cmd;
  EXPECT_TRUE(cleanupCalled);
}

// Verify that ~CtranGpeCmd handles null postKernelCleanup (already consumed).
TEST_F(KernelFlagCmdOwnershipTest, CmdDestructorNullCleanupNoOp) {
  auto* cmd = new CtranGpeCmd;
  cmd->persistent = true;
  cmd->postKernelCleanup = nullptr;
  delete cmd;
}

// Demonstrates the bug scenario: without setPersistent(), the kernel writes
// KERNEL_UNSET after each replay, making the flag reclaimable while
// the graph still holds a baked-in pointer to it.
TEST_F(KernelFlagCmdOwnershipTest, UnprotectedFlagIsReclaimed) {
  constexpr int poolSize = 4;
  auto pool = std::make_unique<KernelFlagPool>(poolSize);
  ASSERT_EQ(pool->size(), poolSize);

  auto* flag = pool->pop();
  ASSERT_NE(flag, nullptr);
  EXPECT_EQ(pool->size(), poolSize - 1);

  // Simulate kernel writing KERNEL_UNSET after replay (the bug scenario:
  // setPersistent() not called, so the flag becomes reclaimable).
  for (int i = 0; i < flag->numGroups_; ++i) {
    flag->flag_[i] = KERNEL_UNSET;
  }
  EXPECT_FALSE(flag->inUse());

  // Pool reclaim takes the flag back — dangerous if a graph still holds it.
  pool->reclaim();
  EXPECT_EQ(pool->size(), poolSize);
}

// Verifies that setPersistent() prevents reclaim regardless of flag values,
// and that reset() does not clear the persistent state.
TEST_F(KernelFlagCmdOwnershipTest, PersistentSurvivesReset) {
  constexpr int poolSize = 4;
  auto pool = std::make_unique<KernelFlagPool>(poolSize);

  auto* flag = pool->pop();
  ASSERT_NE(flag, nullptr);

  flag->setPersistent();

  // reset() clears flags but NOT persistent state.
  flag->reset();
  EXPECT_TRUE(flag->inUse());

  // Multiple resets — persistent still holds.
  flag->reset();
  flag->reset();
  EXPECT_TRUE(flag->inUse());

  // Pool can't reclaim.
  pool->reclaim();
  EXPECT_EQ(pool->size(), poolSize - 1);

  // Only clearPersistent() allows reclaim.
  flag->clearPersistent();
  EXPECT_FALSE(flag->inUse());
  pool->reclaim();
  EXPECT_EQ(pool->size(), poolSize);
}
