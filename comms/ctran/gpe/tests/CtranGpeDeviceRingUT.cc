// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

// Host-side unit tests for the device-ring GPE dispatch registry
// (NCCL_CTRAN_GPE_DEVICE_RING). Covers the mockable core of the ring path: the
// per-comm command registry (id assignment, lookup, erase) and the per-fire
// bookkeeping the GPE worker applies for each ring entry. The device ring
// transport itself (HRDWRingBuffer) is unit-tested in comms/utils/tests, and
// the end-to-end ring consumer is exercised by the single-node GPE/ctran GPU
// tests and multi-node MAST runs.

#include <memory>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>

// CtranGpeCmd's coll.pObj is a std::variant holding a
// std::unique_ptr<alltoallp::AlgoImpl>; constructing/destroying a CtranGpeCmd
// in this TU needs the COMPLETE AlgoImpl type (as CtranGpeImpl.cc does), so
// include its definition directly rather than relying on it being pulled in
// transitively (which only happens under some build configs).
#include "comms/ctran/algos/AllToAll/AllToAllPImpl.h"
#include "comms/ctran/gpe/CtranGpeImpl.h"
#include "comms/ctran/gpe/GpeDeviceRing.h"

namespace {

// A default-constructed CtranGpeCmd is safe to build and destroy on the host:
// its destructor no-ops when kernelFlag and postKernelCleanup are null.
std::unique_ptr<CtranGpeCmd> makeCmd(bool persistent) {
  auto cmd = std::make_unique<CtranGpeCmd>();
  cmd->persistent = persistent;
  return cmd;
}

TEST(GpeDeviceRingCmdRegistryTest, AssignsMonotonicIdsStartingAtOne) {
  GpeDeviceRingCmdRegistry registry;
  auto a = makeCmd(/*persistent=*/true);
  auto b = makeCmd(/*persistent=*/true);
  auto c = makeCmd(/*persistent=*/true);

  const std::vector<ctran::gpe::GpeCmdId> ids = {
      registry.registerCmd(a.get()),
      registry.registerCmd(b.get()),
      registry.registerCmd(c.get())};

  const std::vector<ctran::gpe::GpeCmdId> expected = {1, 2, 3};
  EXPECT_EQ(ids, expected);
  // Each cmd records its own id and is marked as owning a registry entry.
  EXPECT_EQ(a->cmdId, 1u);
  EXPECT_EQ(b->cmdId, 2u);
  EXPECT_EQ(c->cmdId, 3u);
  EXPECT_TRUE(a->inDeviceRingRegistry);
  EXPECT_TRUE(b->inDeviceRingRegistry);
  EXPECT_TRUE(c->inDeviceRingRegistry);
  EXPECT_EQ(registry.size(), 3u);
}

TEST(GpeDeviceRingCmdRegistryTest, LookupResolvesIdToRegisteredCmd) {
  GpeDeviceRingCmdRegistry registry;
  auto a = makeCmd(/*persistent=*/true);
  auto b = makeCmd(/*persistent=*/true);
  const ctran::gpe::GpeCmdId idA = registry.registerCmd(a.get());
  const ctran::gpe::GpeCmdId idB = registry.registerCmd(b.get());

  EXPECT_EQ(registry.lookup(idA), a.get());
  EXPECT_EQ(registry.lookup(idB), b.get());
  // An id that was never registered resolves to nullptr, not garbage — the
  // GPE worker relies on this to skip stale/destroyed ring entries.
  EXPECT_EQ(registry.lookup(9999u), nullptr);
}

TEST(GpeDeviceRingCmdRegistryTest, EraseRemovesEntryAndIsIdempotent) {
  GpeDeviceRingCmdRegistry registry;
  auto a = makeCmd(/*persistent=*/true);
  const ctran::gpe::GpeCmdId id = registry.registerCmd(a.get());
  ASSERT_EQ(registry.lookup(id), a.get());

  registry.erase(id);
  EXPECT_EQ(registry.lookup(id), nullptr);
  EXPECT_EQ(registry.size(), 0u);
  // Erasing again (e.g. a late duplicate) must be safe.
  registry.erase(id);
  EXPECT_EQ(registry.size(), 0u);
}

TEST(GpeDeviceRingCmdRegistryTest, IdsAreUniqueAcrossManyRegistrations) {
  GpeDeviceRingCmdRegistry registry;
  std::vector<std::unique_ptr<CtranGpeCmd>> cmds;
  std::unordered_set<ctran::gpe::GpeCmdId> ids;
  constexpr int kN = 1000;
  for (int i = 0; i < kN; ++i) {
    cmds.push_back(makeCmd(/*persistent=*/true));
    ids.insert(registry.registerCmd(cmds.back().get()));
  }
  EXPECT_EQ(ids.size(), static_cast<size_t>(kN));
  EXPECT_EQ(registry.size(), static_cast<size_t>(kN));
}

TEST(
    GpeDeviceRingCmdRegistryTest,
    RecordRingReplayBumpsInFlightForPersistentCmd) {
  // Each ring entry is one fire of a captured (persistent) cmd.
  // recordRingReplay mirrors the host-node cmdCb: it bumps inFlight so
  // cmdDestroy waits for the GPE worker to finish this fire before freeing the
  // cmd.
  auto cmd = makeCmd(/*persistent=*/true);
  EXPECT_EQ(cmd->inFlight.load(), 0u);

  GpeDeviceRingCmdRegistry::recordRingReplay(cmd.get());
  EXPECT_EQ(cmd->inFlight.load(), 1u);

  GpeDeviceRingCmdRegistry::recordRingReplay(cmd.get());
  EXPECT_EQ(cmd->inFlight.load(), 2u);
}

TEST(GpeDeviceRingCmdRegistryTest, RecordRingReplayIsNoOpForNonPersistentCmd) {
  // Non-persistent (eager) cmds never traverse the ring path, so
  // recordRingReplay must not touch their inFlight (they are freed by the GPE
  // worker directly).
  auto cmd = makeCmd(/*persistent=*/false);
  GpeDeviceRingCmdRegistry::recordRingReplay(cmd.get());
  EXPECT_EQ(cmd->inFlight.load(), 0u);
}

} // namespace
