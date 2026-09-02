// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>

#include "comms/prims/tests/NvlProgressSlotReleaseTest.cuh"

namespace comms::prims {
namespace {

using test::NvlSlotProbe;

// NVL has one non-terminal stage, so unlike the IB suite there is no stage
// sweep here: `Active` is the only state a transfer can unwind from.

// Control. If staging did not actually produce a non-idle slot, the test below
// would pass without proving anything.
TEST(NvlProgressSlotReleaseTest, StagedSlotIsNotIdle) {
  const NvlSlotProbe probe = test::stageNvlSlot();
  EXPECT_EQ(probe.stage, test::kNvlStageActive);
  EXPECT_NE(probe.stage, test::kNvlStageIdle);
  EXPECT_GT(probe.nextByte, 0u) << "slot should be mid-transfer";
}

// The containment property: an aborted transfer leaves the slot reusable, so
// the next kernel on the channel re-initializes instead of trapping. The
// assert_progress_slot_idle() call inside the second kernel launch is itself
// half the assertion -- it traps the whole context if the slot is not
// released, which surfaces here as a CUDA error from the readback.
TEST(NvlProgressSlotReleaseTest, AbandonedSlotIsIdleForTheNextKernel) {
  const NvlSlotProbe probe = test::abandonNvlSlotThenReinit();
  EXPECT_EQ(probe.stage, test::kNvlStageIdle);
  EXPECT_EQ(probe.baseByte, 0u);
  EXPECT_EQ(probe.nextByte, 0u);
  EXPECT_EQ(probe.payloadBytes, 0u);
  EXPECT_EQ(probe.tailPadding, 0u);
  EXPECT_EQ(probe.userBytes, 0u);
  EXPECT_EQ(probe.maxSignalBytes, 0u);
}

// The reserved range is abandoned, not recycled. The peer may still write into
// it over NVLink, so rewinding the channel cursor would hand those bytes to the
// next operation.
TEST(NvlProgressSlotReleaseTest, AbandonKeepsTheChannelCursorAdvanced) {
  const NvlSlotProbe staged = test::stageNvlSlot();
  const NvlSlotProbe abandoned = test::abandonNvlSlotThenReinit();
  EXPECT_GT(staged.sendCursor, 0);
  EXPECT_EQ(abandoned.sendCursor, staged.sendCursor);
}

} // namespace
} // namespace comms::prims
