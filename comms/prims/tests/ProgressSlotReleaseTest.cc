// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <gtest/gtest.h>
#include <cstddef>

#include "comms/prims/tests/ProgressSlotReleaseTest.cuh"

namespace comms::prims {
namespace {

using test::SlotProbe;

// Every non-terminal stage a transfer can unwind from. `Busy` is deliberately
// absent: it belongs to the blocking send()/recv(), which runs its waits to
// completion under SKIP and clears the stage itself.
constexpr int kNonTerminalStages[] = {
    test::kStageWaitLocalCompletion,
    test::kStageWaitSlotFree,
    test::kStageWaitDataReady,
};

// Half a chunk in, so the slot is unambiguously mid-transfer.
constexpr std::size_t kNextByte = 8192;

// Control. If staging did not actually produce a non-idle slot, the test below
// would pass without proving anything.
TEST(ProgressSlotReleaseTest, StagedSlotIsNotIdle) {
  for (const int stage : kNonTerminalStages) {
    const SlotProbe probe = test::stage_slot(stage, kNextByte);
    EXPECT_EQ(probe.stage, stage);
    EXPECT_NE(probe.stage, test::kStageDone) << "stage=" << stage;
  }
}

// The containment property: an aborted transfer leaves the slot reusable, so
// the next kernel on the channel re-initializes instead of trapping. The
// assert_progress_slot_idle() call inside the second kernel launch is itself
// half the assertion -- it traps the whole context if the slot is not released,
// which surfaces here as a CUDA error from the readback.
TEST(ProgressSlotReleaseTest, AbandonedSlotIsIdleForTheNextKernel) {
  for (const int stage : kNonTerminalStages) {
    const SlotProbe probe = test::abandon_then_reinit(stage, kNextByte);
    EXPECT_EQ(probe.stage, test::kStageDone) << "stage=" << stage;
    EXPECT_EQ(probe.nextByte, 0u) << "stage=" << stage;
    EXPECT_EQ(probe.tailPadding, 0u) << "stage=" << stage;
    EXPECT_EQ(probe.baseStep, 0) << "stage=" << stage;
  }
}

// The reserved range is abandoned, not recycled. A peer's RDMA write may still
// land in it, so rewinding the channel cursor would hand those bytes to the
// next operation.
TEST(ProgressSlotReleaseTest, AbandonKeepsTheChannelCursorAdvanced) {
  const SlotProbe staged =
      test::stage_slot(test::kStageWaitSlotFree, kNextByte);
  const SlotProbe abandoned =
      test::abandon_then_reinit(test::kStageWaitSlotFree, kNextByte);
  EXPECT_GT(staged.nextStep, 0);
  EXPECT_EQ(abandoned.nextStep, staged.nextStep);
}

} // namespace
} // namespace comms::prims
